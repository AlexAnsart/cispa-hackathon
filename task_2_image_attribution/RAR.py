import os
import sys
import subprocess
import shutil
import venv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".rar_env"
REPO_DIR = ROOT / "1d-tokenizer"
OUT_DIR = ROOT / "outputs_rar"
WEIGHT_DIR = ROOT / "weights"

# Defaults for direct run (no terminal args needed)
DEFAULT_CLASS_ID = 207
DEFAULT_CLASS_IDS = [1,3,5,9]  # e.g., [207, 282, 404] to generate multiple by default
DEFAULT_NUM_IMAGES = 1
DEFAULT_RAR_SIZE = "rar_xl"  # one of: rar_b, rar_l, rar_xl, rar_xxl


def run(cmd, cwd=None, env=None, check=True, quiet=False):
    # Nicely print the command without dumping large inline code blobs
    display = cmd[:]
    if "-c" in display:
        try:
            i = display.index("-c")
            if i + 1 < len(display):
                display[i + 1] = "<inline>"
        except ValueError:
            pass
    print(f"[run] {' '.join(display)}")
    stdout = subprocess.DEVNULL if quiet else None
    stderr = subprocess.DEVNULL if quiet else None
    return subprocess.run(cmd, cwd=cwd, env=env, check=check, stdout=stdout, stderr=stderr)


def ensure_venv() -> Path:
    """Create a local venv if missing and return its python path."""
    if not VENV_DIR.exists():
        print(f"[setup] Creating venv at {VENV_DIR}")
        builder = venv.EnvBuilder(with_pip=True)
        builder.create(VENV_DIR)
    # Determine python executable inside venv (Windows/Linux)
    if os.name == 'nt':
        py = VENV_DIR / "Scripts" / "python.exe"
    else:
        py = VENV_DIR / "bin" / "python"
    return py


def in_venv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def install_requirements(venv_python: Path):
    # Upgrade pip tooling first
    run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel", "-q"], quiet=True)
    # Clone repo first so we can install its requirements
    if not REPO_DIR.exists():
        print(f"[setup] Cloning bytedance/1d-tokenizer into {REPO_DIR}")
        run(["git", "clone", "https://github.com/bytedance/1d-tokenizer", str(REPO_DIR)])
    else:
        print(f"[setup] Repo exists, pulling latest...")
        run(["git", "pull", "--ff-only"], cwd=str(REPO_DIR))

    # Install repo requirements
    req = REPO_DIR / "requirements.txt"
    deps_marker = VENV_DIR / ".deps_installed"
    if req.exists():
        if not deps_marker.exists():
            print("[setup] Installing repo requirements (first time)")
            run([str(venv_python), "-m", "pip", "install", "-r", str(req), "-q"], quiet=True)
            # Ensure diffusers only if needed
            cp = run([str(venv_python), "-c", "import diffusers"], check=False, quiet=True)
            if cp.returncode != 0:
                run([str(venv_python), "-m", "pip", "install", "diffusers<0.32", "-q"], quiet=True)
            deps_marker.write_text("ok")
        else:
            print("[setup] Requirements already installed; skipping")
    else:
        print("[warn] requirements.txt not found; installing minimal deps")
        run([str(venv_python), "-m", "pip", "install",
             "torch>=2.0.0", "torchvision", "omegaconf", "transformers", "timm",
             "open_clip_torch", "einops", "scipy", "pillow", "accelerate",
             "gdown", "huggingface-hub", "wandb", "torch-fidelity", "torchinfo", "webdataset", "-q"], quiet=True)


def reexec_in_venv(venv_python: Path):
    # Re-exec this script inside the venv
    env = os.environ.copy()
    env["RAR_BOOTSTRAPPED"] = "1"
    cmd = [str(venv_python), str(Path(__file__).resolve())] + sys.argv[1:]
    run(cmd, env=env)
    sys.exit(0)


def hf_download(venv_python: Path, repo_id: str, filename: str, local_dir: Path) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    code = f"""
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id={repo_id!r}, filename={filename!r}, local_dir={str(local_dir)!r})
print(path)
"""
    cp = subprocess.run([str(venv_python), "-c", code], stdout=subprocess.PIPE, text=True, check=True)
    p = Path(cp.stdout.strip())
    if not p.exists():
        raise RuntimeError(f"Download failed for {repo_id}/{filename}")
    return p


def generate_imagenet_class(venv_python: Path, class_id: int, rar_size: str = "rar_xl", num_images: int = 1):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHT_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure weights are present
    print("[weights] Downloading tokenizer and RAR weights if missing...")
    tok_path = hf_download(venv_python, "fun-research/TiTok", "maskgit-vqgan-imagenet-f16-256.bin", WEIGHT_DIR)
    rar_bin = f"{rar_size}.bin"
    rar_path = hf_download(venv_python, "yucornetto/RAR", rar_bin, WEIGHT_DIR)

    # Execute generation inline inside the venv
    code = f"""
import sys
from pathlib import Path
import traceback

REPO_DIR = Path({str(REPO_DIR)!r})
WEIGHT_DIR = Path({str(WEIGHT_DIR)!r})
OUT_DIR = Path({str(OUT_DIR)!r})

try:
    import torch
    from PIL import Image
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))
    import demo_util
    from modeling.titok import PretrainedTokenizer
    from modeling.rar import RAR

    cfg_map = {{
        
        'rar_xl': dict(hidden_size=1280, layers=32, heads=16, mlp=5120),
    }}
    rar_size = {rar_size!r}
    assert rar_size in cfg_map, f"Unsupported rar size: {{rar_size}}"

    config = demo_util.get_config(str(REPO_DIR / 'configs' / 'training' / 'generator' / 'rar.yaml'))
    config.experiment.generator_checkpoint = str(WEIGHT_DIR / f"{{rar_size}}.bin")
    config.model.generator.hidden_size = cfg_map[rar_size]['hidden_size']
    config.model.generator.num_hidden_layers = cfg_map[rar_size]['layers']
    config.model.generator.num_attention_heads = cfg_map[rar_size]['heads']
    config.model.generator.intermediate_size = cfg_map[rar_size]['mlp']
    config.model.vq_model.pretrained_tokenizer_weight = str(WEIGHT_DIR / 'maskgit-vqgan-imagenet-f16-256.bin')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = PretrainedTokenizer(config.model.vq_model.pretrained_tokenizer_weight)
    generator = RAR(config)
    generator.load_state_dict(torch.load(config.experiment.generator_checkpoint, map_location='cpu'))
    generator.eval(); generator.requires_grad_(False); generator.set_random_ratio(0)
    tokenizer.to(device)
    generator.to(device)

    cls_id = int({class_id})
    num_images = int({num_images})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(num_images):
        imgs = demo_util.sample_fn(
            generator=generator,
            tokenizer=tokenizer,
            labels=[cls_id],
            randomize_temperature=1.02,
            guidance_scale=6.9,
            guidance_scale_pow=1.5,
            device=device,
        )
        Image.fromarray(imgs[0]).save(OUT_DIR / f'rar_{{rar_size}}_cls{{cls_id}}_{{i}}.png')
    print('DONE')
except Exception:
    print('[ERROR] Generation failed:')
    traceback.print_exc()
    raise
"""
    run([str(venv_python), "-c", code])


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="RAR-XL one-shot setup and sampling")
    p.add_argument("--class_id", type=int, default=DEFAULT_CLASS_ID, help="ImageNet-1K class id [0..999]")
  
    p.add_argument("--rar_size", type=str, default=DEFAULT_RAR_SIZE, help="RAR model variant (fixed to rar_xl)")
    p.add_argument("--num_images", type=int, default=DEFAULT_NUM_IMAGES, help="Number of images to generate")
    p.add_argument("--class_ids", type=int, nargs='+', help="Generate for multiple class ids [0..999]")
    args = p.parse_args()
    # Enforce XL regardless of user input
    args.rar_size = "rar_xl"

    # Optional: supply class IDs via env var or classes.txt without terminal args
    if args.class_ids is None:
        env_cls = os.environ.get("RAR_CLASS_IDS")
        if env_cls:
            try:
                args.class_ids = [int(x.strip()) for x in env_cls.split(',') if x.strip()]
            except Exception:
                args.class_ids = None
    if args.class_ids is None:
        classes_file = ROOT / "classes.txt"
        if classes_file.exists():
            try:
                raw = classes_file.read_text()
                args.class_ids = [int(x) for x in raw.replace('\n', ' ').split() if x.strip()]
            except Exception:
                args.class_ids = None
    if args.class_ids is None and DEFAULT_CLASS_IDS:
        args.class_ids = list(DEFAULT_CLASS_IDS)

    return args


def main():
    args = parse_args()

    # Phase 1: ensure venv and requirements
    if not in_venv() and os.environ.get("RAR_BOOTSTRAPPED") != "1":
        vpy = ensure_venv()
        install_requirements(vpy)
        reexec_in_venv(vpy)
        return

    # Phase 2: already in venv — clone if needed (done in install), then generate
    # Ensure repo exists (in case venv already existed but repo missing)
    if not REPO_DIR.exists():
        run(["git", "clone", "https://github.com/bytedance/1d-tokenizer", str(REPO_DIR)])

    vpy = Path(sys.executable)
    if args.class_ids:
        for cid in args.class_ids:
            generate_imagenet_class(vpy, class_id=int(cid), rar_size=args.rar_size, num_images=args.num_images)
    else:
        generate_imagenet_class(vpy, class_id=args.class_id, rar_size=args.rar_size, num_images=args.num_images)
    print(f"[done] Images saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
