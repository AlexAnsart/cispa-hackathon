"""
================================================================================
RAR.PY - OPTIMIZED IMAGE GENERATION FOR 4x A100 GPUs
================================================================================

This script generates RAR images for data augmentation with the following optimizations:

1. DEFAULT: Generates 5000 images (optimized for training)
2. DIVERSITY: Uses 50 diverse ImageNet classes for maximum variety
3. PARALLELIZATION: Uses 4 processes (1 per GPU) for ~4x speedup
4. RANDOM SEEDS: Uses random seeds for each image to ensure variety

USAGE:
    python RAR.py
    
    This will generate 5000 images in ~2-3 hours on 4x A100 GPUs (parallelized).
    
    To generate fewer images, use: python RAR.py --num_images 2000

OUTPUT:
    Images saved to: outputs_rar/
    Format: rar_rar_xl_cls{imagenet_class}_{index}.png
"""

import os
import sys
import subprocess
import shutil
import venv
import json
import random
import multiprocessing
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".rar_env"
REPO_DIR = ROOT / "1d-tokenizer"
OUT_DIR = ROOT / "outputs_rar"
WEIGHT_DIR = ROOT / "weights"

# Defaults for direct run (no terminal args needed) - OPTIMIZED
DEFAULT_CLASS_ID = 207
# Use 50 diverse ImageNet classes for better variety (same as VAR.py)
DEFAULT_CLASS_IDS = [
    207, 483, 701, 970,  # Original classes
    15, 23, 45, 67, 89,  # Additional diverse classes
    101, 123, 145, 167, 189,
    201, 223, 245, 267, 289,
    301, 323, 345, 367, 389,
    401, 423, 445, 467, 489,
    501, 523, 545, 567, 589,
    601, 623, 645, 667, 689,
    701, 723, 745, 767, 789,
    801, 823, 845, 867, 889
]
DEFAULT_NUM_IMAGES = 5000  # Optimized for training (was 1)
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


def generate_imagenet_class(venv_python: Path, class_id: int, rar_size: str = "rar_xl", 
                            num_images: int = 1, gpu_id: int = 0, start_idx: int = 0, 
                            seed_offset: int = 0):
    """
    Generate RAR images for a specific ImageNet class.
    
    Args:
        venv_python: Python executable in venv
        class_id: ImageNet class ID
        rar_size: RAR model size
        num_images: Number of images to generate
        gpu_id: GPU device ID to use (for parallelization)
        start_idx: Starting index for image naming
        seed_offset: Offset for random seed (for diversity)
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHT_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure weights are present
    print(f"[GPU {gpu_id}] Downloading tokenizer and RAR weights if missing...")
    tok_path = hf_download(venv_python, "fun-research/TiTok", "maskgit-vqgan-imagenet-f16-256.bin", WEIGHT_DIR)
    rar_bin = f"{rar_size}.bin"
    rar_path = hf_download(venv_python, "yucornetto/RAR", rar_bin, WEIGHT_DIR)

    # Execute generation inline inside the venv
    code = f"""
import sys
from pathlib import Path
import traceback
import random
import numpy as np
import torch

REPO_DIR = Path({str(REPO_DIR)!r})
WEIGHT_DIR = Path({str(WEIGHT_DIR)!r})
OUT_DIR = Path({str(OUT_DIR)!r})

try:
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

    # Use specific GPU for this process
    gpu_id = {gpu_id}
    device = f'cuda:{{gpu_id}}' if torch.cuda.is_available() and gpu_id < torch.cuda.device_count() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[GPU {{gpu_id}}] Using device: {{device}}")

    tokenizer = PretrainedTokenizer(config.model.vq_model.pretrained_tokenizer_weight)
    generator = RAR(config)
    generator.load_state_dict(torch.load(config.experiment.generator_checkpoint, map_location='cpu'))
    generator.eval(); generator.requires_grad_(False); generator.set_random_ratio(0)
    tokenizer.to(device)
    generator.to(device)

    cls_id = int({class_id})
    num_images = int({num_images})
    start_idx = int({start_idx})
    seed_offset = int({seed_offset})
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"[GPU {{gpu_id}}] Generating {{num_images}} images for class {{cls_id}} (starting at index {{start_idx}})")
    
    for i in range(num_images):
        # Use different seed for each image (for diversity)
        img_seed = seed_offset + start_idx + i
        torch.manual_seed(img_seed)
        random.seed(img_seed)
        np.random.seed(img_seed)
        
        imgs = demo_util.sample_fn(
            generator=generator,
            tokenizer=tokenizer,
            labels=[cls_id],
            randomize_temperature=1.02,
            guidance_scale=6.9,
            guidance_scale_pow=1.5,
            device=device,
        )
        img_idx = start_idx + i
        Image.fromarray(imgs[0]).save(OUT_DIR / f'rar_{{rar_size}}_cls{{cls_id}}_{{img_idx:05d}}.png')
        
        if (i + 1) % 100 == 0:
            print(f"[GPU {{gpu_id}}] Generated {{i+1}}/{{num_images}} images for class {{cls_id}}")
    
    print(f"[GPU {{gpu_id}}] DONE: Generated {{num_images}} images for class {{cls_id}}")
except Exception:
    print(f'[GPU {{gpu_id}}] [ERROR] Generation failed:')
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


def generate_worker(args_tuple):
    """Worker function for parallel generation on a specific GPU."""
    (venv_python, class_id, rar_size, num_images, gpu_id, start_idx, seed_offset) = args_tuple
    generate_imagenet_class(venv_python, class_id, rar_size, num_images, gpu_id, start_idx, seed_offset)


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
    
    # Determine number of GPUs available
    try:
        import torch  # type: ignore
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    except ImportError:
        num_gpus = 1
    
    print(f"[info] Detected {num_gpus} GPU(s) available")
    
    # Use diverse classes if not specified
    if args.class_ids is None:
        args.class_ids = DEFAULT_CLASS_IDS[:50]  # Use first 50 diverse classes
    
    # Calculate total images to generate
    total_images = args.num_images
    num_classes = len(args.class_ids) if args.class_ids else 1
    
    if args.class_ids:
        # Generate images for multiple classes
        images_per_class = total_images // num_classes
        remaining_images = total_images % num_classes
        
        print(f"[info] Generating {total_images} images across {num_classes} classes")
        print(f"[info] ~{images_per_class} images per class (with {remaining_images} extra)")
        
        # Use parallelization if multiple GPUs available
        if num_gpus > 1 and total_images > 100:
            print(f"[info] Using {num_gpus} GPUs for parallel generation")
            
            # Distribute work across GPUs
            all_tasks = []
            img_counter = 0
            initial_seed = random.randint(0, 2**31 - 1)
            
            for class_idx, cid in enumerate(args.class_ids):
                class_images = images_per_class + (1 if class_idx < remaining_images else 0)
                
                # Distribute images for this class across GPUs
                images_per_gpu = class_images // num_gpus
                extra_images = class_images % num_gpus
                
                for gpu_id in range(num_gpus):
                    gpu_images = images_per_gpu + (1 if gpu_id < extra_images else 0)
                    if gpu_images > 0:
                        start_idx = img_counter
                        seed_offset = initial_seed + img_counter
                        all_tasks.append((vpy, int(cid), args.rar_size, gpu_images, gpu_id, start_idx, seed_offset))
                        img_counter += gpu_images
            
            # Run in parallel using multiprocessing
            with multiprocessing.Pool(processes=num_gpus) as pool:
                pool.map(generate_worker, all_tasks)
        else:
            # Sequential generation (single GPU or small number of images)
            print(f"[info] Using sequential generation (single process)")
            img_counter = 0
            initial_seed = random.randint(0, 2**31 - 1)
            
            for class_idx, cid in enumerate(args.class_ids):
                class_images = images_per_class + (1 if class_idx < remaining_images else 0)
                start_idx = img_counter
                seed_offset = initial_seed + img_counter
                generate_imagenet_class(vpy, class_id=int(cid), rar_size=args.rar_size, 
                                      num_images=class_images, gpu_id=0, 
                                      start_idx=start_idx, seed_offset=seed_offset)
                img_counter += class_images
    else:
        # Single class generation
        print(f"[info] Generating {total_images} images for class {args.class_id}")
        if num_gpus > 1 and total_images > 100:
            # Distribute across GPUs
            images_per_gpu = total_images // num_gpus
            extra_images = total_images % num_gpus
            initial_seed = random.randint(0, 2**31 - 1)
            
            all_tasks = []
            img_counter = 0
            for gpu_id in range(num_gpus):
                gpu_images = images_per_gpu + (1 if gpu_id < extra_images else 0)
                if gpu_images > 0:
                    start_idx = img_counter
                    seed_offset = initial_seed + img_counter
                    all_tasks.append((vpy, args.class_id, args.rar_size, gpu_images, gpu_id, start_idx, seed_offset))
                    img_counter += gpu_images
            
            with multiprocessing.Pool(processes=num_gpus) as pool:
                pool.map(generate_worker, all_tasks)
        else:
            # Single GPU
            initial_seed = random.randint(0, 2**31 - 1)
            generate_imagenet_class(vpy, class_id=args.class_id, rar_size=args.rar_size, 
                                  num_images=total_images, gpu_id=0, 
                                  start_idx=0, seed_offset=initial_seed)
    
    print(f"[done] All images saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
