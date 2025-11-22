import os
import subprocess
import sys
import urllib.request
import venv
import textwrap

ENV_DIR = "var_env"

# ==============================
# 1. Create a clean venv
# ==============================
if not os.path.exists(ENV_DIR):
    print(f">>> Creating virtual environment: {ENV_DIR}")
    venv.EnvBuilder(with_pip=True).create(ENV_DIR)
else:
    print(f">>> Using existing virtual environment: {ENV_DIR}")

def find_venv_python(env_dir):
    # Windows
    win_dir = os.path.join(env_dir, "Scripts")
    if os.path.exists(win_dir):
        for name in ["python.exe", "python3.exe"]:
            candidate = os.path.join(win_dir, name)
            if os.path.exists(candidate):
                return os.path.abspath(candidate)
    # Unix
    unix_dir = os.path.join(env_dir, "bin")
    if os.path.exists(unix_dir):
        for name in ["python3", "python"]:
            candidate = os.path.join(unix_dir, name)
            if os.path.exists(candidate):
                return os.path.abspath(candidate)
    return sys.executable

VENV_PY = find_venv_python(ENV_DIR)
print(">>> Using venv Python at:", VENV_PY)

# ==============================
# 2. Clone VAR repo if missing
# ==============================
if not os.path.exists("VAR"):
    print(">>> Cloning VAR repo...")
    subprocess.run(["git", "clone", "https://github.com/FoundationVision/VAR.git"], check=True)

os.chdir("VAR")

# ==============================
# 3. Download checkpoints
# ==============================
os.makedirs("checkpoints/var", exist_ok=True)
os.makedirs("checkpoints/vae", exist_ok=True)

def download(url, out_path):
    if not os.path.exists(out_path):
        print(f">>> Downloading {out_path}")
        urllib.request.urlretrieve(url, out_path)
    else:
        print(f">>> Already exists: {out_path}")

download("https://huggingface.co/FoundationVision/var/resolve/main/var_d16.pth",
         "checkpoints/var/var_d16.pth")
download("https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth",
         "checkpoints/vae/vae_ch160v4096z32.pth")

# ==============================
# 4. Install dependencies
# ==============================
print(">>> Installing dependencies in venv")
subprocess.run([VENV_PY, "-m", "pip", "install", "--upgrade", "pip"], check=True)
subprocess.run([VENV_PY, "-m", "pip", "install",
                "torch>=2.0.0", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cu121"], check=True)

# clean torch pin
req_file = "requirements.txt"
if os.path.exists(req_file):
    with open(req_file, "r") as f:
        lines = f.readlines()
    with open(req_file, "w") as f:
        for line in lines:
            if line.strip().startswith("torch"):
                continue
            f.write(line)

subprocess.run([VENV_PY, "-m", "pip", "install", "-r", "requirements.txt"], check=True)

# ==============================
# 5. Write sample.py (generation code)
# ==============================
sample_code = textwrap.dedent("""
    import argparse, os, torch, random, numpy as np
    from PIL import Image
    from models import build_vae_var

    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--ckpt", type=str, required=True)
        parser.add_argument("--vae", type=str, required=True)
        parser.add_argument("--depth", type=int, default=16)
        parser.add_argument("--classes", type=int, nargs="+", default=[207,483,701,970])
        parser.add_argument("--cfg", type=float, default=4.0)
        parser.add_argument("--output", type=str, default="outputs/var_class_samples")
        args = parser.parse_args()

        seed = 0
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        device = "cuda" if torch.cuda.is_available() else "cpu"
        patch_nums = (1,2,3,4,5,6,8,10,13,16)

        vae, var = build_vae_var(V=4096, Cvae=32, ch=160, share_quant_resi=4,
                                 device=device, patch_nums=patch_nums,
                                 num_classes=1000, depth=args.depth, shared_aln=False)
        vae.load_state_dict(torch.load(args.vae, map_location="cpu"))
        var.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
        vae.eval(); var.eval()
        for p in vae.parameters(): p.requires_grad_(False)
        for p in var.parameters(): p.requires_grad_(False)

        labels = torch.tensor(args.classes, device=device, dtype=torch.long)

        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                imgs = var.autoregressive_infer_cfg(
                    B=len(labels), label_B=labels,
                    cfg=args.cfg, top_k=900, top_p=0.95,
                    g_seed=seed, more_smooth=False
                )

        os.makedirs(args.output, exist_ok=True)
        for i, img in enumerate(imgs):
            arr = img.permute(1,2,0).mul(255).clamp(0,255).byte().cpu().numpy()
            out_path = os.path.join(args.output, f"class_{args.classes[i]}_{i}.png")
            Image.fromarray(arr).resize((256,256), Image.LANCZOS).save(out_path)
            print(">>> Saved", out_path)

    if __name__ == "__main__":
        main()
""")

with open("sample.py", "w") as f:
    f.write(sample_code)

# ==============================
# 6. Run sample generation
# ==============================
print(">>> Running class-conditional generation in venv")
os.makedirs("outputs/var_class_samples", exist_ok=True)

subprocess.run([VENV_PY, "sample.py",
                "--ckpt", "checkpoints/var/var_d16.pth",
                "--vae", "checkpoints/vae/vae_ch160v4096z32.pth",
                "--depth", "16",
                "--classes", "207", "483", "701", "970",
                "--output", "outputs/var_class_samples"], check=True)

print(">>> Done! Check images in VAR/outputs/var_class_samples/")
