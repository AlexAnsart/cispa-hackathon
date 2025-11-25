import torch
import numpy as np
from torchvision.utils import save_image
import os

INPUT_PATH = "../natural_images.pt"
OUTPUT_PATH = "simple_attack.npz"
VIS_OUTPUT_DIR = "../vis_output"
NOISE_LEVEL = 0.1

data = torch.load(INPUT_PATH, weights_only=False)
images = data["images"]
image_ids = data["image_ids"]

noise = torch.randn_like(images) * NOISE_LEVEL
adv_images = images + noise
adv_images = torch.clamp(adv_images, 0, 1)

os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)
comparison = torch.cat([images[:5], adv_images[:5]])
save_image(comparison, os.path.join(VIS_OUTPUT_DIR, "simple_attack_sample.png"), nrow=5)

final_images = adv_images.cpu().numpy().astype(np.float32)
final_ids = image_ids.cpu().numpy()

np.savez_compressed(OUTPUT_PATH, images=final_images, image_ids=final_ids)
print(f"Saved submission to: {OUTPUT_PATH}")
