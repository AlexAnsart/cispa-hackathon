import torch
from torchvision.utils import save_image
import os

INPUT_PATH = "../natural_images.pt"
OUTPUT_DIR = "../vis_output"
OUTPUT_FILE = "original_100_images.png"

if not os.path.exists(INPUT_PATH):
    print(f"Error: {INPUT_PATH} not found.")
    exit(1)

data = torch.load(INPUT_PATH, weights_only=False)
images = data["images"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
save_image(images, os.path.join(OUTPUT_DIR, OUTPUT_FILE), nrow=10, padding=2)

print(f"Saved visualization to: {os.path.join(OUTPUT_DIR, OUTPUT_FILE)}")

