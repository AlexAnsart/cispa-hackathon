#!/usr/bin/env python3
import os
import torch
from torchvision import transforms
from torchvision.utils import save_image

GRID_IMAGE_PATH = "../vis_output/original_100_images.png"
OUTPUT_DIR = "../vis_output/extracted_images"
DATASET_PATH = "../natural_images.pt"
GRID_ROWS = 10
GRID_COLS = 10
IMAGE_SIZE = 28
PADDING = 2

def extract_images_from_grid():
    if not os.path.exists(GRID_IMAGE_PATH):
        print(f"Error: Grid image not found at {GRID_IMAGE_PATH}")
        return extract_from_original_dataset()
    
    print(f"Loading grid image from {GRID_IMAGE_PATH}...")
    transform = transforms.ToTensor()
    
    from torchvision.io import read_image
    try:
        grid_tensor = read_image(GRID_IMAGE_PATH).float() / 255.0
        grid_height, grid_width = grid_tensor.shape[1], grid_tensor.shape[2]
    except:
        try:
            from PIL import Image
            grid_img = Image.open(GRID_IMAGE_PATH)
            grid_width, grid_height = grid_img.size
            grid_tensor = transform(grid_img)
        except ImportError:
            print("Error: Need PIL or torchvision.io to load images")
            return extract_from_original_dataset()
    
    print(f"Grid image size: {grid_width}x{grid_height}")
    
    cell_width = (grid_width - (GRID_COLS - 1) * PADDING) // GRID_COLS
    cell_height = (grid_height - (GRID_ROWS - 1) * PADDING) // GRID_ROWS
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    extracted_count = 0
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x_start = col * (cell_width + PADDING)
            y_start = row * (cell_height + PADDING)
            x_end = x_start + cell_width
            y_end = y_start + cell_height
            
            img_cell = grid_tensor[:, y_start:y_end, x_start:x_end]
            image_idx = row * GRID_COLS + col
            output_path = os.path.join(OUTPUT_DIR, f"image_{image_idx:03d}.png")
            save_image(img_cell, output_path)
            extracted_count += 1
            
            if (extracted_count % 10 == 0) or extracted_count == 100:
                print(f"Extracted {extracted_count}/100 images...")
    
    print(f"Extraction complete! Extracted {extracted_count} images.")
    
    files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')]
    file_count = len(files)
    
    if file_count == 100:
        print(f"Success: All 100 images extracted correctly!")
        return True
    else:
        print(f"Error: Expected 100 files, found {file_count}")
        return False


def extract_from_original_dataset():
    print("Loading from original dataset...")
    
    if not os.path.exists(DATASET_PATH):
        print(f"Error: {DATASET_PATH} not found")
        return False
    
    data = torch.load(DATASET_PATH, weights_only=False)
    images = data["images"]
    
    print(f"Loaded {len(images)} images from dataset")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for i in range(len(images)):
        output_path = os.path.join(OUTPUT_DIR, f"image_{i:03d}.png")
        save_image(images[i:i+1], output_path)
        
        if (i + 1) % 10 == 0 or i == 99:
            print(f"Saved {i+1}/100 images...")
    
    files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')]
    file_count = len(files)
    
    if file_count == 100:
        print("Success: All 100 images extracted correctly!")
        return True
    else:
        print(f"Error: Expected 100 files, found {file_count}")
        return False

if __name__ == "__main__":
    success = extract_images_from_grid()
    exit(0 if success else 1)

