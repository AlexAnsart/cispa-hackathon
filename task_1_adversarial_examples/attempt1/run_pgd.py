import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import os

INPUT_PATH = "../natural_images.pt"
OUTPUT_PATH = "submission_pgd.npz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPSILON = 10.0
ALPHA = 0.05
STEPS = 50


def get_surrogate_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    model.to(DEVICE)
    return model


def pgd_attack(model, images, labels):
    adv_images = images.clone().detach().to(DEVICE)
    labels = labels.clone().detach().to(DEVICE)
    originals = images.clone().detach().to(DEVICE)
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)
    upsampler = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Starting PGD Attack (Epsilon={EPSILON}, Steps={STEPS})...")

    for step in range(STEPS):
        adv_images.requires_grad = True
        
        resized = upsampler(adv_images)
        normalized = (resized - mean) / std
        outputs = model(normalized)
        loss = loss_fn(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        
        grad = adv_images.grad.data
        adv_images = adv_images.detach() + ALPHA * grad.sign()
        
        delta = adv_images - originals
        delta_flat = delta.view(delta.shape[0], -1)
        norm = delta_flat.norm(p=2, dim=1).view(delta.shape[0], 1, 1, 1)
        factor = torch.min(torch.ones_like(norm), torch.tensor(EPSILON) / (norm + 1e-6))
        delta = delta * factor
        adv_images = originals + delta
        adv_images = torch.clamp(adv_images, 0, 1).detach()
        
        if step % 10 == 0:
            print(f"  Step {step}: Loss = {loss.item():.4f}")

    return adv_images


if __name__ == "__main__":
    print(f"Running on: {DEVICE}")
    
    data = torch.load(INPUT_PATH, weights_only=False)
    images = data["images"]
    image_ids = data["image_ids"]
    labels = data["labels"]
    
    print(f"Loaded {len(images)} images.")
    
    model = get_surrogate_model()
    adv_images = pgd_attack(model, images, labels)
    
    diff = (adv_images.cpu() - images).view(100, -1)
    l2_dist = torch.norm(diff, p=2, dim=1).mean().item()
    print(f"Final Average L2 Distance: {l2_dist:.4f}")
    
    final_images_np = adv_images.cpu().numpy().astype(np.float32)
    final_ids_np = image_ids.cpu().numpy()
    
    np.savez_compressed(OUTPUT_PATH, images=final_images_np, image_ids=final_ids_np)
    print(f"Saved submission to {OUTPUT_PATH}")
