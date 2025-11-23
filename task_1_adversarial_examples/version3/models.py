"""
Hybrid Ensemble Model Architecture for 28x28 Adversarial Attacks

Strategy:
- Group A: ImageNet pretrained models (28 -> 224 upsampling)
- Group B: CIFAR-10 trained models (28 -> 32 native resolution)

This dual approach maximizes transfer by attacking both high-level semantic 
features (Group A) and low-level pixel patterns (Group B).
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Tuple


class HybridEnsemble(nn.Module):
    """
    Ensemble combining ImageNet models (upsampled) and CIFAR-10 models (native).
    
    The key insight: We don't know if the black-box uses small native models
    or large pretrained models. Attack both fronts simultaneously.
    """
    
    def __init__(self, device='cuda', fast_mode: bool = False):
        super().__init__()
        self.device = device
        self.fast_mode = fast_mode
        
        # ImageNet normalization
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        
        # CIFAR-10 normalization
        self.cifar_mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
        self.cifar_std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(device)
        
        # Upsampling layers
        self.upsample_224 = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        self.upsample_32 = nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False)
        
        # Group A: ImageNet Giants
        self.imagenet_models = self._load_imagenet_models(fast_mode=fast_mode)
        
        # Group B: CIFAR-10 Native (if available, skip in fast mode)
        if not fast_mode:
            self.cifar_models = self._load_cifar_models()
        else:
            self.cifar_models = []  # Skip CIFAR models in fast mode
        
        # Initial weights (uniform)
        num_models = len(self.imagenet_models) + len(self.cifar_models)
        self.weights = [1.0 / num_models] * num_models
        
        print(f"✓ Loaded {len(self.imagenet_models)} ImageNet models + {len(self.cifar_models)} CIFAR models")
        print(f"  Total ensemble size: {num_models}")
    
    def _load_imagenet_models(self, fast_mode: bool = False) -> List[nn.Module]:
        """
        Load diverse ImageNet pretrained models.
        
        Args:
            fast_mode: If True, load only 2 models instead of 4 (for speed)
        """
        models_list = []
        
        # In fast mode, use only 2 most diverse models
        if fast_mode:
            configs = [
                ('ResNet50', lambda: models.resnet50(weights=models.ResNet50_Weights.DEFAULT)),
                ('DenseNet121', lambda: models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)),
            ]
        else:
            configs = [
                ('ResNet50', lambda: models.resnet50(weights=models.ResNet50_Weights.DEFAULT)),
                ('DenseNet121', lambda: models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)),
                ('VGG16_BN', lambda: models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)),
                ('EfficientNet_B0', lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)),
            ]
        
        for name, loader in configs:
            try:
                model = loader()
                model.eval()
                model.to(self.device)
                for param in model.parameters():
                    param.requires_grad = False
                models_list.append(model)
                print(f"  ✓ {name}")
            except Exception as e:
                print(f"  ✗ {name} failed: {e}")
        
        return models_list
    
    def _load_cifar_models(self) -> List[nn.Module]:
        """
        Load CIFAR-10 trained models.
        
        Note: If pretrained CIFAR-10 weights are not available, this will return 
        an empty list. The ensemble will still work with ImageNet models only.
        
        To add CIFAR-10 models, download from:
        https://github.com/huyvnphan/PyTorch_CIFAR10
        """
        models_list = []
        
        try:
            # Attempt to load CIFAR-10 ResNet18
            # This is a placeholder - you would load actual CIFAR-10 weights here
            import warnings
            warnings.filterwarnings('ignore')
            
            # For now, use a standard ResNet18 (not ideal, but adds diversity)
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            model.eval()
            model.to(self.device)
            for param in model.parameters():
                param.requires_grad = False
            models_list.append(model)
            print(f"  ✓ ResNet18 (ImageNet pretrained, adapted)")
            
        except Exception as e:
            print(f"  Note: CIFAR-10 models not loaded ({e})")
            print(f"        Ensemble will use ImageNet models only.")
        
        return models_list
    
    def apply_input_diversity(self, x: torch.Tensor, target_size: int) -> torch.Tensor:
        """
        Apply input diversity transformations adapted for small images.
        
        For 28x28 images, we need subtle transformations to avoid destroying information.
        
        Args:
            x: Input tensor (B, 3, 28, 28)
            target_size: Target upsampling size (32 or 224)
        
        Returns:
            Transformed tensor (B, 3, target_size, target_size)
        """
        B = x.shape[0]
        transformed = []
        
        for i in range(B):
            img = x[i:i+1]
            
            # Random scaling (subtle for small images)
            if target_size == 224:
                scale = torch.rand(1).item() * 0.15 + 0.90  # [0.90, 1.05]
                scale_size = int(target_size * scale)
                scale_size = max(200, min(230, scale_size))
            else:  # target_size == 32
                scale = torch.rand(1).item() * 0.10 + 0.95  # [0.95, 1.05]
                scale_size = int(target_size * scale)
                scale_size = max(30, min(34, scale_size))
            
            # Upsample to scaled size
            upsampler = nn.Upsample(size=(scale_size, scale_size), mode='bilinear', align_corners=False)
            scaled = upsampler(img)
            
            # Random crop/pad to exact target size
            if scale_size != target_size:
                if scale_size > target_size:
                    # Random crop
                    top = torch.randint(0, scale_size - target_size + 1, (1,)).item()
                    left = torch.randint(0, scale_size - target_size + 1, (1,)).item()
                    scaled = scaled[:, :, top:top+target_size, left:left+target_size]
                else:
                    # Random pad
                    pad_total_h = target_size - scale_size
                    pad_total_w = target_size - scale_size
                    pad_top = torch.randint(0, pad_total_h + 1, (1,)).item()
                    pad_left = torch.randint(0, pad_total_w + 1, (1,)).item()
                    pad_bottom = pad_total_h - pad_top
                    pad_right = pad_total_w - pad_left
                    scaled = nn.functional.pad(scaled, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
            
            # Subtle brightness/contrast jitter
            brightness = torch.rand(1).item() * 0.06 + 0.97  # [0.97, 1.03]
            scaled = scaled * brightness
            scaled = torch.clamp(scaled, 0, 1)
            
            transformed.append(scaled)
        
        return torch.cat(transformed, dim=0)
    
    def forward(self, x: torch.Tensor, labels: torch.Tensor, use_diversity: bool = True) -> torch.Tensor:
        """
        Compute ensemble loss for the attack.
        
        Args:
            x: Input images (B, 3, 28, 28) in [0, 1]
            labels: True labels (B,)
            use_diversity: Whether to apply input diversity
        
        Returns:
            Total loss (scalar) - maximize this to attack
        """
        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0.0
        weight_idx = 0
        
        # Group A: ImageNet models (224x224)
        if len(self.imagenet_models) > 0:
            if use_diversity:
                x_224 = self.apply_input_diversity(x, target_size=224)
            else:
                x_224 = self.upsample_224(x)
            
            x_224_norm = (x_224 - self.imagenet_mean) / self.imagenet_std
            
            for model in self.imagenet_models:
                outputs = model(x_224_norm)
                loss = loss_fn(outputs, labels)
                total_loss += self.weights[weight_idx] * loss
                weight_idx += 1
        
        # Group B: CIFAR models (32x32)
        if len(self.cifar_models) > 0:
            if use_diversity:
                x_32 = self.apply_input_diversity(x, target_size=32)
            else:
                x_32 = self.upsample_32(x)
            
            # Use ImageNet normalization for now (CIFAR models are ImageNet pretrained)
            x_32_norm = (x_32 - self.imagenet_mean) / self.imagenet_std
            
            for model in self.cifar_models:
                outputs = model(x_32_norm)
                loss = loss_fn(outputs, labels)
                total_loss += self.weights[weight_idx] * loss
                weight_idx += 1
        
        return total_loss
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get ensemble predictions (for success checking).
        
        Args:
            x: Input images (B, 3, 28, 28) in [0, 1]
        
        Returns:
            Logits (B, num_classes)
        """
        all_logits = []
        
        # Group A: ImageNet models
        if len(self.imagenet_models) > 0:
            x_224 = self.upsample_224(x)
            x_224_norm = (x_224 - self.imagenet_mean) / self.imagenet_std
            
            for model in self.imagenet_models:
                logits = model(x_224_norm)
                all_logits.append(logits)
        
        # Group B: CIFAR models
        if len(self.cifar_models) > 0:
            x_32 = self.upsample_32(x)
            x_32_norm = (x_32 - self.imagenet_mean) / self.imagenet_std
            
            for model in self.cifar_models:
                logits = model(x_32_norm)
                all_logits.append(logits)
        
        # Average logits
        return torch.stack(all_logits).mean(dim=0)
    
    def update_weights(self, new_weights: List[float]):
        """Update model weights based on correlation analysis (Phase 3)."""
        assert len(new_weights) == len(self.weights), "Weight dimension mismatch"
        self.weights = new_weights
        print(f"Updated model weights: {[f'{w:.3f}' for w in new_weights]}")

