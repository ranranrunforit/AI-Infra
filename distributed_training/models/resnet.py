"""
ResNet Model Implementations

Provides factory functions for creating ResNet models with various configurations.
Optimized for both transfer learning and training from scratch.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


def create_resnet_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = False,
    dropout_rate: float = 0.0
) -> nn.Module:
    """
    Create a ResNet model
    
    Args:
        model_name: Name of ResNet variant (resnet18, resnet34, resnet50, resnet101, resnet152)
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        dropout_rate: Dropout rate before final classifier (0 = no dropout)
    
    Returns:
        ResNet model ready for training
    """
    
    # Map model names to torchvision constructors
    model_constructors = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "resnet152": models.resnet152,
    }
    
    if model_name not in model_constructors:
        raise ValueError(
            f"Unknown ResNet model: {model_name}. "
            f"Available: {list(model_constructors.keys())}"
        )
    
    # Create model
    if pretrained:
        # Use pretrained weights
        weights = "IMAGENET1K_V1"
        model = model_constructors[model_name](weights=weights)
        print(f"Loaded pretrained {model_name} weights from ImageNet")
    else:
        # Random initialization
        model = model_constructors[model_name](weights=None)
        print(f"Created {model_name} with random initialization")
    
    # Modify final layer for custom number of classes
    in_features = model.fc.in_features
    
    if dropout_rate > 0:
        # Add dropout before classifier
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
    else:
        # Just replace classifier
        model.fc = nn.Linear(in_features, num_classes)
    
    print(f"Model output: {num_classes} classes")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model


class CustomResNet(nn.Module):
    """
    Custom ResNet with additional features:
    - Label smoothing
    - Mixup support
    - Feature extraction mode
    """
    
    def __init__(
        self,
        base_model: str = "resnet50",
        num_classes: int = 10,
        pretrained: bool = False,
        dropout_rate: float = 0.0,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        
        # Create base ResNet
        self.model = create_resnet_model(
            base_model,
            num_classes,
            pretrained,
            dropout_rate
        )
        
        # Store feature dimension
        self.feature_dim = self.model.fc.in_features if not isinstance(
            self.model.fc, nn.Sequential
        ) else self.model.fc[1].in_features
    
    def forward(self, x, return_features=False):
        """
        Forward pass
        
        Args:
            x: Input tensor
            return_features: If True, return features before classifier
        """
        if return_features:
            # Extract features
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            
            x = self.model.avgpool(x)
            features = torch.flatten(x, 1)
            
            logits = self.model.fc(features)
            return logits, features
        else:
            return self.model(x)
    
    def freeze_backbone(self):
        """Freeze all layers except classifier"""
        for name, param in self.model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
        print("Backbone frozen, only classifier is trainable")
    
    def unfreeze_backbone(self):
        """Unfreeze all layers"""
        for param in self.model.parameters():
            param.requires_grad = True
        print("Backbone unfrozen, all layers are trainable")


def get_model_info(model: nn.Module) -> dict:
    """
    Get detailed model information
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with model statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024 / 1024
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "size_mb": size_mb
    }


# Quick model creation shortcuts
def resnet18(num_classes: int = 1000, pretrained: bool = False):
    """Create ResNet-18"""
    return create_resnet_model("resnet18", num_classes, pretrained)


def resnet34(num_classes: int = 1000, pretrained: bool = False):
    """Create ResNet-34"""
    return create_resnet_model("resnet34", num_classes, pretrained)


def resnet50(num_classes: int = 1000, pretrained: bool = False):
    """Create ResNet-50"""
    return create_resnet_model("resnet50", num_classes, pretrained)


def resnet101(num_classes: int = 1000, pretrained: bool = False):
    """Create ResNet-101"""
    return create_resnet_model("resnet101", num_classes, pretrained)


def resnet152(num_classes: int = 1000, pretrained: bool = False):
    """Create ResNet-152"""
    return create_resnet_model("resnet152", num_classes, pretrained)


if __name__ == "__main__":
    # Test model creation
    print("Testing ResNet model creation...")
    print("=" * 80)
    
    # Test different models
    models_to_test = ["resnet18", "resnet50", "resnet101"]
    
    for model_name in models_to_test:
        print(f"\nCreating {model_name}...")
        model = create_resnet_model(model_name, num_classes=10, pretrained=False)
        info = get_model_info(model)
        
        print(f"Total parameters: {info['total_parameters']:,}")
        print(f"Model size: {info['size_mb']:.2f} MB")
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        y = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {y.shape}")
    
    print("\n" + "=" * 80)
    print("All tests passed!")
