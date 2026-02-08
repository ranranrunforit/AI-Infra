"""
Vision Transformer (ViT) Model Implementations
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


def create_transformer_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = False,
    dropout_rate: float = 0.0
) -> nn.Module:
    """
    Create a Vision Transformer model
    
    Args:
        model_name: Name of ViT variant
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        dropout_rate: Dropout rate
    
    Returns:
        ViT model ready for training
    """
    
    model_constructors = {
        "vit_b_16": models.vit_b_16,
        "vit_b_32": models.vit_b_32,
        "vit_l_16": models.vit_l_16,
        "vit_l_32": models.vit_l_32,
    }
    
    if model_name not in model_constructors:
        raise ValueError(f"Unknown transformer model: {model_name}")
    
    # Create model
    if pretrained:
        weights = "IMAGENET1K_V1"
        model = model_constructors[model_name](weights=weights)
        print(f"Loaded pretrained {model_name} weights")
    else:
        model = model_constructors[model_name](weights=None)
        print(f"Created {model_name} with random initialization")
    
    # Modify classifier head
    in_features = model.heads.head.in_features
    
    if dropout_rate > 0:
        model.heads.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
    else:
        model.heads.head = nn.Linear(in_features, num_classes)
    
    print(f"Model output: {num_classes} classes")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


# Shortcuts
def vit_b_16(num_classes: int = 1000, pretrained: bool = False):
    """Create ViT-Base/16"""
    return create_transformer_model("vit_b_16", num_classes, pretrained)


def vit_l_16(num_classes: int = 1000, pretrained: bool = False):
    """Create ViT-Large/16"""
    return create_transformer_model("vit_l_16", num_classes, pretrained)
