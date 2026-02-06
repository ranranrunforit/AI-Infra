import pytest
import torch
from src.models.resnet import create_resnet_model
from src.models.transformer import create_transformer_model

def test_resnet18_creation():
    model = create_resnet_model("resnet18", num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 10)

def test_resnet50_creation():
    model = create_resnet_model("resnet50", num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 10)
