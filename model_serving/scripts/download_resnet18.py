"""
Download a pretrained ResNet18 model for testing.

This script downloads the ResNet18 model (~45MB) from torchvision
and saves it to the model cache directory for the serving API.

Usage (inside Docker container):
    python /app/scripts/download_resnet18.py

Usage (from host):
    docker exec -it model-serving python /app/scripts/download_resnet18.py

The model is saved to: /tmp/model_cache/resnet18.pt
To delete it later, simply remove that file or restart the container.
"""

import os
# Use /tmp for model downloads (appuser can't write to ~/.cache)
os.environ["TORCH_HOME"] = "/tmp/torch_cache"
os.environ["XDG_CACHE_HOME"] = "/tmp"

import torch
import torchvision.models as models

MODEL_DIR = "/tmp/model_cache"
MODEL_NAME = "resnet18"
MODEL_PATH = f"{MODEL_DIR}/{MODEL_NAME}.pt"


def main():
    print(f"Downloading pretrained ResNet18 model...")

    # Download pretrained ResNet18 (ImageNet, ~45MB)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()

    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  GPU: not available, using CPU")

    # Save the model
    torch.save(model, MODEL_PATH)

    print(f"  Saved to: {MODEL_PATH}")
    print(f"  Size: {__import__('os').path.getsize(MODEL_PATH) / 1024 / 1024:.1f} MB")
    print(f"  Classes: 1000 (ImageNet)")
    print()
    print(f"Test with:")
    print(f'  Invoke-RestMethod -Method Post -Uri "http://localhost:8000/v1/predict" \\')
    print(f'    -ContentType "application/json" \\')
    print(f'    -Body \'{{\"model\": \"{MODEL_NAME}\", \"inputs\": {{\"image\": \"test\"}}, \"parameters\": {{}}}}\'')


if __name__ == "__main__":
    main()
