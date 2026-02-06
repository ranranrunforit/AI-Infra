"""Benchmark different models"""
import torch
import time
from src.models.resnet import create_resnet_model

def benchmark_model(model_name, batch_size=256):
    model = create_resnet_model(model_name, num_classes=10)
    model = model.cuda()
    model.eval()
    
    x = torch.randn(batch_size, 3, 224, 224).cuda()
    
    # Warmup
    for _ in range(10):
        _ = model(x)
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(100):
        _ = model(x)
    
    torch.cuda.synchronize()
    duration = time.time() - start
    
    throughput = (100 * batch_size) / duration
    print(f"{model_name}: {throughput:.1f} images/sec")

if __name__ == "__main__":
    for model in ["resnet18", "resnet50", "resnet101"]:
        benchmark_model(model)
