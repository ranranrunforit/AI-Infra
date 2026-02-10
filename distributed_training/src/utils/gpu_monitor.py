"""GPU monitoring utilities"""
import torch

class GPUMonitor:
    def __init__(self, device_id=0):
        self.device_id = device_id
    
    def get_metrics(self):
        if not torch.cuda.is_available():
            return {}
        
        return {
            "gpu_memory_allocated_mb": float(torch.cuda.memory_allocated(self.device_id) / 1024 / 1024),  # ✅ Converted to float
            "gpu_memory_reserved_mb": float(torch.cuda.memory_reserved(self.device_id) / 1024 / 1024),    # ✅ Converted to float
            "gpu_memory_cached_mb": float(torch.cuda.memory_reserved(self.device_id) / 1024 / 1024),      # ✅ Converted to float
        }
