"""Metrics tracking and logging"""
from pathlib import Path
import json
import time
import torch

class MetricsTracker:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.log_dir / "metrics.jsonl"
    
    def log(self, metrics, step):
        # Convert any torch.Tensor to Python numbers
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                # Convert tensor to Python number
                serializable_metrics[key] = value.item() if value.numel() == 1 else value.tolist()
            else:
                serializable_metrics[key] = value
        
        serializable_metrics['step'] = step
        serializable_metrics['timestamp'] = time.time()
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(serializable_metrics) + '\n')
