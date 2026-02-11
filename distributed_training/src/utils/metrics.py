"""
Metrics Tracking and Logging

Provides comprehensive metrics tracking for training runs including:
- AverageMeter for running statistics
- Accuracy computation utilities
- MetricsTracker with TensorBoard and Prometheus support
- JSON-lines logging for analysis
"""

import time
import json
import logging
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict

import torch

logger = logging.getLogger(__name__)

# Optional TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# Optional Prometheus
try:
    from prometheus_client import Gauge, Counter, Histogram, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    
    Usage:
        meter = AverageMeter()
        meter.update(loss_value, batch_size)
        print(f"Avg loss: {meter.avg:.4f}")
    """
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
        self.min = float('inf')
        self.max = float('-inf')
    
    def update(self, val: float, n: int = 1):
        """
        Update meter with new value.
        
        Args:
            val: New value
            n: Number of samples this value represents
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0
        self.min = min(self.min, val)
        self.max = max(self.max, val)
    
    def __str__(self):
        return f"{self.name}: {self.val:.4f} (avg: {self.avg:.4f})"


def accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    topk: Tuple[int, ...] = (1,)
) -> List[float]:
    """
    Computes the accuracy over the k top predictions.
    
    Args:
        output: Model output logits [batch_size, num_classes]
        target: Ground truth labels [batch_size]
        topk: Tuple of k values to compute top-k accuracy for
    
    Returns:
        List of accuracy values (as percentages) for each k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        result = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            acc = correct_k.mul_(100.0 / batch_size).item()
            result.append(acc)
        
        return result


class MetricsTracker:
    """
    Comprehensive metrics tracker with multiple backends.
    
    Supports:
    - In-memory metric storage with running averages
    - JSONL file logging
    - TensorBoard integration
    - Prometheus gauge export
    
    Usage:
        tracker = MetricsTracker(log_dir="./logs", use_tensorboard=True)
        tracker.update({"loss": 0.5, "accuracy": 85.0}, step=100)
        print(tracker.get_metric("loss"))
        tracker.close()
    """
    
    def __init__(
        self,
        log_dir: str = "./logs",
        use_tensorboard: bool = True,
        enable_prometheus: bool = False,
        prometheus_prefix: str = "training"
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.log_dir / "metrics.jsonl"
        
        # In-memory storage
        self._current_metrics: Dict[str, float] = {}
        self._meters: Dict[str, AverageMeter] = {}
        self._history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # TensorBoard writer
        self._writer = None
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            try:
                self._writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
                logger.info(f"TensorBoard logging to {self.log_dir / 'tensorboard'}")
            except Exception as e:
                logger.warning(f"Could not initialize TensorBoard: {e}")
        
        # Prometheus gauges
        self._prometheus_gauges: Dict[str, Any] = {}
        if enable_prometheus and PROMETHEUS_AVAILABLE:
            self._setup_prometheus(prometheus_prefix)
    
    def _setup_prometheus(self, prefix: str):
        """Setup Prometheus metrics"""
        base_metrics = [
            ("loss", "Training loss"),
            ("accuracy", "Training accuracy"),
            ("val_loss", "Validation loss"),
            ("val_accuracy", "Validation accuracy"),
            ("learning_rate", "Current learning rate"),
            ("throughput", "Training throughput in samples/sec"),
            ("epoch", "Current epoch"),
        ]
        for name, desc in base_metrics:
            self._prometheus_gauges[name] = Gauge(
                f"{prefix}_{name}", desc
            )
    
    def log(self, metrics: Dict[str, Any], step: int):
        """
        Log metrics to JSONL file (backward compatible).
        
        Args:
            metrics: Dict of metric name -> value
            step: Global step number
        """
        # Convert any torch.Tensor to Python numbers
        serializable = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                serializable[key] = value.item() if value.numel() == 1 else value.tolist()
            elif isinstance(value, (int, float, str, bool)):
                serializable[key] = value
            else:
                try:
                    serializable[key] = float(value)
                except (TypeError, ValueError):
                    serializable[key] = str(value)
        
        serializable['step'] = step
        serializable['timestamp'] = time.time()
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(serializable) + '\n')
    
    def update(self, metrics: Dict[str, Any], step: int):
        """
        Update metrics with new values. Writes to all backends.
        
        Args:
            metrics: Dict of metric name -> value
            step: Global step number
        """
        for key, value in metrics.items():
            # Convert tensors
            if isinstance(value, torch.Tensor):
                value = value.item() if value.numel() == 1 else value.tolist()
            
            # Store current value
            self._current_metrics[key] = value
            
            # Update running average
            if key not in self._meters:
                self._meters[key] = AverageMeter(name=key)
            if isinstance(value, (int, float)):
                self._meters[key].update(value)
            
            # Store history
            self._history[key].append({"step": step, "value": value, "time": time.time()})
        
        # Log to file
        self.log(metrics, step)
        
        # Log to TensorBoard
        if self._writer:
            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item() if value.numel() == 1 else float(value)
                if isinstance(value, (int, float)):
                    self._writer.add_scalar(key, value, step)
        
        # Update Prometheus
        for key, gauge in self._prometheus_gauges.items():
            if key in metrics:
                val = metrics[key]
                if isinstance(val, torch.Tensor):
                    val = val.item()
                if isinstance(val, (int, float)):
                    gauge.set(val)
    
    def get_metric(self, name: str) -> Optional[float]:
        """
        Get the current value of a metric.
        
        Args:
            name: Metric name
        
        Returns:
            Current metric value or None if not found
        """
        return self._current_metrics.get(name, None)
    
    def get_average(self, name: str) -> Optional[float]:
        """Get the running average of a metric"""
        if name in self._meters:
            return self._meters[name].avg
        return None
    
    def get_history(self, name: str) -> List[Dict[str, Any]]:
        """Get full history for a metric"""
        return self._history.get(name, [])
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all tracked metrics"""
        summary = {}
        for name, meter in self._meters.items():
            summary[name] = {
                "current": meter.val,
                "average": meter.avg,
                "min": meter.min if meter.min != float('inf') else 0.0,
                "max": meter.max if meter.max != float('-inf') else 0.0,
                "count": meter.count,
            }
        return summary
    
    def reset(self):
        """Reset all meters"""
        for meter in self._meters.values():
            meter.reset()
        self._current_metrics.clear()
    
    def close(self):
        """Close all backends"""
        if self._writer:
            self._writer.close()
            self._writer = None
        logger.info("MetricsTracker closed")
    
    def __del__(self):
        self.close()
