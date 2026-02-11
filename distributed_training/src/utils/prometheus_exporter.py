"""
Prometheus Metrics Exporter

Provides an HTTP endpoint for Prometheus to scrape training and GPU metrics.
Integrates with the GPUMonitor and MetricsTracker for real-time metric export.
"""

import threading
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from prometheus_client import (
        start_http_server,
        Gauge,
        Counter,
        Histogram,
        Info,
        REGISTRY,
        generate_latest,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed, metrics export disabled")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class PrometheusExporter:
    """
    Starts an HTTP server that exposes Prometheus metrics.
    
    Metrics exposed:
    - training_epoch: Current training epoch
    - training_loss: Current training loss
    - training_accuracy: Current training accuracy (%)
    - training_val_loss: Current validation loss
    - training_val_accuracy: Current validation accuracy (%)
    - training_learning_rate: Current learning rate
    - training_throughput: Samples per second
    - training_step: Global training step
    - gpu_memory_allocated_mb: GPU memory allocated
    - gpu_memory_reserved_mb: GPU memory reserved
    - gpu_memory_total_mb: GPU total memory
    - gpu_utilization_pct: GPU utilization %
    - gpu_temperature_c: GPU temperature
    - gpu_power_draw_w: GPU power draw
    
    Usage:
        exporter = PrometheusExporter(port=8080)
        exporter.start()
        
        # During training:
        exporter.update_training_metrics({
            "epoch": 5, "loss": 0.3, "accuracy": 92.5
        })
        exporter.update_gpu_metrics({
            "memory_allocated_mb": 4096, "gpu_utilization_pct": 85
        })
    """
    
    def __init__(self, port: int = 8080):
        self.port = port
        self._server_thread: Optional[threading.Thread] = None
        self._started = False
        
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available, exporter disabled")
            return
        
        # Training metrics
        self.training_epoch = Gauge(
            "training_epoch", "Current training epoch"
        )
        self.training_loss = Gauge(
            "training_loss", "Current training loss"
        )
        self.training_accuracy = Gauge(
            "training_accuracy", "Current training accuracy percentage"
        )
        self.training_val_loss = Gauge(
            "training_val_loss", "Current validation loss"
        )
        self.training_val_accuracy = Gauge(
            "training_val_accuracy", "Current validation accuracy percentage"
        )
        self.training_learning_rate = Gauge(
            "training_learning_rate", "Current learning rate"
        )
        self.training_throughput = Gauge(
            "training_throughput", "Training throughput in samples/sec"
        )
        self.training_step = Gauge(
            "training_step", "Global training step"
        )
        self.training_best_accuracy = Gauge(
            "training_best_accuracy", "Best validation accuracy achieved"
        )
        
        # GPU metrics
        self.gpu_memory_allocated = Gauge(
            "gpu_memory_allocated_mb", "GPU memory allocated in MB", ["device"]
        )
        self.gpu_memory_reserved = Gauge(
            "gpu_memory_reserved_mb", "GPU memory reserved in MB", ["device"]
        )
        self.gpu_memory_total = Gauge(
            "gpu_memory_total_mb", "GPU total memory in MB", ["device"]
        )
        self.gpu_utilization = Gauge(
            "gpu_utilization_pct", "GPU compute utilization %", ["device"]
        )
        self.gpu_temperature = Gauge(
            "gpu_temperature_c", "GPU temperature in Celsius", ["device"]
        )
        self.gpu_power_draw = Gauge(
            "gpu_power_draw_w", "GPU power draw in Watts", ["device"]
        )
        
        # System info
        self.system_info = Info("training_system", "Training system information")
        
        # Counters
        self.checkpoints_saved = Counter(
            "training_checkpoints_saved_total", "Total checkpoints saved"
        )
        self.epochs_completed = Counter(
            "training_epochs_completed_total", "Total epochs completed"
        )
        
        # Histograms
        self.batch_duration = Histogram(
            "training_batch_duration_seconds",
            "Time per training batch",
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
        )
    
    def start(self):
        """Start the Prometheus metrics HTTP server"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        if self._started:
            logger.warning("Prometheus exporter already started")
            return
        
        try:
            start_http_server(self.port)
            self._started = True
            logger.info(f"Prometheus metrics server started on port {self.port}")
            logger.info(f"  Metrics URL: http://localhost:{self.port}/metrics")
        except Exception as e:
            logger.error(f"Failed to start Prometheus metrics server: {e}")
    
    def update_training_metrics(self, metrics: dict):
        """Update training-related Prometheus gauges"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        gauge_map = {
            "epoch": self.training_epoch,
            "loss": self.training_loss,
            "accuracy": self.training_accuracy,
            "val_loss": self.training_val_loss,
            "val_accuracy": self.training_val_accuracy,
            "lr": self.training_learning_rate,
            "learning_rate": self.training_learning_rate,
            "throughput": self.training_throughput,
            "step": self.training_step,
            "best_accuracy": self.training_best_accuracy,
        }
        
        for key, gauge in gauge_map.items():
            if key in metrics:
                try:
                    gauge.set(float(metrics[key]))
                except (TypeError, ValueError):
                    pass
    
    def update_gpu_metrics(self, metrics: dict, device_id: str = "0"):
        """Update GPU-related Prometheus gauges"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        gpu_gauge_map = {
            "memory_allocated_mb": self.gpu_memory_allocated,
            "memory_reserved_mb": self.gpu_memory_reserved,
            "memory_total_mb": self.gpu_memory_total,
            "gpu_utilization_pct": self.gpu_utilization,
            "temperature_c": self.gpu_temperature,
            "power_draw_w": self.gpu_power_draw,
        }
        
        for key, gauge in gpu_gauge_map.items():
            if key in metrics:
                try:
                    gauge.labels(device=device_id).set(float(metrics[key]))
                except (TypeError, ValueError):
                    pass
    
    def record_batch_duration(self, duration_sec: float):
        """Record batch processing time"""
        if PROMETHEUS_AVAILABLE:
            self.batch_duration.observe(duration_sec)
    
    def record_checkpoint_saved(self):
        """Increment checkpoint counter"""
        if PROMETHEUS_AVAILABLE:
            self.checkpoints_saved.inc()
    
    def record_epoch_completed(self):
        """Increment epoch counter"""
        if PROMETHEUS_AVAILABLE:
            self.epochs_completed.inc()
    
    def set_system_info(self, info: dict):
        """Set system info labels"""
        if PROMETHEUS_AVAILABLE:
            str_info = {k: str(v) for k, v in info.items()}
            self.system_info.info(str_info)
