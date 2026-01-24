"""
GPU Metrics Collection using NVIDIA Management Library (NVML)
"""
import logging
import threading
import time
from typing import Optional

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    logging.warning("pynvml not available - GPU metrics will not be collected")

from prometheus_client import Gauge

logger = logging.getLogger(__name__)

# GPU Metrics
gpu_utilization = Gauge(
    'llm_gpu_utilization_percent',
    'GPU utilization percentage',
    ['gpu_id']
)

gpu_memory_used = Gauge(
    'llm_gpu_memory_used_bytes',
    'GPU memory used in bytes',
    ['gpu_id']
)

gpu_memory_total = Gauge(
    'llm_gpu_memory_total_bytes',
    'GPU memory total in bytes',
    ['gpu_id']
)

gpu_temperature = Gauge(
    'llm_gpu_temperature_celsius',
    'GPU temperature in Celsius',
    ['gpu_id']
)

gpu_power_usage = Gauge(
    'llm_gpu_power_watts',
    'GPU power usage in watts',
    ['gpu_id']
)


class GPUMetricsCollector:
    """Collects GPU metrics using NVML"""
    
    def __init__(self, collection_interval: int = 10):
        self.collection_interval = collection_interval
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.initialized = False
        
        if not NVML_AVAILABLE:
            logger.warning("NVML not available, GPU metrics disabled")
            return
            
        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            self.initialized = True
            logger.info(f"NVML initialized successfully. Found {self.device_count} GPU(s)")
        except Exception as e:
            logger.error(f"Failed to initialize NVML: {e}")
            self.initialized = False
    
    def start(self):
        """Start collecting GPU metrics in background thread"""
        if not self.initialized:
            logger.warning("GPU metrics collector not initialized, skipping")
            return
            
        if self.running:
            logger.warning("GPU metrics collector already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._collect_loop, daemon=True)
        self.thread.start()
        logger.info("GPU metrics collector started")
    
    def stop(self):
        """Stop collecting GPU metrics"""
        if not self.running:
            return
            
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        
        if self.initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception as e:
                logger.error(f"Error shutting down NVML: {e}")
        
        logger.info("GPU metrics collector stopped")
    
    def _collect_loop(self):
        """Background loop to collect GPU metrics"""
        while self.running:
            try:
                self._collect_metrics()
            except Exception as e:
                logger.error(f"Error collecting GPU metrics: {e}")
            
            time.sleep(self.collection_interval)
    
    def _collect_metrics(self):
        """Collect metrics from all GPUs"""
        if not self.initialized:
            return
        
        for i in range(self.device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                gpu_id = str(i)
                
                # GPU Utilization
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_utilization.labels(gpu_id=gpu_id).set(util.gpu)
                except Exception as e:
                    logger.debug(f"Could not get utilization for GPU {i}: {e}")
                
                # Memory Usage
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_memory_used.labels(gpu_id=gpu_id).set(mem_info.used)
                    gpu_memory_total.labels(gpu_id=gpu_id).set(mem_info.total)
                except Exception as e:
                    logger.debug(f"Could not get memory info for GPU {i}: {e}")
                
                # Temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    gpu_temperature.labels(gpu_id=gpu_id).set(temp)
                except Exception as e:
                    logger.debug(f"Could not get temperature for GPU {i}: {e}")
                
                # Power Usage
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
                    gpu_power_usage.labels(gpu_id=gpu_id).set(power)
                except Exception as e:
                    logger.debug(f"Could not get power usage for GPU {i}: {e}")
                    
            except Exception as e:
                logger.error(f"Error collecting metrics for GPU {i}: {e}")
    
    def collect_once(self):
        """Collect metrics once (for testing)"""
        if self.initialized:
            self._collect_metrics()


# Global instance
_gpu_collector: Optional[GPUMetricsCollector] = None


def start_gpu_metrics_collector(collection_interval: int = 10):
    """Start the global GPU metrics collector"""
    global _gpu_collector
    
    if _gpu_collector is None:
        _gpu_collector = GPUMetricsCollector(collection_interval)
    
    _gpu_collector.start()
    return _gpu_collector


def stop_gpu_metrics_collector():
    """Stop the global GPU metrics collector"""
    global _gpu_collector
    
    if _gpu_collector is not None:
        _gpu_collector.stop()