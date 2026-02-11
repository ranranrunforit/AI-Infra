"""
GPU Monitoring Utilities

Provides comprehensive GPU monitoring using PyTorch CUDA APIs and optional pynvml.
Tracks memory usage, utilization, temperature, power draw, and clock speeds.
Includes Prometheus metrics export support.
"""

import torch
import time
import logging
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Try to import pynvml for advanced metrics
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logger.debug("pynvml not available, using torch.cuda for GPU metrics")

# Try to import prometheus_client for metrics export
try:
    from prometheus_client import Gauge, Info
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


@dataclass
class GPUInfo:
    """GPU device information"""
    index: int
    name: str
    compute_capability: str
    total_memory_mb: float
    driver_version: str = ""
    cuda_version: str = ""


@dataclass
class GPUMetrics:
    """Real-time GPU metrics"""
    memory_allocated_mb: float = 0.0
    memory_reserved_mb: float = 0.0
    memory_total_mb: float = 0.0
    memory_free_mb: float = 0.0
    memory_utilization_pct: float = 0.0
    gpu_utilization_pct: float = 0.0
    temperature_c: float = 0.0
    power_draw_w: float = 0.0
    power_limit_w: float = 0.0
    clock_sm_mhz: int = 0
    clock_memory_mhz: int = 0
    fan_speed_pct: float = 0.0


def get_gpu_info() -> List[Dict[str, Any]]:
    """
    Get information about all available GPUs.
    
    Returns:
        List of dicts with GPU info. Empty list if no GPUs available.
    """
    if not torch.cuda.is_available():
        return []
    
    gpu_list = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        info = {
            "index": i,
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_mb": props.total_mem / (1024 * 1024),
            "multi_processor_count": props.multi_processor_count,
        }
        
        # Add driver/CUDA version if pynvml available
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                info["driver_version"] = pynvml.nvmlSystemGetDriverVersion()
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info["pci_bus_id"] = pynvml.nvmlDeviceGetPciInfo(handle).busId
                pynvml.nvmlShutdown()
            except Exception:
                pass
        
        gpu_list.append(info)
    
    return gpu_list


def print_gpu_info():
    """Print formatted GPU information to console"""
    gpu_info = get_gpu_info()
    
    if not gpu_info:
        print("No GPUs detected")
        return
    
    print("=" * 60)
    print(f"  GPU Information ({len(gpu_info)} device(s))")
    print("=" * 60)
    
    for gpu in gpu_info:
        print(f"\n  GPU {gpu['index']}: {gpu['name']}")
        print(f"    Compute Capability: {gpu['compute_capability']}")
        print(f"    Total Memory: {gpu['total_memory_mb']:.0f} MB")
        print(f"    Multiprocessors: {gpu['multi_processor_count']}")
        if "driver_version" in gpu:
            print(f"    Driver Version: {gpu['driver_version']}")
    
    print("\n" + "=" * 60)


class GPUMonitor:
    """
    Real-time GPU monitoring with optional Prometheus metrics export.
    
    Provides:
    - Memory tracking (allocated, reserved, free)
    - GPU utilization (requires pynvml)
    - Temperature monitoring
    - Power draw tracking
    - Background monitoring thread
    - Prometheus gauge integration
    
    Usage:
        monitor = GPUMonitor(device_id=0)
        metrics = monitor.get_gpu_metrics()
        print(f"Memory used: {metrics['memory_allocated_mb']:.1f} MB")
        
        # With background monitoring
        monitor.start_background_monitoring(interval=5.0)
        # ... training ...
        monitor.stop_background_monitoring()
        print(monitor.get_peak_metrics())
    """
    
    def __init__(
        self,
        device_id: int = 0,
        enable_prometheus: bool = False,
        prometheus_prefix: str = "gpu"
    ):
        self.device_id = device_id
        self._nvml_handle = None
        self._nvml_initialized = False
        self._background_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._metrics_history: List[Dict[str, float]] = []
        self._peak_memory_mb: float = 0.0
        self._peak_utilization: float = 0.0
        self._last_metrics: Dict[str, float] = {}
        self._last_fetch_time: float = 0.0
        self._cache_duration: float = 1.0  # seconds
        
        # Initialize pynvml if available
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                self._nvml_initialized = True
                logger.info(f"NVML initialized for GPU {device_id}")
            except Exception as e:
                logger.warning(f"Could not initialize NVML: {e}")
        
        # Setup Prometheus gauges
        self._prometheus_gauges = {}
        if enable_prometheus and PROMETHEUS_AVAILABLE:
            self._setup_prometheus_gauges(prometheus_prefix)
    
    def _setup_prometheus_gauges(self, prefix: str):
        """Create Prometheus gauge metrics"""
        gauge_configs = {
            "memory_allocated_mb": "GPU memory allocated in MB",
            "memory_reserved_mb": "GPU memory reserved in MB",
            "memory_total_mb": "GPU total memory in MB",
            "memory_utilization_pct": "GPU memory utilization percentage",
            "gpu_utilization_pct": "GPU compute utilization percentage",
            "temperature_c": "GPU temperature in Celsius",
            "power_draw_w": "GPU power draw in Watts",
        }
        for name, desc in gauge_configs.items():
            self._prometheus_gauges[name] = Gauge(
                f"{prefix}_{name}",
                desc,
                ["gpu_id"]
            )
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get current GPU metrics. Uses cache to avoid excessive queries.
        
        Returns:
            Dict with GPU metrics, all values as Python floats.
        """
        return self.get_gpu_metrics(force=False)
    
    def get_gpu_metrics(self, force: bool = False) -> Dict[str, float]:
        """
        Get detailed GPU metrics.
        
        Args:
            force: If True, bypass cache and fetch fresh metrics.
        
        Returns:
            Dict with metric name -> float value.
        """
        if not torch.cuda.is_available():
            return {}
        
        # Check cache
        now = time.time()
        if not force and (now - self._last_fetch_time) < self._cache_duration:
            return self._last_metrics
        
        metrics = {}
        
        # PyTorch CUDA memory metrics (always available)
        try:
            metrics["memory_allocated_mb"] = float(
                torch.cuda.memory_allocated(self.device_id) / (1024 * 1024)
            )
            metrics["memory_reserved_mb"] = float(
                torch.cuda.memory_reserved(self.device_id) / (1024 * 1024)
            )
            metrics["memory_max_allocated_mb"] = float(
                torch.cuda.max_memory_allocated(self.device_id) / (1024 * 1024)
            )
            
            props = torch.cuda.get_device_properties(self.device_id)
            metrics["memory_total_mb"] = float(props.total_mem / (1024 * 1024))
            metrics["memory_free_mb"] = float(
                metrics["memory_total_mb"] - metrics["memory_reserved_mb"]
            )
            metrics["memory_utilization_pct"] = float(
                (metrics["memory_reserved_mb"] / metrics["memory_total_mb"]) * 100
            ) if metrics["memory_total_mb"] > 0 else 0.0
        except Exception as e:
            logger.debug(f"Error getting CUDA memory metrics: {e}")
        
        # NVML metrics (utilization, temperature, power)
        if self._nvml_initialized and self._nvml_handle:
            try:
                # GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                metrics["gpu_utilization_pct"] = float(util.gpu)
                
                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(
                    self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU
                )
                metrics["temperature_c"] = float(temp)
                
                # Power
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
                    metrics["power_draw_w"] = float(power / 1000.0)  # mW -> W
                except pynvml.NVMLError:
                    pass
                
                try:
                    power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(self._nvml_handle)
                    metrics["power_limit_w"] = float(power_limit / 1000.0)
                except pynvml.NVMLError:
                    pass
                
                # Clock speeds
                try:
                    sm_clock = pynvml.nvmlDeviceGetClockInfo(
                        self._nvml_handle, pynvml.NVML_CLOCK_SM
                    )
                    mem_clock = pynvml.nvmlDeviceGetClockInfo(
                        self._nvml_handle, pynvml.NVML_CLOCK_MEM
                    )
                    metrics["clock_sm_mhz"] = float(sm_clock)
                    metrics["clock_memory_mhz"] = float(mem_clock)
                except pynvml.NVMLError:
                    pass
                
                # Fan speed
                try:
                    fan = pynvml.nvmlDeviceGetFanSpeed(self._nvml_handle)
                    metrics["fan_speed_pct"] = float(fan)
                except pynvml.NVMLError:
                    pass
                    
            except Exception as e:
                logger.debug(f"Error getting NVML metrics: {e}")
        
        # Update peak tracking
        mem_alloc = metrics.get("memory_allocated_mb", 0.0)
        if mem_alloc > self._peak_memory_mb:
            self._peak_memory_mb = mem_alloc
        
        gpu_util = metrics.get("gpu_utilization_pct", 0.0)
        if gpu_util > self._peak_utilization:
            self._peak_utilization = gpu_util
        
        # Update Prometheus gauges
        for name, gauge in self._prometheus_gauges.items():
            if name in metrics:
                gauge.labels(gpu_id=str(self.device_id)).set(metrics[name])
        
        self._last_metrics = metrics
        self._last_fetch_time = now
        return metrics
    
    def get_peak_metrics(self) -> Dict[str, float]:
        """Get peak metrics recorded during monitoring"""
        return {
            "peak_memory_allocated_mb": self._peak_memory_mb,
            "peak_gpu_utilization_pct": self._peak_utilization,
        }
    
    def reset_peak_metrics(self):
        """Reset peak metric tracking"""
        self._peak_memory_mb = 0.0
        self._peak_utilization = 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device_id)
    
    def start_background_monitoring(self, interval: float = 5.0):
        """
        Start background thread that periodically collects GPU metrics.
        
        Args:
            interval: Seconds between metric collections.
        """
        if self._background_thread is not None:
            logger.warning("Background monitoring already running")
            return
        
        self._stop_event.clear()
        
        def _monitor_loop():
            while not self._stop_event.is_set():
                metrics = self.get_gpu_metrics(force=True)
                self._metrics_history.append({
                    "timestamp": time.time(),
                    **metrics
                })
                self._stop_event.wait(interval)
        
        self._background_thread = threading.Thread(
            target=_monitor_loop, daemon=True, name="gpu-monitor"
        )
        self._background_thread.start()
        logger.info(f"Background GPU monitoring started (interval={interval}s)")
    
    def stop_background_monitoring(self) -> List[Dict[str, float]]:
        """
        Stop background monitoring and return collected metrics history.
        
        Returns:
            List of metric snapshots collected during monitoring.
        """
        if self._background_thread is None:
            return self._metrics_history
        
        self._stop_event.set()
        self._background_thread.join(timeout=10)
        self._background_thread = None
        logger.info(f"Background monitoring stopped. {len(self._metrics_history)} snapshots collected.")
        return self._metrics_history
    
    def get_memory_summary(self) -> str:
        """Get a formatted memory summary string"""
        metrics = self.get_gpu_metrics()
        if not metrics:
            return "No GPU available"
        
        return (
            f"GPU {self.device_id} Memory: "
            f"{metrics.get('memory_allocated_mb', 0):.0f}MB allocated / "
            f"{metrics.get('memory_reserved_mb', 0):.0f}MB reserved / "
            f"{metrics.get('memory_total_mb', 0):.0f}MB total "
            f"({metrics.get('memory_utilization_pct', 0):.1f}% used)"
        )
    
    def __del__(self):
        """Cleanup NVML on deletion"""
        self.stop_background_monitoring()
        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
