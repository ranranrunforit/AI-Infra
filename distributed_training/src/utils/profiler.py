"""
Performance Profiling Utilities

Provides profiling tools for training performance analysis:
- Timer context managers (wall clock and CUDA events)
- Throughput calculator
- PyTorch profiler wrapper with TensorBoard export
- Memory profiling utilities
"""

import torch
import time
import logging
import statistics
from contextlib import contextmanager
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


@contextmanager
def profile_time(name: str = ""):
    """Simple wall-clock timer context manager"""
    start = time.time()
    yield
    duration = time.time() - start
    print(f"{name}: {duration:.4f}s")


@contextmanager
def cuda_timer(name: str = "", sync: bool = True):
    """
    CUDA event-based timer for accurate GPU timing.
    
    Args:
        name: Label for the timer
        sync: Whether to synchronize before measuring
    
    Yields:
        Dict that will contain 'elapsed_ms' after the block completes
    """
    result = {}
    
    if torch.cuda.is_available():
        if sync:
            torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        
        yield result
        
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        result['elapsed_ms'] = elapsed_ms
        if name:
            logger.debug(f"{name}: {elapsed_ms:.2f}ms")
    else:
        start = time.time()
        yield result
        elapsed_ms = (time.time() - start) * 1000
        result['elapsed_ms'] = elapsed_ms
        if name:
            logger.debug(f"{name}: {elapsed_ms:.2f}ms (CPU)")


class Profiler:
    """
    Collects and summarizes timing measurements.
    
    Usage:
        profiler = Profiler()
        profiler.record("forward", 12.5)
        profiler.record("backward", 18.3)
        profiler.summary()
    """
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
    
    def record(self, name: str, duration: float):
        """Record a timing measurement in seconds"""
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(duration)
    
    @contextmanager
    def measure(self, name: str):
        """Context manager that records timing automatically"""
        start = time.time()
        yield
        self.record(name, time.time() - start)
    
    def summary(self) -> Dict[str, Dict[str, float]]:
        """Print and return summary statistics for all recorded timings"""
        result = {}
        print("\n" + "=" * 60)
        print("  Profiler Summary")
        print("=" * 60)
        
        for name, times in sorted(self.timings.items()):
            mean_t = statistics.mean(times)
            std_t = statistics.stdev(times) if len(times) > 1 else 0.0
            total_t = sum(times)
            
            result[name] = {
                "mean": mean_t,
                "std": std_t,
                "min": min(times),
                "max": max(times),
                "total": total_t,
                "count": len(times),
            }
            
            print(
                f"  {name:20s}: mean={mean_t:.4f}s  std={std_t:.4f}s  "
                f"min={min(times):.4f}s  max={max(times):.4f}s  "
                f"count={len(times)}  total={total_t:.2f}s"
            )
        
        print("=" * 60 + "\n")
        return result
    
    def reset(self):
        """Clear all recorded timings"""
        self.timings.clear()


class ThroughputCalculator:
    """
    Calculates training throughput (samples/second).
    
    Usage:
        calc = ThroughputCalculator()
        calc.start()
        # ... process batch of 256 ...
        calc.update(256)
        print(f"Throughput: {calc.get_throughput():.1f} samples/sec")
    """
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self._start_time: Optional[float] = None
        self._total_samples = 0
        self._recent_throughputs: List[float] = []
        self._last_time: Optional[float] = None
        self._last_samples = 0
    
    def start(self):
        """Start the throughput timer"""
        self._start_time = time.time()
        self._last_time = self._start_time
        self._total_samples = 0
        self._last_samples = 0
    
    def update(self, num_samples: int):
        """
        Record that num_samples were processed.
        
        Args:
            num_samples: Number of samples processed in this batch
        """
        now = time.time()
        self._total_samples += num_samples
        
        if self._last_time is not None:
            dt = now - self._last_time
            if dt > 0:
                throughput = num_samples / dt
                self._recent_throughputs.append(throughput)
                if len(self._recent_throughputs) > self.window_size:
                    self._recent_throughputs.pop(0)
        
        self._last_time = now
        self._last_samples = num_samples
    
    def get_throughput(self) -> float:
        """Get current throughput (windowed average) in samples/sec"""
        if not self._recent_throughputs:
            return 0.0
        return statistics.mean(self._recent_throughputs)
    
    def get_total_throughput(self) -> float:
        """Get overall throughput since start() in samples/sec"""
        if self._start_time is None:
            return 0.0
        elapsed = time.time() - self._start_time
        if elapsed <= 0:
            return 0.0
        return self._total_samples / elapsed
    
    def get_eta(self, total_samples: int) -> float:
        """
        Estimate time remaining in seconds.
        
        Args:
            total_samples: Total samples to process
        
        Returns:
            Estimated seconds remaining
        """
        throughput = self.get_throughput()
        if throughput <= 0:
            return float('inf')
        remaining = total_samples - self._total_samples
        return max(0, remaining / throughput)


class PyTorchProfiler:
    """
    Wrapper around torch.profiler for easy training profiling.
    
    Usage:
        profiler = PyTorchProfiler(log_dir="./logs/profiler")
        profiler.start()
        for batch in dataloader:
            # ... training step ...
            profiler.step()
        profiler.stop()
    """
    
    def __init__(
        self,
        log_dir: str = "./logs/profiler",
        wait: int = 1,
        warmup: int = 1,
        active: int = 3,
        repeat: int = 2,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._profiler = None
        
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        
        self._schedule = torch.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=repeat
        )
        self._activities = activities
        self._record_shapes = record_shapes
        self._profile_memory = profile_memory
        self._with_stack = with_stack
    
    def start(self):
        """Start the profiler"""
        self._profiler = torch.profiler.profile(
            activities=self._activities,
            schedule=self._schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(self.log_dir)
            ),
            record_shapes=self._record_shapes,
            profile_memory=self._profile_memory,
            with_stack=self._with_stack,
        )
        self._profiler.__enter__()
        logger.info(f"PyTorch profiler started, traces will be saved to {self.log_dir}")
    
    def step(self):
        """Signal end of a profiling step"""
        if self._profiler:
            self._profiler.step()
    
    def stop(self):
        """Stop the profiler"""
        if self._profiler:
            self._profiler.__exit__(None, None, None)
            self._profiler = None
            logger.info("PyTorch profiler stopped")


def get_memory_snapshot() -> Dict[str, float]:
    """Get current CUDA memory snapshot"""
    if not torch.cuda.is_available():
        return {}
    
    return {
        "allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
        "reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
        "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "max_reserved_mb": torch.cuda.max_memory_reserved() / (1024 * 1024),
    }
