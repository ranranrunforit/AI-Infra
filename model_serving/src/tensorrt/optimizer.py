"""
TensorRT Engine Optimizer

This module provides optimization strategies and performance tuning utilities
for TensorRT engines. Includes layer fusion analysis, kernel selection,
and performance profiling.

Features:
- Optimization profile management
- Layer fusion strategies
- Kernel auto-tuning
- Performance benchmarking
- Optimization recommendations
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Predefined optimization strategies."""
    LOW_LATENCY = "low_latency"  # Minimize single-request latency
    HIGH_THROUGHPUT = "high_throughput"  # Maximize batch throughput
    BALANCED = "balanced"  # Balance between latency and throughput
    MEMORY_EFFICIENT = "memory_efficient"  # Minimize memory usage


@dataclass
class OptimizationProfile:
    """Configuration profile for engine optimization."""
    strategy: OptimizationStrategy
    target_batch_size: int
    target_latency_ms: Optional[float] = None
    max_memory_mb: Optional[int] = None
    enable_layer_fusion: bool = True
    enable_kernel_tuning: bool = True
    precision_mode: str = "fp16"
    workspace_size_mb: int = 1024

    def to_dict(self) -> Dict:
        """Convert profile to dictionary."""
        return {
            "strategy": self.strategy.value,
            "target_batch_size": self.target_batch_size,
            "target_latency_ms": self.target_latency_ms,
            "max_memory_mb": self.max_memory_mb,
            "enable_layer_fusion": self.enable_layer_fusion,
            "enable_kernel_tuning": self.enable_kernel_tuning,
            "precision_mode": self.precision_mode,
            "workspace_size_mb": self.workspace_size_mb,
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for engine evaluation."""
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_qps: float
    memory_usage_mb: float
    gpu_utilization_pct: Optional[float] = None

    def __str__(self) -> str:
        """Format metrics as string."""
        return (
            f"Performance Metrics:\n"
            f"  Latency (avg/p50/p95/p99): "
            f"{self.avg_latency_ms:.2f}/{self.p50_latency_ms:.2f}/"
            f"{self.p95_latency_ms:.2f}/{self.p99_latency_ms:.2f} ms\n"
            f"  Throughput: {self.throughput_qps:.1f} QPS\n"
            f"  Memory Usage: {self.memory_usage_mb:.1f} MB\n"
            f"  GPU Utilization: "
            f"{self.gpu_utilization_pct:.1f}%" if self.gpu_utilization_pct else "N/A"
        )


class EngineOptimizer:
    """
    TensorRT engine optimization and analysis toolkit.

    Provides tools for analyzing and optimizing TensorRT engines,
    including performance profiling, layer fusion analysis, and
    optimization recommendations.

    Example:
        >>> optimizer = EngineOptimizer(engine)
        >>> profile = OptimizationProfile(
        ...     strategy=OptimizationStrategy.LOW_LATENCY,
        ...     target_batch_size=1
        ... )
        >>> metrics = optimizer.benchmark(test_data, profile)
        >>> recommendations = optimizer.get_optimization_recommendations(metrics)
    """

    def __init__(self, engine: trt.ICudaEngine):
        """
        Initialize engine optimizer.

        Args:
            engine: TensorRT engine to optimize/analyze
        """
        self.engine = engine
        self.context = engine.create_execution_context()

        # Analyze engine structure
        self._analyze_engine()

        logger.info("Initialized EngineOptimizer")

    def _analyze_engine(self) -> None:
        """Analyze engine structure and characteristics."""
        self.num_layers = 0  # TensorRT doesn't expose layer count in runtime
        self.num_bindings = self.engine.num_bindings
        self.memory_size_mb = self.engine.device_memory_size / (1024 ** 2)

        # Collect input/output information
        self.inputs = {}
        self.outputs = {}

        for i in range(self.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(i)
            dtype = self.engine.get_binding_dtype(i)

            binding_info = {
                "shape": shape,
                "dtype": dtype,
                "index": i,
            }

            if self.engine.binding_is_input(i):
                self.inputs[name] = binding_info
            else:
                self.outputs[name] = binding_info

        logger.info(
            f"Engine analysis: {len(self.inputs)} inputs, {len(self.outputs)} outputs, "
            f"{self.memory_size_mb:.2f} MB device memory"
        )

    def benchmark(
        self,
        test_data: Dict[str, np.ndarray],
        num_iterations: int = 100,
        warmup_iterations: int = 10,
    ) -> PerformanceMetrics:
        """
        Benchmark engine performance.

        Args:
            test_data: Dictionary mapping input names to test data
            num_iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations

        Returns:
            Performance metrics
        """
        logger.info(f"Running benchmark: {warmup_iterations} warmup + {num_iterations} iterations")

        # Allocate buffers
        buffers = self._allocate_buffers(test_data)

        # Warmup
        for _ in range(warmup_iterations):
            self._run_inference(buffers)

        # Benchmark
        latencies = []
        cuda.Context.synchronize()  # Ensure previous operations complete

        for _ in range(num_iterations):
            start_time = time.perf_counter()
            self._run_inference(buffers)
            cuda.Context.synchronize()  # Wait for completion
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

        # Calculate metrics
        latencies = np.array(latencies)
        avg_latency = np.mean(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        throughput = 1000.0 / avg_latency  # QPS for batch size 1

        # Get batch size from first input
        batch_size = list(test_data.values())[0].shape[0]
        throughput *= batch_size

        metrics = PerformanceMetrics(
            avg_latency_ms=avg_latency,
            p50_latency_ms=p50_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            throughput_qps=throughput,
            memory_usage_mb=self.memory_size_mb,
        )

        logger.info(f"Benchmark complete:\n{metrics}")

        # Clean up buffers
        self._free_buffers(buffers)

        return metrics

    def _allocate_buffers(self, test_data: Dict[str, np.ndarray]) -> Dict:
        """
        Allocate GPU buffers for inference.

        Args:
            test_data: Test input data

        Returns:
            Dictionary containing buffer information
        """
        buffers = {
            "inputs": {},
            "outputs": {},
            "bindings": [None] * self.num_bindings,
        }

        # Allocate input buffers
        for name, info in self.inputs.items():
            if name in test_data:
                data = test_data[name]
                dtype = trt.nptype(info["dtype"])

                # Ensure data is contiguous
                host_mem = np.ascontiguousarray(data).astype(dtype)

                # Allocate device memory
                device_mem = cuda.mem_alloc(host_mem.nbytes)

                # Copy to device
                cuda.memcpy_htod(device_mem, host_mem)

                buffers["inputs"][name] = {
                    "host": host_mem,
                    "device": device_mem,
                }
                buffers["bindings"][info["index"]] = int(device_mem)

        # Allocate output buffers
        for name, info in self.outputs.items():
            shape = info["shape"]
            dtype = trt.nptype(info["dtype"])

            # Handle dynamic shapes
            if -1 in shape:
                # Use shape from context
                shape = self.context.get_binding_shape(info["index"])

            size = trt.volume(shape)
            host_mem = np.empty(size, dtype=dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            buffers["outputs"][name] = {
                "host": host_mem,
                "device": device_mem,
            }
            buffers["bindings"][info["index"]] = int(device_mem)

        return buffers

    def _free_buffers(self, buffers: Dict) -> None:
        """Free allocated GPU buffers."""
        for name, buf in buffers["inputs"].items():
            buf["device"].free()

        for name, buf in buffers["outputs"].items():
            buf["device"].free()

    def _run_inference(self, buffers: Dict) -> None:
        """
        Run inference with allocated buffers.

        Args:
            buffers: Pre-allocated buffer dictionary
        """
        # Execute inference
        self.context.execute_v2(bindings=buffers["bindings"])

    def profile_layers(self) -> List[Dict]:
        """
        Profile individual layer performance.

        Returns:
            List of layer profiling information
        """
        logger.info("Profiling engine layers")

        # Enable profiler
        profiler = EngineProfiler()
        self.context.profiler = profiler

        # Run inference to collect profiling data
        # Note: Requires test data, simplified here
        dummy_data = self._create_dummy_inputs()
        buffers = self._allocate_buffers(dummy_data)
        self._run_inference(buffers)
        self._free_buffers(buffers)

        # Get profiling results
        results = profiler.get_results()

        logger.info(f"Profiled {len(results)} layers")
        return results

    def _create_dummy_inputs(self) -> Dict[str, np.ndarray]:
        """Create dummy input data for profiling."""
        dummy_data = {}
        for name, info in self.inputs.items():
            shape = info["shape"]
            dtype = trt.nptype(info["dtype"])

            # Handle dynamic shapes
            if -1 in shape:
                shape = tuple(1 if s == -1 else s for s in shape)

            dummy_data[name] = np.random.randn(*shape).astype(dtype)

        return dummy_data

    def get_optimization_recommendations(
        self,
        metrics: PerformanceMetrics,
        target_profile: OptimizationProfile,
    ) -> List[str]:
        """
        Generate optimization recommendations based on metrics.

        Args:
            metrics: Current performance metrics
            target_profile: Target optimization profile

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        # Latency recommendations
        if target_profile.target_latency_ms:
            if metrics.avg_latency_ms > target_profile.target_latency_ms:
                recommendations.append(
                    f"Latency ({metrics.avg_latency_ms:.2f}ms) exceeds target "
                    f"({target_profile.target_latency_ms}ms). Consider:"
                )
                recommendations.append("  - Use FP16 or INT8 precision")
                recommendations.append("  - Reduce batch size for lower latency")
                recommendations.append("  - Enable layer fusion optimizations")

        # Throughput recommendations
        if target_profile.strategy == OptimizationStrategy.HIGH_THROUGHPUT:
            recommendations.append("For maximum throughput:")
            recommendations.append(f"  - Increase batch size (current: {target_profile.target_batch_size})")
            recommendations.append("  - Use dynamic batching in serving layer")
            recommendations.append("  - Consider multiple engine instances")

        # Memory recommendations
        if target_profile.max_memory_mb:
            if metrics.memory_usage_mb > target_profile.max_memory_mb:
                recommendations.append(
                    f"Memory usage ({metrics.memory_usage_mb:.1f}MB) exceeds limit "
                    f"({target_profile.max_memory_mb}MB). Consider:"
                )
                recommendations.append("  - Reduce workspace size")
                recommendations.append("  - Use lower precision (INT8)")
                recommendations.append("  - Reduce batch size")

        # P99 latency recommendations
        p99_variance = metrics.p99_latency_ms / metrics.avg_latency_ms
        if p99_variance > 2.0:
            recommendations.append(
                f"High P99 latency variance ({p99_variance:.2f}x). Consider:"
            )
            recommendations.append("  - Enable CUDA MPS for consistent performance")
            recommendations.append("  - Reduce concurrent workloads")
            recommendations.append("  - Pin GPU clock speeds")

        if not recommendations:
            recommendations.append("Performance meets target profile. No optimizations needed.")

        return recommendations

    def compare_precisions(
        self,
        engines: Dict[str, trt.ICudaEngine],
        test_data: Dict[str, np.ndarray],
    ) -> Dict[str, PerformanceMetrics]:
        """
        Compare performance across different precision modes.

        Args:
            engines: Dictionary mapping precision name to engine
            test_data: Test input data

        Returns:
            Dictionary mapping precision name to metrics
        """
        results = {}

        logger.info(f"Comparing {len(engines)} precision modes")

        for precision, engine in engines.items():
            logger.info(f"Benchmarking {precision} precision")
            optimizer = EngineOptimizer(engine)
            metrics = optimizer.benchmark(test_data)
            results[precision] = metrics

        # Log comparison
        logger.info("\nPrecision Comparison:")
        for precision, metrics in results.items():
            logger.info(f"{precision}:")
            logger.info(f"  Latency: {metrics.avg_latency_ms:.2f}ms")
            logger.info(f"  Throughput: {metrics.throughput_qps:.1f} QPS")
            logger.info(f"  Memory: {metrics.memory_usage_mb:.1f} MB")

        return results


class EngineProfiler(trt.IProfiler):
    """
    Custom TensorRT profiler for layer-level performance analysis.

    Collects timing information for individual layers during execution.
    """

    def __init__(self):
        super().__init__()
        self.layer_times = []

    def report_layer_time(self, layer_name: str, ms: float) -> None:
        """
        Called by TensorRT for each layer execution.

        Args:
            layer_name: Name of the layer
            ms: Execution time in milliseconds
        """
        self.layer_times.append({
            "name": layer_name,
            "time_ms": ms,
        })

    def get_results(self) -> List[Dict]:
        """
        Get profiling results.

        Returns:
            List of layer timing information
        """
        # Sort by time (descending)
        sorted_results = sorted(
            self.layer_times,
            key=lambda x: x["time_ms"],
            reverse=True
        )
        return sorted_results

    def print_summary(self, top_n: int = 10) -> None:
        """
        Print summary of top N slowest layers.

        Args:
            top_n: Number of layers to display
        """
        results = self.get_results()
        total_time = sum(r["time_ms"] for r in results)

        logger.info(f"\nTop {top_n} slowest layers (Total: {total_time:.2f}ms):")
        for i, result in enumerate(results[:top_n], 1):
            pct = (result["time_ms"] / total_time) * 100
            logger.info(f"{i}. {result['name']}: {result['time_ms']:.3f}ms ({pct:.1f}%)")


# Predefined optimization profiles
OPTIMIZATION_PROFILES = {
    "low_latency": OptimizationProfile(
        strategy=OptimizationStrategy.LOW_LATENCY,
        target_batch_size=1,
        precision_mode="fp16",
        workspace_size_mb=512,
    ),
    "high_throughput": OptimizationProfile(
        strategy=OptimizationStrategy.HIGH_THROUGHPUT,
        target_batch_size=32,
        precision_mode="fp16",
        workspace_size_mb=2048,
    ),
    "int8_optimized": OptimizationProfile(
        strategy=OptimizationStrategy.BALANCED,
        target_batch_size=16,
        precision_mode="int8",
        workspace_size_mb=1024,
    ),
}


# Example usage
if __name__ == "__main__":
    # This would typically be used with a real engine
    logger.info("TensorRT Engine Optimizer ready for use")
    logger.info(f"Available profiles: {list(OPTIMIZATION_PROFILES.keys())}")
