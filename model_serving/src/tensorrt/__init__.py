"""
TensorRT Model Optimization Module

This module provides tools for converting and optimizing models with NVIDIA TensorRT.

Components:
    - converter: Convert PyTorch/ONNX models to TensorRT engines
    - calibrator: INT8 post-training quantization calibration
    - optimizer: Model optimization strategies and layer fusion

Example:
    Basic usage:

        from src.tensorrt import TensorRTConverter, Precision

        converter = TensorRTConverter()
        engine = converter.convert(
            model_path="models/resnet50.onnx",
            precision=Precision.FP16,
            max_batch_size=32
        )

        # Save optimized engine
        converter.save_engine(engine, "models/resnet50-fp16.trt")
"""

from .converter import TensorRTConverter, Precision
from .calibrator import INT8Calibrator
from .optimizer import EngineOptimizer

__all__ = [
    "TensorRTConverter",
    "Precision",
    "INT8Calibrator",
    "EngineOptimizer",
]
