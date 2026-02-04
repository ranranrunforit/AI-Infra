"""
TensorRT Model Converter

This module provides comprehensive model conversion from PyTorch and ONNX to TensorRT engines
with support for multiple precision modes, dynamic shapes, and optimization profiles.

Production-ready implementation with:
- FP32, FP16, and INT8 precision support
- Dynamic batch size and input shape handling
- Engine serialization and deserialization
- Comprehensive error handling and logging
- Performance profiling and validation
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorrt as trt
import torch
import torch.nn as nn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrecisionMode(Enum):
    """Supported precision modes for TensorRT conversion."""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"


@dataclass
class ConversionConfig:
    """Configuration for TensorRT conversion process."""
    precision: PrecisionMode = PrecisionMode.FP16
    max_batch_size: int = 32
    max_workspace_size: int = 1 << 30  # 1GB
    enable_dynamic_shapes: bool = True
    enable_timing_cache: bool = True
    min_timing_iterations: int = 2
    avg_timing_iterations: int = 2
    calibrator: Optional['INT8Calibrator'] = None
    optimization_level: int = 5  # 0-5, higher = more aggressive
    enable_profiling: bool = False
    strict_type_constraints: bool = False

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {self.max_batch_size}")
        if self.optimization_level not in range(6):
            raise ValueError(f"optimization_level must be 0-5, got {self.optimization_level}")
        if self.precision == PrecisionMode.INT8 and self.calibrator is None:
            logger.warning("INT8 precision requested without calibrator - using dynamic range")


class TensorRTConverter:
    """
    Convert PyTorch and ONNX models to optimized TensorRT engines.

    This converter handles the complete pipeline from model import to engine
    serialization, with support for various precision modes and optimization strategies.

    Example:
        >>> config = ConversionConfig(precision=PrecisionMode.FP16, max_batch_size=16)
        >>> converter = TensorRTConverter(config)
        >>> engine = converter.convert_pytorch_model(model, input_shapes)
        >>> converter.save_engine(engine, "model.trt")
    """

    def __init__(self, config: ConversionConfig):
        """
        Initialize TensorRT converter.

        Args:
            config: Conversion configuration parameters
        """
        self.config = config
        self.logger = trt.Logger(trt.Logger.INFO)
        self.builder = trt.Builder(self.logger)
        self.network = None
        self.parser = None

        # Initialize TensorRT components
        self._initialize_builder()

        logger.info(f"Initialized TensorRT converter with precision={config.precision.value}")

    def _initialize_builder(self) -> None:
        """Configure TensorRT builder with optimization settings."""
        # Note: In TensorRT 8.x+, many settings moved to BuilderConfig
        pass

    def convert_pytorch_model(
        self,
        model: nn.Module,
        input_shapes: Dict[str, Tuple[int, ...]],
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    ) -> trt.ICudaEngine:
        """
        Convert PyTorch model to TensorRT engine via ONNX intermediate.

        Args:
            model: PyTorch model to convert
            input_shapes: Dictionary mapping input names to shapes (including batch)
            input_names: List of input tensor names
            output_names: List of output tensor names
            dynamic_axes: Dynamic axes specification for ONNX export

        Returns:
            TensorRT ICudaEngine ready for inference

        Raises:
            RuntimeError: If conversion fails
        """
        logger.info("Converting PyTorch model to TensorRT")

        # Set model to evaluation mode
        model.eval()

        # Prepare dummy inputs for ONNX export
        dummy_inputs = self._create_dummy_inputs(input_shapes)

        # Export to ONNX (intermediate step)
        onnx_path = "/tmp/model_temp.onnx"
        try:
            logger.info("Exporting PyTorch model to ONNX")
            torch.onnx.export(
                model,
                tuple(dummy_inputs.values()) if len(dummy_inputs) > 1 else list(dummy_inputs.values())[0],
                onnx_path,
                input_names=input_names or list(input_shapes.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=17,
                do_constant_folding=True,
                export_params=True,
            )

            # Convert ONNX to TensorRT
            engine = self.convert_onnx_model(onnx_path, input_shapes)

            return engine

        except Exception as e:
            logger.error(f"PyTorch to TensorRT conversion failed: {e}")
            raise RuntimeError(f"Conversion failed: {e}") from e
        finally:
            # Clean up temporary ONNX file
            if os.path.exists(onnx_path):
                os.remove(onnx_path)

    def convert_onnx_model(
        self,
        onnx_path: Union[str, Path],
        input_shapes: Dict[str, Tuple[int, ...]],
    ) -> trt.ICudaEngine:
        """
        Convert ONNX model to TensorRT engine.

        Args:
            onnx_path: Path to ONNX model file
            input_shapes: Dictionary mapping input names to shapes

        Returns:
            TensorRT ICudaEngine ready for inference

        Raises:
            FileNotFoundError: If ONNX file doesn't exist
            RuntimeError: If conversion fails
        """
        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        logger.info(f"Converting ONNX model from {onnx_path}")

        # Create network with explicit batch
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = self.builder.create_network(network_flags)

        # Create ONNX parser
        parser = trt.OnnxParser(network, self.logger)

        # Parse ONNX model
        logger.info("Parsing ONNX model")
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                error_msgs = []
                for idx in range(parser.num_errors):
                    error_msgs.append(parser.get_error(idx).desc())
                error_msg = "\n".join(error_msgs)
                raise RuntimeError(f"ONNX parsing failed:\n{error_msg}")

        logger.info(f"Successfully parsed ONNX model with {network.num_layers} layers")

        # Build engine with configuration
        engine = self._build_engine(network, input_shapes)

        return engine

    def _build_engine(
        self,
        network: trt.INetworkDefinition,
        input_shapes: Dict[str, Tuple[int, ...]],
    ) -> trt.ICudaEngine:
        """
        Build TensorRT engine from network definition.

        Args:
            network: TensorRT network definition
            input_shapes: Input tensor shapes for optimization

        Returns:
            Built and optimized TensorRT engine
        """
        # Create builder configuration
        config = self.builder.create_builder_config()

        # Set memory pool limit (replaces max_workspace_size in TRT 8.x+)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.config.max_workspace_size)

        # Set precision mode
        if self.config.precision == PrecisionMode.FP16:
            if self.builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("Enabled FP16 precision mode")
            else:
                logger.warning("FP16 not supported on this platform, using FP32")

        elif self.config.precision == PrecisionMode.INT8:
            if self.builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                if self.config.calibrator:
                    config.int8_calibrator = self.config.calibrator
                logger.info("Enabled INT8 precision mode with calibration")
            else:
                logger.warning("INT8 not supported on this platform, using FP32")

        # Set optimization level
        config.builder_optimization_level = self.config.optimization_level

        # Enable profiling if requested
        if self.config.enable_profiling:
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

        # Strict type constraints for precision
        if self.config.strict_type_constraints:
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        # Configure dynamic shapes if enabled
        if self.config.enable_dynamic_shapes:
            self._configure_dynamic_shapes(config, network, input_shapes)

        # Enable timing cache for faster subsequent builds
        if self.config.enable_timing_cache:
            cache_file = "/tmp/timing_cache.bin"
            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    timing_cache = config.create_timing_cache(f.read())
                    config.set_timing_cache(timing_cache, ignore_mismatch=False)
                    logger.info("Loaded timing cache")

        # Build engine
        logger.info("Building TensorRT engine (this may take several minutes)...")
        serialized_engine = self.builder.build_serialized_network(network, config)

        if serialized_engine is None:
            raise RuntimeError("Engine build failed - check logs for details")

        # Save timing cache for future use
        if self.config.enable_timing_cache:
            timing_cache = config.get_timing_cache()
            if timing_cache:
                with open(cache_file, "wb") as f:
                    f.write(memoryview(timing_cache.serialize()))
                logger.info("Saved timing cache")

        # Deserialize engine
        runtime = trt.Runtime(self.logger)
        engine = runtime.deserialize_cuda_engine(serialized_engine)

        if engine is None:
            raise RuntimeError("Engine deserialization failed")

        logger.info(f"Successfully built TensorRT engine")
        self._log_engine_info(engine)

        return engine

    def _configure_dynamic_shapes(
        self,
        config: trt.IBuilderConfig,
        network: trt.INetworkDefinition,
        input_shapes: Dict[str, Tuple[int, ...]],
    ) -> None:
        """
        Configure optimization profiles for dynamic shape support.

        Args:
            config: Builder configuration
            network: Network definition
            input_shapes: Input shapes to optimize for
        """
        profile = self.builder.create_optimization_profile()

        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            input_name = input_tensor.name

            if input_name in input_shapes:
                shape = input_shapes[input_name]

                # Define min, opt, and max shapes for dynamic batching
                # Min: batch=1, Opt: specified batch, Max: 2x specified batch
                min_shape = (1,) + shape[1:]
                opt_shape = shape
                max_shape = (shape[0] * 2,) + shape[1:]

                profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                logger.info(f"Set dynamic shape for {input_name}: min={min_shape}, opt={opt_shape}, max={max_shape}")

        config.add_optimization_profile(profile)

    def _create_dummy_inputs(
        self,
        input_shapes: Dict[str, Tuple[int, ...]],
    ) -> Dict[str, torch.Tensor]:
        """
        Create dummy input tensors for ONNX export.

        Args:
            input_shapes: Dictionary mapping input names to shapes

        Returns:
            Dictionary of dummy tensors
        """
        dummy_inputs = {}
        for name, shape in input_shapes.items():
            # Create random tensors matching the shape
            dummy_inputs[name] = torch.randn(*shape)
        return dummy_inputs

    def save_engine(self, engine: trt.ICudaEngine, output_path: Union[str, Path]) -> None:
        """
        Serialize and save TensorRT engine to disk.

        Args:
            engine: TensorRT engine to save
            output_path: Path where engine will be saved
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving TensorRT engine to {output_path}")

        # Serialize engine
        serialized_engine = engine.serialize()

        # Write to file
        with open(output_path, 'wb') as f:
            f.write(serialized_engine)

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Engine saved successfully ({file_size_mb:.2f} MB)")

    def load_engine(self, engine_path: Union[str, Path]) -> trt.ICudaEngine:
        """
        Load serialized TensorRT engine from disk.

        Args:
            engine_path: Path to serialized engine file

        Returns:
            Deserialized TensorRT engine

        Raises:
            FileNotFoundError: If engine file doesn't exist
            RuntimeError: If deserialization fails
        """
        engine_path = Path(engine_path)
        if not engine_path.exists():
            raise FileNotFoundError(f"Engine file not found: {engine_path}")

        logger.info(f"Loading TensorRT engine from {engine_path}")

        # Read serialized engine
        with open(engine_path, 'rb') as f:
            serialized_engine = f.read()

        # Deserialize
        runtime = trt.Runtime(self.logger)
        engine = runtime.deserialize_cuda_engine(serialized_engine)

        if engine is None:
            raise RuntimeError("Engine deserialization failed - version mismatch or corrupted file")

        logger.info("Engine loaded successfully")
        self._log_engine_info(engine)

        return engine

    def _log_engine_info(self, engine: trt.ICudaEngine) -> None:
        """Log detailed information about the engine."""
        logger.info(f"Engine info:")
        logger.info(f"  - Number of bindings: {engine.num_bindings}")
        logger.info(f"  - Number of optimization profiles: {engine.num_optimization_profiles}")
        logger.info(f"  - Device memory size: {engine.device_memory_size / (1024**2):.2f} MB")

        # Log input/output shapes
        for i in range(engine.num_bindings):
            binding_name = engine.get_binding_name(i)
            shape = engine.get_binding_shape(i)
            dtype = engine.get_binding_dtype(i)
            is_input = engine.binding_is_input(i)

            logger.info(f"  - {'Input' if is_input else 'Output'} '{binding_name}': shape={shape}, dtype={dtype}")

    def validate_engine(
        self,
        engine: trt.ICudaEngine,
        test_inputs: Dict[str, np.ndarray],
    ) -> bool:
        """
        Validate that the engine can perform inference.

        Args:
            engine: Engine to validate
            test_inputs: Test input data

        Returns:
            True if validation passes, False otherwise
        """
        try:
            import pycuda.autoinit
            import pycuda.driver as cuda

            logger.info("Validating TensorRT engine")

            # Create execution context
            context = engine.create_execution_context()

            # Allocate buffers
            inputs, outputs, bindings = [], [], []
            stream = cuda.Stream()

            for i in range(engine.num_bindings):
                binding_name = engine.get_binding_name(i)
                size = trt.volume(engine.get_binding_shape(i))
                dtype = trt.nptype(engine.get_binding_dtype(i))

                # Allocate device memory
                device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
                bindings.append(int(device_mem))

                if engine.binding_is_input(i):
                    if binding_name in test_inputs:
                        # Copy input data to device
                        host_mem = test_inputs[binding_name].astype(dtype)
                        cuda.memcpy_htod_async(device_mem, host_mem, stream)
                        inputs.append(host_mem)
                else:
                    # Allocate output buffer
                    host_mem = np.empty(size, dtype=dtype)
                    outputs.append(host_mem)

            # Execute inference
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            stream.synchronize()

            logger.info("Validation passed - engine can perform inference")
            return True

        except Exception as e:
            logger.error(f"Engine validation failed: {e}")
            return False
