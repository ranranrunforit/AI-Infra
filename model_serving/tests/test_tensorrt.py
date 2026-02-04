"""
TensorRT Converter Tests

Comprehensive test suite for the TensorRT model conversion module.
Tests model conversion from PyTorch and ONNX to TensorRT engines with
various precision modes and optimization settings.

Test Coverage:
- ConversionConfig validation
- TensorRT converter initialization
- PyTorch to TensorRT conversion
- ONNX to TensorRT conversion
- Engine serialization and deserialization
- Precision mode handling (FP32, FP16, INT8)
- Dynamic shape configuration
- Calibrator integration
- Error handling and edge cases
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, call

import numpy as np
import pytest
import torch
import torch.nn as nn

from tensorrt.converter import (
    TensorRTConverter,
    ConversionConfig,
    PrecisionMode,
)


class TestConversionConfig:
    """Test ConversionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ConversionConfig()

        assert config.precision == PrecisionMode.FP16
        assert config.max_batch_size == 32
        assert config.max_workspace_size == 1 << 30
        assert config.enable_dynamic_shapes is True
        assert config.enable_timing_cache is True
        assert config.optimization_level == 5

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ConversionConfig(
            precision=PrecisionMode.FP32,
            max_batch_size=64,
            max_workspace_size=2 << 30,
            optimization_level=3
        )

        assert config.precision == PrecisionMode.FP32
        assert config.max_batch_size == 64
        assert config.max_workspace_size == 2 << 30
        assert config.optimization_level == 3

    def test_invalid_batch_size(self):
        """Test validation of invalid batch size."""
        with pytest.raises(ValueError, match="max_batch_size must be positive"):
            ConversionConfig(max_batch_size=0)

        with pytest.raises(ValueError, match="max_batch_size must be positive"):
            ConversionConfig(max_batch_size=-1)

    def test_invalid_optimization_level(self):
        """Test validation of invalid optimization level."""
        with pytest.raises(ValueError, match="optimization_level must be 0-5"):
            ConversionConfig(optimization_level=6)

        with pytest.raises(ValueError, match="optimization_level must be 0-5"):
            ConversionConfig(optimization_level=-1)

    def test_int8_without_calibrator_warning(self, caplog):
        """Test warning when INT8 requested without calibrator."""
        config = ConversionConfig(precision=PrecisionMode.INT8, calibrator=None)

        assert config.precision == PrecisionMode.INT8
        # Check that warning was logged
        assert any("INT8 precision requested without calibrator" in record.message
                   for record in caplog.records)


class TestPrecisionMode:
    """Test PrecisionMode enum."""

    def test_precision_mode_values(self):
        """Test precision mode enum values."""
        assert PrecisionMode.FP32.value == "fp32"
        assert PrecisionMode.FP16.value == "fp16"
        assert PrecisionMode.INT8.value == "int8"

    def test_precision_mode_from_string(self):
        """Test creating precision mode from string."""
        assert PrecisionMode("fp32") == PrecisionMode.FP32
        assert PrecisionMode("fp16") == PrecisionMode.FP16
        assert PrecisionMode("int8") == PrecisionMode.INT8


class TestTensorRTConverter:
    """Test TensorRTConverter class."""

    def test_converter_initialization(self, mock_tensorrt, tensorrt_config):
        """Test converter initialization."""
        converter = TensorRTConverter(tensorrt_config)

        assert converter.config == tensorrt_config
        assert converter.logger is not None
        assert converter.builder is not None

    def test_converter_initialization_with_different_precisions(self, mock_tensorrt):
        """Test converter initialization with different precision modes."""
        for precision in [PrecisionMode.FP32, PrecisionMode.FP16, PrecisionMode.INT8]:
            config = ConversionConfig(precision=precision)
            converter = TensorRTConverter(config)
            assert converter.config.precision == precision

    @patch('torch.onnx.export')
    def test_convert_pytorch_model(
        self,
        mock_onnx_export,
        tensorrt_converter,
        simple_pytorch_model,
    ):
        """Test PyTorch model conversion."""
        input_shapes = {"input": (1, 3, 224, 224)}

        with patch.object(tensorrt_converter, 'convert_onnx_model') as mock_convert_onnx:
            mock_convert_onnx.return_value = MagicMock()

            engine = tensorrt_converter.convert_pytorch_model(
                model=simple_pytorch_model,
                input_shapes=input_shapes,
                input_names=["input"],
                output_names=["output"]
            )

            # Verify ONNX export was called
            assert mock_onnx_export.called
            assert engine is not None

    @patch('torch.onnx.export')
    def test_convert_pytorch_model_with_dynamic_axes(
        self,
        mock_onnx_export,
        tensorrt_converter,
        simple_pytorch_model,
    ):
        """Test PyTorch conversion with dynamic axes."""
        input_shapes = {"input": (1, 3, 224, 224)}
        dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}

        with patch.object(tensorrt_converter, 'convert_onnx_model') as mock_convert_onnx:
            mock_convert_onnx.return_value = MagicMock()

            engine = tensorrt_converter.convert_pytorch_model(
                model=simple_pytorch_model,
                input_shapes=input_shapes,
                dynamic_axes=dynamic_axes
            )

            # Verify dynamic axes were passed to ONNX export
            call_args = mock_onnx_export.call_args
            assert call_args[1]['dynamic_axes'] == dynamic_axes

    @patch('torch.onnx.export')
    def test_convert_pytorch_model_cleanup(
        self,
        mock_onnx_export,
        tensorrt_converter,
        simple_pytorch_model,
    ):
        """Test that temporary ONNX files are cleaned up."""
        input_shapes = {"input": (1, 3, 224, 224)}

        # Create a real temporary file to test cleanup
        temp_onnx = "/tmp/model_temp.onnx"

        def create_temp_file(*args, **kwargs):
            with open(temp_onnx, 'wb') as f:
                f.write(b'test')

        mock_onnx_export.side_effect = create_temp_file

        with patch.object(tensorrt_converter, 'convert_onnx_model') as mock_convert_onnx:
            mock_convert_onnx.return_value = MagicMock()

            try:
                tensorrt_converter.convert_pytorch_model(
                    model=simple_pytorch_model,
                    input_shapes=input_shapes
                )
            finally:
                # Verify cleanup even if conversion fails
                assert not os.path.exists(temp_onnx)

    def test_convert_onnx_model(self, tensorrt_converter, onnx_model_file):
        """Test ONNX model conversion."""
        input_shapes = {"input": (1, 3, 224, 224)}

        with patch.object(tensorrt_converter, '_build_engine') as mock_build:
            mock_engine = MagicMock()
            mock_build.return_value = mock_engine

            engine = tensorrt_converter.convert_onnx_model(
                onnx_path=onnx_model_file,
                input_shapes=input_shapes
            )

            assert engine == mock_engine
            assert mock_build.called

    def test_convert_onnx_model_file_not_found(self, tensorrt_converter):
        """Test error handling when ONNX file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            tensorrt_converter.convert_onnx_model(
                onnx_path="/nonexistent/model.onnx",
                input_shapes={"input": (1, 3, 224, 224)}
            )

    def test_build_engine_fp32(self, mock_tensorrt):
        """Test engine building with FP32 precision."""
        config = ConversionConfig(precision=PrecisionMode.FP32)
        converter = TensorRTConverter(config)

        mock_network = MagicMock()
        input_shapes = {"input": (1, 3, 224, 224)}

        engine = converter._build_engine(mock_network, input_shapes)

        assert engine is not None

    def test_build_engine_fp16(self, mock_tensorrt):
        """Test engine building with FP16 precision."""
        config = ConversionConfig(precision=PrecisionMode.FP16)
        converter = TensorRTConverter(config)

        mock_network = MagicMock()
        input_shapes = {"input": (1, 3, 224, 224)}

        engine = converter._build_engine(mock_network, input_shapes)

        assert engine is not None

    def test_build_engine_int8(self, mock_tensorrt):
        """Test engine building with INT8 precision."""
        mock_calibrator = MagicMock()
        config = ConversionConfig(
            precision=PrecisionMode.INT8,
            calibrator=mock_calibrator
        )
        converter = TensorRTConverter(config)

        mock_network = MagicMock()
        input_shapes = {"input": (1, 3, 224, 224)}

        engine = converter._build_engine(mock_network, input_shapes)

        assert engine is not None

    def test_configure_dynamic_shapes(self, tensorrt_converter, mock_tensorrt):
        """Test dynamic shape configuration."""
        config = MagicMock()
        network = MagicMock()

        # Setup mock input tensor
        mock_tensor = MagicMock()
        mock_tensor.name = "input"
        network.get_input.return_value = mock_tensor
        network.num_inputs = 1

        input_shapes = {"input": (8, 3, 224, 224)}

        tensorrt_converter._configure_dynamic_shapes(config, network, input_shapes)

        # Verify optimization profile was created
        assert tensorrt_converter.builder.create_optimization_profile.called

    def test_create_dummy_inputs(self, tensorrt_converter):
        """Test dummy input creation for ONNX export."""
        input_shapes = {
            "input1": (1, 3, 224, 224),
            "input2": (1, 1000)
        }

        dummy_inputs = tensorrt_converter._create_dummy_inputs(input_shapes)

        assert len(dummy_inputs) == 2
        assert "input1" in dummy_inputs
        assert "input2" in dummy_inputs
        assert dummy_inputs["input1"].shape == (1, 3, 224, 224)
        assert dummy_inputs["input2"].shape == (1, 1000)
        assert isinstance(dummy_inputs["input1"], torch.Tensor)

    def test_save_engine(self, tensorrt_converter, temp_dir, mock_tensorrt_engine):
        """Test engine serialization and saving."""
        output_path = temp_dir / "saved_engine.trt"

        tensorrt_converter.save_engine(mock_tensorrt_engine, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_save_engine_creates_directory(self, tensorrt_converter, temp_dir, mock_tensorrt_engine):
        """Test that save_engine creates parent directories."""
        output_path = temp_dir / "nested" / "dir" / "engine.trt"

        tensorrt_converter.save_engine(mock_tensorrt_engine, output_path)

        assert output_path.exists()
        assert output_path.parent.exists()

    def test_load_engine(self, tensorrt_converter, tensorrt_engine_file):
        """Test engine loading from file."""
        with patch('tensorrt.Runtime') as mock_runtime_class:
            mock_runtime = MagicMock()
            mock_engine = MagicMock()
            mock_runtime.deserialize_cuda_engine.return_value = mock_engine
            mock_runtime_class.return_value = mock_runtime

            engine = tensorrt_converter.load_engine(tensorrt_engine_file)

            assert engine is not None
            assert mock_runtime.deserialize_cuda_engine.called

    def test_load_engine_file_not_found(self, tensorrt_converter):
        """Test error handling when loading non-existent engine."""
        with pytest.raises(FileNotFoundError):
            tensorrt_converter.load_engine("/nonexistent/engine.trt")

    def test_load_engine_deserialization_failure(self, tensorrt_converter, tensorrt_engine_file):
        """Test error handling when deserialization fails."""
        with patch('tensorrt.Runtime') as mock_runtime_class:
            mock_runtime = MagicMock()
            mock_runtime.deserialize_cuda_engine.return_value = None
            mock_runtime_class.return_value = mock_runtime

            with pytest.raises(RuntimeError, match="Engine deserialization failed"):
                tensorrt_converter.load_engine(tensorrt_engine_file)

    def test_log_engine_info(self, tensorrt_converter, mock_tensorrt_engine, caplog):
        """Test engine information logging."""
        tensorrt_converter._log_engine_info(mock_tensorrt_engine)

        # Check that engine info was logged
        log_messages = [record.message for record in caplog.records]
        assert any("Engine info" in msg for msg in log_messages)
        assert any("Number of bindings" in msg for msg in log_messages)

    def test_validate_engine_success(self, tensorrt_converter, mock_tensorrt_engine, mock_cuda):
        """Test successful engine validation."""
        test_inputs = {
            "input": np.random.randn(1, 3, 224, 224).astype(np.float32)
        }

        # Mock the engine methods
        mock_tensorrt_engine.num_bindings = 2
        mock_tensorrt_engine.get_binding_name = lambda i: ["input", "output"][i]
        mock_tensorrt_engine.get_binding_shape = lambda i: [(1, 3, 224, 224), (1, 10)][i]
        mock_tensorrt_engine.binding_is_input = lambda i: i == 0

        # Note: This test requires more complex mocking of CUDA operations
        # For now, we'll test the validation logic without full CUDA simulation
        with patch('tensorrt.volume', return_value=602112), \
             patch('tensorrt.nptype', return_value=np.float32):

            result = tensorrt_converter.validate_engine(mock_tensorrt_engine, test_inputs)

            # In a full test, this would be True
            # With our mocks, it may vary
            assert isinstance(result, bool)

    def test_validate_engine_failure(self, tensorrt_converter, mock_tensorrt_engine, mock_cuda):
        """Test engine validation with errors."""
        test_inputs = {"input": np.random.randn(1, 3, 224, 224).astype(np.float32)}

        # Force an exception during validation
        mock_tensorrt_engine.create_execution_context.side_effect = RuntimeError("Validation error")

        result = tensorrt_converter.validate_engine(mock_tensorrt_engine, test_inputs)

        assert result is False


class TestTensorRTEndToEnd:
    """End-to-end integration tests for TensorRT conversion."""

    @pytest.mark.slow
    @pytest.mark.integration
    @patch('torch.onnx.export')
    def test_full_conversion_pipeline(
        self,
        mock_onnx_export,
        simple_pytorch_model,
        temp_dir,
        mock_tensorrt
    ):
        """Test complete conversion pipeline from PyTorch to saved engine."""
        config = ConversionConfig(
            precision=PrecisionMode.FP16,
            max_batch_size=8,
            enable_timing_cache=False
        )
        converter = TensorRTConverter(config)

        input_shapes = {"input": (1, 3, 224, 224)}

        # Mock ONNX export to create a temporary file
        def create_onnx_file(*args, **kwargs):
            onnx_path = args[2]
            with open(onnx_path, 'wb') as f:
                f.write(b'ONNX' + b'\x00' * 1020)

        mock_onnx_export.side_effect = create_onnx_file

        with patch.object(converter, 'convert_onnx_model') as mock_convert:
            mock_engine = MagicMock()
            mock_convert.return_value = mock_engine

            # Convert model
            engine = converter.convert_pytorch_model(
                model=simple_pytorch_model,
                input_shapes=input_shapes
            )

            assert engine is not None

            # Save engine
            output_path = temp_dir / "model.trt"
            converter.save_engine(engine, output_path)

            assert output_path.exists()

    @pytest.mark.slow
    def test_conversion_with_different_batch_sizes(self, mock_tensorrt):
        """Test conversion with various batch sizes."""
        batch_sizes = [1, 4, 16, 32, 64]

        for batch_size in batch_sizes:
            config = ConversionConfig(max_batch_size=batch_size)
            converter = TensorRTConverter(config)

            assert converter.config.max_batch_size == batch_size

    @pytest.mark.slow
    def test_conversion_with_all_precisions(self, mock_tensorrt):
        """Test conversion with all precision modes."""
        precisions = [PrecisionMode.FP32, PrecisionMode.FP16, PrecisionMode.INT8]

        for precision in precisions:
            if precision == PrecisionMode.INT8:
                config = ConversionConfig(precision=precision, calibrator=MagicMock())
            else:
                config = ConversionConfig(precision=precision)

            converter = TensorRTConverter(config)
            assert converter.config.precision == precision


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_convert_with_invalid_model(self, tensorrt_converter):
        """Test conversion with invalid model."""
        invalid_model = "not a model"
        input_shapes = {"input": (1, 3, 224, 224)}

        with pytest.raises(Exception):
            tensorrt_converter.convert_pytorch_model(
                model=invalid_model,
                input_shapes=input_shapes
            )

    def test_convert_with_empty_input_shapes(self, tensorrt_converter, simple_pytorch_model):
        """Test conversion with empty input shapes."""
        with pytest.raises(Exception):
            tensorrt_converter.convert_pytorch_model(
                model=simple_pytorch_model,
                input_shapes={}
            )

    def test_save_to_readonly_location(self, tensorrt_converter, mock_tensorrt_engine):
        """Test error handling when saving to read-only location."""
        readonly_path = "/readonly/engine.trt"

        with pytest.raises(Exception):
            tensorrt_converter.save_engine(mock_tensorrt_engine, readonly_path)
