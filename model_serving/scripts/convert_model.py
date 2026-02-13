#!/usr/bin/env python3
"""
Model Conversion Script

Converts PyTorch vision models to optimized formats for serving:
- ONNX  (always available, recommended)
- TensorRT (requires tensorrt pip package)

Usage:
    # Export to ONNX (works on any system)
    python scripts/convert_model.py --model resnet18 --format onnx --output /tmp/model_cache/resnet18.onnx

    # Export to TensorRT (requires TensorRT installed)
    python scripts/convert_model.py --model resnet18 --format tensorrt --precision fp16 --output models/resnet18-fp16.trt

Supported models: resnet18, resnet50, resnet101, mobilenet_v2, efficientnet_b0
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Use /tmp for torch cache (appuser can't write to ~/.cache)
os.environ.setdefault("TORCH_HOME", "/tmp/torch_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_pytorch_model(model_name: str):
    """
    Load a pretrained PyTorch vision model.

    Args:
        model_name: Model identifier (resnet18, resnet50, mobilenet_v2, etc.)

    Returns:
        PyTorch model in eval mode
    """
    import torch
    import torchvision.models as models

    model_map = {
        "resnet18": models.resnet18,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "mobilenet_v2": models.mobilenet_v2,
        "efficientnet_b0": models.efficientnet_b0,
    }

    if model_name not in model_map:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported: {list(model_map.keys())}"
        )

    logger.info(f"Loading pretrained {model_name}...")
    try:
        model = model_map[model_name](weights="DEFAULT")
    except TypeError:
        model = model_map[model_name](pretrained=True)

    model.eval()
    logger.info(f"Loaded {model_name} successfully")
    return model


def export_to_onnx(model, output_path: Path, batch_size: int, validate: bool):
    """Export PyTorch model to ONNX format."""
    import torch
    import numpy as np

    logger.info("Exporting to ONNX...")

    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    # Export
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"ONNX model saved to {output_path} ({size_mb:.1f} MB)")

    # Validate
    if validate:
        logger.info("Validating ONNX model...")
        try:
            import onnx
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            logger.info("✓ ONNX model validation passed")
        except ImportError:
            logger.warning("onnx package not installed, skipping validation")

        try:
            import onnxruntime as ort
            session = ort.InferenceSession(str(output_path))
            test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
            outputs = session.run(None, {"input": test_input})
            logger.info(f"✓ Inference test passed (output shape: {outputs[0].shape})")
        except ImportError:
            logger.warning("onnxruntime not installed, skipping inference test")


def export_to_tensorrt(model, output_path: Path, batch_size: int, precision: str,
                       workspace_size: int, calibration_samples: int, validate: bool):
    """Export PyTorch model to TensorRT engine via ONNX."""
    import torch
    import numpy as np

    try:
        from src.tensorrt.converter import TensorRTConverter, PrecisionMode, ConversionConfig
        from src.tensorrt.calibrator import INT8Calibrator, create_calibration_dataset
    except ImportError as e:
        logger.error(f"TensorRT modules not available: {e}")
        logger.error("")
        logger.error("TensorRT is not installed in this container.")
        logger.error("Alternative: export to ONNX instead (works on any system):")
        logger.error(f"  python scripts/convert_model.py --model <name> --format onnx --output <path>.onnx")
        sys.exit(1)

    input_shapes = {"input": (batch_size, 3, 224, 224)}

    calibrator = None
    if precision == "int8":
        logger.info(f"Creating INT8 calibration dataset ({calibration_samples} samples)...")
        calib_data = create_calibration_dataset(
            num_samples=calibration_samples,
            input_shape=(1, 3, 224, 224),
        )
        calibrator = INT8Calibrator(calib_data, batch_size=1)

    config = ConversionConfig(
        precision=PrecisionMode(precision),
        max_batch_size=batch_size,
        max_workspace_size=workspace_size << 30,
        enable_dynamic_shapes=True,
        enable_timing_cache=True,
        calibrator=calibrator,
        optimization_level=5,
    )

    logger.info("Converting to TensorRT (this may take several minutes)...")
    converter = TensorRTConverter(config)
    engine = converter.convert_pytorch_model(
        model=model,
        input_shapes=input_shapes,
        input_names=["input"],
        output_names=["output"],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    converter.save_engine(engine, output_path)
    logger.info(f"TensorRT engine saved to {output_path}")

    if validate:
        logger.info("Validating engine...")
        test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        if converter.validate_engine(engine, {"input": test_input}):
            logger.info("✓ Engine validation passed")
        else:
            logger.error("✗ Engine validation failed")
            sys.exit(1)


def save_pytorch(model, output_path: Path):
    """Save as PyTorch .pt file for direct serving."""
    import torch
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, str(output_path))
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"PyTorch model saved to {output_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch model to optimized serving format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # ONNX export (recommended, works everywhere)
  python scripts/convert_model.py --model resnet18 --format onnx --output /tmp/model_cache/resnet18.onnx

  # PyTorch .pt export (simplest)
  python scripts/convert_model.py --model resnet18 --format pytorch --output /tmp/model_cache/resnet18.pt

  # TensorRT export (requires TensorRT installed)
  python scripts/convert_model.py --model resnet18 --format tensorrt --precision fp16 --output models/resnet18-fp16.trt
        """,
    )
    parser.add_argument(
        "--model", required=True,
        choices=["resnet18", "resnet50", "resnet101", "mobilenet_v2", "efficientnet_b0"],
        help="PyTorch model name",
    )
    parser.add_argument(
        "--format", required=True,
        choices=["onnx", "pytorch", "tensorrt"],
        help="Output format (onnx recommended, tensorrt requires TensorRT installed)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output file path",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "int8"],
        default="fp16",
        help="Precision mode for TensorRT (default: fp16)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Maximum batch size (default: 1)",
    )
    parser.add_argument(
        "--workspace-size", type=int, default=1,
        help="TensorRT workspace size in GB (default: 1)",
    )
    parser.add_argument(
        "--calibration-samples", type=int, default=100,
        help="Number of calibration samples for INT8 (default: 100)",
    )
    parser.add_argument(
        "--no-validate", action="store_true",
        help="Skip validation after conversion",
    )

    args = parser.parse_args()
    output_path = Path(args.output)
    validate = not args.no_validate

    logger.info(f"Converting {args.model} to {args.format}")
    logger.info(f"  Output: {args.output}")

    # Load model
    model = load_pytorch_model(args.model)

    # Convert
    if args.format == "onnx":
        export_to_onnx(model, output_path, args.batch_size, validate)
    elif args.format == "pytorch":
        save_pytorch(model, output_path)
    elif args.format == "tensorrt":
        export_to_tensorrt(
            model, output_path, args.batch_size, args.precision,
            args.workspace_size, args.calibration_samples, validate,
        )

    logger.info("Done!")


if __name__ == "__main__":
    main()
