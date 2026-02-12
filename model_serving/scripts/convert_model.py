#!/usr/bin/env python3
"""
Model Conversion Script for TensorRT Optimization

Converts PyTorch vision models to optimized TensorRT engines with
FP32, FP16, or INT8 precision support.

Usage:
    python scripts/convert_model.py \
        --model resnet50 \
        --precision fp16 \
        --output models/resnet50-fp16.trt

    python scripts/convert_model.py \
        --model resnet50 \
        --precision int8 \
        --batch-size 32 \
        --calibration-samples 500 \
        --output models/resnet50-int8.trt

Supported models: resnet18, resnet50, mobilenet_v2, efficientnet_b0
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
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
        model_name: Model identifier (resnet50, resnet18, mobilenet_v2, etc.)

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
    # Use weights parameter (newer torchvision API)
    try:
        model = model_map[model_name](weights="DEFAULT")
    except TypeError:
        # Fall back to older API
        model = model_map[model_name](pretrained=True)

    model.eval()
    logger.info(f"Loaded {model_name} successfully")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch model to TensorRT engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # FP16 conversion (recommended for RTX GPUs)
  python scripts/convert_model.py --model resnet50 --precision fp16 --output models/resnet50-fp16.trt

  # INT8 conversion with calibration
  python scripts/convert_model.py --model resnet50 --precision int8 --calibration-samples 200 --output models/resnet50-int8.trt
        """,
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["resnet18", "resnet50", "resnet101", "mobilenet_v2", "efficientnet_b0"],
        help="PyTorch model name",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "int8"],
        default="fp16",
        help="TensorRT precision mode (default: fp16)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for TensorRT engine file (.trt)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Maximum batch size (default: 1)",
    )
    parser.add_argument(
        "--workspace-size",
        type=int,
        default=1,
        help="TensorRT workspace size in GB (default: 1)",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=100,
        help="Number of calibration samples for INT8 (default: 100)",
    )
    parser.add_argument(
        "--dynamic-shapes",
        action="store_true",
        default=True,
        help="Enable dynamic shape support (default: True)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate engine after conversion (default: True)",
    )

    args = parser.parse_args()

    logger.info(f"Converting {args.model} to TensorRT ({args.precision})")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Workspace: {args.workspace_size}GB")
    logger.info(f"  Output: {args.output}")

    try:
        import torch
        import numpy as np
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install with: pip install torch torchvision numpy")
        sys.exit(1)

    try:
        from src.tensorrt.converter import TensorRTConverter, PrecisionMode, ConversionConfig
        from src.tensorrt.calibrator import INT8Calibrator, create_calibration_dataset
    except ImportError as e:
        logger.error(f"Failed to import TensorRT modules: {e}")
        logger.error(
            "Make sure TensorRT is installed. "
            "On RTX 5070 (Blackwell), you need CUDA 12.8+ and matching TensorRT."
        )
        sys.exit(1)

    # Load PyTorch model
    model = load_pytorch_model(args.model)

    # Define input shapes
    input_shapes = {"input": (args.batch_size, 3, 224, 224)}

    # Create INT8 calibrator if needed
    calibrator = None
    if args.precision == "int8":
        logger.info(
            f"Creating INT8 calibration dataset "
            f"({args.calibration_samples} samples)..."
        )
        calib_data = create_calibration_dataset(
            num_samples=args.calibration_samples,
            input_shape=(1, 3, 224, 224),
        )
        calibrator = INT8Calibrator(calib_data, batch_size=1)

    # Create conversion config
    config = ConversionConfig(
        precision=PrecisionMode(args.precision),
        max_batch_size=args.batch_size,
        max_workspace_size=args.workspace_size << 30,  # Convert GB to bytes
        enable_dynamic_shapes=args.dynamic_shapes,
        enable_timing_cache=True,
        calibrator=calibrator,
        optimization_level=5,
    )

    # Convert model
    logger.info("Converting to TensorRT (this may take several minutes)...")
    converter = TensorRTConverter(config)
    engine = converter.convert_pytorch_model(
        model=model,
        input_shapes=input_shapes,
        input_names=["input"],
        output_names=["output"],
    )

    # Save engine
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converter.save_engine(engine, output_path)
    logger.info(f"TensorRT engine saved to {output_path}")

    # Validate
    if args.validate:
        logger.info("Validating engine...")
        test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        if converter.validate_engine(engine, {"input": test_input}):
            logger.info("✓ Engine validation passed")
        else:
            logger.error("✗ Engine validation failed")
            sys.exit(1)

    logger.info("Done!")


if __name__ == "__main__":
    main()
