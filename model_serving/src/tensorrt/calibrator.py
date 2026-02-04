"""
TensorRT INT8 Calibrator

This module provides INT8 post-training quantization calibration for TensorRT engines.
Calibration is required to determine optimal quantization scale factors for INT8 precision.

Implements:
- IInt8EntropyCalibrator2 for improved accuracy
- Cache-based calibration for reproducibility
- Batch-wise data feeding
- GPU memory management
"""

import logging
import os
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
import pycuda.autoinit  # Initializes CUDA context
import pycuda.driver as cuda
import tensorrt as trt

logger = logging.getLogger(__name__)


class INT8Calibrator(trt.IInt8EntropyCalibrator2):
    """
    INT8 Entropy Calibrator (v2) for TensorRT quantization.

    This calibrator uses entropy-based optimization to determine quantization scales,
    providing better accuracy than legacy calibrators. It processes calibration data
    in batches and caches results for reproducibility.

    The calibrator implements the IInt8EntropyCalibrator2 interface required by TensorRT
    for INT8 quantization.

    Example:
        >>> data_loader = create_calibration_data_loader()
        >>> calibrator = INT8Calibrator(
        ...     data_loader=data_loader,
        ...     cache_file="calibration.cache",
        ...     batch_size=32
        ... )
        >>> # Use in ConversionConfig
        >>> config = ConversionConfig(
        ...     precision=PrecisionMode.INT8,
        ...     calibrator=calibrator
        ... )
    """

    def __init__(
        self,
        data_loader: Iterator[np.ndarray],
        cache_file: Union[str, Path],
        batch_size: int = 1,
        input_shape: Optional[Tuple[int, ...]] = None,
        algorithm: trt.CalibrationAlgoType = trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2,
    ):
        """
        Initialize INT8 calibrator.

        Args:
            data_loader: Iterator yielding calibration data batches (numpy arrays)
            cache_file: Path to calibration cache file (for caching/reusing calibration)
            batch_size: Batch size for calibration
            input_shape: Expected input shape (excluding batch dimension)
            algorithm: Calibration algorithm to use
        """
        super().__init__()

        self.data_loader = data_loader
        self.cache_file = Path(cache_file)
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.algorithm = algorithm

        # Calibration state
        self.current_batch = None
        self.device_input = None
        self.batch_count = 0
        self.max_batches = 500  # Limit calibration batches for performance

        # Statistics
        self.total_samples_processed = 0

        logger.info(
            f"Initialized INT8 calibrator: batch_size={batch_size}, "
            f"cache_file={cache_file}, algorithm={algorithm}"
        )

    def get_batch_size(self) -> int:
        """
        Get the batch size used for calibration.

        Returns:
            Calibration batch size
        """
        return self.batch_size

    def get_batch(self, names: List[str]) -> Optional[List[int]]:
        """
        Get next batch of calibration data.

        Called by TensorRT during calibration to retrieve input data.

        Args:
            names: List of input tensor names

        Returns:
            List of device memory pointers for each input, or None when done
        """
        try:
            # Check if we've reached the maximum number of batches
            if self.batch_count >= self.max_batches:
                logger.info(
                    f"Calibration complete: processed {self.batch_count} batches "
                    f"({self.total_samples_processed} samples)"
                )
                return None

            # Get next batch from data loader
            try:
                batch = next(self.data_loader)
            except StopIteration:
                logger.info("Data loader exhausted, calibration complete")
                return None

            # Validate batch shape
            if batch.shape[0] != self.batch_size:
                logger.warning(
                    f"Batch size mismatch: expected {self.batch_size}, got {batch.shape[0]}. "
                    "Padding or skipping batch."
                )
                if batch.shape[0] < self.batch_size:
                    # Pad batch to match expected size
                    padding_size = self.batch_size - batch.shape[0]
                    padding = np.zeros(
                        (padding_size,) + batch.shape[1:],
                        dtype=batch.dtype
                    )
                    batch = np.concatenate([batch, padding], axis=0)
                else:
                    # Truncate batch
                    batch = batch[:self.batch_size]

            # Store input shape on first batch
            if self.input_shape is None:
                self.input_shape = batch.shape[1:]
                logger.info(f"Detected input shape: {self.input_shape}")

            # Allocate device memory on first batch
            if self.device_input is None:
                self.device_input = cuda.mem_alloc(batch.nbytes)
                logger.info(f"Allocated {batch.nbytes / (1024**2):.2f} MB of device memory")

            # Copy batch to device
            self.current_batch = np.ascontiguousarray(batch)
            cuda.memcpy_htod(self.device_input, self.current_batch)

            self.batch_count += 1
            self.total_samples_processed += batch.shape[0]

            if self.batch_count % 50 == 0:
                logger.info(f"Calibration progress: {self.batch_count} batches processed")

            # Return device pointers for all inputs (typically just one)
            return [int(self.device_input)]

        except Exception as e:
            logger.error(f"Error in get_batch: {e}")
            return None

    def read_calibration_cache(self) -> Optional[bytes]:
        """
        Read calibration cache from disk.

        If a cache file exists, TensorRT will use cached calibration data
        instead of re-running calibration.

        Returns:
            Calibration cache bytes, or None if cache doesn't exist
        """
        if self.cache_file.exists():
            logger.info(f"Reading calibration cache from {self.cache_file}")
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = f.read()
                logger.info(f"Loaded {len(cache_data)} bytes from calibration cache")
                return cache_data
            except Exception as e:
                logger.error(f"Failed to read calibration cache: {e}")
                return None
        else:
            logger.info("No calibration cache found, will perform calibration")
            return None

    def write_calibration_cache(self, cache: bytes) -> None:
        """
        Write calibration cache to disk.

        Called by TensorRT after calibration completes to save results.

        Args:
            cache: Calibration cache data to save
        """
        try:
            # Create cache directory if it doesn't exist
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Writing calibration cache to {self.cache_file}")
            with open(self.cache_file, 'wb') as f:
                f.write(cache)

            file_size = self.cache_file.stat().st_size
            logger.info(f"Calibration cache saved: {file_size} bytes")

        except Exception as e:
            logger.error(f"Failed to write calibration cache: {e}")

    def clear_cache(self) -> None:
        """Remove calibration cache file."""
        if self.cache_file.exists():
            self.cache_file.unlink()
            logger.info(f"Cleared calibration cache: {self.cache_file}")

    def __del__(self):
        """Clean up GPU memory on deletion."""
        if self.device_input is not None:
            try:
                self.device_input.free()
            except:
                pass


class CalibrationDataLoader:
    """
    Helper class to create calibration data loaders from various sources.

    Provides utilities to generate calibration datasets from:
    - Image directories
    - NumPy arrays
    - PyTorch DataLoaders
    - Random synthetic data
    """

    @staticmethod
    def from_numpy_arrays(
        arrays: List[np.ndarray],
        batch_size: int = 1,
        shuffle: bool = True,
    ) -> Iterator[np.ndarray]:
        """
        Create calibration data loader from list of NumPy arrays.

        Args:
            arrays: List of NumPy arrays (calibration samples)
            batch_size: Batch size
            shuffle: Whether to shuffle data

        Yields:
            Batches of calibration data
        """
        # Stack arrays into single array
        data = np.stack(arrays, axis=0)

        # Shuffle if requested
        if shuffle:
            indices = np.random.permutation(len(data))
            data = data[indices]

        # Yield batches
        num_batches = len(data) // batch_size
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            yield data[start_idx:end_idx]

    @staticmethod
    def from_image_directory(
        directory: Union[str, Path],
        batch_size: int = 1,
        image_size: Tuple[int, int] = (224, 224),
        max_samples: Optional[int] = None,
        preprocessing_fn: Optional[callable] = None,
    ) -> Iterator[np.ndarray]:
        """
        Create calibration data loader from directory of images.

        Args:
            directory: Path to directory containing images
            batch_size: Batch size
            image_size: Target image size (height, width)
            max_samples: Maximum number of samples to use
            preprocessing_fn: Optional preprocessing function

        Yields:
            Batches of preprocessed images
        """
        from PIL import Image

        directory = Path(directory)
        image_paths = list(directory.glob("**/*.jpg")) + list(directory.glob("**/*.png"))

        if max_samples:
            image_paths = image_paths[:max_samples]

        logger.info(f"Found {len(image_paths)} images for calibration")

        batch = []
        for img_path in image_paths:
            try:
                # Load and resize image
                img = Image.open(img_path).convert("RGB")
                img = img.resize(image_size)

                # Convert to numpy array
                img_array = np.array(img).astype(np.float32)

                # Apply preprocessing if provided
                if preprocessing_fn:
                    img_array = preprocessing_fn(img_array)
                else:
                    # Default: normalize to [0, 1]
                    img_array = img_array / 255.0

                # Add to batch
                batch.append(img_array)

                # Yield when batch is full
                if len(batch) == batch_size:
                    yield np.stack(batch, axis=0)
                    batch = []

            except Exception as e:
                logger.warning(f"Failed to load image {img_path}: {e}")

        # Yield remaining samples
        if batch:
            # Pad to batch size if needed
            while len(batch) < batch_size:
                batch.append(batch[-1])
            yield np.stack(batch, axis=0)

    @staticmethod
    def from_random_data(
        input_shape: Tuple[int, ...],
        batch_size: int = 1,
        num_batches: int = 100,
        data_range: Tuple[float, float] = (0.0, 1.0),
    ) -> Iterator[np.ndarray]:
        """
        Create calibration data loader with random synthetic data.

        Useful for testing or when real calibration data is unavailable.

        Args:
            input_shape: Input tensor shape (excluding batch dimension)
            batch_size: Batch size
            num_batches: Number of batches to generate
            data_range: Range of random values (min, max)

        Yields:
            Batches of random data
        """
        logger.warning(
            "Using random calibration data - this may result in suboptimal INT8 accuracy"
        )

        min_val, max_val = data_range
        full_shape = (batch_size,) + input_shape

        for _ in range(num_batches):
            # Generate random data in specified range
            batch = np.random.uniform(min_val, max_val, full_shape).astype(np.float32)
            yield batch


# Example usage and testing
if __name__ == "__main__":
    # Example: Create a calibration data loader from random data
    data_loader = CalibrationDataLoader.from_random_data(
        input_shape=(3, 224, 224),
        batch_size=32,
        num_batches=100,
    )

    # Create calibrator
    calibrator = INT8Calibrator(
        data_loader=data_loader,
        cache_file="calibration_cache.bin",
        batch_size=32,
    )

    logger.info("INT8 Calibrator ready for use")
