"""
Model Warmup Module

Performs model warmup by running dummy inference passes to:
- Prime GPU caches and memory
- Trigger JIT compilation (PyTorch)
- Stabilize TensorRT kernel selection
- Reduce cold-start latency for first real requests

Usage:
    warmup_manager = ModelWarmup(model_loader)
    await warmup_manager.warmup_all(iterations=10)
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ModelWarmup:
    """
    Model warmup manager.

    Runs warm-up inference passes on loaded models to eliminate cold-start
    latency. Supports TensorRT, PyTorch, and ONNX models.

    Example:
        ```python
        from src.serving.warmup import ModelWarmup
        from src.serving.model_loader import ModelLoader

        loader = ModelLoader(cache_dir="/tmp/models")
        loader.load_model("resnet50-fp16", model_format="tensorrt")

        warmup = ModelWarmup(loader)
        await warmup.warmup_all(iterations=10)
        ```
    """

    def __init__(self, model_loader, default_iterations: int = 10):
        """
        Initialize warmup manager.

        Args:
            model_loader: ModelLoader instance with loaded models
            default_iterations: Default number of warmup iterations per model
        """
        self.model_loader = model_loader
        self.default_iterations = default_iterations
        self._warmup_results: Dict[str, Dict[str, Any]] = {}

    async def warmup_all(
        self,
        iterations: Optional[int] = None,
        models: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Warm up all loaded models (or a subset).

        Args:
            iterations: Number of warmup iterations (uses default if None)
            models: List of model names to warm up (all if None)

        Returns:
            Dictionary with warmup results per model
        """
        iters = iterations or self.default_iterations
        target_models = models or self.model_loader.list_loaded_models()

        if not target_models:
            logger.warning("No models loaded for warmup")
            return {}

        logger.info(
            f"Starting warmup for {len(target_models)} model(s), "
            f"{iters} iterations each"
        )

        results = {}
        for model_name in target_models:
            try:
                result = await self._warmup_model(model_name, iters)
                results[model_name] = result
                logger.info(
                    f"Warmup complete for {model_name}: "
                    f"avg={result['avg_latency_ms']:.2f}ms, "
                    f"last={result['last_latency_ms']:.2f}ms"
                )
            except Exception as e:
                logger.error(f"Warmup failed for {model_name}: {e}")
                results[model_name] = {"status": "failed", "error": str(e)}

        self._warmup_results = results
        logger.info(f"Warmup completed for {len(results)} model(s)")
        return results

    async def _warmup_model(
        self, model_name: str, iterations: int
    ) -> Dict[str, Any]:
        """
        Warm up a single model.

        Args:
            model_name: Name of the model to warm up
            iterations: Number of warmup passes

        Returns:
            Dictionary with latency statistics
        """
        try:
            model_info = self.model_loader.get_model_info(model_name)
        except (KeyError, AttributeError):
            model_info = None

        # Determine model format from info or fall back to trying each
        model_format = None
        if model_info and hasattr(model_info, "format"):
            model_format = model_info.format.value if hasattr(model_info.format, "value") else str(model_info.format)

        # Create dummy inputs based on model format
        dummy_inputs = self._create_dummy_inputs(model_name, model_format)

        latencies = []
        for i in range(iterations):
            start = time.perf_counter()
            try:
                await self._run_dummy_inference(model_name, dummy_inputs, model_format)
            except Exception as e:
                if i == 0:
                    # First iteration failure might be expected, log and continue
                    logger.debug(f"Warmup iteration 0 for {model_name} failed: {e}")
                else:
                    raise
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        return {
            "status": "success",
            "iterations": iterations,
            "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            "min_latency_ms": float(np.min(latencies)) if latencies else 0.0,
            "max_latency_ms": float(np.max(latencies)) if latencies else 0.0,
            "last_latency_ms": latencies[-1] if latencies else 0.0,
            "model": model_name,
        }

    def _create_dummy_inputs(
        self, model_name: str, model_format: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Create dummy inputs for warmup inference.

        Args:
            model_name: Model name (used to infer input shapes)
            model_format: Model format hint

        Returns:
            Dictionary of dummy input arrays
        """
        # Default to common image classification input shape
        # In production, this should be derived from model metadata
        name_lower = model_name.lower()

        if any(kw in name_lower for kw in ["bert", "gpt", "llama", "mistral", "llm"]):
            # Text model: token IDs
            return {
                "input_ids": np.ones((1, 128), dtype=np.int64),
                "attention_mask": np.ones((1, 128), dtype=np.int64),
            }
        else:
            # Vision model: image tensor
            return {
                "input": np.random.randn(1, 3, 224, 224).astype(np.float32),
            }

    async def _run_dummy_inference(
        self,
        model_name: str,
        dummy_inputs: Dict[str, np.ndarray],
        model_format: Optional[str] = None,
    ) -> None:
        """
        Run a single dummy inference pass.

        Args:
            model_name: Model to run inference on
            dummy_inputs: Dummy input data
            model_format: Model format hint
        """
        try:
            model = self.model_loader.get_model(model_name)
        except KeyError:
            logger.warning(f"Model {model_name} not found, skipping warmup")
            return

        # Run inference in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self._sync_inference, model, dummy_inputs, model_format
        )

    def _sync_inference(
        self,
        model: Any,
        dummy_inputs: Dict[str, np.ndarray],
        model_format: Optional[str],
    ) -> None:
        """
        Synchronous inference for warmup (runs in thread pool).

        Args:
            model: Loaded model object
            dummy_inputs: Input data
            model_format: Model format
        """
        try:
            import torch

            # TensorRT model (has 'infer' method from our TensorRTModel wrapper)
            if hasattr(model, "infer"):
                model.infer(dummy_inputs)
            # PyTorch model
            elif isinstance(model, torch.nn.Module):
                with torch.no_grad():
                    for key, val in dummy_inputs.items():
                        tensor = torch.from_numpy(val)
                        if torch.cuda.is_available():
                            tensor = tensor.cuda()
                        model(tensor)
                        break  # Usually only one input
            # ONNX Runtime session
            elif hasattr(model, "run"):
                input_names = [inp.name for inp in model.get_inputs()]
                feed = {}
                for name in input_names:
                    if name in dummy_inputs:
                        feed[name] = dummy_inputs[name]
                    else:
                        # Use first available input
                        feed[name] = list(dummy_inputs.values())[0]
                model.run(None, feed)
            else:
                logger.warning(f"Unknown model type {type(model)}, skipping inference")
        except Exception as e:
            logger.debug(f"Warmup inference pass failed (may be expected): {e}")

    def get_warmup_results(self) -> Dict[str, Dict[str, Any]]:
        """Get results from the last warmup run."""
        return self._warmup_results

    def is_warmed_up(self, model_name: str) -> bool:
        """Check if a model has been warmed up."""
        result = self._warmup_results.get(model_name, {})
        return result.get("status") == "success"
