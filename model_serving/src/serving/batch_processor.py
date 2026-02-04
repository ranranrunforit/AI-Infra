"""
Dynamic Batch Processor

Implements intelligent request batching for improved throughput in model serving.
Features adaptive batch sizing, timeout-based flushing, and request queue management.

Key capabilities:
- Automatic batching of concurrent requests
- Configurable timeout for latency control
- Priority queue support
- Queue overflow protection
- Metrics collection
- Graceful shutdown
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class RequestPriority(Enum):
    """Request priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class BatchRequest:
    """
    Container for a single request in the batch queue.

    Attributes:
        request_id: Unique identifier for the request
        data: Input data for inference
        priority: Request priority level
        timestamp: Request arrival time
        future: Future for async result delivery
        metadata: Additional request metadata
    """
    request_id: str
    data: Dict[str, np.ndarray]
    priority: RequestPriority = RequestPriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    future: Optional[asyncio.Future] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: 'BatchRequest') -> bool:
        """Compare by priority (for priority queue)."""
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        return self.timestamp < other.timestamp


@dataclass
class BatchStats:
    """Statistics for batch processing."""
    total_requests: int = 0
    total_batches: int = 0
    avg_batch_size: float = 0.0
    max_batch_size: int = 0
    avg_wait_time_ms: float = 0.0
    max_wait_time_ms: float = 0.0
    timeout_flushes: int = 0
    size_flushes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "total_requests": self.total_requests,
            "total_batches": self.total_batches,
            "avg_batch_size": round(self.avg_batch_size, 2),
            "max_batch_size": self.max_batch_size,
            "avg_wait_time_ms": round(self.avg_wait_time_ms, 2),
            "max_wait_time_ms": round(self.max_wait_time_ms, 2),
            "timeout_flushes": self.timeout_flushes,
            "size_flushes": self.size_flushes,
        }


class DynamicBatchProcessor:
    """
    Dynamic batching engine for model serving.

    Collects incoming requests and batches them together for efficient processing.
    Automatically flushes batches based on size or timeout constraints.

    Example:
        >>> async def inference_fn(batch_data):
        ...     # Process batch
        ...     return results
        >>>
        >>> processor = DynamicBatchProcessor(
        ...     max_batch_size=32,
        ...     timeout_ms=10,
        ...     inference_fn=inference_fn
        ... )
        >>>
        >>> # Submit request
        >>> result = await processor.submit(request_id="req-1", data=input_data)
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        timeout_ms: float = 10.0,
        max_queue_size: int = 1000,
        inference_fn: Optional[Callable] = None,
        enable_priority: bool = False,
    ):
        """
        Initialize dynamic batch processor.

        Args:
            max_batch_size: Maximum batch size before automatic flush
            timeout_ms: Maximum wait time before timeout flush (milliseconds)
            max_queue_size: Maximum queue size (for backpressure)
            inference_fn: Async function to process batches
            enable_priority: Enable priority-based request ordering
        """
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.max_queue_size = max_queue_size
        self.inference_fn = inference_fn
        self.enable_priority = enable_priority

        # Request queue
        if enable_priority:
            import heapq
            self._queue: List[BatchRequest] = []
            self._heapq = heapq
        else:
            self._queue: deque[BatchRequest] = deque()

        # Processing state
        self._queue_lock = asyncio.Lock()
        self._processing_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._running = False

        # Statistics
        self._stats = BatchStats()
        self._batch_sizes: List[int] = []
        self._wait_times: List[float] = []

        logger.info(
            f"Initialized DynamicBatchProcessor: max_batch_size={max_batch_size}, "
            f"timeout_ms={timeout_ms}, priority={enable_priority}"
        )

    async def start(self) -> None:
        """Start the batch processing loop."""
        if self._running:
            logger.warning("Batch processor already running")
            return

        self._running = True
        self._processing_task = asyncio.create_task(self._processing_loop())
        logger.info("Batch processor started")

    async def shutdown(self) -> None:
        """Gracefully shutdown the batch processor."""
        logger.info("Shutting down batch processor")

        self._running = False
        self._shutdown_event.set()

        # Wait for processing task to complete
        if self._processing_task:
            await self._processing_task

        # Process remaining requests
        await self._flush_remaining()

        logger.info("Batch processor shutdown complete")

    async def submit(
        self,
        request_id: str,
        data: Dict[str, np.ndarray],
        priority: RequestPriority = RequestPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Submit a request for batched processing.

        Args:
            request_id: Unique request identifier
            data: Input data dictionary
            priority: Request priority
            metadata: Additional metadata

        Returns:
            Inference result

        Raises:
            asyncio.QueueFull: If queue is full
            RuntimeError: If processor not started
        """
        if not self._running:
            raise RuntimeError("Batch processor not started. Call start() first.")

        # Check queue size
        async with self._queue_lock:
            queue_size = len(self._queue)
            if queue_size >= self.max_queue_size:
                raise asyncio.QueueFull(
                    f"Request queue full ({queue_size}/{self.max_queue_size})"
                )

            # Create request
            future = asyncio.Future()
            request = BatchRequest(
                request_id=request_id,
                data=data,
                priority=priority,
                future=future,
                metadata=metadata or {}
            )

            # Add to queue
            if self.enable_priority:
                self._heapq.heappush(self._queue, request)
            else:
                self._queue.append(request)

            logger.debug(
                f"Request {request_id} queued (queue_size={queue_size + 1}, "
                f"priority={priority.name})"
            )

        # Wait for result
        return await future

    async def _processing_loop(self) -> None:
        """
        Main processing loop.

        Continuously checks for batch readiness and processes batches.
        """
        logger.info("Processing loop started")

        last_flush_time = time.time()

        while self._running or self._queue:
            try:
                # Check if we should flush
                should_flush, reason = await self._should_flush(last_flush_time)

                if should_flush:
                    await self._process_batch(reason)
                    last_flush_time = time.time()
                else:
                    # Small sleep to prevent busy waiting
                    await asyncio.sleep(0.001)

            except Exception as e:
                logger.error(f"Error in processing loop: {e}", exc_info=True)
                await asyncio.sleep(0.1)  # Prevent tight error loop

        logger.info("Processing loop stopped")

    async def _should_flush(self, last_flush_time: float) -> Tuple[bool, str]:
        """
        Determine if batch should be flushed.

        Args:
            last_flush_time: Timestamp of last flush

        Returns:
            Tuple of (should_flush, reason)
        """
        async with self._queue_lock:
            queue_size = len(self._queue)

            # No requests to process
            if queue_size == 0:
                return False, ""

            # Flush if max batch size reached
            if queue_size >= self.max_batch_size:
                return True, "size"

            # Flush if timeout exceeded
            time_since_flush = (time.time() - last_flush_time) * 1000
            if time_since_flush >= self.timeout_ms:
                return True, "timeout"

            return False, ""

    async def _process_batch(self, flush_reason: str) -> None:
        """
        Process a batch of requests.

        Args:
            flush_reason: Reason for batch flush (size/timeout)
        """
        # Extract batch from queue
        async with self._queue_lock:
            batch_requests = self._extract_batch()

            if not batch_requests:
                return

        batch_size = len(batch_requests)
        logger.debug(f"Processing batch of {batch_size} requests (reason={flush_reason})")

        # Update statistics
        self._stats.total_batches += 1
        self._stats.total_requests += batch_size
        self._batch_sizes.append(batch_size)

        if flush_reason == "size":
            self._stats.size_flushes += 1
        elif flush_reason == "timeout":
            self._stats.timeout_flushes += 1

        # Combine batch data
        try:
            batch_data = self._combine_batch_data(batch_requests)

            # Run inference
            if self.inference_fn:
                results = await self._run_inference(batch_data)
            else:
                # Dummy results for testing
                results = [{"output": f"result_{i}"} for i in range(batch_size)]

            # Distribute results
            await self._distribute_results(batch_requests, results)

        except Exception as e:
            logger.error(f"Batch processing failed: {e}", exc_info=True)

            # Set exceptions on all futures
            for request in batch_requests:
                if request.future and not request.future.done():
                    request.future.set_exception(e)

        # Update wait time statistics
        current_time = time.time()
        for request in batch_requests:
            wait_time_ms = (current_time - request.timestamp) * 1000
            self._wait_times.append(wait_time_ms)

        self._update_statistics()

    def _extract_batch(self) -> List[BatchRequest]:
        """
        Extract a batch of requests from the queue.

        Returns:
            List of batch requests
        """
        batch = []
        batch_size = min(len(self._queue), self.max_batch_size)

        for _ in range(batch_size):
            if self.enable_priority:
                if self._queue:
                    request = self._heapq.heappop(self._queue)
                    batch.append(request)
            else:
                if self._queue:
                    request = self._queue.popleft()
                    batch.append(request)

        return batch

    def _combine_batch_data(
        self,
        batch_requests: List[BatchRequest]
    ) -> Dict[str, np.ndarray]:
        """
        Combine individual request data into batched arrays.

        Args:
            batch_requests: List of requests to batch

        Returns:
            Dictionary of batched arrays
        """
        if not batch_requests:
            return {}

        # Get all input names from first request
        input_names = list(batch_requests[0].data.keys())

        # Combine data for each input
        batched_data = {}
        for name in input_names:
            # Collect arrays for this input
            arrays = [req.data[name] for req in batch_requests]

            # Stack into batch
            batched_data[name] = np.stack(arrays, axis=0)

        return batched_data

    async def _run_inference(self, batch_data: Dict[str, np.ndarray]) -> List[Any]:
        """
        Run inference on batched data.

        Args:
            batch_data: Batched input data

        Returns:
            List of results (one per batch item)
        """
        if asyncio.iscoroutinefunction(self.inference_fn):
            results = await self.inference_fn(batch_data)
        else:
            # Run in executor for sync functions
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, self.inference_fn, batch_data)

        return results

    async def _distribute_results(
        self,
        batch_requests: List[BatchRequest],
        results: List[Any]
    ) -> None:
        """
        Distribute results to waiting futures.

        Args:
            batch_requests: Batch requests
            results: Inference results
        """
        if len(results) != len(batch_requests):
            logger.error(
                f"Result count mismatch: {len(results)} results for "
                f"{len(batch_requests)} requests"
            )
            # Set errors on all futures
            for request in batch_requests:
                if request.future and not request.future.done():
                    request.future.set_exception(
                        RuntimeError("Result count mismatch")
                    )
            return

        # Set results on futures
        for request, result in zip(batch_requests, results):
            if request.future and not request.future.done():
                request.future.set_result(result)

    async def _flush_remaining(self) -> None:
        """Process any remaining requests in the queue."""
        while self._queue:
            await self._process_batch("shutdown")

    def _update_statistics(self) -> None:
        """Update aggregate statistics."""
        if self._batch_sizes:
            self._stats.avg_batch_size = np.mean(self._batch_sizes)
            self._stats.max_batch_size = max(self._batch_sizes)

        if self._wait_times:
            self._stats.avg_wait_time_ms = np.mean(self._wait_times)
            self._stats.max_wait_time_ms = max(self._wait_times)

    def get_stats(self) -> BatchStats:
        """
        Get current batch processing statistics.

        Returns:
            BatchStats object
        """
        return self._stats

    def get_queue_size(self) -> int:
        """Get current queue size."""
        return len(self._queue)

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = BatchStats()
        self._batch_sizes = []
        self._wait_times = []
        logger.info("Statistics reset")


# Example usage and testing
async def example_usage():
    """Example of using DynamicBatchProcessor."""

    # Define inference function
    async def mock_inference(batch_data: Dict[str, np.ndarray]) -> List[Dict]:
        """Mock inference function."""
        batch_size = list(batch_data.values())[0].shape[0]
        await asyncio.sleep(0.01)  # Simulate inference time
        return [{"prediction": i} for i in range(batch_size)]

    # Create processor
    processor = DynamicBatchProcessor(
        max_batch_size=8,
        timeout_ms=50,
        inference_fn=mock_inference
    )

    # Start processor
    await processor.start()

    # Submit requests concurrently
    async def submit_request(req_id: int):
        data = {"input": np.random.randn(224, 224, 3).astype(np.float32)}
        result = await processor.submit(f"req-{req_id}", data)
        logger.info(f"Request {req_id} completed: {result}")

    # Create multiple concurrent requests
    tasks = [submit_request(i) for i in range(20)]
    await asyncio.gather(*tasks)

    # Print statistics
    stats = processor.get_stats()
    logger.info(f"Batch Statistics: {stats.to_dict()}")

    # Shutdown
    await processor.shutdown()


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
