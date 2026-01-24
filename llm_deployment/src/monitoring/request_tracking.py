"""
Request tracking utilities for monitoring
"""

import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class RequestTimer:
    """
    Context manager for timing requests and recording metrics
    """

    def __init__(
        self,
        metrics_collector,
        endpoint: str,
    ):
        """
        Initialize timer

        Args:
            metrics_collector: MetricsCollector instance
            endpoint: API endpoint name (e.g., "generate", "rag_generate")
        """
        self.metrics_collector = metrics_collector
        self.endpoint = endpoint
        self.start_time = None
        self.status = "success"
        self.tokens = 0

    def __enter__(self):
        """Start timing when entering context"""
        self.start_time = time.time()
        
        # Increment in-progress counter
        if self.metrics_collector:
            self.metrics_collector.request_in_progress.labels(
                model=self.metrics_collector.model_name, 
                endpoint=self.endpoint
            ).inc()
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Record metrics when exiting context"""
        duration = time.time() - self.start_time

        # Decrement in-progress counter
        if self.metrics_collector:
            self.metrics_collector.request_in_progress.labels(
                model=self.metrics_collector.model_name, 
                endpoint=self.endpoint
            ).dec()

        # Set status based on whether an exception occurred
        if exc_type is not None:
            self.status = "error"
            logger.error(
                f"Request to {self.endpoint} failed after {duration:.3f}s: {exc_val}"
            )

        # Record the request metrics
        if self.metrics_collector:
            self.metrics_collector.record_request(
                endpoint=self.endpoint,
                duration=duration,
                status=self.status,
                tokens=self.tokens,
            )

        logger.debug(
            f"Request to {self.endpoint} completed in {duration:.3f}s "
            f"(status: {self.status}, tokens: {self.tokens})"
        )

    def set_tokens(self, tokens: int):
        """
        Set number of tokens generated
        
        Args:
            tokens: Number of tokens generated in the request
        """
        self.tokens = tokens


class RAGTimer:
    """
    Context manager specifically for timing RAG retrieval operations
    """

    def __init__(self, metrics_collector):
        """
        Initialize RAG timer

        Args:
            metrics_collector: MetricsCollector instance
        """
        self.metrics_collector = metrics_collector
        self.start_time = None
        self.num_chunks = 0

    def __enter__(self):
        """Start timing"""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Record RAG metrics"""
        duration = time.time() - self.start_time

        if exc_type is None and self.metrics_collector:
            self.metrics_collector.record_rag_retrieval(
                num_chunks=self.num_chunks,
                duration=duration
            )

        logger.debug(
            f"RAG retrieval completed in {duration:.3f}s "
            f"({self.num_chunks} chunks)"
        )

    def set_chunks(self, num_chunks: int):
        """
        Set number of chunks retrieved
        
        Args:
            num_chunks: Number of chunks retrieved
        """
        self.num_chunks = num_chunks