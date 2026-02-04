#!/usr/bin/env python3
"""
Throughput Benchmarking Tool

Measures queries per second (QPS) under various concurrency levels.
Uses Locust for load generation and provides detailed performance metrics.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict
import numpy as np
import requests


class ThroughputBenchmark:
    """Throughput benchmarking tool."""
    
    def __init__(self, url: str, model_name: str, duration: int = 60, concurrency: int = 10):
        self.url = url
        self.model_name = model_name
        self.duration = duration
        self.concurrency = concurrency
        self.results = []
    
    def run(self) -> Dict:
        """Run throughput benchmark."""
        print(f"Running throughput benchmark:")
        print(f"  Duration: {self.duration}s")
        print(f"  Concurrency: {self.concurrency}")
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = []
            
            while time.time() - start_time < self.duration:
                future = executor.submit(self._send_request)
                futures.append(future)
            
            # Wait for completion
            for future in as_completed(futures):
                result = future.result()
                if result:
                    self.results.append(result)
        
        return self._calculate_metrics(time.time() - start_time)
    
    def _send_request(self) -> Dict:
        """Send single request."""
        try:
            payload = {
                "model": self.model_name,
                "inputs": {"data": np.random.randn(224, 224, 3).tolist()}
            }
            
            start = time.time()
            response = requests.post(f"{self.url}/v1/predict", json=payload, timeout=10)
            latency = time.time() - start
            
            return {
                "success": response.status_code == 200,
                "latency": latency,
                "timestamp": start
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _calculate_metrics(self, elapsed_time: float) -> Dict:
        """Calculate throughput metrics."""
        successful = [r for r in self.results if r.get("success", False)]
        failed = len(self.results) - len(successful)
        
        latencies = [r["latency"] * 1000 for r in successful]  # Convert to ms
        
        metrics = {
            "duration_seconds": elapsed_time,
            "total_requests": len(self.results),
            "successful_requests": len(successful),
            "failed_requests": failed,
            "success_rate": len(successful) / len(self.results) if self.results else 0,
            "qps": len(successful) / elapsed_time,
            "mean_latency_ms": np.mean(latencies) if latencies else 0,
            "p50_latency_ms": np.percentile(latencies, 50) if latencies else 0,
            "p95_latency_ms": np.percentile(latencies, 95) if latencies else 0,
            "p99_latency_ms": np.percentile(latencies, 99) if latencies else 0,
        }
        
        return metrics
    
    def save_results(self, output_path: str):
        """Save results to JSON."""
        metrics = self._calculate_metrics(self.duration)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Throughput benchmark tool')
    parser.add_argument('--url', default='http://localhost:8000', help='Server URL')
    parser.add_argument('--model', default='test-model', help='Model name')
    parser.add_argument('--duration', type=int, default=60, help='Duration in seconds')
    parser.add_argument('--concurrency', type=int, default=10, help='Concurrent requests')
    parser.add_argument('--output', default='throughput_results.json', help='Output file')
    
    args = parser.parse_args()
    
    benchmark = ThroughputBenchmark(args.url, args.model, args.duration, args.concurrency)
    results = benchmark.run()
    
    print("\nThroughput Metrics:")
    print(f"  QPS: {results['qps']:.2f}")
    print(f"  Success Rate: {results['success_rate']*100:.2f}%")
    print(f"  Mean Latency: {results['mean_latency_ms']:.2f} ms")
    print(f"  P95 Latency: {results['p95_latency_ms']:.2f} ms")
    print(f"  P99 Latency: {results['p99_latency_ms']:.2f} ms")
    
    benchmark.save_results(args.output)


if __name__ == '__main__':
    main()
