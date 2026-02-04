#!/usr/bin/env python3
"""
Latency Benchmarking Tool

Measures inference latency with percentile metrics (P50, P90, P95, P99, P99.9).
Generates visualizations and detailed reports.
"""

import argparse
import json
import time
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import requests


class LatencyBenchmark:
    """Latency benchmarking tool for model serving."""
    
    def __init__(self, url: str, model_name: str, duration: int = 60):
        self.url = url
        self.model_name = model_name
        self.duration = duration
        self.latencies: List[float] = []
    
    def run(self) -> Dict:
        """Run latency benchmark."""
        print(f"Running latency benchmark for {self.duration} seconds...")
        
        start_time = time.time()
        request_count = 0
        
        while time.time() - start_time < self.duration:
            latency = self._measure_single_request()
            if latency is not None:
                self.latencies.append(latency)
                request_count += 1
            
            if request_count % 100 == 0:
                print(f"Completed {request_count} requests...")
        
        return self._calculate_metrics()
    
    def _measure_single_request(self) -> float:
        """Measure latency of single request."""
        try:
            payload = {
                "model": self.model_name,
                "inputs": {"data": np.random.randn(224, 224, 3).tolist()}
            }
            
            start = time.time()
            response = requests.post(f"{self.url}/v1/predict", json=payload, timeout=10)
            latency = (time.time() - start) * 1000  # Convert to ms
            
            if response.status_code == 200:
                return latency
        except Exception as e:
            print(f"Request failed: {e}")
        
        return None
    
    def _calculate_metrics(self) -> Dict:
        """Calculate latency metrics."""
        latencies_np = np.array(self.latencies)
        
        metrics = {
            "count": len(self.latencies),
            "mean": float(np.mean(latencies_np)),
            "median": float(np.median(latencies_np)),
            "std": float(np.std(latencies_np)),
            "min": float(np.min(latencies_np)),
            "max": float(np.max(latencies_np)),
            "p50": float(np.percentile(latencies_np, 50)),
            "p90": float(np.percentile(latencies_np, 90)),
            "p95": float(np.percentile(latencies_np, 95)),
            "p99": float(np.percentile(latencies_np, 99)),
            "p99.9": float(np.percentile(latencies_np, 99.9)),
        }
        
        return metrics
    
    def save_results(self, output_path: str):
        """Save results to JSON."""
        metrics = self._calculate_metrics()
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Results saved to {output_path}")
    
    def plot_results(self, output_path: str):
        """Generate latency distribution plot."""
        plt.figure(figsize=(12, 6))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(self.latencies, bins=50, edgecolor='black')
        plt.xlabel('Latency (ms)')
        plt.ylabel('Frequency')
        plt.title('Latency Distribution')
        
        # Percentiles
        plt.subplot(1, 2, 2)
        metrics = self._calculate_metrics()
        percentiles = ['p50', 'p90', 'p95', 'p99', 'p99.9']
        values = [metrics[p] for p in percentiles]
        
        plt.bar(percentiles, values)
        plt.xlabel('Percentile')
        plt.ylabel('Latency (ms)')
        plt.title('Latency Percentiles')
        
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Latency benchmark tool')
    parser.add_argument('--url', default='http://localhost:8000', help='Server URL')
    parser.add_argument('--model', default='test-model', help='Model name')
    parser.add_argument('--duration', type=int, default=60, help='Duration in seconds')
    parser.add_argument('--output', default='latency_results.json', help='Output file')
    
    args = parser.parse_args()
    
    benchmark = LatencyBenchmark(args.url, args.model, args.duration)
    results = benchmark.run()
    
    print("\nLatency Metrics:")
    for key, value in results.items():
        print(f"  {key}: {value:.2f} ms")
    
    benchmark.save_results(args.output)
    benchmark.plot_results(args.output.replace('.json', '.png'))


if __name__ == '__main__':
    main()
