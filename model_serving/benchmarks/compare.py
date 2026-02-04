#!/usr/bin/env python3
"""
Model Performance Comparison Tool

Compares performance between:
- TensorRT vs PyTorch
- Different precision modes (FP32, FP16, INT8)
- Different batch sizes

Generates comparison reports and visualizations.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np


class PerformanceComparison:
    """Compare performance across models and configurations."""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.results = {}
    
    def load_results(self):
        """Load all benchmark results."""
        for result_file in self.results_dir.glob("*.json"):
            with open(result_file) as f:
                self.results[result_file.stem] = json.load(f)
    
    def generate_comparison_report(self) -> str:
        """Generate comparison report."""
        report = ["# Performance Comparison Report\n"]
        
        for name, metrics in self.results.items():
            report.append(f"\n## {name}\n")
            for key, value in metrics.items():
                if isinstance(value, float):
                    report.append(f"- {key}: {value:.2f}")
                else:
                    report.append(f"- {key}: {value}")
        
        return "\n".join(report)
    
    def plot_comparisons(self, output_path: str):
        """Generate comparison plots."""
        if not self.results:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract common metrics
        names = list(self.results.keys())
        
        # Latency comparison
        if all('mean' in r for r in self.results.values()):
            latencies = [self.results[n].get('mean', 0) for n in names]
            axes[0, 0].bar(names, latencies)
            axes[0, 0].set_title('Mean Latency Comparison')
            axes[0, 0].set_ylabel('Latency (ms)')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Throughput comparison
        if all('qps' in r for r in self.results.values()):
            qps = [self.results[n].get('qps', 0) for n in names]
            axes[0, 1].bar(names, qps)
            axes[0, 1].set_title('Throughput Comparison')
            axes[0, 1].set_ylabel('QPS')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Percentile comparison
        if all('p99' in r for r in self.results.values()):
            p50 = [self.results[n].get('p50', 0) for n in names]
            p99 = [self.results[n].get('p99', 0) for n in names]
            
            x = np.arange(len(names))
            width = 0.35
            
            axes[1, 0].bar(x - width/2, p50, width, label='P50')
            axes[1, 0].bar(x + width/2, p99, width, label='P99')
            axes[1, 0].set_title('Latency Percentiles')
            axes[1, 0].set_ylabel('Latency (ms)')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(names, rotation=45)
            axes[1, 0].legend()
        
        # Success rate comparison
        if all('success_rate' in r for r in self.results.values()):
            success_rates = [self.results[n].get('success_rate', 0) * 100 for n in names]
            axes[1, 1].bar(names, success_rates)
            axes[1, 1].set_title('Success Rate')
            axes[1, 1].set_ylabel('Success Rate (%)')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].set_ylim([0, 105])
        
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Comparison plot saved to {output_path}")
    
    def save_report(self, output_path: str):
        """Save comparison report."""
        report = self.generate_comparison_report()
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Performance comparison tool')
    parser.add_argument('--results-dir', required=True, help='Directory with benchmark results')
    parser.add_argument('--output', default='comparison_report.md', help='Output report file')
    
    args = parser.parse_args()
    
    comparison = PerformanceComparison(args.results_dir)
    comparison.load_results()
    
    print(comparison.generate_comparison_report())
    
    comparison.save_report(args.output)
    comparison.plot_comparisons(args.output.replace('.md', '.png'))


if __name__ == '__main__':
    main()
