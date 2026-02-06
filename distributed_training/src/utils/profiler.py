"""Performance profiling utilities"""
import torch
import time
from contextlib import contextmanager

@contextmanager
def profile_time(name=""):
    start = time.time()
    yield
    duration = time.time() - start
    print(f"{name}: {duration:.4f}s")

class Profiler:
    def __init__(self):
        self.timings = {}
    
    def record(self, name, duration):
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(duration)
    
    def summary(self):
        import statistics
        for name, times in self.timings.items():
            print(f"{name}: mean={statistics.mean(times):.4f}s, std={statistics.stdev(times) if len(times) > 1 else 0:.4f}s")
