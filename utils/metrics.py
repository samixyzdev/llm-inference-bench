"""
Metrics utilities for LLM-Inference-Bench.
"""

import time
import torch
import pynvml
from dataclasses import dataclass, asdict
from typing import List, Dict, Any


@dataclass
class LatencyMetrics:
    """Latency measurement results."""
    time_to_first_token_ms: float
    per_token_latency_ms: float
    total_generation_time_ms: float
    tokens_generated: int


@dataclass
class ThroughputMetrics:
    """Throughput measurement results."""
    tokens_per_second: float
    batch_size: int
    input_tokens: int
    output_tokens: int
    total_time_seconds: float


@dataclass
class MemoryMetrics:
    """GPU memory measurement results."""
    peak_memory_mb: float
    allocated_memory_mb: float
    reserved_memory_mb: float
    model_size_mb: float


@dataclass
class BenchmarkResult:
    """Complete benchmark result for a single run."""
    model_name: str
    quantization: str
    batch_size: int
    input_length: int
    output_length: int
    latency: LatencyMetrics
    throughput: ThroughputMetrics
    memory: MemoryMetrics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "quantization": self.quantization,
            "batch_size": self.batch_size,
            "input_length": self.input_length,
            "output_length": self.output_length,
            "latency": asdict(self.latency),
            "throughput": asdict(self.throughput),
            "memory": asdict(self.memory),
        }


class GPUMonitor:
    """Monitor GPU memory usage."""

    def __init__(self):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def get_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory usage in MB."""
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        return {
            "total_mb": info.total / (1024 * 1024),
            "used_mb": info.used / (1024 * 1024),
            "free_mb": info.free / (1024 * 1024),
        }

    def get_gpu_name(self) -> str:
        """Get GPU name."""
        return pynvml.nvmlDeviceGetName(self.handle)

    def __del__(self):
        try:
            pynvml.nvmlShutdown()
        except:
            pass


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, sync_cuda: bool = True):
        self.sync_cuda = sync_cuda
        self.elapsed_ms = 0

    def __enter__(self):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000


def calculate_tokens_per_second(
    num_tokens: int,
    time_seconds: float
) -> float:
    """Calculate throughput in tokens per second."""
    if time_seconds <= 0:
        return 0.0
    return num_tokens / time_seconds
