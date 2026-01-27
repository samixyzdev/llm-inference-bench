"""
Memory benchmarking for LLM inference.
Measures GPU memory usage during inference.
"""

import torch
import gc
from typing import Optional

import sys
sys.path.append('..')

from utils.metrics import MemoryMetrics, GPUMonitor
from utils.model_loader import get_model_size_mb


def benchmark_memory(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    batch_size: int = 1,
) -> MemoryMetrics:
    """
    Benchmark GPU memory usage during inference.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        batch_size: Batch size for inference

    Returns:
        MemoryMetrics with memory usage data
    """
    device = next(model.parameters()).device

    if not torch.cuda.is_available():
        return MemoryMetrics(
            peak_memory_mb=0,
            allocated_memory_mb=0,
            reserved_memory_mb=0,
            model_size_mb=get_model_size_mb(model),
        )

    # Clear cache and reset peak stats
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Prepare input
    prompts = [prompt] * batch_size
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    # Record memory before inference
    memory_before = torch.cuda.memory_allocated()

    # Run inference
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Record peak memory
    peak_memory = torch.cuda.max_memory_allocated()
    allocated_memory = torch.cuda.memory_allocated()
    reserved_memory = torch.cuda.memory_reserved()

    return MemoryMetrics(
        peak_memory_mb=peak_memory / (1024 * 1024),
        allocated_memory_mb=allocated_memory / (1024 * 1024),
        reserved_memory_mb=reserved_memory / (1024 * 1024),
        model_size_mb=get_model_size_mb(model),
    )


def benchmark_memory_scaling(
    model,
    tokenizer,
    prompt: str,
    batch_sizes: list = [1, 2, 4, 8, 16],
    max_new_tokens: int = 128,
) -> list:
    """
    Benchmark memory usage at different batch sizes.

    Returns:
        List of (batch_size, MemoryMetrics) tuples
    """
    results = []

    for batch_size in batch_sizes:
        print(f"\n--- Batch Size: {batch_size} ---")
        try:
            metrics = benchmark_memory(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
            )
            results.append((batch_size, metrics))
            print(f"Peak Memory: {metrics.peak_memory_mb:.2f} MB "
                  f"({metrics.peak_memory_mb/1024:.2f} GB)")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM at batch size {batch_size}, stopping test")
                torch.cuda.empty_cache()
                break
            raise

    return results


def get_gpu_info() -> dict:
    """Get GPU information."""
    if not torch.cuda.is_available():
        return {"available": False}

    monitor = GPUMonitor()
    memory_info = monitor.get_memory_info()

    return {
        "available": True,
        "name": monitor.get_gpu_name(),
        "total_memory_gb": memory_info["total_mb"] / 1024,
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
    }
