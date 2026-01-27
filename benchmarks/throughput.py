"""
Throughput benchmarking for LLM inference.
Measures tokens generated per second at various batch sizes.
"""

import torch
from typing import List, Dict, Any
from tqdm import tqdm

import sys
sys.path.append('..')

from utils.metrics import Timer, ThroughputMetrics, calculate_tokens_per_second


def benchmark_throughput(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 128,
    num_warmup: int = 2,
    num_runs: int = 5,
    batch_size: int = 1,
) -> ThroughputMetrics:
    """
    Benchmark throughput (tokens per second).

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompts: List of input prompts
        max_new_tokens: Maximum tokens to generate
        num_warmup: Number of warmup runs
        num_runs: Number of benchmark runs
        batch_size: Batch size for inference

    Returns:
        ThroughputMetrics with results
    """
    device = next(model.parameters()).device

    # Prepare batched input
    batch_prompts = prompts[:batch_size]
    if len(batch_prompts) < batch_size:
        batch_prompts = batch_prompts * (batch_size // len(batch_prompts) + 1)
        batch_prompts = batch_prompts[:batch_size]

    inputs = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    input_tokens = inputs.input_ids.shape[1]

    # Warmup runs
    print(f"Running {num_warmup} warmup iterations...")
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

    # Benchmark runs
    print(f"Running {num_runs} benchmark iterations...")
    total_time_ms = 0
    total_output_tokens = 0

    for _ in tqdm(range(num_runs), desc="Benchmarking"):
        torch.cuda.empty_cache()

        with Timer() as timer:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

        total_time_ms += timer.elapsed_ms
        output_tokens = outputs.shape[1] - input_tokens
        total_output_tokens += output_tokens * batch_size

    avg_time_seconds = (total_time_ms / num_runs) / 1000
    avg_output_tokens = total_output_tokens / num_runs

    tokens_per_second = calculate_tokens_per_second(avg_output_tokens, avg_time_seconds)

    return ThroughputMetrics(
        tokens_per_second=tokens_per_second,
        batch_size=batch_size,
        input_tokens=input_tokens,
        output_tokens=int(avg_output_tokens / batch_size),
        total_time_seconds=avg_time_seconds,
    )


def benchmark_throughput_scaling(
    model,
    tokenizer,
    prompts: List[str],
    batch_sizes: List[int] = [1, 2, 4, 8],
    max_new_tokens: int = 128,
    num_runs: int = 3,
) -> List[ThroughputMetrics]:
    """
    Benchmark throughput at multiple batch sizes.

    Returns:
        List of ThroughputMetrics for each batch size
    """
    results = []

    for batch_size in batch_sizes:
        print(f"\n--- Batch Size: {batch_size} ---")
        try:
            metrics = benchmark_throughput(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
                num_runs=num_runs,
            )
            results.append(metrics)
            print(f"Throughput: {metrics.tokens_per_second:.2f} tokens/sec")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM at batch size {batch_size}, stopping scaling test")
                torch.cuda.empty_cache()
                break
            raise

    return results
