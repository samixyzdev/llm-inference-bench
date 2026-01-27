"""
Latency benchmarking for LLM inference.
Measures time-to-first-token (TTFT) and per-token latency.
"""

import torch
import time
from typing import List
from tqdm import tqdm

import sys
sys.path.append('..')

from utils.metrics import Timer, LatencyMetrics


def benchmark_latency(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    num_warmup: int = 2,
    num_runs: int = 10,
) -> LatencyMetrics:
    """
    Benchmark latency metrics including TTFT and per-token latency.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        num_warmup: Number of warmup runs
        num_runs: Number of benchmark runs

    Returns:
        LatencyMetrics with TTFT and per-token latency
    """
    device = next(model.parameters()).device

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    input_length = inputs.input_ids.shape[1]

    # Warmup
    print(f"Running {num_warmup} warmup iterations...")
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

    # Benchmark TTFT (Time to First Token)
    print(f"Measuring TTFT over {num_runs} runs...")
    ttft_times = []

    for _ in tqdm(range(num_runs), desc="TTFT"):
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=1,  # Only generate first token
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ttft_times.append((time.perf_counter() - start) * 1000)

    avg_ttft = sum(ttft_times) / len(ttft_times)

    # Benchmark full generation for per-token latency
    print(f"Measuring per-token latency over {num_runs} runs...")
    generation_times = []
    tokens_generated_list = []

    for _ in tqdm(range(num_runs), desc="Generation"):
        torch.cuda.empty_cache()

        with Timer() as timer:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

        tokens_generated = outputs.shape[1] - input_length
        generation_times.append(timer.elapsed_ms)
        tokens_generated_list.append(tokens_generated)

    avg_generation_time = sum(generation_times) / len(generation_times)
    avg_tokens = sum(tokens_generated_list) / len(tokens_generated_list)

    # Per-token latency = (total_time - TTFT) / (tokens - 1)
    # Because TTFT includes generating the first token
    if avg_tokens > 1:
        per_token_latency = (avg_generation_time - avg_ttft) / (avg_tokens - 1)
    else:
        per_token_latency = avg_generation_time

    return LatencyMetrics(
        time_to_first_token_ms=avg_ttft,
        per_token_latency_ms=per_token_latency,
        total_generation_time_ms=avg_generation_time,
        tokens_generated=int(avg_tokens),
    )


def benchmark_latency_by_input_length(
    model,
    tokenizer,
    base_prompt: str,
    input_lengths: List[int] = [32, 64, 128, 256, 512],
    max_new_tokens: int = 64,
    num_runs: int = 5,
) -> List[dict]:
    """
    Benchmark how latency changes with input length.

    Returns:
        List of dicts with input_length and LatencyMetrics
    """
    results = []

    for length in input_lengths:
        print(f"\n--- Input Length: {length} tokens ---")

        # Extend prompt to target length
        tokens = tokenizer.encode(base_prompt)
        if len(tokens) < length:
            # Repeat the prompt to reach target length
            extended = base_prompt * (length // len(tokens) + 1)
            tokens = tokenizer.encode(extended)[:length]
            prompt = tokenizer.decode(tokens)
        else:
            prompt = tokenizer.decode(tokens[:length])

        try:
            metrics = benchmark_latency(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                num_runs=num_runs,
            )
            results.append({
                "input_length": length,
                "metrics": metrics,
            })
            print(f"TTFT: {metrics.time_to_first_token_ms:.2f}ms, "
                  f"Per-token: {metrics.per_token_latency_ms:.2f}ms")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM at input length {length}, stopping test")
                torch.cuda.empty_cache()
                break
            raise

    return results
