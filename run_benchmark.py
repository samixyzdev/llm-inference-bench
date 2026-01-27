#!/usr/bin/env python3
"""
LLM-Inference-Bench: Main benchmark runner.

Usage:
    python run_benchmark.py --model tiny --quantization fp16
    python run_benchmark.py --model phi2 --quantization int8 --batch-sizes 1 2 4
    python run_benchmark.py --compare-quantization --model tiny
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch

from utils import (
    load_model,
    SUPPORTED_MODELS,
    get_model_size_mb,
    BenchmarkResult,
    GPUMonitor,
    plot_throughput_comparison,
    plot_latency_breakdown,
    plot_memory_usage,
    generate_summary_table,
)
from benchmarks import benchmark_throughput, benchmark_latency, benchmark_memory


# Default prompts for benchmarking
DEFAULT_PROMPTS = [
    "Explain the concept of machine learning in simple terms.",
    "Write a Python function to calculate the factorial of a number.",
    "What are the benefits of using renewable energy sources?",
    "Describe the process of photosynthesis step by step.",
    "How does a neural network learn from data?",
]


def run_single_benchmark(
    model_name: str,
    quantization: str,
    batch_size: int,
    max_new_tokens: int,
    num_runs: int,
    prompts: list,
) -> dict:
    """Run a complete benchmark for a single configuration."""

    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name} ({quantization}) - Batch Size: {batch_size}")
    print(f"{'='*60}\n")

    # Load model
    model, tokenizer = load_model(model_name, quantization=quantization)

    # Run benchmarks
    print("\n[1/3] Benchmarking Throughput...")
    throughput = benchmark_throughput(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        num_runs=num_runs,
    )

    print("\n[2/3] Benchmarking Latency...")
    latency = benchmark_latency(
        model=model,
        tokenizer=tokenizer,
        prompt=prompts[0],
        max_new_tokens=max_new_tokens,
        num_runs=num_runs,
    )

    print("\n[3/3] Benchmarking Memory...")
    memory = benchmark_memory(
        model=model,
        tokenizer=tokenizer,
        prompt=prompts[0],
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )

    # Create result object
    result = BenchmarkResult(
        model_name=model_name,
        quantization=quantization,
        batch_size=batch_size,
        input_length=throughput.input_tokens,
        output_length=throughput.output_tokens,
        latency=latency,
        throughput=throughput,
        memory=memory,
    )

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return result.to_dict()


def run_quantization_comparison(
    model_name: str,
    quantizations: list = ["fp16", "int8", "int4"],
    batch_size: int = 1,
    max_new_tokens: int = 128,
    num_runs: int = 5,
) -> list:
    """Compare different quantization levels for a model."""

    results = []
    for quant in quantizations:
        try:
            result = run_single_benchmark(
                model_name=model_name,
                quantization=quant,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                num_runs=num_runs,
                prompts=DEFAULT_PROMPTS,
            )
            results.append(result)
        except Exception as e:
            print(f"Error benchmarking {quant}: {e}")
            continue

    return results


def run_model_comparison(
    models: list,
    quantization: str = "fp16",
    batch_size: int = 1,
    max_new_tokens: int = 128,
    num_runs: int = 5,
) -> list:
    """Compare different models with same quantization."""

    results = []
    for model_name in models:
        try:
            result = run_single_benchmark(
                model_name=model_name,
                quantization=quantization,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                num_runs=num_runs,
                prompts=DEFAULT_PROMPTS,
            )
            results.append(result)
        except Exception as e:
            print(f"Error benchmarking {model_name}: {e}")
            continue

    return results


def save_results(results: list, output_dir: str = "results"):
    """Save benchmark results to JSON and generate visualizations."""

    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON results
    json_path = f"{output_dir}/benchmark_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Generate visualizations
    if len(results) > 0:
        plot_throughput_comparison(
            results,
            output_path=f"{output_dir}/throughput_{timestamp}.png"
        )
        plot_latency_breakdown(
            results,
            output_path=f"{output_dir}/latency_{timestamp}.png"
        )
        plot_memory_usage(
            results,
            output_path=f"{output_dir}/memory_{timestamp}.png"
        )

        # Print summary table
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        summary_df = generate_summary_table(results)
        print(summary_df.to_string(index=False))
        print("="*60)

        # Save summary CSV
        csv_path = f"{output_dir}/summary_{timestamp}.csv"
        summary_df.to_csv(csv_path, index=False)
        print(f"Summary saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="LLM-Inference-Bench: Benchmark LLM inference performance"
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default="tiny",
        help=f"Model to benchmark. Options: {list(SUPPORTED_MODELS.keys())} or HuggingFace path"
    )
    parser.add_argument(
        "--quantization", "-q",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "int8", "int4"],
        help="Quantization level"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--num-runs", "-n",
        type=int,
        default=5,
        help="Number of benchmark runs"
    )
    parser.add_argument(
        "--compare-quantization",
        action="store_true",
        help="Compare all quantization levels for the model"
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Compare multiple models (tiny, phi2)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results",
        help="Output directory for results"
    )

    args = parser.parse_args()

    # Print GPU info
    if torch.cuda.is_available():
        monitor = GPUMonitor()
        print(f"\nGPU: {monitor.get_gpu_name()}")
        mem_info = monitor.get_memory_info()
        print(f"Total Memory: {mem_info['total_mb']/1024:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("\nWARNING: No GPU available, running on CPU")

    # Run benchmarks
    if args.compare_quantization:
        print(f"\nComparing quantization levels for {args.model}...")
        results = run_quantization_comparison(
            model_name=args.model,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            num_runs=args.num_runs,
        )
    elif args.compare_models:
        print("\nComparing models...")
        results = run_model_comparison(
            models=["tiny", "phi2"],
            quantization=args.quantization,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            num_runs=args.num_runs,
        )
    else:
        results = [run_single_benchmark(
            model_name=args.model,
            quantization=args.quantization,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            num_runs=args.num_runs,
            prompts=DEFAULT_PROMPTS,
        )]

    # Save results
    save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
