"""
Visualization utilities for LLM-Inference-Bench.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Optional


def set_style():
    """Set consistent plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")


def plot_throughput_comparison(
    results: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    title: str = "Throughput Comparison"
) -> plt.Figure:
    """
    Plot throughput comparison across models and quantization levels.
    """
    set_style()

    df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(10, 6))

    x_labels = [f"{r['model_name']}\n({r['quantization']})" for r in results]
    throughputs = [r['throughput']['tokens_per_second'] for r in results]

    bars = ax.bar(x_labels, throughputs, color=sns.color_palette("husl", len(results)))

    ax.set_ylabel("Tokens per Second", fontsize=12)
    ax.set_xlabel("Model (Quantization)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add value labels on bars
    for bar, val in zip(bars, throughputs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_latency_breakdown(
    results: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    title: str = "Latency Breakdown"
) -> plt.Figure:
    """
    Plot latency breakdown (TTFT vs per-token latency).
    """
    set_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    models = [f"{r['model_name']}\n({r['quantization']})" for r in results]
    ttft = [r['latency']['time_to_first_token_ms'] for r in results]
    per_token = [r['latency']['per_token_latency_ms'] for r in results]

    x = range(len(models))
    width = 0.35

    bars1 = ax.bar([i - width/2 for i in x], ttft, width, label='Time to First Token (ms)')
    bars2 = ax.bar([i + width/2 for i in x], per_token, width, label='Per Token Latency (ms)')

    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_xlabel("Model (Quantization)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_memory_usage(
    results: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    title: str = "GPU Memory Usage"
) -> plt.Figure:
    """
    Plot GPU memory usage comparison.
    """
    set_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    models = [f"{r['model_name']}\n({r['quantization']})" for r in results]
    peak_memory = [r['memory']['peak_memory_mb'] / 1024 for r in results]  # Convert to GB

    bars = ax.bar(models, peak_memory, color=sns.color_palette("husl", len(results)))

    ax.set_ylabel("Peak Memory (GB)", fontsize=12)
    ax.set_xlabel("Model (Quantization)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add value labels
    for bar, val in zip(bars, peak_memory):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_batch_scaling(
    results: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    title: str = "Batch Size Scaling"
) -> plt.Figure:
    """
    Plot throughput scaling with batch size.
    """
    set_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    df = pd.DataFrame(results)

    for model in df['model_name'].unique():
        model_data = df[df['model_name'] == model]
        for quant in model_data['quantization'].unique():
            quant_data = model_data[model_data['quantization'] == quant]
            batch_sizes = quant_data['batch_size'].tolist()
            throughputs = [r['tokens_per_second'] for r in quant_data['throughput']]
            ax.plot(batch_sizes, throughputs, marker='o', label=f"{model} ({quant})")

    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Tokens per Second", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def generate_summary_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Generate a summary DataFrame from benchmark results.
    """
    rows = []
    for r in results:
        rows.append({
            "Model": r['model_name'],
            "Quantization": r['quantization'],
            "Batch Size": r['batch_size'],
            "Throughput (tok/s)": f"{r['throughput']['tokens_per_second']:.2f}",
            "TTFT (ms)": f"{r['latency']['time_to_first_token_ms']:.2f}",
            "Per-Token (ms)": f"{r['latency']['per_token_latency_ms']:.2f}",
            "Peak Memory (GB)": f"{r['memory']['peak_memory_mb']/1024:.2f}",
        })

    return pd.DataFrame(rows)
