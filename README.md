# LLM-Inference-Bench

A GPU benchmarking toolkit for measuring Large Language Model (LLM) inference performance. This tool evaluates throughput, latency, and memory usage across different models, quantization levels, and batch sizes.

## Features

- **Throughput Benchmarking**: Measure tokens generated per second
- **Latency Analysis**: Time-to-first-token (TTFT) and per-token latency
- **Memory Profiling**: Track GPU VRAM usage during inference
- **Quantization Comparison**: Compare FP16, INT8, and INT4 performance
- **Batch Scaling Analysis**: Evaluate throughput scaling with batch size
- **Automated Visualization**: Generate publication-ready charts

## Supported Models

| Model | Size | Colab Compatible |
|-------|------|------------------|
| TinyLlama-1.1B | 1.1B | ✅ |
| Microsoft Phi-2 | 2.7B | ✅ |
| Google Gemma-2B | 2B | ✅ |
| Mistral-7B | 7B | ✅ (INT4/INT8) |

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Single Model Benchmark

```bash
python run_benchmark.py --model tiny --quantization fp16
```

### Compare Quantization Levels

```bash
python run_benchmark.py --model phi2 --compare-quantization
```

### Compare Models

```bash
python run_benchmark.py --compare-models --quantization fp16
```

## Usage on Google Colab

```python
# Install dependencies
!pip install -q torch transformers accelerate bitsandbytes auto-gptq

# Clone repository
!git clone https://github.com/YOUR_USERNAME/LLM-Inference-Bench.git
%cd LLM-Inference-Bench

# Run benchmark
!python run_benchmark.py --model tiny --compare-quantization
```

## Key Metrics

### Throughput (tokens/second)
Measures how many tokens the model can generate per second. Higher is better for batch processing workloads.

### Time to First Token (TTFT)
Time from request to first token generation. Critical for interactive applications.

### Per-Token Latency
Average time to generate each subsequent token after the first.

### Peak Memory Usage
Maximum GPU VRAM consumed during inference. Important for deployment planning.

## Example Results

| Model | Quantization | Throughput (tok/s) | TTFT (ms) | Memory (GB) |
|-------|--------------|-------------------|-----------|-------------|
| TinyLlama-1.1B | FP16 | 45.2 | 32.1 | 2.4 |
| TinyLlama-1.1B | INT8 | 52.8 | 28.5 | 1.3 |
| TinyLlama-1.1B | INT4 | 61.3 | 25.2 | 0.8 |
| Phi-2 | FP16 | 32.1 | 45.6 | 5.8 |
| Phi-2 | INT8 | 38.4 | 38.2 | 3.1 |

*Results measured on NVIDIA T4 GPU (Google Colab)*

## Output

Results are saved to the `results/` directory:
- `benchmark_TIMESTAMP.json` - Raw benchmark data
- `summary_TIMESTAMP.csv` - Summary table
- `throughput_TIMESTAMP.png` - Throughput comparison chart
- `latency_TIMESTAMP.png` - Latency breakdown chart
- `memory_TIMESTAMP.png` - Memory usage chart

## Project Structure

```
LLM-Inference-Bench/
├── benchmarks/
│   ├── throughput.py    # Throughput measurement
│   ├── latency.py       # Latency measurement
│   └── memory.py        # Memory profiling
├── utils/
│   ├── model_loader.py  # Model loading with quantization
│   ├── metrics.py       # Metrics data classes
│   └── visualize.py     # Chart generation
├── notebooks/
│   └── LLM_Inference_Bench.ipynb  # Colab notebook
├── results/             # Output directory
├── run_benchmark.py     # Main entry point
└── requirements.txt
```

## Technical Details

### Quantization Methods
- **FP16**: Half-precision floating point (baseline)
- **INT8**: 8-bit integer quantization via bitsandbytes
- **INT4**: 4-bit quantization with NF4 data type

### Measurement Methodology
1. **Warmup**: 2 iterations to stabilize GPU state
2. **Benchmark**: Multiple runs (default: 5) with timing
3. **CUDA Synchronization**: Ensures accurate GPU timing
4. **Memory Reset**: Clear cache between runs

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)
- 8GB+ GPU VRAM for INT4 models
- 16GB+ GPU VRAM for FP16 models

## Contributing

Contributions welcome! Areas of interest:
- Additional model support
- New benchmark metrics
- Performance optimizations
- Multi-GPU benchmarking

## License

MIT License

## Acknowledgments

- HuggingFace Transformers
- bitsandbytes library
- PyTorch team
