"""
Model loading utilities for LLM-Inference-Bench.
Supports various quantization levels and model configurations.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Tuple, Optional


# Supported models optimized for Colab T4 (16GB VRAM)
SUPPORTED_MODELS = {
    "tiny": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "phi2": "microsoft/phi-2",
    "gemma-2b": "google/gemma-2b",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
}


def get_quantization_config(quantization: str) -> Optional[BitsAndBytesConfig]:
    """Get BitsAndBytes quantization config."""
    if quantization == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    elif quantization == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    return None


def load_model(
    model_name: str,
    quantization: str = "fp16",
    device: str = "cuda",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model with specified quantization.

    Args:
        model_name: Model identifier (key from SUPPORTED_MODELS or HuggingFace path)
        quantization: One of 'fp32', 'fp16', 'int8', 'int4'
        device: Target device ('cuda' or 'cpu')

    Returns:
        Tuple of (model, tokenizer)
    """
    # Resolve model name
    model_path = SUPPORTED_MODELS.get(model_name, model_name)

    print(f"Loading {model_path} with {quantization} precision...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure model loading
    quant_config = get_quantization_config(quantization)

    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto" if device == "cuda" else None,
    }

    if quantization == "fp32":
        model_kwargs["torch_dtype"] = torch.float32
    elif quantization == "fp16":
        model_kwargs["torch_dtype"] = torch.float16
    elif quant_config:
        model_kwargs["quantization_config"] = quant_config

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    if device == "cuda" and quant_config is None:
        model = model.to(device)

    model.eval()

    print(f"Model loaded successfully on {next(model.parameters()).device}")

    return model, tokenizer


def get_model_size_mb(model: AutoModelForCausalLM) -> float:
    """Calculate model size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)
