"""
Helper to quantize HunyuanVideoTransformer3DModelPacked and other custom models
to 4-bit using BitsAndBytes.

Replaces all nn.Linear layers with bnb.nn.Linear4bit after loading.
Original weight storage is shared (not cloned) to avoid doubling CPU RAM.
Quantization to 4-bit happens lazily when the module is moved to CUDA.
"""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn


@torch.no_grad()
def quantize_model_to_4bit(
    model: nn.Module,
    compute_dtype: torch.dtype = torch.float16,
    quant_type: str = "nf4",
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Replace all nn.Linear layers with 4-bit BNB Linear4bit.

    Works on any nn.Module. Original weight storage is shared (not cloned)
    to minimize peak CPU RAM.

    Args:
        model: The PyTorch model to quantize
        compute_dtype: Dequantized compute dtype during forward
        quant_type: 'nf4' or 'fp4'
        device: Move model here after quantization

    Returns:
        Quantized model (modified in-place)
    """
    # ------------------------------------------------------------------
    # Phase 1: collect all replacements (don't modify while iterating)
    # ------------------------------------------------------------------
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError(
            "bitsandbytes is required for 4-bit quantization. "
            "Install it with: pip install bitsandbytes"
        )

    replacements = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        parent_path = name.rpartition(".")[0]
        child_name = name.rpartition(".")[2]

        # Walk down from root to find the parent
        parent = model
        if parent_path:
            for part in parent_path.split("."):
                parent = getattr(parent, part)

        # Build the quantized replacement.  BNB stores weights as
        # Params4bit on CPU; 4-bit conversion happens on first CUDA access.
        new_linear = bnb.nn.Linear4bit(
            input_features=module.in_features,
            output_features=module.out_features,
            bias=module.bias is not None,
            compute_dtype=compute_dtype,
            quant_type=quant_type,
            quant_storage=compute_dtype,
        )

        # Share original weight storage (no clone → no RAM spike).
        # Params4bit wraps the tensor; 4-bit quantisation occurs on
        # .cuda() / .to(device='cuda').
        new_linear.weight = bnb.nn.Params4bit(
            module.weight.data,
            requires_grad=False,
            quant_type=quant_type,
        )
        new_linear.weight.compute_dtype = compute_dtype
        new_linear.weight.quant_storage = compute_dtype

        # Share bias storage too
        if module.bias is not None:
            new_linear.bias = module.bias

        replacements.append((parent, child_name, new_linear))

    # ------------------------------------------------------------------
    # Phase 2: apply all replacements
    # ------------------------------------------------------------------
    num_replaced = 0
    for parent, child_name, new_linear in replacements:
        setattr(parent, child_name, new_linear)
        num_replaced += 1

    print(f"Quantized {num_replaced} nn.Linear layers to 4-bit ({quant_type})")

    if device is not None:
        model = model.to(device=device)
        # BNB's Params4bit auto-converts to 4-bit when moved to CUDA

    return model


def _cleanup_hook(module, _input, output):
    """
    Forward hook that frees GPU memory after each transformer block's forward pass.
    
    DynamicSwapInstaller creates temporary GPU copies of parameters when they're
    accessed via __getattr__. Without explicit cleanup, these accumulate on GPU
    and cause OOM. This hook runs torch.cuda.empty_cache() after each block to
    free those temporary copies before the next block loads its parameters.
    """
    # Debug: uncomment to verify hooks fire
    # if hasattr(module, "__class__") and "DynamicSwap" in module.__class__.__name__:
    #     print(f"  cleanup hook fired for {module.__class__.__name__}", flush=True)
    torch.cuda.empty_cache()
    return output


def install_block_cleanup_hooks(transformer_model: torch.nn.Module):
    """
    Install per-block forward hooks that free GPU memory between transformer blocks.
    
    Call after DynamicSwapInstaller.install_model() to prevent OOM on low-VRAM cards.
    
    Args:
        transformer_model: The HunyuanVideoTransformer3DModelPacked instance
    """
    hook_count = 0
    
    # Dual-stream transformer blocks
    if hasattr(transformer_model, 'transformer_blocks'):
        for block in transformer_model.transformer_blocks:
            block.register_forward_hook(_cleanup_hook)
            hook_count += 1
    
    # Single-stream transformer blocks
    if hasattr(transformer_model, 'single_transformer_blocks'):
        for block in transformer_model.single_transformer_blocks:
            block.register_forward_hook(_cleanup_hook)
            hook_count += 1
    
    print(f"Installed {hook_count} block cleanup hooks (frees GPU memory between blocks)")
