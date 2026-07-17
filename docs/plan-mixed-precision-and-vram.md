# Plan: Mixed Precision & VRAM Reduction for FramePack-Studio

## Problem

OOM errors on ~6 GB GPUs:

```
expandable_segments: memory mapping failed with OOM on device 0
while trying to map 20971520 bytes (free: 7143424, total: 6017515520)
```

Only 7–13 MB free out of ~6 GB. The model weights + activations are exhausting VRAM.

---

## Current State of Precision

The codebase is **already almost entirely in `bfloat16`** for the transformer model:

| Location | Code | Purpose |
|---|---|---|
| Model loading (`original_generator.py`, `video_base_generator.py`) | `torch_dtype=torch.bfloat16` | Weights stored as bfloat16 |
| Sampling (`k_diffusion_hunyuan.py`) | `dtype=torch.bfloat16` (default param) | Passes dtype through sampler |
| Transformer forward (`wrapper.py`) | `x = x.to(dtype)` then `...float()` | Casts input to bfloat16, output back to float32 |
| Internal proj_out (`hunyuan_video_packed.py` line ~1067) | `hidden_states.to(dtype=torch.float32)` | Upcast for final projection precision |

### Existing VRAM Mitigations Already in Place

- **`DynamicSwapInstaller`** — CPU offloading for low VRAM mode (per-block parameter transfer)
- **`_cleanup_hooks`** — `torch.cuda.empty_cache()` after each transformer block
- **Manual model offloading** — VAE, text encoders, image encoder moved off GPU when not needed
- **bfloat16 weights** — half the size of float32

### Why `torch.autocast` Mixed Precision Won't Help

Switching from `bfloat16` to `float16` saves **zero memory** — both are 16-bit. `torch.autocast` would only help if large `float32` intermediate computations were running on GPU, but the code already explicitly manages dtype casts. True mixed precision would have **near-zero benefit** here.

---

## Options for Reducing VRAM

| Approach | Feasibility | VRAM Impact | Effort | Notes |
|---|---|---|---|---|
| **Reduce resolution** (e.g., 512→480) | ✅ Trivial | ~40% less for latents | 1 line config | Already adjustable |
| **Reduce `vae_batch_size`** | ✅ Trivial | Less peak VRAM during VAE encode | 0 lines | Already configurable |
| **4-bit quantization** (`bitsandbytes` NF4) | ✅ Code exists (`quantize.py`) | ~60-70% reduction for weights | ~5 lines to call it | `quantize_model_to_4bit()` already implemented but never called |
| **`float8` weight-only quantization** (torch 2.3+) | ⚠️ Moderate | ~50% reduction | Need to implement | |
| **`torch.autocast` mixed precision** | ⚠️ Minimal | Near zero — already bfloat16 | ~2 lines but pointless | |
| **Lower KV cache / attention slicing** | ⚠️ Moderate | Reduces activation memory | Significant refactor | |
| **Gradient checkpointing** | ❌ N/A | — | — | Inference only, not applicable |

---

## Implementation: 4-bit NF4 Quantization ✅ DONE

### What Was Implemented

1. **`modules/settings.py`** — Added `use_4bit_quantization: False` default setting
2. **`modules/generators/base_generator.py`** — Added `self.use_4bit_quantization` from settings + `_apply_4bit_quantization()` method with graceful error handling
3. **All generators** — Added `self._apply_4bit_quantization()` call in `load_model()` after model creation/config, before `DynamicSwapInstaller`:
   - `original_generator.py`
   - `f1_generator.py`
   - `video_base_generator.py`
   - `original_with_endframe_generator.py` (inherits from OriginalModelGenerator)
4. **`modules/interface.py`** — Added "4-bit Quantization (NF4)" checkbox in Settings tab with auto-save
5. **`studio.py`** — Auto-enables 4-bit quantization for low VRAM GPUs (≤6GB)
6. **`diffusers_helper/quantize.py`** — Made `bitsandbytes` import lazy (was top-level, would crash on import if not installed)
7. **`requirements.txt`** — Added `bitsandbytes` as optional dependency with install instructions
8. **Low VRAM warning** — Updated to mention 4-bit quantization option

### How It Works

When `use_4bit_quantization` is enabled:
1. The transformer model is loaded as usual in bfloat16
2. After `.eval()` / `.requires_grad_(False)`, `_apply_4bit_quantization()` replaces all `nn.Linear` layers with `bnb.nn.Linear4bit`
3. Weights are stored as NF4 (4-bit) — reducing model memory from ~12 GB to ~3 GB
4. Quantization happens lazily when parameters are first moved to CUDA (compatible with `DynamicSwapInstaller`)
5. Compute dtype is `torch.float16` for dequantization during forward pass
6. If `bitsandbytes` is not installed, the feature is skipped with a warning

### Key Design Decisions

- **Applied before DynamicSwapInstaller**: Quantization reduces the per-parameter memory before the swap system moves blocks to GPU
- **Graceful fallback**: If `bitsandbytes` isn't installed, a warning is printed and generation continues without quantization
- **Settings-driven**: Users can toggle on/off from the UI; setting persists across restarts
- **Auto-enabled for low VRAM**: Cards with ≤6GB VRAM automatically get 4-bit quantization enabled

### Testing Checklist

- [ ] Verify generation works with 4-bit quantization enabled on a low VRAM GPU
- [ ] Compare output quality with and without quantization
- [ ] Verify LoRA loading still works on quantized models
- [ ] Verify Video, Video F1, Original, F1, and Original with Endframe models all work
- [ ] Verify graceful fallback when `bitsandbytes` is not installed
- [ ] Verify the UI toggle persists across restarts

---

## Alternative: Lower Resolution

If quantization is too invasive, the simplest immediate fix:

```python
# Reduce default resolution from 640 to 480 or 512
resolutionW = 480
resolutionH = 480
```

Latent memory scales with `(H/8) * (W/8)`, so 640→480 saves ~44% on latent VRAM.

---

## Future Considerations

- **float8 quantization** — as a future option if NF4 quality is insufficient
- **Resolution presets** — as a complementary UI option
- **Attention slicing** — for further VRAM reduction on very small GPUs