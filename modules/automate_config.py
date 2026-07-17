"""
Parameter definitions for FramePack Studio automation.

This module defines the exact parameter order and defaults used by the
FramePack Studio Gradio API. The order MUST match the `ips` list in
`modules/interface.py` (line 1422) and the unpacking in
`process_with_queue_update()` (line 1273).
"""

from typing import Any, Dict, List, Optional, OrderedDict
from collections import OrderedDict as OD


# ──────────────────────────────────────────────
# PARAMETER DEFINITION
# Each entry: (ips_index, default, type, description)
# Indices 0–28 are the fixed ips list.
# Indices 29+ are dynamic LoRA slider values (one per loaded LoRA).
# model_type is prepended BEFORE the ips list (index -1 conceptually).
# ──────────────────────────────────────────────

PARAMETER_META: OrderedDict[str, tuple] = OD([
    # ── Fixed parameters (ips list order) ──
    ("input_image",               (0,  None,    "str|None",    "Path to input image file")),
    ("input_video",               (1,  None,    "str|None",    "Path to input video file (for Video models)")),
    ("end_frame_image",           (2,  None,    "str|None",    "Path to end frame image")),
    ("end_frame_strength",        (3,  1.0,     "float",       "End frame influence (0.05–1.0)")),
    ("prompt",                    (4,  "[1s: The person waves hello]", "str", "Generation prompt")),
    ("n_prompt",                  (5,  "",      "str",         "Negative prompt")),
    ("seed",                      (6,  2500,    "int",         "Random seed")),
    ("randomize_seed",            (7,  False,   "bool",        "Randomize seed each job")),
    ("total_second_length",       (8,  6,       "float",       "Video length in seconds")),
    ("latent_window_size",        (9,  5,       "int",         "Latent window size (1–33)")),
    ("steps",                     (10, 25,      "int",         "Diffusion steps (1–100)")),
    ("cfg",                       (11, 1.0,     "float",       "CFG scale (1.0–3.0)")),
    ("gs",                        (12, 10.0,    "float",       "Distilled CFG scale (1.0–32.0)")),
    ("rs",                        (13, 0.0,     "float",       "CFG re-scale (0.0–1.0)")),
    ("cache_type",                (14, "MagCache", "str",      "Cache strategy: None, TeaCache, MagCache")),
    ("teacache_num_steps",        (15, 25,      "int",         "TeaCache steps (1–50)")),
    ("teacache_rel_l1_thresh",    (16, 0.15,    "float",       "TeaCache rel_l1_threshold (0.01–1.0)")),
    ("magcache_threshold",        (17, 0.1,     "float",       "MagCache threshold (0.01–1.0)")),
    ("magcache_max_consecutive_skips", (18, 2,  "int",         "MagCache max consecutive skips (1–5)")),
    ("magcache_retention_ratio",  (19, 0.25,    "float",       "MagCache retention ratio (0.0–1.0)")),
    ("blend_sections",            (20, 4,       "int",         "Blend sections (0–10)")),
    ("latent_type",               (21, "Noise", "str",         "Latent type: Noise, White, Black, Green Screen")),
    ("clean_up_videos",           (22, True,    "bool",        "Clean up intermediate video files")),
    ("selected_loras",            (23, [],      "list",        "List of selected LoRA names")),
    ("resolutionW",               (24, 480,     "int",         "Width (128–768, step 32)")),
    ("resolutionH",               (25, 480,     "int",         "Height (128–768, step 32)")),
    ("combine_with_source",       (26, True,    "bool",        "Combine with source video (Video models)")),
    ("num_cleaned_frames",        (27, 5,       "int",         "Number of context frames (Video models, 2–10)")),
    ("lora_names_states",         (28, [],      "list",        "All loaded LoRA names (gr.State)")),
])


MODEL_TYPES = [
    "Original",
    "Original with Endframe",
    "F1",
    "Video",
    "Video with Endframe",
    "Video F1",
]

LATENT_TYPES = ["Noise", "White", "Black", "Green Screen"]
CACHE_TYPES = ["None", "TeaCache", "MagCache"]


def build_params_list(
    model_type: str,
    params: Dict[str, Any],
    lora_weights_dict: Optional[Dict[str, float]] = None,
) -> list:
    """
    Build the flat parameter list in the exact order expected by the Gradio API.

    Args:
        model_type: One of MODEL_TYPES.
        params: Dict of parameter name -> value (keys matching PARAMETER_META).
        lora_weights_dict: Optional dict of {lora_name: weight} for LoRA weights.

    Returns:
        A flat list: [model_type, input_image, input_video, ..., lora_weights_dict]
    """
    result = [model_type]

    for name, (_, default, _, _) in PARAMETER_META.items():
        value = params.get(name, default)
        result.append(value)

    # Append lora_weights_dict as the last parameter
    if lora_weights_dict:
        result.append(lora_weights_dict)
    else:
        result.append({})

    return result


def get_parameter_count(lora_count: int = 0) -> int:
    """Total number of parameters including model_type and lora_weights_dict."""
    return 1 + len(PARAMETER_META) + 1  # +1 for lora_weights_dict


def match_endpoint_by_input_count(
    endpoints: Dict[str, Any], target_count: int
) -> Optional[str]:
    """
    Find an API endpoint whose input count matches the expected parameter count.

    Used for endpoint discovery when the auto-generated name is unknown.
    """
    for name, info in endpoints.get("named_endpoints", {}).items():
        params = info.get("parameters", [])
        if len(params) == target_count:
            return name

    for idx, info in endpoints.get("unnamed_endpoints", {}).items():
        params = info.get("parameters", [])
        if len(params) == target_count:
            return f"/{idx}"

    return None
