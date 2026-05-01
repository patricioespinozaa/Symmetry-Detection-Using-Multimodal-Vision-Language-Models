"""
molmo_inference.py
------------------
Single-image inference with Molmo2 and coordinate extraction.

Molmo outputs pointing coordinates in the format:
    <point coords="R ID X Y ID X Y ...">
where coordinates are in the 0–1000 range.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
from PIL import Image

from molmo_model import get_model

# ── Prompt ────────────────────────────────────────────────────────────────────

SYMMETRY_PROMPT = (
    "Find the points where the main axis of symmetry intersects the edges of the shape."
)

# ── Coord extraction ──────────────────────────────────────────────────────────

def extract_coords(text: str) -> list[dict]:
    """
    Parse Molmo pointing output and return a list of point dicts.

    Molmo format inside the coords attribute:
        R  ID X Y  ID X Y  ...
    where R is the radius token (first value, discarded) and each point
    is a triplet (object_id, x, y) in the 0–1000 coordinate space.

    Returns:
        List of dicts: [{"obj_id": int, "x": float, "y": float}, ...]
        Empty list if nothing is found or the format is unexpected.
    """
    match = re.search(r'coords="([^"]+)"', text)
    if not match:
        return []

    try:
        raw = [float(n) for n in match.group(1).split()]
    except ValueError:
        return []

    # Drop the leading radius token
    if len(raw) < 4:
        return []
    raw = raw[1:]

    points = []
    for i in range(0, len(raw) - 2, 3):
        points.append({
            "obj_id": int(raw[i]),
            "x": raw[i + 1],
            "y": raw[i + 2],
        })

    return points


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(image_path: str | Path, prompt: str = SYMMETRY_PROMPT) -> dict:
    """
    Run Molmo2 inference on a single image.

    Args:
        image_path: Path to the input PNG.
        prompt:     Text prompt sent to the model.

    Returns:
        {
            "raw_output": str,          # full decoded model output
            "points":     list[dict],   # extracted coords (may be empty)
            "success":    bool,         # True if at least one point was found
        }
    """
    processor, model = get_model()

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text",  "text": prompt},
                {"type": "image", "image": image},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=200)

    raw_output = processor.tokenizer.decode(
        output_ids[0, inputs["input_ids"].size(1):],
        skip_special_tokens=True,
    )

    points = extract_coords(raw_output)

    return {
        "raw_output": raw_output,
        "points":     points,
        "success":    len(points) > 0,
    }
