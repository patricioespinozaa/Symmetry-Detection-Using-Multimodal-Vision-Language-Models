"""
molmo_multiview_runner.py
-------------------------
Batch multi-view pointing inference with Molmo2 at scale.

For each object × resolution × illumination × n_views combination, runs
Molmo2 pointing and saves results to a cumulative JSON file. Results are
accumulated per (object, resolution, illumination) — re-running with new
view groups adds new keys without overwriting existing ones.

Prompt modes
------------
Four modes controlled by --prompt-mode:

  global   All view groups use PROMPT_GLOBAL (one prompt for 1..N images).
  single   All view groups use PROMPT_SINGLE (single-image prompt, called
           once per image — N calls per group).
  multi    All view groups use PROMPT_MULTI  (multi-image prompt, one call
           per group). Falls back to PROMPT_SINGLE when n_views == 1.
  auto     n_views == 1  → PROMPT_SINGLE (one call, single-image format)
           n_views  > 1  → PROMPT_MULTI  (one call, multi-image format)
           This is the recommended mode.

Output structure
----------------
<renders_root>/<symmetry_type>/<object_id>/<image_size>/<illumination>/
    molmo_multiview.json    ← cumulative results, one key per n_views

JSON format
-----------
{
  "6": {
    "prompt_mode": "auto",
    "prompt_used": "...",
    "raw_output": "...",          # list of strings when prompt_mode is "single"
    "points_by_image": {
      "0": [{"obj_id": 1, "x": 450.0, "y": 230.0}, ...],
      "1": [...]
    },
    "images_sent": [...],
    "n_points": 4,
    "n_images_with_points": 3
  },
  "14": { ... },
  ...
}

Resumability
------------
Each (object, resolution, illumination, n_views) combination is skipped if
its key already exists in the JSON.

Usage
-----
    # Recommended: auto mode
    CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \\
        --renders-root ../data/renders \\
        --symmetry-type axis_sym \\
        --view-groups 1 6 14 26 \\
        --prompt-mode auto

    # Global prompt for all view groups
    CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \\
        --renders-root ../data/renders \\
        --symmetry-type axis_sym \\
        --prompt-mode global

    # Single-image prompt, N calls per group
    CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \\
        --renders-root ../data/renders \\
        --symmetry-type axis_sym \\
        --prompt-mode single

    # Multi-image prompt, 1 call per group
    CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \\
        --renders-root ../data/renders \\
        --symmetry-type axis_sym \\
        --prompt-mode multi

    # Two GPUs
    CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \\
        --renders-root ../data/renders \\
        --symmetry-type axis_sym \\
        --gpu-id 0 --num-gpus 2 --prompt-mode auto

    CUDA_VISIBLE_DEVICES=1 python MolmoPointing/molmo_multiview_runner.py \\
        --renders-root ../data/renders \\
        --symmetry-type plane_sym \\
        --gpu-id 1 --num-gpus 2 --prompt-mode auto
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

# ── Prompts ───────────────────────────────────────────────────────────────────

PROMPT_SINGLE = """You are given ONE image of a 3D object.

The object has ONE dominant rotational symmetry axis.

Your task is to estimate the 2D projection of the object's main symmetry axis.

Return TWO distant points lying on the same projected symmetry axis line.

IMPORTANT RULES:
- The two points MUST be different.
- The two points MUST be as far apart as possible along the symmetry axis.
- The points MUST lie on the visible object.
- Use the object's silhouette and geometry to infer the axis.
- Prefer the longest visible symmetry direction through the object.
- Do NOT place both points at the image center.
- If the object is approximately vertical, the axis should also be approximately vertical.

Output ONLY:

<points coords="1 1 X1 Y1 2 X2 Y2">"""


PROMPT_MULTI = """You are given multiple views of the SAME 3D object.

The object has ONE dominant axis of rotational symmetry.

Your task is to estimate the SAME global 3D symmetry axis across all views \
and project that axis into each image.

For each image:
1. Identify the visible projection of the global symmetry axis.
2. Return:
   - one point where the projected axis intersects the object near the TOP,
   - one point where the projected axis intersects the object near the BOTTOM.

IMPORTANT RULES:
- The 2 points of each image MUST be collinear with the axis projection.
- Use the SAME global axis consistently across all views.
- Do NOT place all points at the image center unless the axis projection is truly degenerate.
- Infer the axis from ALL views jointly before answering.
- Even if some views are ambiguous, maintain consistency with the other views.
- Only output points for the dominant rotational symmetry axis.

Output format (one entry per image, separated by semicolons):

<points coords="1 1 Xtop Ytop 2 Xbottom Ybottom; 2 1 Xtop Ytop 2 Xbottom Ybottom; 3 1 Xtop Ytop 2 Xbottom Ybottom">

Where each entry is: image_index obj_id X Y
- obj_id 1 = TOP endpoint
- obj_id 2 = BOTTOM endpoint

Return ONLY the <points ...> block."""


PROMPT_GLOBAL = """You are given one or more images of the SAME 3D object.

The object has ONE dominant rotational symmetry axis.

Your task is to estimate the projection of the object's main symmetry axis in EACH image.

For every image, return TWO points that lie on the projected symmetry axis.

IMPORTANT RULES:
- The two points of each image MUST be different.
- The two points MUST lie on the same symmetry axis line.
- The points should be as far apart as possible along the visible axis.
- The points MUST lie on the visible object region.
- Use the object's geometry and silhouette to infer the axis.
- When multiple images are provided, use all views jointly to maintain consistency.
- Even if some views are ambiguous, keep the predicted axis consistent across views.
- Do NOT collapse points to the image center unless the axis projection is truly degenerate.

Output format:

<points coords="1 1 X1 Y1 2 X2 Y2; 2 1 X1 Y1 2 X2 Y2; 3 1 X1 Y1 2 X2 Y2">

Where each entry is: image_index obj_id X Y
- obj_id 1 = first endpoint of the projected axis
- obj_id 2 = second endpoint of the projected axis

Return ONLY the <points ...> block."""


PROMPT_MODES = ("global", "single", "multi", "auto")

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_ID      = "allenai/Molmo2-8B"
OUTPUT_FILE   = "molmo_multiview.json"

DEFAULT_VIEW_GROUPS   = [1, 6, 14, 26, 42, 62, 86, 114]
DEFAULT_IMAGE_SIZES   = [224, 448, 1136]
DEFAULT_ILLUMINATIONS = ["flat", "brighter", "darker"]

# ── Model singleton ───────────────────────────────────────────────────────────

_processor = None
_model     = None


def get_model():
    global _processor, _model
    if _processor is None or _model is None:
        print(f"[model] Loading {MODEL_ID} ...")
        _processor = AutoProcessor.from_pretrained(
            MODEL_ID, trust_remote_code=True, device_map="auto"
        )
        _model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        _model.eval()
        print("[model] Ready.")
    return _processor, _model


# ── Metadata helpers ──────────────────────────────────────────────────────────

def load_metadata(render_dir: Path) -> list[dict]:
    path = render_dir / "metadata_all.json"
    if not path.exists():
        raise FileNotFoundError(f"metadata_all.json not found: {render_dir}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_n_views_entries(metadata: list[dict], n_views: int) -> list[dict]:
    """First n_views viewpoints, sorted by index."""
    entries = [m for m in metadata if m["index"] < n_views]
    return sorted(entries, key=lambda e: e["index"])


# ── Low-level model call ──────────────────────────────────────────────────────

def _call_model(images: list[Image.Image], prompt: str) -> str:
    """Single model call with 1..N images. Returns raw decoded output."""
    processor, model = get_model()

    content = [{"type": "text", "text": prompt}]
    for img in images:
        content.append({"type": "image", "image": img})

    messages = [{"role": "user", "content": content}]
    inputs   = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    max_new_tokens = 300 + len(images) * 100

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    return processor.tokenizer.decode(
        output_ids[0, inputs["input_ids"].size(1):],
        skip_special_tokens=True,
    )


# ── Coordinate parsers ────────────────────────────────────────────────────────

def parse_single_coords(text: str) -> dict[str, list[dict]]:
    """
    Single-image Molmo2 format:
        <points coords="RADIO  ID X Y  ID X Y ...">
    Returns {"0": [{obj_id, x, y}, ...]}
    """
    match = re.search(r'coords=["\']([^"\']+)["\']', text)
    if not match:
        return {}
    raw = [float(n) for n in match.group(1).split()]
    if len(raw) < 4:
        return {}
    raw = raw[1:]   # drop radius token
    pts = []
    for i in range(0, len(raw) - 2, 3):
        pts.append({"obj_id": int(raw[i]), "x": raw[i + 1], "y": raw[i + 2]})
    return {"0": pts} if pts else {}


def parse_multi_coords(text: str, n_images: int) -> dict[str, list[dict]]:
    """
    Multi-image Molmo2 format:
        <points coords="img_idx obj_id X Y; img_idx obj_id X Y ...">
    img_idx is 1-based; stored as 0-based string keys.
    Returns {"0": [...], "1": [...], ...}
    """
    match = re.search(r'coords=["\']([^"\']+)["\']', text)
    if not match:
        return {}

    result: dict[str, list[dict]] = {}
    for group in match.group(1).split(";"):
        group = group.strip()
        if not group:
            continue
        nums = group.split()
        if len(nums) < 4:
            continue
        try:
            img_idx = int(float(nums[0])) - 1   # 1-based → 0-based
        except ValueError:
            continue
        if img_idx < 0 or img_idx >= n_images:
            continue

        key  = str(img_idx)
        rest = nums[1:]
        pts  = []
        for i in range(0, len(rest) - 2, 3):
            try:
                pts.append({
                    "obj_id": int(float(rest[i])),
                    "x":      float(rest[i + 1]),
                    "y":      float(rest[i + 2]),
                })
            except ValueError:
                continue
        if pts:
            result.setdefault(key, []).extend(pts)

    return result


# ── Inference modes ───────────────────────────────────────────────────────────

def run_global(images: list[Image.Image]) -> dict:
    """One call with all images using PROMPT_GLOBAL."""
    raw           = _call_model(images, PROMPT_GLOBAL)
    points_by_img = parse_multi_coords(raw, n_images=len(images))
    return {
        "prompt_used":     PROMPT_GLOBAL,
        "raw_output":      raw,
        "points_by_image": points_by_img,
    }


def run_single_mode(images: list[Image.Image]) -> dict:
    """
    One call per image using PROMPT_SINGLE.
    raw_output stored as list (one entry per image).
    points_by_image keys correspond to image position in the original list.
    """
    raw_outputs   = []
    points_by_img = {}

    for i, img in enumerate(images):
        raw = _call_model([img], PROMPT_SINGLE)
        raw_outputs.append(raw)
        pts = parse_single_coords(raw)
        if "0" in pts:
            points_by_img[str(i)] = pts["0"]

    return {
        "prompt_used":     PROMPT_SINGLE,
        "raw_output":      raw_outputs,
        "points_by_image": points_by_img,
    }


def run_multi(images: list[Image.Image]) -> dict:
    """One call with all images using PROMPT_MULTI."""
    raw           = _call_model(images, PROMPT_MULTI)
    points_by_img = parse_multi_coords(raw, n_images=len(images))
    return {
        "prompt_used":     PROMPT_MULTI,
        "raw_output":      raw,
        "points_by_image": points_by_img,
    }


def run_inference(
    images:      list[Image.Image],
    prompt_mode: str,
) -> dict:
    """
    Dispatch to the correct inference mode.

    global  → PROMPT_GLOBAL, 1 call for all images
    single  → PROMPT_SINGLE, 1 call per image
    multi   → PROMPT_MULTI,  1 call for all images (fallback to single if n==1)
    auto    → single if n==1, multi if n>1  [recommended]
    """
    n = len(images)

    if prompt_mode == "global":
        return run_global(images)

    elif prompt_mode == "single":
        return run_single_mode(images)

    elif prompt_mode == "multi":
        return run_single_mode(images) if n == 1 else run_multi(images)

    elif prompt_mode == "auto":
        return run_single_mode(images) if n == 1 else run_multi(images)

    else:
        raise ValueError(f"Unknown prompt_mode: {prompt_mode!r}")


# ── JSON helpers ──────────────────────────────────────────────────────────────

def load_results(json_path: Path) -> dict:
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(json_path: Path, data: dict) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ── Per-object processor ──────────────────────────────────────────────────────

def process_object(
    object_dir:   Path,
    view_groups:  list[int],
    sizes:        list[int],
    lightings:    list[str],
    prompt_mode:  str,
) -> None:
    """
    Process one object across all (size, illumination, n_views) combinations.
    Skips any (size, illumination, n_views) already present in the JSON.
    """
    for size in sizes:
        for lighting in lightings:
            render_dir = object_dir / str(size) / lighting
            if not render_dir.exists():
                continue

            try:
                metadata = load_metadata(render_dir)
            except FileNotFoundError:
                continue

            json_path = render_dir / OUTPUT_FILE
            results   = load_results(json_path)

            pending = [n for n in view_groups if str(n) not in results]
            if not pending:
                continue

            for n_views in pending:
                entries = get_n_views_entries(metadata, n_views)
                if not entries:
                    continue

                images = [
                    Image.open(render_dir / e["filename"]).convert("RGB")
                    for e in entries
                ]

                inference     = run_inference(images, prompt_mode)
                points_by_img = inference["points_by_image"]
                n_pts         = sum(len(v) for v in points_by_img.values())

                results[str(n_views)] = {
                    "prompt_mode":           prompt_mode,
                    "prompt_used":           inference["prompt_used"],
                    "raw_output":            inference["raw_output"],
                    "points_by_image":       points_by_img,
                    "images_sent": [
                        {
                            "img_idx":   i,
                            "filename":  e["filename"],
                            "index":     e["index"],
                            "azimuth":   e["azimuth"],
                            "elevation": e["elevation"],
                            "eye":       e["eye"],
                            "R":         e["R"],
                            "T":         e["T"],
                        }
                        for i, e in enumerate(entries)
                    ],
                    "n_points":             n_pts,
                    "n_images_with_points": len(points_by_img),
                }

                # Write after every n_views to survive interruptions
                save_results(json_path, results)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch Molmo2 multi-view pointing — cumulative JSON output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--renders-root",  required=True,
                   help="Root folder produced by data_render.py")
    p.add_argument("--symmetry-type", required=True,
                   choices=["axis_sym", "plane_sym"])

    p.add_argument("--prompt-mode", default="auto", choices=PROMPT_MODES,
                   help=(
                       "global = PROMPT_GLOBAL for all groups (1 call/group); "
                       "single = PROMPT_SINGLE per image (N calls/group); "
                       "multi  = PROMPT_MULTI for all groups (1 call/group, "
                       "fallback to single when n_views==1); "
                       "auto   = single if n_views==1, multi otherwise [recommended]"
                   ))

    p.add_argument("--gpu-id",   type=int, default=0)
    p.add_argument("--num-gpus", type=int, default=1)

    p.add_argument("--view-groups", type=int, nargs="+",
                   default=DEFAULT_VIEW_GROUPS)
    p.add_argument("--sizes",       type=int, nargs="+",
                   default=DEFAULT_IMAGE_SIZES)
    p.add_argument("--lightings",   type=str, nargs="+",
                   default=DEFAULT_ILLUMINATIONS,
                   choices=["flat", "darker", "brighter"])

    return p.parse_args()


def preview(args: argparse.Namespace, objects: list[Path]) -> None:
    mode_desc = {
        "global": "PROMPT_GLOBAL for all groups — 1 call per group",
        "single": "PROMPT_SINGLE per image — N calls per group",
        "multi":  "PROMPT_MULTI for all groups — 1 call per group (fallback to single if n==1)",
        "auto":   "PROMPT_SINGLE if n_views==1, PROMPT_MULTI if n_views>1  [recommended]",
    }
    n_configs = len(args.sizes) * len(args.lightings) * len(args.view_groups)
    print("\n========== MOLMO MULTIVIEW RUNNER ==========")
    print(f"Renders root  : {args.renders_root}")
    print(f"Symmetry type : {args.symmetry_type}")
    print(f"Prompt mode   : {args.prompt_mode}")
    print(f"              : {mode_desc[args.prompt_mode]}")
    print(f"GPU id/total  : {args.gpu_id} / {args.num_gpus}")
    print(f"Objects       : {len(objects)}")
    print(f"Sizes         : {args.sizes}")
    print(f"Lightings     : {args.lightings}")
    print(f"View groups   : {args.view_groups}")
    print(f"Combinations  : {n_configs} per object")
    print(f"(Existing JSON keys are skipped automatically)")
    print("============================================\n")
    if input("Type 'OK' to start: ").strip() != "OK":
        print("Cancelled.")
        sys.exit(0)


def main() -> None:
    args = parse_args()

    symmetry_dir = Path(args.renders_root) / args.symmetry_type
    if not symmetry_dir.exists():
        print(f"[error] Not found: {symmetry_dir}")
        sys.exit(1)

    all_objects = sorted(d for d in symmetry_dir.iterdir() if d.is_dir())
    if not all_objects:
        print(f"[error] No object folders in {symmetry_dir}")
        sys.exit(1)

    objects = all_objects[args.gpu_id :: args.num_gpus]

    preview(args, objects)

    for obj_dir in tqdm(
        objects,
        desc=f"GPU {args.gpu_id}",
        unit="obj",
        dynamic_ncols=True,
        bar_format="{desc:<12} {bar:40} {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ):
        process_object(
            object_dir  = obj_dir,
            view_groups = args.view_groups,
            sizes       = args.sizes,
            lightings   = args.lightings,
            prompt_mode = args.prompt_mode,
        )

    print(f"\n[GPU {args.gpu_id}] Done.")


if __name__ == "__main__":
    main()
