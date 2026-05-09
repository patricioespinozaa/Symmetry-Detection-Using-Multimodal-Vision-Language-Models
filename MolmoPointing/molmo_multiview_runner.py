"""
molmo_multiview_runner.py
-------------------------
Batch multi-view pointing inference with Molmo2 at scale.

For each object × resolution × illumination × n_views combination, runs
Molmo2 pointing and saves results to a cumulative JSON file. Results are
accumulated per (object, resolution, illumination) — re-running with new
view groups adds new keys without overwriting existing ones.

Output structure
----------------
<renders_root>/<symmetry_type>/<object_id>/<image_size>/<illumination>/
    molmo_multiview.json    ← cumulative results, one key per n_views

JSON format
-----------
{
  "6": {
    "raw_output": "...",
    "points_by_image": {
      "0": [{"obj_id": 1, "x": 450.0, "y": 230.0}, ...],
      "1": [...]
    },
    "images_sent": [
      {"img_idx": 0, "filename": "IND_00_..._ROT_000.png",
       "index": 0, "azimuth": 90, "elevation": 89, "eye": [...], "R": [...], "T": [...]},
      ...
    ],
    "n_points": 4,
    "n_images_with_points": 3
  },
  "14": { ... },   ← added on a subsequent run, "6" is preserved
  ...
}

Resumability
------------
Each (object, resolution, illumination, n_views) combination is skipped if
its key already exists in the JSON. This allows:
  - Safe interruption and resumption at any point
  - Adding new view groups (e.g. 114) without re-running existing ones

Coordinate format
-----------------
Molmo2 single-image:   <points coords="RADIO  ID X Y  ID X Y ...">
Molmo2 multi-image:    <points coords="img_idx obj_id X Y ; img_idx obj_id X Y ...">
Coordinates are in 0–1000 scale, origin top-left (no Y inversion needed).

Usage — single GPU
------------------
    CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \\
        --renders-root ../data/renders \\
        --symmetry-type axis_sym \\
        --view-groups 1 6 14 26

Usage — two GPUs, split by symmetry type
-----------------------------------------
    CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \\
        --renders-root ../data/renders \\
        --symmetry-type axis_sym \\
        --gpu-id 0 --num-gpus 2 \\
        --view-groups 1 6 14 26

    CUDA_VISIBLE_DEVICES=1 python MolmoPointing/molmo_multiview_runner.py \\
        --renders-root ../data/renders \\
        --symmetry-type plane_sym \\
        --gpu-id 1 --num-gpus 2 \\
        --view-groups 1 6 14 26

Usage — add new view groups to existing results
------------------------------------------------
    CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \\
        --renders-root ../data/renders \\
        --symmetry-type axis_sym \\
        --view-groups 42 62 86 114
    (keys 1, 6, 14, 26 already in JSON → skipped automatically)
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

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_ID      = "allenai/Molmo2-8B"
OUTPUT_FILE   = "molmo_multiview.json"
PROMPT        = "Point to where the main axis of symmetry intersects the edges of the shape."

DEFAULT_VIEW_GROUPS   = [1, 6, 14, 26, 42, 62, 86, 114]
DEFAULT_IMAGE_SIZES   = [224, 448, 1136]
DEFAULT_ILLUMINATIONS = ["flat", "brighter", "darker"]

# ── Model (singleton per process) ────────────────────────────────────────────

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


def get_rot000_entries(metadata: list[dict], n_views: int) -> list[dict]:
    """First n_views viewpoints, ROT_000 only, sorted by index."""
    entries = [
        m for m in metadata
        if m["index"] < n_views and m["rotation_deg"] == 0
    ]
    return sorted(entries, key=lambda e: e["index"])


# ── Coordinate parsers ────────────────────────────────────────────────────────

def parse_single_image_coords(text: str) -> dict[str, list[dict]]:
    """
    Single-image format: <points coords="RADIO  ID X Y  ID X Y ...">
    Returns {"0": [{"obj_id": int, "x": float, "y": float}, ...]}
    """
    match = re.search(r'coords="([^"]+)"', text)
    if not match:
        return {}
    raw = [float(n) for n in match.group(1).split()]
    if len(raw) < 4:
        return {}
    raw = raw[1:]   # drop radius
    pts = []
    for i in range(0, len(raw) - 2, 3):
        pts.append({"obj_id": int(raw[i]), "x": raw[i + 1], "y": raw[i + 2]})
    return {"0": pts} if pts else {}


def parse_multiimage_coords(text: str, n_images: int) -> dict[str, list[dict]]:
    """
    Multi-image format: <points coords="img_idx obj_id X Y ; img_idx obj_id X Y ...">
    img_idx is 1-based in Molmo output; stored as 0-based string keys.
    Returns {"0": [...], "1": [...], ...}
    """
    match = re.search(r'coords="([^"]+)"', text)
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


def parse_coords(text: str, n_images: int) -> dict[str, list[dict]]:
    """Route to the correct parser based on number of images."""
    if n_images == 1:
        return parse_single_image_coords(text)
    return parse_multiimage_coords(text, n_images)


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(images: list[Image.Image]) -> str:
    """Run Molmo2 on 1..N images, return raw decoded output."""
    processor, model = get_model()

    content = [{"type": "text", "text": PROMPT}]
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

    max_new_tokens = 200 + len(images) * 50

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    return processor.tokenizer.decode(
        output_ids[0, inputs["input_ids"].size(1):],
        skip_special_tokens=True,
    )


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
) -> None:
    """
    Process one object across all (size, illumination, n_views) combinations.
    Skips any (size, illumination, n_views) already present in the JSON.
    """
    object_id = object_dir.name

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

            # Determine which view groups still need processing
            pending = [n for n in view_groups if str(n) not in results]
            if not pending:
                continue   # all groups already done for this config

            for n_views in pending:
                entries = get_rot000_entries(metadata, n_views)
                if not entries:
                    continue

                images = [
                    Image.open(render_dir / e["filename"]).convert("RGB")
                    for e in entries
                ]

                raw            = run_inference(images)
                points_by_img  = parse_coords(raw, n_images=len(images))
                n_pts          = sum(len(v) for v in points_by_img.values())

                results[str(n_views)] = {
                    "raw_output": raw,
                    "points_by_image": points_by_img,
                    "images_sent": [
                        {
                            "img_idx":    i,
                            "filename":   e["filename"],
                            "index":      e["index"],
                            "azimuth":    e["azimuth"],
                            "elevation":  e["elevation"],
                            "eye":        e["eye"],
                            "R":          e["R"],
                            "T":          e["T"],
                        }
                        for i, e in enumerate(entries)
                    ],
                    "n_points":              n_pts,
                    "n_images_with_points":  len(points_by_img),
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

    p.add_argument("--gpu-id",   type=int, default=0,
                   help="Index of this process (0-based)")
    p.add_argument("--num-gpus", type=int, default=1,
                   help="Total number of parallel GPU processes")

    p.add_argument("--view-groups", type=int, nargs="+",
                   default=DEFAULT_VIEW_GROUPS,
                   help="View-group sizes to process (e.g. 1 6 14 26)")
    p.add_argument("--sizes", type=int, nargs="+",
                   default=DEFAULT_IMAGE_SIZES,
                   help="Image sizes to process")
    p.add_argument("--lightings", type=str, nargs="+",
                   default=DEFAULT_ILLUMINATIONS,
                   choices=["flat", "darker", "brighter"])

    return p.parse_args()


def preview(args: argparse.Namespace, objects: list[Path]) -> None:
    n_configs = len(args.sizes) * len(args.lightings) * len(args.view_groups)
    print("\n========== MOLMO MULTIVIEW RUNNER ==========")
    print(f"Renders root  : {args.renders_root}")
    print(f"Symmetry type : {args.symmetry_type}")
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

    # Round-robin slice for this GPU
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
        )

    print(f"\n[GPU {args.gpu_id}] Done.")


if __name__ == "__main__":
    main()
