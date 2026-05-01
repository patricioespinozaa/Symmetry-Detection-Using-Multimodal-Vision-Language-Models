"""
molmo_batch.py
--------------
Processes a single object across all requested view-groups, image sizes,
and illumination modes.

Directory contract (mirrors export_fibonacci_views.py output):

    <renders_root>/<symmetry_type>/<object_id>/<image_size>/<illumination>/
        IND_00_AZ_..._ROT_000.png
        ...
        metadata_all.json

Molmo output is written alongside the renders:

    <renders_root>/<symmetry_type>/<object_id>/<image_size>/<illumination>/molmo/
        IND_00_AZ_..._ROT_000.json      ← coords + raw output
        IND_00_AZ_..._ROT_000_vis.png   ← annotated image

A sentinel file flags completion so molmo_runner.py can skip finished objects:

    <renders_root>/<symmetry_type>/<object_id>/molmo_done.txt

View-group filtering
--------------------
Each folder contains 456 images (114 views × 4 rotations). A group of N
views means the first N viewpoint indices (IND_00 … IND_{N-1}), each with
their 4 rotations — giving N × 4 images. The mapping is read from
metadata_all.json (field "index").

Supported groups: 6, 14, 26, 42, 62, 86, 114
"""

from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from molmo_inference import SYMMETRY_PROMPT, run_inference
from molmo_visualize import save_annotated_image

# ── Constants ─────────────────────────────────────────────────────────────────

VIEW_GROUPS   = [6, 14, 26, 42, 62, 86, 114]
IMAGE_SIZES   = [224, 448, 1024]
ILLUMINATIONS = ["flat", "darker", "brighter"]

DONE_SENTINEL = "molmo_done.txt"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_metadata(folder: Path) -> list[dict]:
    meta_path = folder / "metadata_all.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata_all.json not found in {folder}")
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def _images_for_group(metadata: list[dict], n_views: int) -> list[dict]:
    """Return metadata entries whose viewpoint index < n_views."""
    return [m for m in metadata if m["index"] < n_views]


def _write_result(output_dir: Path, stem: str, result: dict, meta: dict) -> None:
    """Write the JSON result file for one image."""
    payload = {
        "filename":    meta["filename"],
        "index":       meta["index"],
        "azimuth":     meta["azimuth"],
        "elevation":   meta["elevation"],
        "rotation_deg": meta["rotation_deg"],
        "eye":         meta["eye"],
        "R":           meta["R"],
        "T":           meta["T"],
        "prompt":      SYMMETRY_PROMPT,
        "raw_output":  result["raw_output"],
        "points":      result["points"],
        "success":     result["success"],
    }
    json_path = output_dir / f"{stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ── Main entry ────────────────────────────────────────────────────────────────

def process_object(
    object_dir: Path,
    symmetry_type: str,
    sizes: list[int]  = IMAGE_SIZES,
    lightings: list[str] = ILLUMINATIONS,
    view_groups: list[int] = VIEW_GROUPS,
    save_vis: bool = True,
) -> None:
    """
    Run Molmo inference for one object over all requested configurations.

    Args:
        object_dir:    Path to <renders_root>/<symmetry_type>/<object_id>/
        symmetry_type: "axis_sym" or "plane_sym" (used only for logging).
        sizes:         List of image sizes to process.
        lightings:     List of illumination modes to process.
        view_groups:   List of view-group sizes (subset of indices to run).
        save_vis:      Whether to save annotated PNG images.
    """
    object_id = object_dir.name
    sentinel  = object_dir / DONE_SENTINEL

    # ── Skip already-completed objects ────────────────────────────────────────
    if sentinel.exists():
        print(f"[skip] {object_id} already done.")
        return

    # ── Collect all (image_path, meta, output_dir) tuples ─────────────────────
    tasks: list[tuple[Path, dict, Path]] = []

    for size in sizes:
        for lighting in lightings:
            render_dir = object_dir / str(size) / lighting

            if not render_dir.exists():
                print(f"[warn] Missing render folder: {render_dir}")
                continue

            try:
                metadata = _load_metadata(render_dir)
            except FileNotFoundError as e:
                print(f"[warn] {e}")
                continue

            for n_views in view_groups:
                group_entries = _images_for_group(metadata, n_views)
                output_dir = render_dir / "molmo" / f"views_{n_views:03d}"
                output_dir.mkdir(parents=True, exist_ok=True)

                for entry in group_entries:
                    img_path = render_dir / entry["filename"]
                    if not img_path.exists():
                        print(f"[warn] Image not found: {img_path}")
                        continue
                    tasks.append((img_path, entry, output_dir))

    if not tasks:
        print(f"[warn] No tasks found for {object_id}. Skipping.")
        return

    # ── Run inference ──────────────────────────────────────────────────────────
    n_success = 0
    n_no_points = 0

    desc = f"{object_id}"
    for img_path, meta, output_dir in tqdm(tasks, desc=desc, unit="img", leave=False):
        stem = img_path.stem  # e.g. IND_03_AZ_120_EL_+45_ROT_090

        result = run_inference(img_path)

        # Write JSON
        _write_result(output_dir, stem, result, meta)

        # Write annotated PNG
        if save_vis:
            vis_path = output_dir / f"{stem}_vis.png"
            save_annotated_image(img_path, result["points"], vis_path)

        if result["success"]:
            n_success += 1
        else:
            n_no_points += 1

    # ── Mark object as done ───────────────────────────────────────────────────
    total = len(tasks)
    summary = {
        "object_id":   object_id,
        "total_images": total,
        "n_success":   n_success,
        "n_no_points": n_no_points,
    }
    with open(sentinel, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[done] {object_id} | {total} imgs | "
        f"{n_success} with points | {n_no_points} empty"
    )
