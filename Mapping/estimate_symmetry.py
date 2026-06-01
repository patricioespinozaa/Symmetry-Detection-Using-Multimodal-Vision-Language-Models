"""
estimate_symmetry.py
--------------------
Fits a symmetry axis (axis_sym) or plane (plane_sym) from the 3D points
produced by map_to_3d.py, and saves the predicted symmetry to
predicted_symmetry.json alongside the renders.

Fitting method
--------------
axis_sym:
    SVD on the centered 3D hit points → principal component = axis direction.
    Origin = centroid of the hit points, projected onto the true axis origin
    along the predicted direction (keeps origin near 0,0,0 for comparison).

plane_sym:
    SVD on the centered 3D hit points → last component (smallest variance)
    = plane normal. Origin = centroid projected to the plane through (0,0,0).

For both types, all hit points across all n_views groups and all
(size, illumination) configs are pooled together per object before fitting,
unless --per-config is passed (fits independently per size×illumination).

Output
------
<renders_root>/<symmetry_type>/<object_id>/
    predicted_symmetry.json    ← one file per object (pooled across configs)

JSON format — axis_sym
-----------------------
{
  "object_id": "...",
  "symmetry_type": "axis_sym",
  "n_views_predictions": {
    "1":  {"direction": [dx, dy, dz], "origin": [ox, oy, oz], "n_points": 2},
    "6":  {"direction": [...], "origin": [...], "n_points": 6},
    ...
  }
}

JSON format — plane_sym
------------------------
{
  "object_id": "...",
  "symmetry_type": "plane_sym",
  "n_views_predictions": {
    "1":  {"normal": [nx, ny, nz], "origin": [ox, oy, oz], "n_points": 2},
    ...
  }
}

Usage
-----
    python Mapping/estimate_symmetry.py \\
        --renders-root ../data/renders \\
        --symmetry-type axis_sym \\
        --sizes 224 \\
        --lightings flat

    python Mapping/estimate_symmetry.py \\
        --renders-root ../data/renders \\
        --symmetry-type plane_sym
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────

INPUT_FILE     = "mapped_points_3d.json"
OUTPUT_FILE    = "predicted_symmetry.json"

DEFAULT_SIZES       = [224, 448, 1136]
DEFAULT_LIGHTINGS   = ["flat", "brighter", "darker"]


def _exp_filename(base: str, experiment_id: str | None) -> str:
    """Return base unchanged, or base_EXPID.ext when experiment_id is set."""
    if not experiment_id:
        return base
    dot = base.rfind(".")
    return f"{base[:dot]}_{experiment_id}{base[dot:]}"


# ── Fitting ───────────────────────────────────────────────────────────────────

def fit_axis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a 3D axis (line) through a set of points using SVD (PCA).

    Returns:
        direction: (3,) unit vector along the axis
        origin:    (3,) centroid of the points (closest point on axis to origin)
    """
    centroid  = points.mean(axis=0)
    centered  = points - centroid
    _, _, Vt  = np.linalg.svd(centered, full_matrices=False)
    direction = Vt[0]                          # first principal component
    direction /= np.linalg.norm(direction)     # ensure unit length
    return direction, centroid


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a 3D plane through a set of points using SVD (PCA).

    Returns:
        normal: (3,) unit normal to the plane
        origin: (3,) centroid of the points (point on the plane)
    """
    centroid  = points.mean(axis=0)
    centered  = points - centroid
    _, _, Vt  = np.linalg.svd(centered, full_matrices=False)
    normal    = Vt[-1]                         # last principal component (min variance)
    normal   /= np.linalg.norm(normal)
    return normal, centroid


POINT_MODES = ("independent", "midpoint")


def collect_hit_points(
    mapped_json: dict,
    n_views_key: str,
    point_mode:  str = "independent",
) -> np.ndarray | None:
    """
    Extract 3D points from a mapped_points_3d entry for a given n_views key.

    point_mode="independent" (default):
        Every hit point is added to the cloud as-is.
        Use with prompts that return points directly on the axis/plane.

    point_mode="midpoint":
        For each image, obj_id=1 and obj_id=2 are paired and replaced by their
        3D midpoint. Images where either point missed are discarded.
        Use with prompts that return bilateral symmetric pairs — the midpoint
        of each pair lies on the axis/plane instead of the individual points.

    Returns (N, 3) float64 array, or None if fewer than 2 usable points.
    """
    group = mapped_json.get("n_views_results", {}).get(n_views_key)
    if group is None:
        return None

    raw_points = group.get("points_3d", [])

    if point_mode == "independent":
        pts = [
            p["point_3d"]
            for p in raw_points
            if p["hit"] and p["point_3d"] is not None
        ]
        if len(pts) < 2:
            return None
        return np.array(pts, dtype=np.float64)

    elif point_mode == "midpoint":
        # Group hits by img_idx, then by obj_id
        by_img: dict[int, dict[int, np.ndarray]] = {}
        for p in raw_points:
            if not p["hit"] or p["point_3d"] is None:
                continue
            img = p["img_idx"]
            oid = p["obj_id"]
            by_img.setdefault(img, {})[oid] = np.array(p["point_3d"], dtype=np.float64)

        midpoints = [
            (pts[1] + pts[2]) / 2.0
            for pts in by_img.values()
            if 1 in pts and 2 in pts
        ]
        if len(midpoints) < 2:
            return None
        return np.array(midpoints, dtype=np.float64)

    else:
        raise ValueError(f"Unknown point_mode: {point_mode!r}. Choose from {POINT_MODES}.")


# ── Per-object estimation ─────────────────────────────────────────────────────

def process_object(
    object_dir:    Path,
    symmetry_type: str,
    sizes:         list[int],
    lightings:     list[str],
    overwrite:     bool = False,
    experiment_id: str | None = None,
    point_mode:    str = "independent",
) -> None:
    """
    Pool 3D hit points across all (size, illumination) configs and all n_views groups,
    fit the symmetry element per n_views group, and save the result.
    When experiment_id is set, reads mapped_points_3d_<ID>.json and writes
    predicted_symmetry_<ID>.json instead of the default filenames.
    Skips objects where the output already exists unless --overwrite.
    point_mode controls how Molmo's pairs are converted to the SVD point cloud
    (see collect_hit_points for details).
    """
    input_file  = _exp_filename(INPUT_FILE,  experiment_id)
    output_file = _exp_filename(OUTPUT_FILE, experiment_id)

    output_path = object_dir / output_file
    if output_path.exists() and not overwrite:
        return

    # Collect all mapped JSON files for this object
    # Structure: {n_views_key: [array_of_points, ...]}
    points_per_group: dict[str, list[np.ndarray]] = defaultdict(list)

    for size in sizes:
        for lighting in lightings:
            mapped_path = object_dir / str(size) / lighting / input_file
            if not mapped_path.exists():
                continue

            with open(mapped_path, encoding="utf-8") as f:
                mapped = json.load(f)

            for n_views_key in mapped.get("n_views_results", {}).keys():
                pts = collect_hit_points(mapped, n_views_key, point_mode=point_mode)
                if pts is not None:
                    points_per_group[n_views_key].append(pts)

    if not points_per_group:
        return

    n_views_predictions = {}

    for n_views_key, arrays in points_per_group.items():
        # Pool points from all configs for this n_views group
        all_points = np.concatenate(arrays, axis=0)   # (N, 3)

        if len(all_points) < 2:
            continue

        if symmetry_type == "axis_sym":
            direction, origin = fit_axis(all_points)
            n_views_predictions[n_views_key] = {
                "direction": direction.tolist(),
                "origin":    origin.tolist(),
                "n_points":  len(all_points),
            }
        else:  # plane_sym
            normal, origin = fit_plane(all_points)
            n_views_predictions[n_views_key] = {
                "normal":   normal.tolist(),
                "origin":   origin.tolist(),
                "n_points": len(all_points),
            }

    if not n_views_predictions:
        return

    output = {
        "object_id":           object_dir.name,
        "symmetry_type":       symmetry_type,
        "point_mode":          point_mode,
        "n_views_predictions": n_views_predictions,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)



# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit symmetry axis/plane from 3D hit points.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--renders-root",  required=True)
    p.add_argument("--symmetry-type", required=True,
                   choices=["axis_sym", "plane_sym"])
    p.add_argument("--sizes",     type=int, nargs="+", default=DEFAULT_SIZES)
    p.add_argument("--lightings", type=str, nargs="+", default=DEFAULT_LIGHTINGS,
                   choices=["flat", "darker", "brighter"])
    p.add_argument("--gpu-id",   type=int, default=0)
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing predicted_symmetry.json files")
    p.add_argument("--experiment-id", default=None,
                   help=(
                       "Experiment identifier. Reads mapped_points_3d_<ID>.json and "
                       "writes predicted_symmetry_<ID>.json. Must match the --experiment-id "
                       "used in map_to_3d.py."
                   ))
    p.add_argument("--max-objects", type=int, default=None,
                   help="Limit to the first N objects (sorted order).")
    p.add_argument("--point-mode", default="independent", choices=POINT_MODES,
                   help=(
                       "How Molmo's point pairs are converted to the SVD cloud. "
                       "independent = each 3D point enters the cloud as-is (use with "
                       "prompts that return points directly on the axis/plane). "
                       "midpoint = obj_id=1 and obj_id=2 per image are replaced by their "
                       "3D midpoint (use with bilateral symmetric pair prompts)."
                   ))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    symmetry_dir = Path(args.renders_root) / args.symmetry_type
    if not symmetry_dir.exists():
        print(f"[error] Not found: {symmetry_dir}")
        sys.exit(1)

    all_objects = sorted(d for d in symmetry_dir.iterdir() if d.is_dir())
    if args.max_objects:
        all_objects = all_objects[:args.max_objects]
    objects = all_objects[args.gpu_id :: args.num_gpus]

    output_file = _exp_filename(OUTPUT_FILE, args.experiment_id)
    print(f"\nFitting {args.symmetry_type} symmetry for {len(objects)} objects...")
    print(f"Point mode    : {args.point_mode}")
    if args.experiment_id:
        print(f"Experiment ID : {args.experiment_id}  →  {output_file}")
    if args.overwrite:
        print(f"(--overwrite: existing {output_file} will be replaced)")
    else:
        print(f"(Existing {output_file} skipped — use --overwrite to replace)")

    for obj_dir in tqdm(objects, unit="obj", dynamic_ncols=True):
        process_object(
            object_dir    = obj_dir,
            symmetry_type = args.symmetry_type,
            sizes         = args.sizes,
            lightings     = args.lightings,
            overwrite     = args.overwrite,
            experiment_id = args.experiment_id,
            point_mode    = args.point_mode,
        )

    print("Done.")


if __name__ == "__main__":
    main()