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


def collect_hit_points(
    mapped_json: dict,
    n_views_key: str,
) -> np.ndarray | None:
    """
    Extract all 3D hit points from a mapped_points_3d entry for a given n_views key.
    Returns (N, 3) array or None if no hits.
    """
    group = mapped_json.get("n_views_results", {}).get(n_views_key)
    if group is None:
        return None

    pts = [
        p["point_3d"]
        for p in group.get("points_3d", [])
        if p["hit"] and p["point_3d"] is not None
    ]

    if len(pts) < 2:
        return None

    return np.array(pts, dtype=np.float64)


# ── Per-object estimation ─────────────────────────────────────────────────────

def process_object(
    object_dir:    Path,
    symmetry_type: str,
    sizes:         list[int],
    lightings:     list[str],
) -> None:
    """
    Pool 3D hit points across all (size, illumination) configs and all n_views groups,
    fit the symmetry element per n_views group, and save to predicted_symmetry.json.
    Skips objects where the output already exists.
    """
    output_path = object_dir / OUTPUT_FILE
    if output_path.exists():
        return

    # Collect all mapped JSON files for this object
    # Structure: {n_views_key: [array_of_points, ...]}
    points_per_group: dict[str, list[np.ndarray]] = defaultdict(list)

    for size in sizes:
        for lighting in lightings:
            mapped_path = object_dir / str(size) / lighting / INPUT_FILE
            if not mapped_path.exists():
                continue

            with open(mapped_path, encoding="utf-8") as f:
                mapped = json.load(f)

            for n_views_key in mapped.get("n_views_results", {}).keys():
                pts = collect_hit_points(mapped, n_views_key)
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
    return p.parse_args()


def main() -> None:
    args = parse_args()

    symmetry_dir = Path(args.renders_root) / args.symmetry_type
    if not symmetry_dir.exists():
        print(f"[error] Not found: {symmetry_dir}")
        sys.exit(1)

    all_objects = sorted(d for d in symmetry_dir.iterdir() if d.is_dir())
    objects     = all_objects[args.gpu_id :: args.num_gpus]

    print(f"\nFitting {args.symmetry_type} symmetry for {len(objects)} objects...")

    for obj_dir in tqdm(objects, unit="obj", dynamic_ncols=True):
        process_object(
            object_dir    = obj_dir,
            symmetry_type = args.symmetry_type,
            sizes         = args.sizes,
            lightings     = args.lightings,
        )

    print("Done.")


if __name__ == "__main__":
    main()
