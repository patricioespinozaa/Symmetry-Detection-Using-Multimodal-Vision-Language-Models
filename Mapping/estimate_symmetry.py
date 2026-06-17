"""
estimate_symmetry.py
--------------------
Fits a symmetry axis (axis_sym) or plane (plane_sym) from the 3D points
produced by map_to_3d.py.

Generates FOUR estimates per n_views group, forming a 2×2 comparison grid:

                    │  sin SDE   │  con SDE
  ──────────────────┼────────────┼──────────────
  sin RANSAC        │  svd       │  svd_sde
  con RANSAC        │  ransac_svd│  ransac_svd_sde
  ──────────────────┴────────────┴──────────────

  RANSAC  — filtra outliers antes de ajustar (5 % bbox diagonal, 1000 iter)
  SVD     — ajusta eje/plano sobre el conjunto de puntos (todos o inliers)
  SDE     — evalúa la calidad del ajuste por reflexión + KDTree (requiere --objects-root)

Output
------
<renders_root>/<symmetry_type>/<object_id>/
    predicted_symmetry.json

JSON format
-----------
{
  "object_id": "...",
  "symmetry_type": "axis_sym",
  "point_mode": "independent",
  "n_views_predictions": {
    "6": {
      "svd":            {"direction": [...], "origin": [...], "n_points": 12,
                         "n_inliers": null,  "sde": null,  "accepted": null},
      "ransac_svd":     {"direction": [...], "origin": [...], "n_points": 12,
                         "n_inliers": 8,     "sde": null,  "accepted": null},
      "svd_sde":        {"direction": [...], "origin": [...], "n_points": 12,
                         "n_inliers": null,  "sde": 0.031, "accepted": true},
      "ransac_svd_sde": {"direction": [...], "origin": [...], "n_points": 12,
                         "n_inliers": 8,     "sde": 0.021, "accepted": true}
    }, ...
  }
}

Para plane_sym "direction" se reemplaza por "normal".
Cuando no se pasa --objects-root, los campos sde y accepted son null.

Usage
-----
python Mapping/estimate_symmetry.py \
    --renders-root ../data/renders \
    --objects-root ../data/objects \
    --symmetry-type axis_sym \
    --sizes 224 --lightings flat \
    --overwrite \

# Sin SDE (no requiere malla):
python Mapping/estimate_symmetry.py \
    --renders-root ../data/renders \
    --symmetry-type axis_sym
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────

INPUT_FILE   = "mapped_points_3d.json"
OUTPUT_FILE  = "predicted_symmetry.json"

DEFAULT_SIZES     = [224, 448, 1136]
DEFAULT_LIGHTINGS = ["flat", "brighter", "darker"]

POINT_MODES = ("independent", "midpoint")

OBJECTS_SUBDIR = {
    "axis_sym":  "curated_axis_sym_obj",
    "plane_sym": "curated_plane_sym_obj",
}

RANSAC_ITERS  = 1000
RANSAC_THRESH = 0.05   # inlier threshold as fraction of bbox diagonal
SDE_THRESHOLD = 0.05   # accepted if SDE ≤ this value
N_SDE_SAMPLE  = 1000   # max vertices sampled for SDE

METHODS = ("svd", "ransac_svd", "svd_sde", "ransac_svd_sde")


# ── Filename helper ───────────────────────────────────────────────────────────

def _exp_filename(base: str, experiment_id: str | None) -> str:
    """Return base unchanged, or base_EXPID.ext when experiment_id is set."""
    if not experiment_id:
        return base
    dot = base.rfind(".")
    return f"{base[:dot]}_{experiment_id}{base[dot:]}"


# ── Mesh loading ──────────────────────────────────────────────────────────────

def load_mesh_vertices(obj_path: Path) -> np.ndarray | None:
    """Load .obj and return (N, 3) vertex array. Returns None on failure."""
    try:
        import trimesh
        loaded = trimesh.load(str(obj_path), force="mesh", process=False)
        if isinstance(loaded, trimesh.Scene):
            loaded = trimesh.util.concatenate(
                [g for g in loaded.geometry.values()
                 if isinstance(g, trimesh.Trimesh)]
            )
        return np.array(loaded.vertices, dtype=np.float64)
    except Exception:
        return None


# ── SVD fitting ───────────────────────────────────────────────────────────────

def fit_axis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SVD on points → first PC = axis direction, centroid = origin."""
    centroid  = points.mean(axis=0)
    _, _, Vt  = np.linalg.svd(points - centroid, full_matrices=False)
    direction = Vt[0] / (np.linalg.norm(Vt[0]) + 1e-12)
    return direction, centroid


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SVD on points → last PC (min variance) = plane normal, centroid = origin."""
    centroid = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal   = Vt[-1] / (np.linalg.norm(Vt[-1]) + 1e-12)
    return normal, centroid


# ── RANSAC ────────────────────────────────────────────────────────────────────

def ransac_axis(points: np.ndarray, bbox_diag: float,
                n_iter: int = RANSAC_ITERS) -> list[int]:
    """
    RANSAC line fit. Samples 2 points per iteration.
    Returns inlier indices (falls back to all points if < 2 inliers found).
    """
    n = len(points)
    if n < 2:
        return list(range(n))

    threshold = RANSAC_THRESH * bbox_diag
    best: list[int] = []
    rng = np.random.default_rng(42)

    for _ in range(n_iter):
        i, j = rng.choice(n, 2, replace=False)
        d = points[j] - points[i]
        norm_d = np.linalg.norm(d)
        if norm_d < 1e-8:
            continue
        d = d / norm_d

        v = points - points[i]
        dists = np.linalg.norm(v - (v @ d)[:, None] * d, axis=1)
        inliers = np.where(dists < threshold)[0].tolist()
        if len(inliers) > len(best):
            best = inliers

    return best if len(best) >= 2 else list(range(n))


def ransac_plane(points: np.ndarray, bbox_diag: float,
                 n_iter: int = RANSAC_ITERS) -> list[int]:
    """
    RANSAC plane fit. Samples 3 points per iteration.
    Returns inlier indices (falls back to all points if < 2 inliers found).
    """
    n = len(points)
    if n < 3:
        return list(range(n))

    threshold = RANSAC_THRESH * bbox_diag
    best: list[int] = []
    rng = np.random.default_rng(42)

    for _ in range(n_iter):
        idx = rng.choice(n, 3, replace=False)
        p1, p2, p3 = points[idx]
        normal = np.cross(p2 - p1, p3 - p1)
        norm_n = np.linalg.norm(normal)
        if norm_n < 1e-8:
            continue
        normal = normal / norm_n

        dists = np.abs((points - p1) @ normal)
        inliers = np.where(dists < threshold)[0].tolist()
        if len(inliers) > len(best):
            best = inliers

    return best if len(best) >= 2 else list(range(n))


# ── SDE ───────────────────────────────────────────────────────────────────────

def sde_axis(v_sample: np.ndarray, axis_origin: np.ndarray, axis_dir: np.ndarray,
             bbox_diag: float, kdtree: KDTree) -> float:
    """
    SDE for axis symmetry.
    Reflects each sampled vertex through the axis (= 180° rotation about the axis),
    queries nearest neighbour in the mesh sample KDTree.
    Returns mean distance / bbox_diagonal.
    """
    v    = v_sample - axis_origin
    t    = v @ axis_dir
    proj = axis_origin + t[:, None] * axis_dir   # foot of perpendicular on axis
    reflected = 2.0 * proj - v_sample
    dists, _  = kdtree.query(reflected)
    return float(dists.mean() / bbox_diag)


def sde_plane(v_sample: np.ndarray, plane_origin: np.ndarray, plane_normal: np.ndarray,
              bbox_diag: float, kdtree: KDTree) -> float:
    """
    SDE for plane symmetry.
    Reflects each sampled vertex through the predicted plane,
    queries nearest neighbour in the mesh sample KDTree.
    Returns mean distance / bbox_diagonal.
    """
    d         = (v_sample - plane_origin) @ plane_normal
    reflected = v_sample - 2.0 * d[:, None] * plane_normal
    dists, _  = kdtree.query(reflected)
    return float(dists.mean() / bbox_diag)


# ── Spatial clustering ────────────────────────────────────────────────────────

def cluster_points(points: np.ndarray, bbox_diag: float,
                   threshold_fraction: float = 0.05) -> np.ndarray:
    """
    Greedy centroid-based spatial clustering.

    Points within threshold_fraction * bbox_diag of an existing centroid are
    merged into it (centroid updated incrementally). Reduces spatial redundancy
    when multiple views observe the same surface region.

    Returns an array of cluster centroids (one per group), preserving order of
    first encounter. If threshold is too small or fewer than 2 points exist,
    returns the original array unchanged.
    """
    threshold = threshold_fraction * bbox_diag
    if threshold < 1e-8 or len(points) < 2:
        return points

    centroids: list[np.ndarray] = []
    counts: list[int] = []

    for pt in points:
        placed = False
        for k, c in enumerate(centroids):
            if np.linalg.norm(pt - c) < threshold:
                counts[k] += 1
                centroids[k] = c + (pt - c) / counts[k]   # incremental mean
                placed = True
                break
        if not placed:
            centroids.append(pt.copy())
            counts.append(1)

    return np.array(centroids, dtype=np.float64)


# ── Point collection ──────────────────────────────────────────────────────────

def collect_hit_points(
    mapped_json: dict,
    n_views_key: str,
    point_mode:  str = "independent",
) -> np.ndarray | None:
    """
    Extract 3D points from a mapped_points_3d entry for a given n_views key.

    point_mode="independent":
        For each image, both obj_id=1 and obj_id=2 must hit.
        Images where only one point hit are discarded.
        Both points enter the cloud as separate entries (not averaged).
        Use with prompts that return points directly on the axis/plane.

    point_mode="midpoint":
        For each image, obj_id=1 and obj_id=2 are replaced by their 3D midpoint.
        Images where either point missed are discarded.
        Use with bilateral symmetric pair prompts.

    Returns (N, 3) float64 array, or None if fewer than 2 usable points.
    """
    group = mapped_json.get("n_views_results", {}).get(n_views_key)
    if group is None:
        return None

    raw_points = group.get("points_3d", [])

    if point_mode == "independent":
        by_img: dict[int, dict[int, np.ndarray]] = {}
        for p in raw_points:
            if not p["hit"] or p["point_3d"] is None:
                continue
            by_img.setdefault(p["img_idx"], {})[p["obj_id"]] = np.array(
                p["point_3d"], dtype=np.float64
            )
        pts = []
        for img_pts in by_img.values():
            if 1 in img_pts and 2 in img_pts:
                pts.extend([img_pts[1], img_pts[2]])
        if len(pts) < 2:
            return None
        return np.array(pts, dtype=np.float64)

    elif point_mode == "midpoint":
        by_img = {}
        for p in raw_points:
            if not p["hit"] or p["point_3d"] is None:
                continue
            by_img.setdefault(p["img_idx"], {})[p["obj_id"]] = np.array(
                p["point_3d"], dtype=np.float64
            )
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


# ── Entry builder ─────────────────────────────────────────────────────────────

def _make_entry(
    symmetry_type: str,
    vec:           np.ndarray,   # direction (axis) or normal (plane)
    origin:        np.ndarray,
    n_points:      int,
    n_inliers:     int | None,
    sde_val:       float | None,
) -> dict:
    """Build a single estimate dict for one method."""
    vec_key  = "direction" if symmetry_type == "axis_sym" else "normal"
    accepted = bool(sde_val <= SDE_THRESHOLD) if sde_val is not None else None
    return {
        vec_key:     vec.tolist(),
        "origin":    origin.tolist(),
        "n_points":  n_points,
        "n_inliers": n_inliers,
        "sde":       round(sde_val, 6) if sde_val is not None else None,
        "accepted":  accepted,
    }


# ── Per-object estimation ─────────────────────────────────────────────────────

def process_object(
    object_dir:    Path,
    symmetry_type: str,
    sizes:         list[int],
    lightings:     list[str],
    overwrite:     bool = False,
    experiment_id: str | None = None,
    point_mode:    str = "independent",
    objects_dir:   Path | None = None,
    clustering:    bool = False,
) -> None:
    """
    Generates 4 estimates per n_views group (2×2 grid):
      svd            — SVD on all points
      ransac_svd     — RANSAC inlier selection + SVD
      svd_sde        — SVD on all points + SDE evaluation
      ransac_svd_sde — RANSAC + SVD + SDE (full pipeline)

    SDE requires objects_dir (mesh). If not provided, sde=null for all methods.

    When clustering=True, applies greedy centroid-based spatial clustering
    (threshold = 5% bbox diagonal) before fitting. Output is written to a
    separate file with "_cluster" appended to the experiment suffix, allowing
    direct comparison via --experiment-id.
    """
    # Derive output experiment suffix: append "_cluster" when clustering is on
    exp_out     = ((experiment_id + "_cluster") if experiment_id else "cluster") if clustering else experiment_id
    input_file  = _exp_filename(INPUT_FILE,  experiment_id)
    output_file = _exp_filename(OUTPUT_FILE, exp_out)

    output_path = object_dir / output_file
    if output_path.exists() and not overwrite:
        return

    # ── Load mesh for SDE (optional) ──────────────────────────────────────────
    v_sample       = None
    kdtree         = None
    bbox_diag_mesh = None

    if objects_dir is not None:
        obj_path = objects_dir / f"{object_dir.name}.obj"
        vertices = load_mesh_vertices(obj_path)
        if vertices is not None:
            bbox_diag_mesh = float(
                np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
            )
            n_sde    = min(N_SDE_SAMPLE, len(vertices))
            sde_idx  = np.random.default_rng(0).choice(len(vertices), n_sde, replace=False)
            v_sample = vertices[sde_idx]
            kdtree   = KDTree(v_sample)

    # ── Collect 3D points across all (size, lighting) configs ─────────────────
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

    # ── Build 4 estimates per n_views group ───────────────────────────────────
    n_views_predictions: dict = {}
    has_sde = kdtree is not None and bbox_diag_mesh is not None and bbox_diag_mesh > 1e-6

    for n_views_key, arrays in points_per_group.items():
        all_points = np.concatenate(arrays, axis=0)
        if len(all_points) < 2:
            continue

        # Bbox diagonal of the point cloud (RANSAC threshold and clustering)
        bbox_diag_pts = float(
            np.linalg.norm(all_points.max(axis=0) - all_points.min(axis=0))
        )
        if bbox_diag_pts < 1e-6:
            bbox_diag_pts = 1.0

        if clustering:
            all_points = cluster_points(all_points, bbox_diag_pts)
            if len(all_points) < 2:
                continue
            # Recompute bbox after clustering (centroids may be closer together)
            bbox_diag_pts = float(
                np.linalg.norm(all_points.max(axis=0) - all_points.min(axis=0))
            )
            if bbox_diag_pts < 1e-6:
                bbox_diag_pts = 1.0

        n_pts = len(all_points)

        # ── RANSAC → inliers ──────────────────────────────────────────────────
        if symmetry_type == "axis_sym":
            inlier_idx = ransac_axis(all_points, bbox_diag_pts)
        else:
            inlier_idx = ransac_plane(all_points, bbox_diag_pts)
        inliers    = all_points[inlier_idx]
        n_inliers  = len(inliers)

        # ── SVD on all points ─────────────────────────────────────────────────
        if symmetry_type == "axis_sym":
            vec_svd,    origin_svd    = fit_axis(all_points)
            vec_ransac, origin_ransac = fit_axis(inliers)
        else:
            vec_svd,    origin_svd    = fit_plane(all_points)
            vec_ransac, origin_ransac = fit_plane(inliers)

        # ── SDE for both estimates ────────────────────────────────────────────
        sde_svd = sde_ransac = None
        if has_sde:
            if symmetry_type == "axis_sym":
                sde_svd    = sde_axis(v_sample, origin_svd,    vec_svd,    bbox_diag_mesh, kdtree)
                sde_ransac = sde_axis(v_sample, origin_ransac, vec_ransac, bbox_diag_mesh, kdtree)
            else:
                sde_svd    = sde_plane(v_sample, origin_svd,    vec_svd,    bbox_diag_mesh, kdtree)
                sde_ransac = sde_plane(v_sample, origin_ransac, vec_ransac, bbox_diag_mesh, kdtree)

        # ── 4 estimates ───────────────────────────────────────────────────────
        n_views_predictions[n_views_key] = {
            "n_points_raw":   len(np.concatenate(arrays, axis=0)),
            "n_points_fit":   n_pts,
            "svd":            _make_entry(symmetry_type, vec_svd,    origin_svd,    n_pts, None,      None),
            "ransac_svd":     _make_entry(symmetry_type, vec_ransac, origin_ransac, n_pts, n_inliers, None),
            "svd_sde":        _make_entry(symmetry_type, vec_svd,    origin_svd,    n_pts, None,      sde_svd),
            "ransac_svd_sde": _make_entry(symmetry_type, vec_ransac, origin_ransac, n_pts, n_inliers, sde_ransac),
        }

    if not n_views_predictions:
        return

    output = {
        "object_id":           object_dir.name,
        "symmetry_type":       symmetry_type,
        "point_mode":          point_mode,
        "clustering":          clustering,
        "n_views_predictions": n_views_predictions,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit symmetry axis/plane — RANSAC + SVD + SDE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--renders-root",  required=True,
                   help="Root folder of renders (output of data_render.py)")
    p.add_argument("--objects-root",  default=None,
                   help="Root folder containing curated_axis_sym_obj / curated_plane_sym_obj. "
                        "Required for SDE scoring. If omitted, SDE is skipped (sde=null).")
    p.add_argument("--symmetry-type", required=True, choices=["axis_sym", "plane_sym"])
    p.add_argument("--sizes",     type=int, nargs="+", default=DEFAULT_SIZES)
    p.add_argument("--lightings", type=str, nargs="+", default=DEFAULT_LIGHTINGS,
                   choices=["flat", "darker", "brighter"])
    p.add_argument("--gpu-id",   type=int, default=0)
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing predicted_symmetry.json files")
    p.add_argument("--experiment-id", default=None,
                   help="Reads mapped_points_3d_<ID>.json, writes predicted_symmetry_<ID>.json.")
    p.add_argument("--max-objects", type=int, default=None,
                   help="Limit to the first N objects (sorted order).")
    p.add_argument("--point-mode", default="independent", choices=POINT_MODES,
                   help="independent = each 3D point goes to SVD directly. "
                        "midpoint = bilateral pairs (obj_id=1, obj_id=2) replaced by 3D midpoint.")
    p.add_argument("--clustering", action="store_true",
                   help="Apply greedy centroid-based spatial clustering (threshold=5%% bbox diagonal) "
                        "before fitting. Output is written to predicted_symmetry[_EXP]_cluster.json, "
                        "enabling direct comparison with the non-clustered run via --experiment-id.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    symmetry_dir = Path(args.renders_root) / args.symmetry_type
    if not symmetry_dir.exists():
        print(f"[error] Not found: {symmetry_dir}")
        sys.exit(1)

    objects_dir = None
    if args.objects_root:
        objects_dir = Path(args.objects_root) / OBJECTS_SUBDIR[args.symmetry_type]

    all_objects = sorted(d for d in symmetry_dir.iterdir() if d.is_dir())
    if args.max_objects:
        all_objects = all_objects[:args.max_objects]
    objects = all_objects[args.gpu_id :: args.num_gpus]

    output_file = _exp_filename(OUTPUT_FILE, args.experiment_id)
    print(f"\nFitting {args.symmetry_type} symmetry for {len(objects)} objects...")
    print(f"Point mode    : {args.point_mode}")
    print(f"Clustering    : {'enabled (threshold=' + str(int(RANSAC_THRESH*100)) + '% bbox diag)' if args.clustering else 'disabled'}")
    print(f"RANSAC        : {RANSAC_ITERS} iters, threshold={RANSAC_THRESH*100:.0f}% bbox diag")
    print(f"SDE scoring   : {'enabled  (threshold=' + str(SDE_THRESHOLD) + ')' if objects_dir else 'disabled — pass --objects-root to enable'}")
    if args.experiment_id:
        print(f"Experiment ID : {args.experiment_id}  →  {output_file}")
    print(f"({'overwrite' if args.overwrite else 'skip existing'})")

    for obj_dir in tqdm(objects, unit="obj", dynamic_ncols=True):
        process_object(
            object_dir    = obj_dir,
            symmetry_type = args.symmetry_type,
            sizes         = args.sizes,
            lightings     = args.lightings,
            overwrite     = args.overwrite,
            experiment_id = args.experiment_id,
            point_mode    = args.point_mode,
            objects_dir   = objects_dir,
            clustering    = args.clustering,
        )

    print("Done.")


if __name__ == "__main__":
    main()
