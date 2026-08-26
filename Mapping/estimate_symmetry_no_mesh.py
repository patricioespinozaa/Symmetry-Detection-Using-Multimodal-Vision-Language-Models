"""
estimate_symmetry_no_mesh.py
-----------------------------
Mesh-free symmetry estimator (docs/pipeline_sin_malla.md): recovers the
symmetry axis/plane by triangulating camera rays across views, WITHOUT ever
ray-casting against the object's mesh. Runs in parallel to the existing
with-mesh pipeline (Mapping/map_to_3d.py + Mapping/estimate_symmetry.py) --
does not replace it, does not modify its output files.

Prototype validated in test_pipeline_sin_malla.ipynb; migration plan and
rationale for every design choice below in
docs/implementacion_pipeline_sin_malla.md (S2.2/S3). Geometry primitives
(ray_dir_for_point, interpretation_plane_normal, triangulate_line,
widest_pair) live in pipeline_common/triangulation.py.

Input
-----
Reads molmo_multiview_<EXP>.json DIRECTLY from
<renders_root>/<symmetry_type>/<object_id>/<size>/<lighting>/ -- the same
file MolmoPointing/molmo_multiview_runner.py already writes. Does NOT read
mapped_points_3d.json (that file, and the mesh it depends on, belong to the
with-mesh pipeline only).

Output
------
predicted_symmetry_<EXP>.json, one per object (pools all --sizes x
--lightings, same convention as Mapping/estimate_symmetry.py), under a new
method key "triangulation":

    "triangulation": {"direction"/"normal": [...], "origin": [...],
                       "n_points": <int>, "n_views_used": <int>,
                       "n_inliers": null, "sde": null, "accepted": null}

For plane_sym with --max-planes > 1, the entry becomes a list under a
"planes" key instead of a single normal/origin (see docs/implementacion_pipeline_sin_malla.md
S3, Fase 2) -- consumed by Mapping/evaluate.py via the "triangulation_multiplane"
method (evaluate_plane_multi_from_pred / evaluate_plane_multiset), including
SDE_ref/F1_ref when --with-reference-metrics is passed. --max-planes 1
(the default) keeps the single-plane "triangulation" schema fully compatible
with the with-mesh methods.

Usage
-----
    python Mapping/estimate_symmetry_no_mesh.py \\
        --renders-root ../data/renders \\
        --symmetry-type axis_sym \\
        --sizes 224 --lightings flat \\
        --experiment-id axis_v02_nomesh \\
        --overwrite

    python Mapping/estimate_symmetry_no_mesh.py \\
        --renders-root ../data/renders \\
        --symmetry-type plane_sym \\
        --sizes 224 --lightings flat \\
        --experiment-id plane_v02_nomesh \\
        --overwrite
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_common.naming import exp_filename
from pipeline_common.triangulation import (
    get_point_by_obj_id,
    interpretation_plane_normal,
    ray_dir_for_point,
    triangulate_line,
    view_forward_direction,
    widest_pair,
)

# ── Constants ─────────────────────────────────────────────────────────────────

MOLMO_JSON    = "molmo_multiview.json"
MANIFEST_FILE = "manifest.json"
OUTPUT_FILE   = "predicted_symmetry.json"
METHOD_KEY    = "triangulation"

DEFAULT_SIZES       = [224, 448, 1136]
DEFAULT_LIGHTINGS   = ["flat", "brighter", "darker"]
DEFAULT_FOV         = 60.0

EDGE_ON_THRESH_DEFAULT     = 0.5    # |cos(angle)| below this = view is "edge-on" to a plane candidate
DUP_ANGLE_THRESH_DEFAULT   = 15.0   # degrees; candidate planes closer than this are treated as duplicates


# ── Axis: triangulate a single line from all interpretation planes ────────────

def estimate_axis_no_mesh(
    points_by_image: dict, images_sent: list[dict],
    fov_deg: float, image_size: int,
) -> dict:
    """
    One interpretation plane per view (from widest_pair(pts), NOT a fixed
    obj_id 1/2 lookup -- see pipeline_common.triangulation.widest_pair for
    why this is a strict generalization, not a behavior change, for prompts
    that already return exactly 2 points per view). Intersecting >=2 such
    planes (docs/pipeline_sin_malla.md S3.1) gives the 3D axis directly.

    Raises ValueError if fewer than 2 views yield a usable interpretation
    plane.
    """
    centers, normals = [], []
    for img_idx_str, pts in points_by_image.items():
        pair = widest_pair(pts)
        if pair is None:
            continue
        p_a, p_b = pair
        cam = images_sent[int(img_idx_str)]
        C, d_a = ray_dir_for_point(p_a["x"], p_a["y"], cam["R"], cam["T"], fov_deg, image_size)
        _, d_b = ray_dir_for_point(p_b["x"], p_b["y"], cam["R"], cam["T"], fov_deg, image_size)
        n = interpretation_plane_normal(d_a, d_b)
        if n is None:
            continue
        centers.append(C)
        normals.append(n)

    if len(normals) < 2:
        raise ValueError(f"need >=2 valid views, got {len(normals)}")

    point, direction = triangulate_line(centers, normals)
    return {"direction": direction.tolist(), "origin": point.tolist(), "n_views_used": len(normals)}


# ── Plane: single-best candidate (docs/pipeline_sin_malla.md S3.2) ────────────
# Uses get_point_by_obj_id (fixed roles), NOT widest_pair -- bilateral plane
# prompts give obj_id 1/2 a real geometric meaning (left/right mirror pair),
# unlike axis' top/bottom framing. See docs/implementacion_pipeline_sin_malla.md
# S2.1 for why this is NOT generalized the same way as the axis case.

def _line_from_view_pair(
    view_i: int, view_j: int, points_by_image: dict, images_sent: list[dict],
    fov_deg: float, image_size: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    centers, normals = [], []
    for idx in (view_i, view_j):
        pts = points_by_image.get(str(idx), [])
        p1 = get_point_by_obj_id(pts, 1)
        p2 = get_point_by_obj_id(pts, 2)
        if p1 is None or p2 is None:
            return None
        cam = images_sent[idx]
        C, d1 = ray_dir_for_point(p1["x"], p1["y"], cam["R"], cam["T"], fov_deg, image_size)
        _, d2 = ray_dir_for_point(p2["x"], p2["y"], cam["R"], cam["T"], fov_deg, image_size)
        n = interpretation_plane_normal(d1, d2)
        if n is None:
            return None
        centers.append(C)
        normals.append(n)
    return triangulate_line(centers, normals)


def estimate_plane_no_mesh(
    points_by_image: dict, images_sent: list[dict],
    fov_deg: float, image_size: int,
    edge_on_thresh: float = EDGE_ON_THRESH_DEFAULT,
) -> dict:
    """
    Iterative scheme (docs/pipeline_sin_malla.md S3.2): triangulate a
    candidate line per pair of views, cross-product pairs of those lines
    (from independent view-pairs) into candidate normals, score each by how
    "edge-on" the views are to it, keep the best-scored candidate's edge-on
    views and refit the normal (SVD) using only those. Raises ValueError if
    there aren't enough valid views/pairs/candidates.
    """
    view_idxs = sorted(
        int(k) for k, pts in points_by_image.items()
        if get_point_by_obj_id(pts, 1) is not None and get_point_by_obj_id(pts, 2) is not None
    )
    if len(view_idxs) < 4:
        raise ValueError(f"need >=4 valid views (2 independent pairs), got {len(view_idxs)}")

    pair_lines = []
    for i, j in combinations(view_idxs, 2):
        res = _line_from_view_pair(i, j, points_by_image, images_sent, fov_deg, image_size)
        if res is not None:
            point, direction = res
            pair_lines.append(((i, j), point, direction))

    if len(pair_lines) < 2:
        raise ValueError("not enough pair-lines to generate plane candidates")

    view_dirs = {
        idx: view_forward_direction(images_sent[idx]["R"], images_sent[idx]["T"], fov_deg, image_size)
        for idx in view_idxs
    }

    def score_normal(n: np.ndarray) -> float:
        vals = sorted(abs(np.dot(view_dirs[i], n)) for i in view_idxs)
        k = max(2, len(vals) // 2)
        return float(np.mean(vals[:k]))

    candidates = []
    for a in range(len(pair_lines)):
        pa, _, dir_a = pair_lines[a]
        for b in range(a + 1, len(pair_lines)):
            pb, _, dir_b = pair_lines[b]
            if set(pa) & set(pb):
                continue  # require independent view-pairs
            n = np.cross(dir_a, dir_b)
            norm = np.linalg.norm(n)
            if norm < 1e-6:
                continue
            candidates.append(n / norm)

    if not candidates:
        raise ValueError("could not generate any candidate normal (pair-lines too parallel)")

    scored = sorted(candidates, key=score_normal)
    best_normal = scored[0]

    good_views = [i for i in view_idxs if abs(np.dot(view_dirs[i], best_normal)) < edge_on_thresh]

    refit_dirs   = [d for (pa, _, d) in pair_lines if set(pa).issubset(good_views)]
    refit_points = [p for (pa, p, _) in pair_lines if set(pa).issubset(good_views)]

    if len(refit_dirs) >= 2:
        D = np.asarray(refit_dirs)
        _, _, Vt = np.linalg.svd(D)
        refined_normal = Vt[-1]
        refined_normal /= np.linalg.norm(refined_normal)
        used_points = refit_points
    else:
        refined_normal = best_normal
        used_points = [p for (_, p, _) in pair_lines]

    origin = np.mean(used_points, axis=0)

    return {
        "normal": refined_normal.tolist(),
        "origin": origin.tolist(),
        "n_views_used": len(good_views),
        "n_candidates": len(candidates),
        "good_views": good_views,
    }


def detect_planes_no_mesh(
    points_by_image: dict, images_sent: list[dict],
    fov_deg: float, image_size: int,
    edge_on_thresh: float = EDGE_ON_THRESH_DEFAULT,
    max_planes: int = 1,
    dup_angle_thresh_deg: float = DUP_ANGLE_THRESH_DEFAULT,
) -> list[dict]:
    """
    Sequential consolidation (docs/pipeline_sin_malla.md S4, Fase 2 of
    docs/implementacion_pipeline_sin_malla.md): finds plane 1 via
    estimate_plane_no_mesh, removes its edge-on views from the pool, repeats
    on the remainder to find plane 2, 3, etc. Stops when: the pool has <4
    views, the fit fails (not enough independent pair-lines), or the
    candidate normal is essentially identical (<dup_angle_thresh_deg) to an
    already-accepted plane. With max_planes=1 (the default), this is
    equivalent to a single estimate_plane_no_mesh call.
    """
    def angular_error_deg(v1: np.ndarray, v2: np.ndarray) -> float:
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        return float(np.degrees(np.arccos(np.clip(np.abs(np.dot(v1, v2)), 0.0, 1.0))))

    all_view_idxs = sorted(
        int(k) for k, pts in points_by_image.items()
        if get_point_by_obj_id(pts, 1) is not None and get_point_by_obj_id(pts, 2) is not None
    )
    pool = set(all_view_idxs)
    planes: list[dict] = []

    while len(planes) < max_planes:
        if len(pool) < 4:
            break
        sub_points = {k: v for k, v in points_by_image.items() if int(k) in pool}
        try:
            pred = estimate_plane_no_mesh(sub_points, images_sent, fov_deg, image_size, edge_on_thresh)
        except ValueError:
            break

        normal = np.array(pred["normal"])
        if any(angular_error_deg(normal, np.array(p["normal"])) < dup_angle_thresh_deg for p in planes):
            break

        planes.append(pred)
        pool -= set(pred["good_views"])

    return planes


# ── Per-object estimation ─────────────────────────────────────────────────────

def process_object(
    object_dir:    Path,
    symmetry_type: str,
    sizes:         list[int],
    lightings:     list[str],
    overwrite:     bool = False,
    experiment_id: str | None = None,
    edge_on_thresh: float = EDGE_ON_THRESH_DEFAULT,
    max_planes:    int = 1,
) -> None:
    input_file  = exp_filename(MOLMO_JSON, experiment_id)
    output_file = exp_filename(OUTPUT_FILE, experiment_id)
    output_path = object_dir / output_file
    if output_path.exists() and not overwrite:
        return

    # (n_views_key, size, lighting) -> (points_by_image, images_sent, fov_deg, image_size)
    configs: dict[str, list[tuple]] = defaultdict(list)

    for size in sizes:
        for lighting in lightings:
            render_dir = object_dir / str(size) / lighting
            molmo_path = render_dir / input_file
            if not molmo_path.exists():
                continue

            manifest_path = render_dir / MANIFEST_FILE
            if manifest_path.exists():
                with open(manifest_path, encoding="utf-8") as f:
                    manifest = json.load(f)
                image_size = manifest.get("image_size", size)
                fov_deg    = manifest.get("fov", DEFAULT_FOV)
            else:
                image_size = size
                fov_deg    = DEFAULT_FOV

            with open(molmo_path, encoding="utf-8") as f:
                molmo_data = json.load(f)

            for n_views_key, group in molmo_data.items():
                images_sent     = group.get("images_sent", [])
                points_by_image = group.get("points_by_image", {})
                if not points_by_image:
                    continue
                configs[n_views_key].append((points_by_image, images_sent, fov_deg, image_size))

    if not configs:
        return

    n_views_predictions: dict = {}

    for n_views_key, cfg_list in configs.items():
        try:
            if symmetry_type == "axis_sym":
                # Pool interpretation planes across every (size, lighting) config
                # that has this n_views group -- same "pool across configs, fit
                # once per n_views" convention as estimate_symmetry.py.
                centers, normals = [], []
                for points_by_image, images_sent, fov_deg, image_size in cfg_list:
                    for img_idx_str, pts in points_by_image.items():
                        pair = widest_pair(pts)
                        if pair is None:
                            continue
                        p_a, p_b = pair
                        cam = images_sent[int(img_idx_str)]
                        C, d_a = ray_dir_for_point(p_a["x"], p_a["y"], cam["R"], cam["T"], fov_deg, image_size)
                        _, d_b = ray_dir_for_point(p_b["x"], p_b["y"], cam["R"], cam["T"], fov_deg, image_size)
                        n = interpretation_plane_normal(d_a, d_b)
                        if n is None:
                            continue
                        centers.append(C)
                        normals.append(n)
                if len(normals) < 2:
                    continue
                point, direction = triangulate_line(centers, normals)
                pred = {"direction": direction.tolist(), "origin": point.tolist(), "n_views_used": len(normals)}

                n_views_predictions[n_views_key] = {
                    "n_points_raw": len(normals), "n_points_fit": len(normals),
                    METHOD_KEY: {
                        "direction": pred["direction"], "origin": pred["origin"],
                        "n_points": pred["n_views_used"], "n_views_used": pred["n_views_used"],
                        "n_inliers": None, "sde": None, "accepted": None,
                    },
                }

            else:
                # Plane candidates need index-consistent pairing within ONE
                # points_by_image/images_sent set -- process each (size,
                # lighting) config independently and keep the one with the
                # most views used (simple tie-break; true cross-config
                # pooling for plane is a known simplification, see
                # docs/implementacion_pipeline_sin_malla.md).
                best_planes: list[dict] | None = None
                for points_by_image, images_sent, fov_deg, image_size in cfg_list:
                    try:
                        planes = detect_planes_no_mesh(
                            points_by_image, images_sent, fov_deg, image_size,
                            edge_on_thresh=edge_on_thresh, max_planes=max_planes,
                        )
                    except ValueError:
                        continue
                    if not planes:
                        continue
                    if best_planes is None or planes[0]["n_views_used"] > best_planes[0]["n_views_used"]:
                        best_planes = planes

                if not best_planes:
                    continue

                n_points_total = sum(p["n_views_used"] for p in best_planes)
                if max_planes == 1:
                    p = best_planes[0]
                    n_views_predictions[n_views_key] = {
                        "n_points_raw": n_points_total, "n_points_fit": n_points_total,
                        METHOD_KEY: {
                            "normal": p["normal"], "origin": p["origin"],
                            "n_points": p["n_views_used"], "n_views_used": p["n_views_used"],
                            "n_inliers": None, "sde": None, "accepted": None,
                        },
                    }
                else:
                    n_views_predictions[n_views_key] = {
                        "n_points_raw": n_points_total, "n_points_fit": n_points_total,
                        f"{METHOD_KEY}_multiplane": {
                            "planes": [
                                {"normal": p["normal"], "origin": p["origin"],
                                 "n_views_used": p["n_views_used"], "n_candidates": p["n_candidates"],
                                 "good_views": p["good_views"]}
                                for p in best_planes
                            ],
                        },
                    }
        except Exception:
            continue

    if not n_views_predictions:
        return

    output = {
        "object_id":           object_dir.name,
        "symmetry_type":       symmetry_type,
        "point_mode":          METHOD_KEY,
        "clustering_method":   "none",
        "hdbscan_min_samples": None,
        "n_views_predictions": n_views_predictions,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit symmetry axis/plane via multi-view triangulation -- no mesh, no ray-casting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--renders-root",  required=True,
                   help="Root folder of renders (same tree map_to_3d.py/estimate_symmetry.py use)")
    p.add_argument("--symmetry-type", required=True, choices=["axis_sym", "plane_sym"])
    p.add_argument("--sizes",     type=int, nargs="+", default=DEFAULT_SIZES)
    p.add_argument("--lightings", type=str, nargs="+", default=DEFAULT_LIGHTINGS,
                   choices=["flat", "darker", "brighter"])
    p.add_argument("--gpu-id",   type=int, default=0)
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing predicted_symmetry.json files")
    p.add_argument("--experiment-id", default=None,
                   help="Reads molmo_multiview_<ID>.json, writes predicted_symmetry_<ID>.json. "
                        "Use a DIFFERENT id than any with-mesh run to avoid confusion (e.g. "
                        "'axis_v02_nomesh' vs. the with-mesh pipeline's 'axis_v02') -- both "
                        "write to distinct files, there is no collision either way.")
    p.add_argument("--max-objects", type=int, default=None,
                   help="Limit to the first N objects (sorted order).")
    p.add_argument("--edge-on-thresh", type=float, default=EDGE_ON_THRESH_DEFAULT,
                   help="plane_sym only: |cos(angle)| below this counts a view as 'edge-on' "
                        "to a plane candidate.")
    p.add_argument("--max-planes", type=int, default=1,
                   help="plane_sym only: consolidate up to N planes per object (docs/pipeline_sin_malla.md "
                        "S4). 1 (default) keeps the single-plane schema Mapping/evaluate.py already "
                        "understands; >1 writes a 'planes' list that evaluate.py does not yet consume "
                        "(see docs/implementacion_pipeline_sin_malla.md, Fase 2).")
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

    output_file = exp_filename(OUTPUT_FILE, args.experiment_id)
    print(f"\nFitting {args.symmetry_type} symmetry (SIN malla / triangulation) for {len(objects)} objects...")
    print(f"Sizes/lightings : {args.sizes} / {args.lightings}")
    print(f"Output file     : {output_file}")
    if args.experiment_id:
        print(f"Experiment ID   : {args.experiment_id}")
    if args.symmetry_type == "plane_sym":
        print(f"edge_on_thresh  : {args.edge_on_thresh}")
        print(f"max_planes      : {args.max_planes}")
    print(f"({'overwrite' if args.overwrite else 'skip existing'})")

    for obj_dir in tqdm(objects, unit="obj", dynamic_ncols=True):
        process_object(
            object_dir     = obj_dir,
            symmetry_type  = args.symmetry_type,
            sizes          = args.sizes,
            lightings      = args.lightings,
            overwrite      = args.overwrite,
            experiment_id  = args.experiment_id,
            edge_on_thresh = args.edge_on_thresh,
            max_planes     = args.max_planes,
        )

    print(f"\n[GPU {args.gpu_id}] Done.")


if __name__ == "__main__":
    main()
