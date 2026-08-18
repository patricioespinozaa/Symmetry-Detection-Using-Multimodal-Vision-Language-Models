#!/usr/bin/env python3
"""
reference_metrics.py
---------------------
Standalone re-scoring of ALREADY-FITTED plane-symmetry predictions using the
exact SDE and F1 formulas from the reference paper's evaluation scripts
(metric_SDE.py / metric_F1.py — both plane-only).

Does NOT re-run map_to_3d / estimate_symmetry. It reads the normal/origin
already stored in predicted_symmetry_<EXP>.json and only re-scores them, so
it is cheap even against many (experiment, method, n_views) combinations —
no ray-casting, no SVD, no RANSAC. Meant to be pointed at a short list of
"winner" experiment ids picked from the ranking notebook, not the full sweep
(see the discussion that led to this script: running the full clustering x
patch-size cross product for every experiment wasn't worth it either).

Differences vs. the pipeline's own SDE (Mapping/estimate_symmetry.py::sde_plane),
replicated here ON PURPOSE to match the reference paper's numbers:
  - samples 1000 points on the mesh SURFACE (gpy.random_points_on_mesh),
    not a random subset of mesh VERTICES.
  - measures squared distance to the actual triangulated surface
    (gpy.squared_distance with an AABB tree), not nearest-neighbour distance
    to a sampled vertex.
  - reports the MEAN SQUARED distance, unnormalized (no /bbox_diag) — same
    units/scale as the reference script, NOT comparable to the pipeline's own
    sde_mean column without conversion.

F1 treats each of our predictions as a single always-accepted candidate
(confidence=1.0 — our pipeline never emits multiple candidates with a
confidence score, only one best plane per method/n_views). This degenerates
the reference's multi-candidate matching into: for each ground-truth plane,
TP if our one prediction is within the threshold of it (first unmatched GT it
clears) and hasn't already matched a different GT, else FP; every unmatched
GT plane after that is a FN. The counting loop (including its exact
tie-break/iteration order) is copied verbatim from metric_F1.py so the
resulting numbers are computed the same way, quirks included — this is
deliberate, the whole point is a like-for-like comparison against the
reference's reported figures, not a "corrected" metric.

Only plane_sym is supported — the reference functions are plane-only
(calplaneloss reflects through a hyperplane; the F1 matching works on planes
as 4-vectors [nx, ny, nz, d]). Requires gpytoolbox (`pip install gpytoolbox`).

Usage:
    python Mapping/reference_metrics.py \\
        --renders-root ../data/renders --objects-root ../data/objects \\
        --experiment-id plane_v04_1_flowB plane_v04_1_flowC_p5 \\
        --methods svd ransac_svd_sde \\
        --out reference_metrics_plane.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

try:
    import gpytoolbox as gpy
except ImportError:
    print("[error] gpytoolbox is required: pip install gpytoolbox", file=sys.stderr)
    raise

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_common.datasets import OBJECTS_SUBDIR, load_mesh
from pipeline_common.naming import exp_filename

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import parse_true_label, PREDICTED_FILE  # reuse the pipeline's own GT parser

METHODS = ("svd", "ransac_svd", "svd_sde", "ransac_svd_sde")
THRESHOLDS_INLIER = [0.05, 0.1, 0.15, 0.2]   # metric_F1.py's set_threshold
N_SAMPLES_DEFAULT = 1000                      # metric_SDE.py's sample count


# ── Verbatim from the reference scripts (metric_SDE.py / metric_F1.py) ────────
# Do not "clean up" these two functions — the point of this script is that the
# numbers match the reference paper's exactly, quirks included.

def calplaneloss(plane: np.ndarray, vertices: np.ndarray, faces: np.ndarray, points: np.ndarray) -> float:
    points = np.hstack((points, np.ones((points.shape[0], 1))))
    lam = points.dot(plane.T)
    planepoints = points - 2 * lam * plane
    d, ind, b = gpy.squared_distance(planepoints[:, 0:3], vertices, faces, use_aabb=True, use_cpp=True)
    return float(np.mean(d))


# ── Adapters: our stored predictions/labels -> the reference's plane repr ─────

def normal_origin_to_plane(normal: list[float], origin: list[float]) -> np.ndarray:
    """Same convention as the reference scripts: plane = [nx, ny, nz, d], d = -origin . normal."""
    normal = np.asarray(normal, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    d = -origin.dot(normal)
    return np.array([normal[0], normal[1], normal[2], d]).reshape(1, 4)


def gt_planes_for_object(objects_dir: Path, object_id: str) -> list[np.ndarray]:
    txt_path = objects_dir / f"{object_id}.txt"
    if not txt_path.exists():
        return []
    label = parse_true_label(txt_path)
    if label["type"] != "plane":
        return []
    return [normal_origin_to_plane(e["normal"], e["origin"]) for e in label["elements"]]


def f1_match_counts(pred_plane: np.ndarray, gt_planes: list[np.ndarray], threshold_inlier: float) -> tuple[int, int, int]:
    """Verbatim port of the matching loop inside metric_F1.py::f1_score_calc, restricted
    to our single always-accepted predicted plane per object (see module docstring)."""
    mask = np.zeros(len(gt_planes), dtype=bool)
    tp = fp = 0
    for idx, gt in enumerate(gt_planes):
        if mask[idx]:
            continue
        val1 = np.linalg.norm(pred_plane - gt)
        val2 = np.linalg.norm(pred_plane + gt)
        val = min(val1, val2)
        if val < threshold_inlier:
            mask[idx] = True
            tp += 1
        else:
            fp += 1
    fn = int(np.sum(~mask))
    return tp, fp, fn


# ── Mesh + surface-sample cache ────────────────────────────────────────────────
# Building the AABB tree and sampling 1000 surface points is the only real cost
# here (everything else is re-scoring an already-fitted plane) -- do it once per
# object and reuse across every experiment/method/n_views that touches it.

class MeshCache:
    def __init__(self, objects_dir: Path, n_samples: int, seed: int | None):
        self.objects_dir = objects_dir
        self.n_samples = n_samples
        self.seed = seed
        self._cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray] | None] = {}

    def get(self, object_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if object_id in self._cache:
            return self._cache[object_id]
        obj_path = self.objects_dir / f"{object_id}.obj"
        if not obj_path.exists():
            self._cache[object_id] = None
            return None
        try:
            mesh = load_mesh(obj_path)
            v = np.asarray(mesh.vertices, dtype=np.float64)
            f = np.asarray(mesh.faces, dtype=np.int64)
            if self.seed is not None:
                np.random.seed(self.seed)  # gpytoolbox's sampler draws from the numpy global RNG
            sample = gpy.random_points_on_mesh(v, f, self.n_samples)
        except Exception as e:
            print(f"  [warn] mesh load/sample failed for {object_id}: {e}")
            self._cache[object_id] = None
            return None
        self._cache[object_id] = (v, f, sample)
        return self._cache[object_id]


# ── Main scoring loop ──────────────────────────────────────────────────────────

def score_experiment(
    renders_root: Path, objects_dir: Path, experiment_id: str,
    methods: list[str], mesh_cache: MeshCache,
) -> list[dict]:
    sym_dir = renders_root / "plane_sym"
    predicted_file = exp_filename(PREDICTED_FILE, experiment_id)
    object_dirs = sorted(d for d in sym_dir.iterdir() if d.is_dir())

    sde_values: dict[tuple[str, str], list[float]] = {}
    f1_counts: dict[tuple[str, str], dict[float, list[int]]] = {}
    n_objects_seen: dict[tuple[str, str], int] = {}

    for obj_dir in object_dirs:
        object_id = obj_dir.name
        pred_path = obj_dir / predicted_file
        if not pred_path.exists():
            continue
        with open(pred_path, "r", encoding="utf-8") as fh:
            pred = json.load(fh)

        gt_planes = gt_planes_for_object(objects_dir, object_id)
        mesh_data = mesh_cache.get(object_id)

        for n_views_key, entry in pred.get("n_views_predictions", {}).items():
            for method in methods:
                m = entry.get(method)
                if m is None or "normal" not in m:
                    continue
                key = (method, n_views_key)
                n_objects_seen[key] = n_objects_seen.get(key, 0) + 1
                pred_plane = normal_origin_to_plane(m["normal"], m["origin"])

                if mesh_data is not None:
                    v, f, sample = mesh_data
                    sde_values.setdefault(key, []).append(calplaneloss(pred_plane, v, f, sample))

                counts_by_t = f1_counts.setdefault(key, {t: [0, 0, 0] for t in THRESHOLDS_INLIER})
                for t in THRESHOLDS_INLIER:
                    tp, fp, fn = f1_match_counts(pred_plane, gt_planes, t)
                    c = counts_by_t[t]
                    c[0] += tp
                    c[1] += fp
                    c[2] += fn

    rows = []
    for key in sorted(set(sde_values) | set(f1_counts)):
        method, n_views_key = key
        sde_vals = np.array(sde_values.get(key, []))
        f1_per_t = []
        for t in THRESHOLDS_INLIER:
            tp, fp, fn = f1_counts[key][t]
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            f1_per_t.append(f1)
        rows.append({
            "experiment": experiment_id,
            "method": method,
            "n_views": n_views_key,
            "n_objects": n_objects_seen[key],
            "sde_ref_mean": float(sde_vals.mean()) if len(sde_vals) else None,
            "sde_ref_min": float(sde_vals.min()) if len(sde_vals) else None,
            "sde_ref_max": float(sde_vals.max()) if len(sde_vals) else None,
            "f1_ref": float(np.mean(f1_per_t)),
        })
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-score already-fitted plane predictions with the reference paper's exact SDE/F1 formulas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--renders-root", required=True)
    p.add_argument("--objects-root", required=True)
    p.add_argument("--experiment-id", nargs="+", required=True, metavar="EXP_ID",
                   help="One or more experiment ids to re-score. Pass only your ranking "
                        "'winners' -- this is meant to be cheap on a handful of experiments, "
                        "not run over the full sweep.")
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    p.add_argument("--n-samples", type=int, default=N_SAMPLES_DEFAULT,
                   help="Surface points sampled per object (matches the reference's 1000).")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for the surface sampling (reproducible re-runs). Pass -1 to "
                        "leave it unseeded, matching the reference script's own behaviour "
                        "at the cost of run-to-run noise.")
    p.add_argument("--out", default="reference_metrics_plane.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    renders_root = Path(args.renders_root)
    objects_dir = Path(args.objects_root) / OBJECTS_SUBDIR["plane_sym"]
    seed = None if args.seed == -1 else args.seed

    mesh_cache = MeshCache(objects_dir, args.n_samples, seed)

    all_rows: list[dict] = []
    for exp_id in args.experiment_id:
        print(f"\n=== {exp_id} ===")
        rows = score_experiment(renders_root, objects_dir, exp_id, args.methods, mesh_cache)
        if not rows:
            print("  (sin predicciones encontradas -- revisa --experiment-id / --renders-root)")
        for r in rows:
            print(f"  {r['method']:16s} n_views={r['n_views']:>3}  n_obj={r['n_objects']:>4}  "
                  f"SDE_ref(mean)={r['sde_ref_mean']:.6f}  F1_ref={r['f1_ref']:.4f}")
        all_rows.extend(rows)

    if all_rows:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nGuardado: {args.out}")
    else:
        print("\n[warn] nada que guardar -- 0 filas producidas")


if __name__ == "__main__":
    main()
