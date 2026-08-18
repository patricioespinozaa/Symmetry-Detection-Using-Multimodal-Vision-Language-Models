#!/usr/bin/env python3
"""
reference_metrics.py
---------------------
Standalone re-scoring of ALREADY-FITTED plane- and axis-symmetry predictions.

For plane_sym: the exact SDE and F1 formulas from the reference paper's
evaluation scripts (metric_SDE.py / metric_F1.py — both plane-only in the
source repo).

For axis_sym: only an SDE-style metric (`calaxisloss`), NOT a port of
anything in the reference repo -- that repo has no ground-truth-matching
evaluation for axes at all (confirmed by reading it: `evaluation/` only has
a self-supervised DINOv2-feature-invariance proxy for axes, no SDE/F1 against
ground truth). `calaxisloss` is our own extension, built by applying the same
convention the wider literature uses for planar SDE (PRS-Net and follow-ups:
area-weighted surface sampling, squared distance to the true mesh surface,
unnormalized) to axis reflection (180 deg rotation about the line) instead of
planar reflection. No F1 for axis_sym: cross-checked with a literature search
(see conversation) that found no established multi-candidate F1 convention
for axis symmetry detection in 3D -- angular_error/AUC_angular is what the
field actually reports for axes, which this project already computes.

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

f1_match_counts() takes a LIST of predicted planes -- the exact same shape as
metric_F1.py's own matching loop (`for pred in predicted: for idx, gt in
enumerate(planes_gt): ...`), copied verbatim including its tie-break/iteration
order. Today our pipeline only ever fits one plane per (object, method,
n_views), so score_experiment() always calls it with a single-element list
(confidence=1.0 implicit -- our pipeline has no per-candidate confidence
score to filter on). But the function itself no longer assumes that: the day
this pipeline emits several candidate planes per object (e.g. treating the
4 fitting methods as a candidate set, or a future multi-candidate detector
closer to EnhancedBackProjection's up-to-10-candidates-per-object approach),
you pass the full list straight in and get the identical algorithm they use
-- no rewrite needed. See the EnhancedBackProjection comparison discussion:
their confidence/dedup filtering (metric_F1.py's `confidence >= threshold`
and `distPlanes` non-max-suppression before building `predicted`) still isn't
replicated here, since it only matters once real per-candidate confidence
scores exist upstream.

Requires gpytoolbox (`pip install gpytoolbox`).

Usage (plane, SDE_ref + F1_ref):
    python Mapping/reference_metrics.py \\
        --renders-root ../data/renders --objects-root ../data/objects \\
        --symmetry-type plane_sym \\
        --experiment-id plane_v04_1_flowB plane_v04_1_flowC_p5 \\
        --methods svd ransac_svd_sde \\
        --out reference_metrics_plane.csv

Usage (axis, SDE_ref only -- f1_ref comes back as None, see above):
    python Mapping/reference_metrics.py \\
        --renders-root ../data/renders --objects-root ../data/objects \\
        --symmetry-type axis_sym \\
        --experiment-id axis_v05_1_flowC_p5 \\
        --out reference_metrics_axis.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
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


# ── Axis SDE (our own extension, NOT ported from the reference repo) ──────────
# Same convention as calplaneloss (area-weighted surface sample, squared distance
# to the real mesh surface via AABB, unnormalized) applied to axis reflection
# (180 deg rotation about the line) instead of planar reflection. The reflection
# formula itself matches Mapping/estimate_symmetry.py::sde_axis exactly (same
# geometry, "foot of perpendicular on the axis, then 2*proj - point"); what
# changes here is the same trio of literature-convention differences as
# calplaneloss vs. sde_plane: surface sample vs. vertex sample, true-surface
# distance vs. nearest-sampled-point distance, squared+unnormalized vs.
# linear+bbox-normalized.

def calaxisloss(axis_dir: np.ndarray, axis_origin: np.ndarray,
                 vertices: np.ndarray, faces: np.ndarray, points: np.ndarray) -> float:
    t = (points - axis_origin) @ axis_dir
    proj = axis_origin + t[:, None] * axis_dir
    reflected = 2.0 * proj - points
    d, ind, b = gpy.squared_distance(reflected, vertices, faces, use_aabb=True, use_cpp=True)
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


def f1_match_counts(predicted: list[np.ndarray], gt_planes: list[np.ndarray], threshold_inlier: float) -> tuple[int, int, int]:
    """Verbatim port of the matching loop inside metric_F1.py::f1_score_calc.

    `predicted` is a list of candidate planes -- today score_experiment() always
    passes a single-element list (one plane per method/n_views), but this
    function makes no assumption about that: pass in as many candidate planes
    as you have (e.g. all 4 fitting methods as a set, or a future multi-plane
    detector) and it runs the identical algorithm metric_F1.py uses."""
    mask = np.zeros(len(gt_planes), dtype=bool)
    tp = fp = 0
    for pred_plane in predicted:
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


# ── Discovery: every experiment id that has predictions on disk ───────────────
# Same idea as Mapping/run_all_postprocessing.py::discover_experiments, but keyed
# off predicted_symmetry_*.json (already-fitted results) instead of
# molmo_multiview_*.json (raw Molmo output) -- this naturally picks up every
# variant (baseline, _cluster, _hdbscan_ms{2,3,5}, _p{3,5}), not just the base
# experiment ids, since that's what --all is meant to cover ("todos los
# experimentos", matching how the ~1M/~950K prediction-count estimates were
# computed).

def discover_experiment_ids(renders_root: Path, symmetry_type: str) -> list[str]:
    import re
    sym_dir = renders_root / symmetry_type
    found: set[str] = set()
    for f in sym_dir.glob("*/predicted_symmetry_*.json"):
        m = re.match(r"predicted_symmetry_(.+)\.json$", f.name)
        if m:
            found.add(m.group(1))
    return sorted(found)


# ── Main scoring loop ──────────────────────────────────────────────────────────

def score_experiment(
    renders_root: Path, objects_dir: Path, experiment_id: str,
    methods: list[str], mesh_cache: MeshCache, symmetry_type: str,
) -> list[dict]:
    is_axis = symmetry_type == "axis_sym"
    vec_key = "direction" if is_axis else "normal"

    sym_dir = renders_root / symmetry_type
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

        # No F1 for axis: no ground-truth axis matching exists in the reference
        # repo or, per the literature check, in the field generally -- see
        # module docstring. gt_planes stays [] for axis, f1_match_counts is
        # simply never called in that branch below.
        gt_planes = [] if is_axis else gt_planes_for_object(objects_dir, object_id)
        mesh_data = mesh_cache.get(object_id)

        for n_views_key, entry in pred.get("n_views_predictions", {}).items():
            # svd/svd_sde (and ransac_svd/ransac_svd_sde) always share the exact same
            # direction-or-normal/origin -- the "_sde" variant only adds the
            # pipeline's own internal SDE, it doesn't refit anything (see
            # estimate_symmetry.py::_make_entry). Cache the (expensive) re-scoring
            # per distinct (vec, origin) seen for this object/n_views so those
            # duplicate pairs are scored once, not twice.
            plane_cache: dict[tuple, tuple[float | None, dict[float, tuple[int, int, int]]]] = {}

            for method in methods:
                m = entry.get(method)
                if m is None or vec_key not in m:
                    continue
                key = (method, n_views_key)
                n_objects_seen[key] = n_objects_seen.get(key, 0) + 1

                cache_key = (tuple(m[vec_key]), tuple(m["origin"]))
                if cache_key not in plane_cache:
                    sde_val = None
                    if mesh_data is not None:
                        v, f, sample = mesh_data
                        if is_axis:
                            axis_dir = np.asarray(m["direction"], dtype=np.float64)
                            axis_origin = np.asarray(m["origin"], dtype=np.float64)
                            sde_val = calaxisloss(axis_dir, axis_origin, v, f, sample)
                        else:
                            pred_plane = normal_origin_to_plane(m["normal"], m["origin"])
                            sde_val = calplaneloss(pred_plane, v, f, sample)
                    f1_by_t: dict[float, tuple[int, int, int]] = {}
                    if not is_axis:
                        pred_plane = normal_origin_to_plane(m["normal"], m["origin"])
                        f1_by_t = {
                            t: f1_match_counts([pred_plane], gt_planes, t)
                            for t in THRESHOLDS_INLIER
                        }
                    plane_cache[cache_key] = (sde_val, f1_by_t)
                sde_val, f1_by_t = plane_cache[cache_key]

                if sde_val is not None:
                    sde_values.setdefault(key, []).append(sde_val)

                if not is_axis:
                    counts_by_t = f1_counts.setdefault(key, {t: [0, 0, 0] for t in THRESHOLDS_INLIER})
                    for t in THRESHOLDS_INLIER:
                        tp, fp, fn = f1_by_t[t]
                        c = counts_by_t[t]
                        c[0] += tp
                        c[1] += fp
                        c[2] += fn

    rows = []
    for key in sorted(set(sde_values) | set(f1_counts)):
        method, n_views_key = key
        sde_vals = np.array(sde_values.get(key, []))
        f1_ref = None
        if not is_axis:
            f1_per_t = []
            for t in THRESHOLDS_INLIER:
                tp, fp, fn = f1_counts[key][t]
                precision = tp / (tp + fp) if (tp + fp) else 0.0
                recall = tp / (tp + fn) if (tp + fn) else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
                f1_per_t.append(f1)
            f1_ref = float(np.mean(f1_per_t))
        rows.append({
            "experiment": experiment_id,
            "method": method,
            "n_views": n_views_key,
            "n_objects": n_objects_seen[key],
            "sde_ref_mean": float(sde_vals.mean()) if len(sde_vals) else None,
            "sde_ref_min": float(sde_vals.min()) if len(sde_vals) else None,
            "sde_ref_max": float(sde_vals.max()) if len(sde_vals) else None,
            "f1_ref": f1_ref,
        })
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-score already-fitted plane predictions with the reference paper's exact SDE/F1 formulas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--renders-root", required=True)
    p.add_argument("--objects-root", required=True)
    p.add_argument("--symmetry-type", required=True, choices=["plane_sym", "axis_sym"],
                   help="plane_sym gets SDE_ref + F1_ref (both ported from the reference "
                        "repo). axis_sym gets SDE_ref only (f1_ref comes back as None -- "
                        "no F1 convention exists for axis, see module docstring).")
    p.add_argument("--experiment-id", nargs="+", default=None, metavar="EXP_ID",
                   help="One or more experiment ids to re-score. Pass only your ranking "
                        "'winners' -- this is meant to be cheap on a handful of experiments, "
                        "not run over the full sweep. Required unless --all is given.")
    p.add_argument("--all", action="store_true",
                   help="Discover and score every experiment id that has "
                        "predicted_symmetry_*.json on disk for this --symmetry-type "
                        "(baseline + every clustering/patch variant) -- the full sweep, "
                        "several hours. Overrides --experiment-id.")
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    p.add_argument("--n-samples", type=int, default=N_SAMPLES_DEFAULT,
                   help="Surface points sampled per object (matches the reference's 1000).")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for the surface sampling (reproducible re-runs). Pass -1 to "
                        "leave it unseeded, matching the reference script's own behaviour "
                        "at the cost of run-to-run noise.")
    p.add_argument("--out", default=None,
                   help="Default: reference_metrics_plane.csv or reference_metrics_axis.csv, "
                        "depending on --symmetry-type.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.all and not args.experiment_id:
        raise SystemExit("[error] pasa --experiment-id <...> o --all")

    renders_root = Path(args.renders_root)
    objects_dir = Path(args.objects_root) / OBJECTS_SUBDIR[args.symmetry_type]
    seed = None if args.seed == -1 else args.seed
    out_path = args.out or f"reference_metrics_{'axis' if args.symmetry_type == 'axis_sym' else 'plane'}.csv"

    if args.all:
        exp_ids = discover_experiment_ids(renders_root, args.symmetry_type)
        print(f"--all: descubiertos {len(exp_ids)} experiment id(s) con predicciones en "
              f"{renders_root / args.symmetry_type}")
    else:
        exp_ids = args.experiment_id

    mesh_cache = MeshCache(objects_dir, args.n_samples, seed)

    # Escritura incremental: cada experimento termina -> se agrega al CSV al toque.
    # En una corrida de horas (--all), esto evita perder todo si se corta a mitad
    # de camino -- lo ya calculado queda guardado, y se puede retomar filtrando
    # --experiment-id a lo que falte (o simplemente volver a correr, es idempotente
    # salvo por el tiempo perdido en lo que ya estaba).
    fieldnames = ["experiment", "method", "n_views", "n_objects",
                  "sde_ref_mean", "sde_ref_min", "sde_ref_max", "f1_ref"]
    csv_file = open(out_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    n_total_rows = 0
    t0 = time.time()
    for i, exp_id in enumerate(exp_ids, start=1):
        elapsed = time.time() - t0
        eta = ""
        if i > 1:
            eta_s = elapsed / (i - 1) * (len(exp_ids) - i + 1)
            eta = f"  (ETA ~{eta_s/60:.0f} min)"
        print(f"\n=== [{i}/{len(exp_ids)}] {exp_id} ==={eta}")
        rows = score_experiment(renders_root, objects_dir, exp_id, args.methods, mesh_cache, args.symmetry_type)
        if not rows:
            print("  (sin predicciones encontradas)")
        for r in rows:
            f1_str = f"{r['f1_ref']:.4f}" if r["f1_ref"] is not None else "n/a (axis)"
            sde_str = f"{r['sde_ref_mean']:.6f}" if r["sde_ref_mean"] is not None else "n/a"
            print(f"  {r['method']:16s} n_views={r['n_views']:>3}  n_obj={r['n_objects']:>4}  "
                  f"SDE_ref(mean)={sde_str}  F1_ref={f1_str}")
            writer.writerow(r)
        csv_file.flush()
        n_total_rows += len(rows)

    csv_file.close()
    total_min = (time.time() - t0) / 60
    print(f"\nGuardado: {out_path}  ({n_total_rows} filas, {len(exp_ids)} experimentos, {total_min:.1f} min)")


if __name__ == "__main__":
    main()
