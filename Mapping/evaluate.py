"""
evaluate.py
-----------
Compares predicted symmetry (from estimate_symmetry.py) against true labels
(.txt files) and computes evaluation metrics aligned with the paper.

Each predicted_symmetry.json contains FOUR estimates per n_views group (2×2 grid).
Use --method to select which one to evaluate:

    svd            — SVD on all points
    ransac_svd     — RANSAC inlier selection + SVD
    svd_sde        — SVD on all points  (SDE stored in JSON, not recomputed here)
    ransac_svd_sde — RANSAC + SVD  (SDE stored in JSON, not recomputed here)

Merge note (see docs/actualizacion_metricas.md)
------------------------------------------------
Everything that used to live in the standalone `Mapping/reference_metrics.py`
(SDE_ref, F1_ref — the exact formulas from an external reference paper) now
lives here, behind `--with-reference-metrics`. `reference_metrics.py` no
longer exists as a separate file; any script/notebook importing from it
should import from `evaluate` instead (same function names:
`calplaneloss`, `calaxisloss`, `normal_origin_to_plane`, `gt_planes_for_object`,
`f1_match_counts`, `THRESHOLDS_INLIER`, `N_SAMPLES_DEFAULT`).

This merge also fixes a real bug found while writing docs/actualizacion_metricas.md:
the pipeline's own `sde`/`sde_mean`/`auc_sde`/`precision_sde_*` (populated by a
function formerly named `symmetry_distance_error`) was NOT a valid Symmetry
Distance Error — it was twice the mean distance from each vertex to the
predicted plane, which never checks that the *reflected* point lands on real
geometry (a plane through the centroid of a sphere would score ~0 with that
formula regardless of whether the sphere is actually symmetric about it).
That function and its output fields have been REMOVED. `SDE_ref` (below,
opt-in) is now the only SDE this script reports for plane_sym — it is the
verbatim PRS-Net-style convention (area-weighted surface sampling, squared
distance to the real triangulated mesh via an AABB tree), confirmed against
current literature to be the standard convention, unlike the removed formula.
The internal `sde`/`accepted` field inside `predicted_symmetry_<exp>.json`
(computed by `estimate_symmetry.py::sde_axis`/`sde_plane`, KDTree over a
vertex sample) is untouched — it is a separate, legitimate, cheap heuristic
used only to flag `accepted`, not what this script re-scores.

Metrics
-------
axis_sym:
    - Angular error (degrees): angle between predicted and true axis directions
      (sign-agnostic, returns value in [0, 90])
    - Translation error: point-to-line distance from predicted origin to true axis
      (raw, plus a bbox-normalized variant)
    - Precision under threshold: % of objects with angular error < 5°, 10°, 15°
    - AUC: area under the precision-vs-threshold curve (0°–45°, normalized)
    - [--with-reference-metrics] SDE_ref: area-weighted surface sample, squared
      distance to the real mesh after 180°-reflection about the predicted axis,
      unnormalized (own extension of the PRS-Net convention — no F1 exists for
      axes in the literature, confirmed by literature review, see
      reference_metrics.py's former docstring / docs/metricas_evaluacion.md)

plane_sym:
    - Angular error (degrees): angle between predicted and true plane normals
      (best-match against all true planes, sign-agnostic)
    - Translation error: point-to-plane distance from predicted origin to best true plane
      (raw, plus a bbox-normalized variant)
    - Precision under angular threshold: % of objects with angular error < 5°, 10°, 15°
    - AUC: area under the precision-vs-threshold curve (0°–45°, normalized)
    - [--with-reference-metrics] SDE_ref: verbatim port of the reference paper's
      metric_SDE.py (area-weighted surface sample, squared distance to the real
      mesh via AABB tree, unnormalized)
    - [--with-reference-metrics] F1_ref: verbatim port of the reference paper's
      metric_F1.py — greedy matching against ALL true planes at 4 distance
      thresholds {0.05,0.10,0.15,0.20} on the [nx,ny,nz,d] plane vector,
      averaged. Plus `f1_ref_hungarian`, a complementary optimal-assignment
      variant (scipy linear_sum_assignment) matching the convention used by
      more recent multi-candidate benchmarks (Reflect3D/ArchSym, 2025-2026) —
      identical to the greedy result whenever there is only one predicted
      plane per object (today's case), but can diverge once this pipeline
      predicts multiple candidate planes per object.

--with-reference-metrics is opt-in (not the default) because it needs
`gpytoolbox` and is meaningfully more expensive (mesh loading + AABB tree +
surface sampling per object) — same reasoning the old standalone
reference_metrics.py had for being pointed at a short list of "winner"
experiments rather than run on the full sweep by default.

Output
------
Results are saved under <renders_root>/<symmetry_type>/ with filenames that
encode the experiment configuration (sizes + lightings + method):

    eval_{sizes}_{lightings}_{method}_results.json   ← per-object, per-n_views metrics
    eval_{sizes}_{lightings}_{method}_summary.csv    ← aggregated metrics per n_views group

Example filenames:
    eval_s224_flat_svd_results.json
    eval_s224_448_1136_flat_brighter_darker_ransac_svd_sde_summary.csv

JSON format
-----------
{
  "symmetry_type": "axis_sym",
  "method": "ransac_svd",
  "sizes": [224],
  "lightings": ["flat"],
  "objects": {
    "<object_id>": {
      "1": {
        "angular_error_deg":  5.2,
        "translation_error":  0.03,
        "translation_error_normalized": 0.01,
        "precision_5deg":     1,
        "precision_10deg":    1,
        "precision_15deg":    1,
        "n_points":           4,
        "status":             "ok"
      },
      "6": { ... }
    }
  }
}

For plane_sym, additional fields per object/n_views:
    "matched_true_plane":   0,
    "n_true_planes":        2

With --with-reference-metrics, additional fields per object/n_views:
    "sde_ref":                 0.000841                              (both symmetry types)
    "f1_counts_ref":           {"0.05": [tp,fp,fn], ...}              (plane_sym only, raw per-threshold counts)
    "f1_counts_ref_hungarian": {"0.05": [tp,fp,fn], ...}              (plane_sym only)

F1_ref/f1_ref_hungarian themselves only exist at the SUMMARY level (CSV),
never per object — they are dataset-level metrics (TP/FP/FN accumulated
across every object before computing F1 per threshold, exactly like the
reference paper's own evaluation protocol), not an average of a per-object
F1. The per-object f1_counts_ref/f1_counts_ref_hungarian fields above are
what compute_summary accumulates to produce them.

Usage
-----
    python Mapping/evaluate.py \\
        --renders-root ../data/renders \\
        --objects-root ../data/objects \\
        --symmetry-type axis_sym \\
        --sizes 224 --lightings flat \\
        --method svd

    python Mapping/evaluate.py \\
        --renders-root ../data/renders \\
        --objects-root ../data/objects \\
        --symmetry-type plane_sym \\
        --sizes 224 448 --lightings flat brighter darker \\
        --method ransac_svd_sde \\
        --with-reference-metrics
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_common.naming import exp_filename
from pipeline_common.datasets import OBJECTS_SUBDIR, load_mesh, load_mesh_vertices

try:
    import gpytoolbox as gpy
    _HAS_GPYTOOLBOX = True
except ImportError:
    gpy = None
    _HAS_GPYTOOLBOX = False

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY_ASSIGNMENT = True
except ImportError:
    linear_sum_assignment = None
    _HAS_SCIPY_ASSIGNMENT = False

# ── Constants ─────────────────────────────────────────────────────────────────

PREDICTED_FILE = "predicted_symmetry.json"

METHODS = ("svd", "ransac_svd", "svd_sde", "ransac_svd_sde", "triangulation", "triangulation_multiplane")

# Thresholds
ANGULAR_THRESHOLDS = [5, 10, 15]           # degrees

# AUC integration range
AUC_ANGULAR_MAX = 45.0   # degrees — beyond this is considered total failure

# ── Reference-metrics constants (formerly Mapping/reference_metrics.py) ───────
# THRESHOLDS_INLIER / N_SAMPLES_DEFAULT keep their original names so any code
# still doing `from reference_metrics import THRESHOLDS_INLIER` can switch to
# `from evaluate import THRESHOLDS_INLIER` with no other change.

THRESHOLDS_INLIER = [0.05, 0.1, 0.15, 0.2]   # metric_F1.py's set_threshold
N_SAMPLES_DEFAULT = 1000                      # metric_SDE.py's sample count
SDE_REF_SEED_DEFAULT = 0                      # reproducible surface sampling


# ── Experiment filename ───────────────────────────────────────────────────────

def experiment_suffix(sizes: list[int], lightings: list[str]) -> str:
    return "s" + "_".join(str(s) for s in sizes) + "_" + "_".join(lightings)


# ── True label parser ─────────────────────────────────────────────────────────

def parse_true_label(txt_path: Path) -> dict:
    """
    Parse a symmetry .txt label file.
    Normalizes direction/normal vectors on load.
    Returns {"type": "axis"|"plane", "elements": [...]}
    """
    lines    = [l.strip() for l in txt_path.read_text().splitlines() if l.strip()]
    elements = []
    sym_type = None

    for line in lines:
        if line.startswith("axis"):
            sym_type = "axis"
            parts    = line.split()
            vec      = np.array([float(x) for x in parts[1:4]])
            orig     = [float(x) for x in parts[4:7]]
            vec     /= np.linalg.norm(vec)
            elements.append({"direction": vec.tolist(), "origin": orig})
        elif line.startswith("plane"):
            sym_type = "plane"
            parts    = line.split()
            vec      = np.array([float(x) for x in parts[1:4]])
            orig     = [float(x) for x in parts[4:7]]
            vec     /= np.linalg.norm(vec)
            elements.append({"normal": vec.tolist(), "origin": orig})

    return {"type": sym_type, "elements": elements}


def bbox_diagonal(vertices: np.ndarray) -> float:
    return float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))


# ── Geometric metrics ─────────────────────────────────────────────────────────

def angular_error_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in [0, 90] degrees, sign-agnostic."""
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    return float(np.degrees(np.arccos(np.clip(np.abs(np.dot(v1, v2)), 0.0, 1.0))))


def point_to_line_distance(point: np.ndarray,
                            line_origin: np.ndarray,
                            line_dir: np.ndarray) -> float:
    """Distance from point to 3D line (origin + direction)."""
    d = line_dir / np.linalg.norm(line_dir)
    v = point - line_origin
    return float(np.linalg.norm(v - np.dot(v, d) * d))


def point_to_plane_distance(point: np.ndarray,
                             plane_origin: np.ndarray,
                             plane_normal: np.ndarray) -> float:
    """Absolute distance from point to plane."""
    n = plane_normal / np.linalg.norm(plane_normal)
    return float(abs(np.dot(point - plane_origin, n)))


# ── AUC ───────────────────────────────────────────────────────────────────────

def auc_from_errors(errors: list[float],
                    max_error: float,
                    n_steps: int = 100) -> float:
    """
    AUC of the precision-vs-threshold curve, normalized to [0, 1].
    Precision at threshold t = fraction of objects with error < t.
    """
    thresholds = np.linspace(0, max_error, n_steps + 1)
    arr        = np.array(errors)
    precisions = [(arr < t).mean() for t in thresholds]
    # np.trapezoid solo existe desde NumPy 2.0; np.trapz es su predecesor
    # (deprecado pero presente en 1.x). Usamos el que esté disponible.
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz_fn(precisions, thresholds) / max_error)


# ── Reference metrics: SDE_ref (formerly reference_metrics.py::calplaneloss/calaxisloss) ──
# Verbatim from the reference scripts (metric_SDE.py) — do not "clean up",
# the point is that the numbers match the reference paper's exactly, quirks
# included. Requires gpytoolbox; callers must check _HAS_GPYTOOLBOX first.

def calplaneloss(plane: np.ndarray, vertices: np.ndarray, faces: np.ndarray, points: np.ndarray) -> float:
    points = np.hstack((points, np.ones((points.shape[0], 1))))
    lam = points.dot(plane.T)
    planepoints = points - 2 * lam * plane
    d, ind, b = gpy.squared_distance(planepoints[:, 0:3], vertices, faces, use_aabb=True, use_cpp=True)
    return float(np.mean(d))


# Axis SDE_ref (our own extension, NOT ported from the reference repo — that
# repo has no ground-truth-matching evaluation for axes at all, confirmed by
# reading it: its evaluation/ only has a self-supervised DINOv2-feature-
# invariance proxy for axes). Applies the same convention as calplaneloss
# (area-weighted surface sample, squared distance to the real mesh surface via
# AABB, unnormalized) to axis reflection (180° rotation about the line)
# instead of planar reflection. The reflection formula itself matches
# estimate_symmetry.py::sde_axis exactly (same geometry: foot of perpendicular
# on the axis, then 2*proj - point).

def calaxisloss(axis_dir: np.ndarray, axis_origin: np.ndarray,
                 vertices: np.ndarray, faces: np.ndarray, points: np.ndarray) -> float:
    t = (points - axis_origin) @ axis_dir
    proj = axis_origin + t[:, None] * axis_dir
    reflected = 2.0 * proj - points
    d, ind, b = gpy.squared_distance(reflected, vertices, faces, use_aabb=True, use_cpp=True)
    return float(np.mean(d))


def sample_surface_points(mesh, n_samples: int = N_SAMPLES_DEFAULT,
                           seed: int | None = SDE_REF_SEED_DEFAULT):
    """
    Area-weighted surface sample used by SDE_ref (both axis and plane).
    Returns (vertices, faces, sample_points). Requires gpytoolbox.
    """
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    if seed is not None:
        np.random.seed(seed)  # gpytoolbox's sampler draws from the numpy global RNG
    sample = gpy.random_points_on_mesh(v, f, n_samples)
    return v, f, sample


# ── Reference metrics: F1_ref (formerly reference_metrics.py) ─────────────────

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


def f1_match_counts(predicted: list[np.ndarray], gt_planes: list[np.ndarray],
                     threshold_inlier: float) -> tuple[int, int, int]:
    """
    Verbatim port of the matching loop inside the reference paper's
    metric_F1.py::f1_score_calc — greedy, in list order, not an optimal
    assignment (see f1_match_counts_hungarian for that). `predicted` is a
    list of candidate planes; today this pipeline only ever fits one plane
    per (object, method, n_views), so it's always called with a single-
    element list, but the function itself makes no such assumption — pass in
    as many candidates as a future multi-plane detector produces.
    """
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


def f1_match_counts_hungarian(predicted: list[np.ndarray], gt_planes: list[np.ndarray],
                               threshold_inlier: float) -> tuple[int, int, int]:
    """
    Complementary variant of f1_match_counts: optimal (minimum-cost) bipartite
    assignment between predicted and GT planes instead of greedy-by-list-order,
    matching the convention used by more recent multi-candidate symmetry
    benchmarks (Reflect3D/CVPR 2025, ArchSym/2026) rather than PRS-Net's
    original greedy loop. With a single predicted plane (this pipeline's
    current output) this is mathematically identical to f1_match_counts —
    they only diverge once multiple candidate planes are predicted per object.
    Requires scipy; callers must check _HAS_SCIPY_ASSIGNMENT first.
    """
    n_pred, n_gt = len(predicted), len(gt_planes)
    if n_pred == 0 or n_gt == 0:
        return 0, n_pred, n_gt

    cost = np.zeros((n_pred, n_gt))
    for i, pred_plane in enumerate(predicted):
        for j, gt in enumerate(gt_planes):
            cost[i, j] = min(np.linalg.norm(pred_plane - gt), np.linalg.norm(pred_plane + gt))

    row_idx, col_idx = linear_sum_assignment(cost)
    tp = int(np.sum(cost[row_idx, col_idx] < threshold_inlier))
    fp = n_pred - tp
    fn = n_gt - tp
    return tp, fp, fn


def _f1_from_counts_by_threshold(counts_by_t: dict[float, tuple[int, int, int]]) -> float:
    """
    Mean F1 across THRESHOLDS_INLIER, given (tp, fp, fn) per threshold.

    IMPORTANT: the reference paper's F1_ref accumulates TP/FP/FN GLOBALLY
    across the whole dataset per threshold, then computes one F1 per
    threshold from those totals — it is NOT the mean of a per-object F1
    (those give different numbers, especially when n_true_planes varies
    across objects). Always feed this function dataset-level accumulated
    counts (see compute_summary), never call it on a single object's counts
    and treat the result as "the" F1_ref.
    """
    f1_per_t = []
    for tp, fp, fn in counts_by_t.values():
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_per_t.append(f1)
    return float(np.mean(f1_per_t))


# ── Multi-plane metrics (scaffolding — see docs/actualizacion_metricas.md §3.3) ─
# NOT wired into evaluate_object/compute_summary yet: today's
# predicted_symmetry.json stores exactly one plane per (object, method,
# n_views). Call this directly once a multi-plane detector (up to 3 planes,
# matching the curated dataset's 1/2/3-plane objects) produces a LIST of
# candidate planes per object — it fills the gap noted in
# docs/verificacion_metricas_literatura.md B.1 ("no hay concepto de recall
# sobre el conjunto completo de simetrías"), using the project's own angular
# convention (ANGULAR_THRESHOLDS) instead of F1_ref's plane-vector distance.

def evaluate_plane_multiset(pred_planes: list[dict], true_elements: list[dict],
                             angular_threshold_deg: float = 15.0) -> dict:
    """
    Greedy best-match recall/precision over a SET of predicted planes against
    the FULL set of GT planes for an object (unlike evaluate_plane, which only
    ever reports the single best match and never penalizes un-found GT planes).

    Args:
    - pred_planes: list of {"normal": [...], "origin": [...]} candidates
    - true_elements: GT planes for this object (label["elements"])
    - angular_threshold_deg: a predicted/GT pair counts as matched if their
      angular_error_deg is below this (reuses the project's own convention,
      default aligned with ANGULAR_THRESHOLDS[-1])

    Returns: {"n_planes_predicted", "n_true_planes", "n_planes_matched",
              "recall_planes", "precision_planes"}
    """
    n_pred = len(pred_planes)
    n_true = len(true_elements)
    matched_gt: set[int] = set()

    for pred in pred_planes:
        p_normal = np.array(pred["normal"])
        best_idx, best_ang = -1, float("inf")
        for idx, true in enumerate(true_elements):
            if idx in matched_gt:
                continue
            ang = angular_error_deg(p_normal, np.array(true["normal"]))
            if ang < best_ang:
                best_idx, best_ang = idx, ang
        if best_idx >= 0 and best_ang < angular_threshold_deg:
            matched_gt.add(best_idx)

    n_matched = len(matched_gt)
    return {
        "n_planes_predicted": n_pred,
        "n_true_planes":      n_true,
        "n_planes_matched":   n_matched,
        "recall_planes":      round(n_matched / n_true, 4) if n_true else None,
        "precision_planes":   round(n_matched / n_pred, 4) if n_pred else None,
    }


def evaluate_plane_multi_from_pred(pred_planes: list[dict], true_elements: list[dict],
                                    angular_threshold_deg: float = ANGULAR_THRESHOLDS[-1],
                                    ref_ctx: dict | None = None) -> dict:
    """
    Per-object wrapper around evaluate_plane_multiset for evaluate_object's
    dispatch: adds the "status" field the rest of the pipeline
    (compute_summary) checks on every per-object metrics dict, and an
    "n_points" field for parity with evaluate_axis/evaluate_plane (sum of
    n_views_used across the predicted planes -- there's no single "n_points"
    for a whole set, this is the closest sensible analog).

    angular_threshold_deg defaults to ANGULAR_THRESHOLDS[-1] (15°, the most
    permissive of the project's own thresholds) rather than
    evaluate_plane_multiset's own 15° default reused verbatim -- same value
    today, but ties this wrapper's default to the project's threshold
    convention instead of duplicating a magic number.

    Reference metrics (SDE_ref/F1_ref, --with-reference-metrics): this was
    already validated in test_pipeline_sin_malla.ipynb's own evaluation
    section for exactly this multi-plane case -- calplaneloss and
    f1_match_counts/f1_match_counts_hungarian already accept a LIST of
    predicted planes (they were designed for the multi-candidate case from
    the metrics refactor onward, see docs/actualizacion_metricas.md), so this
    just calls them with pred_planes as-is instead of a single-element list:
      - sde_ref_per_plane: one calplaneloss per predicted plane (there's no
        single "the" SDE_ref for a whole set, unlike F1 which is inherently
        a set-level metric).
      - f1_counts_ref/f1_counts_ref_hungarian: raw per-threshold (tp, fp, fn)
        counts over the FULL predicted-plane list vs. the FULL GT list, same
        dataset-level-accumulation convention compute_summary already uses
        for the single-plane case (see _f1_from_counts_by_threshold).
    """
    m = evaluate_plane_multiset(pred_planes, true_elements, angular_threshold_deg)
    m["status"]   = "ok"
    m["n_points"] = sum(p.get("n_views_used", 0) for p in pred_planes)

    if ref_ctx is not None and pred_planes:
        mesh_v, mesh_f, sample = ref_ctx["mesh_v"], ref_ctx["mesh_f"], ref_ctx["sample"]
        pred_planes_ref = [normal_origin_to_plane(p["normal"], p["origin"]) for p in pred_planes]

        m["sde_ref_per_plane"] = [
            round(calplaneloss(pp, mesh_v, mesh_f, sample), 8) for pp in pred_planes_ref
        ]

        gt_planes = ref_ctx.get("gt_planes_ref") or []
        if gt_planes:
            m["f1_counts_ref"] = {
                str(t): list(f1_match_counts(pred_planes_ref, gt_planes, t))
                for t in THRESHOLDS_INLIER
            }
            if _HAS_SCIPY_ASSIGNMENT:
                m["f1_counts_ref_hungarian"] = {
                    str(t): list(f1_match_counts_hungarian(pred_planes_ref, gt_planes, t))
                    for t in THRESHOLDS_INLIER
                }

    return m


# ── Per-object evaluation ─────────────────────────────────────────────────────

def evaluate_axis(pred: dict, true_elements: list[dict],
                   vertices: np.ndarray | None = None,
                   ref_ctx: dict | None = None) -> dict:
    true      = true_elements[0]
    t_dir     = np.array(true["direction"])
    t_orig    = np.array(true["origin"])
    p_dir     = np.array(pred["direction"])
    p_orig    = np.array(pred["origin"])

    ang_err   = angular_error_deg(p_dir, t_dir)
    trans_err = point_to_line_distance(p_orig, t_orig, t_dir)

    m = {
        "angular_error_deg":  round(ang_err,   4),
        "translation_error":  round(trans_err, 6),
        "n_points":           pred.get("n_points", 0),
        "status":             "ok",
    }
    if vertices is not None:
        diag = bbox_diagonal(vertices)
        m["translation_error_normalized"] = round(trans_err / diag, 6) if diag > 0 else None
    else:
        m["translation_error_normalized"] = None

    for t in ANGULAR_THRESHOLDS:
        m[f"precision_{t}deg"] = 1 if ang_err < t else 0

    if ref_ctx is not None:
        mesh_v, mesh_f, sample = ref_ctx["mesh_v"], ref_ctx["mesh_f"], ref_ctx["sample"]
        m["sde_ref"] = round(calaxisloss(p_dir, p_orig, mesh_v, mesh_f, sample), 8)

    return m


def evaluate_plane(pred: dict,
                   true_elements: list[dict],
                   vertices: np.ndarray | None,
                   ref_ctx: dict | None = None) -> dict:
    p_normal = np.array(pred["normal"])
    p_origin = np.array(pred["origin"])

    # Best-match true plane
    best_ang, best_idx, best_dist = float("inf"), -1, float("inf")
    for idx, true in enumerate(true_elements):
        ang  = angular_error_deg(p_normal, np.array(true["normal"]))
        dist = point_to_plane_distance(p_origin,
                                        np.array(true["origin"]),
                                        np.array(true["normal"]))
        if ang < best_ang:
            best_ang, best_idx, best_dist = ang, idx, dist

    m = {
        "angular_error_deg":  round(best_ang,  4),
        "translation_error":  round(best_dist, 6),
        "matched_true_plane": best_idx,
        "n_true_planes":      len(true_elements),
        "n_points":           pred.get("n_points", 0),
        "status":             "ok",
    }
    if vertices is not None:
        diag = bbox_diagonal(vertices)
        m["translation_error_normalized"] = round(best_dist / diag, 6) if diag > 0 else None
    else:
        m["translation_error_normalized"] = None

    for t in ANGULAR_THRESHOLDS:
        m[f"precision_{t}deg"] = 1 if best_ang < t else 0

    if ref_ctx is not None:
        mesh_v, mesh_f, sample = ref_ctx["mesh_v"], ref_ctx["mesh_f"], ref_ctx["sample"]
        pred_plane = normal_origin_to_plane(pred["normal"], pred["origin"])
        m["sde_ref"] = round(calplaneloss(pred_plane, mesh_v, mesh_f, sample), 8)

        gt_planes = ref_ctx.get("gt_planes_ref") or []
        if gt_planes:
            # Raw per-threshold (tp, fp, fn) counts, NOT a per-object F1 —
            # F1_ref is a dataset-level metric (TP/FP/FN accumulated across
            # ALL objects before computing F1 per threshold, see
            # _f1_from_counts_by_threshold docstring); compute_summary does
            # that accumulation. Lists (not tuples) so this round-trips
            # through json.dump/json.load unchanged.
            m["f1_counts_ref"] = {
                str(t): list(f1_match_counts([pred_plane], gt_planes, t))
                for t in THRESHOLDS_INLIER
            }
            if _HAS_SCIPY_ASSIGNMENT:
                m["f1_counts_ref_hungarian"] = {
                    str(t): list(f1_match_counts_hungarian([pred_plane], gt_planes, t))
                    for t in THRESHOLDS_INLIER
                }

    return m


def evaluate_object(object_dir: Path,
                    true_label: dict,
                    symmetry_type: str,
                    vertices: np.ndarray | None,
                    method: str,
                    predicted_file: str = PREDICTED_FILE,
                    ref_ctx: dict | None = None) -> dict | None:
    pred_path = object_dir / predicted_file
    if not pred_path.exists():
        return None

    with open(pred_path, encoding="utf-8") as f:
        predicted = json.load(f)

    results = {}
    for n_views_key, method_preds in predicted.get("n_views_predictions", {}).items():
        pred = method_preds.get(method) if isinstance(method_preds, dict) else None
        if pred is None:
            results[n_views_key] = {"status": f"method '{method}' not found"}
            continue
        try:
            if symmetry_type == "axis_sym":
                metrics = evaluate_axis(pred, true_label["elements"], vertices, ref_ctx)
            elif isinstance(pred, dict) and "planes" in pred:
                # Multi-plane prediction (Mapping/estimate_symmetry_no_mesh.py
                # --max-planes > 1, key "triangulation_multiplane") -- recall/
                # precision over the FULL set of GT planes, not a best-match
                # angular error. See docs/actualizacion_metricas.md S3.3 and
                # docs/implementacion_pipeline_sin_malla.md S3 Fase 2.
                metrics = evaluate_plane_multi_from_pred(pred["planes"], true_label["elements"], ref_ctx=ref_ctx)
            else:
                metrics = evaluate_plane(pred, true_label["elements"], vertices, ref_ctx)
        except Exception as e:
            metrics = {"status": f"error: {e}"}
        results[n_views_key] = metrics

    return results


# ── Summary ───────────────────────────────────────────────────────────────────

def compute_summary(all_results: dict, symmetry_type: str,
                    n_total: int | None = None) -> dict[str, dict]:
    """
    Aggregate per-object results into per-n_views summary statistics.

    Angular metrics (angular_error, precision@k, AUC) are computed over ALL
    n_total objects: objects without a valid prediction contribute angular_error=90°
    and precision=0, reflecting total failure rather than being silently excluded.

    Translation and SDE_ref metrics are computed only over objects with valid
    predictions — there is no principled worst-case value to impute.

    F1_ref/f1_ref_hungarian follow the reference paper's own convention:
    TP/FP/FN are accumulated GLOBALLY across every object in the dataset
    (per n_views, per threshold) before computing F1 — NOT averaged from a
    per-object F1. See evaluate_plane's f1_counts_ref field and
    _f1_from_counts_by_threshold's docstring for why this matters.

    n_total defaults to len(all_results) when not provided.
    """
    _n_total = n_total if n_total is not None else len(all_results)

    # Collect valid-prediction data per n_views group
    grouped: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    # F1_ref: dataset-level TP/FP/FN accumulation per n_views, per threshold,
    # per matching strategy — kept separate from `grouped` since it isn't a
    # simple list-of-values-to-average like everything else here.
    f1_totals: dict[str, dict[str, dict[float, list[int]]]] = defaultdict(
        lambda: {"greedy": {t: [0, 0, 0] for t in THRESHOLDS_INLIER},
                 "hungarian": {t: [0, 0, 0] for t in THRESHOLDS_INLIER}}
    )

    # Also discover every n_views key seen across any object
    all_nv: set[str] = set()

    for obj_results in all_results.values():
        if obj_results is None:
            continue
        all_nv.update(obj_results.keys())
        for nv, m in obj_results.items():
            if m.get("status") != "ok":
                continue
            g = grouped[nv]

            if "n_planes_predicted" in m:
                # Multi-plane result (evaluate_plane_multi_from_pred) -- a
                # completely different metric shape from everything below
                # (recall/precision over the GT plane SET, no single
                # angular_error/translation_error to speak of). Aggregated
                # separately in the per-nv loop further down.
                g["n_planes_predicted"].append(m["n_planes_predicted"])
                g["n_true_planes"].append(m["n_true_planes"])
                g["n_planes_matched"].append(m["n_planes_matched"])
                if m.get("recall_planes") is not None:
                    g["recall_planes"].append(m["recall_planes"])
                if m.get("precision_planes") is not None:
                    g["precision_planes"].append(m["precision_planes"])

                # SDE_ref/F1_ref (--with-reference-metrics) DO apply to
                # multi-plane predictions too -- see
                # evaluate_plane_multi_from_pred. sde_ref is inherently
                # per-plane (no single "the" SDE_ref for a whole set), so
                # every plane's score is pooled into the same "sde_ref" list
                # the single-plane branch below also feeds -- compute_summary
                # doesn't care which mode contributed a value, it's the same
                # mean/min/max either way. F1_ref reuses the exact same
                # dataset-level accumulation (f1_totals) as the single-plane
                # branch, unconditionally on the mode.
                if m.get("sde_ref_per_plane"):
                    g["sde_ref"].extend(m["sde_ref_per_plane"])
                if m.get("f1_counts_ref") is not None:
                    totals = f1_totals[nv]["greedy"]
                    for t_str, (tp, fp, fn) in m["f1_counts_ref"].items():
                        t = float(t_str)
                        totals[t][0] += tp; totals[t][1] += fp; totals[t][2] += fn
                if m.get("f1_counts_ref_hungarian") is not None:
                    totals_h = f1_totals[nv]["hungarian"]
                    for t_str, (tp, fp, fn) in m["f1_counts_ref_hungarian"].items():
                        t = float(t_str)
                        totals_h[t][0] += tp; totals_h[t][1] += fp; totals_h[t][2] += fn
                continue

            g["angular_error_deg"].append(m["angular_error_deg"])
            g["translation_error"].append(m["translation_error"])
            if m.get("translation_error_normalized") is not None:
                g["translation_error_normalized"].append(m["translation_error_normalized"])
            g["n_points"].append(m.get("n_points", 0))
            for t in ANGULAR_THRESHOLDS:
                g[f"precision_{t}deg"].append(m.get(f"precision_{t}deg", 0))
            if m.get("sde_ref") is not None:
                g["sde_ref"].append(m["sde_ref"])

            if m.get("f1_counts_ref") is not None:
                totals = f1_totals[nv]["greedy"]
                for t_str, (tp, fp, fn) in m["f1_counts_ref"].items():
                    t = float(t_str)
                    totals[t][0] += tp; totals[t][1] += fp; totals[t][2] += fn
            if m.get("f1_counts_ref_hungarian") is not None:
                totals_h = f1_totals[nv]["hungarian"]
                for t_str, (tp, fp, fn) in m["f1_counts_ref_hungarian"].items():
                    t = float(t_str)
                    totals_h[t][0] += tp; totals_h[t][1] += fp; totals_h[t][2] += fn

    summary = {}
    for nv in sorted(all_nv, key=int):
        data = grouped.get(nv, defaultdict(list))

        if data.get("n_planes_predicted"):
            # Multi-plane summary: entirely different shape from the regular
            # (axis / single-plane) branch below -- no angular_error/AUC/
            # precision@theta here, see evaluate_plane_multi_from_pred.
            n_valid = len(data["n_planes_predicted"])
            s: dict = {
                "n_total":                 _n_total,
                "n_objects":               n_valid,
                "n_planes_predicted_mean": round(float(np.mean(data["n_planes_predicted"])), 4),
                "n_true_planes_mean":      round(float(np.mean(data["n_true_planes"])), 4),
                "n_planes_matched_mean":   round(float(np.mean(data["n_planes_matched"])), 4),
                "recall_planes_mean":      round(float(np.mean(data["recall_planes"])), 4) if data.get("recall_planes") else None,
                "precision_planes_mean":   round(float(np.mean(data["precision_planes"])), 4) if data.get("precision_planes") else None,
            }

            # SDE_ref/F1_ref (--with-reference-metrics), same conventions as
            # the single-plane branch below: sde_ref is a flat pool of
            # per-plane scores (mean/min/max); F1_ref/f1_ref_hungarian come
            # from the dataset-level TP/FP/FN accumulation in f1_totals.
            if data.get("sde_ref"):
                sde_ref_arr = np.array(data["sde_ref"])
                s["sde_ref_mean"] = round(float(sde_ref_arr.mean()), 8)
                s["sde_ref_min"]  = round(float(sde_ref_arr.min()),  8)
                s["sde_ref_max"]  = round(float(sde_ref_arr.max()),  8)
            if nv in f1_totals:
                greedy_counts = f1_totals[nv]["greedy"]
                if any(sum(c) > 0 for c in greedy_counts.values()):
                    s["f1_ref"] = round(_f1_from_counts_by_threshold(greedy_counts), 4)
                hungarian_counts = f1_totals[nv]["hungarian"]
                if any(sum(c) > 0 for c in hungarian_counts.values()):
                    s["f1_ref_hungarian"] = round(_f1_from_counts_by_threshold(hungarian_counts), 4)

            summary[nv] = s
            continue

        n_valid = len(data.get("angular_error_deg", []))
        n_no_pred = _n_total - n_valid   # objects that failed or were never processed

        # ── Angular metrics: all N objects (no-pred → 90°) ────────────────────
        all_ang  = data.get("angular_error_deg", []) + [90.0] * n_no_pred
        ang_arr  = np.array(all_ang)

        s: dict = {
            "n_total":              _n_total,
            "n_objects":            n_valid,   # kept for backward compat
            "angular_error_mean":   round(float(ang_arr.mean()),     4),
            "angular_error_median": round(float(np.median(ang_arr)), 4),
            "angular_error_std":    round(float(ang_arr.std()),      4),
            "auc_angular":          round(auc_from_errors(all_ang, AUC_ANGULAR_MAX), 4),
        }

        for t in ANGULAR_THRESHOLDS:
            valid_p = data.get(f"precision_{t}deg", [])
            all_p   = valid_p + [0] * n_no_pred
            s[f"precision_{t}deg"] = round(float(np.mean(all_p)), 4)

        # ── Translation metrics: valid objects only (no imputation) ───────────
        dist = data.get("translation_error", [])
        if dist:
            dist_arr = np.array(dist)
            s["translation_error_mean"]   = round(float(dist_arr.mean()),       6)
            s["translation_error_median"] = round(float(np.median(dist_arr)),   6)
            s["translation_error_std"]    = round(float(dist_arr.std()),        6)
        else:
            s["translation_error_mean"]   = None
            s["translation_error_median"] = None
            s["translation_error_std"]    = None

        dist_norm = data.get("translation_error_normalized", [])
        if dist_norm:
            dn_arr = np.array(dist_norm)
            s["translation_error_normalized_mean"]   = round(float(dn_arr.mean()),     6)
            s["translation_error_normalized_median"] = round(float(np.median(dn_arr)), 6)
        else:
            s["translation_error_normalized_mean"]   = None
            s["translation_error_normalized_median"] = None

        pts = data.get("n_points", [])
        s["n_points_mean"] = round(float(np.mean(pts)), 2) if pts else 0.0

        # ── Reference metrics: valid objects only (opt-in, see --with-reference-metrics) ──
        if data.get("sde_ref"):
            sde_ref_arr = np.array(data["sde_ref"])
            s["sde_ref_mean"] = round(float(sde_ref_arr.mean()), 8)
            s["sde_ref_min"]  = round(float(sde_ref_arr.min()),  8)
            s["sde_ref_max"]  = round(float(sde_ref_arr.max()),  8)

        # F1_ref: dataset-level accumulation (see compute_summary docstring),
        # NOT a mean of per-object values.
        if symmetry_type == "plane_sym" and nv in f1_totals:
            greedy_counts = f1_totals[nv]["greedy"]
            if any(sum(c) > 0 for c in greedy_counts.values()):
                s["f1_ref"] = round(_f1_from_counts_by_threshold(greedy_counts), 4)
            hungarian_counts = f1_totals[nv]["hungarian"]
            if any(sum(c) > 0 for c in hungarian_counts.values()):
                s["f1_ref_hungarian"] = round(_f1_from_counts_by_threshold(hungarian_counts), 4)

        summary[nv] = s
    return summary


def write_csv(summary: dict, csv_path: Path, symmetry_type: str,
              with_reference_metrics: bool = False) -> None:
    is_multiplane = any("n_planes_predicted_mean" in s for s in summary.values())

    if is_multiplane:
        # Entirely different metric shape (see compute_summary) -- no
        # angular_error/AUC/precision@theta columns here. SDE_ref/F1_ref DO
        # apply (evaluate_plane_multi_from_pred computes them over the full
        # predicted-plane list), same --with-reference-metrics gate as the
        # single-plane branch.
        fieldnames = [
            "n_views", "n_total", "n_objects",
            "n_planes_predicted_mean", "n_true_planes_mean",
            "n_planes_matched_mean", "recall_planes_mean", "precision_planes_mean",
        ]
        if with_reference_metrics:
            fieldnames += ["sde_ref_mean", "sde_ref_min", "sde_ref_max", "f1_ref", "f1_ref_hungarian"]
    else:
        base = [
            "n_views", "n_total", "n_objects",
            "angular_error_mean", "angular_error_median", "angular_error_std",
            "translation_error_mean", "translation_error_median", "translation_error_std",
            "translation_error_normalized_mean", "translation_error_normalized_median",
            "auc_angular", "n_points_mean",
        ] + [f"precision_{t}deg" for t in ANGULAR_THRESHOLDS]

        ref_extra = []
        if with_reference_metrics:
            ref_extra = ["sde_ref_mean", "sde_ref_min", "sde_ref_max"]
            if symmetry_type == "plane_sym":
                ref_extra += ["f1_ref", "f1_ref_hungarian"]

        fieldnames = base + ref_extra

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore",
                                restval="")
        writer.writeheader()
        for nv, stats in summary.items():
            writer.writerow({"n_views": nv, **stats})


# ── Bulk re-scoring mode (formerly reference_metrics.py's --all) ──────────────
# Distinct from the per-run flow above: instead of evaluating ONE
# (experiment_id, method) combination against angular/translation GT and
# writing a full per-object results.json, this scans predicted_symmetry_*.json
# files ALREADY on disk for MANY experiment ids x methods at once and writes a
# single combined CSV of just SDE_ref/F1_ref (cheap re-scoring, no ray-casting,
# no SVD/RANSAC re-fit). Kept for docs/actualizacion_metricas.md's promise
# that nothing from reference_metrics.py would be lost in the merge —
# `ranking_postprocesamiento.ipynb` §10-11 reads exactly this CSV shape
# (reference_metrics_axis.csv / reference_metrics_plane.csv).

class _MeshCache:
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
            result = sample_surface_points(mesh, self.n_samples, self.seed)
        except Exception as e:
            print(f"  [warn] mesh load/sample failed for {object_id}: {e}")
            self._cache[object_id] = None
            return None
        self._cache[object_id] = result
        return result


def discover_experiment_ids(renders_root: Path, symmetry_type: str) -> list[str]:
    import re
    sym_dir = renders_root / symmetry_type
    found: set[str] = set()
    for f in sym_dir.glob("*/predicted_symmetry_*.json"):
        m = re.match(r"predicted_symmetry_(.+)\.json$", f.name)
        if m:
            found.add(m.group(1))
    return sorted(found)


def score_experiment_bulk(
    renders_root: Path, objects_dir: Path, experiment_id: str,
    methods: list[str], mesh_cache: "_MeshCache", symmetry_type: str,
) -> list[dict]:
    is_axis = symmetry_type == "axis_sym"
    vec_key = "direction" if is_axis else "normal"

    sym_dir = renders_root / symmetry_type
    predicted_file = exp_filename(PREDICTED_FILE, experiment_id)
    object_dirs = sorted(d for d in sym_dir.iterdir() if d.is_dir())

    sde_values: dict[tuple[str, str], list[float]] = {}
    f1_counts_greedy: dict[tuple[str, str], dict[float, list[int]]] = {}
    f1_counts_hungarian: dict[tuple[str, str], dict[float, list[int]]] = {}
    n_objects_seen: dict[tuple[str, str], int] = {}

    for obj_dir in object_dirs:
        object_id = obj_dir.name
        pred_path = obj_dir / predicted_file
        if not pred_path.exists():
            continue
        with open(pred_path, "r", encoding="utf-8") as fh:
            pred = json.load(fh)

        gt_planes = [] if is_axis else gt_planes_for_object(objects_dir, object_id)
        mesh_data = mesh_cache.get(object_id)

        for n_views_key, entry in pred.get("n_views_predictions", {}).items():
            # svd/svd_sde (and ransac_svd/ransac_svd_sde) always share the exact same
            # direction-or-normal/origin -- cache the (expensive) re-scoring per
            # distinct (vec, origin) seen for this object/n_views so duplicate
            # pairs are scored once, not twice.
            local_cache: dict[tuple, tuple] = {}

            for method in methods:
                m = entry.get(method)
                if m is None or vec_key not in m:
                    continue
                key = (method, n_views_key)
                n_objects_seen[key] = n_objects_seen.get(key, 0) + 1

                cache_key = (tuple(m[vec_key]), tuple(m["origin"]))
                if cache_key not in local_cache:
                    sde_val = None
                    f1g: dict[float, tuple[int, int, int]] = {}
                    f1h: dict[float, tuple[int, int, int]] = {}
                    if mesh_data is not None:
                        v, f, sample = mesh_data
                        if is_axis:
                            sde_val = calaxisloss(
                                np.asarray(m["direction"], dtype=np.float64),
                                np.asarray(m["origin"], dtype=np.float64), v, f, sample,
                            )
                        else:
                            pred_plane = normal_origin_to_plane(m["normal"], m["origin"])
                            sde_val = calplaneloss(pred_plane, v, f, sample)
                            f1g = {t: f1_match_counts([pred_plane], gt_planes, t) for t in THRESHOLDS_INLIER}
                            if _HAS_SCIPY_ASSIGNMENT:
                                f1h = {t: f1_match_counts_hungarian([pred_plane], gt_planes, t) for t in THRESHOLDS_INLIER}
                    local_cache[cache_key] = (sde_val, f1g, f1h)
                sde_val, f1g, f1h = local_cache[cache_key]

                if sde_val is not None:
                    sde_values.setdefault(key, []).append(sde_val)
                if not is_axis:
                    cg = f1_counts_greedy.setdefault(key, {t: [0, 0, 0] for t in THRESHOLDS_INLIER})
                    for t in THRESHOLDS_INLIER:
                        tp, fp, fn = f1g[t]
                        cg[t][0] += tp; cg[t][1] += fp; cg[t][2] += fn
                    if f1h:
                        ch = f1_counts_hungarian.setdefault(key, {t: [0, 0, 0] for t in THRESHOLDS_INLIER})
                        for t in THRESHOLDS_INLIER:
                            tp, fp, fn = f1h[t]
                            ch[t][0] += tp; ch[t][1] += fp; ch[t][2] += fn

    def _f1_mean(counts: dict[tuple[str, str], dict[float, list[int]]], key) -> float | None:
        if key not in counts:
            return None
        f1_per_t = []
        for t in THRESHOLDS_INLIER:
            tp, fp, fn = counts[key][t]
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall    = tp / (tp + fn) if (tp + fn) else 0.0
            f1_per_t.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
        return float(np.mean(f1_per_t))

    rows = []
    for key in sorted(set(sde_values) | set(f1_counts_greedy)):
        method, n_views_key = key
        sde_vals = np.array(sde_values.get(key, []))
        rows.append({
            "experiment":       experiment_id,
            "method":           method,
            "n_views":          n_views_key,
            "n_objects":        n_objects_seen[key],
            "sde_ref_mean":     float(sde_vals.mean()) if len(sde_vals) else None,
            "sde_ref_min":      float(sde_vals.min())  if len(sde_vals) else None,
            "sde_ref_max":      float(sde_vals.max())  if len(sde_vals) else None,
            "f1_ref":           _f1_mean(f1_counts_greedy, key),
            "f1_ref_hungarian": _f1_mean(f1_counts_hungarian, key),
        })
    return rows


def run_bulk_rescore(args: argparse.Namespace) -> None:
    """Entry point for the bulk re-scoring mode (--all or --experiment-ids)."""
    import time

    if not _HAS_GPYTOOLBOX:
        print("[error] bulk re-scoring requires gpytoolbox: pip install gpytoolbox", file=sys.stderr)
        sys.exit(1)

    renders_root = Path(args.renders_root)
    objects_dir  = Path(args.objects_root) / OBJECTS_SUBDIR[args.symmetry_type]
    ref_seed     = None if args.ref_seed == -1 else args.ref_seed
    out_path     = args.out or f"reference_metrics_{'axis' if args.symmetry_type == 'axis_sym' else 'plane'}.csv"

    if args.all:
        exp_ids = discover_experiment_ids(renders_root, args.symmetry_type)
        print(f"--all: descubiertos {len(exp_ids)} experiment id(s) con predicciones en "
              f"{renders_root / args.symmetry_type}")
    else:
        exp_ids = args.experiment_ids

    mesh_cache = _MeshCache(objects_dir, args.ref_n_samples, ref_seed)

    fieldnames = ["experiment", "method", "n_views", "n_objects",
                  "sde_ref_mean", "sde_ref_min", "sde_ref_max", "f1_ref", "f1_ref_hungarian"]
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
        rows = score_experiment_bulk(renders_root, objects_dir, exp_id, args.methods, mesh_cache, args.symmetry_type)
        if not rows:
            print("  (sin predicciones encontradas)")
        for r in rows:
            f1_str  = f"{r['f1_ref']:.4f}" if r["f1_ref"] is not None else "n/a"
            sde_str = f"{r['sde_ref_mean']:.6f}" if r["sde_ref_mean"] is not None else "n/a"
            print(f"  {r['method']:16s} n_views={r['n_views']:>3}  n_obj={r['n_objects']:>4}  "
                  f"SDE_ref(mean)={sde_str}  F1_ref={f1_str}")
            writer.writerow(r)
        csv_file.flush()
        n_total_rows += len(rows)

    csv_file.close()
    total_min = (time.time() - t0) / 60
    print(f"\nGuardado: {out_path}  ({n_total_rows} filas, {len(exp_ids)} experimentos, {total_min:.1f} min)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate predicted symmetry vs true labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--renders-root",  required=True)
    p.add_argument("--objects-root",  required=True)
    p.add_argument("--symmetry-type", required=True,
                   choices=["axis_sym", "plane_sym"])
    p.add_argument("--sizes",     type=int, nargs="+", default=[224, 448, 1136],
                   help="Sizes used in the experiment (affects output filename)")
    p.add_argument("--lightings", type=str, nargs="+",
                   default=["flat", "brighter", "darker"],
                   choices=["flat", "darker", "brighter"],
                   help="Lightings used in the experiment (affects output filename)")
    p.add_argument("--experiment-id", default=None,
                   help=(
                       "Experiment identifier. Reads predicted_symmetry_<ID>.json and "
                       "writes eval_*_<ID>_results.json. Must match the --experiment-id "
                       "used in estimate_symmetry.py."
                   ))
    p.add_argument("--method", default=None, choices=METHODS,
                   help="Which estimation method to evaluate. Required unless --all/"
                        "--experiment-ids (bulk re-scoring mode) is used.")
    p.add_argument("--max-objects", type=int, default=None,
                   help="Limit to the first N objects (sorted order).")
    p.add_argument("--with-reference-metrics", action="store_true",
                   help=(
                       "Also compute SDE_ref (both symmetry types) and F1_ref/"
                       "f1_ref_hungarian (plane_sym only) — the exact formulas from "
                       "an external reference paper (formerly the standalone "
                       "reference_metrics.py). Opt-in: needs gpytoolbox and is "
                       "meaningfully more expensive (mesh + AABB tree + surface "
                       "sampling per object) — point this at a short list of "
                       "'winner' experiments, not the full sweep, same reasoning "
                       "the old standalone script had."
                   ))
    p.add_argument("--ref-n-samples", type=int, default=N_SAMPLES_DEFAULT,
                   help="Surface points sampled per object for SDE_ref (matches the reference's 1000).")
    p.add_argument("--ref-seed", type=int, default=SDE_REF_SEED_DEFAULT,
                   help="Seed for SDE_ref surface sampling. Pass -1 to leave unseeded.")

    # Bulk re-scoring mode (formerly reference_metrics.py's --all): cheap
    # SDE_ref/F1_ref-only re-scoring across MANY already-fitted experiments/
    # methods at once, into a single combined CSV — no ray-casting, no SVD/
    # RANSAC re-fit, no per-object results.json. Mutually exclusive with the
    # normal per-run flow above (--method/--sizes/--lightings are ignored here).
    bulk = p.add_argument_group("bulk re-scoring mode (SDE_ref/F1_ref only, formerly reference_metrics.py --all)")
    bulk.add_argument("--experiment-ids", nargs="+", default=None, metavar="EXP_ID",
                       help="One or more experiment ids to re-score in bulk mode. "
                            "Pass only your ranking 'winners' -- meant to be cheap on a "
                            "handful of experiments, not the full sweep.")
    bulk.add_argument("--all", action="store_true",
                       help="Bulk mode: discover and re-score every experiment id that has "
                            "predicted_symmetry_*.json on disk for this --symmetry-type. "
                            "Overrides --experiment-ids. Several hours over the full sweep.")
    bulk.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS,
                       help="Bulk mode only: which methods to re-score (default: all 4).")
    bulk.add_argument("--out", default=None,
                       help="Bulk mode only: output CSV path. Default: "
                            "reference_metrics_plane.csv or reference_metrics_axis.csv.")
    return p.parse_args()


def _build_ref_ctx(mesh, objects_dir: Path, object_id: str, symmetry_type: str,
                    n_samples: int, seed: int | None) -> dict:
    mesh_v, mesh_f, sample = sample_surface_points(mesh, n_samples, seed)
    ctx = {"mesh_v": mesh_v, "mesh_f": mesh_f, "sample": sample}
    if symmetry_type == "plane_sym":
        ctx["gt_planes_ref"] = gt_planes_for_object(objects_dir, object_id)
    return ctx


def main() -> None:
    args = parse_args()

    if args.all or args.experiment_ids:
        run_bulk_rescore(args)
        return

    if args.method is None:
        print("[error] --method es obligatorio salvo que uses el modo bulk "
              "(--all o --experiment-ids)", file=sys.stderr)
        sys.exit(1)

    if args.with_reference_metrics and not _HAS_GPYTOOLBOX:
        print("[error] --with-reference-metrics requires gpytoolbox: pip install gpytoolbox", file=sys.stderr)
        sys.exit(1)
    if args.with_reference_metrics and args.symmetry_type == "plane_sym" and not _HAS_SCIPY_ASSIGNMENT:
        print("[warn] scipy not available — f1_ref_hungarian will be skipped, f1_ref (greedy) still computed.")

    symmetry_dir   = Path(args.renders_root) / args.symmetry_type
    objects_subdir = OBJECTS_SUBDIR[args.symmetry_type]
    objects_dir    = Path(args.objects_root) / objects_subdir

    if not symmetry_dir.exists():
        print(f"[error] Not found: {symmetry_dir}")
        sys.exit(1)

    suffix         = experiment_suffix(args.sizes, args.lightings)
    exp_suffix     = f"_{args.experiment_id}" if args.experiment_id else ""
    eval_json_path = symmetry_dir / f"eval_{suffix}{exp_suffix}_{args.method}_results.json"
    eval_csv_path  = symmetry_dir / f"eval_{suffix}{exp_suffix}_{args.method}_summary.csv"
    predicted_file = exp_filename(PREDICTED_FILE, args.experiment_id)

    all_object_dirs = sorted(d for d in symmetry_dir.iterdir() if d.is_dir())
    if args.max_objects:
        all_object_dirs = all_object_dirs[:args.max_objects]

    n_attempted = len(all_object_dirs)   # denominator for all metrics
    ref_seed = None if args.ref_seed == -1 else args.ref_seed

    print(f"\nEvaluating {n_attempted} objects  [{args.symmetry_type}]")
    print(f"Method        : {args.method}")
    print(f"Experiment    : sizes={args.sizes}  lightings={args.lightings}")
    if args.experiment_id:
        print(f"Experiment ID : {args.experiment_id}  →  reads {predicted_file}")
    print(f"Reference metrics (SDE_ref/F1_ref): {'enabled' if args.with_reference_metrics else 'disabled (pass --with-reference-metrics)'}")
    print(f"Results JSON  : {eval_json_path}")
    print(f"Summary CSV   : {eval_csv_path}\n")

    all_results: dict[str, dict | None] = {}

    for obj_dir in tqdm(all_object_dirs, unit="obj", dynamic_ncols=True):
        txt_path = objects_dir / f"{obj_dir.name}.txt"
        if not txt_path.exists():
            all_results[obj_dir.name] = None
            continue

        true_label = parse_true_label(txt_path)

        # Vertices are cheap and used for translation_error_normalized on both
        # symmetry types now (previously plane_sym only).
        vertices = load_mesh_vertices(objects_dir / f"{obj_dir.name}.obj")

        ref_ctx = None
        if args.with_reference_metrics:
            try:
                mesh = load_mesh(objects_dir / f"{obj_dir.name}.obj")
                ref_ctx = _build_ref_ctx(
                    mesh, objects_dir, obj_dir.name, args.symmetry_type,
                    args.ref_n_samples, ref_seed,
                )
            except Exception as e:
                print(f"  [warn] reference-metrics setup failed for {obj_dir.name}: {e}")

        all_results[obj_dir.name] = evaluate_object(
            obj_dir, true_label, args.symmetry_type, vertices,
            method=args.method,
            predicted_file=predicted_file,
            ref_ctx=ref_ctx,
        )

    # Save JSON
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "symmetry_type": args.symmetry_type,
            "method":        args.method,
            "sizes":         args.sizes,
            "lightings":     args.lightings,
            "objects":       all_results,
        }, f, indent=2)
    print(f"Saved: {eval_json_path}")

    # Save CSV + print table
    summary = compute_summary(all_results, args.symmetry_type, n_total=n_attempted)
    write_csv(summary, eval_csv_path, args.symmetry_type, args.with_reference_metrics)
    print(f"Saved: {eval_csv_path}\n")

    # Console summary table
    is_multiplane = any("n_planes_predicted_mean" in s for s in summary.values())

    if is_multiplane:
        # docs/actualizacion_metricas.md S3.3/S3.4 -- recall/precision over
        # the GT plane SET, not a best-match angular error.
        cols = (f"{'n_views':<8} {'n_obj/tot':<10} {'n_pred':>7} {'n_true':>7} "
                f"{'n_match':>8} {'recall':>7} {'precision':>9}")
        if args.with_reference_metrics:
            cols += f"  {'SDE_ref':>10} {'F1_ref':>7} {'F1_ref_H':>8}"
        print(cols)
        print("─" * len(cols))
        for nv, s in summary.items():
            n_valid = s["n_objects"]
            n_tot   = s.get("n_total", n_valid)
            recall     = s.get("recall_planes_mean")
            precision  = s.get("precision_planes_mean")
            recall_str = f"{recall:>7.4f}" if recall is not None else f"{'—':>7}"
            prec_str   = f"{precision:>9.4f}" if precision is not None else f"{'—':>9}"
            row = (f"{nv:<8} {f'{n_valid}/{n_tot}':<10} "
                   f"{s['n_planes_predicted_mean']:>7.2f} "
                   f"{s['n_true_planes_mean']:>7.2f} "
                   f"{s['n_planes_matched_mean']:>8.2f} "
                   f"{recall_str} {prec_str}")
            if args.with_reference_metrics:
                sde_ref = s.get("sde_ref_mean")
                row += f"  {sde_ref:>10.6f}" if sde_ref is not None else f"  {'—':>10}"
                f1  = s.get("f1_ref")
                f1h = s.get("f1_ref_hungarian")
                row += f" {f1:>7.4f}" if f1 is not None else f" {'—':>7}"
                row += f" {f1h:>8.4f}" if f1h is not None else f" {'—':>8}"
            print(row)
        return

    # n_obj = valid predictions; n_total = all attempted (denominator for angular metrics)
    cols = (f"{'n_views':<8} {'n_obj/tot':<10} {'ang_mean':>9} {'ang_med':>8} "
            f"{'AUC_ang':>8} {'p@5°':>7} {'p@10°':>7} {'trans_mean':>11}")
    if args.with_reference_metrics:
        cols += f"  {'SDE_ref':>10}"
        if args.symmetry_type == "plane_sym":
            cols += f" {'F1_ref':>7} {'F1_ref_H':>8}"
    print(cols)
    print("─" * len(cols))
    for nv, s in summary.items():
        n_valid = s["n_objects"]
        n_tot   = s.get("n_total", n_valid)
        t_mean  = s.get("translation_error_mean")
        t_str   = f"{t_mean:>11.5f}" if t_mean is not None else f"{'—':>11}"
        row = (f"{nv:<8} {f'{n_valid}/{n_tot}':<10} "
               f"{s['angular_error_mean']:>9.2f} "
               f"{s['angular_error_median']:>8.2f} "
               f"{s['auc_angular']:>8.4f} "
               f"{s.get('precision_5deg', 0):>7.3f} "
               f"{s.get('precision_10deg', 0):>7.3f} "
               f"{t_str}")
        if args.with_reference_metrics:
            sde_ref = s.get("sde_ref_mean")
            row += f"  {sde_ref:>10.6f}" if sde_ref is not None else f"  {'—':>10}"
            if args.symmetry_type == "plane_sym":
                f1  = s.get("f1_ref")
                f1h = s.get("f1_ref_hungarian")
                row += f" {f1:>7.4f}" if f1 is not None else f" {'—':>7}"
                row += f" {f1h:>8.4f}" if f1h is not None else f" {'—':>8}"
        print(row)


if __name__ == "__main__":
    main()
