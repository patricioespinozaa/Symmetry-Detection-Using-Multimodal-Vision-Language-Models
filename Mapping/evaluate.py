"""
evaluate.py
-----------
Compares predicted symmetry (from estimate_symmetry.py) against true labels
(.txt files) and computes evaluation metrics aligned with the paper.

Metrics
-------
axis_sym:
    - Angular error (degrees): angle between predicted and true axis directions
      (sign-agnostic, returns value in [0, 90])
    - Translation error: point-to-line distance from predicted origin to true axis
    - Precision under threshold: % of objects with angular error < 5°, 10°, 15°
    - AUC: area under the precision-vs-threshold curve (0°–45°, normalized)

plane_sym:
    - Angular error (degrees): angle between predicted and true plane normals
      (best-match against all true planes, sign-agnostic)
    - Translation error: point-to-plane distance from predicted origin to best true plane
    - SDE (Symmetry Distance Error): mean distance between mesh vertices and their
      reflections through the predicted plane, normalized by bbox diagonal
    - Precision under SDE threshold: % of objects with SDE < 0.01, 0.02 (bbox diagonal)
    - AUC: area under the precision-vs-SDE-threshold curve (0–0.10, normalized)
    - Precision under angular threshold: % of objects with angular error < 5°, 10°, 15°

Output
------
Results are saved under <renders_root>/<symmetry_type>/ with filenames that
encode the experiment configuration (sizes + lightings):

    eval_{sizes}_{lightings}_results.json   ← per-object, per-n_views metrics
    eval_{sizes}_{lightings}_summary.csv    ← aggregated metrics per n_views group

Example filenames:
    eval_s224_flat_results.json
    eval_s224_448_1136_flat_brighter_darker_results.json

JSON format
-----------
{
  "symmetry_type": "axis_sym",
  "sizes": [224],
  "lightings": ["flat"],
  "objects": {
    "<object_id>": {
      "1": {
        "angular_error_deg":  5.2,
        "translation_error":  0.03,
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
    "sde":                  0.008,
    "precision_sde_010":    1,      ← SDE < 0.01
    "precision_sde_020":    1,      ← SDE < 0.02
    "matched_true_plane":   0,
    "n_true_planes":        2

Usage
-----
    python Mapping/evaluate.py \\
        --renders-root ../data/renders \\
        --objects-root ../data/objects \\
        --symmetry-type axis_sym \\
        --sizes 224 \\
        --lightings flat

    python Mapping/evaluate.py \\
        --renders-root ../data/renders \\
        --objects-root ../data/objects \\
        --symmetry-type plane_sym \\
        --sizes 224 448 \\
        --lightings flat brighter darker
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────

PREDICTED_FILE = "predicted_symmetry.json"
OBJECTS_SUBDIR = {
    "axis_sym":  "curated_axis_sym_obj",
    "plane_sym": "curated_plane_sym_obj",
}

# Thresholds
ANGULAR_THRESHOLDS = [5, 10, 15]           # degrees
SDE_THRESHOLDS     = [0.01, 0.02]          # fraction of bbox diagonal

# AUC integration ranges
AUC_ANGULAR_MAX = 45.0   # degrees — beyond this is considered total failure
AUC_SDE_MAX     = 0.10   # SDE cap


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


# ── Mesh loading ──────────────────────────────────────────────────────────────

def load_mesh_vertices(obj_path: Path) -> np.ndarray | None:
    """Load .obj and return (N, 3) vertex array. Returns None on failure."""
    try:
        scene_or_mesh = trimesh.load(str(obj_path), force="mesh", process=False)
        if isinstance(scene_or_mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(
                [g for g in scene_or_mesh.geometry.values()
                 if isinstance(g, trimesh.Trimesh)]
            )
        else:
            mesh = scene_or_mesh
        return np.array(mesh.vertices, dtype=np.float64)
    except Exception:
        return None


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


def symmetry_distance_error(vertices: np.ndarray,
                             plane_origin: np.ndarray,
                             plane_normal: np.ndarray,
                             bbox_diag: float) -> float:
    """
    SDE = mean( 2 * |dot(v - origin, n)| ) / bbox_diagonal

    Equivalent to mean distance between each vertex and its reflection
    through the plane, normalized by the bounding box diagonal.
    """
    n         = plane_normal / np.linalg.norm(plane_normal)
    distances = 2.0 * np.abs((vertices - plane_origin) @ n)
    return float(distances.mean() / bbox_diag)


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
    return float(np.trapz(precisions, thresholds) / max_error)


# ── Per-object evaluation ─────────────────────────────────────────────────────

def evaluate_axis(pred: dict, true_elements: list[dict]) -> dict:
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
    for t in ANGULAR_THRESHOLDS:
        m[f"precision_{t}deg"] = 1 if ang_err < t else 0
    return m


def evaluate_plane(pred: dict,
                   true_elements: list[dict],
                   vertices: np.ndarray | None) -> dict:
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
    for t in ANGULAR_THRESHOLDS:
        m[f"precision_{t}deg"] = 1 if best_ang < t else 0

    # SDE
    if vertices is not None:
        diag = bbox_diagonal(vertices)
        if diag > 0:
            sde = symmetry_distance_error(vertices, p_origin, p_normal, diag)
            m["sde"] = round(sde, 6)
            for t in SDE_THRESHOLDS:
                key = f"precision_sde_{int(t * 1000):03d}"
                m[key] = 1 if sde < t else 0
        else:
            m["sde"] = None
    else:
        m["sde"] = None

    return m


def evaluate_object(object_dir: Path,
                    true_label: dict,
                    symmetry_type: str,
                    vertices: np.ndarray | None) -> dict | None:
    pred_path = object_dir / PREDICTED_FILE
    if not pred_path.exists():
        return None

    with open(pred_path, encoding="utf-8") as f:
        predicted = json.load(f)

    results = {}
    for n_views_key, pred in predicted.get("n_views_predictions", {}).items():
        try:
            if symmetry_type == "axis_sym":
                metrics = evaluate_axis(pred, true_label["elements"])
            else:
                metrics = evaluate_plane(pred, true_label["elements"], vertices)
        except Exception as e:
            metrics = {"status": f"error: {e}"}
        results[n_views_key] = metrics

    return results


# ── Summary ───────────────────────────────────────────────────────────────────

def compute_summary(all_results: dict, symmetry_type: str) -> dict[str, dict]:
    grouped: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for obj_results in all_results.values():
        if obj_results is None:
            continue
        for nv, m in obj_results.items():
            if m.get("status") != "ok":
                continue
            g = grouped[nv]
            g["angular_error_deg"].append(m["angular_error_deg"])
            g["translation_error"].append(m["translation_error"])
            g["n_points"].append(m.get("n_points", 0))
            for t in ANGULAR_THRESHOLDS:
                g[f"precision_{t}deg"].append(m.get(f"precision_{t}deg", 0))
            if symmetry_type == "plane_sym" and m.get("sde") is not None:
                g["sde"].append(m["sde"])
                for t in SDE_THRESHOLDS:
                    key = f"precision_sde_{int(t * 1000):03d}"
                    g[key].append(m.get(key, 0))

    summary = {}
    for nv, data in sorted(grouped.items(), key=lambda x: int(x[0])):
        ang  = np.array(data["angular_error_deg"])
        dist = np.array(data["translation_error"])
        pts  = np.array(data["n_points"])

        s = {
            "n_objects":                len(ang),
            "angular_error_mean":       round(float(ang.mean()),        4),
            "angular_error_median":     round(float(np.median(ang)),    4),
            "angular_error_std":        round(float(ang.std()),         4),
            "translation_error_mean":   round(float(dist.mean()),       6),
            "translation_error_median": round(float(np.median(dist)),   6),
            "translation_error_std":    round(float(dist.std()),        6),
            "n_points_mean":            round(float(pts.mean()),        2),
            "auc_angular":              round(auc_from_errors(
                                            data["angular_error_deg"],
                                            AUC_ANGULAR_MAX), 4),
        }
        for t in ANGULAR_THRESHOLDS:
            vals = np.array(data[f"precision_{t}deg"])
            s[f"precision_{t}deg"] = round(float(vals.mean()), 4)

        if symmetry_type == "plane_sym" and data.get("sde"):
            sde_arr = np.array(data["sde"])
            s["sde_mean"]   = round(float(sde_arr.mean()),      6)
            s["sde_median"] = round(float(np.median(sde_arr)),  6)
            s["sde_std"]    = round(float(sde_arr.std()),       6)
            s["auc_sde"]    = round(auc_from_errors(data["sde"], AUC_SDE_MAX), 4)
            for t in SDE_THRESHOLDS:
                key  = f"precision_sde_{int(t * 1000):03d}"
                vals = np.array(data[key])
                s[key] = round(float(vals.mean()), 4)

        summary[nv] = s
    return summary


def write_csv(summary: dict, csv_path: Path, symmetry_type: str) -> None:
    base = [
        "n_views", "n_objects",
        "angular_error_mean", "angular_error_median", "angular_error_std",
        "translation_error_mean", "translation_error_median", "translation_error_std",
        "auc_angular", "n_points_mean",
    ] + [f"precision_{t}deg" for t in ANGULAR_THRESHOLDS]

    plane_extra = (
        ["sde_mean", "sde_median", "sde_std", "auc_sde"]
        + [f"precision_sde_{int(t*1000):03d}" for t in SDE_THRESHOLDS]
    ) if symmetry_type == "plane_sym" else []

    fieldnames = base + plane_extra
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for nv, stats in summary.items():
            writer.writerow({"n_views": nv, **stats})


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
    return p.parse_args()


def main() -> None:
    args = parse_args()

    symmetry_dir   = Path(args.renders_root) / args.symmetry_type
    objects_subdir = OBJECTS_SUBDIR[args.symmetry_type]
    objects_dir    = Path(args.objects_root) / objects_subdir

    if not symmetry_dir.exists():
        print(f"[error] Not found: {symmetry_dir}")
        sys.exit(1)

    suffix         = experiment_suffix(args.sizes, args.lightings)
    eval_json_path = symmetry_dir / f"eval_{suffix}_results.json"
    eval_csv_path  = symmetry_dir / f"eval_{suffix}_summary.csv"

    all_object_dirs = sorted(d for d in symmetry_dir.iterdir() if d.is_dir())
    print(f"\nEvaluating {len(all_object_dirs)} objects  [{args.symmetry_type}]")
    print(f"Experiment    : sizes={args.sizes}  lightings={args.lightings}")
    print(f"Results JSON  : {eval_json_path}")
    print(f"Summary CSV   : {eval_csv_path}\n")

    all_results: dict[str, dict | None] = {}

    for obj_dir in tqdm(all_object_dirs, unit="obj", dynamic_ncols=True):
        txt_path = objects_dir / f"{obj_dir.name}.txt"
        if not txt_path.exists():
            all_results[obj_dir.name] = None
            continue

        true_label = parse_true_label(txt_path)

        vertices = None
        if args.symmetry_type == "plane_sym":
            vertices = load_mesh_vertices(objects_dir / f"{obj_dir.name}.obj")

        all_results[obj_dir.name] = evaluate_object(
            obj_dir, true_label, args.symmetry_type, vertices
        )

    # Save JSON
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "symmetry_type": args.symmetry_type,
            "sizes":         args.sizes,
            "lightings":     args.lightings,
            "objects":       all_results,
        }, f, indent=2)
    print(f"Saved: {eval_json_path}")

    # Save CSV + print table
    summary = compute_summary(all_results, args.symmetry_type)
    write_csv(summary, eval_csv_path, args.symmetry_type)
    print(f"Saved: {eval_csv_path}\n")

    # Console summary table
    cols = (f"{'n_views':<8} {'n_obj':<6} {'ang_mean':>9} {'ang_med':>8} "
            f"{'AUC_ang':>8} {'p@5°':>7} {'p@10°':>7} {'trans_mean':>11}")
    if args.symmetry_type == "plane_sym":
        cols += f"  {'sde_mean':>9} {'AUC_sde':>8} {'p@SDE1%':>8}"
    print(cols)
    print("─" * len(cols))
    for nv, s in summary.items():
        row = (f"{nv:<8} {s['n_objects']:<6} "
               f"{s['angular_error_mean']:>9.2f} "
               f"{s['angular_error_median']:>8.2f} "
               f"{s['auc_angular']:>8.4f} "
               f"{s.get('precision_5deg', 0):>7.3f} "
               f"{s.get('precision_10deg', 0):>7.3f} "
               f"{s['translation_error_mean']:>11.5f}")
        if args.symmetry_type == "plane_sym":
            row += (f"  {s.get('sde_mean', 0):>9.5f} "
                    f"{s.get('auc_sde', 0):>8.4f} "
                    f"{s.get('precision_sde_010', 0):>8.3f}")
        print(row)


if __name__ == "__main__":
    main()