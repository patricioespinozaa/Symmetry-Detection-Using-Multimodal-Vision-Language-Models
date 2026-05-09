"""
evaluate.py
-----------
Compares predicted symmetry (from estimate_symmetry.py) against true labels
(.txt files) and computes evaluation metrics.

Metrics
-------
axis_sym:
    - Angular error (degrees): angle between predicted and true axis directions
      min(angle(pred, true), angle(pred, -true)) — handles sign ambiguity
    - Origin distance: distance between predicted origin and true axis line
      (point-to-line distance, since axis origin can vary along the line)

plane_sym (per predicted plane vs best-matching true plane):
    - Angular error (degrees): angle between predicted and true plane normals
    - Origin distance: distance between predicted origin and true plane
    - Best-match strategy: Hungarian matching minimizing angular error

Output
------
<renders_root>/<symmetry_type>/
    evaluation_results.json    ← all objects, all n_views groups
    evaluation_summary.csv     ← mean/median metrics per n_views group

JSON format
-----------
{
  "symmetry_type": "axis_sym",
  "objects": {
    "<object_id>": {
      "1":  {"angular_error_deg": 5.2, "origin_dist": 0.03, "status": "ok"},
      "6":  {...},
      ...
    },
    ...
  }
}

Usage
-----
    python Mapping/evaluate.py \\
        --renders-root ../data/renders \\
        --objects-root ../data/objects \\
        --symmetry-type axis_sym

    python Mapping/evaluate.py \\
        --renders-root ../data/renders \\
        --objects-root ../data/objects \\
        --symmetry-type plane_sym
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────

PREDICTED_FILE  = "predicted_symmetry.json"
EVAL_JSON       = "evaluation_results.json"
EVAL_CSV        = "evaluation_summary.csv"
OBJECTS_SUBDIR  = {"axis_sym": "curated_axis_sym_obj",
                   "plane_sym": "curated_plane_sym_obj"}


# ── True label parser ─────────────────────────────────────────────────────────

def parse_true_label(txt_path: Path) -> dict:
    """
    Parse a symmetry .txt file.

    axis_sym format:
        1
        axis DX DY DZ  OX OY OZ
        N_ANGLES
        angles A1 A2 ...

    plane_sym format:
        N_PLANES
        plane NX NY NZ  OX OY OZ
        plane NX NY NZ  OX OY OZ
        ...

    Returns:
        {
          "type": "axis" | "plane",
          "elements": [
            {"direction": [dx,dy,dz], "origin": [ox,oy,oz]},  # axis
            {"normal":    [nx,ny,nz], "origin": [ox,oy,oz]},  # plane
            ...
          ]
        }
    """
    lines = [l.strip() for l in txt_path.read_text().splitlines() if l.strip()]
    elements = []
    sym_type = None

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("axis"):
            sym_type = "axis"
            parts    = line.split()
            vec      = [float(x) for x in parts[1:4]]
            orig     = [float(x) for x in parts[4:7]]
            # Normalize direction
            v = np.array(vec)
            v /= np.linalg.norm(v)
            elements.append({"direction": v.tolist(), "origin": orig})
            i += 1
        elif line.startswith("plane"):
            sym_type = "plane"
            parts    = line.split()
            vec      = [float(x) for x in parts[1:4]]
            orig     = [float(x) for x in parts[4:7]]
            v = np.array(vec)
            v /= np.linalg.norm(v)
            elements.append({"normal": v.tolist(), "origin": orig})
            i += 1
        else:
            i += 1

    return {"type": sym_type, "elements": elements}


# ── Geometric metrics ─────────────────────────────────────────────────────────

def angular_error_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Angle in degrees between two unit vectors.
    Handles sign ambiguity (axis/normal direction is undetermined).
    Returns value in [0, 90].
    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    cos_angle = np.clip(np.abs(np.dot(v1, v2)), 0.0, 1.0)  # abs for sign ambiguity
    return float(np.degrees(np.arccos(cos_angle)))


def point_to_line_distance(point: np.ndarray,
                            line_origin: np.ndarray,
                            line_dir: np.ndarray) -> float:
    """
    Distance from a point to a 3D line defined by origin + direction.
    """
    d   = line_dir / np.linalg.norm(line_dir)
    v   = point - line_origin
    return float(np.linalg.norm(v - np.dot(v, d) * d))


def point_to_plane_distance(point: np.ndarray,
                             plane_origin: np.ndarray,
                             plane_normal: np.ndarray) -> float:
    """
    Signed distance from a point to a plane (normal + origin).
    Returns absolute value.
    """
    n = plane_normal / np.linalg.norm(plane_normal)
    return float(abs(np.dot(point - plane_origin, n)))


# ── Per-object evaluation ─────────────────────────────────────────────────────

def evaluate_axis(pred: dict, true_elements: list[dict]) -> dict:
    """
    Compare predicted axis vs true axis.
    Returns metrics dict.
    """
    true   = true_elements[0]
    t_dir  = np.array(true["direction"])
    t_orig = np.array(true["origin"])

    p_dir  = np.array(pred["direction"])
    p_orig = np.array(pred["origin"])

    ang_err  = angular_error_deg(p_dir, t_dir)
    orig_dist = point_to_line_distance(p_orig, t_orig, t_dir)

    return {
        "angular_error_deg": round(ang_err, 4),
        "origin_dist":       round(orig_dist, 6),
        "n_points":          pred.get("n_points", 0),
        "status":            "ok",
    }


def evaluate_plane(pred: dict, true_elements: list[dict]) -> dict:
    """
    Compare predicted plane vs best-matching true plane (min angular error).
    Returns metrics dict including which true plane was matched.
    """
    p_normal = np.array(pred["normal"])
    p_origin = np.array(pred["origin"])

    best_ang  = float("inf")
    best_idx  = -1
    best_dist = float("inf")

    for idx, true in enumerate(true_elements):
        t_normal = np.array(true["normal"])
        t_origin = np.array(true["origin"])

        ang  = angular_error_deg(p_normal, t_normal)
        dist = point_to_plane_distance(p_origin, t_origin, t_normal)

        if ang < best_ang:
            best_ang  = ang
            best_idx  = idx
            best_dist = dist

    return {
        "angular_error_deg":   round(best_ang, 4),
        "origin_dist":         round(best_dist, 6),
        "matched_true_plane":  best_idx,
        "n_true_planes":       len(true_elements),
        "n_points":            pred.get("n_points", 0),
        "status":              "ok",
    }


def evaluate_object(
    object_dir:    Path,
    true_label:    dict,
    symmetry_type: str,
) -> dict | None:
    """
    Load predicted_symmetry.json and evaluate all n_views groups.
    Returns dict {n_views_key: metrics} or None if file missing.
    """
    pred_path = object_dir / PREDICTED_FILE
    if not pred_path.exists():
        return None

    with open(pred_path, encoding="utf-8") as f:
        predicted = json.load(f)

    true_elements = true_label["elements"]
    results       = {}

    for n_views_key, pred in predicted.get("n_views_predictions", {}).items():
        try:
            if symmetry_type == "axis_sym":
                metrics = evaluate_axis(pred, true_elements)
            else:
                metrics = evaluate_plane(pred, true_elements)
        except Exception as e:
            metrics = {"status": f"error: {e}"}

        results[n_views_key] = metrics

    return results


# ── Summary statistics ────────────────────────────────────────────────────────

def compute_summary(all_results: dict) -> dict[str, dict]:
    """
    Compute mean / median / std of angular_error_deg and origin_dist
    per n_views group across all objects.
    """
    from collections import defaultdict

    grouped: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for obj_results in all_results.values():
        if obj_results is None:
            continue
        for n_views_key, metrics in obj_results.items():
            if metrics.get("status") != "ok":
                continue
            grouped[n_views_key]["angular_error_deg"].append(
                metrics["angular_error_deg"])
            grouped[n_views_key]["origin_dist"].append(
                metrics["origin_dist"])
            grouped[n_views_key]["n_points"].append(
                metrics.get("n_points", 0))

    summary = {}
    for n_views_key, data in sorted(grouped.items(), key=lambda x: int(x[0])):
        ang  = np.array(data["angular_error_deg"])
        dist = np.array(data["origin_dist"])
        pts  = np.array(data["n_points"])
        summary[n_views_key] = {
            "n_objects":             len(ang),
            "angular_error_mean":    round(float(ang.mean()),   4),
            "angular_error_median":  round(float(np.median(ang)), 4),
            "angular_error_std":     round(float(ang.std()),    4),
            "origin_dist_mean":      round(float(dist.mean()),  6),
            "origin_dist_median":    round(float(np.median(dist)), 6),
            "origin_dist_std":       round(float(dist.std()),   6),
            "n_points_mean":         round(float(pts.mean()),   2),
        }

    return summary


def write_csv(summary: dict, csv_path: Path) -> None:
    fieldnames = [
        "n_views", "n_objects",
        "angular_error_mean", "angular_error_median", "angular_error_std",
        "origin_dist_mean",   "origin_dist_median",   "origin_dist_std",
        "n_points_mean",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for n_views_key, stats in summary.items():
            row = {"n_views": n_views_key, **stats}
            writer.writerow(row)


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
    return p.parse_args()


def main() -> None:
    args = parse_args()

    symmetry_dir   = Path(args.renders_root) / args.symmetry_type
    objects_subdir = OBJECTS_SUBDIR[args.symmetry_type]
    objects_dir    = Path(args.objects_root) / objects_subdir

    if not symmetry_dir.exists():
        print(f"[error] Not found: {symmetry_dir}")
        sys.exit(1)

    all_object_dirs = sorted(d for d in symmetry_dir.iterdir() if d.is_dir())

    print(f"\nEvaluating {len(all_object_dirs)} objects ({args.symmetry_type})...")

    all_results: dict[str, dict | None] = {}

    for obj_dir in tqdm(all_object_dirs, unit="obj", dynamic_ncols=True):
        txt_path = objects_dir / f"{obj_dir.name}.txt"
        if not txt_path.exists():
            print(f"  [warn] No label: {txt_path}")
            all_results[obj_dir.name] = None
            continue

        true_label = parse_true_label(txt_path)
        result     = evaluate_object(obj_dir, true_label, args.symmetry_type)
        all_results[obj_dir.name] = result

    # Save full results
    eval_json_path = symmetry_dir / EVAL_JSON
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "symmetry_type": args.symmetry_type,
            "objects":       all_results,
        }, f, indent=2)
    print(f"\nResults saved: {eval_json_path}")

    # Summary CSV
    summary      = compute_summary(all_results)
    eval_csv_path = symmetry_dir / EVAL_CSV
    write_csv(summary, eval_csv_path)
    print(f"Summary CSV:   {eval_csv_path}")

    # Print summary table
    print(f"\n{'n_views':<10} {'n_obj':<8} {'ang_err_mean':<16} "
          f"{'ang_err_median':<18} {'origin_dist_mean':<18} {'pts_mean'}")
    print("-" * 80)
    for n_views_key, s in summary.items():
        print(
            f"{n_views_key:<10} {s['n_objects']:<8} "
            f"{s['angular_error_mean']:<16} {s['angular_error_median']:<18} "
            f"{s['origin_dist_mean']:<18} {s['n_points_mean']}"
        )


if __name__ == "__main__":
    main()
