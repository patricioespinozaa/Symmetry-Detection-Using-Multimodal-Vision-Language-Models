# trivial_baselines.py
#
# No-information reference scores for the no-mesh pipeline: what the SAME
# metrics Mapping/evaluate.py reports would give for predictors that never
# look at an image -- a fixed world axis (X/Y/Z) for axis_sym, a fixed set of
# world-aligned planes through the render look-at point (the origin) for
# plane_sym. Motivated by docs/reporte_resultados_sin_malla.md: on the 30
# local sandbox objects, axis GT directions are randomly oriented (so the
# chance reference is a uniform random direction) while plane GT normals are
# all within ~1.4 deg of a world axis (ShapeNet canonical pose), so a fixed
# "always X" plane already scores recall ~0.74 / f1_ref ~0.74 there. This
# script repeats that check over EVERY object's ground-truth .txt, so the
# comparison in the report rests on the full dataset, not a 30-object sample.
#
# Reads only data/objects/*/<id>.txt -- no renders, no Molmo2, no GPU.
#
#   python Pipeline_Experiments/diagnostics/trivial_baselines.py --objects-root ../data/objects
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline_common.datasets import OBJECTS_SUBDIR  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Mapping"))
from evaluate import (  # noqa: E402
    ANGULAR_THRESHOLDS, AUC_ANGULAR_MAX, THRESHOLDS_INLIER, _f1_from_counts_by_threshold,
    angular_error_deg, auc_from_errors, evaluate_plane_multiset, f1_match_counts,
    f1_match_counts_hungarian, normal_origin_to_plane, parse_true_label,
)

WORLD_AXES = {"X": np.array([1.0, 0.0, 0.0]), "Y": np.array([0.0, 1.0, 0.0]), "Z": np.array([0.0, 0.0, 1.0])}
AXIS_PREDICTORS = ["X", "Y", "Z"]
PLANE_PREDICTORS = ["X", "Y", "Z", "XZ", "XY", "XYZ"]


def load_labels(objects_root: Path, symmetry_type: str) -> list[list[dict]]:
    """ Load every object's GT elements for one symmetry type.

    Args:
        * objects_root: folder holding curated_{axis,plane}_sym_obj/
        * symmetry_type: "axis_sym" or "plane_sym"

    Returns:
        * list[list[dict]]: one list of GT elements (evaluate.py format) per object

    """
    folder = objects_root / OBJECTS_SUBDIR[symmetry_type]
    return [parse_true_label(p)["elements"] for p in sorted(folder.glob("*.txt"))]


def nearest_world_axis(vec: list[float]) -> tuple[str, float]:
    """ Return (axis name, angle in degrees) of the world axis closest to vec, sign-agnostic. """
    angles = {name: angular_error_deg(np.asarray(vec), axis) for name, axis in WORLD_AXES.items()}
    name = min(angles, key=angles.get)
    return name, angles[name]


def orientation_report(labels: list[list[dict]], key: str) -> dict:
    """ Summarize how aligned GT vectors are with the world axes.

    Args:
        * labels: GT elements per object
        * key: "direction" (axis) or "normal" (plane)

    Returns:
        * dict: counts by nearest axis and fractions within 5/15 degrees

    """
    nearest = [nearest_world_axis(el[key]) for els in labels for el in els]
    angles = np.array([a for _, a in nearest])
    return {
        "n_elements": len(nearest),
        "by_axis": dict(Counter(name for name, _ in nearest)),
        "frac_within_5deg": float(np.mean(angles < 5)),
        "frac_within_15deg": float(np.mean(angles < 15)),
        "random_ref_within_15deg": 3 * (1 - np.cos(np.radians(15))),
    }


def score_axis_predictor(labels: list[list[dict]], axis_name: str) -> dict:
    """ Angular-error summary for "always predict world axis <axis_name>", evaluate.py's formulas. """
    errors = [angular_error_deg(WORLD_AXES[axis_name], np.asarray(els[0]["direction"])) for els in labels]
    row = {
        "predictor": f"always {axis_name}",
        "angular_error_mean": round(float(np.mean(errors)), 2),
        "angular_error_median": round(float(np.median(errors)), 2),
        "auc_angular": round(auc_from_errors(errors, AUC_ANGULAR_MAX), 4),
    }
    for t in ANGULAR_THRESHOLDS:
        row[f"precision_{t}deg"] = round(float(np.mean([e < t for e in errors])), 4)
    return row


def uniform_random_axis_reference() -> dict:
    """ Closed-form scores of a uniformly random direction (density sin(t) on [0, 90] deg). """
    t = np.radians(AUC_ANGULAR_MAX)
    row = {
        "predictor": "uniform random direction (analytic)",
        "angular_error_mean": round(float(np.degrees(1.0)), 2),
        "angular_error_median": 60.0,
        "auc_angular": round(float(1 - np.sin(t) / t), 4),
    }
    for thr in ANGULAR_THRESHOLDS:
        row[f"precision_{thr}deg"] = round(float(1 - np.cos(np.radians(thr))), 4)
    return row


def score_plane_predictor(labels: list[list[dict]], axes: str) -> dict:
    """ Recall/precision/F1_ref for "always predict the world planes named in <axes>" through the origin. """
    preds = [{"normal": WORLD_AXES[a].tolist(), "origin": [0.0, 0.0, 0.0]} for a in axes]
    pred_vecs = [normal_origin_to_plane(p["normal"], p["origin"]).reshape(4) for p in preds]

    recalls, precisions = [], []
    greedy = {t: [0, 0, 0] for t in THRESHOLDS_INLIER}
    hungarian = {t: [0, 0, 0] for t in THRESHOLDS_INLIER}
    for els in labels:
        res = evaluate_plane_multiset(preds, els)
        recalls.append(res["recall_planes"])
        precisions.append(res["precision_planes"])
        gt_vecs = [normal_origin_to_plane(el["normal"], el["origin"]).reshape(4) for el in els]
        for t in THRESHOLDS_INLIER:
            for acc, fn in ((greedy, f1_match_counts), (hungarian, f1_match_counts_hungarian)):
                tp, fp, fnn = fn(pred_vecs, gt_vecs, t)
                acc[t][0] += tp
                acc[t][1] += fp
                acc[t][2] += fnn

    return {
        "predictor": "always {" + ",".join(axes) + "}",
        "n_planes_predicted": len(preds),
        "recall_planes_mean": round(float(np.mean(recalls)), 4),
        "precision_planes_mean": round(float(np.mean(precisions)), 4),
        "f1_ref": round(_f1_from_counts_by_threshold({t: tuple(c) for t, c in greedy.items()}), 4),
        "f1_ref_hungarian": round(_f1_from_counts_by_threshold({t: tuple(c) for t, c in hungarian.items()}), 4),
    }


def main() -> None:
    """ CLI entrypoint: print orientation stats and trivial-predictor scores for both symmetry types. """
    p = argparse.ArgumentParser(description="No-information reference scores for the no-mesh pipeline.")
    p.add_argument("--objects-root", required=True)
    args = p.parse_args()
    objects_root = Path(args.objects_root)

    axis_labels = load_labels(objects_root, "axis_sym")
    print(f"=== axis_sym: {len(axis_labels)} objects ===")
    print("GT axis orientation:", orientation_report(axis_labels, "direction"))
    for row in [uniform_random_axis_reference()] + [score_axis_predictor(axis_labels, a) for a in AXIS_PREDICTORS]:
        print("  ", row)

    plane_labels = load_labels(objects_root, "plane_sym")
    print(f"\n=== plane_sym: {len(plane_labels)} objects, "
          f"{sum(len(els) for els in plane_labels) / len(plane_labels):.3f} GT planes/object ===")
    print("GT plane-normal orientation:", orientation_report(plane_labels, "normal"))
    for axes in PLANE_PREDICTORS:
        print("  ", score_plane_predictor(plane_labels, axes))


if __name__ == "__main__":
    main()
