# Monte Carlo random baseline: a predictor that never looks at the renders and
# draws its axis direction / plane normal uniformly on the sphere. It is scored
# with the same rules as Mapping/evaluate.py (best match over every GT plane,
# 90 deg imputed for objects without a prediction, F1_ref TP/FP/FN accumulated
# over the dataset), reusing that module's functions and constants wherever
# they are not per-object Python loops. The vectorized F1 counts are checked
# against evaluate.f1_match_counts / f1_match_counts_hungarian in
# RandomBaseline/tests/test_random_baseline.py.
from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Mapping"))
from pipeline_common.datasets import OBJECTS_SUBDIR  # noqa: E402
from evaluate import (  # noqa: E402
    ANGULAR_THRESHOLDS, AUC_ANGULAR_MAX, THRESHOLDS_INLIER, _f1_from_counts_by_threshold,
    auc_from_errors, normal_origin_to_plane, parse_true_label,
)

IMPUTED_ERROR_DEG = 90.0
GT_VECTOR_KEY = {"axis_sym": "direction", "plane_sym": "normal"}


@dataclass
class GroundTruth:
    """ GT symmetry elements of a set of objects, NaN-padded to the max elements per object. """

    symmetry_type: str
    object_ids: list[str]
    vectors: np.ndarray          # (N, K, 3) unit axis directions / plane normals
    planes: np.ndarray | None    # (N, K, 4) [nx, ny, nz, d] as in F1_ref (plane_sym only)
    n_elements: np.ndarray       # (N,) GT elements per object

    @property
    def n_objects(self) -> int:
        """ Number of objects. """
        return len(self.object_ids)


def load_ground_truth(objects_root: Path, symmetry_type: str,
                      object_ids: list[str] | None = None) -> GroundTruth:
    """ Load GT labels from <objects_root>/curated_*_sym_obj/<id>.txt.

    Args:
        * objects_root: folder holding curated_axis_sym_obj/ and curated_plane_sym_obj/
        * symmetry_type: "axis_sym" or "plane_sym"
        * object_ids: objects to load, in order; every .txt in the folder when None

    Returns:
        * GroundTruth: padded GT arrays, one row per object

    """
    folder = Path(objects_root) / OBJECTS_SUBDIR[symmetry_type]
    if object_ids is None:
        object_ids = sorted(p.stem for p in folder.glob("*.txt"))
    if not object_ids:
        raise FileNotFoundError(f"no ground-truth .txt labels found in {folder}")

    key = GT_VECTOR_KEY[symmetry_type]
    elements = []
    for object_id in object_ids:
        txt_path = folder / f"{object_id}.txt"
        if not txt_path.exists():
            raise FileNotFoundError(f"missing ground-truth label: {txt_path}")
        elements.append(parse_true_label(txt_path)["elements"])

    n, k = len(elements), max(len(els) for els in elements)
    vectors = np.full((n, k, 3), np.nan)
    planes = np.full((n, k, 4), np.nan) if symmetry_type == "plane_sym" else None
    for i, els in enumerate(elements):
        for j, el in enumerate(els):
            vectors[i, j] = el[key]
            if planes is not None:
                planes[i, j] = normal_origin_to_plane(el[key], el["origin"]).reshape(4)

    return GroundTruth(
        symmetry_type=symmetry_type,
        object_ids=list(object_ids),
        vectors=vectors,
        planes=planes,
        n_elements=np.array([len(els) for els in elements]),
    )


def random_directions(rng: np.random.Generator, size: tuple[int, ...] | int) -> np.ndarray:
    """ Directions uniform on the unit sphere: normalized standard Gaussian vectors (Muller 1959).

    Args:
        * rng: NumPy random generator
        * size: leading shape of the output (an int gives shape (size, 3))

    Returns:
        * np.ndarray: unit vectors with shape (*size, 3)

    """
    shape = (size,) if isinstance(size, int) else tuple(size)
    v = rng.normal(size=(*shape, 3))
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def best_match_errors(directions: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """ Sign-agnostic angular error to the closest GT element, as evaluate_axis/evaluate_plane.

    Args:
        * directions: (..., N, 3) predicted unit directions/normals
        * vectors: (N, K, 3) NaN-padded GT unit vectors

    Returns:
        * np.ndarray: (..., N) errors in degrees, in [0, 90], rounded like evaluate.py

    """
    cos = np.abs(np.einsum("...nd,nkd->...nk", directions, vectors))
    cos = np.where(np.isnan(cos), -1.0, cos)
    best = np.clip(cos.max(axis=-1), 0.0, 1.0)
    return np.round(np.degrees(np.arccos(best)), 4)


def f1_counts(directions: np.ndarray,
              planes: np.ndarray) -> tuple[dict[float, tuple[int, int, int]],
                                           dict[float, tuple[int, int, int]]]:
    """ Dataset-level F1_ref (tp, fp, fn) per threshold for one random plane per object.

    The random plane passes through the origin (d = 0): objects are normalized
    and centred, so this is the plane through the object's centre. With a
    single predicted plane the greedy loop of evaluate.f1_match_counts scores
    every GT plane independently (match -> tp, else fp and fn), and the
    Hungarian variant keeps only the closest GT plane.

    Args:
        * directions: (M, 3) random unit normals, objects with a prediction only
        * planes: (M, K, 4) NaN-padded GT planes of those objects

    Returns:
        * tuple: (greedy counts, hungarian counts), each {threshold: (tp, fp, fn)}

    """
    pred = np.concatenate([directions, np.zeros((len(directions), 1))], axis=1)[:, None, :]
    dist = np.minimum(np.linalg.norm(pred - planes, axis=2), np.linalg.norm(pred + planes, axis=2))
    valid = ~np.isnan(dist)
    n_gt_total = int(valid.sum())
    closest = np.where(valid, dist, np.inf).min(axis=1)

    greedy, hungarian = {}, {}
    for t in THRESHOLDS_INLIER:
        tp = int((valid & (np.nan_to_num(dist, nan=np.inf) < t)).sum())
        greedy[t] = (tp, n_gt_total - tp, n_gt_total - tp)
        tp_h = int((closest < t).sum())
        hungarian[t] = (tp_h, len(directions) - tp_h, n_gt_total - tp_h)
    return greedy, hungarian


def summarize_errors(errors: np.ndarray) -> dict[str, float]:
    """ Angular summary metrics over all objects, same formulas as evaluate.compute_summary.

    Args:
        * errors: (N,) angular errors in degrees, 90 already imputed for missing objects

    Returns:
        * dict[str, float]: metric name -> value

    """
    summary = {
        "angular_error_mean": float(errors.mean()),
        "angular_error_median": float(np.median(errors)),
        "angular_error_std": float(errors.std()),
        "auc_angular": auc_from_errors(errors.tolist(), AUC_ANGULAR_MAX),
    }
    for t in ANGULAR_THRESHOLDS:
        summary[f"precision_{t}deg"] = float(np.mean(errors < t))
    return summary


def simulate_random_baseline(gt: GroundTruth, n_repetitions: int = 1000, seed: int = 0,
                             has_prediction: np.ndarray | None = None,
                             n_with_prediction: int | None = None) -> dict[str, np.ndarray]:
    """ Score the uniform random predictor n_repetitions times on the given objects.

    Coverage follows the method being compared: either the exact objects that
    had a prediction (has_prediction, when per-object results are available),
    or a random subset of n_with_prediction objects redrawn every repetition
    (when only the summary counts are known). Objects outside it get 90 deg and
    contribute nothing to F1_ref, exactly as in evaluate.compute_summary.

    Args:
        * gt: ground truth of the evaluated objects
        * n_repetitions: number of simulated datasets
        * seed: RNG seed (same seed -> same draws, so rows are comparable)
        * has_prediction: (N,) bool mask of objects the method predicted
        * n_with_prediction: number of objects the method predicted; all when both are None

    Returns:
        * dict[str, np.ndarray]: metric name -> (n_repetitions,) simulated values;
          f1_ref / f1_ref_hungarian only for plane_sym

    """
    n = gt.n_objects
    if has_prediction is None and n_with_prediction is None:
        n_with_prediction = n
    rng = np.random.default_rng(seed)
    results: dict[str, list[float]] = defaultdict(list)

    for _ in range(n_repetitions):
        if has_prediction is not None:
            mask = np.asarray(has_prediction, dtype=bool)
        else:
            mask = np.zeros(n, dtype=bool)
            mask[rng.choice(n, size=n_with_prediction, replace=False)] = True

        directions = random_directions(rng, n)
        errors = np.where(mask, best_match_errors(directions, gt.vectors), IMPUTED_ERROR_DEG)
        for name, value in summarize_errors(errors).items():
            results[name].append(value)

        if gt.planes is not None and mask.any():
            greedy, hungarian = f1_counts(directions[mask], gt.planes[mask])
            results["f1_ref"].append(_f1_from_counts_by_threshold(greedy))
            results["f1_ref_hungarian"].append(_f1_from_counts_by_threshold(hungarian))

    return {name: np.array(values) for name, values in results.items()}
