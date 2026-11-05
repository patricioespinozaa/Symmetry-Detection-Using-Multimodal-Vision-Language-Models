# Monte Carlo random baseline for multi-plane predictions (triangulation_multiplane):
# every object receives a SET of planes with uniform random normals, through the
# origin, at least the estimator's duplicate threshold apart. It is scored with
# the rules of evaluate_plane_multi_from_pred and compute_summary: greedy recall
# and precision at 15 deg per object, F1_ref TP/FP/FN accumulated over the dataset.
# The vectorized scoring is checked against evaluate.py in
# RandomBaseline/tests/test_random_baseline.py.
from __future__ import annotations

from collections import defaultdict

import numpy as np

from simulation import (
    ANGULAR_THRESHOLDS, THRESHOLDS_INLIER, _f1_from_counts_by_threshold, GroundTruth,
    random_directions,
)
from estimate_symmetry_no_mesh import DUP_ANGLE_THRESH_DEFAULT
from evaluate import linear_sum_assignment

MATCH_THRESHOLD_DEG = float(ANGULAR_THRESHOLDS[-1])
Counts = dict[float, tuple[int, int, int]]


def draw_plane_counts(rng: np.random.Generator, n_objects: int, mean_planes: float) -> np.ndarray:
    """ Planes per object that keep a given mean, as even as possible, on random objects.

    Args:
        * rng: NumPy random generator
        * n_objects: number of objects that receive planes
        * mean_planes: target mean number of planes per object

    Returns:
        * np.ndarray: (n_objects,) integer plane counts

    """
    base, extra = divmod(int(round(mean_planes * n_objects)), n_objects)
    counts = np.full(n_objects, base, dtype=int)
    counts[rng.choice(n_objects, size=extra, replace=False)] += 1
    return counts


def random_plane_sets(rng: np.random.Generator, counts: np.ndarray,
                      min_separation_deg: float = DUP_ANGLE_THRESH_DEFAULT) -> np.ndarray:
    """ Random normals per object; sets with planes too close together are redrawn whole.

    Args:
        * rng: NumPy random generator
        * counts: (M,) planes per object
        * min_separation_deg: smallest sign-agnostic angle allowed within a set

    Returns:
        * np.ndarray: (M, P, 3) unit normals, P = counts.max(); unused rows are never scored

    """
    n_objects, max_planes = len(counts), int(counts.max())
    normals = random_directions(rng, (n_objects, max_planes))
    if min_separation_deg <= 0 or max_planes < 2:
        return normals

    used = np.arange(max_planes) < counts[:, None]
    max_cos = np.cos(np.radians(min_separation_deg))
    while True:
        too_close = np.zeros(n_objects, dtype=bool)
        for i in range(max_planes):
            for j in range(i + 1, max_planes):
                cos = np.abs(np.einsum("nd,nd->n", normals[:, i], normals[:, j]))
                too_close |= used[:, i] & used[:, j] & (cos > max_cos)
        if not too_close.any():
            return normals
        normals[too_close] = random_directions(rng, (int(too_close.sum()), max_planes))


def multiset_scores(normals: np.ndarray, counts: np.ndarray,
                    vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ Per-object recall and precision, same greedy rule as evaluate_plane_multiset.

    Args:
        * normals: (M, P, 3) predicted unit normals
        * counts: (M,) planes predicted per object
        * vectors: (M, K, 3) NaN-padded GT unit normals

    Returns:
        * tuple[np.ndarray, np.ndarray]: (M,) recall and (M,) precision, rounded to 4 dp

    """
    cos = np.abs(np.einsum("npd,nkd->npk", normals, vectors))
    angles = np.degrees(np.arccos(np.clip(np.nan_to_num(cos, nan=0.0), 0.0, 1.0)))
    gt_valid = ~np.isnan(vectors[:, :, 0])
    angles = np.where(gt_valid[:, None, :], angles, np.inf)

    rows = np.arange(len(counts))
    matched = np.zeros(gt_valid.shape, dtype=bool)
    for p in range(normals.shape[1]):
        available = np.where(matched, np.inf, angles[:, p])
        best = available.argmin(axis=1)
        hit = (p < counts) & (available[rows, best] < MATCH_THRESHOLD_DEG)
        matched[rows[hit], best[hit]] = True

    n_matched = matched.sum(axis=1)
    return np.round(n_matched / gt_valid.sum(axis=1), 4), np.round(n_matched / counts, 4)


def assigned_costs(dist: np.ndarray, counts: np.ndarray, n_true: np.ndarray) -> np.ndarray:
    """ Plane-vector distances of the optimal assignment of each object.

    Args:
        * dist: (M, P, K) predicted-to-GT distances
        * counts: (M,) planes predicted per object
        * n_true: (M,) GT planes per object

    Returns:
        * np.ndarray: (M, min(P, K)) assigned distances, padded with inf

    """
    costs = np.full((len(counts), min(dist.shape[1:])), np.inf)
    for i, (n_pred, n_gt) in enumerate(zip(counts, n_true)):
        block = dist[i, :n_pred, :n_gt]
        if min(n_pred, n_gt) == 1:
            costs[i, 0] = block.min()
        else:
            row, col = linear_sum_assignment(block)
            costs[i, :len(row)] = block[row, col]
    return costs


def multiset_f1_counts(normals: np.ndarray, counts: np.ndarray,
                       planes: np.ndarray) -> tuple[Counts, Counts]:
    """ Dataset-level F1_ref (tp, fp, fn) per threshold for a set of planes per object.

    Follows evaluate.f1_match_counts literally: each prediction walks over every
    GT plane not matched yet, and every failed comparison adds one false positive.

    Args:
        * normals: (M, P, 3) predicted unit normals; every plane has d = 0
        * counts: (M,) planes predicted per object
        * planes: (M, K, 4) NaN-padded GT planes

    Returns:
        * tuple[Counts, Counts]: greedy and Hungarian counts, {threshold: (tp, fp, fn)}

    """
    n_objects, max_planes, max_gt = normals.shape[0], normals.shape[1], planes.shape[1]
    pred = np.concatenate([normals, np.zeros((n_objects, max_planes, 1))], axis=2)[:, :, None, :]
    gt = planes[:, None, :, :]
    dist = np.minimum(np.linalg.norm(pred - gt, axis=3), np.linalg.norm(pred + gt, axis=3))
    gt_valid = ~np.isnan(planes[:, :, 0])
    n_true = gt_valid.sum(axis=1)
    n_pred_total, n_gt_total = int(counts.sum()), int(n_true.sum())
    costs = assigned_costs(dist, counts, n_true)

    greedy, hungarian = {}, {}
    for t in THRESHOLDS_INLIER:
        matched = np.zeros(gt_valid.shape, dtype=bool)
        tp = fp = 0
        for p in range(max_planes):
            for g in range(max_gt):
                active = (p < counts) & gt_valid[:, g] & ~matched[:, g]
                hit = active & (dist[:, p, g] < t)
                matched[:, g] |= hit
                tp += int(hit.sum())
                fp += int(active.sum()) - int(hit.sum())
        greedy[t] = (tp, fp, n_gt_total - tp)
        tp_h = int((costs < t).sum())
        hungarian[t] = (tp_h, n_pred_total - tp_h, n_gt_total - tp_h)
    return greedy, hungarian


def adjusted_metrics(recall: float, precision: float, coverage: float) -> dict[str, float]:
    """ Coverage-adjusted recall and F1 as defined in the thesis (R_adj, F1_adj).

    Args:
        * recall: mean recall over objects with a prediction
        * precision: mean precision over objects with a prediction
        * coverage: fraction of objects with a prediction

    Returns:
        * dict[str, float]: recall_adj and f1_adj

    """
    recall_adj = recall * coverage
    total = recall_adj + precision
    f1_adj = 2 * recall_adj * precision / total if total else 0.0
    return {"recall_adj": recall_adj, "f1_adj": f1_adj}


def _object_mask(rng: np.random.Generator, n_objects: int, has_prediction: np.ndarray | None,
                 n_with_prediction: int | None) -> np.ndarray:
    """ Objects that receive planes: an exact mask, a random subset, or all of them.

    Args:
        * rng: NumPy random generator
        * n_objects: total objects
        * has_prediction: (N,) exact mask, or None
        * n_with_prediction: size of a random subset, used when there is no mask

    Returns:
        * np.ndarray: (N,) boolean mask

    """
    if has_prediction is not None:
        return np.asarray(has_prediction, dtype=bool)
    if n_with_prediction is None or n_with_prediction == n_objects:
        return np.ones(n_objects, dtype=bool)
    mask = np.zeros(n_objects, dtype=bool)
    mask[rng.choice(n_objects, size=n_with_prediction, replace=False)] = True
    return mask


def _object_counts(rng: np.random.Generator, mask: np.ndarray, plane_counts: np.ndarray | None,
                   mean_planes: float | None) -> np.ndarray:
    """ Planes per selected object: the method's counts when known, else a target mean.

    Args:
        * rng: NumPy random generator
        * mask: (N,) objects that receive planes
        * plane_counts: (N,) planes the method predicted, 0 where it predicted nothing
        * mean_planes: mean planes per object, used when plane_counts is None

    Returns:
        * np.ndarray: (mask.sum(),) planes per selected object

    """
    if plane_counts is None:
        return draw_plane_counts(rng, int(mask.sum()), mean_planes)
    counts = plane_counts[mask].copy()
    missing = counts == 0
    if missing.any():
        counts[missing] = rng.choice(plane_counts[plane_counts > 0], size=int(missing.sum()))
    return counts


def simulate_random_multiplane(
    gt: GroundTruth, n_repetitions: int = 1000, seed: int = 0,
    has_prediction: np.ndarray | None = None, n_with_prediction: int | None = None,
    plane_counts: np.ndarray | None = None, mean_planes: float | None = None,
    min_separation_deg: float = DUP_ANGLE_THRESH_DEFAULT,
) -> dict[str, np.ndarray]:
    """ Score the random plane-set predictor n_repetitions times on the given objects.

    Objects without a prediction add nothing to any metric except the adjusted
    ones, as in compute_summary. Selected objects where plane_counts is 0 draw
    their count from the method's own counts in every repetition.

    Args:
        * gt: plane_sym ground truth
        * n_repetitions: number of simulated datasets
        * seed: RNG seed
        * has_prediction: (N,) exact mask of objects that receive planes
        * n_with_prediction: size of a random subset redrawn per repetition; all when both are None
        * plane_counts: (N,) planes the method predicted per object
        * mean_planes: mean planes per object, when plane_counts is unknown
        * min_separation_deg: smallest angle allowed between planes of one object

    Returns:
        * dict[str, np.ndarray]: metric name -> (n_repetitions,) simulated values

    """
    if gt.planes is None:
        raise ValueError("multi-plane baseline needs plane_sym ground truth")
    if (plane_counts is None) == (mean_planes is None):
        raise ValueError("give exactly one of plane_counts or mean_planes")

    rng = np.random.default_rng(seed)
    results: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_repetitions):
        mask = _object_mask(rng, gt.n_objects, has_prediction, n_with_prediction)
        counts = _object_counts(rng, mask, plane_counts, mean_planes)
        normals = random_plane_sets(rng, counts, min_separation_deg)

        recall, precision = multiset_scores(normals, counts, gt.vectors[mask])
        recall_mean, precision_mean = float(recall.mean()), float(precision.mean())
        results["n_planes_predicted_mean"].append(float(counts.mean()))
        results["recall_planes_mean"].append(recall_mean)
        results["precision_planes_mean"].append(precision_mean)
        adjusted = adjusted_metrics(recall_mean, precision_mean, float(mask.mean()))
        for name, value in adjusted.items():
            results[name].append(value)

        greedy, hungarian = multiset_f1_counts(normals, counts, gt.planes[mask])
        results["f1_ref"].append(_f1_from_counts_by_threshold(greedy))
        results["f1_ref_hungarian"].append(_f1_from_counts_by_threshold(hungarian))

    return {name: np.array(values) for name, values in results.items()}
