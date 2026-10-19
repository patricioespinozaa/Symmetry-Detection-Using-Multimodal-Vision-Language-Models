# Consistency checks for the random baseline: the vectorized scoring must give
# exactly what Mapping/evaluate.py gives, and the simulation must reproduce the
# closed-form values. Synthetic GT only, no dataset needed.
#
#   python RandomBaseline/tests/test_random_baseline.py     (or pytest)
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import chance  # noqa: E402
from significance import compare_metric  # noqa: E402
from simulation import (  # noqa: E402
    ANGULAR_THRESHOLDS, AUC_ANGULAR_MAX, THRESHOLDS_INLIER, GroundTruth, best_match_errors,
    f1_counts, random_directions, simulate_random_baseline,
)
from evaluate import (  # noqa: E402
    angular_error_deg, f1_match_counts, f1_match_counts_hungarian, normal_origin_to_plane,
)


def _synthetic_gt(symmetry_type: str, n_elements: list[int], seed: int = 1) -> GroundTruth:
    """ Random GT with the given number of elements per object; planes get a random offset.

    Args:
        * symmetry_type: "axis_sym" or "plane_sym"
        * n_elements: GT elements per object
        * seed: RNG seed

    Returns:
        * GroundTruth: synthetic ground truth

    """
    rng = np.random.default_rng(seed)
    n, k = len(n_elements), max(n_elements)
    vectors = np.full((n, k, 3), np.nan)
    planes = np.full((n, k, 4), np.nan) if symmetry_type == "plane_sym" else None
    for i, m in enumerate(n_elements):
        vectors[i, :m] = random_directions(rng, m)
        if planes is not None:
            for j in range(m):
                origin = rng.normal(scale=0.05, size=3)
                planes[i, j] = normal_origin_to_plane(vectors[i, j], origin).reshape(4)
    return GroundTruth(symmetry_type, [f"obj{i}" for i in range(n)], vectors, planes,
                       np.array(n_elements))


def test_best_match_matches_evaluate() -> None:
    """ best_match_errors == min over GT of evaluate.angular_error_deg (rounded to 4 dp). """
    gt = _synthetic_gt("plane_sym", [1, 2, 3] * 20)
    directions = random_directions(np.random.default_rng(2), gt.n_objects)
    ours = best_match_errors(directions, gt.vectors)
    for i in range(gt.n_objects):
        ref = min(angular_error_deg(directions[i], gt.vectors[i, j])
                  for j in range(gt.n_elements[i]))
        assert abs(ours[i] - round(ref, 4)) < 1e-9


def test_f1_counts_match_evaluate() -> None:
    """ Vectorized F1 counts == summed evaluate.f1_match_counts / f1_match_counts_hungarian. """
    gt = _synthetic_gt("plane_sym", [1, 2, 3] * 200)
    rng = np.random.default_rng(3)
    directions = random_directions(rng, gt.n_objects)
    # half the predictions are near-copies of a GT normal so that every threshold sees matches
    near = rng.random(gt.n_objects) < 0.5
    directions[near] = gt.vectors[near, 0] + rng.normal(scale=0.03, size=(near.sum(), 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    greedy, hungarian = f1_counts(directions, gt.planes)
    for t in THRESHOLDS_INLIER:
        ref_g, ref_h = np.zeros(3, dtype=int), np.zeros(3, dtype=int)
        for i in range(gt.n_objects):
            pred = [np.append(directions[i], 0.0).reshape(1, 4)]
            gts = [gt.planes[i, j].reshape(1, 4) for j in range(gt.n_elements[i])]
            ref_g += f1_match_counts(pred, gts, t)
            ref_h += f1_match_counts_hungarian(pred, gts, t)
        assert tuple(ref_g) == greedy[t], (t, ref_g, greedy[t])
        assert tuple(ref_h) == hungarian[t], (t, ref_h, hungarian[t])
        assert greedy[t][0] > 0


def test_simulation_matches_closed_form() -> None:
    """ Single-GT simulation reproduces the closed-form values at full and partial coverage. """
    gt = _synthetic_gt("axis_sym", [1] * 850)
    for coverage in (1.0, 0.8):
        sim = simulate_random_baseline(gt, n_repetitions=400, seed=0,
                                       n_with_prediction=int(round(coverage * 850)))
        analytic = chance.analytic_summary(ANGULAR_THRESHOLDS, AUC_ANGULAR_MAX, coverage)
        assert abs(sim["angular_error_mean"].mean() - analytic["angular_error_mean"]) < 0.2
        assert abs(sim["angular_error_median"].mean() - analytic["angular_error_median"]) < 0.5
        assert abs(sim["angular_error_std"].mean() - analytic["angular_error_std"]) < 0.3
        assert abs(sim["auc_angular"].mean() - analytic["auc_angular"]) < 0.005
        for t in ANGULAR_THRESHOLDS:
            key = f"precision_{t}deg"
            assert abs(sim[key].mean() - analytic[key]) < 0.003
        # spread of the dataset mean at full coverage: 21.56 deg / sqrt(850)
        if coverage == 1.0:
            spread = chance.std_error_deg() / math.sqrt(850)
            assert abs(sim["angular_error_mean"].std() - spread) < 0.1


def test_closed_form_values() -> None:
    """ Closed-form values quoted in docs/literature/baseline_aleatorio.md. """
    assert abs(chance.mean_error_deg() - 57.2958) < 1e-3
    assert abs(chance.median_error_deg() - 60.0) < 1e-9
    assert abs(chance.std_error_deg() - 21.5597) < 1e-3   # sqrt(pi - 3) rad
    assert abs(chance.auc_angular(45.0) - 0.0997) < 1e-4
    assert abs(chance.precision_at(10.0) - 0.0152) < 1e-4
    assert abs(chance.mean_error_deg(800 / 850) - 59.22) < 0.01


def test_multi_plane_is_easier() -> None:
    """ Expected random error drops with more perpendicular GT planes: ~57.3 / 42.0 / 31.9. """
    eye = np.eye(3)
    for n_planes, chance_mean in ((1, 57.30), (2, 42.0), (3, 31.9)):
        vectors = np.full((2000, 3, 3), np.nan)
        vectors[:, :n_planes] = eye[:n_planes]
        gt = GroundTruth("plane_sym", [str(i) for i in range(2000)], vectors, None,
                         np.full(2000, n_planes))
        sim = simulate_random_baseline(gt, n_repetitions=100, seed=0)
        assert abs(sim["angular_error_mean"].mean() - chance_mean) < 0.3


def test_compare_metric_orientation() -> None:
    """ A clearly better method is flagged for lower- and higher-is-better metrics. """
    rng = np.random.default_rng(0)
    random_err = rng.normal(57.3, 0.74, size=1000)
    assert compare_metric(25.0, random_err, higher_is_better=False)["better_than_random"]
    assert not compare_metric(57.3, random_err, higher_is_better=False)["better_than_random"]
    random_auc = rng.normal(0.1, 0.01, size=1000)
    assert compare_metric(0.6, random_auc, higher_is_better=True)["better_than_random"]
    assert not compare_metric(0.05, random_auc, higher_is_better=True)["better_than_random"]


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed")
