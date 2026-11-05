# Consistency checks for the random baseline: the vectorized scoring must give
# exactly what Mapping/evaluate.py gives, and the simulation must reproduce the
# closed-form values. Synthetic GT only, no dataset needed.
#
#   python RandomBaseline/tests/test_random_baseline.py     (or pytest)
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import chance  # noqa: E402
import latex_tables  # noqa: E402
import run_baseline  # noqa: E402
from significance import compare_metric  # noqa: E402
from simulation import (  # noqa: E402
    ANGULAR_THRESHOLDS, AUC_ANGULAR_MAX, THRESHOLDS_INLIER, GroundTruth, best_match_errors,
    f1_counts, random_directions, simulate_random_baseline,
)
from multiplane import (  # noqa: E402
    adjusted_metrics, draw_plane_counts, multiset_f1_counts, multiset_scores, random_plane_sets,
    simulate_random_multiplane,
)
from evaluate import (  # noqa: E402
    angular_error_deg, compute_summary, evaluate_plane_multiset, f1_match_counts,
    f1_match_counts_hungarian, normal_origin_to_plane,
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


def _plane_sets_near_gt(gt: GroundTruth, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """ Plane sets of 1-3 planes per object; some sit near a GT normal so that matches occur.

    Args:
        * gt: plane_sym ground truth
        * seed: RNG seed

    Returns:
        * tuple[np.ndarray, np.ndarray]: (M, 3, 3) unit normals and (M,) planes per object

    """
    rng = np.random.default_rng(seed)
    counts = rng.integers(1, 4, size=gt.n_objects)
    normals = random_plane_sets(rng, counts)
    for i in range(gt.n_objects):
        for p in range(counts[i]):
            if rng.random() < 0.5:
                j = rng.integers(gt.n_elements[i])
                noisy = gt.vectors[i, j] + rng.normal(scale=0.04, size=3)
                normals[i, p] = noisy / np.linalg.norm(noisy)
    return normals, counts


def test_multiset_scores_match_evaluate() -> None:
    """ Vectorized recall/precision == evaluate_plane_multiset on every object. """
    gt = _synthetic_gt("plane_sym", [1, 2, 3] * 150)
    normals, counts = _plane_sets_near_gt(gt, 4)
    recall, precision = multiset_scores(normals, counts, gt.vectors)
    n_matches = 0
    for i in range(gt.n_objects):
        pred = [{"normal": normals[i, p].tolist(), "origin": [0.0, 0.0, 0.0]}
                for p in range(counts[i])]
        true = [{"normal": gt.vectors[i, j].tolist(), "origin": [0.0, 0.0, 0.0]}
                for j in range(gt.n_elements[i])]
        ref = evaluate_plane_multiset(pred, true)
        assert ref["recall_planes"] == recall[i], (i, ref, recall[i])
        assert ref["precision_planes"] == precision[i], (i, ref, precision[i])
        n_matches += ref["n_planes_matched"]
    assert n_matches > 0


def test_multiset_f1_counts_match_evaluate() -> None:
    """ Vectorized multi-plane F1 counts == summed evaluate.py counts, greedy and Hungarian. """
    gt = _synthetic_gt("plane_sym", [1, 2, 3] * 100)
    normals, counts = _plane_sets_near_gt(gt, 5)
    greedy, hungarian = multiset_f1_counts(normals, counts, gt.planes)
    for t in THRESHOLDS_INLIER:
        ref_g, ref_h = np.zeros(3, dtype=int), np.zeros(3, dtype=int)
        for i in range(gt.n_objects):
            pred = [np.append(normals[i, p], 0.0).reshape(1, 4) for p in range(counts[i])]
            gts = [gt.planes[i, j].reshape(1, 4) for j in range(gt.n_elements[i])]
            ref_g += f1_match_counts(pred, gts, t)
            ref_h += f1_match_counts_hungarian(pred, gts, t)
        assert tuple(ref_g) == greedy[t], (t, ref_g, greedy[t])
        assert tuple(ref_h) == hungarian[t], (t, ref_h, hungarian[t])
        assert greedy[t][0] > 0


def test_one_plane_equals_single_plane_counts() -> None:
    """ With one plane per object the multi-plane F1 counts equal the single-plane ones. """
    gt = _synthetic_gt("plane_sym", [1, 2, 3] * 100)
    directions = random_directions(np.random.default_rng(6), gt.n_objects)
    counts = np.ones(gt.n_objects, dtype=int)
    assert multiset_f1_counts(directions[:, None, :], counts, gt.planes) == f1_counts(
        directions, gt.planes)


def test_random_recall_closed_form() -> None:
    """ With 1 GT and K independent planes: recall = 1 - (1 - p)^K, with p = 1 - cos(15 deg). """
    gt = _synthetic_gt("plane_sym", [1] * 850)
    p = 1 - math.cos(math.radians(15))
    for k in (1, 2, 3):
        sim = simulate_random_multiplane(gt, 300, 0, mean_planes=k, min_separation_deg=0)
        expected = 1 - (1 - p) ** k
        assert abs(sim["recall_planes_mean"].mean() - expected) < 0.004
        assert abs(sim["precision_planes_mean"].mean() - expected / k) < 0.002
        assert np.allclose(sim["recall_adj"], sim["recall_planes_mean"])


def test_plane_sets_keep_separation() -> None:
    """ Every pair of planes used by an object respects the minimum angle. """
    rng = np.random.default_rng(7)
    counts = rng.integers(1, 4, size=2000)
    normals = random_plane_sets(rng, counts, min_separation_deg=15.0)
    for i in np.flatnonzero(counts > 1):
        for a in range(counts[i]):
            for b in range(a + 1, counts[i]):
                assert angular_error_deg(normals[i, a], normals[i, b]) >= 15.0


def test_plane_counts_keep_mean() -> None:
    """ Drawn counts keep the mean and differ by at most one plane. """
    counts = draw_plane_counts(np.random.default_rng(8), 850, 2.3596)
    assert abs(counts.mean() - 2.3596) < 1 / 850
    assert counts.max() - counts.min() <= 1


def test_exact_counts_and_mask_are_used() -> None:
    """ Exact counts apply to their objects; missing ones draw from the method's counts. """
    gt = _synthetic_gt("plane_sym", [1, 2] * 100)
    plane_counts = np.array([3] * 100 + [0] * 100)
    mask = plane_counts > 0
    matched = simulate_random_multiplane(gt, 5, 0, has_prediction=mask, plane_counts=plane_counts)
    assert np.allclose(matched["n_planes_predicted_mean"], 3.0)
    full = simulate_random_multiplane(gt, 5, 0, plane_counts=plane_counts)
    assert np.allclose(full["n_planes_predicted_mean"], 3.0)
    assert np.allclose(matched["recall_adj"], matched["recall_planes_mean"] * 0.5)


def test_adjusted_metrics_formula() -> None:
    """ R_adj and F1_adj follow the thesis definition. """
    adjusted = adjusted_metrics(0.8, 0.5, 0.5)
    assert abs(adjusted["recall_adj"] - 0.4) < 1e-12
    assert abs(adjusted["f1_adj"] - 2 * 0.4 * 0.5 / 0.9) < 1e-12
    assert adjusted_metrics(0.0, 0.0, 1.0)["f1_adj"] == 0.0


def test_compute_summary_agrees_with_simulation_scoring() -> None:
    """ compute_summary on per-object results gives the same recall and precision means. """
    gt = _synthetic_gt("plane_sym", [1, 2, 3] * 40)
    normals, counts = _plane_sets_near_gt(gt, 9)
    recall, precision = multiset_scores(normals, counts, gt.vectors)
    objects = {}
    for i in range(gt.n_objects):
        pred = [{"normal": normals[i, p].tolist(), "origin": [0.0, 0.0, 0.0]}
                for p in range(counts[i])]
        true = [{"normal": gt.vectors[i, j].tolist(), "origin": [0.0, 0.0, 0.0]}
                for j in range(gt.n_elements[i])]
        objects[f"obj{i}"] = {"14": {**evaluate_plane_multiset(pred, true), "status": "ok"}}
    summary = compute_summary(objects, "plane_sym", n_total=gt.n_objects)["14"]
    assert abs(summary["recall_planes_mean"] - recall.mean()) < 1e-4
    assert abs(summary["precision_planes_mean"] - precision.mean()) < 1e-4


def _compare_args(mode: str) -> argparse.Namespace:
    """ Arguments of the compare subcommand with few repetitions. """
    return argparse.Namespace(repetitions=20, seed=0, coverage_mode=mode)


def test_compare_summary_csv_multiplane_row() -> None:
    """ A multi-plane CSV row is compared, with adjusted metrics, instead of being skipped. """
    gt = _synthetic_gt("plane_sym", [1, 2] * 50)
    row = {"experiment": "exp", "method": "triangulation_multiplane", "n_views": 14,
           "n_total": 100, "n_objects": 50, "n_planes_predicted_mean": 2.5,
           "recall_planes_mean": 0.8, "precision_planes_mean": 0.5, "f1_ref": 0.4,
           "f1_ref_hungarian": ""}
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "summary.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        for mode, coverage in (("all", 1.0), ("matched", 0.5)):
            rows = run_baseline.compare_summary_csv(gt, path, _compare_args(mode))
            by_metric = {r["metric"]: r for r in rows}
            assert set(by_metric) == {"recall_planes_mean", "precision_planes_mean", "recall_adj",
                                      "f1_adj", "f1_ref"}
            assert abs(by_metric["recall_adj"]["method_value"] - 0.4) < 1e-12
            assert by_metric["f1_ref"]["n_planes_mean"] == 2.5
            assert by_metric["f1_ref"]["random_coverage"] == coverage


def test_compare_eval_results_multiplane() -> None:
    """ Per-object multi-plane results are compared using the method's exact plane counts. """
    gt = _synthetic_gt("plane_sym", [1, 2] * 20)
    objects = {}
    for i, object_id in enumerate(gt.object_ids):
        ok = {"status": "ok", "n_planes_predicted": 2, "n_true_planes": int(gt.n_elements[i]),
              "n_planes_matched": 1, "recall_planes": 0.5, "precision_planes": 0.5}
        objects[object_id] = {"14": ok} if i % 2 == 0 else None
    original = run_baseline.load_ground_truth
    run_baseline.load_ground_truth = lambda *_: GroundTruth(
        "plane_sym", sorted(gt.object_ids),
        gt.vectors[np.argsort(gt.object_ids)], gt.planes[np.argsort(gt.object_ids)],
        gt.n_elements[np.argsort(gt.object_ids)])
    try:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "eval_results.json"
            path.write_text(json.dumps({"method": "m", "objects": objects}), encoding="utf-8")
            rows = run_baseline.compare_eval_results(Path(folder), "plane_sym", path,
                                                     _compare_args("matched"))
    finally:
        run_baseline.load_ground_truth = original
    by_metric = {r["metric"]: r for r in rows}
    assert by_metric["recall_planes_mean"]["n_objects"] == 20
    assert by_metric["recall_planes_mean"]["random_coverage"] == 0.5
    assert abs(by_metric["recall_adj"]["method_value"] - 0.25) < 1e-12


def test_plane_multi_table() -> None:
    """ The multi-plane LaTeX table has one column per K and one row per metric. """
    gt = _synthetic_gt("plane_sym", [1, 2] * 50)
    simulated = {k: simulate_random_multiplane(gt, 10, 0, mean_planes=k) for k in (1, 2, 3)}
    table = latex_tables.plane_multi_table(gt, simulated, 10, 15.0)
    assert "tab:plane_azar_multi" in table
    assert "3 planos por objeto" in table
    assert table.count(r"\\") == len(latex_tables.MULTIPLANE_ROWS) + 1


def _write_labels(root: Path) -> None:
    """ Writes synthetic labels in the dataset's .txt format, with known answers.

    Args:
        * root: folder that will hold curated_axis_sym_obj/ and curated_plane_sym_obj/

    """
    axis_dir, plane_dir = root / "curated_axis_sym_obj", root / "curated_plane_sym_obj"
    axis_dir.mkdir()
    plane_dir.mkdir()
    for i in range(20):
        (axis_dir / f"a{i}.txt").write_text("1\naxis 0 1 0 0 0 0\n4\nangles 180 90\n")
        (plane_dir / f"p{i}.txt").write_text("1\nplane 1 0 0 -0.0 0.0 0.0\n")
    for i in range(20, 25):
        (plane_dir / f"p{i}.txt").write_text("2\nplane 1 0 0 0 0 0\nplane 0 0 1 0 0 0\n")


def test_cap4_file_has_both_baselines() -> None:
    """ The cap4 file holds all tables, and fixed axes score as the labels dictate. """
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        _write_labels(root)
        out = root / "salida" / "cap4_baselines.tex"
        run_baseline.run_cap4(argparse.Namespace(objects_root=str(root), repetitions=5, seed=0,
                                                 out=str(out)))
        text = out.read_text(encoding="utf-8")
    for label in ("axis_azar", "axis_fijos", "plane_azar", "plane_azar_multi", "plane_fijos",
                  "plane_fijos_multi"):
        assert rf"\label{{tab:base_{label}}}" in text, label
    assert text.count(r"\begin{table}") == 6
    assert "25 de las 30" in text or "20 de las 30" in text
    # Y predicts the axis exactly, and X is exact on every plane object; a perfect AUC is
    # 0.995 because auc_from_errors counts errors strictly below each threshold, 0 included
    assert r"Siempre $Y$ & 0.995 & 0.0 & 0.0 & 1.000 & 1.000 & 1.000 \\" in text
    assert r"Siempre $X$ & 0.995 & 0.0 & 0.0 & 1.000 & 1.000 & 1.000 \\" in text
    # {X,Z} finds all planes: recall 1, precision (20 * 0.5 + 5 * 1) / 25, F1 from global counts
    assert r"Siempre $\{X,Z\}$ & 2 & 1.000 & 0.600 &" in text


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed")
