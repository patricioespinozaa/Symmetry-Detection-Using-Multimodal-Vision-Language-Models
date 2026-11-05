# CLI for the random baseline (docs/literature/baseline_aleatorio.md).
#
#   reference: chance scores of the random predictor on the dataset itself
#              (closed form + simulation over the real GT planes), optionally
#              at several coverages.
#   compare:   compares evaluate.py outputs against the random predictor,
#              read from summary/comparison CSVs or from eval_*_results.json
#              (the latter knows exactly which objects the method missed,
#              used by --coverage-mode matched). By default the random
#              predictor answers EVERY object (--coverage-mode all), while the
#              method keeps its 90 deg imputation for the objects it missed:
#              the strictest comparison for the method. --coverage-mode matched
#              gives the random predictor the method's coverage instead.
#              Multi-plane rows are compared against a random predictor with
#              the method's number of planes (RandomBaseline/multiplane.py).
#   latex:     LaTeX tables of the random baseline on the full dataset: axial,
#              planar with one plane, and planar with several planes.
#   cap4:      one .tex with both baselines on the full dataset: random predictors
#              and fixed world-axis predictors (Pipeline_Experiments/diagnostics/
#              trivial_baselines.py), by default Tesis_DCC_MDS/cap4_baselines.tex.
#
#   python RandomBaseline/run_baseline.py reference --objects-root ../data/objects
#   python RandomBaseline/run_baseline.py reference --objects-root ../data/objects \
#       --symmetry-type plane_sym --n-planes 1 2 3
#   python RandomBaseline/run_baseline.py latex --objects-root ../data/objects
#   python RandomBaseline/run_baseline.py cap4 --objects-root ../data/objects
#   python RandomBaseline/run_baseline.py compare --objects-root ../data/objects \
#       --symmetry-type axis_sym --summary-csv <comparison.csv> [...]
#   python RandomBaseline/run_baseline.py compare --objects-root ../data/objects \
#       --symmetry-type plane_sym --eval-results <eval_..._results.json> [...]
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chance  # noqa: E402
import latex_tables  # noqa: E402
from significance import COMPARED_METRICS, MULTIPLANE_METRICS, compare_metric  # noqa: E402
from simulation import (  # noqa: E402
    ANGULAR_THRESHOLDS, AUC_ANGULAR_MAX, GroundTruth, best_match_errors,
    load_ground_truth, simulate_random_baseline, summarize_errors,
)
from multiplane import (  # noqa: E402
    DUP_ANGLE_THRESH_DEFAULT, adjusted_metrics, simulate_random_multiplane,
)
from evaluate import compute_summary  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "Pipeline_Experiments" / "diagnostics"))
from trivial_baselines import (  # noqa: E402
    AXIS_PREDICTORS, PLANE_PREDICTORS, WORLD_AXES, load_labels, orientation_report,
    score_plane_predictor,
)

SYMMETRY_TYPES = ("axis_sym", "plane_sym")
REFERENCE_METRICS = ["angular_error_mean", "angular_error_median", "angular_error_std",
                     "auc_angular", *[f"precision_{t}deg" for t in ANGULAR_THRESHOLDS],
                     "f1_ref", "f1_ref_hungarian"]
COVERAGE_MODES = ("all", "matched")
COMPARE_FIELDS = ["experiment", "method", "n_views", "n_total", "n_objects", "coverage",
                  "random_coverage",
                  "metric", "method_value", "random_mean", "random_std", "random_ci_low",
                  "random_ci_high", "z", "p_empirical", "p_normal", "better_than_random",
                  "n_planes_mean"]
LATEX_N_PLANES = (1, 2, 3)
CAP4_OUTPUT = REPO / "Tesis_DCC_MDS" / "cap4_baselines.tex"
CAP4_HEADER = ("% Generado por RandomBaseline/run_baseline.py cap4. "
               "Requiere booktabs y adjustbox.\n\n")


def _to_float(value: str | float | None) -> float | None:
    """ Parse a CSV cell into a float, None for empty/missing cells. """
    if value is None or value == "":
        return None
    return float(value)


def _write_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    """ Write rows to path as CSV, floats with 6 significant digits (keeps tiny p-values).

    Args:
        * rows: one dict per output row
        * path: destination file (parent folders are created)
        * fieldnames: column order

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: float(f"{v:.6g}") if isinstance(v, float) else v
                             for k, v in row.items()})
    print(f"Saved: {path}")


def _gt_profile(gt: GroundTruth) -> str:
    """ One-line description of how many GT elements the objects have. """
    counts = Counter(int(k) for k in gt.n_elements)
    parts = ", ".join(f"{k} GT: {counts[k]}" for k in sorted(counts))
    return f"{gt.n_objects} objects ({parts})"


def method_multiplane_metrics(values: dict, coverage: float) -> dict[str, float | None]:
    """ A method's multi-plane summary values plus its coverage-adjusted recall and F1.

    Args:
        * values: summary values, already parsed into floats or None
        * coverage: fraction of objects the method predicted

    Returns:
        * dict[str, float | None]: value per MULTIPLANE_METRICS key

    """
    metrics = {m: values.get(m) for m in MULTIPLANE_METRICS}
    recall, precision = values.get("recall_planes_mean"), values.get("precision_planes_mean")
    if recall is not None and precision is not None:
        metrics.update(adjusted_metrics(recall, precision, coverage))
    return metrics


def compare_rows(gt: GroundTruth, base: dict, method_metrics: dict[str, float | None],
                 random_metrics: dict[str, np.ndarray],
                 compared: dict[str, bool] = COMPARED_METRICS) -> list[dict]:
    """ Build one output row per metric present for both the method and the random baseline.

    Args:
        * gt: ground truth (only used for its symmetry type)
        * base: experiment/method/n_views/coverage columns shared by every row
        * method_metrics: the method's summary values
        * random_metrics: simulated random-baseline values per metric
        * compared: metric -> True when higher is better

    Returns:
        * list[dict]: rows with COMPARE_FIELDS columns

    """
    rows = []
    for metric, higher_is_better in compared.items():
        if metric.startswith("f1_ref") and gt.symmetry_type != "plane_sym":
            continue
        value = method_metrics.get(metric)
        if value is None or metric not in random_metrics:
            continue
        stats = compare_metric(value, random_metrics[metric], higher_is_better)
        rows.append({**base, "metric": metric, "method_value": value, **stats})
    return rows


def print_compare(rows: list[dict]) -> None:
    """ Print the comparison as a console table. """
    header = (f"{'experiment':<34} {'nv':>3} {'cov':>6} {'metric':<22} {'method':>9} "
              f"{'random':>9} {'random 95% CI':>19} {'p':>7}  better")
    print(header)
    print("-" * len(header))
    for r in rows:
        ci = f"[{r['random_ci_low']:.4f}, {r['random_ci_high']:.4f}]"
        print(f"{r['experiment'][:34]:<34} {r['n_views']:>3} {r['coverage']:>6.3f} "
              f"{r['metric']:<22} {r['method_value']:>9.4f} {r['random_mean']:>9.4f} "
              f"{ci:>19} {r['p_empirical']:>7.4f}  {'yes' if r['better_than_random'] else 'no'}")


def run_reference(args: argparse.Namespace) -> None:
    """ Chance scores of the random predictor on the dataset, closed form and simulated.

    Args:
        * args: parsed CLI arguments of the "reference" subcommand

    """
    rows = []
    for symmetry_type in args.symmetry_type:
        gt = load_ground_truth(Path(args.objects_root), symmetry_type)
        print(f"\n=== {symmetry_type}: {_gt_profile(gt)} ===")
        print(f"{'coverage':>8} {'metric':<22} {'closed form*':>12} {'simulated':>10} "
              f"{'95% interval':>19}")
        for coverage in args.coverage:
            n_with = int(round(coverage * gt.n_objects))
            simulated = simulate_random_baseline(gt, args.repetitions, args.seed,
                                                 n_with_prediction=n_with)
            analytic = chance.analytic_summary(ANGULAR_THRESHOLDS, AUC_ANGULAR_MAX, coverage)
            for metric in REFERENCE_METRICS:
                if metric not in simulated:
                    continue
                values = simulated[metric]
                row = {
                    "symmetry_type": symmetry_type,
                    "n_objects_gt": gt.n_objects,
                    "coverage": coverage,
                    "metric": metric,
                    "closed_form_single_gt": analytic.get(metric),
                    "random_mean": float(values.mean()),
                    "random_std": float(values.std(ddof=1)),
                    "random_ci_low": float(np.percentile(values, 2.5)),
                    "random_ci_high": float(np.percentile(values, 97.5)),
                }
                rows.append(row)
                closed = f"{row['closed_form_single_gt']:.4f}" if metric in analytic else "-"
                ci = f"[{row['random_ci_low']:.4f}, {row['random_ci_high']:.4f}]"
                print(f"{coverage:>8.3f} {metric:<22} {closed:>12} "
                      f"{row['random_mean']:>10.4f} {ci:>19}")
            if symmetry_type == "plane_sym":
                rows.extend(reference_multiplane_rows(gt, args, coverage, n_with))
    print("\n* closed form assumes ONE GT element per object; the simulation uses the real "
          "GT planes (best match), so for plane_sym the simulated column is the valid one.")
    if args.out:
        _write_csv(rows, Path(args.out), list(dict.fromkeys(k for r in rows for k in r)))


def reference_multiplane_rows(gt: GroundTruth, args: argparse.Namespace, coverage: float,
                              n_with: int) -> list[dict]:
    """ Chance scores with a fixed number of random planes per object (--n-planes).

    Args:
        * gt: plane_sym ground truth
        * args: parsed CLI arguments of the "reference" subcommand
        * coverage: fraction of objects that receive planes
        * n_with: number of objects that receive planes

    Returns:
        * list[dict]: one row per (n_planes, metric)

    """
    rows = []
    for n_planes in args.n_planes:
        simulated = simulate_random_multiplane(gt, args.repetitions, args.seed,
                                               n_with_prediction=n_with, mean_planes=n_planes)
        for metric in MULTIPLANE_METRICS:
            values = simulated[metric]
            row = {
                "symmetry_type": gt.symmetry_type,
                "n_objects_gt": gt.n_objects,
                "coverage": coverage,
                "n_planes": n_planes,
                "metric": metric,
                "random_mean": float(values.mean()),
                "random_std": float(values.std(ddof=1)),
                "random_ci_low": float(np.percentile(values, 2.5)),
                "random_ci_high": float(np.percentile(values, 97.5)),
            }
            rows.append(row)
            ci = f"[{row['random_ci_low']:.4f}, {row['random_ci_high']:.4f}]"
            print(f"{coverage:>8.3f} {f'{metric} K={n_planes}':<26} {'-':>8} "
                  f"{row['random_mean']:>10.4f} {ci:>19}")
    return rows


def compare_summary_csv(gt: GroundTruth, csv_path: Path, args: argparse.Namespace) -> list[dict]:
    """ Compare every row of an evaluate.py summary CSV or compare_results comparison CSV.

    With --coverage-mode all the random predictor answers all n_total objects.
    With matched, only n_total/n_objects are known per row, so which objects
    the random predictor answers is redrawn at random in every repetition.
    Multi-plane rows only report the mean planes per object, so the random
    predictor keeps that mean with counts as even as possible.

    Args:
        * gt: ground truth of the full dataset (must have n_total objects)
        * csv_path: CSV with n_views, n_total, n_objects and summary metric columns
        * args: parsed CLI arguments of the "compare" subcommand

    Returns:
        * list[dict]: comparison rows

    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))

    rows, cache = [], {}
    for r in csv_rows:
        n_total, n_objects = int(r["n_total"]), int(r["n_objects"])
        if n_total != gt.n_objects:
            raise ValueError(f"{csv_path.name}: n_total={n_total} but {gt.n_objects} GT labels; "
                             "use --eval-results to get the exact object list")
        n_random = n_total if args.coverage_mode == "all" else n_objects
        mean_planes = _to_float(r.get("n_planes_predicted_mean"))
        if mean_planes is not None:
            key = (n_random, mean_planes)
            if key not in cache:
                cache[key] = simulate_random_multiplane(gt, args.repetitions, args.seed,
                                                        n_with_prediction=n_random,
                                                        mean_planes=mean_planes)
        else:
            key = n_random
            if key not in cache:
                cache[key] = simulate_random_baseline(gt, args.repetitions, args.seed,
                                                      n_with_prediction=n_random)
        base = {
            "experiment": r.get("experiment") or csv_path.stem,
            "method": r.get("method", ""),
            "n_views": int(r["n_views"]),
            "n_total": n_total,
            "n_objects": n_objects,
            "coverage": n_objects / n_total,
            "random_coverage": n_random / n_total,
        }
        if mean_planes is not None:
            base["n_planes_mean"] = mean_planes
            values = {m: _to_float(r.get(m)) for m in MULTIPLANE_METRICS}
            method_metrics = method_multiplane_metrics(values, n_objects / n_total)
            rows.extend(compare_rows(gt, base, method_metrics, cache[key], MULTIPLANE_METRICS))
        else:
            method_metrics = {m: _to_float(r.get(m)) for m in COMPARED_METRICS}
            rows.extend(compare_rows(gt, base, method_metrics, cache[key]))
    return rows


def compare_eval_results(objects_root: Path, symmetry_type: str, json_path: Path,
                         args: argparse.Namespace) -> list[dict]:
    """ Compare an eval_*_results.json against the random baseline.

    With --coverage-mode matched the random predictor answers exactly the
    objects the method answered; with all, every object. In multi-plane rows
    the random predictor gets the method's plane count on each object, and
    objects the method missed draw a count from the method's own counts.

    Method metrics are recomputed with evaluate.compute_summary from the same
    per-object results, so they match the summary CSV evaluate.py wrote.

    Args:
        * objects_root: folder holding curated_*_sym_obj/
        * symmetry_type: "axis_sym" or "plane_sym"
        * json_path: per-object results written by Mapping/evaluate.py
        * args: parsed CLI arguments of the "compare" subcommand

    Returns:
        * list[dict]: comparison rows

    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    objects = data["objects"]
    object_ids = sorted(objects)
    gt = load_ground_truth(objects_root, symmetry_type, object_ids)
    summary = compute_summary(objects, symmetry_type, n_total=len(object_ids))

    rows = []
    for nv, s in summary.items():
        per_object = [(objects[oid] or {}).get(nv, {}) for oid in object_ids]
        has_prediction = np.array([m.get("status") == "ok" for m in per_object])
        random_mask = has_prediction if args.coverage_mode == "matched" else None
        is_multiplane = "n_planes_predicted_mean" in s
        if is_multiplane:
            plane_counts = np.array([m.get("n_planes_predicted", 0) if m.get("status") == "ok"
                                     else 0 for m in per_object])
            random_metrics = simulate_random_multiplane(gt, args.repetitions, args.seed,
                                                        has_prediction=random_mask,
                                                        plane_counts=plane_counts)
        else:
            random_metrics = simulate_random_baseline(gt, args.repetitions, args.seed,
                                                      has_prediction=random_mask)
        base = {
            "experiment": json_path.stem,
            "method": data.get("method", ""),
            "n_views": int(nv),
            "n_total": len(object_ids),
            "n_objects": int(has_prediction.sum()),
            "coverage": float(has_prediction.mean()),
            "random_coverage": float(has_prediction.mean()) if random_mask is not None else 1.0,
        }
        if is_multiplane:
            base["n_planes_mean"] = s["n_planes_predicted_mean"]
            method_metrics = method_multiplane_metrics(s, float(has_prediction.mean()))
            rows.extend(compare_rows(gt, base, method_metrics, random_metrics, MULTIPLANE_METRICS))
        else:
            rows.extend(compare_rows(gt, base, s, random_metrics))
    return rows


def run_compare(args: argparse.Namespace) -> None:
    """ Compare evaluate.py outputs against the random baseline (see --coverage-mode).

    Args:
        * args: parsed CLI arguments of the "compare" subcommand

    """
    objects_root = Path(args.objects_root)
    rows = []

    if args.summary_csv:
        gt = load_ground_truth(objects_root, args.symmetry_type)
        print(f"{args.symmetry_type}: {_gt_profile(gt)}")
        for path in args.summary_csv:
            rows.extend(compare_summary_csv(gt, Path(path), args))

    for path in args.eval_results or []:
        rows.extend(compare_eval_results(objects_root, args.symmetry_type, Path(path), args))

    if not rows:
        print("No comparable rows found.")
        return
    print()
    print_compare(rows)
    _write_csv(rows, Path(args.out), COMPARE_FIELDS)


def run_latex(args: argparse.Namespace) -> None:
    """ Write the random-baseline LaTeX tables (azar_axis, azar_plane, azar_plane_multi).

    Args:
        * args: parsed CLI arguments of the "latex" subcommand

    """
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for symmetry_type in args.symmetry_type:
        gt = load_ground_truth(Path(args.objects_root), symmetry_type)
        simulated = simulate_random_baseline(gt, args.repetitions, args.seed)
        if symmetry_type == "axis_sym":
            closed_form = chance.analytic_summary(ANGULAR_THRESHOLDS, AUC_ANGULAR_MAX)
            table = latex_tables.axis_table(gt, simulated, closed_form, args.repetitions)
            path = out_dir / "azar_axis.tex"
        else:
            table = latex_tables.plane_table(gt, simulated, args.repetitions)
            path = out_dir / "azar_plane.tex"
        path.write_text(latex_tables.HEADER + table, encoding="utf-8")
        print(f"{symmetry_type}: {_gt_profile(gt)} -> Saved: {path}")
        if symmetry_type == "plane_sym":
            multi = {k: simulate_random_multiplane(gt, args.repetitions, args.seed, mean_planes=k)
                     for k in LATEX_N_PLANES}
            table = latex_tables.plane_multi_table(gt, multi, args.repetitions,
                                                   DUP_ANGLE_THRESH_DEFAULT)
            path = out_dir / "azar_plane_multi.tex"
            path.write_text(latex_tables.HEADER + table, encoding="utf-8")
            print(f"{symmetry_type}: multi-plane -> Saved: {path}")


def fixed_axis_scores(gt: GroundTruth, predictors: list[str]) -> dict[str, dict[str, float]]:
    """ Angular metrics of predicting the same world axis for every object.

    Args:
        * gt: ground truth; the error is measured against the closest GT element
        * predictors: world-axis names

    Returns:
        * dict[str, dict[str, float]]: LaTeX predictor label -> summarize_errors output

    """
    scores = {}
    for name in predictors:
        directions = np.tile(WORLD_AXES[name], (gt.n_objects, 1))
        errors = best_match_errors(directions, gt.vectors)
        scores[f"Siempre ${name}$"] = summarize_errors(errors)
    return scores


def cap4_tables(objects_root: Path, repetitions: int, seed: int) -> list[str]:
    """ Build the tables of both baselines, random and fixed world axes.

    Args:
        * objects_root: folder holding curated_*_sym_obj/
        * repetitions: simulated datasets per random baseline
        * seed: RNG seed

    Returns:
        * list[str]: LaTeX tables, with tab:base_* labels so that they can coexist with cap4.tex

    """
    axis_gt = load_ground_truth(objects_root, "axis_sym")
    plane_gt = load_ground_truth(objects_root, "plane_sym")
    print(f"axis_sym: {_gt_profile(axis_gt)}\nplane_sym: {_gt_profile(plane_gt)}")

    closed_form = chance.analytic_summary(ANGULAR_THRESHOLDS, AUC_ANGULAR_MAX)
    axis_random = simulate_random_baseline(axis_gt, repetitions, seed)
    plane_random = simulate_random_baseline(plane_gt, repetitions, seed)
    multi_random = {k: simulate_random_multiplane(plane_gt, repetitions, seed, mean_planes=k)
                    for k in LATEX_N_PLANES}

    plane_labels = load_labels(objects_root, "plane_sym")
    by_axis = orientation_report(plane_labels, "normal")
    n_x, n_normals = by_axis["by_axis"].get("X", 0), by_axis["n_elements"]
    plane_sets = []
    for axes in PLANE_PREDICTORS:
        row = score_plane_predictor(plane_labels, axes)
        name = axes if len(axes) == 1 else r"\{" + ",".join(axes) + r"\}"
        plane_sets.append({**row, "predictor": f"Siempre ${name}$"})

    return [
        latex_tables.axis_table(axis_gt, axis_random, closed_form, repetitions)
        .replace("tab:axis_azar", "tab:base_axis_azar"),
        latex_tables.trivial_angular_table(
            fixed_axis_scores(axis_gt, AXIS_PREDICTORS),
            "Línea base de ejes fijos para simetría axial: se predice siempre el mismo eje del "
            f"mundo, sin mirar las imágenes, para los {axis_gt.n_objects} objetos.",
            "tab:base_axis_fijos"),
        latex_tables.plane_table(plane_gt, plane_random, repetitions)
        .replace("tab:plane_azar", "tab:base_plane_azar"),
        latex_tables.plane_multi_table(plane_gt, multi_random, repetitions,
                                       DUP_ANGLE_THRESH_DEFAULT)
        .replace("tab:plane_azar_multi", "tab:base_plane_azar_multi"),
        latex_tables.trivial_angular_table(
            fixed_axis_scores(plane_gt, list(WORLD_AXES)),
            "Línea base de ejes fijos para simetría planar, un plano por objeto: la normal es "
            f"siempre el mismo eje del mundo y el plano pasa por el origen, para los "
            f"{plane_gt.n_objects} objetos. El error se mide contra el plano verdadero más "
            f"cercano. {n_x} de las {n_normals} normales verdaderas están más cerca de $X$ que "
            "de los otros ejes.",
            "tab:base_plane_fijos"),
        latex_tables.trivial_plane_set_table(
            plane_sets,
            "Línea base de ejes fijos para simetría planar por conjunto de planos: se predicen "
            "siempre los mismos planos del mundo, que pasan por el origen, para los "
            f"{plane_gt.n_objects} objetos. Recall y precisión usan el emparejamiento voraz a "
            r"$15^\circ$ de la evaluación multiplano.",
            "tab:base_plane_fijos_multi"),
    ]


def run_cap4(args: argparse.Namespace) -> None:
    """ Write one .tex with the tables of both baselines.

    Args:
        * args: parsed CLI arguments of the "cap4" subcommand

    """
    tables = cap4_tables(Path(args.objects_root), args.repetitions, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(CAP4_HEADER + "\n".join(tables), encoding="utf-8")
    print(f"Saved: {out} ({len(tables)} tables)")


def parse_args() -> argparse.Namespace:
    """ Parse CLI arguments. """
    p = argparse.ArgumentParser(description="Uniform random-direction baseline for axis/plane "
                                            "symmetry (docs/literature/baseline_aleatorio.md).")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--objects-root", required=True,
                        help="folder with curated_axis_sym_obj/ and curated_plane_sym_obj/")
    common.add_argument("--repetitions", type=int, default=1000,
                        help="simulated datasets per row (min. empirical p = 1/(R+1))")
    common.add_argument("--seed", type=int, default=0)

    ref = sub.add_parser("reference", parents=[common],
                         help="chance scores of the random predictor on the dataset")
    ref.add_argument("--symmetry-type", nargs="+", choices=SYMMETRY_TYPES,
                     default=list(SYMMETRY_TYPES))
    ref.add_argument("--coverage", type=float, nargs="+", default=[1.0],
                     help="fraction(s) of objects that receive a prediction")
    ref.add_argument("--n-planes", type=int, nargs="+", default=[],
                     help="also score plane_sym with this many random planes per object")
    ref.add_argument("--out", default=None, help="optional CSV path")

    cmp_ = sub.add_parser("compare", parents=[common],
                          help="compare evaluate.py outputs against the random baseline")
    cmp_.add_argument("--symmetry-type", required=True, choices=SYMMETRY_TYPES)
    cmp_.add_argument("--summary-csv", nargs="+", default=None,
                      help="eval_*_summary.csv or compare_results *_comparison.csv files")
    cmp_.add_argument("--eval-results", nargs="+", default=None,
                      help="eval_*_results.json files (exact per-object coverage)")
    cmp_.add_argument("--coverage-mode", choices=COVERAGE_MODES, default="all",
                      help="all: random predictor answers every object (default); "
                           "matched: same coverage as the method")
    cmp_.add_argument("--out", default="RandomBaseline/outputs/random_baseline_comparison.csv")

    tex = sub.add_parser("latex", parents=[common],
                         help="LaTeX tables of the random baseline (axial and planar)")
    tex.add_argument("--symmetry-type", nargs="+", choices=SYMMETRY_TYPES,
                     default=list(SYMMETRY_TYPES))
    tex.add_argument("--out-dir", default="docs/resultados/tablas_azar")

    cap4 = sub.add_parser("cap4", parents=[common],
                          help="one .tex with the random and the fixed-axis baselines")
    cap4.add_argument("--out", default=str(CAP4_OUTPUT))

    args = p.parse_args()
    if args.command == "compare" and not (args.summary_csv or args.eval_results):
        p.error("compare needs --summary-csv and/or --eval-results")
    return args


def main() -> None:
    """ CLI entrypoint. """
    args = parse_args()
    if args.command == "reference":
        run_reference(args)
    elif args.command == "latex":
        run_latex(args)
    elif args.command == "cap4":
        run_cap4(args)
    else:
        run_compare(args)


if __name__ == "__main__":
    main()
