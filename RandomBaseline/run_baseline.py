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
#   latex:     LaTeX tables of the random baseline on the full dataset, one
#              for axial and one for planar symmetry.
#
#   python RandomBaseline/run_baseline.py reference --objects-root ../data/objects
#   python RandomBaseline/run_baseline.py latex --objects-root ../data/objects
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
from significance import COMPARED_METRICS, compare_metric  # noqa: E402
from simulation import (  # noqa: E402
    ANGULAR_THRESHOLDS, AUC_ANGULAR_MAX, GroundTruth,
    load_ground_truth, simulate_random_baseline,
)
from evaluate import compute_summary  # noqa: E402

SYMMETRY_TYPES = ("axis_sym", "plane_sym")
REFERENCE_METRICS = ["angular_error_mean", "angular_error_median", "angular_error_std",
                     "auc_angular", *[f"precision_{t}deg" for t in ANGULAR_THRESHOLDS],
                     "f1_ref", "f1_ref_hungarian"]
COVERAGE_MODES = ("all", "matched")
COMPARE_FIELDS = ["experiment", "method", "n_views", "n_total", "n_objects", "coverage",
                  "random_coverage",
                  "metric", "method_value", "random_mean", "random_std", "random_ci_low",
                  "random_ci_high", "z", "p_empirical", "p_normal", "better_than_random"]


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


def compare_rows(gt: GroundTruth, base: dict, method_metrics: dict[str, float | None],
                 random_metrics: dict[str, np.ndarray]) -> list[dict]:
    """ Build one output row per metric present for both the method and the random baseline.

    Args:
        * gt: ground truth (only used for its symmetry type)
        * base: experiment/method/n_views/coverage columns shared by every row
        * method_metrics: the method's summary values
        * random_metrics: simulated random-baseline values per metric

    Returns:
        * list[dict]: rows with COMPARE_FIELDS columns

    """
    rows = []
    for metric, higher_is_better in COMPARED_METRICS.items():
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
    print("\n* closed form assumes ONE GT element per object; the simulation uses the real "
          "GT planes (best match), so for plane_sym the simulated column is the valid one.")
    if args.out:
        _write_csv(rows, Path(args.out), list(rows[0].keys()))


def compare_summary_csv(gt: GroundTruth, csv_path: Path, args: argparse.Namespace) -> list[dict]:
    """ Compare every row of an evaluate.py summary CSV or compare_results comparison CSV.

    With --coverage-mode all the random predictor answers all n_total objects.
    With matched, only n_total/n_objects are known per row, so which objects
    the random predictor answers is redrawn at random in every repetition.

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
        if _to_float(r.get("n_planes_predicted_mean")) is not None:
            print(f"  [skip] {csv_path.name} nv={r['n_views']}: multi-plane row "
                  "(recall/precision over plane sets), no single-plane random baseline")
            continue
        n_total, n_objects = int(r["n_total"]), int(r["n_objects"])
        if n_total != gt.n_objects:
            raise ValueError(f"{csv_path.name}: n_total={n_total} but {gt.n_objects} GT labels; "
                             "use --eval-results to get the exact object list")
        n_random = n_total if args.coverage_mode == "all" else n_objects
        if n_random not in cache:
            cache[n_random] = simulate_random_baseline(gt, args.repetitions, args.seed,
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
        method_metrics = {m: _to_float(r.get(m)) for m in COMPARED_METRICS}
        rows.extend(compare_rows(gt, base, method_metrics, cache[n_random]))
    return rows


def compare_eval_results(objects_root: Path, symmetry_type: str, json_path: Path,
                         args: argparse.Namespace) -> list[dict]:
    """ Compare an eval_*_results.json against the random baseline.

    With --coverage-mode matched the random predictor answers exactly the
    objects the method answered; with all, every object.

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
        if "n_planes_predicted_mean" in s:
            print(f"  [skip] {json_path.name} nv={nv}: multi-plane results")
            continue
        per_object = [(objects[oid] or {}).get(nv, {}) for oid in object_ids]
        has_prediction = np.array([m.get("status") == "ok" for m in per_object])
        random_mask = has_prediction if args.coverage_mode == "matched" else None
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
    """ Write the random-baseline LaTeX tables (azar_axis.tex, azar_plane.tex).

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
    else:
        run_compare(args)


if __name__ == "__main__":
    main()
