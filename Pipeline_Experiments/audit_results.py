# audit_results.py
#
# Terminal audit for a Pipeline_Experiments run: replays the SAME config a
# run_batch.py invocation used (source experiment_ids x applicable variants)
# and reports what actually landed on disk for each combination -- counts,
# not metrics -- so silent per-object failures (e.g. "need >=2 valid views"
# swallowed by process_object's `except Exception: continue`) show up as
# n_pred < n_molmo instead of hiding inside a file you'd have to open by hand.
#
# Usage:
#   python Pipeline_Experiments/audit_results.py --config Pipeline_Experiments/configs/full_sweep.yaml
#   python Pipeline_Experiments/audit_results.py --config ... --only-problems
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from pipeline_common.naming import exp_filename  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "Mapping"))
from evaluate import experiment_suffix  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RunConfig, VariantConfig, build_output_experiment_id, load_config  # noqa: E402
from triangulation_variants import get_variant  # noqa: E402

MOLMO_JSON = "molmo_multiview.json"
PREDICTED_FILE = "predicted_symmetry.json"


def _count_molmo(all_objects: list[Path], source_id: str, sizes: list[int], lightings: list[str]) -> int:
    """ Count objects with a molmo_multiview_<source_id>.json in any configured (size, lighting). """
    input_file = exp_filename(MOLMO_JSON, source_id)
    n = 0
    for obj_dir in all_objects:
        if any((obj_dir / str(size) / lighting / input_file).exists()
               for size in sizes for lighting in lightings):
            n += 1
    return n


def _count_pred(all_objects: list[Path], output_id: str) -> int:
    """ Count objects with a predicted_symmetry_<output_id>.json directly under the object dir. """
    output_file = exp_filename(PREDICTED_FILE, output_id)
    return sum(1 for obj_dir in all_objects if (obj_dir / output_file).exists())


def _eval_coverage(
    renders_root: Path, symmetry_type: str, output_id: str, method: str,
    sizes: list[int], lightings: list[str],
) -> tuple[bool, int, int]:
    """ Read eval_..._results.json (if present) and count "ok" vs total (object, n_views) entries.

    Args:
        * renders_root: root folder of renders
        * symmetry_type: "axis_sym" or "plane_sym"
        * output_id: the --experiment-id evaluate.py was run with
        * method: "triangulation" or "triangulation_multiplane"
        * sizes: sizes evaluate.py was run with
        * lightings: lightings evaluate.py was run with

    Returns:
        * tuple[bool, int, int]: (file_exists, n_ok_entries, n_total_entries)

    """
    suffix = experiment_suffix(sizes, lightings)
    path = renders_root / symmetry_type / f"eval_{suffix}_{output_id}_{method}_results.json"
    if not path.exists():
        return False, 0, 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return True, 0, 0

    objects = data.get("objects", {})
    n_ok = n_total = 0
    for obj_result in objects.values():
        if obj_result is None:
            n_total += 1
            continue
        for m in obj_result.values():
            n_total += 1
            if isinstance(m, dict) and m.get("status") == "ok":
                n_ok += 1
    return True, n_ok, n_total


def audit(config: RunConfig) -> list[dict]:
    """ Build one audit row per (symmetry_type, source_experiment_id, variant).

    Args:
        * config: the same RunConfig a run_batch.py invocation would load

    Returns:
        * list[dict]: rows with n_molmo/n_pred/eval coverage and a computed
          "problem" flag

    """
    rows = []
    for symmetry_type in config.symmetry_types:
        symmetry_dir = config.renders_root / symmetry_type
        if not symmetry_dir.exists():
            continue
        all_objects = sorted(d for d in symmetry_dir.iterdir() if d.is_dir())
        if config.max_objects:
            all_objects = all_objects[:config.max_objects]
        n_dataset = len(all_objects)

        source_ids = config.resolved_experiment_ids(symmetry_type)
        variants = [vc for vc in config.variants if symmetry_type in get_variant(vc.variant).SYMMETRY_TYPES]

        for source_id in sorted(source_ids):
            n_molmo = _count_molmo(all_objects, source_id, config.sizes, config.lightings)

            for variant_config in variants:
                output_id = build_output_experiment_id(source_id, variant_config)
                method = ("triangulation_multiplane"
                          if symmetry_type == "plane_sym" and variant_config.max_planes > 1
                          else "triangulation")
                n_pred = _count_pred(all_objects, output_id)
                eval_exists, n_ok, n_eval_total = _eval_coverage(
                    config.renders_root, symmetry_type, output_id, method, config.sizes, config.lightings,
                )

                problems = []
                if n_molmo == 0:
                    problems.append("NO MOLMO POINTS")
                elif n_pred < n_molmo:
                    problems.append(f"pred missing {n_molmo - n_pred}/{n_molmo}")
                if config.run_evaluate and n_pred > 0 and not eval_exists:
                    problems.append("evaluate never ran")
                elif eval_exists and n_ok < n_eval_total:
                    problems.append(f"eval status!=ok {n_eval_total - n_ok}/{n_eval_total}")

                rows.append({
                    "symmetry_type": symmetry_type, "source": source_id, "variant": variant_config.variant,
                    "output_id": output_id, "n_dataset": n_dataset, "n_molmo": n_molmo, "n_pred": n_pred,
                    "eval_exists": eval_exists, "n_ok": n_ok, "n_eval_total": n_eval_total,
                    "problems": "; ".join(problems) if problems else "-",
                })
    return rows


def print_table(rows: list[dict], only_problems: bool) -> None:
    """ Print the audit table, grouped by symmetry_type, plus a problem-count summary. """
    header = (f"{'sym':10s} {'source':32s} {'variant':6s} {'n_molmo':>7s} {'n_pred':>7s} "
              f"{'eval':>5s} {'ok/total':>10s}  problems")
    shown = [r for r in rows if not only_problems or r["problems"] != "-"]

    print(header)
    print("-" * len(header))
    for r in shown:
        eval_col = "-" if not r["eval_exists"] else f"{r['n_ok']}/{r['n_eval_total']}" if r["n_eval_total"] else "y"
        print(f"{r['symmetry_type']:10s} {r['source']:32s} {r['variant']:6s} "
              f"{r['n_molmo']:7d} {r['n_pred']:7d} "
              f"{'y' if r['eval_exists'] else 'n':>5s} {eval_col:>10s}  {r['problems']}")

    n_problem_rows = sum(1 for r in rows if r["problems"] != "-")
    print("-" * len(header))
    print(f"{len(rows)} combinations audited, {n_problem_rows} with problems"
          f"{f' ({len(shown)} shown)' if only_problems else ''}")


def main() -> None:
    """ CLI entrypoint. """
    p = argparse.ArgumentParser(
        description="Audit what actually ran for a Pipeline_Experiments config -- counts, not metrics.",
    )
    p.add_argument("--config", required=True, help="Same YAML config run_batch.py was given.")
    p.add_argument("--only-problems", action="store_true", help="Only print rows with something to fix.")
    args = p.parse_args()

    config = load_config(args.config)
    rows = audit(config)
    print_table(rows, args.only_problems)


if __name__ == "__main__":
    main()
