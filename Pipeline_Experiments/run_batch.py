# run_batch.py
#
# Top-level, config-driven orchestrator: reads a YAML config (see
# configs/experiments.example.yaml) and runs every configured triangulation
# ablation / diagnostic over EVERY configured (or auto-discovered)
# --experiment-id prompt run, instead of hand-writing one shell command per
# prompt the way the Experiments/ sandbox notebooks did. Optionally chains
# Mapping/evaluate.py and Mapping/compare_results_no_mesh.py so a single
# invocation goes from "predictions" to "an updated comparison CSV".
#
# Usage:
#   python Pipeline_Experiments/run_batch.py --config Pipeline_Experiments/configs/experiments.example.yaml
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RunConfig, VariantConfig, build_output_experiment_id, load_config  # noqa: E402
from estimate_symmetry_variants import process_object  # noqa: E402
from triangulation_variants import get_variant  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "diagnostics"))
from diagnose_center_bias import run_diagnostic as run_center_bias  # noqa: E402


def _applicable_variants(config: RunConfig, symmetry_type: str) -> list[VariantConfig]:
    """ Configured variants whose module supports symmetry_type. """
    return [vc for vc in config.variants if symmetry_type in get_variant(vc.variant).SYMMETRY_TYPES]


def run_variants(
    config: RunConfig, shard_id: int = 0, num_shards: int = 1,
) -> list[tuple[str, str, str]]:
    """ Run every configured triangulation variant over every resolved experiment_id.

    Pure CPU work (numpy + gpytoolbox for expF) -- no GPU involved, so
    "sharding" here means splitting the OBJECT list across parallel OS
    processes on this machine's CPU cores, not across GPUs. Each shard writes
    disjoint predicted_symmetry_<ID>.json files (one per object), so running
    several shards concurrently is safe; only run evaluate/compare (which
    each need EVERY object's output to exist) after all shards finish, e.g.
    with --skip-evaluate --skip-compare on every sharded invocation and a
    final unsharded --skip-variants pass.

    Args:
        * config: the loaded run configuration
        * shard_id: this process's shard index, in [0, num_shards)
        * num_shards: total number of parallel shards (1 = no sharding)

    Returns:
        * list[tuple[str, str, str]]: (symmetry_type, output_experiment_id,
          method) for every combination actually processed -- fed to
          run_evaluate/run_compare below (identical across shards, since
          every shard sees the same experiment_ids x variants, just a
          disjoint slice of objects)

    """
    generated: list[tuple[str, str, str]] = []
    mesh_ctx_cache: dict = {}

    for symmetry_type in config.symmetry_types:
        source_ids = config.resolved_experiment_ids(symmetry_type)
        if not source_ids:
            print(f"[warn] No experiment_ids resolved for {symmetry_type} -- skipping.")
            continue

        variants = _applicable_variants(config, symmetry_type)
        if not variants:
            print(f"[warn] No configured variant applies to {symmetry_type} -- skipping.")
            continue

        symmetry_dir = config.renders_root / symmetry_type
        all_objects = sorted(d for d in symmetry_dir.iterdir() if d.is_dir())
        if config.max_objects:
            all_objects = all_objects[:config.max_objects]
        all_objects = all_objects[shard_id::num_shards]

        print(f"\n[{symmetry_type}] shard {shard_id}/{num_shards}: {len(source_ids)} source "
              f"experiment_id(s) x {len(variants)} variant(s) x {len(all_objects)} objects")

        for source_experiment_id in source_ids:
            for variant_config in variants:
                output_id = build_output_experiment_id(source_experiment_id, variant_config)
                for obj_dir in tqdm(all_objects, unit="obj", dynamic_ncols=True,
                                    desc=f"{source_experiment_id}->{variant_config.variant}"):
                    process_object(
                        object_dir=obj_dir, symmetry_type=symmetry_type, sizes=config.sizes,
                        lightings=config.lightings, source_experiment_id=source_experiment_id,
                        variant_config=variant_config, objects_root=config.objects_root,
                        overwrite=config.overwrite, mesh_ctx_cache=mesh_ctx_cache,
                    )
                method = ("triangulation_multiplane"
                          if symmetry_type == "plane_sym" and variant_config.max_planes > 1
                          else "triangulation")
                generated.append((symmetry_type, output_id, method))

    return generated


def run_diagnostics(config: RunConfig) -> None:
    """ Run every configured diagnostic over every resolved experiment_id.

    Args:
        * config: the loaded run configuration

    """
    if not config.diagnostics:
        return

    for symmetry_type in config.symmetry_types:
        source_ids = config.resolved_experiment_ids(symmetry_type)
        if not source_ids:
            continue

        if "center_bias" in config.diagnostics:
            all_detail, all_summary = [], []
            for experiment_id in source_ids:
                detail, summary = run_center_bias(
                    config.renders_root, symmetry_type, experiment_id,
                    config.sizes, config.lightings, config.max_objects,
                )
                all_detail += detail
                all_summary += summary
                for row in summary:
                    print(f"  [center_bias][{symmetry_type}][{experiment_id}] n_views={row['n_views']:>4}  "
                          f"pct_noncenital_centered={row['pct_noncenital_centered']:.2%}  -> {row['verdict']}")

            out_dir = config.results_dir / "diagnostics"
            out_dir.mkdir(parents=True, exist_ok=True)
            import csv
            for name, rows in (("center_bias_detail", all_detail), ("center_bias_summary", all_summary)):
                if not rows:
                    continue
                path = out_dir / f"{symmetry_type}_{name}.csv"
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"  Wrote {len(rows)} rows -> {path}")


def run_evaluate(config: RunConfig, generated: list[tuple[str, str, str]]) -> None:
    """ Chain Mapping/evaluate.py for every newly-generated (symmetry_type, experiment_id, method).

    Args:
        * config: the loaded run configuration
        * generated: output of run_variants()

    """
    if not config.run_evaluate or not generated:
        return

    evaluate_py = REPO_ROOT / "Mapping" / "evaluate.py"
    for symmetry_type, experiment_id, method in generated:
        cmd = [
            sys.executable, str(evaluate_py),
            "--renders-root", str(config.renders_root), "--objects-root", str(config.objects_root),
            "--symmetry-type", symmetry_type,
            "--sizes", *[str(s) for s in config.sizes], "--lightings", *config.lightings,
            "--experiment-id", experiment_id, "--method", method,
        ]
        if config.with_reference_metrics:
            cmd.append("--with-reference-metrics")
        cmd += config.extra_evaluate_args
        print("  $", " ".join(cmd))
        subprocess.run(cmd, check=True)


def run_compare(config: RunConfig) -> None:
    """ Chain Mapping/compare_results_no_mesh.py once per configured symmetry_type. """
    if not config.run_compare:
        return

    compare_py = REPO_ROOT / "Mapping" / "compare_results_no_mesh.py"
    for symmetry_type in config.symmetry_types:
        cmd = [
            sys.executable, str(compare_py),
            "--renders-root", str(config.renders_root), "--symmetry-type", symmetry_type,
            "--sizes", *[str(s) for s in config.sizes], "--lightings", *config.lightings,
            "--csv-dir", str(config.results_dir), "--no-plots",
        ]
        print("  $", " ".join(cmd))
        subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    """ Parse CLI arguments. """
    p = argparse.ArgumentParser(
        description="Config-driven batch runner for Pipeline_Experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True, help="Path to a YAML config file.")
    p.add_argument("--skip-variants", action="store_true")
    p.add_argument("--skip-diagnostics", action="store_true")
    p.add_argument("--skip-evaluate", action="store_true", help="Overrides evaluation.run_evaluate.")
    p.add_argument("--skip-compare", action="store_true", help="Overrides evaluation.run_compare.")
    p.add_argument("--shard-id", type=int, default=0,
                   help="This process's shard index (CPU parallelism across the object list -- "
                        "no GPU is used anywhere in this module). Launch --num-shards processes "
                        "with --shard-id 0..num_shards-1, each with --skip-evaluate --skip-compare "
                        "--skip-diagnostics, then one final unsharded run with --skip-variants to "
                        "run diagnostics/evaluate/compare once over the complete object set.")
    p.add_argument("--num-shards", type=int, default=1)
    return p.parse_args()


def main() -> None:
    """ CLI entrypoint. """
    args = parse_args()
    config = load_config(args.config)

    if args.skip_evaluate:
        config.run_evaluate = False
    if args.skip_compare:
        config.run_compare = False

    if args.num_shards > 1:
        # run_evaluate/run_compare already reflect --skip-evaluate/--skip-compare
        # (mutated above); diagnostics has no such config-level flag, so check
        # --skip-diagnostics directly.
        needs_full_set = (
            config.run_evaluate or config.run_compare
            or (config.diagnostics and not args.skip_diagnostics)
        )
        if needs_full_set:
            print("[error] --num-shards > 1 together with diagnostics/evaluate/compare would "
                  "read/score a partial object set (and race across shards writing the same "
                  "output files). Pass --skip-diagnostics --skip-evaluate --skip-compare on every "
                  "sharded invocation, then run once more with --skip-variants (no sharding needed "
                  "there) to run diagnostics/evaluate/compare over the complete object set.")
            sys.exit(1)

    generated: list[tuple[str, str, str]] = []
    if not args.skip_variants:
        generated = run_variants(config, shard_id=args.shard_id, num_shards=args.num_shards)
    if not args.skip_diagnostics:
        run_diagnostics(config)

    run_evaluate(config, generated)
    run_compare(config)

    print("\nDone.")
    if generated:
        print(f"Generated {len(generated)} (symmetry_type, experiment_id, method) combinations:")
        for row in generated:
            print(f"  {row}")


if __name__ == "__main__":
    main()
