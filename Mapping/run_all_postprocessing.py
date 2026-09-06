#!/usr/bin/env python3
"""Sweep the post-Molmo pipeline (map_to_3d -> estimate_symmetry -> evaluate)
over every Molmo experiment already found on disk, for both symmetry types —
plus, for each of those, the full clustering (C2 greedy, C3 hdbscan ms=2/3/5)
and patch-size backprojection (p3, p5) variant sweep.

Meant to be left running unattended (tmux) after all Molmo inference runs
have finished. It auto-discovers which experiment ids actually have
molmo_multiview_<EXP_ID>.json files under --renders-root (instead of a
hardcoded list), so it stays correct regardless of exactly which prompts /
Flow B / Flow C runs were executed. Each experiment's files are isolated by
--experiment-id (see MolmoPointing/Experiments.md "File isolation guarantee"),
so nothing here can overwrite another experiment's output.

Per base experiment this produces 8 variants total: the baseline (C1, patch 1),
C2 (greedy), C3 (hdbscan min_samples 2/3/5), and patch-size 3/5 — each
evaluated with all 4 fitting methods. This is deliberately more than
POST_MOLMO_PIPELINE_COMMANDS.md section 6 recommends (there, the sweep is run
only on top of one already-picked winning prompt) — here it runs for every
discovered experiment, trading extra compute for a fuller comparison table.
Pass --no-clustering-patch-sweep to fall back to baseline-only, matching that
narrower recommendation.

Resumable by construction: map_to_3d.py / estimate_symmetry.py already skip
per-object files that exist unless --overwrite is passed, and this script
additionally skips an entire experiment/variant's stages once all 4 evaluate
summary CSVs exist for it. Re-running this script after an interruption
(Ctrl-C, crash, tmux killed) just fast-forwards through finished work.

Usage (from repo root, inside tmux):
    python Mapping/run_all_postprocessing.py

See --help for flags (--overwrite, --export-examples, --only, --dry-run,
--no-clustering-patch-sweep, ...).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime, date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
METHODS = ["svd", "ransac_svd", "svd_sde", "ransac_svd_sde"]
PATCH_SIZES = [3, 5]
HDBSCAN_MIN_SAMPLES = [2, 3, 5]

# --experiment-id -> --point-mode, per MolmoPointing/Experiments.md's
# registered-prompt table. Flow B reuses its base prompt's mode (looked up
# by stripping "_flowB"); Flow C always uses "all" regardless of base prompt.
POINT_MODE = {
    "axis_v00": "independent", "axis_v01": "midpoint",   "axis_v02": "midpoint",
    "axis_v03": "midpoint",    "axis_v04": "independent", "axis_v05": "independent",
    "axis_v00_1": "independent", "axis_v01_1": "midpoint",   "axis_v02_1": "independent",
    "axis_v03_1": "midpoint",    "axis_v04_1": "independent", "axis_v05_1": "independent",
    "plane_v00": "independent", "plane_v01": "midpoint",   "plane_v02": "independent",
    "plane_v03": "midpoint",    "plane_v04": "independent", "plane_v05": "independent",
    "plane_v00_1": "independent", "plane_v01_1": "midpoint",   "plane_v02_1": "independent",
    "plane_v03_1": "midpoint",    "plane_v04_1": "independent", "plane_v05_1": "independent",
    # Round 4/5 (see MolmoPointing/prompts_registry.py DESCRIPTIONS)
    "axis_v08": "independent", "plane_v08": "independent",
    "axis_lit2_grid": "independent", "axis_lit3_cot": "independent",
}

# Suffixes that mark a downstream sweep artifact (clustering / patch-size),
# never a raw Molmo experiment. Section 6 of POST_MOLMO_PIPELINE_COMMANDS.md
# handles these separately, on top of a single already-picked winner — skip here.
_SWEEP_SUFFIX_RE = re.compile(r"(_cluster|_hdbscan_ms\d+|_p[0-9]+)$")


def point_mode_for(exp_id: str) -> str | None:
    if exp_id.endswith("_flowC"):
        return "all"
    if exp_id.endswith("_flowB"):
        base = exp_id[: -len("_flowB")]
        return POINT_MODE.get(base)
    return POINT_MODE.get(exp_id)


def discover_experiments(renders_root: Path, sym: str, sizes: list[int], lightings: list[str]) -> list[str]:
    found: set[str] = set()
    sym_dir = renders_root / sym
    for size in sizes:
        for lighting in lightings:
            for f in sym_dir.glob(f"*/{size}/{lighting}/molmo_multiview_*.json"):
                m = re.match(r"molmo_multiview_(.+)\.json$", f.name)
                if m:
                    found.add(m.group(1))
    return sorted(found)


class Logger:
    """Writes a concise, greppable line per stage to --log-file, and a full
    (stdout+stderr) transcript of every subprocess to --full-log-file."""

    def __init__(self, log_path: Path, full_log_path: Path):
        self.log_path = log_path
        self.full_log_path = full_log_path
        for p in (log_path, full_log_path):
            p.parent.mkdir(parents=True, exist_ok=True)

    def _ts(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def line(self, msg: str) -> None:
        text = f"[{self._ts()}] {msg}"
        print(text, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def full(self, msg: str) -> None:
        with open(self.full_log_path, "a", encoding="utf-8") as f:
            f.write(msg if msg.endswith("\n") else msg + "\n")


def run_stage(logger: Logger, exp_id: str, stage: str, cmd: list[str], dry_run: bool) -> bool:
    cmd_str = " ".join(cmd)
    if dry_run:
        logger.line(f"{exp_id:28s} | {stage:22s} | DRY-RUN | {cmd_str}")
        return True

    logger.line(f"{exp_id:28s} | {stage:22s} | START")
    logger.full(f"\n===== {exp_id} | {stage} | {datetime.now()} =====\n$ {cmd_str}\n")
    t0 = time.time()
    proc = subprocess.Popen(
        cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    tail_lines: list[str] = []
    assert proc.stdout is not None
    for raw_line in proc.stdout:
        logger.full(raw_line.rstrip("\n"))
        tail_lines.append(raw_line.rstrip("\n"))
        if len(tail_lines) > 20:
            tail_lines.pop(0)
    proc.wait()
    dt = time.time() - t0

    if proc.returncode == 0:
        logger.line(f"{exp_id:28s} | {stage:22s} | DONE    | {dt:7.1f}s")
        return True

    logger.line(f"{exp_id:28s} | {stage:22s} | FAIL    | {dt:7.1f}s | exit={proc.returncode}")
    for tl in tail_lines:
        logger.line(f"{'':28s} |   stderr> {tl}")
    return False


def experiment_already_done(renders_root: Path, sym: str, exp_id: str) -> bool:
    sym_dir = renders_root / sym
    return all(
        (sym_dir / f"eval_s224_flat_{exp_id}_{method}_summary.csv").exists()
        for method in METHODS
    )


def clean_corrupt_json_files(logger: Logger, renders_root: Path, sym: str, exp_id: str) -> int:
    """Per-object files (molmo_multiview_<EXP>.json, mapped_points_3d_<EXP>.json,
    predicted_symmetry_<EXP>.json) only get skip-if-exists treatment downstream —
    a 0-byte/truncated file (left by an earlier interrupted run, OR written moments
    ago by map_to_3d.py itself in this very run) is silently treated as "already
    done" and crashes the reading stage with a JSONDecodeError, which aborts the
    entire experiment (not just the one bad object) unless purged first. Call this
    BOTH before map_to_3d (cleans stale corruption from a prior run) AND again right
    after map_to_3d succeeds, before estimate_symmetry reads its output (catches
    corruption map_to_3d just introduced in this run)."""
    import json as _json

    sym_dir = renders_root / sym
    removed: list[tuple[Path, str]] = []
    for f in sym_dir.glob(f"**/*_{exp_id}.json"):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                _json.load(fh)
        except Exception as e:
            try:
                f.unlink()
                removed.append((f, str(e)))
            except OSError:
                pass
    if removed:
        logger.line(f"{exp_id:28s} | {'corrupt-scan':22s} | CLEAN   | removed {len(removed)} unreadable json file(s)")
        for f, e in removed:
            logger.line(f"{'':28s} |   removed> {f} ({e})")
    return len(removed)


def compare_one(logger: Logger, args, sym: str, variant_id: str) -> None:
    renders = Path(args.renders_root)
    results = Path(args.results_root)
    run_stage(
        logger, variant_id, "compare_results",
        [
            sys.executable, "Mapping/compare_results.py",
            "--renders-root", str(renders), "--symmetry-type", sym,
            "--sizes", *map(str, args.sizes), "--lightings", *args.lightings,
            "--total-objects", str(args.total_objects),
            "--experiment-id", variant_id,
            "--save-dir", str(results / sym / "per_experiment" / variant_id / "plots"),
            "--csv-dir", str(results / sym / "per_experiment" / variant_id),
        ],
        args.dry_run,
    )


def evaluate_all_methods(logger: Logger, args, sym: str, variant_id: str) -> None:
    renders = Path(args.renders_root)
    objects = Path(args.objects_root)
    common = [
        "--renders-root", str(renders), "--objects-root", str(objects),
        "--symmetry-type", sym, "--sizes", *map(str, args.sizes),
        "--lightings", *args.lightings, "--experiment-id", variant_id,
    ]
    for method in METHODS:
        run_stage(
            logger, variant_id, f"evaluate[{method}]",
            [sys.executable, "Mapping/evaluate.py", *common, "--method", method],
            args.dry_run,
        )


def run_core_stages(logger: Logger, args, sym: str, exp_id: str, mode: str) -> bool:
    """map_to_3d -> estimate_symmetry -> evaluate x4 for the C1/patch-1 baseline.
    Returns True if usable results exist for exp_id afterward (fresh or already done)."""
    renders = Path(args.renders_root)
    objects = Path(args.objects_root)
    common = [
        "--renders-root", str(renders), "--objects-root", str(objects),
        "--symmetry-type", sym, "--sizes", *map(str, args.sizes),
        "--lightings", *args.lightings, "--experiment-id", exp_id,
    ]

    if experiment_already_done(renders, sym, exp_id) and not args.overwrite:
        logger.line(f"{exp_id:28s} | {'core stages':22s} | SKIP    | all 4 eval summaries already exist")
        return True

    if not args.dry_run:
        clean_corrupt_json_files(logger, renders, sym, exp_id)

    ok = run_stage(
        logger, exp_id, "map_to_3d",
        [sys.executable, "Mapping/map_to_3d.py", *common, "--overwrite", "--yes"]
        if args.overwrite else
        [sys.executable, "Mapping/map_to_3d.py", *common, "--yes"],
        args.dry_run,
    )
    if ok:
        if not args.dry_run:
            clean_corrupt_json_files(logger, renders, sym, exp_id)
        ok = run_stage(
            logger, exp_id, "estimate_symmetry",
            [sys.executable, "Mapping/estimate_symmetry.py", *common, "--point-mode", mode]
            + (["--overwrite"] if args.overwrite else []),
            args.dry_run,
        )
    if not ok:
        logger.line(f"{exp_id:28s} | {'core stages':22s} | ABORTED | skipping evaluate/compare for this experiment")
        return False

    evaluate_all_methods(logger, args, sym, exp_id)
    return True


def run_clustering_sweep(logger: Logger, args, sym: str, exp_id: str, mode: str) -> None:
    """C2 (greedy) + C3 (hdbscan ms=2/3/5) — reuses the baseline's existing
    mapped_points_3d_<exp_id>.json (patch-size 1), no new map_to_3d call."""
    renders = Path(args.renders_root)
    objects = Path(args.objects_root)
    common = [
        "--renders-root", str(renders), "--objects-root", str(objects),
        "--symmetry-type", sym, "--sizes", *map(str, args.sizes),
        "--lightings", *args.lightings, "--experiment-id", exp_id,
        "--point-mode", mode,
    ]

    cluster_variants = [(f"{exp_id}_cluster", ["--clustering-method", "greedy"])]
    for ms in HDBSCAN_MIN_SAMPLES:
        cluster_variants.append(
            (f"{exp_id}_hdbscan_ms{ms}", ["--clustering-method", "hdbscan", "--hdbscan-min-samples", str(ms)])
        )

    for variant_id, cluster_args in cluster_variants:
        if experiment_already_done(renders, sym, variant_id) and not args.overwrite:
            logger.line(f"{variant_id:28s} | {'estimate_symmetry':22s} | SKIP    | all 4 eval summaries already exist")
        else:
            ok = run_stage(
                logger, variant_id, "estimate_symmetry",
                [sys.executable, "Mapping/estimate_symmetry.py", *common, *cluster_args, "--overwrite"],
                args.dry_run,
            )
            if not ok:
                logger.line(f"{variant_id:28s} | {'clustering sweep':22s} | ABORTED | skipping evaluate/compare for this variant")
                continue
            evaluate_all_methods(logger, args, sym, variant_id)
        if not args.no_per_experiment_compare:
            compare_one(logger, args, sym, variant_id)


def run_patch_sweep(logger: Logger, args, sym: str, exp_id: str, mode: str) -> None:
    """Backprojection patch-size 3 and 5 — new map_to_3d call per patch size."""
    renders = Path(args.renders_root)
    objects = Path(args.objects_root)

    for patch in PATCH_SIZES:
        variant_id = f"{exp_id}_p{patch}"
        if experiment_already_done(renders, sym, variant_id) and not args.overwrite:
            logger.line(f"{variant_id:28s} | {'patch sweep':22s} | SKIP    | all 4 eval summaries already exist")
            if not args.no_per_experiment_compare:
                compare_one(logger, args, sym, variant_id)
            continue

        common_base = [
            "--renders-root", str(renders), "--objects-root", str(objects),
            "--symmetry-type", sym, "--sizes", *map(str, args.sizes),
            "--lightings", *args.lightings,
        ]
        ok = run_stage(
            logger, variant_id, "map_to_3d",
            [sys.executable, "Mapping/map_to_3d.py", *common_base,
             "--experiment-id", exp_id, "--patch-size", str(patch), "--overwrite", "--yes"],
            args.dry_run,
        )
        if ok:
            if not args.dry_run:
                clean_corrupt_json_files(logger, renders, sym, variant_id)
            ok = run_stage(
                logger, variant_id, "estimate_symmetry",
                [sys.executable, "Mapping/estimate_symmetry.py", *common_base,
                 "--experiment-id", variant_id, "--point-mode", mode, "--overwrite"],
                args.dry_run,
            )
        if not ok:
            logger.line(f"{variant_id:28s} | {'patch sweep':22s} | ABORTED | skipping evaluate/compare for this variant")
            continue
        evaluate_all_methods(logger, args, sym, variant_id)
        if not args.no_per_experiment_compare:
            compare_one(logger, args, sym, variant_id)


def process_experiment(
    logger: Logger, args, sym: str, exp_id: str, mode: str,
) -> None:
    results = Path(args.results_root)

    core_ok = run_core_stages(logger, args, sym, exp_id, mode)
    if not core_ok:
        return

    if not args.no_clustering_patch_sweep:
        run_clustering_sweep(logger, args, sym, exp_id, mode)
        run_patch_sweep(logger, args, sym, exp_id, mode)

    if not args.no_per_experiment_compare:
        compare_one(logger, args, sym, exp_id)

    if args.export_examples:
        renders = Path(args.renders_root)
        objects = Path(args.objects_root)
        run_stage(
            logger, exp_id, "export_examples",
            [
                sys.executable, "Mapping/export_viz_samples.py",
                "--renders-root", str(renders), "--objects-root", str(objects),
                "--symmetry-type", sym, "--sizes", *map(str, args.sizes),
                "--lightings", *args.lightings, "--experiment-id", exp_id,
                "--method", "svd", "--n-views", "14", "--n-samples", "10",
                "--results-dir", str(results),
            ],
            args.dry_run,
        )


def final_consolidation(logger: Logger, args) -> None:
    renders = Path(args.renders_root)
    results = Path(args.results_root)
    for sym in ["axis_sym", "plane_sym"]:
        run_stage(
            logger, sym, "consolidated_compare",
            [
                sys.executable, "Mapping/compare_results.py",
                "--renders-root", str(renders), "--symmetry-type", sym,
                "--sizes", *map(str, args.sizes), "--lightings", *args.lightings,
                "--total-objects", str(args.total_objects),
                "--save-dir", str(results / sym / "plots"),
                "--csv-dir", str(results),
            ],
            args.dry_run,
        )

    if args.dry_run:
        return

    try:
        import pandas as pd
    except ImportError:
        logger.line(f"{'merge':28s} | {'consolidated_compare':22s} | SKIP    | pandas not available")
        return

    today_dir = results / f"experiments_{date.today().strftime('%d_%m_%Y')}"
    frames = []
    for sym in ["axis_sym", "plane_sym"]:
        p = today_dir / f"{sym}_comparison.csv"
        if p.exists():
            df = pd.read_csv(p)
            df.insert(0, "symmetry_type", sym)
            frames.append(df)
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        out_path = today_dir / "all_experiments_comparison.csv"
        combined.to_csv(out_path, index=False)
        logger.line(f"{'merge':28s} | {'consolidated_compare':22s} | DONE    | wrote {out_path} ({len(combined)} rows)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run map_to_3d -> estimate_symmetry -> evaluate -> compare for every "
                    "Molmo experiment found on disk, both symmetry types, unattended.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--renders-root", default="../data/renders")
    p.add_argument("--objects-root", default="../data/objects")
    p.add_argument("--results-root", default="../results")
    p.add_argument("--sizes", type=int, nargs="+", default=[224])
    p.add_argument("--lightings", type=str, nargs="+", default=["flat"])
    p.add_argument("--total-objects", type=int, default=850)
    p.add_argument("--overwrite", action="store_true",
                   help="Force re-running map_to_3d/estimate_symmetry even if outputs exist.")
    p.add_argument("--no-clustering-patch-sweep", action="store_true",
                   help="Skip the clustering (C2/C3) and patch-size (p3/p5) variant sweep — "
                        "baseline (C1, patch-1) only per experiment.")
    p.add_argument("--no-per-experiment-compare", action="store_true",
                   help="Skip the per-experiment compare_results.py call (section 2), "
                        "including per-variant compare when the sweep is enabled.")
    p.add_argument("--export-examples", action="store_true",
                   help="Also run export_viz_samples.py per experiment (section 2b, slower, writes images).")
    p.add_argument("--only", nargs="+", default=None, metavar="EXP_ID",
                   help="Restrict to these experiment ids only (for testing).")
    p.add_argument("--exclude", nargs="+", default=[], metavar="EXP_ID")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would run without executing anything.")
    p.add_argument("--log-file", default=str(REPO_ROOT / "log_postprocesos.txt"))
    p.add_argument("--full-log-file", default=str(REPO_ROOT / "log_postprocesos_full.txt"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = Logger(Path(args.log_file), Path(args.full_log_file))
    renders_root = Path(args.renders_root)

    logger.line("=" * 100)
    logger.line(f"run_all_postprocessing START | renders-root={renders_root} | overwrite={args.overwrite} | dry_run={args.dry_run}")

    plan: list[tuple[str, str, str]] = []  # (sym, exp_id, mode)
    for sym in ["axis_sym", "plane_sym"]:
        exp_ids = discover_experiments(renders_root, sym, args.sizes, args.lightings)
        for exp_id in exp_ids:
            if args.only and exp_id not in args.only:
                continue
            if exp_id in args.exclude:
                continue
            if _SWEEP_SUFFIX_RE.search(exp_id):
                logger.line(f"{exp_id:28s} | {'discovery':22s} | SKIP    | clustering/patch-size sweep artifact, handle via section 6 manually")
                continue
            mode = point_mode_for(exp_id)
            if mode is None:
                logger.line(f"{exp_id:28s} | {'discovery':22s} | WARN    | unknown experiment id, not in POINT_MODE registry and no _flowB/_flowC suffix — add it to the script's registry and re-run with --only {exp_id}")
                continue
            plan.append((sym, exp_id, mode))

    variants_per_exp = 1 if args.no_clustering_patch_sweep else (1 + 1 + len(HDBSCAN_MIN_SAMPLES) + len(PATCH_SIZES))
    logger.line(
        f"Discovered {len(plan)} base experiment(s) to process "
        f"({variants_per_exp} variant(s) each -> ~{len(plan) * variants_per_exp} total rows, "
        f"{len(plan) * variants_per_exp * len(METHODS)} evaluate calls):"
    )
    for sym, exp_id, mode in plan:
        logger.line(f"  - {sym:10s} {exp_id:24s} point-mode={mode}")

    t0 = time.time()
    for i, (sym, exp_id, mode) in enumerate(plan, start=1):
        logger.line(f"----- [{i}/{len(plan)}] {sym} / {exp_id} -----")
        process_experiment(logger, args, sym, exp_id, mode)

    logger.line("Running final consolidated compare_results (section 4) for axis_sym and plane_sym...")
    final_consolidation(logger, args)

    dt = time.time() - t0
    logger.line(f"run_all_postprocessing DONE | {len(plan)} experiment(s) | total wall time {dt/60:.1f} min")
    logger.line("=" * 100)


if __name__ == "__main__":
    main()
