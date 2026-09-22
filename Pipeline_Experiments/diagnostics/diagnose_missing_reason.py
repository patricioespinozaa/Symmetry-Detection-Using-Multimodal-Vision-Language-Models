# diagnose_missing_reason.py
#
# Targeted follow-up to Pipeline_Experiments/audit_results.py: that script
# only counts n_molmo vs n_pred per (source_experiment_id, variant); it can't
# tell whether a missing predicted_symmetry_<ID>.json is expected ("Molmo2
# didn't return >=2/>=4 usable points for this object") or a real bug hiding
# behind estimate_symmetry_variants.py::process_object's swallowed
# exceptions. This re-runs process_object for exactly the objects currently
# missing an output file, with its on_error hook wired up, and classifies
# every failure instead of silently skipping it.
#
# Usage (run against the SAME source_experiment_id/variant an audit_results.py
# row flagged as "pred missing N/850" -- start with the biggest outliers):
#   python Pipeline_Experiments/diagnostics/diagnose_missing_reason.py \
#       --renders-root ../data/renders --symmetry-type axis_sym \
#       --experiment-id axis_v05_1_flowC --variant expA \
#       --out ../results/diagnostics/axis_v05_1_flowC_expA_missing_reasons.csv
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline_common.naming import exp_filename  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import VariantConfig, build_output_experiment_id  # noqa: E402
from estimate_symmetry_variants import MOLMO_JSON, OUTPUT_FILE, process_object  # noqa: E402

# ValueError messages process_object already raises for "not enough data" --
# anything else caught by on_error is a real, uncategorized failure.
_EXPECTED_MESSAGE_PREFIXES = (
    "need >=2 valid views",
    "need >=4 valid views",
    "not enough pair-lines",
    "could not generate any candidate normal",
    "detect_planes returned zero accepted planes",
    "no (size, lighting) config produced",
)


def classify(exc: Exception) -> str:
    """ Label one captured failure as expected (insufficient Molmo2 data) or unexpected.

    Args:
        * exc: the exception process_object's on_error hook received

    Returns:
        * str: "insufficient_data" if it matches a known "not enough points/views"
          message, "UNEXPECTED" otherwise (a candidate real bug worth reading
          the traceback for)

    """
    if isinstance(exc, ValueError) and str(exc).startswith(_EXPECTED_MESSAGE_PREFIXES):
        return "insufficient_data"
    return "UNEXPECTED"


def find_missing_objects(
    renders_root: Path, symmetry_type: str, source_experiment_id: str, output_experiment_id: str,
) -> list[Path]:
    """ List object dirs that have Molmo2 points for the source prompt but no output file yet.

    Args:
        * renders_root: root folder of renders
        * symmetry_type: "axis_sym" or "plane_sym"
        * source_experiment_id: the prompt run whose points feed the variant
        * output_experiment_id: the --experiment-id the variant would write

    Returns:
        * list[Path]: object directories missing predicted_symmetry_<output_experiment_id>.json

    """
    symmetry_dir = Path(renders_root) / symmetry_type
    input_file = exp_filename(MOLMO_JSON, source_experiment_id)
    output_file = exp_filename(OUTPUT_FILE, output_experiment_id)

    missing = []
    for obj_dir in sorted(d for d in symmetry_dir.iterdir() if d.is_dir()):
        has_molmo = any(obj_dir.glob(f"*/*/{input_file}"))
        if has_molmo and not (obj_dir / output_file).exists():
            missing.append(obj_dir)
    return missing


def main() -> None:
    """ CLI entrypoint: diagnose why a (source_experiment_id, variant) has missing predictions. """
    p = argparse.ArgumentParser(
        description="Re-run process_object for objects with a missing predicted_symmetry file, "
                    "surfacing the real exception instead of the silent `continue` it normally does.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--renders-root", required=True)
    p.add_argument("--objects-root", default=None, help="Required only for --variant expF.")
    p.add_argument("--symmetry-type", required=True, choices=["axis_sym", "plane_sym"])
    p.add_argument("--experiment-id", required=True, help="Source prompt run (audit_results.py 'source').")
    p.add_argument("--variant", required=True, choices=["expA", "expB", "expC", "expD", "expE", "expF"])
    p.add_argument("--filter-policy", default="none", choices=["none", "per_point", "any", "both"])
    p.add_argument("--max-planes", type=int, default=1, help="plane_sym only.")
    p.add_argument("--edge-on-thresh", type=float, default=0.5, help="plane_sym only.")
    p.add_argument("--dup-angle-thresh", type=float, default=15.0, help="plane_sym only.")
    p.add_argument("--sde-gate", type=float, default=0.02, help="plane_sym + expF only.")
    p.add_argument("--sizes", type=int, nargs="+", default=[224])
    p.add_argument("--lightings", type=str, nargs="+", default=["flat"])
    p.add_argument("--max-objects", type=int, default=None,
                    help="Cap how many currently-missing objects to re-run (for a quick look).")
    p.add_argument("--out", required=True, help="Per-failure detail CSV.")
    args = p.parse_args()

    variant_config = VariantConfig(
        variant=args.variant, filter_policy=args.filter_policy, max_planes=args.max_planes,
        edge_on_thresh=args.edge_on_thresh, dup_angle_thresh=args.dup_angle_thresh, sde_gate=args.sde_gate,
    )
    output_experiment_id = build_output_experiment_id(args.experiment_id, variant_config)

    renders_root = Path(args.renders_root)
    missing = find_missing_objects(renders_root, args.symmetry_type, args.experiment_id, output_experiment_id)
    if args.max_objects:
        missing = missing[:args.max_objects]
    print(f"{len(missing)} object(s) currently missing predicted_symmetry_{output_experiment_id}.json")
    if not missing:
        return

    rows: list[dict] = []
    still_missing: set[str] = {d.name for d in missing}

    def on_error(object_id: str, n_views_key: str, exc: Exception) -> None:
        rows.append({
            "object_id": object_id, "n_views": n_views_key,
            "category": classify(exc), "exc_type": type(exc).__name__, "message": str(exc),
        })

    objects_root = Path(args.objects_root) if args.objects_root else None
    mesh_ctx_cache: dict = {}
    for obj_dir in tqdm(missing, unit="obj", dynamic_ncols=True,
                        desc=f"{args.symmetry_type}/{output_experiment_id}"):
        process_object(
            obj_dir, args.symmetry_type, args.sizes, args.lightings, args.experiment_id, variant_config,
            objects_root=objects_root, overwrite=False, mesh_ctx_cache=mesh_ctx_cache, on_error=on_error,
        )
        if (obj_dir / exp_filename(OUTPUT_FILE, output_experiment_id)).exists():
            still_missing.discard(obj_dir.name)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {len(rows)} failure rows -> {args.out}")

    counts = Counter(r["category"] for r in rows)
    print(f"\ninsufficient_data (expected): {counts['insufficient_data']}")
    print(f"UNEXPECTED (possible bug):    {counts['UNEXPECTED']}")
    if counts["UNEXPECTED"]:
        unexpected_msgs = Counter(
            f"{r['exc_type']}: {r['message']}" for r in rows if r["category"] == "UNEXPECTED"
        )
        print("\nUNEXPECTED failures, grouped by exact message:")
        for msg, n in unexpected_msgs.most_common(20):
            print(f"  {n:4d}x  {msg}")

    print(f"\n{len(missing) - len(still_missing)}/{len(missing)} previously-missing objects now have a "
          f"prediction (this run also backfills them, it does not just diagnose).")
    print(f"{len(still_missing)} object(s) still produced zero valid n_views_key.")


if __name__ == "__main__":
    main()
