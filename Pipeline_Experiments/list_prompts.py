# list_prompts.py
#
# Terminal audit: shows every molmo_multiview_<ID>.json experiment_id found
# on disk for a symmetry_type, how many objects each one covers, and which
# ones config.discover_experiment_ids would drop as legacy "<X>_nomesh"
# duplicates of a bare "<X>" -- i.e. exactly what `experiment_ids: auto`
# would hand to run_batch.py, before you spend CPU time on it.
#
# Usage:
#   python Pipeline_Experiments/list_prompts.py --renders-root ../data/renders
#   python Pipeline_Experiments/list_prompts.py --renders-root ../data/renders --symmetry-type plane_sym
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import NOMESH_SUFFIX, discover_experiment_ids  # noqa: E402

_MOLMO_JSON_EXP_RE = re.compile(r"^molmo_multiview_(.+)\.json$")


def scan(renders_root: Path, symmetry_type: str) -> dict[str, set[str]]:
    """ Map every raw experiment_id found to the set of object_ids that have it.

    Args:
        * renders_root: root folder of renders (same tree every Mapping/
          script uses)
        * symmetry_type: "axis_sym" or "plane_sym"

    Returns:
        * dict[str, set[str]]: experiment_id -> object_ids with a
          molmo_multiview_<id>.json under <symmetry_dir>/<object_id>/*/*/

    """
    symmetry_dir = Path(renders_root) / symmetry_type
    coverage: dict[str, set[str]] = defaultdict(set)
    if not symmetry_dir.exists():
        return coverage

    for path in symmetry_dir.glob("*/*/*/molmo_multiview*.json"):
        match = _MOLMO_JSON_EXP_RE.match(path.name)
        if match:
            object_id = path.parents[2].name   # <symmetry_dir>/<object_id>/<size>/<lighting>/<file>
            coverage[match.group(1)].add(object_id)
    return coverage


def print_report(renders_root: Path, symmetry_type: str) -> None:
    """ Print the coverage/dedup report for one symmetry_type. """
    coverage = scan(renders_root, symmetry_type)
    all_objects = set().union(*coverage.values()) if coverage else set()
    deduped = set(discover_experiment_ids(renders_root, symmetry_type))

    print(f"\n=== {symmetry_type} ({len(all_objects)} objects with any Molmo2 output) ===")
    print(f"{'experiment_id':34s} {'n_objects':>10s}  status")
    print("-" * 72)
    for exp_id in sorted(coverage):
        n = len(coverage[exp_id])
        if exp_id in deduped:
            status = "kept"
        elif exp_id.endswith(NOMESH_SUFFIX) and exp_id[: -len(NOMESH_SUFFIX)] in coverage:
            status = f"dropped (duplicate of '{exp_id[: -len(NOMESH_SUFFIX)]}')"
        else:
            status = "dropped"
        print(f"{exp_id:34s} {n:10d}  {status}")

    print("-" * 72)
    print(f"Raw ids found      : {len(coverage)}")
    print(f"After dedup (auto) : {len(deduped)}  <- this many run_batch.py would actually process")


def main() -> None:
    """ CLI entrypoint. """
    p = argparse.ArgumentParser(
        description="Audit molmo_multiview_<ID>.json experiment_ids on disk -- what "
                    "`experiment_ids: auto` would hand to run_batch.py.",
    )
    p.add_argument("--renders-root", required=True)
    p.add_argument("--symmetry-type", nargs="+", default=["axis_sym", "plane_sym"],
                   choices=["axis_sym", "plane_sym"])
    args = p.parse_args()

    for symmetry_type in args.symmetry_type:
        print_report(Path(args.renders_root), symmetry_type)


if __name__ == "__main__":
    main()
