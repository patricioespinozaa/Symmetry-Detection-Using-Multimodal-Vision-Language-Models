"""
cleanup_experiments.py
----------------------
Removes specific n_views keys from molmo_multiview.json files.
Useful when iterating on prompts — delete only the keys you want to rerun
without touching other results in the same JSON.

If after removing the specified keys the JSON becomes empty, the file is
deleted entirely. If it still has other keys, it is rewritten without the
removed ones.

Usage
-----
    # Dry run — shows what would be deleted without changing anything
    python Mapping/cleanup_experiments.py \\
        --renders-root ../data/renders \\
        --symmetry-type axis_sym \\
        --sizes 224 \\
        --lightings flat \\
        --view-groups 1 6 14 26 \\
        --dry-run

    # Actually delete (production file)
    python Mapping/cleanup_experiments.py \\
        --renders-root ../data/renders \\
        --symmetry-type axis_sym \\
        --sizes 224 \\
        --lightings flat \\
        --view-groups 1 6 14 26

    # Experiment file (--experiment-id)
    python Mapping/cleanup_experiments.py \\
        --renders-root ../data/renders \\
        --symmetry-type plane_sym \\
        --sizes 224 --lightings flat \\
        --view-groups 1 6 \\
        --experiment-id plane_v00

    # All sizes and lightings
    python Mapping/cleanup_experiments.py \\
        --renders-root ../data/renders \\
        --symmetry-type axis_sym \\
        --view-groups 1 6 14 26
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────

MOLMO_JSON_BASE = "molmo_multiview.json"

DEFAULT_SIZES       = [224, 448, 1136]
DEFAULT_ILLUMINATIONS = ["flat", "brighter", "darker"]


# ── Core logic ────────────────────────────────────────────────────────────────

def _molmo_filename(experiment_id: str | None) -> str:
    if not experiment_id:
        return MOLMO_JSON_BASE
    dot = MOLMO_JSON_BASE.rfind(".")
    return f"{MOLMO_JSON_BASE[:dot]}_{experiment_id}{MOLMO_JSON_BASE[dot:]}"


def cleanup_object(
    object_dir:    Path,
    view_groups:   list[str],    # string keys, e.g. ["1", "6", "14", "26"]
    sizes:         list[int],
    lightings:     list[str],
    dry_run:       bool,
    experiment_id: str | None = None,
) -> dict:
    """
    Remove specified view-group keys from all matching molmo_multiview[_EXP].json
    files under this object directory.

    Returns a summary dict with counts of files modified, deleted, and skipped.
    """
    modified  = 0
    deleted   = 0
    skipped   = 0
    json_name = _molmo_filename(experiment_id)

    for size in sizes:
        for lighting in lightings:
            json_path = object_dir / str(size) / lighting / json_name

            if not json_path.exists():
                skipped += 1
                continue

            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)

            keys_present = [k for k in view_groups if k in data]

            if not keys_present:
                skipped += 1
                continue

            # Remove the requested keys
            remaining = {k: v for k, v in data.items() if k not in view_groups}

            if dry_run:
                if remaining:
                    print(f"  [dry-run] Would remove keys {keys_present} from {json_path}")
                    modified += 1
                else:
                    print(f"  [dry-run] Would delete (empty after removal): {json_path}")
                    deleted += 1
                continue

            if remaining:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(remaining, f, indent=2)
                modified += 1
            else:
                json_path.unlink()
                deleted += 1

    return {"modified": modified, "deleted": deleted, "skipped": skipped}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Remove specific n_views keys from molmo_multiview.json files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--renders-root",  required=True,
                   help="Root folder of renders")
    p.add_argument("--symmetry-type", required=True,
                   choices=["axis_sym", "plane_sym"])
    p.add_argument("--view-groups",   required=True, type=int, nargs="+",
                   help="View-group keys to remove (e.g. 1 6 14 26)")
    p.add_argument("--sizes",     type=int, nargs="+", default=DEFAULT_SIZES)
    p.add_argument("--lightings", type=str, nargs="+",
                   default=DEFAULT_ILLUMINATIONS,
                   choices=["flat", "darker", "brighter"])
    p.add_argument("--experiment-id", default=None,
                   help="Experiment ID. Targets molmo_multiview_<ID>.json instead of "
                        "molmo_multiview.json. Must match the --experiment-id used in the runner.")
    p.add_argument("--dry-run", action="store_true",
                   help="Preview what would be deleted without making changes")
    return p.parse_args()


def preview(args: argparse.Namespace, n_objects: int) -> None:
    mode      = "DRY RUN — no changes will be made" if args.dry_run else "LIVE — files will be modified/deleted"
    json_name = _molmo_filename(args.experiment_id)
    print("\n========== CLEANUP EXPERIMENTS ==========")
    print(f"Renders root  : {args.renders_root}")
    print(f"Symmetry type : {args.symmetry_type}")
    print(f"Target file   : {json_name}")
    if args.experiment_id:
        print(f"Experiment ID : {args.experiment_id}")
    print(f"View groups   : {args.view_groups}  ← keys to remove")
    print(f"Sizes         : {args.sizes}")
    print(f"Lightings     : {args.lightings}")
    print(f"Objects       : {n_objects}")
    print(f"Mode          : {mode}")
    print("==========================================\n")
    if not args.dry_run:
        confirm = input("Type 'OK' to proceed: ").strip()
        if confirm != "OK":
            print("Cancelled.")
            sys.exit(0)


def main() -> None:
    args = parse_args()

    symmetry_dir = Path(args.renders_root) / args.symmetry_type
    if not symmetry_dir.exists():
        print(f"[error] Not found: {symmetry_dir}")
        sys.exit(1)

    all_objects  = sorted(d for d in symmetry_dir.iterdir() if d.is_dir())
    view_groups  = [str(n) for n in args.view_groups]   # convert to string keys

    preview(args, len(all_objects))

    total_modified = 0
    total_deleted  = 0
    total_skipped  = 0

    for obj_dir in tqdm(all_objects, unit="obj", dynamic_ncols=True):
        summary = cleanup_object(
            object_dir    = obj_dir,
            view_groups   = view_groups,
            sizes         = args.sizes,
            lightings     = args.lightings,
            dry_run       = args.dry_run,
            experiment_id = args.experiment_id,
        )
        total_modified += summary["modified"]
        total_deleted  += summary["deleted"]
        total_skipped  += summary["skipped"]

    print(f"\n{'[dry-run] ' if args.dry_run else ''}Done.")
    print(f"  JSON files modified (keys removed) : {total_modified}")
    print(f"  JSON files deleted  (now empty)    : {total_deleted}")
    print(f"  Configs skipped     (no JSON/keys) : {total_skipped}")


if __name__ == "__main__":
    main()
