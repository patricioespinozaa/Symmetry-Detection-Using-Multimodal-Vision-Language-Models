"""
molmo_runner.py
---------------
Entry point for batch Molmo2 pointing inference.

Splits the list of objects into N equal slices and assigns each slice to a
GPU via the --gpu-id / --num-gpus arguments.  Each process runs independently
— just set CUDA_VISIBLE_DEVICES before launching.

Typical 2-GPU setup (run each in its own tmux pane):

    CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_runner.py \\
        --renders-root ../data/renders \\
        --symmetry-type axis_sym \\
        --gpu-id 0 --num-gpus 2

    CUDA_VISIBLE_DEVICES=1 python MolmoPointing/molmo_runner.py \\
        --renders-root ../data/renders \\
        --symmetry-type plane_sym \\
        --gpu-id 1 --num-gpus 2

You can also override sizes, lightings, and view-groups:

    --sizes 224 448 1024
    --lightings flat darker brighter
    --view-groups 6 14 26 42 62 86 114

Resumability
------------
Objects are skipped if <object_dir>/molmo_done.txt already exists.
To reprocess a specific object, delete that file and rerun.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

from molmo_batch import (
    ILLUMINATIONS,
    IMAGE_SIZES,
    VIEW_GROUPS,
    process_object,
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Batch Molmo2 pointing inference — single GPU slice.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--renders-root",  type=str, required=True,
                   help="Root folder produced by data_render.py")
    p.add_argument("--symmetry-type", type=str, required=True,
                   choices=["axis_sym", "plane_sym"],
                   help="Symmetry type subfolder to process")

    p.add_argument("--gpu-id",   type=int, default=0,
                   help="Index of this process (0-based)")
    p.add_argument("--num-gpus", type=int, default=1,
                   help="Total number of parallel GPU processes")

    p.add_argument("--sizes",       type=int,  nargs="+", default=IMAGE_SIZES,
                   help="Image sizes to process")
    p.add_argument("--lightings",   type=str,  nargs="+", default=ILLUMINATIONS,
                   choices=["flat", "darker", "brighter"],
                   help="Illumination modes to process")
    p.add_argument("--view-groups", type=int,  nargs="+", default=VIEW_GROUPS,
                   help="View-group sizes (first N viewpoint indices)")

    p.add_argument("--no-vis", action="store_true",
                   help="Skip saving annotated PNG images (faster, less disk)")

    return p.parse_args()


# ── Preview ───────────────────────────────────────────────────────────────────

def preview(args, object_dirs: list[Path]) -> None:
    n_configs = len(args.sizes) * len(args.lightings) * len(args.view_groups)
    # Approximate: each view_group n means n×4 images per config
    # Use max group for upper-bound estimate
    max_imgs = max(args.view_groups) * 4
    est_total = len(object_dirs) * n_configs * max_imgs

    print("\n========== MOLMO RUNNER — EXECUTION PLAN ==========")
    print(f"Renders root:    {args.renders_root}")
    print(f"Symmetry type:   {args.symmetry_type}")
    print(f"GPU id / total:  {args.gpu_id} / {args.num_gpus}")
    print(f"Objects (slice): {len(object_dirs)}")
    print(f"Sizes:           {args.sizes}")
    print(f"Lightings:       {args.lightings}")
    print(f"View groups:     {args.view_groups}")
    print(f"Save vis PNG:    {not args.no_vis}")
    print(f"Est. max images: ~{est_total:,}  (upper bound, completed objects skipped)")
    print("====================================================\n")

    confirm = input("Type 'OK' to start: ").strip()
    if confirm != "OK":
        print("Cancelled.")
        sys.exit(0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    renders_root  = Path(args.renders_root)
    symmetry_dir  = renders_root / args.symmetry_type

    if not symmetry_dir.exists():
        print(f"[error] Directory not found: {symmetry_dir}")
        sys.exit(1)

    # Collect and sort all object directories
    all_objects = sorted([d for d in symmetry_dir.iterdir() if d.is_dir()])

    if not all_objects:
        print(f"[error] No object folders found in {symmetry_dir}")
        sys.exit(1)

    # Slice for this GPU: round-robin assignment
    gpu_objects = all_objects[args.gpu_id :: args.num_gpus]

    preview(args, gpu_objects)

    # Process
    for obj_dir in tqdm(
        gpu_objects,
        desc=f"GPU {args.gpu_id}",
        unit="obj",
        dynamic_ncols=True,
        bar_format="{desc:<12} {bar:40} {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ):
        process_object(
            object_dir    = obj_dir,
            symmetry_type = args.symmetry_type,
            sizes         = args.sizes,
            lightings     = args.lightings,
            view_groups   = args.view_groups,
            save_vis      = not args.no_vis,
        )

    print(f"\n[GPU {args.gpu_id}] All done.")


if __name__ == "__main__":
    main()
