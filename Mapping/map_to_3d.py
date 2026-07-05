"""
map_to_3d.py
------------
Projects Molmo2 2D pointing coordinates onto the 3D mesh surface via ray casting.

For each object, reads molmo_multiview.json (produced by molmo_multiview_runner.py),
reconstructs the camera ray for each predicted point, intersects it with the mesh,
and saves the resulting 3D points to mapped_points_3d.json alongside the renders.

Camera model
------------
PyTorch3D FoVPerspectiveCamera with fov=60 (degrees), square image.
R and T are stored in metadata_all.json per viewpoint.

Coordinate convention
---------------------
- Molmo2 coords: (x, y) in [0, 1000], origin top-left
- Pixel coords:  (px, py) normalized to [-1, 1] NDC (top-left = (-1, -1))
- Ray casting:   trimesh, world space (same as .obj)

Patch-based backprojection
--------------------------
--patch-size 1 (default) casts one ray per Molmo point, unchanged from the
original behavior. --patch-size 3/5 instead averages the 3D hit points of a
patch_size x patch_size grid of sub-rays around the point (see
pipeline_common.camera.cast_ray_patch), which stabilizes the result against
grazing-angle localization noise. Output goes to a separate
mapped_points_3d_p{patch_size}[_EXP].json file so it never collides with the
exact-mode output.

Output
------
<renders_root>/<symmetry_type>/<object_id>/<image_size>/<illumination>/
    mapped_points_3d.json                    (--patch-size 1, default)
    mapped_points_3d_p3.json                 (--patch-size 3)

JSON format
-----------
{
  "object_id": "...",
  "symmetry_type": "...",
  "image_size": 224,
  "illumination": "flat",
  "fov_deg": 60.0,
  "patch_size": 1,
  "n_views_results": {
    "1": {
      "images_sent": [...],          # from molmo_multiview.json
      "points_3d": [                 # one entry per Molmo point
        {
          "img_idx":    0,           # which image in the sent set (0-based)
          "obj_id":     1,           # Molmo obj_id
          "molmo_x":    450.0,       # original Molmo coord
          "molmo_y":    230.0,
          "hit":        true,        # whether ray intersected the mesh
          "point_3d":   [x, y, z],  # 3D intersection point (null if no hit)
          "face_id":    42,          # mesh face index (null if no hit)
          # present only when --patch-size > 1:
          "patch_size":    3,
          "n_patch_hits":  7,
          "n_patch_total": 9
        },
        ...
      ],
      "n_hits":    2,
      "n_misses":  0,
    },
    "6": { ... },
    ...
  }
}

Usage
-----
    python Mapping/map_to_3d.py \\
        --renders-root ../data/renders \\
        --objects-root ../data/objects \\
        --symmetry-type axis_sym \\
        --sizes 224 \\
        --lightings flat \\
        --fov 60.0

    # Two GPUs (CPU-only task, but uses same object slicing for parallelism)
    python Mapping/map_to_3d.py \\
        --renders-root ../data/renders \\
        --objects-root ../data/objects \\
        --symmetry-type axis_sym \\
        --gpu-id 0 --num-gpus 2

    # Patch-based backprojection (h=3)
    python Mapping/map_to_3d.py \\
        --renders-root ../data/renders \\
        --objects-root ../data/objects \\
        --symmetry-type axis_sym \\
        --sizes 224 --lightings flat \\
        --patch-size 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_common.naming import exp_filename
from pipeline_common.datasets import OBJECTS_SUBDIR, load_mesh
from pipeline_common.camera import molmo_to_ndc, build_camera_rays, cast_ray, cast_ray_patch

# ── Constants ─────────────────────────────────────────────────────────────────

FOV_DEG        = 60.0
OUTPUT_FILE    = "mapped_points_3d.json"
MOLMO_JSON     = "molmo_multiview.json"
MANIFEST_FILE  = "manifest.json"

DEFAULT_SIZES       = [224, 448, 1136]
DEFAULT_LIGHTINGS   = ["flat", "brighter", "darker"]
PATCH_SIZES         = (1, 3, 5)


# ── Per-object mapping ────────────────────────────────────────────────────────

def process_object(
    object_dir:    Path,
    obj_path:      Path,
    sizes:         list[int],
    lightings:     list[str],
    fov_deg:       float,
    overwrite:     bool = False,
    experiment_id: str | None = None,
    patch_size:    int = 1,
) -> None:
    """
    Map Molmo2 predictions to 3D for all (size, illumination) configs of one object.
    Skips configs where the output JSON already exists unless --overwrite.
    When experiment_id is set, reads molmo_multiview_<ID>.json and writes
    mapped_points_3d_<ID>.json instead of the default filenames.

    patch_size == 1 (default) reproduces exact single-ray backprojection,
    unchanged from the original behavior. patch_size in (3, 5) averages a
    patch_size x patch_size grid of sub-rays per point (see
    pipeline_common.camera.cast_ray_patch) and writes to a separate
    p{patch_size}-tagged output file.
    """
    object_id   = object_dir.name
    molmo_json  = exp_filename(MOLMO_JSON, experiment_id)
    patch_tag   = f"p{patch_size}" if patch_size != 1 else None
    exp_out     = (f"{experiment_id}_{patch_tag}" if experiment_id else patch_tag) if patch_tag else experiment_id
    output_file = exp_filename(OUTPUT_FILE, exp_out)

    # Load mesh once per object (shared across all size/lighting configs)
    try:
        mesh = load_mesh(obj_path)
    except Exception as e:
        print(f"  [warn] Could not load mesh {obj_path}: {e}")
        return

    for size in sizes:
        for lighting in lightings:
            render_dir = object_dir / str(size) / lighting
            if not render_dir.exists():
                continue

            molmo_json_path = render_dir / molmo_json
            if not molmo_json_path.exists():
                continue

            output_path = render_dir / output_file
            if output_path.exists() and not overwrite:
                continue   # already mapped — skip

            # Read image_size from manifest if available, else use folder name
            manifest_path = render_dir / MANIFEST_FILE
            if manifest_path.exists():
                with open(manifest_path, encoding="utf-8") as f:
                    manifest   = json.load(f)
                image_size = manifest.get("image_size", size)
                fov        = manifest.get("fov", fov_deg)
            else:
                image_size = size
                fov        = fov_deg

            with open(molmo_json_path, encoding="utf-8") as f:
                molmo_data = json.load(f)

            n_views_results = {}

            for n_views_key, group in molmo_data.items():
                images_sent     = group.get("images_sent", [])
                points_by_image = group.get("points_by_image", {})

                points_3d_all = []
                n_hits = 0
                n_misses = 0

                for img_idx_str, pts in points_by_image.items():
                    img_idx = int(img_idx_str)

                    # Get camera params for this image
                    if img_idx >= len(images_sent):
                        continue
                    cam = images_sent[img_idx]
                    R   = cam["R"]
                    T   = cam["T"]

                    for pt in pts:
                        x       = pt["x"]
                        y       = pt["y"]
                        obj_id  = pt["obj_id"]

                        if patch_size == 1:
                            ndc_x, ndc_y = molmo_to_ndc(x, y)
                            ray_origin, ray_dir = build_camera_rays(
                                ndc_x, ndc_y, R, T, fov, image_size
                            )
                            hit_result = cast_ray(mesh, ray_origin, ray_dir)
                        else:
                            hit_result = cast_ray_patch(
                                mesh, R, T, x, y, patch_size, image_size, image_size, fov
                            )

                        entry = {
                            "img_idx":  img_idx,
                            "obj_id":   obj_id,
                            "molmo_x":  x,
                            "molmo_y":  y,
                            "hit":      hit_result["hit"],
                            "point_3d": hit_result["point_3d"],
                            "face_id":  hit_result["face_id"],
                        }
                        if patch_size != 1:
                            entry["patch_size"]   = hit_result["patch_size"]
                            entry["n_patch_hits"] = hit_result["n_patch_hits"]
                            entry["n_patch_total"] = hit_result["n_patch_total"]
                        points_3d_all.append(entry)

                        if hit_result["hit"]:
                            n_hits += 1
                        else:
                            n_misses += 1

                n_views_results[n_views_key] = {
                    "images_sent": images_sent,
                    "points_3d":   points_3d_all,
                    "n_hits":      n_hits,
                    "n_misses":    n_misses,
                }

            output = {
                "object_id":       object_id,
                "symmetry_type":   object_dir.parent.name,
                "image_size":      image_size,
                "illumination":    lighting,
                "fov_deg":         fov,
                "patch_size":      patch_size,
                "n_views_results": n_views_results,
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Map Molmo2 2D coords to 3D mesh surface via ray casting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--renders-root",  required=True,
                   help="Root folder of renders (produced by data_render.py)")
    p.add_argument("--objects-root",  required=True,
                   help="Root folder containing curated_axis_sym_obj / curated_plane_sym_obj")
    p.add_argument("--symmetry-type", required=True,
                   choices=["axis_sym", "plane_sym"])

    p.add_argument("--gpu-id",   type=int, default=0)
    p.add_argument("--num-gpus", type=int, default=1)

    p.add_argument("--sizes",     type=int, nargs="+", default=DEFAULT_SIZES)
    p.add_argument("--lightings", type=str, nargs="+", default=DEFAULT_LIGHTINGS,
                   choices=["flat", "darker", "brighter"])
    p.add_argument("--fov",       type=float, default=FOV_DEG)
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing mapped_points_3d.json files")
    p.add_argument("--experiment-id", default=None,
                   help=(
                       "Experiment identifier. Reads molmo_multiview_<ID>.json and "
                       "writes mapped_points_3d_<ID>.json. Must match the --experiment-id "
                       "used in molmo_multiview_runner.py."
                   ))
    p.add_argument("--max-objects", type=int, default=None,
                   help="Limit to the first N objects (sorted order).")
    p.add_argument("--yes", "-y", action="store_true",
                   help="Skip the confirmation prompt (useful for automated loops).")
    p.add_argument("--patch-size", type=int, default=1, choices=PATCH_SIZES,
                   help=(
                       "1 = exact single-ray backprojection (default, unchanged output). "
                       "3 or 5 = average a patch_size x patch_size grid of sub-rays per "
                       "point, written to a separate p{patch_size}-tagged output file."
                   ))

    return p.parse_args()


def preview(args: argparse.Namespace, objects: list[Path]) -> None:
    patch_tag   = f"p{args.patch_size}" if args.patch_size != 1 else None
    exp_out     = (f"{args.experiment_id}_{patch_tag}" if args.experiment_id else patch_tag) if patch_tag else args.experiment_id
    output_file = exp_filename(OUTPUT_FILE, exp_out)
    print("\n========== MAP TO 3D ==========")
    print(f"Renders root  : {args.renders_root}")
    print(f"Objects root  : {args.objects_root}")
    print(f"Symmetry type : {args.symmetry_type}")
    if args.experiment_id:
        print(f"Experiment ID : {args.experiment_id}  →  {output_file}")
    print(f"GPU id/total  : {args.gpu_id} / {args.num_gpus}")
    print(f"Objects       : {len(objects)}")
    print(f"Sizes         : {args.sizes}")
    print(f"Lightings     : {args.lightings}")
    print(f"FoV           : {args.fov}°")
    print(f"Patch size    : {args.patch_size}" + (" (exact, single ray)" if args.patch_size == 1 else f" ({args.patch_size}x{args.patch_size} averaged sub-rays)"))
    print(f"Output file   : {output_file}")
    print(f"Overwrite     : {args.overwrite}")
    if not args.overwrite:
        print(f"(Existing {output_file} skipped — use --overwrite to replace)")
    else:
        print(f"(Existing {output_file} will be overwritten)")
    print("================================\n")
    if not args.yes:
        if input("Type 'OK' to start: ").strip() != "OK":
            print("Cancelled.")
            sys.exit(0)


def main() -> None:
    args = parse_args()

    symmetry_dir = Path(args.renders_root) / args.symmetry_type
    if not symmetry_dir.exists():
        print(f"[error] Not found: {symmetry_dir}")
        sys.exit(1)

    objects_subdir = OBJECTS_SUBDIR[args.symmetry_type]
    objects_dir    = Path(args.objects_root) / objects_subdir

    all_objects = sorted(d for d in symmetry_dir.iterdir() if d.is_dir())
    if args.max_objects:
        all_objects = all_objects[:args.max_objects]
    objects = all_objects[args.gpu_id :: args.num_gpus]

    preview(args, objects)

    for obj_dir in tqdm(
        objects,
        desc=f"Mapping",
        unit="obj",
        dynamic_ncols=True,
        bar_format="{desc:<12} {bar:40} {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ):
        obj_path = objects_dir / f"{obj_dir.name}.obj"
        if not obj_path.exists():
            print(f"  [warn] .obj not found: {obj_path}")
            continue

        process_object(
            object_dir    = obj_dir,
            obj_path      = obj_path,
            sizes         = args.sizes,
            lightings     = args.lightings,
            fov_deg       = args.fov,
            overwrite     = args.overwrite,
            experiment_id = args.experiment_id,
            patch_size    = args.patch_size,
        )

    print(f"\nDone.")


if __name__ == "__main__":
    main()