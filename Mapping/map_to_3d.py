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

Output
------
<renders_root>/<symmetry_type>/<object_id>/<image_size>/<illumination>/
    mapped_points_3d.json

JSON format
-----------
{
  "object_id": "...",
  "symmetry_type": "...",
  "image_size": 224,
  "illumination": "flat",
  "fov_deg": 60.0,
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
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────

FOV_DEG        = 60.0
OUTPUT_FILE    = "mapped_points_3d.json"
MOLMO_JSON     = "molmo_multiview.json"
MANIFEST_FILE  = "manifest.json"

DEFAULT_SIZES       = [224, 448, 1136]
DEFAULT_LIGHTINGS   = ["flat", "brighter", "darker"]
OBJECTS_SUBDIR      = {"axis_sym": "curated_axis_sym_obj",
                       "plane_sym": "curated_plane_sym_obj"}


# ── Camera helpers ────────────────────────────────────────────────────────────

def molmo_to_ndc(x: float, y: float, image_size: int) -> tuple[float, float]:
    """
    Convert Molmo2 coords (0–1000, top-left origin) to
    NDC ([-1, 1], top-left = (-1, +1) in OpenGL convention).

    Molmo x → NDC x:  (x / 1000) * 2 - 1        (left=-1, right=+1)
    Molmo y → NDC y:  1 - (y / 1000) * 2         (top=+1,  bottom=-1)
    """
    ndc_x = (x / 1000.0) * 2.0 - 1.0
    ndc_y = 1.0 - (y / 1000.0) * 2.0
    return ndc_x, ndc_y


def build_camera_rays(
    ndc_x: float,
    ndc_y: float,
    R: list[list[float]],
    T: list[float],
    fov_deg: float,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a camera ray in world space for a given NDC point.

    PyTorch3D convention:
      - Camera looks along +Z in camera space
      - R, T transform world → camera:  p_cam = R @ p_world + T
      - To go camera → world:           p_world = R^T @ (p_cam - T)

    The ray origin is the camera centre in world space.
    The ray direction is computed from the NDC point using the FoV.

    Args:
        ndc_x, ndc_y: NDC coordinates in [-1, 1]
        R:            3×3 rotation matrix (world → camera), row-major
        T:            translation vector (world → camera)
        fov_deg:      field of view in degrees (full angle)
        image_size:   image width = height in pixels

    Returns:
        ray_origin:    (3,) world-space camera centre
        ray_direction: (3,) normalized world-space ray direction
    """
    R_np = np.array(R, dtype=np.float64)       # (3, 3)
    T_np = np.array(T, dtype=np.float64)       # (3,)

    # Camera centre in world space: C = -R^T @ T
    ray_origin = -R_np.T @ T_np                # (3,)

    # Half-tangent for the full FoV
    half_tan = np.tan(np.deg2rad(fov_deg) / 2.0)

    # Direction in camera space (looking along +Z)
    # x scales with ndc_x, y scales with ndc_y (aspect=1 → same scale)
    dir_cam = np.array([
        ndc_x * half_tan,
        ndc_y * half_tan,
        1.0,
    ], dtype=np.float64)

    # Rotate to world space: dir_world = R^T @ dir_cam
    dir_world = R_np.T @ dir_cam
    dir_world /= np.linalg.norm(dir_world)

    return ray_origin, dir_world


# ── Mesh loading ──────────────────────────────────────────────────────────────

def load_mesh(obj_path: Path) -> trimesh.Trimesh:
    """Load .obj as a trimesh, merging geometry if the file has multiple meshes."""
    scene_or_mesh = trimesh.load(str(obj_path), force="mesh", process=False)
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in scene_or_mesh.geometry.values()
             if isinstance(g, trimesh.Trimesh)]
        )
    else:
        mesh = scene_or_mesh
    return mesh


# ── Ray casting ───────────────────────────────────────────────────────────────

def cast_ray(
    mesh: trimesh.Trimesh,
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
) -> dict:
    """
    Cast a single ray against the mesh.

    Returns dict with:
        hit       (bool)
        point_3d  (list[float] | None)
        face_id   (int | None)
    """
    origins    = ray_origin[None, :]      # (1, 3)
    directions = ray_direction[None, :]   # (1, 3)

    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=origins,
        ray_directions=directions,
        multiple_hits=False,
    )

    if len(locations) == 0:
        return {"hit": False, "point_3d": None, "face_id": None}

    # Take the closest hit
    distances = np.linalg.norm(locations - ray_origin, axis=1)
    closest   = np.argmin(distances)

    return {
        "hit":      True,
        "point_3d": locations[closest].tolist(),
        "face_id":  int(index_tri[closest]),
    }


# ── Per-object mapping ────────────────────────────────────────────────────────

def process_object(
    object_dir:   Path,
    obj_path:     Path,
    sizes:        list[int],
    lightings:    list[str],
    fov_deg:      float,
    overwrite:    bool = False,
) -> None:
    """
    Map Molmo2 predictions to 3D for all (size, illumination) configs of one object.
    Skips configs where mapped_points_3d.json already exists unless --overwrite.
    """
    object_id = object_dir.name

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

            molmo_json_path = render_dir / MOLMO_JSON
            if not molmo_json_path.exists():
                continue

            output_path = render_dir / OUTPUT_FILE
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

                        ndc_x, ndc_y = molmo_to_ndc(x, y, image_size)

                        ray_origin, ray_dir = build_camera_rays(
                            ndc_x, ndc_y, R, T, fov, image_size
                        )

                        hit_result = cast_ray(mesh, ray_origin, ray_dir)

                        entry = {
                            "img_idx":  img_idx,
                            "obj_id":   obj_id,
                            "molmo_x":  x,
                            "molmo_y":  y,
                            "hit":      hit_result["hit"],
                            "point_3d": hit_result["point_3d"],
                            "face_id":  hit_result["face_id"],
                        }
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

    return p.parse_args()


def preview(args: argparse.Namespace, objects: list[Path]) -> None:
    print("\n========== MAP TO 3D ==========")
    print(f"Renders root  : {args.renders_root}")
    print(f"Objects root  : {args.objects_root}")
    print(f"Symmetry type : {args.symmetry_type}")
    print(f"GPU id/total  : {args.gpu_id} / {args.num_gpus}")
    print(f"Objects       : {len(objects)}")
    print(f"Sizes         : {args.sizes}")
    print(f"Lightings     : {args.lightings}")
    print(f"FoV           : {args.fov}°")
    print(f"Overwrite      : {args.overwrite}")
    if not args.overwrite:
        print(f"(Existing {OUTPUT_FILE} skipped — use --overwrite to replace)")
    else:
        print(f"(Existing {OUTPUT_FILE} will be overwritten)")
    print("================================\n")
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
    objects     = all_objects[args.gpu_id :: args.num_gpus]

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
            object_dir = obj_dir,
            obj_path   = obj_path,
            sizes      = args.sizes,
            lightings  = args.lightings,
            fov_deg    = args.fov,
            overwrite  = args.overwrite,
        )

    print(f"\nDone.")


if __name__ == "__main__":
    main()