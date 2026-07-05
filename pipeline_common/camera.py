"""
camera.py
---------
Camera/ray math shared by the ray-casting stage (Mapping/map_to_3d.py) and the
Polyscope debug viewer (Mapping/visualize_rays.py).

Coordinate conventions
----------------------
- Molmo2 coords: (x, y) in [0, 1000], origin top-left, independent of the
  image's actual pixel resolution.
- NDC: [-1, 1], top-left = (-1, +1) (OpenGL-style).
- PyTorch3D row-vector convention: p_cam = p_world @ R + T, so the camera
  centre in world space is C = -(R @ T).
"""
from __future__ import annotations

import numpy as np
import trimesh


def molmo_to_ndc(x: float, y: float) -> tuple[float, float]:
    """Convert Molmo2 coords (0-1000, top-left origin) to NDC ([-1, 1])."""
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

    Args:
        ndc_x, ndc_y: NDC coordinates in [-1, 1]
        R:            3x3 rotation matrix (world -> camera), row-major
        T:            translation vector (world -> camera)
        fov_deg:      field of view in degrees (full angle)
        image_size:   image width = height in pixels (unused; kept for
                      call-site symmetry with cast_ray_patch)

    Returns:
        ray_origin:    (3,) world-space camera centre
        ray_direction: (3,) normalized world-space ray direction
    """
    R_np = np.array(R, dtype=np.float64)
    T_np = np.array(T, dtype=np.float64)

    ray_origin = -(R_np @ T_np)

    half_tan = np.tan(np.deg2rad(fov_deg) / 2.0)
    dir_cam = np.array([
        ndc_x * half_tan,
        ndc_y * half_tan,
        1.0,
    ], dtype=np.float64)

    dir_world = R_np @ dir_cam
    dir_world /= np.linalg.norm(dir_world)

    return ray_origin, dir_world


def cast_ray(
    mesh: trimesh.Trimesh,
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
) -> dict:
    """
    Cast a single ray against the mesh. Keeps the closest intersection.

    Returns dict with:
        hit       (bool)
        point_3d  (list[float] | None)
        face_id   (int | None)
    """
    origins    = ray_origin[None, :]
    directions = ray_direction[None, :]

    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=origins,
        ray_directions=directions,
        multiple_hits=False,
    )

    if len(locations) == 0:
        return {"hit": False, "point_3d": None, "face_id": None}

    distances = np.linalg.norm(locations - ray_origin, axis=1)
    closest   = np.argmin(distances)

    return {
        "hit":      True,
        "point_3d": locations[closest].tolist(),
        "face_id":  int(index_tri[closest]),
    }


def cast_ray_patch(
    mesh: trimesh.Trimesh,
    R: list[list[float]],
    T: list[float],
    x: float,
    y: float,
    patch_size: int,
    image_width: int,
    image_height: int,
    fov_deg: float,
) -> dict:
    """
    Patch-based backprojection: casts a patch_size x patch_size grid of
    sub-rays around the Molmo (x, y) point and averages the 3D hit points of
    the sub-rays that hit the mesh.

    Exact single-pixel ray casting is sensitive to grazing angles: when a ray
    is nearly tangent to the surface, small pixel-localization errors produce
    large jumps in the resulting 3D point. Averaging a small neighborhood of
    sub-rays stabilizes the result against local curvature and VLM
    localization noise.

    Only meant for patch_size > 1 -- patch_size == 1 should go through
    cast_ray directly (identical result, without the averaging overhead).

    Returns dict with:
        hit            (bool)       -- whether ANY sub-ray hit
        point_3d       (list[float] | None) -- mean of the hitting sub-rays
        face_id        (int | None) -- the center sub-ray's face, else the
                                        first hitting sub-ray's face
        patch_size     (int)
        n_patch_hits   (int)        -- how many of the patch_size**2 sub-rays hit
        n_patch_total  (int)        -- patch_size**2
    """
    ndc_x, ndc_y = molmo_to_ndc(x, y)
    half   = patch_size // 2
    dx_ndc = 2.0 / image_width
    dy_ndc = 2.0 / image_height

    hit_points: list[list[float]] = []
    center_face_id: int | None = None
    first_face_id:  int | None = None

    for drow in range(-half, half + 1):
        for dcol in range(-half, half + 1):
            # Molmo y grows downward; NDC y grows upward, so row offsets flip sign.
            sub_ndc_x = ndc_x + dcol * dx_ndc
            sub_ndc_y = ndc_y - drow * dy_ndc

            origin, direction = build_camera_rays(
                sub_ndc_x, sub_ndc_y, R, T, fov_deg, image_width
            )
            result = cast_ray(mesh, origin, direction)

            if result["hit"]:
                hit_points.append(result["point_3d"])
                if first_face_id is None:
                    first_face_id = result["face_id"]
                if drow == 0 and dcol == 0:
                    center_face_id = result["face_id"]

    n_total = patch_size * patch_size
    n_hits  = len(hit_points)

    if n_hits == 0:
        return {
            "hit": False, "point_3d": None, "face_id": None,
            "patch_size": patch_size, "n_patch_hits": 0, "n_patch_total": n_total,
        }

    avg_point = np.asarray(hit_points, dtype=np.float64).mean(axis=0)
    return {
        "hit":      True,
        "point_3d": avg_point.tolist(),
        "face_id":  center_face_id if center_face_id is not None else first_face_id,
        "patch_size":    patch_size,
        "n_patch_hits":  n_hits,
        "n_patch_total": n_total,
    }
