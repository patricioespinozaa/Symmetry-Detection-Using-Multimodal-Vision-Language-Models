"""
triangulation.py
-----------------
Geometry for the mesh-free symmetry pipeline (docs/pipeline_sin_malla.md):
recovers a 3D line (the symmetry axis, or a line lying within the symmetry
plane) by triangulating "interpretation planes" built from calibrated camera
rays -- never touches the mesh. Used by Mapping/estimate_symmetry_no_mesh.py.

Migrated from the prototype validated in test_pipeline_sin_malla.ipynb (see
docs/implementacion_pipeline_sin_malla.md S2.1 for the full design writeup);
the only functional change from the notebook version is `widest_pair`, which
did not exist there, and that `fov_deg`/`image_size` are function parameters
here instead of notebook-global constants (a real dataset has objects at
different sizes/fov, unlike the notebook's single test object).
"""
from __future__ import annotations

import numpy as np

from pipeline_common.camera import build_camera_rays, molmo_to_ndc


def ray_dir_for_point(
    x: float, y: float,
    R: list[list[float]], T: list[float],
    fov_deg: float, image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """World-space (camera_center, ray_direction) for a Molmo2 (x, y) point."""
    ndc_x, ndc_y = molmo_to_ndc(x, y)
    return build_camera_rays(ndc_x, ndc_y, R, T, fov_deg, image_size)


def view_forward_direction(
    R: list[list[float]], T: list[float],
    fov_deg: float, image_size: int,
) -> np.ndarray:
    """World-space direction the camera is looking (its optical axis)."""
    _, direction = build_camera_rays(0.0, 0.0, R, T, fov_deg, image_size)
    return direction


def interpretation_plane_normal(dir_a: np.ndarray, dir_b: np.ndarray) -> np.ndarray | None:
    """
    Normal of the plane containing the camera center and two rays from it.
    Any 3D line whose projection in this view passes through both (x_a, y_a)
    and (x_b, y_b) must lie entirely in this plane (Bartoli & Sturm 2005) --
    see docs/pipeline_sin_malla.md S3.1. Returns None if the two rays are
    (numerically) parallel, i.e. the two points are too close together.
    """
    n = np.cross(dir_a, dir_b)
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return None
    return n / norm


def triangulate_line(
    camera_centers: list[np.ndarray],
    plane_normals: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Intersects >=2 interpretation planes (each: normal n_i, passes through
    camera center C_i) to recover a 3D line, by least squares:
      - direction: right singular vector of smallest singular value of the
        stacked normals (the common null-space direction -- the line
        direction lies in every plane, so it's orthogonal to every normal).
      - a point on the line: solves n_i . p = n_i . C_i by least squares.

    Requires len(camera_centers) == len(plane_normals) >= 2.
    """
    N = np.asarray(plane_normals, dtype=np.float64)   # (k, 3)
    C = np.asarray(camera_centers, dtype=np.float64)  # (k, 3)

    _, _, Vt = np.linalg.svd(N)
    direction = Vt[-1]
    direction /= np.linalg.norm(direction)

    b = np.einsum("ij,ij->i", N, C)
    point, *_ = np.linalg.lstsq(N, b, rcond=None)

    return point, direction


def get_point_by_obj_id(pts: list[dict], obj_id: int) -> dict | None:
    """Look up a point by its fixed obj_id within one view's point list."""
    return next((p for p in pts if p["obj_id"] == obj_id), None)


def widest_pair(pts: list[dict]) -> tuple[dict, dict] | None:
    """
    Of ALL the points a view returned (regardless of obj_id), returns the
    pair with the largest 2D pixel distance between them -- or None if fewer
    than 2 points are available.

    Generalizes get_point_by_obj_id(pts, 1)/(pts, 2) so callers don't depend
    on a fixed-role convention ("top"/"bottom"): with exactly 2 points
    (Flow A) this returns those same 2 points (the only pair possible), so
    it is a strict generalization, not a behavior change, for every prompt
    already validated with the fixed-obj_id convention. It also degrades
    gracefully with free-form point counts (returns None with 0-1 points,
    and picks the most-separated pair rather than every combination when
    there are 3+, avoiding injecting multiple near-duplicate interpretation
    planes from the same view -- see docs/implementacion_pipeline_sin_malla.md
    S2.1 for the full geometric justification and known limits (a view with
    exactly 1 point per image, as observed for some Flow C prompts, still
    cannot be used -- no pair exists regardless of selection strategy).
    """
    if len(pts) < 2:
        return None
    best_pair, best_dist_sq = None, -1.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dx = pts[i]["x"] - pts[j]["x"]
            dy = pts[i]["y"] - pts[j]["y"]
            dist_sq = dx * dx + dy * dy
            if dist_sq > best_dist_sq:
                best_dist_sq, best_pair = dist_sq, (pts[i], pts[j])
    return best_pair
