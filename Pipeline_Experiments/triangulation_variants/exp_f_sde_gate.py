# EXP-F -- PRS-Net-style acceptance gate (Gao et al., IEEE TVCG 2021,
# DOI:10.1109/TVCG.2020.3003823), plane only.
#
# Same sequential multi-plane consolidation as the baseline
# detect_planes_no_mesh, but a candidate plane is only accepted if its REAL
# SDE_ref (area-weighted surface sample, squared distance to the actual mesh
# after reflection, via gpytoolbox -- the same formula Mapping/evaluate.py
# uses to SCORE predictions) falls below a gate threshold. Motivated by the
# baseline over-predicting planes (n_planes_predicted ~= 1.9 vs GT ~= 1.18 at
# n_views=14, see Experiments/sandbox_pipeline_server_extended.ipynb): this
# uses the mesh at ESTIMATION time to prune spurious extra planes, unlike
# every other variant in this module, which is intentionally mesh-free.
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Mapping"))
from estimate_symmetry_no_mesh import estimate_plane_no_mesh  # noqa: E402
from evaluate import calplaneloss, normal_origin_to_plane  # noqa: E402
from pipeline_common.triangulation import get_point_by_obj_id  # noqa: E402

SYMMETRY_TYPES = ("plane_sym",)
NEEDS_MESH = True

SDE_GATE_DEFAULT = 0.02   # candidate planes above this real SDE_ref are rejected


def detect_planes(
    points_by_image: dict, images_sent: list[dict], fov_deg: float, image_size: int,
    edge_on_thresh: float, max_planes: int, dup_angle_thresh_deg: float, mesh_ctx: dict,
) -> list[dict]:
    """ Sequential multi-plane consolidation gated by the real mesh's SDE_ref.

    Args:
        * points_by_image: Molmo2 points, one list per view index (string keys)
        * images_sent: camera pose entries (R/T), one per view, aligned by index
        * fov_deg: camera field of view in degrees
        * image_size: render size in pixels (square)
        * edge_on_thresh: |cos(angle)| below this counts a view as "edge-on"
        * max_planes: stop once this many planes have been accepted
        * dup_angle_thresh_deg: candidate planes closer than this are duplicates
        * mesh_ctx: {"vertices", "faces", "surface_sample", "sde_gate"} -- see
          estimate_symmetry_variants.py::_build_mesh_ctx. Required (unlike
          every other variant, which accepts mesh_ctx=None).

    Returns:
        * list[dict]: accepted planes, in acceptance order, each with an
          extra "sde_ref_gate" field recording the real SDE_ref that let it
          pass the gate

    """
    if mesh_ctx is None:
        raise ValueError("exp_f_sde_gate requires mesh_ctx (pass --objects-root to the CLI)")

    mesh_v, mesh_f, sample = mesh_ctx["vertices"], mesh_ctx["faces"], mesh_ctx["surface_sample"]
    sde_gate = mesh_ctx.get("sde_gate", SDE_GATE_DEFAULT)

    def _ang(v1: np.ndarray, v2: np.ndarray) -> float:
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        return float(np.degrees(np.arccos(np.clip(np.abs(np.dot(v1, v2)), 0.0, 1.0))))

    all_view_idxs = sorted(
        int(k) for k, pts in points_by_image.items()
        if get_point_by_obj_id(pts, 1) is not None and get_point_by_obj_id(pts, 2) is not None
    )
    pool = set(all_view_idxs)
    planes: list[dict] = []

    while len(planes) < max_planes:
        if len(pool) < 4:
            break
        sub_points = {k: v for k, v in points_by_image.items() if int(k) in pool}
        try:
            pred = estimate_plane_no_mesh(sub_points, images_sent, fov_deg, image_size, edge_on_thresh)
        except ValueError:
            break

        normal = np.array(pred["normal"])
        if any(_ang(normal, np.array(p["normal"])) < dup_angle_thresh_deg for p in planes):
            break

        plane_vec = normal_origin_to_plane(pred["normal"], pred["origin"])
        real_sde = calplaneloss(plane_vec, mesh_v, mesh_f, sample)
        # `real_sde > sde_gate` is False for NaN too, so without the explicit
        # isfinite check a degenerate mesh triangle (gpytoolbox divides by a
        # zero-area face's normal in barycentric_coordinates.py -> NaN) would
        # silently PASS the gate instead of failing it -- reject, don't accept.
        if not np.isfinite(real_sde) or real_sde > sde_gate:
            break   # candidate doesn't hold up against the real surface -- stop consolidating

        pred["sde_ref_gate"] = real_sde
        planes.append(pred)
        pool -= set(pred["good_views"])

    return planes
