#!/usr/bin/env python3
"""
visualize_rays.py
-----------------
Debugging tool for the 2D→3D ray-casting pipeline.

Shows, for a single object and n_views group:
  - 3D mesh (gray)
  - Camera positions (blue spheres, one per view sent to Molmo2)
  - Hit rays  (green):  camera → mesh intersection point
  - Miss rays (orange): camera → camera + direction × ray_length
  - Hit points on the mesh surface (yellow spheres)
  - Ground-truth symmetry axis or plane, when --show-gt is given

This script recomputes all rays from scratch using the camera parameters
stored in molmo_multiview.json, so it does NOT require map_to_3d.py to
have been run first.

Requires
--------
    polyscope, trimesh, numpy

Usage
-----
python Mapping/visualize_rays.py \
    --object-id 1a9c1cbf1ca9ca24274623f5a5d0bcdc \
    --renders-root ../data/renders \
    --objects-root ../data/objects \
    --symmetry-type axis_sym \
    --size 224 \
    --lighting flat \
    --n-views 6 \
    --show-gt

# Longer miss rays, thicker tubes
python Mapping/visualize_rays.py \
    --object-id 1a9c1cbf1ca9ca24274623f5a5d0bcdc \
    --renders-root ../data/renders \
    --objects-root ../data/objects \
    --symmetry-type axis_sym \
    --n-views 14 \
    --ray-length 3.0 \
    --ray-radius 0.006
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

try:
    import polyscope as ps
except ImportError:
    sys.exit("[error] polyscope not installed — pip install polyscope")


# ── Constants ─────────────────────────────────────────────────────────────────

OBJECTS_SUBDIR = {
    "axis_sym":  "curated_axis_sym_obj",
    "plane_sym": "curated_plane_sym_obj",
}
MOLMO_JSON = "molmo_multiview.json"
FOV_DEG    = 60.0

# Colors (RGB in [0,1])
COLOR_MESH      = (0.75, 0.75, 0.75)
COLOR_CAMERA    = (0.20, 0.40, 0.90)   # blue
COLOR_HIT_RAY   = (0.15, 0.80, 0.25)   # green
COLOR_MISS_RAY  = (1.00, 0.55, 0.10)   # orange
COLOR_HIT_PT    = (1.00, 0.95, 0.10)   # yellow
COLOR_GT_AXIS   = (0.15, 0.40, 0.90)   # blue
COLOR_GT_PLANE  = (0.10, 0.80, 0.35)   # green


# ── Camera / ray helpers ──────────────────────────────────────────────────────

def molmo_to_ndc(x: float, y: float) -> tuple[float, float]:
    return (x / 1000.0) * 2.0 - 1.0, 1.0 - (y / 1000.0) * 2.0


def camera_ray(
    ndc_x: float,
    ndc_y: float,
    R: list,
    T: list,
    fov_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (origin, unit_direction) in world space.
    Convention: p_cam = R @ p_world + T  →  origin = −R^T @ T
    """
    R_np = np.array(R, dtype=np.float64)
    T_np = np.array(T, dtype=np.float64)
    origin   = -R_np.T @ T_np
    half_tan = np.tan(np.deg2rad(fov_deg) / 2.0)
    dir_cam  = np.array([ndc_x * half_tan, ndc_y * half_tan, 1.0])
    dir_world = R_np.T @ dir_cam
    dir_world /= np.linalg.norm(dir_world)
    return origin, dir_world


def cast_ray(
    mesh: trimesh.Trimesh,
    origin: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray | None:
    """Returns the closest hit point or None."""
    locs, _, _ = mesh.ray.intersects_location(
        ray_origins=origin[None],
        ray_directions=direction[None],
        multiple_hits=False,
    )
    if len(locs) == 0:
        return None
    dists = np.linalg.norm(locs - origin, axis=1)
    return locs[np.argmin(dists)]


# ── Mesh loading ──────────────────────────────────────────────────────────────

def load_mesh(obj_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(obj_path), force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(
            [g for g in loaded.geometry.values()
             if isinstance(g, trimesh.Trimesh)]
        )
    return loaded


# ── Ground-truth parser ───────────────────────────────────────────────────────

def load_gt(txt_path: Path) -> list[dict]:
    symmetries = []
    lines = [l.strip() for l in txt_path.read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    i = 0
    n = int(lines[i]); i += 1
    for _ in range(n):
        tokens = lines[i].split(); i += 1
        vals   = [float(v) for v in tokens[1:]]
        if tokens[0] == "axis":
            symmetries.append({"type": "axis",  "direction": vals[:3], "origin": vals[3:6]})
        elif tokens[0] == "plane":
            symmetries.append({"type": "plane", "normal":    vals[:3], "origin": vals[3:6]})
    return symmetries


# ── Geometry helpers (plane quad, axis segment) ───────────────────────────────

def _translate_mat(tx, ty, tz):
    m = np.eye(4, dtype=np.float32); m[:3, 3] = [tx, ty, tz]; return m

def _rotation_z_mat(theta):
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.float32)

def plane_quad(normal, origin, scale):
    n = np.asarray(normal, dtype=np.float64)
    n /= np.linalg.norm(n)
    a, b, c = float(n[0]), float(n[1]), float(n[2])
    h = float(np.sqrt(a**2 + c**2))
    base = np.array([[0,-scale,-scale],[0,scale,-scale],
                     [0,scale,scale],[0,-scale,scale]], dtype=np.float32)
    if h < 1e-7:
        T = _translate_mat(*origin) @ _rotation_z_mat(np.pi / 2)
    else:
        Rzinv = np.array([[h,-b,0,0],[b,h,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
        Ryinv = np.array([[a/h,0,-c/h,0],[0,1,0,0],[c/h,0,a/h,0],[0,0,0,1]], dtype=np.float32)
        T = _translate_mat(*origin) @ Ryinv @ Rzinv
    pts_h = np.concatenate([base.T, np.ones((1, 4))], axis=0)
    return (T @ pts_h)[:3].T.astype(np.float64)

def plane_faces():
    return np.array([[0,1,3],[3,1,2]], dtype=np.int32)

def axis_endpoints(direction, origin, half_length):
    d = np.asarray(direction, dtype=np.float64); d /= np.linalg.norm(d)
    o = np.asarray(origin,    dtype=np.float64)
    return np.array([o - half_length * d, o + half_length * d])


# ── Polyscope curve-network batching ──────────────────────────────────────────

def make_curve_network(segments: list[tuple[np.ndarray, np.ndarray]]):
    """
    Convert a list of (start, end) pairs into (nodes, edges) arrays
    for ps.register_curve_network.
    """
    n  = len(segments)
    nodes = np.zeros((2 * n, 3), dtype=np.float64)
    edges = np.zeros((n, 2),     dtype=np.int32)
    for i, (s, e) in enumerate(segments):
        nodes[2*i]   = s
        nodes[2*i+1] = e
        edges[i]     = [2*i, 2*i+1]
    return nodes, edges


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize Molmo2 camera rays in 3D with Polyscope.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--object-id",     required=True)
    p.add_argument("--renders-root",  required=True)
    p.add_argument("--objects-root",  required=True)
    p.add_argument("--symmetry-type", required=True,
                   choices=["axis_sym", "plane_sym"])
    p.add_argument("--n-views",   type=int, required=True,
                   help="n_views group to visualize (e.g. 6)")
    p.add_argument("--size",      type=int, default=224)
    p.add_argument("--lighting",  default="flat",
                   choices=["flat", "darker", "brighter"])
    p.add_argument("--fov",       type=float, default=FOV_DEG)
    p.add_argument("--show-gt",   action="store_true")
    p.add_argument("--ray-length",  type=float, default=None,
                   help="Length of miss rays. Default: 2 × bbox diagonal.")
    p.add_argument("--ray-radius",  type=float, default=0.004,
                   help="Tube radius for all rays.")
    p.add_argument("--hit-radius",  type=float, default=0.015,
                   help="Sphere radius for hit points.")
    p.add_argument("--cam-radius",  type=float, default=0.020,
                   help="Sphere radius for camera positions.")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    renders_root = Path(args.renders_root)
    objects_root = Path(args.objects_root)
    sym_type     = args.symmetry_type
    object_id    = args.object_id
    nv_key       = str(args.n_views)

    render_dir = renders_root / sym_type / object_id / str(args.size) / args.lighting
    obj_path   = objects_root / OBJECTS_SUBDIR[sym_type] / f"{object_id}.obj"
    txt_path   = objects_root / OBJECTS_SUBDIR[sym_type] / f"{object_id}.txt"
    molmo_path = render_dir / MOLMO_JSON

    for p, lbl in [(obj_path, ".obj"), (molmo_path, MOLMO_JSON)]:
        if not p.exists():
            sys.exit(f"[error] {lbl} not found: {p}")

    # ── Load mesh ──────────────────────────────────────────────────────────────
    print(f"Loading mesh : {obj_path}")
    mesh      = load_mesh(obj_path)
    vertices  = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.faces,    dtype=np.int32)
    bbox_diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))

    ray_length = args.ray_length if args.ray_length is not None else 2.0 * bbox_diag
    half_len   = 0.6 * bbox_diag   # for GT axis segment
    plane_sc   = 0.5 * bbox_diag   # for GT plane quad

    # ── Load Molmo predictions ─────────────────────────────────────────────────
    with open(molmo_path, encoding="utf-8") as f:
        molmo_data = json.load(f)

    if nv_key not in molmo_data:
        avail = list(molmo_data.keys())
        sys.exit(f"[error] n_views={nv_key} not in {MOLMO_JSON}. Available: {avail}")

    group           = molmo_data[nv_key]
    images_sent     = group.get("images_sent", [])
    points_by_image = group.get("points_by_image", {})

    # ── Compute rays ───────────────────────────────────────────────────────────
    cam_positions: list[np.ndarray] = []
    hit_segments:  list[tuple[np.ndarray, np.ndarray]] = []
    miss_segments: list[tuple[np.ndarray, np.ndarray]] = []
    hit_points:    list[np.ndarray] = []

    seen_cam_origins: set[tuple] = set()

    for img_idx_str, pts in points_by_image.items():
        img_idx = int(img_idx_str)
        if img_idx >= len(images_sent):
            continue

        cam   = images_sent[img_idx]
        R, T  = cam["R"], cam["T"]

        # Camera position (use 'eye' if present, else compute from R,T)
        if "eye" in cam:
            cam_origin = np.array(cam["eye"], dtype=np.float64)
        else:
            R_np = np.array(R, dtype=np.float64)
            T_np = np.array(T, dtype=np.float64)
            cam_origin = -R_np.T @ T_np

        key = tuple(cam_origin.round(6))
        if key not in seen_cam_origins:
            cam_positions.append(cam_origin)
            seen_cam_origins.add(key)

        for pt in pts:
            ndc_x, ndc_y = molmo_to_ndc(pt["x"], pt["y"])
            origin, direction = camera_ray(ndc_x, ndc_y, R, T, args.fov)

            hit_pt = cast_ray(mesh, origin, direction)

            if hit_pt is not None:
                hit_segments.append((origin, hit_pt))
                hit_points.append(hit_pt)
            else:
                endpoint = origin + direction * ray_length
                miss_segments.append((origin, endpoint))

    # ── GT ─────────────────────────────────────────────────────────────────────
    gt_symmetries = []
    if args.show_gt and txt_path.exists():
        gt_symmetries = load_gt(txt_path)

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\nObject       : {object_id}  ({sym_type})")
    print(f"n_views group: {nv_key}  ({len(images_sent)} images sent)")
    print(f"Cameras      : {len(cam_positions)} unique positions")
    print(f"Hit rays     : {len(hit_segments)}")
    print(f"Miss rays    : {len(miss_segments)}")
    print(f"Bbox diagonal: {bbox_diag:.4f}")
    print(f"Ray length   : {ray_length:.4f}  (miss rays)")
    if args.show_gt:
        print(f"GT elements  : {len(gt_symmetries)}")

    # ── Polyscope ──────────────────────────────────────────────────────────────
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_background_color((1.0, 1.0, 1.0))
    ps.look_at((2.0, 2.0, 2.0), (0.0, 0.0, 0.0))
    ps.set_ground_plane_mode("shadow_only")

    # Mesh
    ps.register_surface_mesh("mesh", vertices, triangles).set_color(COLOR_MESH)

    # Camera positions
    if cam_positions:
        cams = np.array(cam_positions)
        ps.register_point_cloud("cameras", cams,
                                radius=args.cam_radius).set_color(COLOR_CAMERA)

    # Hit rays
    if hit_segments:
        nodes, edges = make_curve_network(hit_segments)
        ps.register_curve_network("hit_rays", nodes, edges,
                                  radius=args.ray_radius).set_color(COLOR_HIT_RAY)

    # Miss rays
    if miss_segments:
        nodes, edges = make_curve_network(miss_segments)
        ps.register_curve_network("miss_rays", nodes, edges,
                                  radius=args.ray_radius).set_color(COLOR_MISS_RAY)

    # Hit points
    if hit_points:
        pts = np.array(hit_points)
        ps.register_point_cloud("hit_points", pts,
                                radius=args.hit_radius).set_color(COLOR_HIT_PT)

    # GT symmetry
    for i, gt in enumerate(gt_symmetries):
        origin = np.array(gt["origin"], dtype=np.float64)
        if gt["type"] == "axis":
            pts   = axis_endpoints(gt["direction"], origin, half_len)
            edges = np.array([[0, 1]], dtype=np.int32)
            net   = ps.register_curve_network(f"gt_axis_{i}", pts, edges,
                                              radius=args.ray_radius * 1.5)
            net.set_color(COLOR_GT_AXIS)
        elif gt["type"] == "plane":
            verts = plane_quad(gt["normal"], origin, plane_sc)
            faces = plane_faces()
            m     = ps.register_surface_mesh(f"gt_plane_{i}", verts, faces)
            m.set_color(COLOR_GT_PLANE)
            m.set_transparency(0.40)

    # Legend
    print()
    print("Legend")
    print(f"  Gray mesh     : 3D object")
    print(f"  Blue spheres  : camera positions ({len(cam_positions)} views)")
    print(f"  Green lines   : hit rays  ({len(hit_segments)} rays → mesh surface)")
    print(f"  Orange lines  : miss rays ({len(miss_segments)} rays → no intersection)")
    print(f"  Yellow spheres: hit points on mesh surface")
    if gt_symmetries:
        print(f"  Blue line     : GT axis" if any(g["type"]=="axis"  for g in gt_symmetries) else "")
        print(f"  Green plane   : GT plane" if any(g["type"]=="plane" for g in gt_symmetries) else "")
    print()
    print("Launching Polyscope — close the window to exit.")

    ps.show()


if __name__ == "__main__":
    main()
