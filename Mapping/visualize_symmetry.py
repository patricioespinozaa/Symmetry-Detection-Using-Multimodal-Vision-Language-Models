#!/usr/bin/env python3
"""
visualize_symmetry.py
---------------------
Interactive 3D visualization of predicted (and optionally ground-truth)
symmetries for a single ShapeNet object using Polyscope.

Shows
-----
  - 3D mesh (gray)
  - Molmo2 3D hit points, one point cloud per n_views group (color-coded)
  - Predicted symmetry axis (red line) or plane (semi-transparent red quad)
  - Ground-truth symmetry (blue axis / green plane), when --show-gt is given

Requires
--------
    pip install polyscope trimesh numpy

Usage
-----
    # Axis symmetry — show all n_views groups, overlay GT
python Mapping/visualize_symmetry.py \
    --object-id <id> \
    --renders-root  ../data/renders \
    --objects-root  ../data/objects \
    --symmetry-type axis_sym \
    --size 224 \
    --lighting flat \
    --show-gt

    # Plane symmetry — show only n_views=14
    python Mapping/visualize_symmetry.py \\
        --object-id <id> \\
        --renders-root  ../data/renders \\
        --objects-root  ../data/objects \\
        --symmetry-type plane_sym \\
        --n-views 14

    # Custom geometry scales
    python Mapping/visualize_symmetry.py \\
        --object-id <id> \\
        --renders-root ../data/renders \\
        --objects-root ../data/objects \\
        --symmetry-type axis_sym \\
        --axis-length 0.8 \\
        --point-radius 0.015
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import polyscope as ps
except ImportError:
    sys.exit("[error] polyscope not installed — run: pip install polyscope")

try:
    import trimesh
except ImportError:
    sys.exit("[error] trimesh not installed — run: pip install trimesh")


# ── Constants ─────────────────────────────────────────────────────────────────

OBJECTS_SUBDIR: dict[str, str] = {
    "axis_sym":  "curated_axis_sym_obj",
    "plane_sym": "curated_plane_sym_obj",
}

# One distinct color per n_views group (RGB in [0, 1])
GROUP_COLORS: dict[str, tuple[float, float, float]] = {
    "1":   (0.20, 0.60, 1.00),   # blue
    "6":   (0.20, 0.85, 0.40),   # green
    "14":  (1.00, 0.65, 0.10),   # orange
    "26":  (0.85, 0.20, 0.85),   # purple
    "42":  (0.10, 0.90, 0.90),   # cyan
    "62":  (1.00, 0.40, 0.40),   # salmon
    "86":  (0.70, 0.50, 0.20),   # brown
    "114": (0.55, 0.55, 0.95),   # lavender
}
FALLBACK_COLOR = (0.70, 0.70, 0.70)

# Predicted symmetry: red
PRED_COLOR = (0.93, 0.10, 0.10)
# Ground-truth: blue (axis) / green (plane)
GT_AXIS_COLOR  = (0.15, 0.40, 0.90)
GT_PLANE_COLOR = (0.10, 0.80, 0.35)


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _translate_mat(tx: float, ty: float, tz: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[:3, 3] = [tx, ty, tz]
    return m


def _rotation_z_mat(theta: float) -> np.ndarray:
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return np.array([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1],
    ], dtype=np.float32)


def build_plane_quad(
    normal: np.ndarray,
    origin: np.ndarray,
    scale: float,
) -> np.ndarray:
    """
    Return (4, 3) corners of a square centred at `origin` lying in the
    plane defined by `normal`.  `scale` is the half-side length.

    The approach mirrors the SymmetryPlane.compute_geometry() method in
    symmetries.py: rotate a canonical YZ-plane quad to face `normal`.
    """
    normal = normal / np.linalg.norm(normal)
    a, b, c = float(normal[0]), float(normal[1]), float(normal[2])
    h = float(np.sqrt(a**2 + c**2))

    base = np.array([
        [0, -scale, -scale],
        [0,  scale, -scale],
        [0,  scale,  scale],
        [0, -scale,  scale],
    ], dtype=np.float32)  # canonical quad in YZ plane (normal = X)

    if h < 1e-7:
        # Normal is nearly Y; rotate 90° around Z
        transform = _translate_mat(*origin) @ _rotation_z_mat(np.pi / 2)
    else:
        Rzinv = np.array([
            [ h, -b, 0, 0],
            [ b,  h, 0, 0],
            [ 0,  0, 1, 0],
            [ 0,  0, 0, 1],
        ], dtype=np.float32)
        Ryinv = np.array([
            [a/h, 0, -c/h, 0],
            [  0, 1,    0, 0],
            [c/h, 0,  a/h, 0],
            [  0, 0,    0, 1],
        ], dtype=np.float32)
        transform = _translate_mat(*origin) @ Ryinv @ Rzinv

    pts_h = np.concatenate([base.T, np.ones((1, 4))], axis=0)  # (4, 4)
    return (transform @ pts_h)[:3].T.astype(np.float64)         # (4, 3)


def plane_triangles() -> np.ndarray:
    """Two triangles covering the canonical quad vertices [0, 1, 2, 3]."""
    return np.array([[0, 1, 3], [3, 1, 2]], dtype=np.int32)


def axis_endpoints(
    direction: np.ndarray,
    origin: np.ndarray,
    half_length: float,
) -> np.ndarray:
    """Return (2, 3) array of the two endpoints of the axis segment."""
    d = np.asarray(direction, dtype=np.float64)
    d /= np.linalg.norm(d)
    o = np.asarray(origin, dtype=np.float64)
    return np.array([o - half_length * d, o + half_length * d])


# ── Ground-truth parser ───────────────────────────────────────────────────────

def load_gt(txt_path: Path) -> list[dict]:
    """
    Parse a ShapeNet symmetry annotation .txt file.

    Returns a list of dicts, each with:
        axis_sym  → {"type": "axis",  "direction": [...], "origin": [...]}
        plane_sym → {"type": "plane", "normal":    [...], "origin": [...]}
    """
    symmetries: list[dict] = []
    with open(txt_path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

    i = 0
    n_sym = int(lines[i]); i += 1

    for _ in range(n_sym):
        tokens = lines[i].split(); i += 1
        kind   = tokens[0]
        vals   = [float(v) for v in tokens[1:]]

        if kind == "axis":
            symmetries.append({
                "type":      "axis",
                "direction": vals[:3],
                "origin":    vals[3:6],
            })
        elif kind == "plane":
            symmetries.append({
                "type":   "plane",
                "normal": vals[:3],
                "origin": vals[3:6],
            })

    return symmetries


# ── Polyscope helpers ─────────────────────────────────────────────────────────

def register_axis(
    name: str,
    direction: np.ndarray,
    origin: np.ndarray,
    half_length: float,
    color: tuple[float, float, float],
    radius: float = 0.008,
) -> None:
    pts   = axis_endpoints(direction, origin, half_length)
    edges = np.array([[0, 1]], dtype=np.int32)
    net   = ps.register_curve_network(name, pts, edges, radius=radius)
    net.set_color(color)


def register_plane(
    name: str,
    normal: np.ndarray,
    origin: np.ndarray,
    scale: float,
    color: tuple[float, float, float],
    transparency: float = 0.45,
) -> None:
    verts = build_plane_quad(normal, origin, scale)
    faces = plane_triangles()
    mesh  = ps.register_surface_mesh(name, verts, faces)
    mesh.set_color(color)
    mesh.set_transparency(transparency)


# ── Mesh loading ──────────────────────────────────────────────────────────────

def load_mesh(obj_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(obj_path), force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(
            [g for g in loaded.geometry.values()
             if isinstance(g, trimesh.Trimesh)]
        )
    return loaded


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize 3D symmetry predictions with Polyscope.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--object-id",     required=True,
                   help="Object ID to visualize (folder name under <symmetry_type>/)")
    p.add_argument("--renders-root",  required=True,
                   help="Root folder of renders (produced by data_render.py)")
    p.add_argument("--objects-root",  required=True,
                   help="Root containing curated_*_obj/ subfolders")
    p.add_argument("--symmetry-type", required=True,
                   choices=["axis_sym", "plane_sym"])

    p.add_argument("--size",     type=int, default=224,
                   help="Image size subfolder to read mapped_points_3d.json from")
    p.add_argument("--lighting", default="flat",
                   choices=["flat", "darker", "brighter"])

    p.add_argument("--n-views",  type=int, default=None,
                   help="Show only this n_views group; omit to show all")
    p.add_argument("--show-gt",  action="store_true",
                   help="Overlay ground-truth symmetry from .txt annotation")

    p.add_argument("--axis-length", type=float, default=None,
                   help="Half-length of axis segment. Default: 0.6 × bbox diagonal")
    p.add_argument("--plane-scale", type=float, default=None,
                   help="Half-side of plane quad.   Default: 0.5 × bbox diagonal")
    p.add_argument("--point-radius", type=float, default=0.012,
                   help="Radius of 3D hit point spheres")
    p.add_argument("--axis-radius",  type=float, default=0.008,
                   help="Radius of axis tube")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    renders_root = Path(args.renders_root)
    objects_root = Path(args.objects_root)
    sym_type     = args.symmetry_type
    object_id    = args.object_id

    # ── File paths ─────────────────────────────────────────────────────────────
    obj_render_dir = renders_root / sym_type / object_id
    render_dir     = obj_render_dir / str(args.size) / args.lighting

    obj_path    = objects_root / OBJECTS_SUBDIR[sym_type] / f"{object_id}.obj"
    txt_path    = objects_root / OBJECTS_SUBDIR[sym_type] / f"{object_id}.txt"
    pred_path   = obj_render_dir / "predicted_symmetry.json"
    mapped_path = render_dir / "mapped_points_3d.json"

    # ── Validation ─────────────────────────────────────────────────────────────
    missing = [(p, lbl) for p, lbl in [
        (obj_path,  ".obj mesh"),
        (pred_path, "predicted_symmetry.json"),
    ] if not p.exists()]
    if missing:
        for p, lbl in missing:
            print(f"[error] {lbl} not found: {p}")
        sys.exit(1)

    # ── Load mesh ──────────────────────────────────────────────────────────────
    print(f"Loading mesh  : {obj_path}")
    mesh      = load_mesh(obj_path)
    vertices  = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.faces,    dtype=np.int32)

    bbox_diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    half_len  = args.axis_length if args.axis_length is not None else 0.6 * bbox_diag
    plane_sc  = args.plane_scale if args.plane_scale is not None else 0.5 * bbox_diag

    # ── Load predictions ───────────────────────────────────────────────────────
    print(f"Loading preds : {pred_path}")
    with open(pred_path, encoding="utf-8") as f:
        pred = json.load(f)

    predictions = pred.get("n_views_predictions", {})

    if args.n_views is not None:
        key = str(args.n_views)
        if key not in predictions:
            avail = list(predictions.keys())
            sys.exit(f"[error] n_views={key} not in predicted_symmetry.json. "
                     f"Available: {avail}")
        predictions = {key: predictions[key]}

    # ── Load 3D hit points ─────────────────────────────────────────────────────
    hit_points: dict[str, np.ndarray] = {}
    if mapped_path.exists():
        print(f"Loading hits  : {mapped_path}")
        with open(mapped_path, encoding="utf-8") as f:
            mapped = json.load(f)

        for key, group in mapped.get("n_views_results", {}).items():
            if args.n_views is not None and key != str(args.n_views):
                continue
            pts = [
                p["point_3d"]
                for p in group.get("points_3d", [])
                if p["hit"] and p["point_3d"] is not None
            ]
            if pts:
                hit_points[key] = np.array(pts, dtype=np.float64)
    else:
        print(f"[warn] mapped_points_3d.json not found at {mapped_path} "
              f"— skipping hit points")

    # ── Load ground truth ──────────────────────────────────────────────────────
    gt_symmetries: list[dict] = []
    if args.show_gt:
        if txt_path.exists():
            gt_symmetries = load_gt(txt_path)
            print(f"GT symmetries : {len(gt_symmetries)} element(s) from {txt_path.name}")
        else:
            print(f"[warn] .txt not found: {txt_path} — skipping GT")

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    print(f"Object        : {object_id}  ({sym_type})")
    print(f"Mesh          : {len(vertices):,} vertices · {len(triangles):,} faces")
    print(f"Bbox diagonal : {bbox_diag:.4f}")
    print(f"Axis half-len : {half_len:.4f}")
    print(f"Plane scale   : {plane_sc:.4f}")
    print(f"Pred groups   : {list(predictions.keys())}")
    print(f"Hit groups    : {list(hit_points.keys())}")
    if args.show_gt:
        print(f"GT elements   : {len(gt_symmetries)}")

    # ── Polyscope setup ────────────────────────────────────────────────────────
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_background_color((1.0, 1.0, 1.0))
    ps.look_at((2.0, 2.0, 2.0), (0.0, 0.0, 0.0))
    ps.set_ground_plane_mode("shadow_only")

    # ── Mesh ───────────────────────────────────────────────────────────────────
    ps.register_surface_mesh("mesh", vertices, triangles).set_color((0.75, 0.75, 0.75))

    # ── Hit points (one cloud per n_views group) ───────────────────────────────
    for key, pts in hit_points.items():
        color = GROUP_COLORS.get(key, FALLBACK_COLOR)
        cloud = ps.register_point_cloud(f"hits_nv{key}", pts,
                                        radius=args.point_radius)
        cloud.set_color(color)
        print(f"  hits  nv={key:>3}: {len(pts)} point(s)  color={color}")

    # ── Predicted symmetry ─────────────────────────────────────────────────────
    for key, sym in predictions.items():
        origin = np.array(sym["origin"], dtype=np.float64)
        label  = f"pred_nv{key}"

        if sym_type == "axis_sym":
            direction = np.array(sym["direction"], dtype=np.float64)
            register_axis(label, direction, origin,
                          half_len, PRED_COLOR, args.axis_radius)
            print(f"  pred  nv={key:>3}: axis  dir={np.round(direction,3).tolist()}")
        else:
            normal = np.array(sym["normal"], dtype=np.float64)
            register_plane(label, normal, origin, plane_sc, PRED_COLOR)
            print(f"  pred  nv={key:>3}: plane n={np.round(normal,3).tolist()}")

    # ── Ground-truth symmetry ──────────────────────────────────────────────────
    for i, gt in enumerate(gt_symmetries):
        origin = np.array(gt["origin"], dtype=np.float64)

        if gt["type"] == "axis":
            direction = np.array(gt["direction"], dtype=np.float64)
            register_axis(f"gt_axis_{i}", direction, origin,
                          half_len, GT_AXIS_COLOR, args.axis_radius * 0.8)
        elif gt["type"] == "plane":
            normal = np.array(gt["normal"], dtype=np.float64)
            register_plane(f"gt_plane_{i}", normal, origin,
                           plane_sc, GT_PLANE_COLOR, transparency=0.30)

    # ── Launch ─────────────────────────────────────────────────────────────────
    print("\nLaunching Polyscope — close the window to exit.")
    print()
    print("Legend")
    print("  Gray mesh          : 3D object")
    for key, color in GROUP_COLORS.items():
        if key in hit_points:
            print(f"  Color {color} : Molmo2 hit points (n_views={key})")
    print(f"  Red               : Predicted {sym_type.replace('_', ' ')}")
    if args.show_gt:
        print(f"  Blue              : GT axis" if sym_type == "axis_sym"
              else f"  Green             : GT plane")
    print()

    ps.show()


if __name__ == "__main__":
    main()
