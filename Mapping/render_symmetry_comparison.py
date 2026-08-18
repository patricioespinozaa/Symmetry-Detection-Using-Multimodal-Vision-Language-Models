#!/usr/bin/env python3
"""
render_symmetry_comparison.py
------------------------------
Renders the object's own 3D mesh twice with Polyscope -- once with the
ground-truth axis/plane overlaid, once with the predicted one -- takes a
screenshot of each, and composites them into a single print-ready "versus"
figure saved at high DPI.

Unlike export_symmetry_overlay.py, this does NOT need any pre-rendered
photograph (molmo_multiview.json / PNG renders): only the object mesh
(<object_id>.obj/.txt) and predicted_symmetry[_EXP].json are required.

Colors match visualize_rays.py, so figures from either tool read consistently:
    GT axis         : blue    (0.15, 0.40, 0.90)
    GT plane        : green   (0.10, 0.80, 0.35)
    Predicted axis  : red     (0.90, 0.15, 0.15)
    Predicted plane : magenta (0.85, 0.10, 0.85)

Usage
-----
python Mapping/render_symmetry_comparison.py \\
    --object-id 1edaab172fcc23ab5238dc5d98b43ffd \\
    --json-root results/plane_v04_1/viz_samples/good \\
    --objects-root ../data/objects \\
    --symmetry-type plane_sym \\
    --n-views 1 \\
    --experiment-id plane_v04_1 \\
    --pred-method svd \\
    --out results/plane_v04_1/render_good.png \\
    --dpi 800

Requires
--------
    polyscope, trimesh, matplotlib, pillow, numpy
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
    sys.exit("[error] polyscope not installed -- pip install polyscope")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_common.naming import exp_filename
from pipeline_common.datasets import OBJECTS_SUBDIR, load_mesh
from pipeline_common.clustering import cluster_points


# ── Constants ─────────────────────────────────────────────────────────────────

PRED_JSON_BASE   = "predicted_symmetry.json"
MAPPED_JSON_BASE = "mapped_points_3d.json"
METHODS          = ["svd", "ransac_svd", "svd_sde", "ransac_svd_sde"]
POINT_MODES      = ("independent", "midpoint", "all")

COLOR_MESH       = (0.78, 0.78, 0.78)
COLOR_GT_AXIS    = (0.15, 0.40, 0.90)   # blue
COLOR_GT_PLANE   = (0.10, 0.80, 0.35)   # green
COLOR_PRED_AXIS  = (0.90, 0.15, 0.15)   # red
COLOR_PRED_PLANE = (0.85, 0.10, 0.85)   # magenta
COLOR_HIT_PT     = (1.00, 0.95, 0.10)   # yellow -- Molmo2 hit points


def collect_hit_points(mapped_json: dict, nv_key: str,
                       point_mode: str = "independent") -> np.ndarray | None:
    """
    Extract 3D hit points from mapped_points_3d.json for one n_views group.
    Mirrors visualize_rays.py's _collect_hit_points / estimate_symmetry's
    collect_hit_points -- keep in sync.
    """
    group = mapped_json.get("n_views_results", {}).get(nv_key)
    if group is None:
        return None
    raw = group.get("points_3d", [])

    if point_mode == "all":
        pts = [np.array(p["point_3d"], dtype=np.float64)
               for p in raw if p["hit"] and p["point_3d"] is not None]
        return np.array(pts, dtype=np.float64) if len(pts) >= 1 else None

    by_img: dict[int, dict[int, np.ndarray]] = {}
    for p in raw:
        if not p["hit"] or p["point_3d"] is None:
            continue
        by_img.setdefault(p["img_idx"], {})[p["obj_id"]] = np.array(
            p["point_3d"], dtype=np.float64
        )

    if point_mode == "midpoint":
        pts = [(d[1] + d[2]) / 2.0 for d in by_img.values() if 1 in d and 2 in d]
    else:
        pts = []
        for d in by_img.values():
            if 1 in d and 2 in d:
                pts.extend([d[1], d[2]])

    return np.array(pts, dtype=np.float64) if len(pts) >= 1 else None


# ── Ground-truth parser (mirrors visualize_rays.load_gt) ─────────────────────

def load_gt(txt_path: Path) -> list[dict]:
    symmetries = []
    lines = [l.strip() for l in txt_path.read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    i = 0
    n = int(lines[i]); i += 1
    for _ in range(n):
        tokens = lines[i].split(); i += 1
        vals = [float(v) for v in tokens[1:]]
        if tokens[0] == "axis":
            symmetries.append({"type": "axis", "direction": vals[:3], "origin": vals[3:6]})
        elif tokens[0] == "plane":
            symmetries.append({"type": "plane", "normal": vals[:3], "origin": vals[3:6]})
    return symmetries


# ── Geometry helpers (mirror visualize_rays.py) ───────────────────────────────

def axis_endpoints(direction, origin, half_length):
    d = np.asarray(direction, dtype=np.float64); d /= np.linalg.norm(d)
    o = np.asarray(origin,    dtype=np.float64)
    return np.array([o - half_length * d, o + half_length * d])


def plane_quad(normal, origin, scale):
    n = np.asarray(normal, dtype=np.float64)
    n /= np.linalg.norm(n)
    a, b, c = float(n[0]), float(n[1]), float(n[2])
    h = float(np.sqrt(a**2 + c**2))
    base = np.array([[0,-scale,-scale],[0,scale,-scale],
                      [0,scale,scale],[0,-scale,scale]], dtype=np.float32)

    def translate(tx, ty, tz):
        m = np.eye(4, dtype=np.float32); m[:3, 3] = [tx, ty, tz]; return m

    def rot_z(theta):
        cth, sth = float(np.cos(theta)), float(np.sin(theta))
        return np.array([[cth,-sth,0,0],[sth,cth,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.float32)

    if h < 1e-7:
        T = translate(*origin) @ rot_z(np.pi / 2)
    else:
        Rzinv = np.array([[h,-b,0,0],[b,h,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
        Ryinv = np.array([[a/h,0,-c/h,0],[0,1,0,0],[c/h,0,a/h,0],[0,0,0,1]], dtype=np.float32)
        T = translate(*origin) @ Ryinv @ Rzinv

    pts_h = np.concatenate([base.T, np.ones((1, 4))], axis=0)
    return (T @ pts_h)[:3].T.astype(np.float64)


def plane_faces() -> np.ndarray:
    return np.array([[0, 1, 3], [3, 1, 2]], dtype=np.int32)


def angular_error_deg(a, b) -> float:
    a = np.asarray(a, dtype=np.float64); a /= np.linalg.norm(a)
    b = np.asarray(b, dtype=np.float64); b /= np.linalg.norm(b)
    cos_t = float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_t)))


# ── Polyscope rendering ────────────────────────────────────────────────────────

def render_panel(vertices, triangles, kind, element, color, out_png, window_size, cam_dist,
                  hit_points=None, point_radius=0.02):
    """Registers mesh + one symmetry element (+ optional hit points), screenshots, then clears."""
    ps.remove_all_structures()
    ps.register_surface_mesh("mesh", vertices, triangles).set_color(COLOR_MESH)

    if kind == "axis":
        pts  = element
        net  = ps.register_curve_network("elem", pts, np.array([[0, 1]], dtype=np.int32),
                                         radius=0.012)
        net.set_color(color)
    else:
        verts = element
        m = ps.register_surface_mesh("elem", verts, plane_faces())
        m.set_color(color)
        m.set_transparency(0.5)

    if hit_points is not None and len(hit_points) > 0:
        ps.register_point_cloud("molmo_points", hit_points,
                                radius=point_radius).set_color(COLOR_HIT_PT)

    ps.look_at((cam_dist, cam_dist, cam_dist), (0.0, 0.0, 0.0))
    ps.screenshot(str(out_png))


def _content_bbox(img: Image.Image, bg=(255, 255, 255)) -> tuple[int, int, int, int] | None:
    """Bounding box (top, bottom, left, right) of non-background pixels, or None if blank."""
    arr = np.asarray(img.convert("RGB"))
    mask = np.any(np.abs(arr.astype(int) - np.array(bg)) > 8, axis=-1)
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def crop_pair_to_shared_content(img_a: Image.Image, img_b: Image.Image,
                                 bg=(255, 255, 255), pad=20
                                 ) -> tuple[Image.Image, Image.Image]:
    """
    Crop both images to the SAME box (the union of each one's content bbox).
    Both panels come from the same camera/window size, so this keeps them at
    identical dimensions -- otherwise GT and predicted planes/axes with
    different on-screen extents crop to different aspect ratios and the
    panel titles end up at different heights.
    """
    box_a = _content_bbox(img_a, bg)
    box_b = _content_bbox(img_b, bg)
    boxes = [b for b in (box_a, box_b) if b is not None]
    if not boxes:
        return img_a, img_b

    top    = min(b[0] for b in boxes)
    bottom = max(b[1] for b in boxes)
    left   = min(b[2] for b in boxes)
    right  = max(b[3] for b in boxes)

    w, h = img_a.size
    top    = max(top - pad, 0)
    left   = max(left - pad, 0)
    bottom = min(bottom + pad, h)
    right  = min(right + pad, w)

    crop_box = (left, top, right, bottom)
    return img_a.crop(crop_box), img_b.crop(crop_box)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render GT-vs-predicted symmetry comparison figure for one object.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--object-id",     required=True)
    p.add_argument("--json-root",     required=True,
                   help="Root with predicted_symmetry[_EXP].json (e.g. results/<exp>/viz_samples/good).")
    p.add_argument("--objects-root",  required=True)
    p.add_argument("--symmetry-type", required=True, choices=["axis_sym", "plane_sym"])
    p.add_argument("--n-views",       type=int, required=True)
    p.add_argument("--size",     type=int, default=224,
                   help="Only used to locate mapped_points_3d[_EXP].json for --show-points.")
    p.add_argument("--lighting", default="flat", choices=["flat", "darker", "brighter"],
                   help="Only used to locate mapped_points_3d[_EXP].json for --show-points.")
    p.add_argument("--experiment-id", default=None)
    p.add_argument("--pred-method",   default="svd", choices=METHODS)
    p.add_argument("--out", required=True, help="Output image path (.png/.pdf).")
    p.add_argument("--dpi", type=int, default=800)
    p.add_argument("--window-size", type=int, default=1200,
                   help="Polyscope render resolution (square, in px) per panel.")
    p.add_argument("--cam-dist", type=float, default=2.2,
                   help="Camera distance from origin along (1,1,1), in object-space units.")
    p.add_argument("--show-points", action="store_true",
                   help="Overlay the raw 3D hit points Molmo2's 2D points back-projected to "
                        "(yellow spheres), on the Predicted panel only.")
    p.add_argument("--point-mode", default="independent", choices=POINT_MODES,
                   help="Point collection mode for --show-points. Must match the mode used by "
                        "estimate_symmetry.py for this experiment ('all' = Flow C).")
    p.add_argument("--point-radius", type=float, default=0.02,
                   help="Sphere radius for --show-points.")
    p.add_argument("--cluster", action="store_true",
                   help="Apply the same greedy centroid clustering used before RANSAC+SVD "
                        "(pipeline_common.clustering.cluster_points, threshold=0.05*bbox_diag "
                        "of the raw hit points) before overlaying --show-points.")
    p.add_argument("--no-title", action="store_true",
                   help="Skip the 'Ground truth' / 'Predicted (error = ...)' matplotlib titles, "
                        "for embedding the panels in a document that sets its own labels/font.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    json_root    = Path(args.json_root)
    objects_root = Path(args.objects_root)
    sym_type     = args.symmetry_type
    object_id    = args.object_id
    nv_key       = str(args.n_views)

    pred_name   = exp_filename(PRED_JSON_BASE,   args.experiment_id)
    mapped_name = exp_filename(MAPPED_JSON_BASE, args.experiment_id)
    pred_path   = json_root / sym_type / object_id / pred_name
    mapped_path = json_root / sym_type / object_id / str(args.size) / args.lighting / mapped_name
    txt_path    = objects_root / OBJECTS_SUBDIR[sym_type] / f"{object_id}.txt"
    obj_path    = objects_root / OBJECTS_SUBDIR[sym_type] / f"{object_id}.obj"

    required = [(pred_path, "predicted_symmetry"), (txt_path, "GT .txt"), (obj_path, ".obj")]
    if args.show_points:
        required.append((mapped_path, "mapped_points_3d"))
    for p, lbl in required:
        if not p.exists():
            sys.exit(f"[error] {lbl} not found: {p}")

    mesh      = load_mesh(obj_path)
    vertices  = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.faces,    dtype=np.int32)
    bbox_diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    half_len  = 0.6 * bbox_diag
    plane_sc  = 0.5 * bbox_diag

    gt_kind = "axis" if sym_type == "axis_sym" else "plane"
    gt_symmetries = [g for g in load_gt(txt_path) if g["type"] == gt_kind]
    if not gt_symmetries:
        sys.exit(f"[error] no GT {gt_kind} entry found in {txt_path}")

    with open(pred_path, encoding="utf-8") as f:
        pred_data = json.load(f)
    nv_preds = pred_data["n_views_predictions"].get(nv_key, {})
    pred = nv_preds.get(args.pred_method)
    if pred is None:
        sys.exit(f"[error] --pred-method {args.pred_method} not available for n_views={nv_key}. "
                  f"Available: {list(nv_preds.keys())}")

    # Objects can have more than one GT axis/plane (e.g. planar objects with
    # 2-3 valid reflection planes). Match the prediction against whichever GT
    # candidate it's actually closest to, mirroring evaluate.py's min-over-GT
    # angular error -- otherwise we might score against an unrelated GT
    # element and report a large error that doesn't reflect the real fit.
    vec_key = "direction" if sym_type == "axis_sym" else "normal"
    gt, err_deg = min(
        ((g, angular_error_deg(g[vec_key], pred[vec_key])) for g in gt_symmetries),
        key=lambda pair: pair[1],
    )
    if len(gt_symmetries) > 1:
        print(f"[info] object has {len(gt_symmetries)} GT {gt_kind}s; "
              f"showing the closest match to the prediction ({err_deg:.2f} deg).")

    if sym_type == "axis_sym":
        gt_elem   = axis_endpoints(gt["direction"],   gt["origin"],   half_len)
        pred_elem = axis_endpoints(pred["direction"], pred["origin"], half_len)
        color_gt, color_pred = COLOR_GT_AXIS, COLOR_PRED_AXIS
    else:
        gt_elem   = plane_quad(gt["normal"],   gt["origin"],   plane_sc)
        pred_elem = plane_quad(pred["normal"], pred["origin"], plane_sc)
        color_gt, color_pred = COLOR_GT_PLANE, COLOR_PRED_PLANE

    hit_points = None
    if args.show_points:
        with open(mapped_path, encoding="utf-8") as f:
            mapped_data = json.load(f)
        hit_points = collect_hit_points(mapped_data, nv_key, args.point_mode)
        if hit_points is None:
            print(f"[warn] --show-points: no hit points found for n_views={nv_key} "
                  f"(point_mode={args.point_mode}); skipping.")
        else:
            print(f"[info] {len(hit_points)} Molmo2 hit points overlaid on the Predicted panel.")
            if args.cluster:
                pts_bbox_diag = float(np.linalg.norm(
                    hit_points.max(axis=0) - hit_points.min(axis=0)))
                hit_points = cluster_points(hit_points, pts_bbox_diag, 0.05)
                print(f"[info] clustered down to {len(hit_points)} centroids "
                      f"(threshold=0.05*{pts_bbox_diag:.4f}).")

    # ── Polyscope: one session, two screenshots ───────────────────────────────
    ps.init()
    ps.set_window_size(args.window_size, args.window_size)
    ps.set_up_dir("y_up")
    ps.set_background_color((1.0, 1.0, 1.0))
    ps.set_ground_plane_mode("none")
    ps.set_screenshot_extension(".png")

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_gt   = out_dir / f".{object_id}_gt_tmp.png"
    tmp_pred = out_dir / f".{object_id}_pred_tmp.png"

    render_panel(vertices, triangles, gt_kind, gt_elem,   color_gt,
                 tmp_gt.resolve(),   args.window_size, args.cam_dist)
    render_panel(vertices, triangles, gt_kind, pred_elem, color_pred,
                 tmp_pred.resolve(), args.window_size, args.cam_dist,
                 hit_points=hit_points, point_radius=args.point_radius)

    img_gt, img_pred = crop_pair_to_shared_content(Image.open(tmp_gt), Image.open(tmp_pred))
    tmp_gt.unlink(missing_ok=True)
    tmp_pred.unlink(missing_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.4))
    for ax, img, title in [
        (axes[0], img_gt,   "Ground truth"),
        (axes[1], img_pred, f"Predicted  (error = {err_deg:.1f}°)"),
    ]:
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if not args.no_title:
            ax.set_title(title, fontsize=20, fontweight="bold", y=1.02)

    fig.tight_layout()

    out_path = Path(args.out)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved       : {out_path}  ({args.dpi} dpi)")
    print(f"Angular err : {err_deg:.2f} deg  (GT vs. predicted {args.pred_method})")


if __name__ == "__main__":
    main()
