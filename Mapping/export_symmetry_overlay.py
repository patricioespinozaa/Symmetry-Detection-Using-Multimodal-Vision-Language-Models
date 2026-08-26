#!/usr/bin/env python3
"""
export_symmetry_overlay.py
---------------------------
Generates a print-ready "versus" figure for a single object: its own rendered
photograph with the ground-truth axis/plane overlaid (left panel) next to the
same photograph with the predicted axis/plane overlaid (right panel).

Colors match the Polyscope debug viewer (Mapping/visualize_rays.py), so
figures from either tool read consistently:
    GT axis         : blue    (0.15, 0.40, 0.90)
    GT plane        : green   (0.10, 0.80, 0.35)
    Predicted axis  : red     (0.90, 0.15, 0.15)
    Predicted plane : magenta (0.85, 0.10, 0.85)

The view (which of the N rendered photos to use) is auto-selected as the one
Molmo2 returned the most points for, unless --view-index is given explicitly.

Usage
-----
python Mapping/export_symmetry_overlay.py \\
    --object-id 1edaab172fcc23ab5238dc5d98b43ffd \\
    --json-root results/plane_v04_1/viz_samples/good \\
    --photos-root ../data/renders \\
    --objects-root ../data/objects \\
    --symmetry-type plane_sym \\
    --size 224 --lighting flat \\
    --n-views 1 \\
    --experiment-id plane_v04_1 \\
    --pred-method svd \\
    --out results/plane_v04_1/overlay_good.png \\
    --dpi 800

Note on --photos-root
----------------------
export_viz_samples.py only copies the pipeline JSONs into
results/<exp>/viz_samples/{good,bad}/, not the rendered PNGs (too heavy).
--photos-root must point at wherever the actual <object_id>/<size>/<lighting>/
*.png files live (typically the original ../data/renders tree). If those
renders aren't on this machine, copy the handful of PNGs you need over from
wherever the multi-view rendering was run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_common.naming import exp_filename
from pipeline_common.datasets import OBJECTS_SUBDIR, load_mesh_vertices
from pipeline_common.camera import project_point
from pipeline_common.viz_colors import (
    COLOR_GT_AXIS,
    COLOR_GT_PLANE,
    COLOR_PRED_AXIS,
    COLOR_PRED_PLANE,
)


# ── Constants ─────────────────────────────────────────────────────────────────

MOLMO_JSON_BASE = "molmo_multiview.json"
PRED_JSON_BASE  = "predicted_symmetry.json"
FOV_DEG         = 60.0
METHODS         = ["svd", "ransac_svd", "svd_sde", "ransac_svd_sde"]


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


# ── Geometry helpers (mirror visualize_rays.py's plane_quad / axis_endpoints) ─

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


def angular_error_deg(a, b) -> float:
    a = np.asarray(a, dtype=np.float64); a /= np.linalg.norm(a)
    b = np.asarray(b, dtype=np.float64); b /= np.linalg.norm(b)
    cos_t = float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_t)))


def choose_view(points_by_image: dict, images_sent: list) -> int:
    """Pick the image index Molmo2 returned the most points for."""
    best_idx, best_n = 0, -1
    for img_idx_str, pts in points_by_image.items():
        img_idx = int(img_idx_str)
        if img_idx >= len(images_sent):
            continue
        if len(pts) > best_n:
            best_n, best_idx = len(pts), img_idx
    return best_idx


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_axis(ax, pts2d, color, label):
    (x0, y0), (x1, y1) = pts2d
    ax.add_line(Line2D([x0, x1], [y0, y1], color=color, linewidth=2.5, label=label))


def draw_plane(ax, pts2d, color, label):
    poly = Polygon(pts2d, closed=True, facecolor=color, edgecolor=color,
                    alpha=0.35, linewidth=2.0, label=label)
    ax.add_patch(poly)


def render_panel(ax, photo, kind, pts2d, color, label, image_size, title):
    ax.imshow(photo, extent=[0, image_size, image_size, 0])
    if kind == "axis":
        draw_axis(ax, pts2d, color, label)
    else:
        draw_plane(ax, pts2d, color, label)
    ax.set_xlim(0, image_size)
    ax.set_ylim(image_size, 0)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export a GT-vs-predicted symmetry overlay figure on the object's own rendered photo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--object-id",     required=True)
    p.add_argument("--json-root",     required=True,
                   help="Root with molmo_multiview[_EXP].json / predicted_symmetry[_EXP].json "
                        "(e.g. results/<exp>/viz_samples/good).")
    p.add_argument("--photos-root",   required=True,
                   help="Root with the rendered PNGs (e.g. ../data/renders). See module docstring.")
    p.add_argument("--objects-root",  required=True)
    p.add_argument("--symmetry-type", required=True, choices=["axis_sym", "plane_sym"])
    p.add_argument("--size",     type=int, default=224)
    p.add_argument("--lighting", default="flat", choices=["flat", "darker", "brighter"])
    p.add_argument("--fov",      type=float, default=FOV_DEG)
    p.add_argument("--n-views",  type=int, required=True)
    p.add_argument("--experiment-id", default=None)
    p.add_argument("--pred-method",   default="svd", choices=METHODS)
    p.add_argument("--view-index", type=int, default=None,
                   help="Force a specific rendered view. Default: auto-pick the view "
                        "Molmo2 returned the most points for.")
    p.add_argument("--out", required=True, help="Output image path (.png/.pdf).")
    p.add_argument("--dpi", type=int, default=800)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    json_root    = Path(args.json_root)
    photos_root  = Path(args.photos_root)
    objects_root = Path(args.objects_root)
    sym_type     = args.symmetry_type
    object_id    = args.object_id
    nv_key       = str(args.n_views)

    json_dir   = json_root / sym_type / object_id / str(args.size) / args.lighting
    molmo_name = exp_filename(MOLMO_JSON_BASE, args.experiment_id)
    pred_name  = exp_filename(PRED_JSON_BASE,  args.experiment_id)
    molmo_path = json_dir  / molmo_name
    pred_path  = json_root / sym_type / object_id / pred_name
    txt_path   = objects_root / OBJECTS_SUBDIR[sym_type] / f"{object_id}.txt"
    obj_path   = objects_root / OBJECTS_SUBDIR[sym_type] / f"{object_id}.obj"

    for p, lbl in [(molmo_path, "molmo_multiview"), (pred_path, "predicted_symmetry"),
                   (txt_path, "GT .txt"), (obj_path, ".obj")]:
        if not p.exists():
            sys.exit(f"[error] {lbl} not found: {p}")

    with open(molmo_path, encoding="utf-8") as f:
        molmo_data = json.load(f)
    if nv_key not in molmo_data:
        sys.exit(f"[error] n_views={nv_key} not in {molmo_name}. Available: {list(molmo_data.keys())}")

    group           = molmo_data[nv_key]
    images_sent     = group["images_sent"]
    points_by_image = group["points_by_image"]

    view_idx = args.view_index if args.view_index is not None \
        else choose_view(points_by_image, images_sent)
    if not (0 <= view_idx < len(images_sent)):
        sys.exit(f"[error] view-index {view_idx} out of range (0..{len(images_sent) - 1})")

    cam = images_sent[view_idx]
    R, T, filename = cam["R"], cam["T"], cam["filename"]

    photo_path = photos_root / sym_type / object_id / str(args.size) / args.lighting / filename
    if not photo_path.exists():
        sys.exit(f"[error] photo not found: {photo_path}\n"
                 f"    Rendered PNGs may live on a different machine than the exported JSONs "
                 f"-- copy {filename} over, or point --photos-root at the right renders folder.")
    photo = Image.open(photo_path).convert("RGB")

    vertices  = load_mesh_vertices(obj_path)
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
    # 2-3 valid reflection planes). Match against whichever GT candidate the
    # prediction is actually closest to, mirroring evaluate.py's min-over-GT
    # angular error.
    vec_key = "direction" if sym_type == "axis_sym" else "normal"
    gt, err_deg = min(
        ((g, angular_error_deg(g[vec_key], pred[vec_key])) for g in gt_symmetries),
        key=lambda pair: pair[1],
    )
    if len(gt_symmetries) > 1:
        print(f"[info] object has {len(gt_symmetries)} GT {gt_kind}s; "
              f"showing the closest match to the prediction ({err_deg:.2f} deg).")

    if sym_type == "axis_sym":
        gt_3d   = axis_endpoints(gt["direction"],   gt["origin"],   half_len)
        pred_3d = axis_endpoints(pred["direction"], pred["origin"], half_len)
    else:
        gt_3d   = plane_quad(gt["normal"],   gt["origin"],   plane_sc)
        pred_3d = plane_quad(pred["normal"], pred["origin"], plane_sc)

    def project_all(pts3d):
        proj, behind_any = [], False
        for pt in pts3d:
            px, py, behind = project_point(pt, R, T, args.fov, args.size)
            behind_any = behind_any or behind
            proj.append((px, py))
        return proj, behind_any

    gt_2d,   gt_behind   = project_all(gt_3d)
    pred_2d, pred_behind = project_all(pred_3d)
    if gt_behind or pred_behind:
        print("[warn] part of the GT/predicted element projects behind the camera in this view; "
              "try --view-index to pick a different one.")

    color_gt   = COLOR_GT_AXIS   if gt_kind == "axis" else COLOR_GT_PLANE
    color_pred = COLOR_PRED_AXIS if gt_kind == "axis" else COLOR_PRED_PLANE

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
    render_panel(axes[0], photo, gt_kind, gt_2d,   color_gt,   f"GT {gt_kind}",
                 args.size, title="Ground truth")
    render_panel(axes[1], photo, gt_kind, pred_2d, color_pred, f"Pred. {gt_kind} ({args.pred_method})",
                 args.size, title=f"Predicted  (error = {err_deg:.1f}°)")
    fig.suptitle(f"{object_id}  ·  {sym_type}  ·  n_views={nv_key}  ·  view #{view_idx}",
                 fontsize=9, y=1.02)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved       : {out_path}  ({args.dpi} dpi)")
    print(f"View used   : img_idx={view_idx}  ({filename})")
    print(f"Angular err : {err_deg:.2f} deg  (GT vs. predicted {args.pred_method})")


if __name__ == "__main__":
    main()
