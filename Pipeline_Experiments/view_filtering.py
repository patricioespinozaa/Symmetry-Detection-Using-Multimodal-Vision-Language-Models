# Generalizes Mapping/estimate_symmetry_no_mesh.py::filter_points_on_object
# (which only implements one policy -- drop individual off-object points,
# invalidate the view if too few survive) into the 3 named view-discard
# policies from Experiments/sandbox_pipeline_server_final.ipynb section 14,
# plus a "none" no-op. See docs/diagnostico_conditioning_axis.md S7 for why
# this matters: points off the rendered silhouette don't explain
# angular_error on their own, but correlate with translation_error outliers.
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

BACKGROUND_THRESH = 250   # RGB channel > this = flat white background (PyTorch3D HardFlatShader default)

FILTER_POLICIES = ("none", "per_point", "any", "both")


def _is_on_object(pixel: np.ndarray) -> bool:
    """ True if a rendered RGB pixel belongs to the object (not the flat white background). """
    return not bool(np.all(pixel[:3] > BACKGROUND_THRESH))


def _molmo_xy_to_pixel(x: float, y: float, img_w: int, img_h: int) -> tuple[int, int]:
    """ Convert a Molmo2 0-1000 coordinate to a pixel index in a loaded render.

    Args:
        * x: Molmo2 x coordinate, 0-1000
        * y: Molmo2 y coordinate, 0-1000
        * img_w: loaded render width in pixels
        * img_h: loaded render height in pixels

    Returns:
        * tuple[int, int]: (pixel_x, pixel_y), clamped to the image bounds

    """
    px = int(round((x / 1000.0) * img_w))
    py = int(round((y / 1000.0) * img_h))
    return min(max(px, 0), img_w - 1), min(max(py, 0), img_h - 1)


def _load_render(render_dir: Path, filename: str, image_cache: dict) -> np.ndarray | None:
    """ Load (and cache) one render as an RGB array, or None if the PNG is missing. """
    if filename not in image_cache:
        img_path = render_dir / filename
        image_cache[filename] = np.array(Image.open(img_path).convert("RGB")) if img_path.exists() else None
    return image_cache[filename]


def filter_points_on_object(
    points_by_image: dict, images_sent: list[dict], render_dir: Path,
    policy: str = "per_point", min_points: int = 2, image_cache: dict | None = None,
) -> dict:
    """ Apply one of 4 off-object view-discard policies before triangulation.

    Args:
        * points_by_image: Molmo2 points, one list per view index (string keys)
        * images_sent: camera pose entries (needs "filename"), one per view,
          aligned by index
        * render_dir: directory holding the rendered PNGs for this
          object/size/lighting config
        * policy: "none" (no filtering, return unchanged); "per_point" (drop
          individual off-object points, invalidate the view if fewer than
          min_points survive -- today's production behavior); "any"
          (invalidate the whole view, keeping none of its points, if ANY
          point is off-object); "both" (invalidate the whole view only if
          ALL of its points are off-object, otherwise keep every point
          unmodified, including the off-object one)
        * min_points: only used by "per_point" -- minimum on-object points a
          view must keep to stay valid
        * image_cache: optional dict shared by the caller across n_views
          groups of the same object, so the same PNG is decoded once

    Returns:
        * dict: filtered copy of points_by_image (input is not mutated)

    """
    if policy not in FILTER_POLICIES:
        raise ValueError(f"Unknown filter policy {policy!r}. Available: {FILTER_POLICIES}")
    if policy == "none":
        return points_by_image

    if image_cache is None:
        image_cache = {}

    filtered: dict = {}
    for img_idx_str, pts in points_by_image.items():
        if not pts:
            filtered[img_idx_str] = pts
            continue

        cam = images_sent[int(img_idx_str)]
        img = _load_render(render_dir, cam["filename"], image_cache)
        if img is None:
            filtered[img_idx_str] = pts   # can't classify without the PNG -- leave untouched
            continue

        img_h, img_w = img.shape[0], img.shape[1]
        on_object = []
        for p in pts:
            px, py = _molmo_xy_to_pixel(p["x"], p["y"], img_w, img_h)
            on_object.append(_is_on_object(img[py, px]))

        if policy == "per_point":
            kept = [p for p, ok in zip(pts, on_object) if ok]
            filtered[img_idx_str] = kept if len(kept) >= min_points else []
        elif policy == "any":
            filtered[img_idx_str] = pts if all(on_object) else []
        else:   # "both"
            filtered[img_idx_str] = [] if not any(on_object) else pts

    return filtered
