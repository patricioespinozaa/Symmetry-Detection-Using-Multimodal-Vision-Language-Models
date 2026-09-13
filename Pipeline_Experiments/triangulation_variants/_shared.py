from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np

from pipeline_common.triangulation import (
    get_point_by_obj_id,
    interpretation_plane_normal,
    ray_dir_for_point,
    view_forward_direction,
    widest_pair,
)

PairSelector = Callable[[list[dict]], "tuple[dict, dict] | None"]
WeightFn = Callable[[dict, dict], float]


# ── Point-pair selection strategies ────────────────────────────────────────

def role_pair_selector(pts: list[dict]) -> tuple[dict, dict] | None:
    """ Baseline plane-side pair selection: the fixed obj_id 1/2 bilateral roles.

    Args:
        * pts: points returned for one view

    Returns:
        * tuple[dict, dict] | None: (obj_id 1, obj_id 2), or None if either is missing

    """
    p1 = get_point_by_obj_id(pts, 1)
    p2 = get_point_by_obj_id(pts, 2)
    if p1 is None or p2 is None:
        return None
    return p1, p2


def ransac_line_2d(
    pts: list[dict], inlier_thresh: float = 20.0, n_iters: int = 200, seed: int = 42,
) -> tuple[dict, dict] | None:
    """ EXP-C (Recker et al., WACV 2013): RANSAC-fit a 2D line through the points
    a view returned and pick the pair of INLIERS farthest apart, instead of
    trusting every returned point equally.

    With exactly 2 points (every prompt in production today) this is a
    deterministic pass-through, identical to widest_pair/role_pair_selector --
    it only changes behavior for >=3-point prompts (the "6pts" family), where
    it discards points that don't lie near the consensus line before choosing
    the widest surviving pair.

    Args:
        * pts: points returned for one view (ignores obj_id -- generalizes
          both the axis "widest pair" and plane "role pair" baselines)
        * inlier_thresh: max perpendicular distance (Molmo 0-1000 px units)
          from the candidate line to count as an inlier
        * n_iters: number of random 2-point line hypotheses to try
        * seed: RNG seed, fixed for reproducibility across runs

    Returns:
        * tuple[dict, dict] | None: the widest inlier pair, or None if fewer
          than 2 points are available

    """
    if len(pts) < 2:
        return None
    if len(pts) == 2:
        return pts[0], pts[1]

    rng = np.random.default_rng(seed)
    xy = np.array([[p["x"], p["y"]] for p in pts], dtype=np.float64)
    idx_pairs = list(combinations(range(len(pts)), 2))

    best_inliers, best_count = None, -1
    for _ in range(n_iters):
        i, j = idx_pairs[rng.integers(0, len(idx_pairs))]
        line_vec = xy[j] - xy[i]
        norm = np.linalg.norm(line_vec)
        if norm < 1e-6:
            continue
        line_dir = line_vec / norm
        normal = np.array([-line_dir[1], line_dir[0]])
        dist = np.abs((xy - xy[i]) @ normal)
        inliers = np.where(dist < inlier_thresh)[0]
        if len(inliers) > best_count:
            best_count, best_inliers = len(inliers), inliers

    if best_inliers is None or len(best_inliers) < 2:
        return widest_pair(pts)

    inlier_pts = [pts[k] for k in best_inliers]
    return widest_pair(inlier_pts)


# ── Confidence weight functions ────────────────────────────────────────────

def uniform_weight(p_a: dict, p_b: dict) -> float:
    """ Constant weight of 1.0 -- makes weighted_triangulate_line numerically
    identical to the baseline's unweighted triangulate_line. """
    return 1.0


def pixel_length(p_a: dict, p_b: dict) -> float:
    """ EXP-A (Bartoli & Sturm, CVIU 2005): raw 2D pixel separation between the
    two points used to build one view's interpretation plane -- a longer
    baseline makes that plane's normal less sensitive to Molmo2's per-point
    localization noise.

    Args:
        * p_a: point dict with "x"/"y" keys, Molmo 0-1000 coordinates
        * p_b: point dict with "x"/"y" keys, Molmo 0-1000 coordinates

    Returns:
        * float: Euclidean distance in Molmo coordinate units

    """
    return float(np.hypot(p_a["x"] - p_b["x"], p_a["y"] - p_b["y"]))


def edge_distance_fraction(p: dict, molmo_scale: float = 1000.0) -> float:
    """ Fraction of the image half-extent a point sits away from the nearest edge.

    Args:
        * p: point dict with "x"/"y" keys, Molmo 0-1000 coordinates
        * molmo_scale: the fixed Molmo2 coordinate scale (1000)

    Returns:
        * float: 0.0 exactly on an edge, 0.5 at the exact image center

    """
    x, y = p["x"], p["y"]
    return float(min(x, molmo_scale - x, y, molmo_scale - y) / molmo_scale)


def edge_aware_weight(p_a: dict, p_b: dict, edge_margin: float = 0.1) -> float:
    """ EXP-D (AssemblyHands-X, arXiv:2509.23888): pixel-separation weight
    (pixel_length) penalized when either point sits close to the image
    border, where clipped FOV / partial-object visibility makes Molmo2's
    localization less trustworthy.

    Args:
        * p_a: point dict with "x"/"y" keys
        * p_b: point dict with "x"/"y" keys
        * edge_margin: fraction of the image half-extent at which the
          penalty stops applying (points farther than this from every edge
          get full weight)

    Returns:
        * float: pixel_length(p_a, p_b) scaled down toward 0 as either point
          approaches the border

    """
    edge_factor = min(edge_distance_fraction(p_a), edge_distance_fraction(p_b))
    edge_factor = float(np.clip(edge_factor / edge_margin, 0.0, 1.0))
    return pixel_length(p_a, p_b) * edge_factor


# ── Weighted linear algebra ─────────────────────────────────────────────────

def weighted_null_direction(rows: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """ Unit vector minimizing the weighted sum of squared dot-products with each row.

    Args:
        * rows: (k, 3) array, one row per observation (interpretation-plane
          normal, or candidate line direction)
        * weights: (k,) non-negative reliability weight per row

    Returns:
        * np.ndarray: (3,) unit vector -- the weighted counterpart of the
          unweighted SVD's smallest right-singular vector (see
          pipeline_common.triangulation.triangulate_line)

    """
    w = np.sqrt(np.clip(np.asarray(weights, dtype=np.float64), 1e-9, None))
    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64) * w[:, None])
    direction = vt[-1]
    return direction / np.linalg.norm(direction)


def weighted_triangulate_line(
    camera_centers: list[np.ndarray], plane_normals: list[np.ndarray], weights: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """ Weighted counterpart of pipeline_common.triangulation.triangulate_line.

    Args:
        * camera_centers: one (3,) world-space camera center per interpretation plane
        * plane_normals: one (3,) unit normal per interpretation plane
        * weights: one non-negative reliability weight per interpretation plane

    Returns:
        * tuple[np.ndarray, np.ndarray]: (point_on_line, unit_direction)

    """
    n = np.asarray(plane_normals, dtype=np.float64)
    c = np.asarray(camera_centers, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)

    direction = weighted_null_direction(n, w)

    sqrt_w = np.sqrt(np.clip(w, 1e-9, None))
    a = n * sqrt_w[:, None]
    b = np.einsum("ij,ij->i", n, c) * sqrt_w
    point, *_ = np.linalg.lstsq(a, b, rcond=None)
    return point, direction


def closest_point_to_rays(
    origins: list[np.ndarray], directions: list[np.ndarray], weights: list[float] | None = None,
) -> np.ndarray:
    """ EXP-B building block: least-squares point closest to a set of 3D rays.

    Args:
        * origins: one (3,) world-space origin per ray
        * directions: one (3,) (not necessarily normalized) direction per ray
        * weights: one non-negative reliability weight per ray, or None for uniform

    Returns:
        * np.ndarray: (3,) point minimizing the weighted sum of squared
          perpendicular distances to every ray

    """
    if weights is None:
        weights = [1.0] * len(origins)

    a = np.zeros((3, 3))
    b = np.zeros(3)
    for center, direction, weight in zip(origins, directions, weights):
        d = direction / np.linalg.norm(direction)
        proj = np.eye(3) - np.outer(d, d)
        a += weight * proj
        b += weight * (proj @ center)
    point, *_ = np.linalg.lstsq(a, b, rcond=None)
    return point


# ── Generic axis/plane estimators, parametrized by pair_selector + weight_fn ─
# Baseline behavior (Mapping/estimate_symmetry_no_mesh.py) is exactly
# pair_selector={widest_pair for axis, role_pair_selector for plane},
# weight_fn=uniform_weight -- every EXP-A/C/D variant below only swaps one of
# these two arguments, so the diff against the baseline stays legible.

def generic_estimate_axis(
    points_by_image: dict, images_sent: list[dict], fov_deg: float, image_size: int,
    pair_selector: PairSelector, weight_fn: WeightFn,
) -> dict:
    """ Axis triangulation generalized over point-pair selection and per-view weight.

    Args:
        * points_by_image: Molmo2 points, one list per view index (string keys)
        * images_sent: camera pose entries (R/T), one per view, aligned by index
        * fov_deg: camera field of view in degrees
        * image_size: render size in pixels (square)
        * pair_selector: callable(pts) -> (p_a, p_b) | None
        * weight_fn: callable(p_a, p_b) -> float

    Returns:
        * dict: {"direction", "origin", "n_views_used"}

    """
    centers, normals, weights = [], [], []
    for img_idx_str, pts in points_by_image.items():
        pair = pair_selector(pts)
        if pair is None:
            continue
        p_a, p_b = pair
        cam = images_sent[int(img_idx_str)]
        c, d_a = ray_dir_for_point(p_a["x"], p_a["y"], cam["R"], cam["T"], fov_deg, image_size)
        _, d_b = ray_dir_for_point(p_b["x"], p_b["y"], cam["R"], cam["T"], fov_deg, image_size)
        n = interpretation_plane_normal(d_a, d_b)
        if n is None:
            continue
        centers.append(c)
        normals.append(n)
        weights.append(weight_fn(p_a, p_b))

    if len(normals) < 2:
        raise ValueError(f"need >=2 valid views, got {len(normals)}")

    point, direction = weighted_triangulate_line(centers, normals, weights)
    return {"direction": direction.tolist(), "origin": point.tolist(), "n_views_used": len(normals)}


def _generic_line_from_pair(
    view_i: int, view_j: int, points_by_image: dict, images_sent: list[dict],
    fov_deg: float, image_size: int, pair_selector: PairSelector, weight_fn: WeightFn,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """ Generalized counterpart of estimate_symmetry_no_mesh._line_from_view_pair. """
    centers, normals, weights = [], [], []
    for idx in (view_i, view_j):
        pts = points_by_image.get(str(idx), [])
        pair = pair_selector(pts)
        if pair is None:
            return None
        p_a, p_b = pair
        cam = images_sent[idx]
        c, d_a = ray_dir_for_point(p_a["x"], p_a["y"], cam["R"], cam["T"], fov_deg, image_size)
        _, d_b = ray_dir_for_point(p_b["x"], p_b["y"], cam["R"], cam["T"], fov_deg, image_size)
        n = interpretation_plane_normal(d_a, d_b)
        if n is None:
            return None
        centers.append(c)
        normals.append(n)
        weights.append(weight_fn(p_a, p_b))
    point, direction = weighted_triangulate_line(centers, normals, weights)
    return point, direction, float(np.mean(weights))


def generic_estimate_plane(
    points_by_image: dict, images_sent: list[dict], fov_deg: float, image_size: int,
    edge_on_thresh: float, pair_selector: PairSelector, weight_fn: WeightFn,
) -> dict:
    """ Single-plane estimation generalized over point-pair selection and pair-line weight.

    Same 4-step scheme as estimate_symmetry_no_mesh.estimate_plane_no_mesh
    (candidate lines per view-pair -> candidate normals via cross product ->
    score by edge-on-ness -> SVD refit), but the pair-line construction uses
    pair_selector (instead of the fixed obj_id 1/2 lookup) and the refit step
    weights each pair-line by weight_fn (instead of treating every pair-line
    equally).

    Args:
        * points_by_image: Molmo2 points, one list per view index (string keys)
        * images_sent: camera pose entries (R/T), one per view, aligned by index
        * fov_deg: camera field of view in degrees
        * image_size: render size in pixels (square)
        * edge_on_thresh: |cos(angle)| below this counts a view as "edge-on"
        * pair_selector: callable(pts) -> (p_a, p_b) | None
        * weight_fn: callable(p_a, p_b) -> float

    Returns:
        * dict: {"normal", "origin", "n_views_used", "n_candidates", "good_views"}

    """
    view_idxs = sorted(int(k) for k, pts in points_by_image.items() if pair_selector(pts) is not None)
    if len(view_idxs) < 4:
        raise ValueError(f"need >=4 valid views (2 independent pairs), got {len(view_idxs)}")

    pair_lines = []
    for i, j in combinations(view_idxs, 2):
        res = _generic_line_from_pair(i, j, points_by_image, images_sent, fov_deg, image_size,
                                       pair_selector, weight_fn)
        if res is not None:
            point, direction, weight = res
            pair_lines.append(((i, j), point, direction, weight))

    if len(pair_lines) < 2:
        raise ValueError("not enough pair-lines to generate plane candidates")

    view_dirs = {
        idx: view_forward_direction(images_sent[idx]["R"], images_sent[idx]["T"], fov_deg, image_size)
        for idx in view_idxs
    }

    def score_normal(n: np.ndarray) -> float:
        vals = sorted(abs(np.dot(view_dirs[i], n)) for i in view_idxs)
        k = max(2, len(vals) // 2)
        return float(np.mean(vals[:k]))

    candidates = []
    for a in range(len(pair_lines)):
        pa, _, dir_a, _ = pair_lines[a]
        for b in range(a + 1, len(pair_lines)):
            pb, _, dir_b, _ = pair_lines[b]
            if set(pa) & set(pb):
                continue
            n = np.cross(dir_a, dir_b)
            norm = np.linalg.norm(n)
            if norm < 1e-6:
                continue
            candidates.append(n / norm)

    if not candidates:
        raise ValueError("could not generate any candidate normal (pair-lines too parallel)")

    scored = sorted(candidates, key=score_normal)
    best_normal = scored[0]
    good_views = [i for i in view_idxs if abs(np.dot(view_dirs[i], best_normal)) < edge_on_thresh]

    refit_dirs    = [d for (pa, _, d, _) in pair_lines if set(pa).issubset(good_views)]
    refit_points  = [p for (pa, p, _, _) in pair_lines if set(pa).issubset(good_views)]
    refit_weights = [w for (pa, _, _, w) in pair_lines if set(pa).issubset(good_views)]

    if len(refit_dirs) >= 2:
        refined_normal = weighted_null_direction(np.asarray(refit_dirs), np.asarray(refit_weights))
        used_points = refit_points
    else:
        refined_normal = best_normal
        used_points = [p for (_, p, _, _) in pair_lines]

    origin = np.mean(used_points, axis=0)
    return {
        "normal": refined_normal.tolist(), "origin": origin.tolist(),
        "n_views_used": len(good_views), "n_candidates": len(candidates), "good_views": good_views,
    }


def generic_detect_planes(
    points_by_image: dict, images_sent: list[dict], fov_deg: float, image_size: int,
    edge_on_thresh: float, max_planes: int, dup_angle_thresh_deg: float,
    pair_selector: PairSelector, weight_fn: WeightFn,
) -> list[dict]:
    """ Sequential multi-plane consolidation generalized over pair_selector/weight_fn.

    Same stopping rules as estimate_symmetry_no_mesh.detect_planes_no_mesh:
    pool has <4 views, the fit fails, or the candidate normal duplicates an
    already-accepted plane.

    Args:
        * points_by_image: Molmo2 points, one list per view index (string keys)
        * images_sent: camera pose entries (R/T), one per view, aligned by index
        * fov_deg: camera field of view in degrees
        * image_size: render size in pixels (square)
        * edge_on_thresh: |cos(angle)| below this counts a view as "edge-on"
        * max_planes: stop once this many planes have been accepted
        * dup_angle_thresh_deg: candidate planes closer than this to an
          already-accepted plane are treated as duplicates
        * pair_selector: callable(pts) -> (p_a, p_b) | None
        * weight_fn: callable(p_a, p_b) -> float

    Returns:
        * list[dict]: accepted planes, in acceptance order

    """
    def _ang(v1: np.ndarray, v2: np.ndarray) -> float:
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        return float(np.degrees(np.arccos(np.clip(np.abs(np.dot(v1, v2)), 0.0, 1.0))))

    all_view_idxs = sorted(int(k) for k, pts in points_by_image.items() if pair_selector(pts) is not None)
    pool = set(all_view_idxs)
    planes: list[dict] = []

    while len(planes) < max_planes:
        if len(pool) < 4:
            break
        sub_points = {k: v for k, v in points_by_image.items() if int(k) in pool}
        try:
            pred = generic_estimate_plane(sub_points, images_sent, fov_deg, image_size,
                                           edge_on_thresh, pair_selector, weight_fn)
        except ValueError:
            break

        normal = np.array(pred["normal"])
        if any(_ang(normal, np.array(p["normal"])) < dup_angle_thresh_deg for p in planes):
            break

        planes.append(pred)
        pool -= set(pred["good_views"])

    return planes
