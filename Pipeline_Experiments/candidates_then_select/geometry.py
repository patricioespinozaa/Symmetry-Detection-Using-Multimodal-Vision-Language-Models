# Candidate generation/drawing for EXP-LIT-1 (Candidates-then-Select --
# CVPC arXiv:2512.04686; ZeroDex arXiv:2606.19340). Ported from the oracle/
# random SIMULATION in Experiments/sandbox_literatura.ipynb into the pieces
# needed for REAL Molmo2 inference: draw numbered candidates on the actual
# render and let the model pick one, instead of picking the closest-to-GT
# candidate by hand.
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

K_CANDIDATES_DEFAULT = 4
DELTA_CANDIDATES_DEFAULT = 30.0   # radius, Molmo 0-1000 coordinate units
CANDIDATE_SEED_DEFAULT = 42

MARKER_COLOR = (255, 40, 40)


def generate_candidates(
    center_x: float, center_y: float,
    k: int = K_CANDIDATES_DEFAULT, delta: float = DELTA_CANDIDATES_DEFAULT,
    seed: int = CANDIDATE_SEED_DEFAULT,
) -> list[dict]:
    """ Generate k candidates in a disk of radius delta around a center point, plus the center itself.

    Args:
        * center_x: pass-1 point's x, Molmo 0-1000 coordinates
        * center_y: pass-1 point's y, Molmo 0-1000 coordinates
        * k: number of extra candidates to generate around the center
        * delta: candidate disk radius, Molmo 0-1000 coordinate units
        * seed: RNG seed -- fixed so the same pass-1 point always yields the
          same candidate set across re-runs

    Returns:
        * list[dict]: k+1 points ({"x", "y"}), candidate 0 is the original center

    """
    rng = np.random.default_rng(seed)
    candidates = [{"x": float(center_x), "y": float(center_y)}]
    angles = rng.uniform(0, 2 * np.pi, k)
    radii = rng.uniform(delta * 0.4, delta, k)
    for angle, radius in zip(angles, radii):
        candidates.append({
            "x": float(np.clip(center_x + radius * np.cos(angle), 5, 995)),
            "y": float(np.clip(center_y + radius * np.sin(angle), 5, 995)),
        })
    return candidates


def draw_candidates(image: Image.Image, candidates: list[dict]) -> Image.Image:
    """ Draw numbered circular markers (1-indexed) for each candidate on a copy of image.

    Args:
        * image: the render to mark up (left untouched -- a copy is returned)
        * candidates: points in Molmo 0-1000 coordinates, as returned by
          generate_candidates

    Returns:
        * Image.Image: a new RGB image with the markers drawn on top

    """
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    w, h = marked.size
    radius = max(5, int(min(w, h) * 0.015))
    try:
        font = ImageFont.truetype("arial.ttf", size=max(14, int(min(w, h) * 0.045)))
    except OSError:
        font = ImageFont.load_default()

    for i, c in enumerate(candidates, start=1):
        px, py = c["x"] / 1000.0 * w, c["y"] / 1000.0 * h
        draw.ellipse([px - radius, py - radius, px + radius, py + radius],
                     outline=MARKER_COLOR, width=2)
        draw.text((px + radius + 2, py - radius), str(i), fill=MARKER_COLOR, font=font)
    return marked
