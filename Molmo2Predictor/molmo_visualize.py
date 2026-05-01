"""
molmo_visualize.py
------------------
Save a PNG with Molmo pointing results overlaid on the source image.
Coordinates are in the 0–1000 range with Y-axis inverted (Molmo convention).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
from PIL import Image


def save_annotated_image(
    image_path: str | Path,
    points: list[dict],
    output_path: str | Path,
    show_labels: bool = True,
    title: str = "Molmo2 — symmetry axis intersection points",
) -> None:
    """
    Overlay predicted symmetry points on the image and save to disk.

    Args:
        image_path:   Source PNG.
        points:       List of {"obj_id", "x", "y"} dicts (0–1000 coords).
        output_path:  Where to save the annotated PNG.
        show_labels:  Whether to draw obj_id labels next to each point.
        title:        Figure title.
    """
    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.imshow(img)

    for pt in points:
        # Scale from 0–1000 to pixel space; invert Y (Molmo origin = bottom-left)
        px = pt["x"] * (width  / 1000.0)
        py = height - pt["y"] * (height / 1000.0)

        ax.scatter(
            px, py,
            s=120,
            c="red",
            edgecolors="white",
            linewidths=1.5,
            zorder=10,
        )

        if show_labels:
            ax.text(
                px + 5, py + 5,
                str(pt["obj_id"]),
                color="white",
                fontsize=8,
                fontweight="bold",
                zorder=11,
            )

    ax.set_title(title, fontsize=9)
    ax.axis("off")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02, dpi=100)
    plt.close(fig)
