"""viz_colors.py

Shared RGB color constants for the Polyscope/matplotlib visualization scripts
under Mapping/ (visualize_rays.py, visualize_symmetry.py,
export_symmetry_overlay.py, render_symmetry_comparison.py). These constants
were previously copy-pasted verbatim across those files; consolidated here so
the GT/predicted color convention only needs to change in one place.

All values are RGB tuples in [0, 1].
"""

from __future__ import annotations

COLOR_MESH: tuple[float, float, float] = (0.75, 0.75, 0.75)
COLOR_CAMERA: tuple[float, float, float] = (0.20, 0.40, 0.90)  # blue
COLOR_HIT_RAY: tuple[float, float, float] = (0.15, 0.80, 0.25)  # green
COLOR_MISS_RAY: tuple[float, float, float] = (1.00, 0.55, 0.10)  # orange
COLOR_HIT_PT: tuple[float, float, float] = (1.00, 0.95, 0.10)  # yellow
COLOR_CLUSTER_PT: tuple[float, float, float] = (0.95, 0.30, 0.15)  # red-orange (cluster centroids)

COLOR_GT_AXIS: tuple[float, float, float] = (0.15, 0.40, 0.90)  # blue
COLOR_GT_PLANE: tuple[float, float, float] = (0.10, 0.80, 0.35)  # green
COLOR_PRED_AXIS: tuple[float, float, float] = (0.90, 0.15, 0.15)  # red
COLOR_PRED_PLANE: tuple[float, float, float] = (0.85, 0.10, 0.85)  # magenta
