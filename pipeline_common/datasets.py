"""
datasets.py
-----------
Dataset path conventions and mesh loading shared across Mapping/ and
InteractiveViewer/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

OBJECTS_SUBDIR: dict[str, str] = {
    "axis_sym":  "curated_axis_sym_obj",
    "plane_sym": "curated_plane_sym_obj",
}


def load_mesh(obj_path: Path) -> trimesh.Trimesh:
    """Load .obj as a trimesh, merging geometry if the file has multiple meshes."""
    scene_or_mesh = trimesh.load(str(obj_path), force="mesh", process=False)
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in scene_or_mesh.geometry.values()
             if isinstance(g, trimesh.Trimesh)]
        )
    else:
        mesh = scene_or_mesh
    return mesh


def load_mesh_vertices(obj_path: Path) -> np.ndarray | None:
    """Load .obj and return an (N, 3) vertex array. Returns None on failure."""
    try:
        return np.array(load_mesh(obj_path).vertices, dtype=np.float64)
    except Exception:
        return None
