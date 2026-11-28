# Registry of mesh-free triangulation variants, each a drop-in alternative to
# Mapping/estimate_symmetry_no_mesh.py's baseline estimator. Selected by name from
# Pipeline_Experiments/estimate_symmetry_variants.py.
#
# Only EXP-E (iterative residual reweighting, axis_sym) remains: it is the axis
# estimator of the thesis' final method. The ablations EXP-A..D and EXP-F were
# not reported and were removed from the repo; they remain in the git history.
from __future__ import annotations

import sys
from pathlib import Path

# The variant module imports from pipeline_common.* -- make sure the repo root is
# importable even if this package is imported without going through
# estimate_symmetry_variants.py first (which also adds it, redundantly but harmlessly).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from . import exp_e_iterative_reweight

VARIANTS = {"expE": exp_e_iterative_reweight}


def get_variant(name: str):
    """ Look up a registered triangulation variant module by its EXP-id.

    Args:
        * name: "expE"

    Returns:
        * module: exposes SYMMETRY_TYPES, NEEDS_MESH, and estimate_axis
          and/or (estimate_plane and detect_planes) depending on SYMMETRY_TYPES

    """
    if name not in VARIANTS:
        raise KeyError(f"Unknown variant {name!r}. Available: {sorted(VARIANTS)}")
    return VARIANTS[name]
