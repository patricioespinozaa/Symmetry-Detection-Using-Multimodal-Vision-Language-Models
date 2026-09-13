# Registry of mesh-free triangulation ablations (EXP-A..F, ported from
# Experiments/sandbox_pipeline_server_extended.ipynb and
# Experiments/sandbox_pipeline_server_final.ipynb), each a drop-in
# alternative to Mapping/estimate_symmetry_no_mesh.py's baseline estimator.
# Selected by name from Pipeline_Experiments/estimate_symmetry_variants.py.
from __future__ import annotations

import sys
from pathlib import Path

# Every variant module imports from pipeline_common.* -- make sure the repo
# root is importable even if this package is imported without going through
# estimate_symmetry_variants.py/run_batch.py first (both of which also add
# it, redundantly but harmlessly).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from . import (
    exp_a_weighted,
    exp_b_point_then_direction,
    exp_c_ransac2d,
    exp_d_confidence,
    exp_e_iterative_reweight,
    exp_f_sde_gate,
)

VARIANTS = {
    "expA": exp_a_weighted,
    "expB": exp_b_point_then_direction,
    "expC": exp_c_ransac2d,
    "expD": exp_d_confidence,
    "expE": exp_e_iterative_reweight,
    "expF": exp_f_sde_gate,
}


def get_variant(name: str):
    """ Look up a registered triangulation variant module by its EXP-id.

    Args:
        * name: one of "expA".."expF"

    Returns:
        * module: exposes SYMMETRY_TYPES, NEEDS_MESH, and estimate_axis
          and/or (estimate_plane and detect_planes) depending on SYMMETRY_TYPES

    """
    if name not in VARIANTS:
        raise KeyError(f"Unknown variant {name!r}. Available: {sorted(VARIANTS)}")
    return VARIANTS[name]
