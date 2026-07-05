"""
naming.py
---------
Shared --experiment-id filename suffixing, used by every stage of the
pipeline (MolmoPointing/, Mapping/) to keep experiment output files separate
from production output files without touching the production path.
"""
from __future__ import annotations


def exp_filename(base: str, experiment_id: str | None) -> str:
    """
    Return base unchanged, or base_EXPID.ext when experiment_id is set.

    Args:
        base: e.g. "molmo_multiview.json"
        experiment_id: e.g. "axis_v01" or None

    Returns:
        "molmo_multiview.json" when experiment_id is None
        "molmo_multiview_axis_v01.json" when experiment_id is "axis_v01"
    """
    if not experiment_id:
        return base
    dot = base.rfind(".")
    return f"{base[:dot]}_{experiment_id}{base[dot:]}"
