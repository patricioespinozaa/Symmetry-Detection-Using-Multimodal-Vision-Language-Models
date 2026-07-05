"""
view_selection.py
------------------
Seeded selection of a "description view" from a render metadata list, used
by MolmoPointing's Flow B (pointing con descripcion) and Flow C (descripcion
y pointing integrados) to pick one representative view for a free-form
Molmo2 description call.

Lives here rather than in MolmoPointing/ because it operates purely on
stage-1 render metadata (metadata_all.json entries), not on anything
Molmo-specific.
"""
from __future__ import annotations

import hashlib

import numpy as np


def select_description_view(
    object_id: str,
    metadata_entries: list[dict],
    elevation_range: tuple[float, float] = (-60.0, 60.0),
) -> dict:
    """
    Select one metadata entry to use for object description.

    Filters to entries with elevation strictly inside elevation_range (avoids
    near-pole views, which project an axis to a point or a plane to a line
    for most objects), then draws one entry uniformly at random using a
    per-object deterministic seed:

        seed = int(MD5(object_id), 16) mod 2**31

    A per-object seed (rather than one global seed) keeps the choice
    reproducible per object while ensuring different objects are described
    from different viewpoints.

    Falls back to the full entry list if none fall inside elevation_range
    (degenerate metadata -- should not happen with the standard 114-view
    Fibonacci sampling, but keeps this function total).

    Args:
        object_id: the object's identifier (used to derive the RNG seed)
        metadata_entries: list of metadata dicts (as returned by
            molmo_multiview_runner.load_metadata), each with at least
            "index" and "elevation" keys
        elevation_range: (lo, hi) in degrees, open interval

    Returns:
        the selected metadata entry (dict)
    """
    lo, hi = elevation_range
    candidates = [e for e in metadata_entries if lo < e["elevation"] < hi]
    if not candidates:
        candidates = list(metadata_entries)
    candidates = sorted(candidates, key=lambda e: e["index"])

    seed = int(hashlib.md5(object_id.encode("utf-8")).hexdigest(), 16) % (2 ** 31)
    idx  = int(np.random.default_rng(seed).integers(0, len(candidates)))
    return candidates[idx]
