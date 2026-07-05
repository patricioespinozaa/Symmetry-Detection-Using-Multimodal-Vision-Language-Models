"""
prompts_registry.py
-------------------
Auto-discovers prompt variants from the prompts/ directory structure:

    prompts/
      axis/
        v00_single.txt    ← prompt_id "axis_v00", single-image mode
        v00_multi.txt     ← prompt_id "axis_v00", multi-image mode
        v01_single.txt    ← prompt_id "axis_v01"
        ...
      plane/
        v00_single.txt    ← prompt_id "plane_v00"
        v00_multi.txt
        ...

Adding a new prompt requires NO changes to this file:
    1. Create  prompts/axis/v01_single.txt  (prompt text for n_views == 1)
    2. Create  prompts/axis/v01_multi.txt   (prompt text for n_views  > 1)
    3. Optionally add a description in DESCRIPTIONS below.

Public API
----------
    from prompts_registry import get_prompt, list_prompts

    entry    = get_prompt("axis_v01")
    p_single = entry["single"]   # str
    p_multi  = entry["multi"]    # str
"""

from __future__ import annotations

from pathlib import Path

# Optional descriptions:
# Add an entry here to document what a specific variant tests.
# If a prompt_id is not listed, description defaults to the filename.

DESCRIPTIONS: dict[str, str] = {
    # Axis variants
    "axis_v00":  "Axis projection — two far-apart points directly on the projected axis",
    "axis_v01":  "Bilateral pair — left/right mirror points equidistant from the axis",
    "axis_v02":  "Widest silhouette extrema — leftmost and rightmost at the widest cross-section",
    "axis_v03":  "Structural feature pairs — visually symmetric elements (handles, holes, ribs)",
    "axis_v04":  "Polar extremes — topmost and bottommost points where the axis exits the surface",
    "axis_v05":  "Axis centerline — one point upper half + one point lower half on the centerline",
    # Plane variants
    "plane_v00": "Plane trace — top and bottom points on the plane's surface intersection",
    "plane_v01": "Bilateral pair — left/right mirror points equidistant from the symmetry plane",
    "plane_v02": "Plane seam — two points directly on the plane's surface trace (top and bottom)",
    "plane_v03": "Structural feature pairs — corresponding symmetric elements across the plane",
    "plane_v04": "Silhouette midpoints — horizontal center of left-right extent at top and bottom",
    "plane_v05": "Plane trace extremes — two most distant points along the visible plane trace",
    # Axis v1 variants (improved with centerline definition + upper/lower split + fallbacks)
    "axis_v00_1": "Axis projection v1 — upper/lower split, centerline definition, no silhouette edges",
    "axis_v01_1": "Bilateral pair v1 — height diversity across views, midpoint on centerline verified",
    "axis_v02_1": "Widest cross-section v1 — returns centerline midpoints at widest + top/bottom heights",
    "axis_v03_1": "Structural pairs v1 — midpoint verification + widest cross-section fallback",
    "axis_v04_1": "Polar extremes v1 — horizontal center verification + flat-surface fallback",
    "axis_v05_1": "Axis centerline v1 — explicit step-0 global axis identification, consistency check",
    # Plane v1 variants (improved with top/bottom split + midpoint verification + height diversity)
    "plane_v00_1": "Plane trace v1 — explicit top/bottom split + horizontal center guidance",
    "plane_v01_1": "Bilateral pair v1 — height diversity across views, midpoint on trace verified",
    "plane_v02_1": "Plane seam v1 — global plane step-0, consistency across views, top/bottom enforced",
    "plane_v03_1": "Structural pairs v1 — midpoint verification + widest cross-section fallback",
    "plane_v04_1": "Silhouette midpoints v1 — step-0 global plane ID + cross-view consistency check",
    "plane_v05_1": "Plane trace v1 — topmost/bottommost on trace, vertical separation explicit",
}


_PROMPTS_DIR = Path(__file__).parent / "prompts"

def _load_prompts() -> dict[str, dict[str, str]]:
    """
    Scan prompts/axis/ and prompts/plane/ for *_single.txt + *_multi.txt pairs.
    Returns a dict mapping prompt_id → {symmetry_type, description, single, multi}.
    """
    registry: dict[str, dict[str, str]] = {}

    for sym_type in ("axis", "plane"):
        sym_dir = _PROMPTS_DIR / sym_type
        if not sym_dir.exists():
            continue

        single_files = sorted(sym_dir.glob("*_single.txt"))
        for single_path in single_files:
            variant   = single_path.stem.replace("_single", "")   # e.g. "v00"
            multi_path = single_path.with_name(f"{variant}_multi.txt")

            if not multi_path.exists():
                print(f"[prompts_registry] Warning: {multi_path.name} not found "
                      f"for variant {variant!r} — skipping.")
                continue

            prompt_id = f"{sym_type}_{variant}"   # e.g. "axis_v00"
            registry[prompt_id] = {
                "symmetry_type": f"{sym_type}_sym",
                "description":   DESCRIPTIONS.get(prompt_id, f"{sym_type} {variant}"),
                "single":        single_path.read_text(encoding="utf-8").rstrip(),
                "multi":         multi_path.read_text(encoding="utf-8").rstrip(),
            }

    return registry


# Built once at import time
PROMPTS: dict[str, dict[str, str]] = _load_prompts()


# Public API
def get_prompt(prompt_id: str) -> dict[str, str]:
    """
    Return the registry entry for prompt_id. Raises KeyError if not found.
    Args:
    - prompt_id: e.g. "axis_v00", "plane_v01"
    Returns:
    - dict with keys: "symmetry_type", "description", "single", "multi"
    """
    if prompt_id not in PROMPTS:
        raise KeyError(
            f"Unknown prompt_id: {prompt_id!r}. "
            f"Available: {list(PROMPTS)}"
        )
    return PROMPTS[prompt_id]


def list_prompts() -> None:
    """Print all discovered prompts with their descriptions."""
    if not PROMPTS:
        print("No prompts found. Add .txt files to MolmoPointing/prompts/axis/ or prompts/plane/")
        return
    print(f"{'ID':<20} {'TYPE':<12} DESCRIPTION")
    print("─" * 65)
    for pid, entry in sorted(PROMPTS.items()):
        print(f"{pid:<20} {entry['symmetry_type']:<12} {entry['description']}")


# Flow B/C prompt plumbing
#
# Separate from the PROMPTS/_load_prompts() variant-testing registry above
# (which is keyed on single/multi pairs per prompt_id, one axis of A/B
# testing). These are flow plumbing -- fixed prompt text used to run the
# single-view description (Flow B) and description+point (Flow C) pre-pass,
# not experimental variants themselves.

_FLOW_PROMPTS_DIR = _PROMPTS_DIR / "description"


def load_flow_prompt(name: str) -> str:
    """
    Load a Flow B/C prompt file from prompts/description/<name>.txt.

    Args:
    - name: one of "describe", "describe_and_point_axis", "describe_and_point_plane"
    Returns:
    - the prompt text (stripped)
    """
    path = _FLOW_PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Flow prompt {name!r} not found: {path}. "
            f"Expected files in {_FLOW_PROMPTS_DIR}."
        )
    return path.read_text(encoding="utf-8").rstrip()


if __name__ == "__main__":
    list_prompts()