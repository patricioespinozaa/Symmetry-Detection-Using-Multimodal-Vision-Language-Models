"""
prompts_registry.py
-------------------
Central registry of all prompt variants for the symmetry-detection experiments.

Each entry maps a prompt_id to a dict with:
    symmetry_type : "axis_sym" | "plane_sym"  (informational, not enforced)
    description   : short note on what this variant tests
    single        : prompt text used for single-image inference (n_views == 1)
    multi         : prompt text used for multi-image inference  (n_views  > 1)

Usage
-----
    from prompts_registry import get_prompt, list_prompts

    entry  = get_prompt("axis_v00")
    p_single = entry["single"]
    p_multi  = entry["multi"]

Adding a new prompt
-------------------
1. Copy an existing block below.
2. Give it a unique prompt_id  (e.g. "axis_v03", "plane_v02").
3. Edit "single" and "multi" texts.
4. Run the experiment with:
       --prompt-id axis_v03 --experiment-id axis_v03
"""

from __future__ import annotations

# ── Axis-symmetry prompt texts ─────────────────────────────────────────────────

_AXIS_V00_SINGLE = """You are given ONE image of a 3D object.

The object has ONE dominant rotational symmetry axis.

Your task is to estimate the 2D projection of the object's main symmetry axis.

Return TWO distant points lying on the same projected symmetry axis line.

IMPORTANT RULES:
- The two points MUST be different.
- The two points MUST be as far apart as possible along the symmetry axis.
- The points MUST lie on the visible object.
- Use the object's silhouette and geometry to infer the axis.
- Prefer the longest visible symmetry direction through the object.
- Do NOT place both points at the image center.
- If the object is approximately vertical, the axis should also be approximately vertical.

Output ONLY:

<points coords="1 1 X1 Y1 2 X2 Y2">"""


_AXIS_V00_MULTI = """You are given multiple views of the SAME 3D object.

The object has ONE dominant axis of rotational symmetry.

Your task is to estimate the SAME global 3D symmetry axis across all views \
and project that axis into each image.

For each image:
1. Identify the visible projection of the global symmetry axis.
2. Return:
   - one point where the projected axis intersects the object near the TOP and within the object pixels,
   - one point where the projected axis intersects the object near the BOTTOM and within the object pixels.

IMPORTANT RULES:
- The 2 points of each image MUST be collinear with the axis projection.
- Use the SAME global axis consistently across all views.
- Do NOT place all points at the image center unless the axis projection is truly degenerate.
- Infer the axis from ALL views jointly before answering.
- Even if some views are ambiguous, maintain consistency with the other views.
- Only output points for the dominant rotational symmetry axis.

Output format (one entry per image, separated by semicolons):

<points coords="1 1 Xtop Ytop 2 Xbottom Ybottom; 2 1 Xtop Ytop 2 Xbottom Ybottom; 3 1 Xtop Ytop 2 Xbottom Ybottom">

Where each entry is: image_index obj_id X Y
- obj_id 1 = TOP endpoint
- obj_id 2 = BOTTOM endpoint

Return ONLY the <points ...> block."""


# ── Plane-symmetry prompt texts ────────────────────────────────────────────────

_PLANE_V00_SINGLE = """You are given ONE image of a 3D object.

The object has ONE dominant plane of reflective symmetry.

Your task is to estimate the 2D projection of the symmetry plane boundary visible in this image.

Return TWO distant points that both lie on the visible intersection line of the symmetry plane with the object's surface.

IMPORTANT RULES:
- The two points MUST be different.
- The two points MUST be as far apart as possible along the visible symmetry boundary.
- The points MUST lie on the visible object surface.
- Use the object's bilateral symmetry to infer the mirror plane.
- The symmetry plane divides the object into two mirror-image halves; find the dividing boundary.
- Do NOT place both points at the image center.

Output ONLY:

<points coords="1 1 X1 Y1 2 X2 Y2">"""


_PLANE_V00_MULTI = """You are given multiple views of the SAME 3D object.

The object has ONE dominant plane of reflective symmetry.

Your task is to estimate the SAME global 3D symmetry plane across all views \
and project its visible boundary into each image.

For each image:
1. Identify the visible trace of the global symmetry plane on the object's surface.
2. Return:
   - one point on the symmetry plane trace near the TOP of the object,
   - one point on the symmetry plane trace near the BOTTOM of the object.

IMPORTANT RULES:
- Use the SAME global plane consistently across all views.
- The 2 points of each image MUST lie on the symmetry plane's visible intersection with the object.
- Infer the plane from ALL views jointly before answering.
- Even if some views are ambiguous, maintain consistency with the other views.

Output format (one entry per image, separated by semicolons):

<points coords="1 1 Xtop Ytop 2 Xbottom Ybottom; 2 1 Xtop Ytop 2 Xbottom Ybottom; 3 1 Xtop Ytop 2 Xbottom Ybottom">

Where each entry is: image_index obj_id X Y
- obj_id 1 = TOP endpoint on the symmetry plane trace
- obj_id 2 = BOTTOM endpoint on the symmetry plane trace

Return ONLY the <points ...> block."""


# ── Registry ──────────────────────────────────────────────────────────────────

PROMPTS: dict[str, dict[str, str]] = {

    # ── Axis variants ──────────────────────────────────────────────────────────
    "axis_v00": {
        "symmetry_type": "axis_sym",
        "description":   "Baseline axis prompt (current production)",
        "single":        _AXIS_V00_SINGLE,
        "multi":         _AXIS_V00_MULTI,
    },
    # Add new axis variants below:
    # "axis_v01": {
    #     "symmetry_type": "axis_sym",
    #     "description":   "Describe what this variant tests",
    #     "single":        "...",
    #     "multi":         "...",
    # },

    # ── Plane variants ─────────────────────────────────────────────────────────
    "plane_v00": {
        "symmetry_type": "plane_sym",
        "description":   "Baseline plane prompt",
        "single":        _PLANE_V00_SINGLE,
        "multi":         _PLANE_V00_MULTI,
    },
    # Add new plane variants below:
    # "plane_v01": {
    #     "symmetry_type": "plane_sym",
    #     "description":   "Describe what this variant tests",
    #     "single":        "...",
    #     "multi":         "...",
    # },
}


# ── Public API ────────────────────────────────────────────────────────────────

def get_prompt(prompt_id: str) -> dict[str, str]:
    """Return the registry entry for prompt_id. Raises KeyError if not found."""
    if prompt_id not in PROMPTS:
        available = list(PROMPTS.keys())
        raise KeyError(
            f"Unknown prompt_id: {prompt_id!r}. "
            f"Available: {available}"
        )
    return PROMPTS[prompt_id]


def list_prompts() -> None:
    """Print all registered prompts with their descriptions."""
    print(f"{'ID':<20} {'TYPE':<12} DESCRIPTION")
    print("─" * 65)
    for pid, entry in PROMPTS.items():
        print(f"{pid:<20} {entry['symmetry_type']:<12} {entry['description']}")
