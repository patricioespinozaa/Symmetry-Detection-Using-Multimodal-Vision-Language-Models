"""
prompts_registry.py
-------------------
MolmoPointPointing's own copy of MolmoPointing/prompts_registry.py's
auto-discovery logic -- deliberately duplicated rather than imported, so this
module has no dependency on MolmoPointing/ and nothing there needs to change
to support MolmoPoint-8B.

Unlike Molmo2/Qwen3-VL, MolmoPoint needs FOUR prompt files per variant, not
two: empirically (see molmopoint_multiview_runner.py's module docstring),
asking MolmoPoint to point to two different things in one call (e.g. "the top
pole AND the bottom pole") makes it fall back to free-text description with no
pointing tokens at all. The only reliable pattern found is one call per
target, each a single short "point to <noun>" instruction. So every variant
needs a _top and a _bottom prompt, each in a _single and _multi form:

    prompts/
      axis/
        v04_molmopoint_single_top.txt      <- prompt_id "axis_v04_molmopoint", n_views==1, top target
        v04_molmopoint_single_bottom.txt   <- same variant, bottom target
        v04_molmopoint_multi_top.txt       <- n_views>1, top target
        v04_molmopoint_multi_bottom.txt    <- n_views>1, bottom target
      plane/
        v05_1_molmopoint_single_top.txt
        v05_1_molmopoint_single_bottom.txt
        v05_1_molmopoint_multi_top.txt
        v05_1_molmopoint_multi_bottom.txt

Adding a new prompt requires NO changes to this file: drop all four files
for a new <variant> under prompts/axis/ or prompts/plane/ and it's discovered.

Public API
----------
    from prompts_registry import get_prompt, list_prompts

    entry = get_prompt("axis_v04_molmopoint")
    entry["single_top"], entry["single_bottom"]   # str, used when n_views == 1
    entry["multi_top"],  entry["multi_bottom"]     # str, used when n_views  > 1
"""

from __future__ import annotations

from pathlib import Path

# Optional descriptions:
# Add an entry here to document what a specific variant tests. If a prompt_id
# is not listed, description defaults to the filename.
DESCRIPTIONS: dict[str, str] = {}


_PROMPTS_DIR = Path(__file__).parent / "prompts"
_PARTS = ("single_top", "single_bottom", "multi_top", "multi_bottom")


def _load_prompts() -> dict[str, dict[str, str]]:
    """
    Scan prompts/axis/ and prompts/plane/ for the four *_single_top.txt /
    *_single_bottom.txt / *_multi_top.txt / *_multi_bottom.txt files that make
    up one variant. Returns a dict mapping
    prompt_id -> {symmetry_type, description, single_top, single_bottom,
    multi_top, multi_bottom}. A variant missing any of the four files is
    skipped with a warning, same policy as MolmoPointing/prompts_registry.py
    has for its single/multi pair.
    """
    registry: dict[str, dict[str, str]] = {}

    for sym_type in ("axis", "plane"):
        sym_dir = _PROMPTS_DIR / sym_type
        if not sym_dir.exists():
            continue

        top_files = sorted(sym_dir.glob("*_single_top.txt"))
        for top_path in top_files:
            variant = top_path.stem.replace("_single_top", "")   # e.g. "v04_molmopoint"
            paths   = {part: sym_dir / f"{variant}_{part}.txt" for part in _PARTS}

            missing = [p.name for p in paths.values() if not p.exists()]
            if missing:
                print(f"[prompts_registry] Warning: {missing} not found "
                      f"for variant {variant!r} — skipping.")
                continue

            prompt_id = f"{sym_type}_{variant}"   # e.g. "axis_v04_molmopoint"
            entry = {
                "symmetry_type": f"{sym_type}_sym",
                "description":   DESCRIPTIONS.get(prompt_id, f"{sym_type} {variant}"),
            }
            entry.update({part: paths[part].read_text(encoding="utf-8").rstrip() for part in _PARTS})
            registry[prompt_id] = entry

    return registry


# Built once at import time
PROMPTS: dict[str, dict[str, str]] = _load_prompts()


def get_prompt(prompt_id: str) -> dict[str, str]:
    """
    Return the registry entry for prompt_id. Raises KeyError if not found.
    Args:
    - prompt_id: e.g. "axis_v04_molmopoint", "plane_v05_1_molmopoint"
    Returns:
    - dict with keys: "symmetry_type", "description",
      "single_top", "single_bottom", "multi_top", "multi_bottom"
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
        print("No prompts found. Add .txt files to MolmoPointPointing/prompts/axis/ or prompts/plane/")
        return
    print(f"{'ID':<28} {'TYPE':<12} DESCRIPTION")
    print("─" * 75)
    for pid, entry in sorted(PROMPTS.items()):
        print(f"{pid:<28} {entry['symmetry_type']:<12} {entry['description']}")


if __name__ == "__main__":
    list_prompts()
