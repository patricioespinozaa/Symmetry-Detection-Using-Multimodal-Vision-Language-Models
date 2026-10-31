"""
prompts_registry.py
-------------------
QwenPointing's own copy of MolmoPointing/prompts_registry.py's auto-discovery
logic -- deliberately duplicated rather than imported, so this module has no
dependency on MolmoPointing/ and nothing there needs to change to support
Qwen3-VL-8B-Instruct.

Auto-discovers prompt variants from this folder's own prompts/ directory:

    prompts/
      axis/
        v04_qwen3vl_single.txt   <- prompt_id "axis_v04_qwen3vl", single-image mode
        v04_qwen3vl_multi.txt    <- prompt_id "axis_v04_qwen3vl", multi-image mode
      plane/
        v05_1_qwen3vl_single.txt <- prompt_id "plane_v05_1_qwen3vl"
        v05_1_qwen3vl_multi.txt
        ...

Adding a new prompt requires NO changes to this file:
    1. Create  prompts/axis/<variant>_single.txt
    2. Create  prompts/axis/<variant>_multi.txt
    3. Optionally add a description in DESCRIPTIONS below.

Public API
----------
    from prompts_registry import get_prompt, list_prompts

    entry    = get_prompt("axis_v04_qwen3vl")
    p_single = entry["single"]   # str
    p_multi  = entry["multi"]    # str
"""

from __future__ import annotations

from pathlib import Path

# Optional descriptions:
# Add an entry here to document what a specific variant tests. If a prompt_id
# is not listed, description defaults to the filename.
DESCRIPTIONS: dict[str, str] = {}


_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompts() -> dict[str, dict[str, str]]:
    """
    Scan prompts/axis/ and prompts/plane/ for *_single.txt + *_multi.txt pairs.
    Returns a dict mapping prompt_id -> {symmetry_type, description, single, multi}.
    """
    registry: dict[str, dict[str, str]] = {}

    for sym_type in ("axis", "plane"):
        sym_dir = _PROMPTS_DIR / sym_type
        if not sym_dir.exists():
            continue

        single_files = sorted(sym_dir.glob("*_single.txt"))
        for single_path in single_files:
            variant    = single_path.stem.replace("_single", "")   # e.g. "v04_qwen3vl"
            multi_path = single_path.with_name(f"{variant}_multi.txt")

            if not multi_path.exists():
                print(f"[prompts_registry] Warning: {multi_path.name} not found "
                      f"for variant {variant!r} — skipping.")
                continue

            prompt_id = f"{sym_type}_{variant}"   # e.g. "axis_v04_qwen3vl"
            registry[prompt_id] = {
                "symmetry_type": f"{sym_type}_sym",
                "description":   DESCRIPTIONS.get(prompt_id, f"{sym_type} {variant}"),
                "single":        single_path.read_text(encoding="utf-8").rstrip(),
                "multi":         multi_path.read_text(encoding="utf-8").rstrip(),
            }

    return registry


# Built once at import time
PROMPTS: dict[str, dict[str, str]] = _load_prompts()


def get_prompt(prompt_id: str) -> dict[str, str]:
    """
    Return the registry entry for prompt_id. Raises KeyError if not found.
    Args:
    - prompt_id: e.g. "axis_v04_qwen3vl", "plane_v05_1_qwen3vl"
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
        print("No prompts found. Add .txt files to QwenPointing/prompts/axis/ or prompts/plane/")
        return
    print(f"{'ID':<28} {'TYPE':<12} DESCRIPTION")
    print("─" * 75)
    for pid, entry in sorted(PROMPTS.items()):
        print(f"{pid:<28} {entry['symmetry_type']:<12} {entry['description']}")


if __name__ == "__main__":
    list_prompts()
