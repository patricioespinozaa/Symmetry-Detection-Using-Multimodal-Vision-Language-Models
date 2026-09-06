"""
molmo_multiview_runner.py
-------------------------
Batch multi-view pointing inference with Molmo2 at scale.

For each object × resolution × illumination × n_views combination, runs
Molmo2 pointing and saves results to a cumulative JSON file. Results are
accumulated per (object, resolution, illumination) — re-running with new
view groups adds new keys without overwriting existing ones.

Experiment mode
---------------
Use --experiment-id and --prompt-id together to run isolated prompt experiments:

    --experiment-id EXP_ID   Changes output filename to molmo_multiview_EXP_ID.json.
                             Production file (molmo_multiview.json) is never touched.
    --prompt-id     PROMPT   Loads single/multi prompt texts from prompts_registry.py.
                             Run `python MolmoPointing/prompts_registry.py` to list
                             all registered prompts.
    --max-objects   N        Limits processing to the first N objects (sorted order),
                             useful for quick subset experiments.

Prompt modes
------------
Four modes controlled by --prompt-mode:

  global   All view groups use PROMPT_GLOBAL (one prompt for 1..N images).
  single   All view groups use PROMPT_SINGLE (single-image prompt, called
           once per image — N calls per group).
  multi    All view groups use PROMPT_MULTI  (multi-image prompt, one call
           per group). Falls back to PROMPT_SINGLE when n_views == 1.
  auto     n_views == 1  → PROMPT_SINGLE (one call, single-image format)
           n_views  > 1  → PROMPT_MULTI  (one call, multi-image format)
           This is the recommended mode.

Flows
-----
Three flows controlled by --flow (orthogonal to --prompt-mode/--prompt-id,
which still control the *main* N-view pointing call in all three flows):

  a   Direct pointing (default). No semantic context, no extra model call.
      Byte-for-byte identical to the original single-flow behavior.
  b   Pointing con descripcion. One extra single-view call on a seeded
      description view (elevation in (-60°, 60°), per-object deterministic
      seed) asks the model to describe the object; that description is
      prepended to the base pointing prompt before the normal N-view call.
  c   Descripcion y pointing integrados. Same seeded description view, but
      the extra call asks the model to describe the object AND give a
      short qualitative hint of where the symmetry axis/plane is located
      (e.g. "runs vertically from the spout tip to the base center") --
      no landmark list, no pixel coordinates (see
      parse_describe_and_location / describe_and_point_{axis,plane}.txt).
      That description + location hint replace the base --prompt-id prompt
      entirely for the main N-view call: the model is explicitly asked to
      find points ON the symmetry axis/plane in each image (obj_id just
      enumerates however many points that image has, up to a small cap --
      no cross-view identity tracking, see build_flow_c_prompts). Falls
      back to Flow B-style description-only augmentation of the base
      prompt if the pre-pass yields no parseable location hint.

--flow b/c only affect --prompt-mode single/multi/auto (whichever
prompt_single/prompt_multi ends up used); --prompt-mode global does not
support prompt overrides at all (not even via --prompt-id), so --flow b/c
has no effect when combined with it.

Output structure
----------------
<renders_root>/<symmetry_type>/<object_id>/<image_size>/<illumination>/
    molmo_multiview.json              ← production (no --experiment-id)
    molmo_multiview_<EXP_ID>.json     ← experiment

JSON format
-----------
{
  "6": {
    "experiment_id": "axis_v01",       # null when no --experiment-id
    "prompt_id":     "axis_v01",       # null when no --prompt-id
    "prompt_mode": "auto",
    "prompt_used": "...",
    "raw_output": "...",
    "points_by_image": {
      "0": [{"obj_id": 1, "x": 450.0, "y": 230.0}, ...],
      "1": [...]
    },
    "images_sent": [...],
    "n_points": 4,
    "n_images_with_points": 3
    # present only when --flow b or c (absent for flow a):
    "flow": "b",
    "description_used": "A wooden chair with four legs and a slatted back.",
    "description_view_idx": 47,
    "description_view_filename": "IND_47_AZ_020_EL_+59.png",
    "location_hint": null   # flow c only: e.g. "runs vertically through the lid and base"
  },
  ...
}

Resumability
------------
Each (object, resolution, illumination, n_views) combination is skipped if
its key already exists in the JSON *for the current --flow* (a stored key's
flow defaults to "a" when absent, so re-running with --flow a never
reprocesses anything, and re-running the same --experiment-id under a
different --flow reprocesses rather than silently reusing another flow's
result). Use a distinct --experiment-id per flow to keep artifacts separate.

Usage
-----
    # Production run (recommended mode)
    CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \\
        --renders-root ../data/renders \\
        --symmetry-type axis_sym \\
        --view-groups 1 6 14 26 \\
        --prompt-mode auto

    # Prompt experiment on first 50 objects
    CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \\
        --renders-root ../data/renders \\
        --symmetry-type axis_sym \\
        --sizes 224 --lightings flat \\
        --view-groups 1 6 14 26 \\
        --max-objects 50 \\
        --prompt-id axis_v01 \\
        --experiment-id axis_v01 \\
        --prompt-mode auto

    # Flow B (pointing con descripcion), using the best Flow-A prompt as base
    CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \\
        --renders-root ../data/renders \\
        --symmetry-type axis_sym \\
        --sizes 224 --lightings flat \\
        --view-groups 1 6 14 26 \\
        --prompt-id axis_v05_1 \\
        --flow b --experiment-id axis_v05_1_flowB \\
        --prompt-mode auto

    # Flow C (descripcion y pointing integrados)
    CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \\
        --renders-root ../data/renders \\
        --symmetry-type axis_sym \\
        --sizes 224 --lightings flat \\
        --view-groups 1 6 14 26 \\
        --prompt-id axis_v05_1 \\
        --flow c --experiment-id axis_v05_1_flowC \\
        --prompt-mode auto

    # Grid-overlay experiment (axis_lit2_grid requires --grid-overlay)
    CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \\
        --renders-root ../data/renders \\
        --symmetry-type axis_sym \\
        --sizes 224 --lightings flat \\
        --view-groups 1 6 14 26 \\
        --prompt-id axis_lit2_grid --grid-overlay \\
        --experiment-id axis_lit2_grid \\
        --prompt-mode auto

    # Two GPUs
    CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \\
        --renders-root ../data/renders --symmetry-type axis_sym \\
        --gpu-id 0 --num-gpus 2 --prompt-mode auto

    CUDA_VISIBLE_DEVICES=1 python MolmoPointing/molmo_multiview_runner.py \\
        --renders-root ../data/renders --symmetry-type plane_sym \\
        --gpu-id 1 --num-gpus 2 --prompt-mode auto
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

# Allow importing prompts_registry from this directory
sys.path.insert(0, str(Path(__file__).parent))
from prompts_registry import PROMPTS, get_prompt, list_prompts, load_flow_prompt

# Allow importing pipeline_common from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_common.view_selection import select_description_view

# ── Default prompt texts ────────────────────

PROMPT_SINGLE = """You are given ONE image of a 3D object.

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


PROMPT_MULTI = """You are given multiple views of the SAME 3D object.

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


PROMPT_GLOBAL = """You are given one or more images of the SAME 3D object.

The object has ONE dominant rotational symmetry axis.

Your task is to estimate the projection of the object's main symmetry axis in EACH image.

For every image, return TWO points that lie on the projected symmetry axis.

IMPORTANT RULES:
- The two points of each image MUST be different.
- The two points MUST lie on the same symmetry axis line.
- The points should be as far apart as possible along the visible axis.
- The points MUST lie on the visible object region.
- Use the object's geometry and silhouette to infer the axis.
- When multiple images are provided, use all views jointly to maintain consistency.
- Even if some views are ambiguous, keep the predicted axis consistent across views.
- Do NOT collapse points to the image center unless the axis projection is truly degenerate.

Output format:

<points coords="1 1 X1 Y1 2 X2 Y2; 2 1 X1 Y1 2 X2 Y2; 3 1 X1 Y1 2 X2 Y2">

Where each entry is: image_index obj_id X Y
- obj_id 1 = first endpoint of the projected axis
- obj_id 2 = second endpoint of the projected axis

Return ONLY the <points ...> block."""


PROMPT_MODES = ("global", "single", "multi", "auto")
FLOW_MODES   = ("a", "b", "c")
DESCRIPTION_ELEVATION_RANGE = (-60.0, 60.0)

# Constants

MODEL_ID      = "allenai/Molmo2-8B"
OUTPUT_FILE   = "molmo_multiview.json"

DEFAULT_VIEW_GROUPS   = [1, 6, 14, 26, 42, 62, 86, 114]
DEFAULT_IMAGE_SIZES   = [224, 448, 1136]
DEFAULT_ILLUMINATIONS = ["flat", "brighter", "darker"]

# Grid overlay for --grid-overlay (axis_lit2_grid, EXP-LIT-2 style, MOKA arXiv:2403.03174).
# Same 5x5 / A-E x 1-5 cell-labeling convention already validated (as a pure post-hoc
# simulation, not a real Molmo call) in Experiments/sandbox_literatura.ipynb.
GRID_N    = 5
GRID_COLS = [chr(65 + i) for i in range(GRID_N)]   # ["A", "B", "C", "D", "E"]
GRID_ROWS = list(range(1, GRID_N + 1))              # [1, 2, 3, 4, 5]


def apply_grid_overlay(image: Image.Image, n: int = GRID_N) -> Image.Image:
    """
    Returns a NEW image (the input is left untouched) with an n x n grid drawn
    on top: gridlines plus a column letter (A, B, ...) + row number (1, 2, ...)
    label in each cell's top-left corner, e.g. "C3" for the center cell of a
    5x5 grid. Purely a visual aid for the model's input -- does not touch the
    underlying render on disk, so ray casting / camera geometry downstream are
    unaffected.
    """
    grid_img = image.copy()
    draw     = ImageDraw.Draw(grid_img)
    w, h     = grid_img.size
    step_x, step_y = w / n, h / n

    line_color  = (255, 60, 60)
    label_color = (255, 60, 60)
    try:
        font = ImageFont.truetype("arial.ttf", size=max(12, int(min(step_x, step_y) * 0.18)))
    except OSError:
        font = ImageFont.load_default()

    for i in range(1, n):
        x = round(i * step_x)
        draw.line([(x, 0), (x, h)], fill=line_color, width=1)
        y = round(i * step_y)
        draw.line([(0, y), (w, y)], fill=line_color, width=1)

    for row in range(n):
        for col in range(n):
            label = f"{GRID_COLS[col]}{GRID_ROWS[row]}"
            x, y  = round(col * step_x) + 3, round(row * step_y) + 2
            draw.text((x, y), label, fill=label_color, font=font)

    return grid_img


# Filename helper
def _exp_filename(base: str, experiment_id: str | None) -> str:
    """
    Return base unchanged, or base_EXPID.ext when experiment_id is set.
    Args:
    - base: e.g. "molmo_multiview.json"
    - experiment_id: e.g. "axis_v01" or None
    Returns:
    - "molmo_multiview.json" when experiment_id is None
    - "molmo_multiview_axis_v01.json" when experiment_id is "axis_v01"
    """
    if not experiment_id:
        return base
    dot = base.rfind(".")
    return f"{base[:dot]}_{experiment_id}{base[dot:]}"


# Model singleton
_processor = None
_model     = None

def get_model():
    """Load the model and processor if not already loaded. Returns (processor, model)."""
    global _processor, _model
    if _processor is None or _model is None:
        print(f"[model] Loading {MODEL_ID} ...")
        _processor = AutoProcessor.from_pretrained(
            MODEL_ID, trust_remote_code=True, device_map="auto", use_fast=True
        )
        _model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            device_map="auto",
            dtype=torch.bfloat16,
        )
        _model.eval()
        print("[model] Ready.")
    return _processor, _model


# Metadata helpers
def load_metadata(render_dir: Path) -> list[dict]:
    """
    Load metadata_all.json from render_dir. Raises FileNotFoundError if not found.
    Args:
    - render_dir: directory containing metadata_all.json
    Returns:
    - list of metadata entries, where each entry is a dict with keys:
      - filename: str
      - index: int
      - azimuth: float
      - elevation: float
      - eye: str
      - R: list (3x3 rotation matrix)
      - T: list (3D translation vector)
    """
    path = render_dir / "metadata_all.json"
    if not path.exists():
        raise FileNotFoundError(f"metadata_all.json not found: {render_dir}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_n_views_entries(metadata: list[dict], n_views: int) -> list[dict]:
    """
    Select n_views entries evenly spaced across the full metadata sequence.

    Uses linspace to guarantee uniform angular coverage regardless of n_views.
    Taking the first n_views entries from a 114-point Fibonacci sequence would
    concentrate all cameras near the north pole (elevation +66° to +90° for
    n_views=6, and +33° to +90° for n_views=26).

    Args:
    - metadata: list of metadata entries (dicts with "index" key)
    - n_views: number of views to select
    Returns:
    - list of n_views metadata entries with uniform index spacing, sorted by index
    """
    total = len(metadata)
    if n_views >= total:
        return sorted(metadata, key=lambda e: e["index"])
    indices = {int(round(i)) for i in np.linspace(0, total - 1, n_views)}
    entries = [m for m in metadata if m["index"] in indices]
    return sorted(entries, key=lambda e: e["index"])


# Low-level model call
def _call_model(images: list[Image.Image], prompt: str) -> str:
    """
    Single model call with 1..N images. Returns raw decoded output.
    Args:
    - images: list of PIL Images to send (1..N)
    - prompt: prompt text to use for this call
    Returns:
    - raw decoded text output from the model (the <points ...> block)
    """
    processor, model = get_model()

    content = [{"type": "text", "text": prompt}]
    for img in images:
        content.append({"type": "image", "image": img})

    messages = [{"role": "user", "content": content}]
    inputs   = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    max_new_tokens = 2048

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    n_input = inputs["input_ids"].size(1)
    del inputs                  # free GPU tensors immediately
    text = processor.tokenizer.decode(
        output_ids[0, n_input:],
        skip_special_tokens=True,
    )
    del output_ids
    torch.cuda.empty_cache()
    return text, n_input


# Coordinate parsers 
def parse_single_coords(text: str) -> dict[str, list[dict]]:
    """
    Single-image Molmo2 format:
        <points coords="RADIO  ID X Y  ID X Y ...">
    Returns {"0": [{obj_id, x, y}, ...]}

    Args:
    - text: raw text output from the model containing the <points ...> block
    Returns:
    - dict with a single key "0" mapping to a list of points, where each point is a dict with keys.
    """
    match = re.search(r'coords=["\']([^"\']+)["\']', text)
    if not match:
        return {}
    raw = [float(n) for n in match.group(1).split()]
    if len(raw) < 4:
        return {}
    raw = raw[1:]   # drop radius token
    pts = []
    for i in range(0, len(raw) - 2, 3):
        pts.append({"obj_id": int(raw[i]), "x": raw[i + 1], "y": raw[i + 2]})
    return {"0": pts} if pts else {}


def parse_multi_coords(text: str, n_images: int) -> dict[str, list[dict]]:
    """
    Multi-image Molmo2 format:
        <points coords="img_idx obj_id X Y; img_idx obj_id X Y ...">
    img_idx is 1-based; stored as 0-based string keys.
    Returns {"0": [...], "1": [...], ...}

    Args:
    - text: raw text output from the model containing the <points ...> block
    - n_images: number of images that were sent to the model (for validation)
    Returns:
    - dict mapping image index (as string) to list of points, where each point is a dict with keys:
      - obj_id: int
      - x: float
      - y: float
    """
    match = re.search(r'coords=["\']([^"\']+)["\']', text)
    if not match:
        return {}

    result: dict[str, list[dict]] = {}
    for group in match.group(1).split(";"):
        group = group.strip()
        if not group:
            continue
        nums = group.split()
        if len(nums) < 4:
            continue
        try:
            img_idx = int(float(nums[0])) - 1   # 1-based → 0-based
        except ValueError:
            continue
        if img_idx < 0 or img_idx >= n_images:
            continue

        key  = str(img_idx)
        rest = nums[1:]
        pts  = []
        for i in range(0, len(rest) - 2, 3):
            try:
                pts.append({
                    "obj_id": int(float(rest[i])),
                    "x":      float(rest[i + 1]),
                    "y":      float(rest[i + 2]),
                })
            except ValueError:
                continue
        if pts:
            result.setdefault(key, []).extend(pts)

    return result


def parse_describe_and_location(text: str) -> dict:
    """
    Flow C's single-view combined describe+locate response: free text
    description followed by a one-line qualitative hint of where the
    symmetry axis/plane is located -- no landmark list, no labels, no pixel
    coordinates (see prompts/description/describe_and_point_{axis,plane}.txt).
    Localization of actual points happens entirely in the main N-view
    pointing call (see build_flow_c_prompts), using only the description and
    this location hint as context.

    Splits the text on the "Axis location:" / "Plane location:" marker line
    (either wording accepted, case-insensitively) to recover the free-text
    description separately from the location hint.

    Args:
    - text: raw text output from the model
    Returns:
    - {"description": str, "location_hint": str} -- location_hint is "" if
      the model didn't produce a parseable marker line (graceful fallback:
      the whole response is treated as the description, location hint left
      empty -- callers then fall back to Flow B-style behavior).
    """
    match = re.search(r'\n\s*(?:Axis|Plane) location:\s*(.*)', text, re.DOTALL | re.IGNORECASE)
    if match:
        description   = text[:match.start()].strip()
        location_hint = match.group(1).strip()
    else:
        description, location_hint = text.strip(), ""

    return {"description": description, "location_hint": location_hint}


def augment_prompt_with_description(base_prompt: str, description: str) -> str:
    """
    Flow B: prepend a semantic description of the object (generated from a
    single reference view) ahead of the base pointing prompt's instructions.
    """
    return (
        f"Context: the object being shown is described as follows:\n"
        f"\"{description}\"\n\n"
        f"Use this context to inform where the structurally relevant points are.\n\n"
        f"{base_prompt}"
    )


FLOW_C_POINTS_PER_IMAGE = 2   # exact point count/image for Flow C's main pointing call --
                              # matches the geometric minimum the no-mesh triangulation
                              # needs per view (2 points -> one interpretation plane; see
                              # docs/pipeline_sin_malla.md S3.1 and
                              # docs/implementacion_pipeline_sin_malla.md). Was previously
                              # "at most 3, fewer OK", which let Molmo return 0-1 points/image
                              # inconsistently once several images shared one call -- confirmed
                              # empirically (axis_v05_1_flowC and plane_v04_1_flowC both
                              # collapsed to 1 point/image at n_views>1), not just a theoretical
                              # concern.


def build_flow_c_prompts(
    description:   str,
    location_hint: str,
    symmetry_type: str,
) -> tuple[str, str]:
    """
    Flow C: build the main N-view pointing prompts from the single-view
    pre-pass's free-text description and qualitative axis/plane location
    hint (see parse_describe_and_location /
    describe_and_point_{axis,plane}.txt) -- NOT an augmentation of the base
    Flow-A prompt (kept separate so the wording below can explicitly
    reinstate the "find points ON the axis/plane" goal and the
    anti-degenerate rules).

    Unlike an earlier iteration of this design, points are NOT tracked by
    persistent cross-view identity: obj_id here just enumerates the (always
    exactly FLOW_C_POINTS_PER_IMAGE) points a given image independently
    returns -- the model is not asked to "re-find" a specific named
    landmark, only to locate points on the symmetry axis/plane using the
    description and location hint as context. This is simpler and, per
    empirical testing, more reliable: asking Molmo to track K named
    landmarks across many views (in one batched multi-image call) was
    observed to degenerate into mechanically evenly-spaced grid/line
    patterns of points and duplicated coordinates instead of genuine
    per-point reasoning -- the same failure mode noted for multi-point
    prompting in the ZeroKey paper. The IMPORTANT RULES below explicitly
    target those observed failures.

    Point count fixed at exactly FLOW_C_POINTS_PER_IMAGE=2 (not "up to N,
    fewer OK" as in an earlier version): with a flexible cap, Molmo
    empirically collapsed to 1 point/image once several images shared a
    single call, which is unusable for the no-mesh triangulation pipeline
    (needs 2 points/view minimum to define an interpretation plane, for
    both axis_sym and plane_sym -- see docs/pipeline_sin_malla.md S3.1/S3.2).
    Downstream consumers should NOT assume the two points have fixed
    semantic roles (e.g. "top"/"bottom") across views -- pick whichever pair
    is most useful per algorithm (e.g. widest_pair() in
    pipeline_common/triangulation.py), same as this function's docstring
    already established for obj_id identity.

    Returns (prompt_single, prompt_multi).
    """
    label = "axis" if symmetry_type == "axis_sym" else "plane"

    if symmetry_type == "axis_sym":
        task_single = "find TWO points that lie ON the object's symmetry axis in THIS image"
        task_multi  = "find TWO points that lie ON the object's symmetry axis"
        consistency = "The symmetry axis is the SAME across all views -- keep it consistent."
    else:
        task_single = (
            "find TWO points that lie ON the visible trace of the object's symmetry plane "
            "(the line/seam dividing it into two mirror halves) in THIS image"
        )
        task_multi = (
            "find TWO points that lie ON the visible trace of the object's symmetry plane "
            "(the line/seam dividing it into two mirror halves)"
        )
        consistency = "The symmetry plane is the SAME across all views -- keep it consistent."

    context = (
        f"Context: the object being shown is described as follows:\n"
        f"\"{description}\"\n\n"
        f"A separate reference view of this SAME object was used to estimate where the "
        f"symmetry {label} is located: \"{location_hint}\"\n\n"
    )

    anti_degenerate_rules = (
        f"IMPORTANT RULES:\n"
        f"- Return EXACTLY {FLOW_C_POINTS_PER_IMAGE} points -- not fewer, not more -- even if "
        f"you are not fully confident; give your best two independent estimates.\n"
        f"- The two points MUST be different from each other and as far apart as possible "
        f"along the {label} -- do NOT repeat the same (X, Y) coordinate for both.\n"
        f"- Each point MUST be independently reasoned from the object's visible geometry -- do "
        f"NOT output evenly-spaced or grid-like coordinates.\n"
        f"- Do NOT force a point at the image center.\n"
    )

    prompt_multi = context + (
        f"You are given multiple views of the SAME 3D object.\n\n"
        f"Your task: for EACH image below, {task_multi}, using the description and the "
        f"{label} location above as context. {consistency}\n\n"
        f"{anti_degenerate_rules}"
        f"- Number the two points 1 and 2 within each image -- these numbers do NOT need "
        f"to refer to the same physical feature across different images, they just tell the "
        f"two points of that image apart.\n\n"
        f"Output format (one entry per image, separated by semicolons):\n\n"
        f'<points coords="1 1 X1 Y1 2 X2 Y2; 2 1 X1 Y1 2 X2 Y2; 3 1 X1 Y1 2 X2 Y2">\n\n'
        f"Where each entry is: image_index obj_id X Y\n\n"
        f"Return ONLY the <points ...> block."
    )

    prompt_single = context + (
        f"You are given ONE image of the SAME 3D object.\n\n"
        f"Your task: {task_single}, using the description and the {label} location above as "
        f"context.\n\n"
        f"{anti_degenerate_rules}\n"
        f"Output ONLY:\n\n"
        f'<points coords="1 1 X1 Y1 2 X2 Y2">'
    )

    return prompt_single, prompt_multi


def prepare_flow_context(
    flow:                      str,
    render_dir:                Path,
    object_id:                 str,
    metadata:                  list[dict],
    describe_prompt:           str | None,
    describe_and_point_prompt: str | None,
) -> dict:
    """
    Runs the Flow B/C single-view pre-pass: selects a description view (seeded
    per-object, filtered to non-degenerate elevations) and makes exactly one
    extra model call on that single image.

    Flow "b": calls with describe_prompt, free-text output only.
    Flow "c": calls with describe_and_point_prompt, parses both a description
              and a qualitative axis/plane location hint from the combined
              response.

    Args:
    - flow: "b" or "c"
    - render_dir: directory containing the rendered images for this
      (object, size, illumination) config
    - object_id: used to derive the per-object view-selection seed
    - metadata: full list of metadata entries (as returned by load_metadata)
    - describe_prompt: prompt text for flow "b"
    - describe_and_point_prompt: prompt text for flow "c"

    Returns:
    - {"description": str, "view_index": int, "view_filename": str,
       "location_hint": str | None, "extra_tokens": int}
      location_hint is None for flow "b" (no locating done in the pre-pass).
    """
    view = select_description_view(object_id, metadata, DESCRIPTION_ELEVATION_RANGE)
    image = Image.open(render_dir / view["filename"]).convert("RGB")

    if flow == "b":
        raw, n_tok = _call_model([image], describe_prompt)
        description, location_hint = raw.strip(), None
    else:  # flow == "c"
        raw, n_tok = _call_model([image], describe_and_point_prompt)
        parsed = parse_describe_and_location(raw)
        description, location_hint = parsed["description"], parsed["location_hint"]

    return {
        "description":   description,
        "view_index":    view["index"],
        "view_filename": view["filename"],
        "location_hint": location_hint,
        "extra_tokens":  n_tok,
    }


def build_augmented_prompts(
    flow_context:  dict,
    prompt_single: str,
    prompt_multi:  str,
) -> tuple[str, str]:
    """
    Flow B (and Flow C's fallback when its pre-pass yields no location
    hint): augments the base prompt_single/prompt_multi with the
    flow_context's free-text description only, ahead of the base prompt's
    own instructions.

    Returns (augmented_prompt_single, augmented_prompt_multi).
    """
    augment = lambda p: augment_prompt_with_description(p, flow_context["description"])
    return augment(prompt_single), augment(prompt_multi)


# Inference modes
def run_global(images: list[Image.Image], prompt: str = PROMPT_GLOBAL) -> dict:
    """
    One call with all images using the given prompt (global format).
    Args:
    - images: list of PIL Images to send (1..N)
    - prompt: prompt text to use for this call
    Returns:
    - dict with keys: "prompt_used", "raw_output", "points_by_image", "n_input_tokens"
    """
    raw, n_tok    = _call_model(images, prompt)
    points_by_img = parse_multi_coords(raw, n_images=len(images))
    return {
        "prompt_used":     prompt,
        "raw_output":      raw,
        "points_by_image": points_by_img,
        "n_input_tokens":  n_tok,
    }


def run_single_mode(images: list[Image.Image], prompt: str = PROMPT_SINGLE) -> dict:
    """
    One call per image using the given single-image prompt.
    raw_output stored as list (one entry per image).
    Args:
    - images: list of PIL Images to send (1..N)
    - prompt: prompt text to use for each call
    Returns:
    - dict with keys: "prompt_used", "raw_output", "points_by_image", "n_input_tokens"
    """
    raw_outputs   = []
    points_by_img = {}
    total_tokens  = 0

    for i, img in enumerate(images):
        raw, n_tok = _call_model([img], prompt)
        raw_outputs.append(raw)
        total_tokens += n_tok
        pts = parse_single_coords(raw)
        if "0" in pts:
            points_by_img[str(i)] = pts["0"]

    return {
        "prompt_used":     prompt,
        "raw_output":      raw_outputs,
        "points_by_image": points_by_img,
        "n_input_tokens":  total_tokens,
    }


def run_multi(images: list[Image.Image], prompt: str = PROMPT_MULTI) -> dict:
    """
    One call with all images using the given multi-image prompt.
    Args:
    - images: list of PIL Images to send (1..N)
    - prompt: prompt text to use for this call
    Returns:
    - dict with keys: "prompt_used", "raw_output", "points_by_image", "n_input_tokens"
    """
    raw, n_tok    = _call_model(images, prompt)
    points_by_img = parse_multi_coords(raw, n_images=len(images))
    return {
        "prompt_used":     prompt,
        "raw_output":      raw,
        "points_by_image": points_by_img,
        "n_input_tokens":  n_tok,
    }


def run_inference(
    images:        list[Image.Image],
    prompt_mode:   str,
    prompt_single: str = PROMPT_SINGLE,
    prompt_multi:  str = PROMPT_MULTI,
    prompt_global: str = PROMPT_GLOBAL,
) -> dict:
    """
    Dispatch to the correct inference mode.

    global  → run_global,      1 call  for all images
    single  → run_single_mode, 1 call  per image
    multi   → run_multi,       1 call  for all images (fallback to single if n==1)
    auto    → single if n==1, multi if n>1  [recommended]
    """
    n = len(images)

    if prompt_mode == "global":
        return run_global(images, prompt=prompt_global)
    elif prompt_mode == "single":
        return run_single_mode(images, prompt=prompt_single)
    elif prompt_mode == "multi":
        return run_single_mode(images, prompt=prompt_single) if n == 1 \
               else run_multi(images, prompt=prompt_multi)
    elif prompt_mode == "auto":
        return run_single_mode(images, prompt=prompt_single) if n == 1 \
               else run_multi(images, prompt=prompt_multi)
    else:
        raise ValueError(f"Unknown prompt_mode: {prompt_mode!r}")


# JSON helpers
def load_results(json_path: Path) -> dict:
    """
    Load existing results from json_path, or return empty dict if file doesn't exist.
    Args:
    - json_path: path to the JSON file to load
    Returns:
    - dict containing the loaded results, or empty dict if file doesn't exist
    """
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(json_path: Path, data: dict) -> None:
    """
    Save results to json_path, creating parent directories if needed.
    Args:
    - json_path: path to the JSON file to save
    - data: dict containing the results to save
    Returns:
    - None
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# Per-object processor 
def process_object(
    object_dir:    Path,
    view_groups:   list[int],
    sizes:         list[int],
    lightings:     list[str],
    prompt_mode:   str,
    experiment_id: str | None = None,
    prompt_id:     str | None = None,
    prompt_single: str = PROMPT_SINGLE,
    prompt_multi:  str = PROMPT_MULTI,
    token_log:     dict | None = None,
    flow:          str = "a",
    describe_prompt:           str | None = None,
    describe_and_point_prompt: str | None = None,
    grid_overlay:  bool = False,
) -> None:
    """
    Process one object across all (size, illumination, n_views) combinations.
    Skips any (size, illumination, n_views) already present in the JSON for
    the current flow. token_log: optional dict {n_views: [token_counts]} to
    accumulate across objects.

    flow == "a" (default) is byte-for-byte identical to the original
    single-flow behavior: no extra model call, no extra JSON keys. flow in
    ("b", "c") runs one extra single-view description/describe+point call
    per (size, lighting) config (see prepare_flow_context) and augments
    prompt_single/prompt_multi with its result before the normal N-view
    pointing call.

    grid_overlay: when True, draws the 5x5 lettered/numbered grid (see
    apply_grid_overlay) on every image sent to the model for the main
    N-view pointing call (used by --prompt-id axis_lit2_grid). Only affects
    the in-memory copy sent to Molmo -- "images_sent" in the output JSON
    still records the original render filenames, so downstream mapping
    (which re-reads the plain PNGs) is unaffected.
    """
    output_file   = _exp_filename(OUTPUT_FILE, experiment_id)
    symmetry_type = object_dir.parent.name

    for size in sizes:
        for lighting in lightings:
            render_dir = object_dir / str(size) / lighting
            if not render_dir.exists():
                continue

            try:
                metadata = load_metadata(render_dir)
            except FileNotFoundError:
                continue

            json_path = render_dir / output_file
            results   = load_results(json_path)

            pending = [
                n for n in view_groups
                if str(n) not in results or results[str(n)].get("flow", "a") != flow
            ]
            if not pending:
                continue

            flow_context = None
            if flow != "a":
                for existing in results.values():
                    if (isinstance(existing, dict) and existing.get("flow") == flow
                            and "description_used" in existing):
                        flow_context = {
                            "description":   existing["description_used"],
                            "view_index":    existing.get("description_view_idx"),
                            "view_filename": existing.get("description_view_filename"),
                            "location_hint": existing.get("location_hint"),
                            "extra_tokens":  0,
                        }
                        break
                if flow_context is None:
                    flow_context = prepare_flow_context(
                        flow, render_dir, object_dir.name, metadata,
                        describe_prompt, describe_and_point_prompt,
                    )

            for n_views in pending:
                entries = get_n_views_entries(metadata, n_views)
                if not entries:
                    continue

                images = [
                    Image.open(render_dir / e["filename"]).convert("RGB")
                    for e in entries
                ]
                if grid_overlay:
                    images = [apply_grid_overlay(img) for img in images]

                if flow == "a":
                    call_prompt_single, call_prompt_multi = prompt_single, prompt_multi
                elif flow == "c" and flow_context.get("location_hint"):
                    call_prompt_single, call_prompt_multi = build_flow_c_prompts(
                        flow_context["description"], flow_context["location_hint"], symmetry_type
                    )
                else:
                    call_prompt_single, call_prompt_multi = build_augmented_prompts(
                        flow_context, prompt_single, prompt_multi
                    )

                inference     = run_inference(
                    images, prompt_mode,
                    prompt_single=call_prompt_single,
                    prompt_multi=call_prompt_multi,
                )
                points_by_img = inference["points_by_image"]
                n_pts         = sum(len(v) for v in points_by_img.values())
                n_tok         = inference.get("n_input_tokens", 0)

                if token_log is not None:
                    token_log.setdefault(n_views, []).append(n_tok)

                results[str(n_views)] = {
                    "experiment_id":         experiment_id,
                    "prompt_id":             prompt_id,
                    "prompt_mode":           prompt_mode,
                    "prompt_used":           inference["prompt_used"],
                    "raw_output":            inference["raw_output"],
                    "points_by_image":       points_by_img,
                    "images_sent": [
                        {
                            "img_idx":   i,
                            "filename":  e["filename"],
                            "index":     e["index"],
                            "azimuth":   e["azimuth"],
                            "elevation": e["elevation"],
                            "eye":       e["eye"],
                            "R":         e["R"],
                            "T":         e["T"],
                        }
                        for i, e in enumerate(entries)
                    ],
                    "n_points":             n_pts,
                    "n_images_with_points": len(points_by_img),
                    "n_input_tokens":       n_tok,
                }

                if flow != "a":
                    results[str(n_views)]["flow"]                     = flow
                    results[str(n_views)]["description_used"]         = flow_context["description"]
                    results[str(n_views)]["description_view_idx"]     = flow_context["view_index"]
                    results[str(n_views)]["description_view_filename"] = flow_context["view_filename"]
                    if flow == "c":
                        results[str(n_views)]["location_hint"] = flow_context["location_hint"]
                    results[str(n_views)]["n_input_tokens"] += flow_context["extra_tokens"]

                # Write after every n_views to survive interruptions
                save_results(json_path, results)


# CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch Molmo2 multi-view pointing — cumulative JSON output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--renders-root",  required=True,
                   help="Root folder produced by data_render.py")
    p.add_argument("--symmetry-type", required=True,
                   choices=["axis_sym", "plane_sym"])

    p.add_argument("--prompt-mode", default="auto", choices=PROMPT_MODES,
                   help=(
                       "global = PROMPT_GLOBAL for all groups (1 call/group); "
                       "single = PROMPT_SINGLE per image (N calls/group); "
                       "multi  = PROMPT_MULTI for all groups (1 call/group, "
                       "fallback to single when n_views==1); "
                       "auto   = single if n_views==1, multi otherwise [recommended]"
                   ))
    p.add_argument("--flow", default="a", choices=FLOW_MODES,
                   help=(
                       "a = direct pointing, no semantic context [default, unchanged behavior]; "
                       "b = pointing con descripcion -- one extra single-view describe call "
                       "prepended to the pointing prompt; "
                       "c = descripcion y pointing integrados -- one extra single-view "
                       "describe+point call, whose result seeds the pointing prompt. "
                       "Both b/c pick a seeded description view per object "
                       "(elevation in (-60, 60)) and only affect --prompt-mode single/multi/auto "
                       "(not global, which does not support prompt overrides)."
                   ))

    # Experiment flags
    p.add_argument("--experiment-id", default=None,
                   help=(
                       "Experiment identifier. When set, results are saved to "
                       "molmo_multiview_<ID>.json instead of molmo_multiview.json. "
                       "Use together with --prompt-id for prompt experiments."
                   ))
    p.add_argument("--prompt-id", default=None,
                   help=(
                       "Prompt variant from prompts_registry.py "
                       "(e.g. axis_v01, plane_v02). "
                       "Run `python MolmoPointing/prompts_registry.py` to list options."
                   ))
    p.add_argument("--grid-overlay", action="store_true",
                   help=(
                       "Draw a 5x5 lettered/numbered grid (A-E x 1-5) on every image sent to "
                       "the model for the main N-view call (see apply_grid_overlay). Only the "
                       "in-memory copy sent to Molmo is affected; required for --prompt-id "
                       "axis_lit2_grid, unused by every other prompt."
                   ))
    p.add_argument("--max-objects", type=int, default=None,
                   help="Limit to the first N objects (sorted order). Useful for subset experiments.")
    p.add_argument("--list-prompts", action="store_true",
                   help="List all registered prompts and exit.")
    p.add_argument("--yes", "-y", action="store_true",
                   help="Skip the confirmation prompt (useful for automated loops).")

    p.add_argument("--gpu-id",   type=int, default=0)
    p.add_argument("--num-gpus", type=int, default=1)

    p.add_argument("--view-groups", type=int, nargs="+",
                   default=DEFAULT_VIEW_GROUPS)
    p.add_argument("--sizes",       type=int, nargs="+",
                   default=DEFAULT_IMAGE_SIZES)
    p.add_argument("--lightings",   type=str, nargs="+",
                   default=DEFAULT_ILLUMINATIONS,
                   choices=["flat", "darker", "brighter"])

    return p.parse_args()


def preview(args: argparse.Namespace, objects: list[Path],
            prompt_single: str, prompt_multi: str) -> None:
    mode_desc = {
        "global": "PROMPT_GLOBAL for all groups — 1 call per group",
        "single": "PROMPT_SINGLE per image — N calls per group",
        "multi":  "PROMPT_MULTI for all groups — 1 call per group (fallback to single if n==1)",
        "auto":   "PROMPT_SINGLE if n_views==1, PROMPT_MULTI if n_views>1  [recommended]",
    }
    n_configs = len(args.sizes) * len(args.lightings) * len(args.view_groups)
    output_file = _exp_filename(OUTPUT_FILE, args.experiment_id)

    print("\n========== MOLMO MULTIVIEW RUNNER ==========")
    print(f"Renders root  : {args.renders_root}")
    print(f"Symmetry type : {args.symmetry_type}")
    print(f"Flow          : {args.flow}")
    print(f"Prompt mode   : {args.prompt_mode}")
    print(f"              : {mode_desc[args.prompt_mode]}")
    if args.flow != "a" and args.prompt_mode == "global":
        print(f"[warn] --flow {args.flow} has no effect under --prompt-mode global "
              f"(global mode doesn't support prompt overrides)")
    if args.experiment_id:
        print(f"Experiment ID : {args.experiment_id}")
        print(f"Prompt ID     : {args.prompt_id or '(default hardcoded)'}")
        print(f"Output file   : {output_file}")
    if args.max_objects:
        print(f"Max objects   : {args.max_objects}")
    print(f"GPU id/total  : {args.gpu_id} / {args.num_gpus}")
    print(f"Objects       : {len(objects)}")
    print(f"Sizes         : {args.sizes}")
    print(f"Lightings     : {args.lightings}")
    print(f"View groups   : {args.view_groups}")
    print(f"Combinations  : {n_configs} per object")
    print(f"(Existing JSON keys are skipped automatically)")
    print("============================================\n")
    if not args.yes:
        if input("Type 'OK' to start: ").strip() != "OK":
            print("Cancelled.")
            sys.exit(0)


def main() -> None:
    args = parse_args()

    if args.list_prompts:
        list_prompts()
        sys.exit(0)

    # Validate prompt_id early
    prompt_single, prompt_multi = PROMPT_SINGLE, PROMPT_MULTI
    if args.prompt_id:
        try:
            entry = get_prompt(args.prompt_id)
        except KeyError as e:
            print(f"[error] {e}")
            print("Run with --list-prompts to see available prompt IDs.")
            sys.exit(1)
        prompt_single = entry["single"]
        prompt_multi  = entry["multi"]

    if args.prompt_id == "axis_lit2_grid" and not args.grid_overlay:
        print("[error] --prompt-id axis_lit2_grid requires --grid-overlay "
              "(the prompt references grid cell labels that won't exist in the image otherwise).")
        sys.exit(1)
    if args.grid_overlay and args.prompt_id != "axis_lit2_grid":
        print(f"[warn] --grid-overlay is set but --prompt-id is {args.prompt_id!r}, "
              f"not 'axis_lit2_grid' -- the model will see a grid but the prompt won't mention it "
              f"unless you wrote it that way.")

    # Load Flow B/C prompts, failing fast if the expected files are missing
    describe_prompt = describe_and_point_prompt = None
    if args.flow == "b":
        try:
            describe_prompt = load_flow_prompt("describe")
        except FileNotFoundError as e:
            print(f"[error] {e}")
            sys.exit(1)
    elif args.flow == "c":
        try:
            point_variant = "axis" if args.symmetry_type == "axis_sym" else "plane"
            describe_and_point_prompt = load_flow_prompt(f"describe_and_point_{point_variant}")
        except FileNotFoundError as e:
            print(f"[error] {e}")
            sys.exit(1)

    symmetry_dir = Path(args.renders_root) / args.symmetry_type
    if not symmetry_dir.exists():
        print(f"[error] Not found: {symmetry_dir}")
        sys.exit(1)

    all_objects = sorted(d for d in symmetry_dir.iterdir() if d.is_dir())
    if not all_objects:
        print(f"[error] No object folders in {symmetry_dir}")
        sys.exit(1)

    if args.max_objects:
        all_objects = all_objects[:args.max_objects]

    objects = all_objects[args.gpu_id :: args.num_gpus]

    preview(args, objects, prompt_single, prompt_multi)

    token_log: dict[int, list[int]] = {}

    for obj_dir in tqdm(
        objects,
        desc=f"GPU {args.gpu_id}",
        unit="obj",
        dynamic_ncols=True,
        bar_format="{desc:<12} {bar:40} {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ):
        process_object(
            object_dir    = obj_dir,
            view_groups   = args.view_groups,
            sizes         = args.sizes,
            lightings     = args.lightings,
            prompt_mode   = args.prompt_mode,
            experiment_id = args.experiment_id,
            prompt_id     = args.prompt_id,
            prompt_single = prompt_single,
            prompt_multi  = prompt_multi,
            token_log     = token_log,
            flow          = args.flow,
            describe_prompt           = describe_prompt,
            describe_and_point_prompt = describe_and_point_prompt,
            grid_overlay  = args.grid_overlay,
        )

    print(f"\n[GPU {args.gpu_id}] Done.")

    if token_log:
        exp_tag   = f"_{args.experiment_id}" if args.experiment_id else ""
        tok_path  = symmetry_dir / f"token_counts{exp_tag}.txt"
        with open(tok_path, "w", encoding="utf-8") as f:
            f.write(f"# Token counts per n_views — experiment: {args.experiment_id or '(none)'}\n")
            f.write(f"# prompt_mode: {args.prompt_mode}  sizes: {args.sizes}  lightings: {args.lightings}\n")
            f.write(f"{'n_views':>8}  {'n_objects':>9}  {'mean':>8}  {'min':>8}  {'max':>8}\n")
            for nv in sorted(token_log):
                counts = token_log[nv]
                f.write(
                    f"{nv:>8}  {len(counts):>9}  "
                    f"{sum(counts)/len(counts):>8.0f}  "
                    f"{min(counts):>8}  {max(counts):>8}\n"
                )



if __name__ == "__main__":
    main()