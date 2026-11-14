# Variante "_subset" de qwen3vl_multiview_runner.py para renders parciales (--subset-n-views).
# Copia deliberada: el original lo usan pipelines en curso y no se modifica.
# Diferencia: las vistas se eligen sobre la secuencia Fibonacci completa
# ("fibonacci_total" en metadata_all.json), no sobre len(metadata).
"""
qwen3vl_multiview_runner.py
----------------------------
Batch multi-view pointing inference with Qwen3-VL-8B-Instruct
(Qwen/Qwen3-VL-8B-Instruct).

This module is deliberately self-contained: it duplicates (rather than
imports) the handful of pure helpers MolmoPointing/molmo_multiview_runner.py
also has (load_metadata, get_n_views_entries, load_results, save_results,
_exp_filename), so nothing under MolmoPointing/ needs to change, and this
folder has zero import-time coupling to it. The output JSON schema
(points_by_image, images_sent, n_points, ...) is kept identical on purpose,
so the existing no-mesh triangulation pipeline (Mapping/estimate_symmetry_no_mesh.py,
Mapping/evaluate.py) can read it the same way once pointed at the right
filename. See ../guia_instalacion_molmopoint_qwen3vl.md for the model-specific
background.

Output goes to qwen3vl_multiview[_<EXP>].json in the SAME render_dir Molmo2
writes molmo_multiview[_<EXP>].json to -- different filename, so the two never
collide or get skipped against each other.

What's different from Molmo2, and why
--------------------------------------
- Qwen3-VL answers with free text containing a JSON block (key "point_2d"), not
  Molmo2's "<points coords=...>" tag -- _parse_points() below tolerates the
  common ways that comes back malformed (```json fences, a bare dict instead of
  a list, trailing commas, text before/after the JSON).
- Qwen3-VL's coordinates are already relative 0-1000 (same scale
  molmo_multiview*.json uses), NOT pixels -- --qwen-scale-denominator exists only
  because the official denominator (1000 vs 999) is unconfirmed by the model
  card (guide section 3.3); verify with the single-point smoke test before
  changing it from the default.
- How Qwen3-VL assigns points to images in a multi-image call is undocumented.
  _parse_points() only trusts an explicit "image" key per point (validated
  against both 0- and 1-based indexing, like molmo_multiview_runner.py's
  parse_multi_coords() already does for Molmo2). Points with the key missing
  or out of range are dropped and counted in "n_points_dropped", NOT silently
  assigned to image 0 -- the prompt (--prompt-id) must explicitly ask for an
  "image" field per point for --prompt-mode multi/auto to recover anything on
  n_views > 1. If that proves unreliable, fall back to --prompt-mode single
  (one call per view) and document the exception, per the guide's checklist.

Deliberately NOT replicated here (out of scope per the guide): Molmo2's
--flow b/c semantic pre-pass and --grid-overlay.

Usage
-----
    CUDA_VISIBLE_DEVICES=0 python QwenPointing/qwen3vl_multiview_runner.py \\
        --renders-root ../data/renders \\
        --symmetry-type plane_sym \\
        --prompt-id plane_v05_1_qwen3vl \\
        --view-groups 6 14 26 \\
        --sizes 224 --lightings flat \\
        --prompt-mode auto

--prompt-id must name a registered variant from this folder's own
prompts_registry.py, i.e. files prompts/axis/<variant>_single.txt +
<variant>_multi.txt (or prompts/plane/...) under QwenPointing/ must already
exist -- this script does not ship default prompt text, since the guide
explicitly leaves prompt authoring out of scope.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).parent))
from prompts_registry import get_prompt, list_prompts

# Constants
MODEL_ID    = "Qwen/Qwen3-VL-8B-Instruct"
OUTPUT_FILE = "qwen3vl_multiview.json"

PROMPT_MODES = ("single", "multi", "auto")

DEFAULT_VIEW_GROUPS       = [1, 6, 14, 26, 42, 62, 86, 114]
DEFAULT_IMAGE_SIZES       = [224, 448, 1136]
DEFAULT_ILLUMINATIONS     = ["flat", "brighter", "darker"]
DEFAULT_MAX_NEW_TOKENS    = 3072   # 1024 truncated mid-JSON at n_views=26 in the smoke test (guide S3.3).
                                    # Raising this further does NOT fix every n_views=26 failure: in 2/5
                                    # plane objects the model fell into a genuine repetition loop (same
                                    # point_2d repeated under fabricated "image" indices beyond the real
                                    # image count) rather than running out of budget -- a model behavior
                                    # issue, not something max_new_tokens can paper over.
DEFAULT_SCALE_DENOMINATOR = 1000.0   # unconfirmed vs. 999 (guide S3.3) -- verify with smoke test


# Shared helpers (duplicated from MolmoPointing/molmo_multiview_runner.py by design -- see
# module docstring)
def _exp_filename(base: str, experiment_id: str | None) -> str:
    """
    Return base unchanged, or base_EXPID.ext when experiment_id is set.
    Args:
    - base: e.g. "qwen3vl_multiview.json"
    - experiment_id: e.g. "plane_v05_1_qwen3vl" or None
    Returns:
    - "qwen3vl_multiview.json" when experiment_id is None
    - "qwen3vl_multiview_plane_v05_1_qwen3vl.json" when experiment_id is set
    """
    if not experiment_id:
        return base
    dot = base.rfind(".")
    return f"{base[:dot]}_{experiment_id}{base[dot:]}"


def load_metadata(render_dir: Path) -> list[dict]:
    """
    Load metadata_all.json from render_dir. Raises FileNotFoundError if not found.
    Args:
    - render_dir: directory containing metadata_all.json
    Returns:
    - list of metadata entries, where each entry is a dict with keys:
      filename, index, azimuth, elevation, eye, R (3x3 rotation matrix), T (3D translation).
    """
    path = render_dir / "metadata_all.json"
    if not path.exists():
        raise FileNotFoundError(f"metadata_all.json not found: {render_dir}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_n_views_entries(metadata: list[dict], n_views: int) -> list[dict]:
    """
    Select n_views entries evenly spaced across the full metadata sequence
    (via linspace over index, to guarantee uniform angular coverage -- same
    rationale as MolmoPointing/molmo_multiview_runner.py's version).
    Args:
    - metadata: list of metadata entries (dicts with "index" key)
    - n_views: number of views to select
    Returns:
    - list of n_views metadata entries with uniform index spacing, sorted by index
    """
    # Size of the full Fibonacci sequence. Partial renders (export_fibonacci_views.py
    # --subset-n-views) store it per entry as "fibonacci_total", so the indices picked
    # here are the same ones a full render would give; old full renders lack the
    # field and fall back to len(metadata), the previous behaviour.
    total = max(int(m.get("fibonacci_total", len(metadata))) for m in metadata)
    if n_views >= total:
        wanted = set(range(total))
    else:
        wanted = {int(round(i)) for i in np.linspace(0, total - 1, n_views)}
    entries = [m for m in metadata if m["index"] in wanted]
    missing = sorted(wanted - {m["index"] for m in entries})
    if missing:
        raise ValueError(
            f"n_views={n_views} needs Fibonacci indices {missing} (of {total}) that were "
            f"not rendered; this render only has {len(metadata)} views -- re-render with "
            f"--subset-n-views including {n_views}"
        )
    return sorted(entries, key=lambda e: e["index"])


def load_results(json_path: Path) -> dict:
    """Load existing results from json_path, or return empty dict if file doesn't exist."""
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(json_path: Path, data: dict) -> None:
    """Save results to json_path, creating parent directories if needed."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# Model singleton
_processor = None
_model     = None


def get_model(
    revision:   str | None = None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    use_flash_attention: bool = False,
):
    """Load the processor and model if not already loaded. Returns (processor, model)."""
    global _processor, _model
    if _processor is None or _model is None:
        print(f"[model] Loading {MODEL_ID} ...")
        processor_kwargs = {}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels
        _processor = AutoProcessor.from_pretrained(
            MODEL_ID, revision=revision, **processor_kwargs,
        )

        model_kwargs = {"dtype": "auto", "device_map": "auto", "revision": revision}
        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        _model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, **model_kwargs)
        _model.eval()
        print("[model] Ready.")
    return _processor, _model


# Low-level model call
def _call_model(images: list[Image.Image], prompt: str, max_new_tokens: int) -> tuple[str, int]:
    """
    Single Qwen3-VL call with 1..N images. Images are listed before the text
    prompt (official cookbook convention, guide section 3.2). Returns
    (raw_text, n_input_tokens).
    """
    processor, model = get_model()

    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    n_input = inputs.input_ids.size(1)

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    output_ids = output_ids[:, n_input:]
    text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    del inputs, output_ids
    return text, n_input


# JSON parsing -- tolerant of fenced blocks, bare dicts, trailing commas
_JSON_FENCE_RE     = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _extract_json_payload(text: str) -> str:
    """
    Pulls the most likely JSON payload out of a free-text model response:
    prefers the content of a ```json ... ``` fence if present, otherwise the
    substring between the first '[' or '{' and the last matching ']' or '}'.
    """
    fence = _JSON_FENCE_RE.search(text)
    if fence:
        return fence.group(1).strip()

    start_candidates = [i for i in (text.find("["), text.find("{")) if i != -1]
    if not start_candidates:
        return text.strip()
    start = min(start_candidates)
    end_bracket = "]" if text[start] == "[" else "}"
    end = text.rfind(end_bracket)
    if end == -1 or end < start:
        return text.strip()
    return text[start:end + 1].strip()


def _parse_points(text: str) -> tuple[list[dict], str | None]:
    """
    Parses Qwen3-VL's JSON pointing output into a flat list of
    {"point_2d": [x, y], "image": int | None} dicts.

    Tolerates: a ```json fence, a bare dict instead of a list of dicts,
    trailing commas. Returns (points, parse_error) -- points is [] and
    parse_error is a short description when parsing fails outright, so the
    caller can log the raw text for auditing (guide section 4) instead of
    silently losing the failure.
    """
    payload = _extract_json_payload(text)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        payload_fixed = _TRAILING_COMMA_RE.sub(r"\1", payload)
        try:
            data = json.loads(payload_fixed)
        except json.JSONDecodeError as e:
            return [], f"JSONDecodeError: {e}"

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return [], f"Unexpected JSON top-level type: {type(data).__name__}"

    points = []
    for item in data:
        if not isinstance(item, dict):
            continue
        image_idx = item.get("image")
        # The item's own point_2d, plus any point_2d nested in a dict-valued field --
        # observed in practice: instead of two flat list entries, the model sometimes
        # nests a second point inside the first, e.g. {"image": 0, "label": "top",
        # "point_2d": [...], "bottom": {"point_2d": [...]}}. Both get the same
        # "image" (the nested point has no index of its own to read).
        for candidate in [item] + [v for v in item.values() if isinstance(v, dict)]:
            point_2d = candidate.get("point_2d")
            if not isinstance(point_2d, (list, tuple)) or len(point_2d) != 2:
                continue
            points.append({"point_2d": [float(point_2d[0]), float(point_2d[1])], "image": image_idx})

    if not points:
        return [], "No valid point_2d entries found"
    return points, None


def points_to_by_image(
    points: list[dict], n_images: int, scale_denominator: float,
) -> tuple[dict[str, list[dict]], int]:
    """
    Convert parsed {"point_2d", "image"} dicts into the pipeline's
    {"<img_idx 0-based>": [{"obj_id", "x", "y"}, ...]} schema.

    For a single-image call (n_images == 1) the "image" key is irrelevant --
    everything goes to image 0, matching run_single_mode()'s own per-call
    semantics. For multi-image calls, a point is kept only if "image" is
    present and resolves (0- or 1-based) to a valid index in [0, n_images) --
    see the module docstring for why this doesn't guess when the key is
    missing. obj_id is assigned as a per-image running count (1, 2, ...)
    since Qwen3-VL's JSON has no equivalent identifier.

    Returns (points_by_image, n_dropped).
    """
    result: dict[str, list[dict]] = {}
    dropped = 0
    counters: dict[str, int] = {}

    for pt in points:
        x_1000 = pt["point_2d"][0] / scale_denominator * 1000.0
        y_1000 = pt["point_2d"][1] / scale_denominator * 1000.0

        if n_images == 1:
            img_idx = 0
        else:
            raw_idx = pt["image"]
            if raw_idx is None:
                dropped += 1
                continue
            try:
                raw_idx = int(raw_idx)
            except (TypeError, ValueError):
                dropped += 1
                continue
            if 0 <= raw_idx < n_images:
                img_idx = raw_idx
            elif 0 <= raw_idx - 1 < n_images:
                img_idx = raw_idx - 1
            else:
                dropped += 1
                continue

        key = str(img_idx)
        counters[key] = counters.get(key, 0) + 1
        result.setdefault(key, []).append(
            {"obj_id": counters[key], "x": x_1000, "y": y_1000}
        )

    return result, dropped


# Inference modes
def run_single_mode(
    images: list[Image.Image], prompt: str, max_new_tokens: int, scale_denominator: float,
) -> dict:
    """One call per image. raw_output stored as a list (one entry per image)."""
    raw_outputs:   list[str] = []
    points_by_img: dict[str, list[dict]] = {}
    parse_errors:  list[str] = []
    total_tokens  = 0
    total_dropped = 0

    for i, img in enumerate(images):
        raw, n_tok = _call_model([img], prompt, max_new_tokens)
        raw_outputs.append(raw)
        total_tokens += n_tok

        points, err = _parse_points(raw)
        if err:
            parse_errors.append(f"image {i}: {err}")
        by_img, dropped = points_to_by_image(points, n_images=1, scale_denominator=scale_denominator)
        total_dropped += dropped
        if "0" in by_img:
            points_by_img[str(i)] = by_img["0"]

    return {
        "prompt_used":             prompt,
        "raw_output":               raw_outputs,
        "points_by_image":          points_by_img,
        "n_input_tokens":           total_tokens,
        "n_points_dropped":         total_dropped,
        "raw_output_parse_errors":  parse_errors or None,
    }


def run_multi(
    images: list[Image.Image], prompt: str, max_new_tokens: int, scale_denominator: float,
) -> dict:
    """One call with all images."""
    raw, n_tok = _call_model(images, prompt, max_new_tokens)
    points, err = _parse_points(raw)
    points_by_img, dropped = points_to_by_image(
        points, n_images=len(images), scale_denominator=scale_denominator,
    )
    return {
        "prompt_used":             prompt,
        "raw_output":              raw,
        "points_by_image":         points_by_img,
        "n_input_tokens":          n_tok,
        "n_points_dropped":        dropped,
        "raw_output_parse_errors": [err] if err else None,
    }


def run_inference(
    images:            list[Image.Image],
    prompt_mode:       str,
    prompt_single:     str,
    prompt_multi:      str,
    max_new_tokens:    int,
    scale_denominator: float,
) -> dict:
    """
    single  -> run_single_mode, 1 call  per image [documented fallback for
               unreliable multi-image indexing, guide S3.3/S6]
    multi   -> run_multi,       1 call  for all images (fallback to single if n==1)
    auto    -> single if n==1, multi if n>1  [parity with Molmo2's default]
    """
    n = len(images)
    if prompt_mode == "single":
        return run_single_mode(images, prompt_single, max_new_tokens, scale_denominator)
    if prompt_mode in ("multi", "auto"):
        return run_single_mode(images, prompt_single, max_new_tokens, scale_denominator) if n == 1 \
               else run_multi(images, prompt_multi, max_new_tokens, scale_denominator)
    raise ValueError(f"Unknown prompt_mode: {prompt_mode!r}")


# Per-object processor
def process_object(
    object_dir:        Path,
    view_groups:       list[int],
    sizes:             list[int],
    lightings:         list[str],
    prompt_mode:       str,
    prompt_single:     str,
    prompt_multi:      str,
    experiment_id:     str | None,
    prompt_id:         str | None,
    max_new_tokens:    int,
    scale_denominator: float,
    revision:          str | None,
    min_pixels:        int | None,
    max_pixels:        int | None,
    token_log:         dict | None = None,
) -> None:
    """
    Process one object across all (size, illumination, n_views) combinations.
    Skips any (size, illumination, n_views) already present in the output JSON.
    """
    output_file = _exp_filename(OUTPUT_FILE, experiment_id)

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

            pending = [n for n in view_groups if str(n) not in results]
            if not pending:
                continue

            for n_views in pending:
                entries = get_n_views_entries(metadata, n_views)
                if not entries:
                    continue

                images = [
                    Image.open(render_dir / e["filename"]).convert("RGB")
                    for e in entries
                ]

                inference = run_inference(
                    images, prompt_mode, prompt_single, prompt_multi,
                    max_new_tokens, scale_denominator,
                )
                points_by_img = inference["points_by_image"]
                n_pts         = sum(len(v) for v in points_by_img.values())
                n_tok         = inference.get("n_input_tokens", 0)

                if token_log is not None:
                    token_log.setdefault(n_views, []).append(n_tok)

                results[str(n_views)] = {
                    "model_id":              MODEL_ID,
                    "model_revision":        revision,
                    "experiment_id":         experiment_id,
                    "prompt_id":             prompt_id,
                    "prompt_mode":           prompt_mode,
                    "scale_denominator":     scale_denominator,
                    "min_pixels":            min_pixels,
                    "max_pixels":            max_pixels,
                    "prompt_used":           inference["prompt_used"],
                    "raw_output":            inference["raw_output"],
                    "raw_output_parse_errors": inference.get("raw_output_parse_errors"),
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
                    "n_points_dropped":     inference.get("n_points_dropped", 0),
                    "n_images_with_points": len(points_by_img),
                    "n_input_tokens":       n_tok,
                }

                # Write after every n_views to survive interruptions
                save_results(json_path, results)


# CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch Qwen3-VL-8B-Instruct multi-view pointing — cumulative JSON output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--renders-root",  required=True,
                   help="Root folder produced by data_render.py")
    p.add_argument("--symmetry-type", required=True,
                   choices=["axis_sym", "plane_sym"])

    p.add_argument("--prompt-mode", default="auto", choices=PROMPT_MODES,
                   help=(
                       "single = one call per image (N calls/group) -- the documented "
                       "fallback if multi-image indexing proves unreliable; "
                       "multi  = one call for all images (fallback to single when n_views==1); "
                       "auto   = single if n_views==1, multi otherwise [default, parity with Molmo2]"
                   ))

    p.add_argument("--prompt-id", required=True,
                   help=(
                       "Prompt variant from THIS folder's prompts_registry.py, e.g. "
                       "plane_v05_1_qwen3vl. For --prompt-mode multi/auto with n_views>1, the "
                       "prompt MUST ask the model to include an explicit \"image\" index per "
                       "point in its JSON -- otherwise those points are dropped (see module "
                       "docstring). Requires prompts/{axis,plane}/<variant>_single.txt + "
                       "<variant>_multi.txt under QwenPointing/ to already exist -- this "
                       "script ships no default prompt text."
                   ))
    p.add_argument("--experiment-id", default=None,
                   help=(
                       "Output filename becomes qwen3vl_multiview_<ID>.json instead of "
                       "qwen3vl_multiview.json (always separate from Molmo2's "
                       "molmo_multiview*.json -- only needed here to separate multiple "
                       "Qwen3-VL prompt experiments from each other)."
                   ))
    p.add_argument("--revision", default=None,
                   help="Pin a specific Qwen/Qwen3-VL-8B-Instruct revision (commit hash). "
                        "Recorded in the output JSON for reproducibility (guide section 1.3/6).")
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--scale-denominator", type=float, default=DEFAULT_SCALE_DENOMINATOR,
                   help="Divisor Qwen3-VL's point_2d values are relative to. UNCONFIRMED "
                        "(1000 vs 999, guide S3.3) -- verify with the smoke test first.")
    p.add_argument("--min-pixels", type=int, default=None,
                   help="Fixed lower bound on per-image token resolution, passed to the "
                        "processor. Leave unset for the model's own default, but once chosen "
                        "for a run, keep it fixed and record it (guide S3.3) -- it changes "
                        "tokens/image and thus the token-budget comparison across models.")
    p.add_argument("--max-pixels", type=int, default=None,
                   help="Fixed upper bound on per-image token resolution. See --min-pixels.")
    p.add_argument("--flash-attention", action="store_true",
                   help="Use attn_implementation='flash_attention_2' (requires flash-attn "
                        "installed and GPU support, guide section 1.2).")

    p.add_argument("--max-objects", type=int, default=None,
                   help="Limit to the first N objects (sorted order). Useful for subset experiments.")
    p.add_argument("--list-prompts", action="store_true",
                   help="List all registered prompts and exit.")
    p.add_argument("--yes", "-y", action="store_true",
                   help="Skip the confirmation prompt (useful for automated loops).")

    p.add_argument("--gpu-id",   type=int, default=0)
    p.add_argument("--num-gpus", type=int, default=1)

    p.add_argument("--view-groups", type=int, nargs="+", default=DEFAULT_VIEW_GROUPS)
    p.add_argument("--sizes",       type=int, nargs="+", default=DEFAULT_IMAGE_SIZES)
    p.add_argument("--lightings",   type=str, nargs="+", default=DEFAULT_ILLUMINATIONS,
                   choices=["flat", "darker", "brighter"])

    return p.parse_args()


def preview(args: argparse.Namespace, objects: list[Path]) -> None:
    n_configs   = len(args.sizes) * len(args.lightings) * len(args.view_groups)
    output_file = _exp_filename(OUTPUT_FILE, args.experiment_id)

    print("\n========== QWEN3-VL MULTIVIEW RUNNER ==========")
    print(f"Model          : {MODEL_ID}  (revision={args.revision or 'latest'})")
    print(f"Renders root   : {args.renders_root}")
    print(f"Symmetry type  : {args.symmetry_type}")
    print(f"Prompt mode    : {args.prompt_mode}")
    if args.prompt_mode in ("multi", "auto") and any(v > 1 for v in args.view_groups):
        print(f"               : relies on an explicit \"image\" key per point in the "
              f"model's JSON for n_views>1 -- see --prompt-id help if points get dropped")
    print(f"Prompt ID      : {args.prompt_id}")
    print(f"Output file    : {output_file}")
    print(f"Scale denom.   : {args.scale_denominator}  (unconfirmed -- see guide section 3.3)")
    print(f"Max new tokens : {args.max_new_tokens}")
    print(f"min/max pixels : {args.min_pixels} / {args.max_pixels}")
    if args.max_objects:
        print(f"Max objects    : {args.max_objects}")
    print(f"GPU id/total   : {args.gpu_id} / {args.num_gpus}")
    print(f"Objects        : {len(objects)}")
    print(f"Sizes          : {args.sizes}")
    print(f"Lightings      : {args.lightings}")
    print(f"View groups    : {args.view_groups}")
    print(f"Combinations   : {n_configs} per object")
    print(f"(Existing JSON keys are skipped automatically)")
    print("=================================================\n")
    if not args.yes:
        if input("Type 'OK' to start: ").strip() != "OK":
            print("Cancelled.")
            sys.exit(0)


def main() -> None:
    args = parse_args()

    if args.list_prompts:
        list_prompts()
        sys.exit(0)

    try:
        entry = get_prompt(args.prompt_id)
    except KeyError as e:
        print(f"[error] {e}")
        print("Run with --list-prompts to see available prompt IDs.")
        sys.exit(1)
    prompt_single, prompt_multi = entry["single"], entry["multi"]

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

    preview(args, objects)

    # Load the model eagerly (rather than lazily on first _call_model) so
    # --min-pixels/--max-pixels/--flash-attention are applied before any call.
    get_model(
        revision=args.revision, min_pixels=args.min_pixels, max_pixels=args.max_pixels,
        use_flash_attention=args.flash_attention,
    )

    token_log: dict[int, list[int]] = {}

    for obj_dir in tqdm(
        objects,
        desc=f"GPU {args.gpu_id}",
        unit="obj",
        dynamic_ncols=True,
        bar_format="{desc:<12} {bar:40} {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ):
        process_object(
            object_dir        = obj_dir,
            view_groups       = args.view_groups,
            sizes             = args.sizes,
            lightings         = args.lightings,
            prompt_mode       = args.prompt_mode,
            prompt_single     = prompt_single,
            prompt_multi      = prompt_multi,
            experiment_id     = args.experiment_id,
            prompt_id         = args.prompt_id,
            max_new_tokens    = args.max_new_tokens,
            scale_denominator = args.scale_denominator,
            revision          = args.revision,
            min_pixels        = args.min_pixels,
            max_pixels        = args.max_pixels,
            token_log         = token_log,
        )

    print(f"\n[GPU {args.gpu_id}] Done.")

    if token_log:
        exp_tag  = f"_{args.experiment_id}" if args.experiment_id else ""
        tok_path = symmetry_dir / f"token_counts_qwen3vl{exp_tag}.txt"
        with open(tok_path, "w", encoding="utf-8") as f:
            f.write(f"# Token counts per n_views — experiment: {args.experiment_id or '(none)'}\n")
            f.write(f"# model: {MODEL_ID}  prompt_mode: {args.prompt_mode}  "
                    f"sizes: {args.sizes}  lightings: {args.lightings}\n")
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
