"""
molmopoint_multiview_runner.py
-------------------------------
Batch multi-view pointing inference with MolmoPoint-8B (allenai/MolmoPoint-8B).

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

Output goes to molmopoint_multiview[_<EXP>].json in the SAME render_dir Molmo2
writes molmo_multiview[_<EXP>].json to -- different filename, so the two never
collide or get skipped against each other.

What's different from Molmo2, and why
--------------------------------------
- MolmoPoint doesn't write coordinates as text: each point is three special
  tokens (<PATCH>/<SUBPATCH>/<LOCATION>). Decoding them requires the pointing
  metadata the processor attaches to the chat-template inputs (token_pooling,
  subpatch_mapping, image_sizes) plus model.extract_image_points() -- there is
  no raw "<points coords=...>" string to regex, unlike Molmo2's text parsing.
- extract_image_points() returns coordinates in PIXELS of the original image,
  not the 0-1000 scale molmo_multiview*.json uses elsewhere in the pipeline.
  points_to_by_image() below does that conversion (guide section 4) before
  anything is written to disk, so downstream code never has to know which
  model produced a given file.
- Whether `image_num` (the model's per-point view index) is 0- or 1-based is NOT
  documented in the model card. --image-num-base defaults to 0; run the two-image
  smoke test in the guide (section 5, step 2) before trusting it, and flip to 1
  with --image-num-base if that test shows otherwise.
- ONE PROMPT CAN ONLY ASK FOR ONE TARGET. Empirically (tested live against the
  model on real renders, not just the model card): asking MolmoPoint to point
  to two different things in a single call -- "point to the top pole and the
  bottom pole", "point to: X, Y", two sentences back to back -- makes it answer
  with free-text description instead of emitting ANY pointing tokens (or, in
  one observed case, pointing tokens for only the first target and prose for
  the second). Even a single target degrades to prose if the sentence that
  asks for it carries a trailing qualifier ("...pole of the rotation axis",
  "topmost point on the plane's trace") or any preceding "use the SAME axis
  across views" consistency framing -- cutting the point-to clause down to a
  bare "point to the top pole." (short noun phrase, no trailing qualifier, no
  "and") is what made it reliably emit real <PATCH>/<SUBPATCH>/<LOCATION>
  tokens. As a result this runner makes TWO separate model calls per
  n_views-group -- one asking only for the top/first target, one asking only
  for the bottom/second target -- then merges their points, tagging the
  top-call's points obj_id=1 and the bottom-call's obj_id=2 (the model's own
  object_id is just a per-call running index, not a stable role marker, so it
  is discarded in favor of this call-provenance tag). This is why
  prompts_registry.py here requires FOUR prompt files per variant
  (single_top/single_bottom/multi_top/multi_bottom), not two -- and why the
  guide's own "use the SAME global axis in every image" cross-view-consistency
  instruction could not be kept: adding it to the prompt was tested and broke
  pointing entirely (zero points), so that explicit consistency ask had to be
  dropped. A consequence worth knowing before comparing against Molmo2: a
  single call can occasionally return more than one point for the same image
  (the model finding more than one candidate match for an ambiguous object,
  e.g. a lattice/grid-shaped object with several locally-"top" features) --
  points_to_by_image() does not cap this, so some images may end up with more
  than the clean 2-points-per-image Molmo2's own axis_v04/plane_v05_1 always
  produce.

Deliberately NOT replicated here (out of scope per the guide, which only asks to
repeat axis_v04 / plane_v05_1): molmo_multiview_runner.py's --flow b/c semantic
pre-pass and --grid-overlay. Both are Molmo2-specific experiments, not part of
the base direct-pointing protocol this guide describes.

Usage
-----
    CUDA_VISIBLE_DEVICES=1 python MolmoPointPointing/molmopoint_multiview_runner.py \\
        --renders-root ../data/renders \\
        --symmetry-type axis_sym \\
        --prompt-id axis_v04_molmopoint \\
        --view-groups 6 14 26 \\
        --sizes 224 --lightings flat \\
        --prompt-mode auto

--prompt-id must name a registered variant from this folder's own
prompts_registry.py, i.e. the four files prompts/axis/<variant>_single_top.txt
+ _single_bottom.txt + _multi_top.txt + _multi_bottom.txt (or prompts/plane/...)
under MolmoPointPointing/ must already exist -- this script does not ship
default prompt text, since the guide explicitly leaves prompt authoring out of
scope.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).parent))
from prompts_registry import get_prompt, list_prompts

# Constants
MODEL_ID    = "allenai/MolmoPoint-8B"
OUTPUT_FILE = "molmopoint_multiview.json"

PROMPT_MODES = ("single", "multi", "auto")

DEFAULT_VIEW_GROUPS    = [1, 6, 14, 26, 42, 62, 86, 114]
DEFAULT_IMAGE_SIZES    = [224, 448, 1136]
DEFAULT_ILLUMINATIONS  = ["flat", "brighter", "darker"]
DEFAULT_MAX_NEW_TOKENS = 400   # ~3 tokens/point (guide S2.3); raise for many views/points


# Shared helpers (duplicated from MolmoPointing/molmo_multiview_runner.py by design -- see
# module docstring)
def _exp_filename(base: str, experiment_id: str | None) -> str:
    """
    Return base unchanged, or base_EXPID.ext when experiment_id is set.
    Args:
    - base: e.g. "molmopoint_multiview.json"
    - experiment_id: e.g. "axis_v04_molmopoint" or None
    Returns:
    - "molmopoint_multiview.json" when experiment_id is None
    - "molmopoint_multiview_axis_v04_molmopoint.json" when experiment_id is set
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
    total = len(metadata)
    if n_views >= total:
        return sorted(metadata, key=lambda e: e["index"])
    indices = {int(round(i)) for i in np.linspace(0, total - 1, n_views)}
    entries = [m for m in metadata if m["index"] in indices]
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


def get_model(revision: str | None = None):
    """
    Load the processor and model if not already loaded. Returns (processor, model).

    dtype is forced to bfloat16 rather than "auto" (the guide's example) because
    allenai/MolmoPoint-8B's config.json declares dtype float32 -- "auto" loads the
    full fp32 checkpoint (~2x the memory), which alone nearly fills a 24GB GPU and
    OOMs as soon as generate() needs any activation memory. device_map is pinned
    to {"": 0} (the single visible GPU) rather than "auto": with device_map="auto"
    deciding to CPU-offload part of the vision tower, this model's custom remote
    code raises `NotImplementedError: Cannot copy out of meta tensor; no data!` on
    the first forward pass -- an offload/dispatch incompatibility in its modeling
    code, not something fixable from the caller's side.
    """
    global _processor, _model
    if _processor is None or _model is None:
        print(f"[model] Loading {MODEL_ID} ...")
        _processor = AutoProcessor.from_pretrained(
            MODEL_ID, trust_remote_code=True, padding_side="left", revision=revision,
        )
        _model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": 0}, revision=revision,
        )
        _model.eval()
        print("[model] Ready.")
    return _processor, _model


# Low-level model call
def _call_model(
    images: list[Image.Image], prompt: str, max_new_tokens: int,
) -> tuple[list, str, int]:
    """
    Single MolmoPoint call with 1..N images.

    skip_special_tokens=False and the explicit logits_processor are both required
    (guide section 2.3) -- without them the <PATCH>/<SUBPATCH>/<LOCATION> tokens
    that encode points either get stripped before decoding or aren't generated
    correctly in the first place.

    Returns:
    - points: list of [object_id, image_num, x_px, y_px] rows (model.extract_image_points
      output, pixel coordinates in the ORIGINAL image, not the 0-1000 pipeline scale)
    - raw_text: the decoded text (including special tokens) -- kept for the output
      JSON's "raw_output" field, same role as Molmo2's raw decoded text
    - n_input_tokens: input token count, for the same token-accounting the pipeline
      already does for Molmo2
    """
    processor, model = get_model()

    content = [{"type": "text", "text": prompt}]
    content += [{"type": "image", "image": img} for img in images]
    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        padding=True,
        return_pointing_metadata=True,
    )
    metadata = inputs.pop("metadata")
    inputs   = {k: v.to(model.device) for k, v in inputs.items()}
    n_input  = inputs["input_ids"].size(1)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output_ids = model.generate(
            **inputs,
            logits_processor=model.build_logit_processor_from_inputs(inputs),
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated = output_ids[:, n_input:]
    raw_text  = processor.post_process_image_text_to_text(
        generated, skip_special_tokens=False, clean_up_tokenization_spaces=False,
    )[0]

    points = model.extract_image_points(
        raw_text, metadata["token_pooling"], metadata["subpatch_mapping"], metadata["image_sizes"],
    )

    del inputs, output_ids, generated
    torch.cuda.empty_cache()
    return list(points), raw_text, n_input


# Coordinate conversion -- pixels -> pipeline's 0-1000 scale (guide section 4)
def points_to_by_image(
    points: list, images: list[Image.Image], image_num_base: int, obj_id: int,
) -> dict[str, list[dict]]:
    """
    Convert MolmoPoint's [object_id, image_num, x_px, y_px] rows into the
    pipeline's {"<img_idx 0-based>": [{"obj_id", "x", "y"}, ...]} schema, with
    x/y rescaled from pixels to the 0-1000 scale molmo_multiview*.json already
    uses for Molmo2 (a_escala_molmo2 in the guide): x_1000 = x_px / width * 1000.

    image_num_base is subtracted from each row's image_num before use -- 0 if
    MolmoPoint's image_num is already 0-based, 1 if it's 1-based (see the
    module docstring: unverified, confirm with the smoke test before changing
    the --image-num-base default).

    obj_id overrides the model's own per-row object_id (1 for a top-target
    call, 2 for a bottom-target call -- see the module docstring on why every
    call here asks for exactly one target): the model's object_id is just a
    running index over however many matches it found IN THIS CALL, not a
    stable role marker, so using it directly would conflate "first point
    found" with "top point" only by coincidence. A call that returns more
    than one point for the same image (ambiguous object, see module
    docstring) means more than one entry sharing this same obj_id for that
    image -- not deduplicated, since the no-mesh pipeline reads
    points_by_image as a flat per-image list rather than requiring unique
    obj_ids.

    Rows whose resulting index falls outside [0, len(images)) are dropped
    silently, matching molmo_multiview_runner.py's existing behavior for
    out-of-range image indices.
    """
    result: dict[str, list[dict]] = {}
    for row in points:
        _model_obj_id, image_num, x_px, y_px = row[0], row[1], row[2], row[3]
        img_idx = int(image_num) - image_num_base
        if img_idx < 0 or img_idx >= len(images):
            continue
        width, height = images[img_idx].size
        x_1000 = float(x_px) / width * 1000.0
        y_1000 = float(y_px) / height * 1000.0
        result.setdefault(str(img_idx), []).append(
            {"obj_id": obj_id, "x": x_1000, "y": y_1000}
        )
    return result


def _merge_points_by_image(
    a: dict[str, list[dict]], b: dict[str, list[dict]],
) -> dict[str, list[dict]]:
    """Union two points_by_image dicts, concatenating lists on shared image keys."""
    merged = {k: list(v) for k, v in a.items()}
    for k, v in b.items():
        merged.setdefault(k, []).extend(v)
    return merged


# Inference modes -- each issues TWO calls (top target, then bottom target),
# never one call asking for both (see module docstring: that reliably breaks
# pointing entirely).
def run_single_mode(
    images: list[Image.Image],
    prompt_top: str, prompt_bottom: str,
    max_new_tokens: int, image_num_base: int,
) -> dict:
    """One top call + one bottom call PER IMAGE. raw_output stored as parallel lists."""
    raw_top:    list[str] = []
    raw_bottom: list[str] = []
    points_by_img: dict[str, list[dict]] = {}
    total_tokens = 0

    for i, img in enumerate(images):
        pts_top, text_top, n_tok_top = _call_model([img], prompt_top, max_new_tokens)
        pts_bot, text_bot, n_tok_bot = _call_model([img], prompt_bottom, max_new_tokens)
        raw_top.append(text_top)
        raw_bottom.append(text_bot)
        total_tokens += n_tok_top + n_tok_bot

        by_top = points_to_by_image(pts_top, [img], image_num_base, obj_id=1)
        by_bot = points_to_by_image(pts_bot, [img], image_num_base, obj_id=2)
        merged = _merge_points_by_image(by_top, by_bot)
        if "0" in merged:
            points_by_img[str(i)] = merged["0"]

    return {
        "prompt_used":     {"top": prompt_top, "bottom": prompt_bottom},
        "raw_output":      {"top": raw_top, "bottom": raw_bottom},
        "points_by_image": points_by_img,
        "n_input_tokens":  total_tokens,
    }


def run_multi(
    images: list[Image.Image],
    prompt_top: str, prompt_bottom: str,
    max_new_tokens: int, image_num_base: int,
) -> dict:
    """One top call + one bottom call, each covering ALL images at once."""
    pts_top, text_top, n_tok_top = _call_model(images, prompt_top, max_new_tokens)
    pts_bot, text_bot, n_tok_bot = _call_model(images, prompt_bottom, max_new_tokens)

    by_top = points_to_by_image(pts_top, images, image_num_base, obj_id=1)
    by_bot = points_to_by_image(pts_bot, images, image_num_base, obj_id=2)
    points_by_img = _merge_points_by_image(by_top, by_bot)

    return {
        "prompt_used":     {"top": prompt_top, "bottom": prompt_bottom},
        "raw_output":      {"top": text_top, "bottom": text_bot},
        "points_by_image": points_by_img,
        "n_input_tokens":  n_tok_top + n_tok_bot,
    }


def run_inference(
    images: list[Image.Image],
    prompt_mode: str,
    prompt_single_top: str, prompt_single_bottom: str,
    prompt_multi_top: str, prompt_multi_bottom: str,
    max_new_tokens: int,
    image_num_base: int,
) -> dict:
    """
    single  -> run_single_mode, 1 top call + 1 bottom call  per image
    multi   -> run_multi,       1 top call + 1 bottom call  for all images (fallback to single if n==1)
    auto    -> single if n==1, multi if n>1  [recommended, same convention as Molmo2]
    """
    n = len(images)
    if prompt_mode == "single":
        return run_single_mode(images, prompt_single_top, prompt_single_bottom, max_new_tokens, image_num_base)
    if prompt_mode in ("multi", "auto"):
        return run_single_mode(images, prompt_single_top, prompt_single_bottom, max_new_tokens, image_num_base) \
               if n == 1 else \
               run_multi(images, prompt_multi_top, prompt_multi_bottom, max_new_tokens, image_num_base)
    raise ValueError(f"Unknown prompt_mode: {prompt_mode!r}")


# Per-object processor
def process_object(
    object_dir:            Path,
    view_groups:           list[int],
    sizes:                 list[int],
    lightings:             list[str],
    prompt_mode:           str,
    prompt_single_top:     str,
    prompt_single_bottom:  str,
    prompt_multi_top:      str,
    prompt_multi_bottom:   str,
    experiment_id:         str | None,
    prompt_id:             str | None,
    max_new_tokens:        int,
    image_num_base:        int,
    revision:              str | None,
    token_log:             dict | None = None,
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
                    images, prompt_mode,
                    prompt_single_top, prompt_single_bottom,
                    prompt_multi_top, prompt_multi_bottom,
                    max_new_tokens, image_num_base,
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
                    "image_num_base":        image_num_base,
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

                # Write after every n_views to survive interruptions
                save_results(json_path, results)


# CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch MolmoPoint-8B multi-view pointing — cumulative JSON output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--renders-root",  required=True,
                   help="Root folder produced by data_render.py")
    p.add_argument("--symmetry-type", required=True,
                   choices=["axis_sym", "plane_sym"])

    p.add_argument("--prompt-mode", default="auto", choices=PROMPT_MODES,
                   help=(
                       "single = one call per image (N calls/group); "
                       "multi  = one call for all images (fallback to single when n_views==1); "
                       "auto   = single if n_views==1, multi otherwise [recommended]"
                   ))

    p.add_argument("--prompt-id", required=True,
                   help=(
                       "Prompt variant from THIS folder's prompts_registry.py, e.g. "
                       "axis_v04_molmopoint. Requires prompts/{axis,plane}/<variant>_single.txt "
                       "+ <variant>_multi.txt under MolmoPointPointing/ to already exist -- "
                       "this script ships no default prompt text. Run "
                       "`python MolmoPointPointing/prompts_registry.py` to list what's registered."
                   ))
    p.add_argument("--experiment-id", default=None,
                   help=(
                       "Output filename becomes molmopoint_multiview_<ID>.json instead of "
                       "molmopoint_multiview.json (always separate from Molmo2's "
                       "molmo_multiview*.json -- only needed here to separate multiple "
                       "MolmoPoint prompt experiments from each other)."
                   ))
    p.add_argument("--revision", default=None,
                   help="Pin a specific allenai/MolmoPoint-8B revision (commit hash). "
                        "Recorded in the output JSON for reproducibility (guide section 1.3/6).")
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                   help="Raise if outputs look truncated with many views/points (guide S2.3).")
    p.add_argument("--image-num-base", type=int, default=0, choices=[0, 1],
                   help="Whether MolmoPoint's image_num is 0- or 1-based. UNVERIFIED default "
                        "-- confirm with the two-image smoke test (guide section 5) first.")

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

    print("\n========== MOLMOPOINT MULTIVIEW RUNNER ==========")
    print(f"Model         : {MODEL_ID}  (revision={args.revision or 'latest'})")
    print(f"Renders root  : {args.renders_root}")
    print(f"Symmetry type : {args.symmetry_type}")
    print(f"Prompt mode   : {args.prompt_mode}")
    print(f"Prompt ID     : {args.prompt_id}")
    print(f"Output file   : {output_file}")
    print(f"Image num base: {args.image_num_base}  (unverified -- see guide section 5)")
    print(f"Max new tokens: {args.max_new_tokens}")
    if args.max_objects:
        print(f"Max objects   : {args.max_objects}")
    print(f"GPU id/total  : {args.gpu_id} / {args.num_gpus}")
    print(f"Objects       : {len(objects)}")
    print(f"Sizes         : {args.sizes}")
    print(f"Lightings     : {args.lightings}")
    print(f"View groups   : {args.view_groups}")
    print(f"Combinations  : {n_configs} per object")
    print(f"(Existing JSON keys are skipped automatically)")
    print("==================================================\n")
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
    prompt_single_top,    prompt_single_bottom = entry["single_top"], entry["single_bottom"]
    prompt_multi_top,     prompt_multi_bottom  = entry["multi_top"],  entry["multi_bottom"]

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

    token_log: dict[int, list[int]] = {}

    for obj_dir in tqdm(
        objects,
        desc=f"GPU {args.gpu_id}",
        unit="obj",
        dynamic_ncols=True,
        bar_format="{desc:<12} {bar:40} {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ):
        process_object(
            object_dir            = obj_dir,
            view_groups           = args.view_groups,
            sizes                 = args.sizes,
            lightings             = args.lightings,
            prompt_mode           = args.prompt_mode,
            prompt_single_top     = prompt_single_top,
            prompt_single_bottom  = prompt_single_bottom,
            prompt_multi_top      = prompt_multi_top,
            prompt_multi_bottom   = prompt_multi_bottom,
            experiment_id         = args.experiment_id,
            prompt_id             = args.prompt_id,
            max_new_tokens        = args.max_new_tokens,
            image_num_base        = args.image_num_base,
            revision              = args.revision,
            token_log             = token_log,
        )

    print(f"\n[GPU {args.gpu_id}] Done.")

    if token_log:
        exp_tag  = f"_{args.experiment_id}" if args.experiment_id else ""
        tok_path = symmetry_dir / f"token_counts_molmopoint{exp_tag}.txt"
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
