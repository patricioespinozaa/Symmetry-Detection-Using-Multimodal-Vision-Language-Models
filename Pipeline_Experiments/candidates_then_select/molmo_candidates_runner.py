# molmo_candidates_runner.py
#
# Real Molmo2 inference for EXP-LIT-1 (Candidates-then-Select -- CVPC
# arXiv:2512.04686; ZeroDex arXiv:2606.19340), unlike the oracle/random
# SIMULATION in Experiments/sandbox_literatura.ipynb: reads a prior prompt
# run's points as pass-1 anchors, marks K+1 numbered candidates around each
# point on the actual render, and asks Molmo2 to pick one (pass 2).
#
# Needs a GPU (loads allenai/Molmo2-8B) -- run on the server, same as
# MolmoPointing/molmo_multiview_runner.py, whose conventions (resumable
# cumulative JSON, --gpu-id/--num-gpus sharding, --experiment-id file
# suffixing) this script mirrors so its output
# (molmo_multiview_<OUT_EXP>.json) can be consumed unmodified by
# Mapping/estimate_symmetry_no_mesh.py or estimate_symmetry_variants.py, as
# if it were an ordinary prompt run.
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from pipeline_common.naming import exp_filename  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry import (  # noqa: E402
    CANDIDATE_SEED_DEFAULT, DELTA_CANDIDATES_DEFAULT, K_CANDIDATES_DEFAULT,
    draw_candidates, generate_candidates,
)

MOLMO_JSON = "molmo_multiview.json"
MODEL_ID = "allenai/Molmo2-8B"

ROLE_DESCRIPTIONS = {
    "axis_sym": {
        1: "the TOP pole of the object's rotational symmetry axis",
        2: "the BOTTOM pole of the object's rotational symmetry axis",
    },
    "plane_sym": {
        1: "the horizontal midpoint of the object's left-right extent, near the TOP",
        2: "the horizontal midpoint of the object's left-right extent, near the BOTTOM",
    },
}

SELECTION_PROMPT_TEMPLATE = """You are given ONE image of a 3D object, with {n_candidates} \
candidate points marked and numbered (in red) on top of it.

Exactly one of these numbered candidates correctly marks {role_description}.

Look carefully at the object's geometry and choose the candidate closest to \
the correct location. If you are unsure, pick your single best guess.

Output ONLY the chosen candidate number, in this exact format:

<selected id="N">"""

_processor = None
_model = None


def get_model():
    """ Load the model/processor once and reuse across calls. """
    global _processor, _model
    if _processor is None or _model is None:
        print(f"[model] Loading {MODEL_ID} ...")
        _processor = AutoProcessor.from_pretrained(
            MODEL_ID, trust_remote_code=True, device_map="auto", use_fast=True,
        )
        _model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, trust_remote_code=True, device_map="auto", dtype=torch.bfloat16,
        )
        _model.eval()
        print("[model] Ready.")
    return _processor, _model


def call_selection(image: Image.Image, prompt: str) -> str:
    """ One single-image Molmo2 call asking it to choose among the marked candidates.

    Args:
        * image: the candidate-marked render
        * prompt: the selection prompt (see SELECTION_PROMPT_TEMPLATE)

    Returns:
        * str: raw decoded model output

    """
    processor, model = get_model()
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt}, {"type": "image", "image": image},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=32)
    n_input = inputs["input_ids"].size(1)
    del inputs
    text = processor.tokenizer.decode(output_ids[0, n_input:], skip_special_tokens=True)
    del output_ids
    torch.cuda.empty_cache()
    return text


def parse_selected_id(text: str, n_candidates: int) -> int | None:
    """ Parse the chosen 1-indexed candidate number out of the model's raw output.

    Args:
        * text: raw decoded model output
        * n_candidates: how many candidates were offered (for bounds-checking)

    Returns:
        * int | None: the chosen candidate index (1-indexed), or None if
          nothing parseable / out of range was found

    """
    match = re.search(r'id=["\']?(\d+)["\']?', text)
    if not match:
        match = re.search(r"\b(\d+)\b", text)
    if not match:
        return None
    idx = int(match.group(1))
    return idx if 1 <= idx <= n_candidates else None


def load_results(json_path: Path) -> dict:
    """ Load existing cumulative results, or an empty dict if the file doesn't exist yet. """
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(json_path: Path, data: dict) -> None:
    """ Save cumulative results, creating parent directories as needed. """
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def process_view_point(
    image: Image.Image, p: dict, symmetry_type: str,
    k_candidates: int, delta_candidates: float, candidate_seed: int,
) -> dict:
    """ Run pass 2 (candidate generation + selection) for one pass-1 point.

    Args:
        * image: the plain render for this view (unmarked)
        * p: the pass-1 point ({"obj_id", "x", "y"}) from the base experiment
        * symmetry_type: "axis_sym" or "plane_sym" (selects the role description)
        * k_candidates: extra candidates to generate around p
        * delta_candidates: candidate disk radius, Molmo 0-1000 units
        * candidate_seed: RNG seed for candidate generation

    Returns:
        * dict: {"obj_id", "x", "y", "n_candidates", "selected_idx", "raw_output"}
          -- x/y are the SELECTED candidate's coordinates (falls back to the
          pass-1 point if the model's answer couldn't be parsed)

    """
    candidates = generate_candidates(p["x"], p["y"], k=k_candidates, delta=delta_candidates,
                                      seed=candidate_seed)
    marked_image = draw_candidates(image, candidates)
    role = ROLE_DESCRIPTIONS.get(symmetry_type, {}).get(
        p["obj_id"], f"the point labeled {p['obj_id']} in the original instructions",
    )
    prompt = SELECTION_PROMPT_TEMPLATE.format(n_candidates=len(candidates), role_description=role)
    raw = call_selection(marked_image, prompt)
    selected_idx = parse_selected_id(raw, len(candidates))
    chosen = candidates[selected_idx - 1] if selected_idx is not None else {"x": p["x"], "y": p["y"]}

    return {
        "obj_id": p["obj_id"], "x": chosen["x"], "y": chosen["y"],
        "n_candidates": len(candidates), "selected_idx": selected_idx, "raw_output": raw,
    }


def process_object(
    object_dir: Path, sizes: list[int], lightings: list[str],
    base_experiment_id: str, output_experiment_id: str,
    k_candidates: int, delta_candidates: float, candidate_seed: int,
) -> None:
    """ Run the candidates-then-select pass for every (size, lighting, n_views) of one object.

    Args:
        * object_dir: <renders_root>/<symmetry_type>/<object_id>
        * sizes: render sizes to process
        * lightings: illuminations to process
        * base_experiment_id: prior prompt run to read pass-1 points from
        * output_experiment_id: --experiment-id to write results under
        * k_candidates: extra candidates generated per pass-1 point
        * delta_candidates: candidate disk radius, Molmo 0-1000 units
        * candidate_seed: RNG seed for candidate generation

    """
    symmetry_type = object_dir.parent.name
    input_file = exp_filename(MOLMO_JSON, base_experiment_id)
    output_file = exp_filename(MOLMO_JSON, output_experiment_id)

    for size in sizes:
        for lighting in lightings:
            render_dir = object_dir / str(size) / lighting
            base_path = render_dir / input_file
            if not base_path.exists():
                continue
            with open(base_path, encoding="utf-8") as f:
                base_data = json.load(f)

            output_path = render_dir / output_file
            results = load_results(output_path)

            for n_views_key, group in base_data.items():
                if n_views_key in results:
                    continue
                images_sent = group.get("images_sent", [])
                points_by_image = group.get("points_by_image", {})
                if not images_sent or not points_by_image:
                    continue

                image_cache: dict[str, Image.Image] = {}
                new_points_by_image: dict[str, list[dict]] = {}
                for img_idx_str, pts in points_by_image.items():
                    if not pts:
                        new_points_by_image[img_idx_str] = []
                        continue
                    cam = images_sent[int(img_idx_str)]
                    filename = cam["filename"]
                    if filename not in image_cache:
                        img_path = render_dir / filename
                        image_cache[filename] = Image.open(img_path).convert("RGB") if img_path.exists() else None
                    image = image_cache[filename]
                    if image is None:
                        new_points_by_image[img_idx_str] = pts
                        continue

                    selected = [
                        process_view_point(image, p, symmetry_type, k_candidates,
                                           delta_candidates, candidate_seed)
                        for p in pts
                    ]
                    new_points_by_image[img_idx_str] = [
                        {"obj_id": s["obj_id"], "x": s["x"], "y": s["y"]} for s in selected
                    ]

                results[n_views_key] = {
                    "experiment_id": output_experiment_id,
                    "base_experiment_id": base_experiment_id,
                    "k_candidates": k_candidates, "delta_candidates": delta_candidates,
                    "points_by_image": new_points_by_image,
                    "images_sent": images_sent,
                    "n_points": sum(len(v) for v in new_points_by_image.values()),
                }
                save_results(output_path, results)   # survive interruptions


def parse_args() -> argparse.Namespace:
    """ Parse CLI arguments. """
    p = argparse.ArgumentParser(
        description="EXP-LIT-1 (Candidates-then-Select), real Molmo2 inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--renders-root", required=True)
    p.add_argument("--symmetry-type", required=True, choices=["axis_sym", "plane_sym"])
    p.add_argument("--base-experiment-id", required=True,
                   help="Prior prompt run to read pass-1 points from.")
    p.add_argument("--experiment-id", default=None,
                   help="Output experiment_id. Defaults to <base-experiment-id>_candidates.")
    p.add_argument("--k-candidates", type=int, default=K_CANDIDATES_DEFAULT)
    p.add_argument("--delta-candidates", type=float, default=DELTA_CANDIDATES_DEFAULT)
    p.add_argument("--candidate-seed", type=int, default=CANDIDATE_SEED_DEFAULT)
    p.add_argument("--sizes", type=int, nargs="+", default=[224])
    p.add_argument("--lightings", type=str, nargs="+", default=["flat"],
                   choices=["flat", "darker", "brighter"])
    p.add_argument("--max-objects", type=int, default=None)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--yes", "-y", action="store_true")
    return p.parse_args()


def main() -> None:
    """ CLI entrypoint. """
    args = parse_args()
    output_experiment_id = args.experiment_id or f"{args.base_experiment_id}_candidates"

    symmetry_dir = Path(args.renders_root) / args.symmetry_type
    if not symmetry_dir.exists():
        print(f"[error] Not found: {symmetry_dir}")
        sys.exit(1)

    all_objects = sorted(d for d in symmetry_dir.iterdir() if d.is_dir())
    if args.max_objects:
        all_objects = all_objects[:args.max_objects]
    objects = all_objects[args.gpu_id::args.num_gpus]

    print("\n===== EXP-LIT-1 Candidates-then-Select =====")
    print(f"Base experiment_id   : {args.base_experiment_id}")
    print(f"Output experiment_id : {output_experiment_id}")
    print(f"K candidates / delta : {args.k_candidates} / {args.delta_candidates}")
    print(f"Objects              : {len(objects)}")
    print(f"Calls per point      : 1 (2 points/view typical -> ~2x the base run's call count)")
    print("=============================================\n")
    if not args.yes and input("Type 'OK' to start: ").strip() != "OK":
        print("Cancelled.")
        sys.exit(0)

    for obj_dir in tqdm(objects, unit="obj", dynamic_ncols=True):
        process_object(
            object_dir=obj_dir, sizes=args.sizes, lightings=args.lightings,
            base_experiment_id=args.base_experiment_id, output_experiment_id=output_experiment_id,
            k_candidates=args.k_candidates, delta_candidates=args.delta_candidates,
            candidate_seed=args.candidate_seed,
        )

    print(f"\n[GPU {args.gpu_id}] Done.")


if __name__ == "__main__":
    main()
