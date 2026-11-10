# MolmoPoint & Qwen3-VL Pointing — Implementation & Replication Guide

## Description

`MolmoPointPointing/` and `QwenPointing/` are two additional, fully
independent pointing backends added alongside `MolmoPointing/` (Molmo2-8B),
so the thesis can compare Molmo2 against **MolmoPoint-8B**
(`allenai/MolmoPoint-8B`) and **Qwen3-VL-8B-Instruct**
(`Qwen/Qwen3-VL-8B-Instruct`) on the same multi-view pointing task
(`axis_v04`/`plane_v05_1` and later variants such as `plane_v02`), feeding the
same mesh-free triangulation + evaluation pipeline
(`Mapping/estimate_symmetry_no_mesh.py`, `Mapping/evaluate.py`).

This doc exists specifically so the whole thing can be **replicated on a
different server from scratch** — environment, folder layout, exact
commands, and (most importantly) the non-obvious bugs/behaviors that were
only found by actually running both models against real renders. The
original planning document, `guia_instalacion_molmopoint_qwen3vl.md` (repo
root), proposed the integration up front and flagged several things as
"unverified" — this doc records what was actually true once tested live
against the models on GPU, which in a few places differs from that original
plan (see "What changed vs. the original plan" below).

## Why two new top-level folders instead of extending `MolmoPointing/`

Explicit design constraint from the start: **`MolmoPointing/` (Molmo2's own
runner, prompts, registry) must never be modified or imported from** —
redundancy was preferred over coupling, so a bug or future change in one
model's code can never silently affect another's. Concretely:

- `MolmoPointPointing/molmopoint_multiview_runner.py` and
  `QwenPointing/qwen3vl_multiview_runner.py` each duplicate (not import) the
  handful of pure helpers `molmo_multiview_runner.py` also has
  (`load_metadata`, `get_n_views_entries`, `load_results`, `save_results`,
  `_exp_filename`).
- Each folder has its **own** `prompts_registry.py` and `prompts/{axis,plane}/`
  tree — adding a prompt variant for MolmoPoint or Qwen3-VL never touches
  `MolmoPointing/prompts/`.
- Each model writes to its **own** default output filename —
  `molmopoint_multiview.json` / `qwen3vl_multiview.json` — never
  `molmo_multiview.json`. All three files can sit in the same
  `<renders_root>/<symmetry_type>/<object_id>/<size>/<lighting>/` directory
  with zero collision risk, even with no `--experiment-id`.

```
MolmoPointPointing/
├── molmopoint_multiview_runner.py
├── prompts_registry.py                 # own copy, scans ITS OWN prompts/
└── prompts/{axis,plane}/<variant>_{single,multi}_{top,bottom}.txt

QwenPointing/
├── qwen3vl_multiview_runner.py
├── prompts_registry.py                 # own copy, scans ITS OWN prompts/
└── prompts/{axis,plane}/<variant>_{single,multi}.txt
```

Note the asymmetry: MolmoPoint needs **four** prompt files per variant
(`_single_top`/`_single_bottom`/`_multi_top`/`_multi_bottom`), Qwen3-VL needs
only **two** (`_single`/`_multi`) — this is not cosmetic, see below.

## Environment — nothing new to install

Both runners use **the exact same conda env already set up for Molmo2**
(`tesis_env`: `transformers==4.57.1`, `torch==2.6.0+cu124`, `accelerate`,
`pillow`, `einops`, `tqdm`, `numpy`; `gpytoolbox`/`scipy`/`pandas` for the
evaluation stage). Neither `qwen-vl-utils` nor `molmo_utils` is imported by
either runner — Qwen3-VL is driven purely through
`transformers.Qwen3VLForConditionalGeneration` + `AutoProcessor`, and
MolmoPoint purely through `AutoModelForImageTextToText` +
`model.extract_image_points()` (both `trust_remote_code=True`, downloaded
automatically from Hugging Face on first run — ~16–20GB each, cached under
`~/.cache/huggingface`).

```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate tesis_env
```

On a brand-new server: follow `README.md`'s existing "Installation" section
(PyTorch3D + the pinned `transformers==4.57.1`/`accelerate`/etc.) — that's
sufficient for all three models, no extra `pip install` needed for
MolmoPoint/Qwen3-VL specifically.

**GPU**: both models are ~16–18GB in bf16, i.e. close to a single 24GB GPU's
capacity once you add activation memory — they cannot both be loaded on the
same GPU at once. Always run MolmoPoint's stages to completion, let the
process exit (frees the GPU), *then* run Qwen3-VL's. Check `nvidia-smi`
before starting anything to confirm which GPU is actually free (shared
servers may have a GPU reserved by someone else).

## Model-specific implementation notes (the parts that actually mattered)

### MolmoPoint-8B

**Loading bugs, both load-time, both silent/confusing if you don't know
about them:**

1. `AutoModelForImageTextToText.from_pretrained(..., dtype="auto", ...)`
   **OOMs on a 24GB GPU.** `allenai/MolmoPoint-8B`'s own `config.json`
   declares `"dtype": "float32"`, so `dtype="auto"` loads the full fp32
   checkpoint (~2x the expected memory) — it alone consumes ~23.6GB before
   `generate()` even starts, with zero room left for activations/KV-cache.
   **Fix: force `dtype=torch.bfloat16` explicitly.**
2. `device_map="auto"` decides (correctly, given the fp32-vs-bf16 confusion
   above) that the model doesn't fit on one GPU and tries to CPU-offload part
   of the vision tower — which then crashes with
   `NotImplementedError: Cannot copy out of meta tensor; no data!` inside
   MolmoPoint's own custom remote-code `modeling_molmo_point.py` (an
   offload/dispatch incompatibility in that code, not something fixable from
   the caller's side). **Fix: pin `device_map={"": 0}`** (the single visible
   GPU under `CUDA_VISIBLE_DEVICES`) so no offload path is ever taken.

Both fixes are in `MolmoPointPointing/molmopoint_multiview_runner.py`'s
`get_model()`.

**Prompt sensitivity — this is the single most important finding, and it
drove a real architecture change.** MolmoPoint communicates points via three
special tokens (`<PATCH>`/`<SUBPATCH>`/`<LOCATION>`), decoded with
`model.extract_image_points()` using metadata the processor attaches via
`return_pointing_metadata=True`. Getting it to actually *emit* those tokens
(rather than answer in plain prose, which parses to zero points) turned out
to require very specific phrasing, found by live trial-and-error against the
model on real renders — none of this is documented on the model card:

- **Asking for two targets in one call reliably breaks pointing entirely**
  (model answers in free-text prose with zero special tokens), whether
  phrased as "point to X and Y", "point to: X, Y", or even two back-to-back
  imperative sentences in the same call. Sometimes it partially recovers
  (grounds the *first* target, describes the second in prose) — never both.
- **Any "use the SAME axis/plane across views" consistency framing breaks
  it too**, zero points, even when it's just one clause added before an
  otherwise-working single-target instruction.
- **A trailing qualifier on the noun phrase breaks it** — `"point to the top
  pole of the rotation axis"` fails; the bare `"point to the top pole"`
  works. Same for plane: `"...topmost point on the plane's trace"` fails;
  `"...top of the seam"` works.
- **Superlatives + "of" also breaks it** — `"point to the topmost point of
  the seam"` fails; `"point to the top of the seam"` (no superlative) works.

**Resulting design**: `molmopoint_multiview_runner.py` issues **two separate
model calls per `n_views` batch** — one asking only for the top/first target,
one asking only for the bottom/second target — then merges the results,
tagging every point from the top-call `obj_id=1` and every point from the
bottom-call `obj_id=2` (MolmoPoint's own per-call `object_id` is just a
running index over matches found *in that call*, not a stable role marker,
so it's discarded in favor of this call-provenance tag). This is why every
prompt variant needs four files, not two — see "Adding a new prompt variant"
below.

**Known, accepted model-quality limitations** (not bugs): a single call can
return more than one point for the same image on an ambiguous/symmetric
object (e.g. a lattice/grid shape with several locally-plausible "top"
features) — not deduplicated, since the no-mesh pipeline reads
`points_by_image` as a flat per-image list. And the top-call and bottom-call
occasionally return the *same* coordinate (top==bottom collapse) on some
objects — a real accuracy limitation, observed at a low but non-zero rate.

### Qwen3-VL-8B-Instruct

No loading bugs — `dtype="auto"`/`device_map="auto"` work fine (the
checkpoint's declared dtype is accurate). Output is free text containing a
JSON block (`point_2d: [x, y]`, 0–1000 scale, top-left origin, same
convention Molmo2 already uses), not a special-token format, so the
failure modes are different:

- **The model sometimes nests the second point inside the first** instead of
  two flat list entries — e.g. `{"image": 0, "label": "top", "point_2d":
  [...], "bottom": {"point_2d": [...]}}` — despite the prompt explicitly
  showing a flat-list example and an explicit "do NOT nest" rule. Fixed two
  ways: (1) `_parse_points()` in `qwen3vl_multiview_runner.py` defensively
  flattens any dict-valued field that itself contains a `point_2d`, so a
  nested point is recovered rather than lost; (2) the multi-image prompt
  (`prompts/{axis,plane}/*_multi.txt`) spells out the exact flat-list
  format and explicitly forbids nesting, which reduced (but did not fully
  eliminate) the nesting behavior.
- **Multi-image batching has an undocumented, unreliable image-index
  convention.** The model has no documented way of saying "this point
  belongs to image N" when given several images in one call. The runner
  only trusts an explicit `"image"` key per point (checked against both
  0- and 1-based indexing); anything missing or out of range is dropped
  and counted in `n_points_dropped` rather than guessed.
- **At `n_views=26` with multi-image batching, the model can fall into a
  genuine repetition loop** — repeating the same `point_2d` over and over
  under *fabricated* `"image"` indices that exceed the real image count
  (e.g. `"image": 31, 32...` for a 26-image call) — and never terminates
  within any reasonable token budget. Confirmed by inspecting raw output:
  this is a model behavior issue, not a `max_new_tokens` shortfall (raising
  the budget just produces more garbage before hitting the limit).

**Resulting production choice: `--prompt-mode single`** (one call per
image, not `auto`/`multi`) for the real experiment runs. This was validated
head-to-head against multi-image batching on the same 10-object sample:
single-image mode gave **0 dropped points and exactly 2 points/image across
every object**, markedly cleaner than multi-image batching, and it entirely
sidesteps both the image-index ambiguity and the `n_views=26` repetition
loop (since every call only ever contains one image). The flip side — one
model call per image instead of one call per `n_views` group — is far
*cheaper* per call too, so the net runtime cost is comparable, not worse.

MolmoPoint was **also** validated with `--prompt-mode single` on the same
10-object sample for the same reason (consistency, and because the model
card's own usage example is single/few-image, not large batches) — it
performed at least as well as multi-image batching there too. **Both models'
real production runs used `--prompt-mode single`.**

## Adding a new prompt variant

Dropping new prompt files into either folder's `prompts/axis/` or
`prompts/plane/` registers a new `prompt_id` automatically — **no code
changes needed** in either `prompts_registry.py`.

**Qwen3-VL** (two files per variant):
```
QwenPointing/prompts/axis/<variant>_qwen3vl_single.txt
QwenPointing/prompts/axis/<variant>_qwen3vl_multi.txt
```
registers prompt_id `axis_<variant>_qwen3vl`. The multi-image prompt must
explicitly ask for a flat JSON list with an `"image": <0-based index>` field
per point (see any existing `*_multi.txt` for the exact wording/format the
parser expects) if you ever use `--prompt-mode multi`/`auto` — not needed for
`--prompt-mode single`, which is what production actually uses.

**MolmoPoint** (**four** files per variant — top/bottom split, see above):
```
MolmoPointPointing/prompts/axis/<variant>_molmopoint_single_top.txt
MolmoPointPointing/prompts/axis/<variant>_molmopoint_single_bottom.txt
MolmoPointPointing/prompts/axis/<variant>_molmopoint_multi_top.txt
MolmoPointPointing/prompts/axis/<variant>_molmopoint_multi_bottom.txt
```
registers prompt_id `axis_<variant>_molmopoint`. Each of the four files must
be a **short, single-target, bare "point to X" instruction** — no second
target, no cross-view consistency clause, no trailing qualifier on the noun
phrase, no superlative+"of" construction. Copy the wording pattern of an
existing working prompt (e.g. `v04_molmopoint_multi_top.txt`) rather than
starting from Molmo2's own (much more verbose, rule-heavy) prompt style —
that style is exactly what breaks pointing.

List what's registered in either folder:
```bash
python MolmoPointPointing/prompts_registry.py
python QwenPointing/prompts_registry.py
```

## Running inference

Always smoke-test a new prompt variant on a handful of objects before the
full corpus — both the MolmoPoint prompt sensitivity and the Qwen3-VL
nesting/repetition issues above were only found this way. Use a throwaway
`--experiment-id` so nothing pollutes the real `molmopoint_multiview.json`/
`qwen3vl_multiview.json` files:

```bash
export CUDA_VISIBLE_DEVICES=1   # whichever GPU is actually free, check nvidia-smi first

python MolmoPointPointing/molmopoint_multiview_runner.py \
    --renders-root /path/to/data/renders --symmetry-type axis_sym \
    --prompt-id axis_v04_molmopoint --prompt-mode single \
    --sizes 224 --lightings flat --view-groups 6 \
    --max-objects 5 --experiment-id smoketest --yes

python QwenPointing/qwen3vl_multiview_runner.py \
    --renders-root /path/to/data/renders --symmetry-type axis_sym \
    --prompt-id axis_v04_qwen3vl --prompt-mode single \
    --sizes 224 --lightings flat --view-groups 6 \
    --max-objects 5 --experiment-id smoketest --yes
```

Inspect `molmopoint_multiview_smoketest.json`/`qwen3vl_multiview_smoketest.json`
under each sampled object's `224/flat/` directory: check `n_points`,
`n_images_with_points` (should be `points-per-image × n_views`, typically
2×), and (Qwen3-VL) `n_points_dropped`/`raw_output_parse_errors`.

**Production run** (both models, both symmetry types, sequentially — see GPU
note above — in `tmux` so it survives a dropped SSH connection):

```bash
tmux new -s single_exp
# inside tmux:
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate tesis_env
export CUDA_VISIBLE_DEVICES=1
cd /path/to/Symmetry-Detection-Using-Multimodal-Vision-Language-Models
RENDERS_ROOT=/path/to/data/renders
LOGDIR=logs; mkdir -p "$LOGDIR"

python MolmoPointPointing/molmopoint_multiview_runner.py \
    --renders-root "$RENDERS_ROOT" --symmetry-type axis_sym \
    --prompt-id axis_v04_molmopoint --prompt-mode single \
    --view-groups 6 14 26 --sizes 224 --lightings flat --yes \
    2>&1 | tee -a "$LOGDIR/molmopoint_axis.log"

python MolmoPointPointing/molmopoint_multiview_runner.py \
    --renders-root "$RENDERS_ROOT" --symmetry-type plane_sym \
    --prompt-id plane_v05_1_molmopoint --prompt-mode single \
    --view-groups 6 14 26 --sizes 224 --lightings flat --yes \
    2>&1 | tee -a "$LOGDIR/molmopoint_plane.log"

python QwenPointing/qwen3vl_multiview_runner.py \
    --renders-root "$RENDERS_ROOT" --symmetry-type axis_sym \
    --prompt-id axis_v04_qwen3vl --prompt-mode single \
    --view-groups 6 14 26 --sizes 224 --lightings flat --yes \
    2>&1 | tee -a "$LOGDIR/qwen3vl_axis.log"

python QwenPointing/qwen3vl_multiview_runner.py \
    --renders-root "$RENDERS_ROOT" --symmetry-type plane_sym \
    --prompt-id plane_v05_1_qwen3vl --prompt-mode single \
    --view-groups 6 14 26 --sizes 224 --lightings flat --yes \
    2>&1 | tee -a "$LOGDIR/qwen3vl_plane.log"
```
(`;`-separated commands, not `&&` — a crash on one stage shouldn't block the
rest. Detach: `Ctrl+b d`. Reattach: `tmux attach -t single_exp`.) Note
`--view-groups 1 6 14 26` matches Molmo2's own `axis_v04`/`plane_v05_1`
production runs exactly — check the existing `molmo_multiview_axis_v04.json`
keys on your dataset before assuming this list; it may differ per
experiment.

No `--experiment-id` here → writes the default
`molmopoint_multiview.json`/`qwen3vl_multiview.json`, independent of
Molmo2's `molmo_multiview*.json` and of each other.

### Verifying full coverage

Self-contained (pure stdlib, no conda env needed) — confirms, per symmetry
type and model, that every object with a `224/flat` render has the file,
all expected `n_views` keys, the right `prompt_id` tagged in each, and
non-empty `points_by_image`:

```bash
python3 << 'EOF'
import json
from pathlib import Path

RENDERS_ROOT = Path("/path/to/data/renders")
EXPECTED_NVIEWS = {"6", "14", "26"}
MODELS = {
    "molmopoint": ("molmopoint_multiview.json", {"axis_sym": "axis_v04_molmopoint", "plane_sym": "plane_v05_1_molmopoint"}),
    "qwen3vl":    ("qwen3vl_multiview.json",    {"axis_sym": "axis_v04_qwen3vl",    "plane_sym": "plane_v05_1_qwen3vl"}),
}

for sym in ["axis_sym", "plane_sym"]:
    sym_dir = RENDERS_ROOT / sym
    rendered = [d for d in sorted(sym_dir.iterdir())
                if d.is_dir() and (d / "224" / "flat" / "metadata_all.json").exists()]
    print(f"\n=== {sym}: {len(rendered)} rendered (224/flat) objects ===")
    for model, (fname, prompt_ids) in MODELS.items():
        n_has_file, n_complete, n_wrong_prompt = 0, 0, 0
        missing, incomplete = [], []
        for d in rendered:
            p = d / "224" / "flat" / fname
            if not p.exists():
                missing.append(d.name); continue
            n_has_file += 1
            try:
                data = json.load(open(p))
            except Exception:
                incomplete.append(d.name); continue
            if not EXPECTED_NVIEWS.issubset(data.keys()):
                incomplete.append(d.name); continue
            ok = True
            for nv in EXPECTED_NVIEWS:
                entry = data[nv]
                if entry.get("prompt_id") != prompt_ids[sym]:
                    n_wrong_prompt += 1; ok = False; break
                if not entry.get("points_by_image"):
                    ok = False; break
            (n_complete := n_complete + 1) if ok else incomplete.append(d.name)
        print(f"  {model:12s}: file present {n_has_file}/{len(rendered)}  |  "
              f"n_views complete+non-empty {n_complete}/{len(rendered)}  |  wrong prompt_id hits: {n_wrong_prompt}")
        if missing:    print(f"      missing ({len(missing)}): {missing[:8]}")
        if incomplete: print(f"      incomplete ({len(incomplete)}): {incomplete[:8]}")
EOF
```

## Post-processing (triangulation + evaluation)

`Mapping/estimate_symmetry_no_mesh.py` got one small, additive change: a
`--model {molmo2,molmopoint,qwen3vl}` flag (default `molmo2`, which is
byte-for-byte identical to its pre-existing behavior — no triangulation/
geometry code was touched). It controls only which inference file is read
and tags the output so it never collides with Molmo2's own:

| `--model` | reads | writes |
|---|---|---|
| `molmo2` (default) | `molmo_multiview[_<EXP>].json` | `predicted_symmetry[_<EXP>].json` |
| `molmopoint` | `molmopoint_multiview[_<EXP>].json` | `predicted_symmetry_molmopoint[_<EXP>].json` |
| `qwen3vl` | `qwen3vl_multiview[_<EXP>].json` | `predicted_symmetry_qwen3vl[_<EXP>].json` |

`Mapping/evaluate.py` needed **zero changes** — its existing `--experiment-id`
flag is reused as the model tag, since it only needs to find
`predicted_symmetry_<tag>.json`.

```bash
RENDERS_ROOT=/path/to/data/renders
OBJECTS_ROOT=/path/to/data/objects   # curated_axis_sym_obj / curated_plane_sym_obj (ground truth)

for model in molmopoint qwen3vl; do
  python Mapping/estimate_symmetry_no_mesh.py \
      --renders-root "$RENDERS_ROOT" --symmetry-type axis_sym \
      --model "$model" --sizes 224 --lightings flat

  python Mapping/estimate_symmetry_no_mesh.py \
      --renders-root "$RENDERS_ROOT" --symmetry-type plane_sym \
      --model "$model" --sizes 224 --lightings flat

  python Mapping/evaluate.py \
      --renders-root "$RENDERS_ROOT" --objects-root "$OBJECTS_ROOT" \
      --symmetry-type axis_sym --sizes 224 --lightings flat \
      --method triangulation --experiment-id "$model" --with-reference-metrics

  python Mapping/evaluate.py \
      --renders-root "$RENDERS_ROOT" --objects-root "$OBJECTS_ROOT" \
      --symmetry-type plane_sym --sizes 224 --lightings flat \
      --method triangulation --experiment-id "$model" --with-reference-metrics
done
```

This writes `eval_s224_flat_<model>_triangulation_{results.json,summary.csv}`
under `<renders_root>/<symmetry_type>/`. `--with-reference-metrics` adds
`SDE_ref` (both symmetry types) and `F1_ref`/`F1_ref_hungarian` (plane only —
there's no established reference F1 metric for a single rotational axis;
see the project's root `CLAUDE.md`).

### Combining into one comparison CSV per model/symmetry-type

`Mapping/compare_results_no_mesh.py` is **not** usable for this as-is — it
only recognizes `--experiment-id` values that look like Molmo2's own
`axis_vNN`/`plane_vNN` prompt-variant naming, which `molmopoint`/`qwen3vl`
don't match. Rather than edit that shared script, combine the one summary
CSV's columns directly (same columns `evaluate.py` already writes, just
prepending `experiment`/`method`):

```bash
for model in molmopoint qwen3vl; do
  for sym in axis_sym plane_sym; do
    python3 -c "
import pandas as pd
df = pd.read_csv('$RENDERS_ROOT/$sym/eval_s224_flat_${model}_triangulation_summary.csv')
df.insert(0, 'method', 'triangulation')
df.insert(0, 'experiment', '$model')
df.to_csv('$RENDERS_ROOT/$sym/${sym}_${model}_comparison.csv', index=False)
"
  done
done
```

Then copy into the repo under the matching model folder
(`MolmoPointPointing/experiments_<DD_MM_YYYY>/`,
`QwenPointing/experiments_<DD_MM_YYYY>/`) and commit — see existing
`experiments_*` folders in either directory for the established convention.

## Known pitfalls

- **Interrupting a run (Ctrl-C, SSH drop, manual kill) can leave a
  truncated/empty output file for whichever object was in flight.** Both
  `estimate_symmetry_no_mesh.py` and the model runners use skip-existing
  logic (only `str(n_views) not in results` / `output_path.exists()`), which
  treats "file exists" as "already done" — it does **not** validate the
  file's contents. A 0-byte file from an interrupted run is silently
  skipped forever afterward, and `evaluate.py` then crashes with a
  `JSONDecodeError` on it, aborting the *entire* evaluation run partway
  through (no partial CSV gets written at all for that symmetry
  type/model). This actually happened once in this project
  (`predicted_symmetry_molmopoint.json` for one `axis_sym` object was
  empty after an interrupted triangulation run, silently skipped on the
  rerun, and `evaluate.py` died at object 778/850 with nothing written).
  **Fix pattern**: if a run was ever interrupted, before trusting its
  output, scan for empty/corrupt files and delete them (not
  `--overwrite` the whole corpus — just the broken ones) so they get
  regenerated on the next run:
  ```bash
  python3 -c "
  import json, os
  from pathlib import Path
  for d in sorted(Path('$RENDERS_ROOT/axis_sym').iterdir()):
      p = d / 'predicted_symmetry_molmopoint.json'
      if p.exists():
          try: json.load(open(p))
          except Exception: print('CORRUPT:', p); os.remove(p)
  "
  ```
- **GPU 0 may be reserved by someone else on a shared server** — always
  check `nvidia-smi` and set `CUDA_VISIBLE_DEVICES` explicitly before
  starting anything; don't assume GPU 0 is free.
- **MolmoPoint and Qwen3-VL cannot share one 24GB GPU** — always run one
  model's stages to completion (process exits, frees the GPU) before
  starting the other's.

## What changed vs. the original plan (`guia_instalacion_molmopoint_qwen3vl.md`)

That document proposed the integration and correctly flagged several things
as unverified. Once actually tested against the models on GPU:

- `dtype="auto"` for MolmoPoint was proposed per the model card's own
  example — in practice it OOMs; `torch.bfloat16` must be forced (see above).
- The guide's `--prompt-mode auto`/`multi` (all views in one call) was the
  proposed default for both models. In practice, `--prompt-mode single`
  (one call per image) proved more reliable for both and is what the real
  production runs used.
- The guide anticipated MolmoPoint's prompt could follow the same
  multi-point, rule-heavy style as Molmo2's own prompts, just adapted in
  wording. In practice this reliably produces zero pointing tokens; the
  real implementation needed the two-call-per-target architecture described
  above, which the guide did not anticipate.
- `image_num_base` (0- vs. 1-based) for MolmoPoint was flagged as unverified
  in the guide; moot for `--prompt-mode single` since there's only ever one
  image per call (no index ambiguity).
- Qwen3-VL's point-scale denominator (1000 vs. 999, flagged unverified in
  the guide) was never contradicted by any observation; `--scale-denominator
  1000` (the default) was used throughout.
