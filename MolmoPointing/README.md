# MolmoPointing — Batch Molmo2 Pointing Inference

Runs **Molmo2-8B** symmetry-axis pointing inference over the rendered image dataset.
Supports multi-GPU parallelism (one process per GPU), full resumability at the
`(object, size, illumination, n_views)` level, and four prompt modes.

---

## File structure

```
MolmoPointing/
├── molmo_multiview_runner.py   # Entry point — all-in-one batch runner
├── prompts_registry.py         # Auto-discovers prompt variants from prompts/
├── prompts/
│   ├── axis/                   # axis_v00-v05, axis_v00_1-v05_1 (original + improved)
│   ├── plane/                  # plane_v00-v05, plane_v00_1-v05_1
│   └── description/            # Flow B/C prompts (describe.txt, describe_and_point_{axis,plane}.txt)
├── Experiments.md               # Prompt table + full experiment loops
└── PROMPT_IMPROVEMENTS_v1.md    # Rationale for v0 -> v_1 prompt changes
```

---

## Input structure

Expects the output produced by `data_render.py` / `export_fibonacci_views.py`:

```
<renders_root>/<symmetry_type>/<object_id>/<image_size>/<illumination>/
    IND_00_AZ_090_EL_+89.png
    ...                                   # 114 images (one per viewpoint)
    metadata_all.json
```

Only `ROT_000` images are used (one image per viewpoint, no 2D rotations).

---

## Output structure

Results are written alongside the renders as a **cumulative JSON** file:

```
<renders_root>/<symmetry_type>/<object_id>/<image_size>/<illumination>/
    molmo_multiview.json    ← one key per n_views group, accumulated across runs
```

### JSON format

```json
{
  "1": {
    "prompt_mode":           "auto",
    "prompt_used":           "...",
    "raw_output":            "...",
    "points_by_image": {
      "0": [{"obj_id": 1, "x": 450.0, "y": 230.0},
            {"obj_id": 2, "x": 452.0, "y": 810.0}]
    },
    "images_sent": [
      {
        "img_idx":   0,
        "filename":  "IND_00_AZ_090_EL_+89.png",
        "index":     0,
        "azimuth":   90,
        "elevation": 89,
        "eye":       [x, y, z],
        "R":         [[...], [...], [...]],
        "T":         [x, y, z]
      }
    ],
    "n_points":             2,
    "n_images_with_points": 1
  },
  "6":  { ... },
  "14": { ... },
  "26": { ... }
}
```

> **Note:** when `prompt_mode` is `single`, `raw_output` is a **list of strings**
> (one per image) instead of a single string, since N separate model calls are made.

> **Note:** when `--flow b` or `--flow c` is used, each n_views entry gains extra keys
> not present under the default `--flow a` — see [Flows](#flows) below.

### Coordinate format

Coordinates are in **0–1000 scale**, origin **top-left** (no Y inversion needed).

```python
px = point["x"] / 1000 * image_width
py = point["y"] / 1000 * image_height   # no Y inversion
```

---

## Prompt modes

Four modes are available via `--prompt-mode`:

| Mode | Prompt used | Model calls per group | When to use |
|---|---|---|---|
| `auto` | `PROMPT_SINGLE` if n==1, `PROMPT_MULTI` if n>1 | 1 | **Recommended** |
| `single` | `PROMPT_SINGLE` for every image | N (one per image) | Maximum quality, slower |
| `multi` | `PROMPT_MULTI` for all images (fallback to single if n==1) | 1 | Multi-image only |
| `global` | `PROMPT_GLOBAL` for all images | 1 | Baseline comparison |

### Prompt descriptions

**`PROMPT_SINGLE`** — used for one image at a time. Asks the model to return two
distant points along the projected symmetry axis, as far apart as possible.

**`PROMPT_MULTI`** — used when N > 1 images are sent together in a single call.
Asks the model to infer a single consistent global 3D axis across all views and
project it into each image, returning top and bottom endpoints per image.

**`PROMPT_GLOBAL`** — handles 1 or N images with the same prompt. Useful as a
baseline but empirically performs worse than `auto` on both single and multi-image
cases.

---

## Flows

Three flows are available via `--flow`, orthogonal to `--prompt-mode`/`--prompt-id`
(those still control the *main* N-view pointing call in all three flows):

| Flow | Description | Extra model calls |
|---|---|---|
| `a` (default) | Direct pointing, no semantic context. Identical to the original single-flow behavior. | 0 |
| `b` | Pointing con descripción — a seeded single-view call asks the model to describe the object; the description is prepended to the base pointing prompt. | 1 per (size, lighting) config |
| `c` | Descripción y pointing integrados — a seeded single-view call asks the model to describe the object AND give a short qualitative hint of where the axis/plane is located (plain text, no coordinates). That description + location hint replace the base `--prompt-id` prompt for the main N-view call: the model is explicitly asked to find points ON the symmetry axis/plane in each image (exactly 2 points/image, no cross-view identity tracking). | 1 per (size, lighting) config |

The description view is picked per object with a **deterministic seed**
(`MD5(object_id) mod 2^31`), filtered to elevation in `(-60°, +60°)` to avoid
near-pole views that project an axis to a point or a plane to a line. See
`pipeline_common/view_selection.py`.

**Flow C now requires exactly 2 points/image** (`FLOW_C_POINTS_PER_IMAGE = 2`
in `molmo_multiview_runner.py`) — an earlier version asked for "up to 3,
fewer OK", but that flexible cap empirically collapsed to 1 point/image once
several images shared a single call (confirmed on both `axis_v05_1_flowC`
and `plane_v04_1_flowC`), which is unusable for the mesh-free triangulation
pipeline (needs ≥2 points/view to define an interpretation plane — see
`docs/pipeline_sin_malla.md` §3.1). `Mapping/estimate_symmetry.py` should
still be run with **`--point-mode all`** for Flow C experiments (not
`independent`): `all` tolerates images where only one of the two points
hits the mesh, whereas `independent` discards the whole image in that case.
See `build_flow_c_prompts` and `parse_describe_and_location` in
`molmo_multiview_runner.py`, and `docs/archive/EXPERIMENT_ROADMAP.md` §4 for
the full command sequence (kept for historical reference — see
`docs/archive/` for why it's no longer at the repo root).
`--prompt-id` is still required for `--flow c` as the fallback base prompt
for the rare case where the pre-pass produces no parseable location hint
for an object.

Note: an earlier iteration asked the pre-pass to name several labeled
landmarks and had the main call re-locate each one by identity across
views (obj_id tracked a specific named feature). Empirically this
degenerated into mechanically evenly-spaced grid/line patterns instead of
genuine per-point reasoning (angular error ~90°, chance level). The current
design drops per-landmark identity tracking, fixes points/image at exactly
2, and adds explicit anti-degenerate rules (no grid/line patterns, no
duplicate coordinates, no inventing points) targeting that failure mode.

`--flow b`/`c` only affect `--prompt-mode single/multi/auto`; `--prompt-mode global`
does not support prompt overrides at all (not even via `--prompt-id`), so `--flow`
has no effect when combined with it (a warning is printed).

```bash
# Flow B, using the best Flow-A prompt as base
CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \
    --renders-root ../data/renders --symmetry-type axis_sym \
    --sizes 224 --lightings flat --view-groups 1 6 14 26 \
    --prompt-id axis_v05_1 --flow b --experiment-id axis_v05_1_flowB \
    --prompt-mode auto

# Flow C
CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \
    --renders-root ../data/renders --symmetry-type axis_sym \
    --sizes 224 --lightings flat --view-groups 1 6 14 26 \
    --prompt-id axis_v05_1 --flow c --experiment-id axis_v05_1_flowC \
    --prompt-mode auto
```

Resumability is flow-aware: a stored entry's `flow` defaults to `"a"` when
absent, so `--flow a` reruns never reprocess anything, but switching `--flow`
under the same `--experiment-id` reprocesses rather than silently reusing the
other flow's result. Use a distinct `--experiment-id` per flow to keep
artifacts separate.

---

## Usage

### Single GPU — recommended starting point

```bash
CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \
    --renders-root ../data/renders \
    --symmetry-type axis_sym \
    --sizes 224 \
    --lightings flat \
    --view-groups 6 \
    --prompt-mode auto

CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \
    --renders-root ../data/renders \
    --symmetry-type plane_sym \
    --sizes 224 \
    --lightings flat \
    --view-groups 1 6 14 26 \
    --prompt-mode auto
```

### Two GPUs — split by symmetry type

Run each command in a separate `tmux` session:

```bash
# GPU 0 — axis symmetry
CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \
    --renders-root ../data/renders \
    --symmetry-type axis_sym \
    --sizes 224 --lightings flat \
    --view-groups 1 6 14 26 \
    --prompt-mode auto

# GPU 1 — plane symmetry
CUDA_VISIBLE_DEVICES=1 python MolmoPointing/molmo_multiview_runner.py \
    --renders-root ../data/renders \
    --symmetry-type plane_sym \
    --sizes 224 --lightings flat \
    --view-groups 1 6 14 26 \
    --prompt-mode auto
```

### Two GPUs — split same symmetry type across both GPUs

```bash
# GPU 0 — objects 0, 2, 4, ...  (425 objects)
CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \
    --renders-root ../data/renders \
    --symmetry-type axis_sym \
    --gpu-id 0 --num-gpus 2 \
    --view-groups 1 6 14 26 \
    --prompt-mode auto

# GPU 1 — objects 1, 3, 5, ...  (425 objects)
CUDA_VISIBLE_DEVICES=1 python MolmoPointing/molmo_multiview_runner.py \
    --renders-root ../data/renders \
    --symmetry-type axis_sym \
    --gpu-id 1 --num-gpus 2 \
    --view-groups 1 6 14 26 \
    --prompt-mode auto
```

Objects are distributed with **round-robin assignment** — `gpu_id=0` takes indices
0, 2, 4, …; `gpu_id=1` takes 1, 3, 5, …

### Add new view groups to existing results

Existing keys are never overwritten. Running with new groups simply adds them:

```bash
CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \
    --renders-root ../data/renders \
    --symmetry-type axis_sym \
    --view-groups 42 62 86 114 \
    --prompt-mode auto
# keys 1, 6, 14, 26 already in JSON → skipped automatically
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--renders-root` | *(required)* | Root folder from `data_render.py` |
| `--symmetry-type` | *(required)* | `axis_sym` or `plane_sym` |
| `--prompt-mode` | `auto` | `auto`, `single`, `multi`, or `global` |
| `--flow` | `a` | `a` (direct), `b` (pointing + description), `c` (description + pointing integrados) |
| `--gpu-id` | `0` | Index of this process (0-based) |
| `--num-gpus` | `1` | Total parallel processes |
| `--sizes` | `224 448 1136` | Image sizes to process |
| `--lightings` | `flat brighter darker` | Illumination modes |
| `--view-groups` | `1 6 14 26 42 62 86 114` | View-group sizes |

---

## Resumability

Results accumulate in `molmo_multiview.json`. Each `(size, illumination, n_views)`
combination is skipped if its key already exists in the JSON.

To rerun a specific combination, use `cleanup_experiments.py` (see `Mapping/README.md`):

```bash
# Preview what would be deleted
python Mapping/cleanup_experiments.py \
    --renders-root ../data/renders \
    --symmetry-type axis_sym \
    --sizes 224 --lightings flat \
    --view-groups 1 6 14 26 \
    --dry-run

# Execute
python Mapping/cleanup_experiments.py \
    --renders-root ../data/renders \
    --symmetry-type axis_sym \
    --sizes 224 --lightings flat \
    --view-groups 1 6 14 26
```

---

## Scale estimate

| Parameter | Value |
|---|---|
| Objects | 850 per symmetry type (1,700 total) |
| View groups | 8 (1, 6, 14, 26, 42, 62, 86, 114) |
| Sizes | 3 (224, 448, 1136) |
| Lightings | 3 (flat, darker, brighter) |
| **Calls per object — all groups, all configs** | 8 × 3 × 3 = **72 calls** |
| **Total calls — 1,700 objects** | ~**122,400 calls** |

For initial experiments, restrict to a single size and illumination to reduce by 9×:

```bash
--sizes 224 --lightings flat   # 7 calls per object instead of 63
```

---

## tmux cheatsheet

```bash
# Create sessions
tmux new -s molmo_axis
tmux new -s molmo_plane

# Detach without killing the process
Ctrl+B, then D

# Reattach
tmux attach -t molmo_axis
tmux attach -t molmo_plane

# List active sessions
tmux ls
```
