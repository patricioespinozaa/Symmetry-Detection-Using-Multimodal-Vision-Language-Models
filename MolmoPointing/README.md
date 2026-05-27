# MolmoPointing — Batch Molmo2 Pointing Inference

Runs **Molmo2-8B** symmetry-axis pointing inference over the rendered image dataset.
Supports multi-GPU parallelism (one process per GPU), full resumability at the
`(object, size, illumination, n_views)` level, and four prompt modes.

---

## File structure

```
MolmoPointing/
└── molmo_multiview_runner.py   # Entry point — all-in-one batch runner
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
| View groups | 7 (1, 6, 14, 26, 42, 62, 86, 114) |
| Sizes | 3 (224, 448, 1136) |
| Lightings | 3 (flat, darker, brighter) |
| **Calls per object — all groups, all configs** | 7 × 3 × 3 = **63 calls** |
| **Total calls — 1,700 objects** | ~**107,100 calls** |

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
