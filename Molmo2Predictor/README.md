# MolmoPointing — Batch Molmo2 Pointing Inference

Runs **Molmo2-8B** symmetry-axis pointing inference over the rendered image dataset.
Supports multi-GPU parallelism (one process per GPU) and full resumability at the object level.

---

## File structure

```
MolmoPointing/
├── molmo_model.py      # Model + processor loader (singleton per process)
├── molmo_inference.py  # Single-image inference and coordinate extraction
├── molmo_visualize.py  # Annotated PNG output
├── molmo_batch.py      # Processes one object across all groups/sizes/lightings
└── molmo_runner.py     # Entry point — slices objects across GPUs
```

---

## Input structure

Expects the output produced by `data_render.py` / `export_fibonacci_views.py`:

```
<renders_root>/<symmetry_type>/<object_id>/<image_size>/<illumination>/
    IND_00_AZ_090_EL_+89_ROT_000.png
    IND_00_AZ_090_EL_+89_ROT_090.png
    ...                                   # 114 views × 4 rotations = 456 images
    metadata_all.json
```

---

## Output structure

Results are written alongside the renders, inside a `molmo/views_{N}/` subfolder:

```
<renders_root>/<symmetry_type>/<object_id>/<image_size>/<illumination>/molmo/views_{N}/
    IND_00_AZ_090_EL_+89_ROT_000.json       ← coords + raw model output
    IND_00_AZ_090_EL_+89_ROT_000_vis.png    ← annotated image (optional)
    ...
```

A sentinel file marks completed objects and enables resumability:

```
<renders_root>/<symmetry_type>/<object_id>/molmo_done.txt
```

### View-group filtering (`views_{N}`)

Each folder holds 456 images (114 views × 4 rotations). A group of **N** views
uses only the first N viewpoint indices (`IND_00` → `IND_{N-1}`), each with
their 4 rotations — giving **N × 4** images per config.

Supported groups: `6, 14, 26, 42, 62, 86, 114`

### JSON output fields

Each `.json` file contains:

| Field | Description |
|---|---|
| `filename` | Source image filename |
| `index` | Viewpoint index |
| `azimuth` | Azimuth angle (degrees) |
| `elevation` | Elevation angle (degrees) |
| `rotation_deg` | 2D rotation applied (0, 90, 180, 270) |
| `eye` | Camera position `[x, y, z]` |
| `R` | Camera rotation matrix |
| `T` | Camera translation vector |
| `prompt` | Prompt sent to the model |
| `raw_output` | Full decoded model output |
| `points` | List of `{obj_id, x, y}` dicts in 0–1000 coords |
| `success` | `true` if at least one point was detected |

Coordinates use Molmo's 0–1000 range with **Y-axis inverted** (origin at bottom-left).

---

## Usage

### Single GPU

```bash
CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_runner.py \
  --renders-root ../data/renders \
  --symmetry-type axis_sym
```

### Two GPUs — split by symmetry type (recommended)

Run each command in a separate `tmux` session:

```bash
# GPU 0 — axis symmetry
CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_runner.py \
  --renders-root ../data/renders \
  --symmetry-type axis_sym \
  --gpu-id 0 --num-gpus 2

# GPU 1 — plane symmetry
CUDA_VISIBLE_DEVICES=1 python MolmoPointing/molmo_runner.py \
  --renders-root ../data/renders \
  --symmetry-type plane_sym \
  --gpu-id 1 --num-gpus 2
```

### Two GPUs — split same symmetry type across both GPUs

```bash
CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_runner.py \
  --renders-root ../data/renders \
  --symmetry-type axis_sym \
  --gpu-id 0 --num-gpus 2

CUDA_VISIBLE_DEVICES=1 python MolmoPointing/molmo_runner.py \
  --renders-root ../data/renders \
  --symmetry-type axis_sym \
  --gpu-id 1 --num-gpus 2
```

Objects are distributed with round-robin assignment (`gpu_id=0` takes indices 0, 2, 4, …; `gpu_id=1` takes 1, 3, 5, …).

### Skip annotated images (faster, less disk usage)

```bash
CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_runner.py \
  --renders-root ../data/renders \
  --symmetry-type axis_sym \
  --no-vis
```

### Override sizes, lightings, and view groups

```bash
CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_runner.py \
  --renders-root ../data/renders \
  --symmetry-type axis_sym \
  --sizes 224 448 \
  --lightings flat \
  --view-groups 26 114
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--renders-root` | *(required)* | Root folder from `data_render.py` |
| `--symmetry-type` | *(required)* | `axis_sym` or `plane_sym` |
| `--gpu-id` | `0` | Index of this process (0-based) |
| `--num-gpus` | `1` | Total parallel processes |
| `--sizes` | `224 448 1024` | Image sizes to process |
| `--lightings` | `flat darker brighter` | Illumination modes |
| `--view-groups` | `6 14 26 42 62 86 114` | View-group sizes |
| `--no-vis` | `False` | Skip annotated PNG output |

---

## Resumability

- **Object-level:** if `molmo_done.txt` exists inside an object folder, that object is skipped entirely.
- To reprocess a specific object, delete its sentinel file and rerun:

```bash
rm ../data/renders/axis_sym/<object_id>/molmo_done.txt
```

- To reprocess everything, delete all sentinels:

```bash
find ../data/renders -name "molmo_done.txt" -delete
```

---

## Scale estimate

| Parameter | Value |
|---|---|
| Objects | 1,700 |
| View groups | 7 (6 → 114) |
| Sizes | 3 (224, 448, 1024) |
| Lightings | 3 (flat, darker, brighter) |
| Images per group (max) | 114 × 4 = 456 |
| **Upper bound (all groups)** | ~1,700 × 7 × 3 × 3 × avg(group×4) ≈ **10M+ inferences** |

Use `--no-vis` to avoid disk saturation at this scale, or restrict to a subset of sizes/lightings for initial experiments.

---

## tmux cheatsheet

```bash
# Create sessions
tmux new -s molmo_axis
tmux new -s molmo_plane

# Detach (keep running)
Ctrl+B, then D

# Reattach
tmux attach -t molmo_axis

# List sessions
tmux ls
```
