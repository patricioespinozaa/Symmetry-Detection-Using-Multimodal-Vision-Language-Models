# Symmetry Detection Using Multimodal Vision-Language Models

**Hypothesis:** Molmo2-8B (a VLM with pointing capability) can identify structural 3D symmetry points from 2D multi-view images without direct access to 3D geometry.

The model receives rendered views of a 3D object and returns 2D pointing coordinates. These points are then lifted to 3D via ray casting and used to fit the symmetry axis or plane, which is compared against the ground-truth ShapeNet labels.

---

## Repository structure

```
Symmetry-Detection-Using-Multimodal-Vision-Language-Models/
│
├── ImagesGenerator/               # Render .obj files to multi-view images
│   ├── export_fibonacci_views.py  # Fibonacci-sphere sampling; renders one object
│   └── README.md
│
├── MolmoPointing/                 # Molmo2-8B inference over rendered images
│   ├── molmo_multiview_runner.py  # Main batch runner
│   ├── prompts_registry.py        # Auto-discovers prompts from prompts/
│   ├── prompts/
│   │   ├── axis/                  # axis_v00–v05 × {single, multi}.txt
│   │   └── plane/                 # plane_v00–v05 × {single, multi}.txt
│   ├── Experiments.md             # Prompt table + full experiment loops
│   └── README.md
│
├── Mapping/                       # 2D predictions → 3D symmetry evaluation
│   ├── map_to_3d.py               # Ray casting: Molmo 2D coords → 3D hit points
│   ├── estimate_symmetry.py       # SVD fit: 3D points → axis or plane (4 methods)
│   ├── evaluate.py                # Metrics: angular error, AUC, precision@k, SDE
│   ├── compare_results.py         # Table + plots across experiments/methods
│   ├── visualize_rays.py          # Debug: Polyscope 3D ray viewer
│   ├── cleanup_experiments.py     # Remove n_views keys from JSON results
│   └── README.md
│
├── InteractiveViewer/             # Polyscope viewer for objects and GT symmetries
│   ├── view_symmetries.py
│   └── README.md
│
├── ExploratoryDataAnalysis/
│   └── EDA.ipynb
│
├── Examples/                      # One axis_sym + one plane_sym example object
│   ├── objects/
│   └── renders/
│
├── utils/                         # Shared utilities
│   ├── data_render.py             # Batch renderer (calls export_fibonacci_views.py)
│   └── ...
│
├── cleanup_all_experiments.sh     # Delete all experiment files for all prompts
└── find_extra_renders.sh          # Audit render directories vs source .obj files
```

---

## Data structure

```
data/                              (outside the repo, typically ../data/)
├── objects/
│   ├── curated_axis_sym_obj/      # 850 .obj meshes + .txt labels
│   └── curated_plane_sym_obj/     # 850 .obj meshes + .txt labels
│
└── renders/
    ├── axis_sym/
    │   └── <object_id>/
    │       └── <size>/<lighting>/
    │           ├── IND_00_AZ_090_EL_+89.png   # rendered images
    │           ├── ...
    │           ├── metadata_all.json           # viewpoint index, azimuth, elevation
    │           └── molmo_multiview[_EXP].json  # Molmo2 predictions + camera R,T,fov
    └── plane_sym/
        └── <object_id>/...
```

### True label format (`.txt`)

```
# axis_sym
1
axis DX DY DZ  OX OY OZ     ← direction + origin

# plane_sym (1–3 planes)
2
plane NX NY NZ  OX OY OZ
plane NX NY NZ  OX OY OZ
```

---

## Installation

```bash
conda create -n tesis_env python=3.10 -y
conda activate tesis_env

# PyTorch + PyTorch3D (CUDA 12.4)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

pip install pytorch3d==0.7.9+pt2.6.0cu124 \
  --extra-index-url https://miropsota.github.io/torch_packages_builder

# Molmo2 inference
pip install transformers==4.57.1 accelerate==1.10.1 \
  pillow==11.3.0 einops==0.8.1 tqdm==4.67.1 molmo_utils

# Mapping pipeline
pip install trimesh scipy pandas matplotlib

# Optional: 3D visualization (visualize_rays.py, InteractiveViewer/)
pip install polyscope
```

---

## Full pipeline (step by step)

### Step 1 — Render objects

```bash
# GPU 0: axis symmetry
CUDA_VISIBLE_DEVICES=0 python3 -m utils.data_render \
  --input-folder ../data/objects/curated_axis_sym_obj \
  --output-folder ../data/renders \
  --symmetry-type axis_sym \
  --repo-views 114 \
  --sizes 224 448 1136 \
  --lightings flat darker brighter

# GPU 1: plane symmetry
CUDA_VISIBLE_DEVICES=1 python3 -m utils.data_render \
  --input-folder ../data/objects/curated_plane_sym_obj \
  --output-folder ../data/renders \
  --symmetry-type plane_sym \
  --repo-views 114 \
  --sizes 224 448 1136 \
  --lightings flat darker brighter
```

Each object produces images under `<renders_root>/<symmetry_type>/<object_id>/<size>/<lighting>/`.

### Step 2 — Molmo2 inference (GPU required)

```bash
CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \
    --renders-root ../data/renders \
    --symmetry-type axis_sym \
    --sizes 224 --lightings flat \
    --view-groups 1 6 14 26 \
    --prompt-mode auto --yes
```

Results accumulate in `molmo_multiview.json`. Already-computed keys are skipped automatically. See `MolmoPointing/README.md` for multi-GPU and experiment variants.

### Step 3 — Map 2D points to 3D

```bash
python Mapping/map_to_3d.py \
    --renders-root ../data/renders \
    --objects-root ../data/objects \
    --symmetry-type axis_sym \
    --sizes 224 --lightings flat \
    --overwrite --yes
```

### Step 4 — Fit symmetry (4 methods)

```bash
python Mapping/estimate_symmetry.py \
    --renders-root ../data/renders \
    --objects-root ../data/objects \
    --symmetry-type axis_sym \
    --sizes 224 --lightings flat \
    --point-mode independent \
    --overwrite
```

`--point-mode` must match the prompt strategy (see `MolmoPointing/Experiments.md`):
- `independent` — points lie directly on the axis/plane
- `midpoint` — bilateral pairs; midpoint is computed before SVD

### Step 5 — Evaluate vs ground truth

```bash
for METHOD in svd ransac_svd svd_sde ransac_svd_sde; do
    python Mapping/evaluate.py \
        --renders-root ../data/renders \
        --objects-root ../data/objects \
        --symmetry-type axis_sym \
        --sizes 224 --lightings flat \
        --method $METHOD
done
```

### Step 6 — Compare results

```bash
python Mapping/compare_results.py \
    --renders-root ../data/renders \
    --symmetry-type axis_sym \
    --save-dir ../results/plots \
    --csv-dir ../results
```

---

## Running prompt experiments

To compare multiple prompt strategies, use `--experiment-id` to isolate results from production files. See `MolmoPointing/Experiments.md` for the full loop (axis + plane, all 6 prompts each).

```bash
EXP=axis_v02
MODE=independent   # from the prompt table in Experiments.md

CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \
    --renders-root ../data/renders --symmetry-type axis_sym \
    --sizes 224 --lightings flat --view-groups 1 6 14 26 \
    --prompt-id $EXP --experiment-id $EXP --prompt-mode auto --yes

python Mapping/map_to_3d.py \
    --renders-root ../data/renders --objects-root ../data/objects \
    --symmetry-type axis_sym --sizes 224 --lightings flat \
    --experiment-id $EXP --overwrite --yes

python Mapping/estimate_symmetry.py \
    --renders-root ../data/renders --objects-root ../data/objects \
    --symmetry-type axis_sym --sizes 224 --lightings flat \
    --experiment-id $EXP --point-mode $MODE --overwrite

for METHOD in svd ransac_svd svd_sde ransac_svd_sde; do
    python Mapping/evaluate.py \
        --renders-root ../data/renders --objects-root ../data/objects \
        --symmetry-type axis_sym --sizes 224 --lightings flat \
        --experiment-id $EXP --method $METHOD
done
```

Files with `--experiment-id` get an `_<ID>` suffix; production files are never touched.

To delete all experiment files and start clean:

```bash
bash cleanup_all_experiments.sh          # dry-run (shows counts)
bash cleanup_all_experiments.sh --delete # execute
```

---

## Key design choices

| Choice | Detail |
|---|---|
| **View sampling** | Fibonacci sphere — near-uniform angular coverage; 8 groups: 1, 6, 14, 26, 42, 62, 86, 114 views |
| **Molmo2 output scale** | Coordinates in [0, 1000], independent of image resolution |
| **Camera convention** | PyTorch3D row-vector: `p_cam = p_world @ R + T` |
| **Symmetry fitting** | SVD: axis = first PC (max variance), plane normal = last PC (min variance) |
| **RANSAC threshold** | 5% of point cloud bounding-box diagonal |
| **SDE** | `mean(2 × |signed distance to plane|) / bbox_diag` — normalized Symmetry Distance Error |
| **Evaluation** | Sign-agnostic angular error [0°, 90°]; best-match for plane_sym (up to 3 GT planes) |
| **Dataset** | ShapeNet, 850 objects per symmetry type, normalized to unit cube |

---

## tmux cheatsheet

Long jobs should run inside `tmux` so they survive terminal disconnections.

```bash
tmux new -s <session-name>   # create session
# run your command inside the session
Ctrl+B, then D               # detach (process keeps running)
tmux attach -t <name>        # reattach
tmux ls                      # list sessions
```

---

## Module documentation

| Module | README |
|---|---|
| Rendering | [ImagesGenerator/README.md](ImagesGenerator/README.md) |
| Molmo2 inference + experiments | [MolmoPointing/README.md](MolmoPointing/README.md) |
| Mapping + evaluation | [Mapping/README.md](Mapping/README.md) |
| Interactive viewer | [InteractiveViewer/README.md](InteractiveViewer/README.md) |
