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
│   ├── data_render.py             # Batch renderer (calls export_fibonacci_views.py)
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
│   ├── estimate_symmetry.py       # SVD/RANSAC fit (mesh-based): 3D points → axis or plane (4 methods)
│   ├── estimate_symmetry_no_mesh.py # Mesh-free fit via multi-view ray triangulation (up to 3 planes)
│   ├── evaluate.py                # Metrics: angular error, AUC, precision@k, SDE_ref, F1_ref (+ bulk --all mode)
│   ├── compare_results.py         # Table + plots across experiments/methods
│   ├── visualize_rays.py          # Debug: Polyscope 3D ray viewer
│   ├── visualize_symmetry.py      # Debug: Polyscope GT vs. predicted symmetry viewer
│   ├── export_symmetry_overlay.py # Batch-export overlay renders
│   ├── export_viz_samples.py      # Batch-export sample visualizations + README
│   ├── render_symmetry_comparison.py # Side-by-side comparison renders
│   ├── run_all_postprocessing.py  # Sweep entrypoint: map_to_3d → estimate → evaluate
│   ├── cleanup_experiments.py     # Remove n_views keys from JSON results
│   ├── cleanup_all_experiments.sh # Delete all experiment files for all prompts
│   ├── find_extra_renders.sh      # Audit render directories vs source .obj files
│   ├── archive/                   # Superseded one-off diagnostic scripts, kept for reference
│   │   ├── audit_view_indices.py       # superseded by audit_view_indices_v2.py
│   │   └── check_origin_compactness.py # superseded by check_origin_full.py
│   ├── audit_view_indices_v2.py, check_independent_vs_midpoint.py,
│   │   check_origin_full.py, check_sde_vs_angular.py  # ad hoc analysis scripts
│   │   tied to specific thesis findings — see each file's docstring
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
├── pipeline_common/                # Shared helpers used across all stages
│   ├── naming.py                   # --experiment-id filename suffixing
│   ├── datasets.py                 # OBJECTS_SUBDIR + mesh loading
│   ├── camera.py                   # NDC/ray math, ray casting (exact + patch)
│   ├── clustering.py                # Greedy centroid + HDBSCAN clustering
│   ├── view_selection.py           # Seeded description-view selection (Flow B/C)
│   └── triangulation.py            # Ray/interpretation-plane primitives (mesh-free pipeline)
│
└── docs/                          # Design docs, metric definitions, audits (see below)
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

# HDBSCAN clustering (estimate_symmetry.py --clustering-method hdbscan)
pip install scikit-learn

# Optional: reference metrics (evaluate.py --with-reference-metrics / --all / --experiment-ids)
pip install gpytoolbox rtree

# Optional: 3D visualization (visualize_rays.py, visualize_symmetry.py, render_symmetry_comparison.py)
pip install polyscope

# Optional: 3D visualization (InteractiveViewer/)
pip install open3d
```

Or use the pinned environment directly (Linux/CUDA 12.4 only — captured from the
remote GPU server, will not install as-is on Windows/CPU-only machines):

```bash
conda env create -f environment.yml
```

---

## Full pipeline (step by step)

### Step 1 — Render objects

```bash
# GPU 0: axis symmetry
CUDA_VISIBLE_DEVICES=0 python3 -m ImagesGenerator.data_render \
  --input-folder ../data/objects/curated_axis_sym_obj \
  --output-folder ../data/renders \
  --symmetry-type axis_sym \
  --repo-views 114 \
  --sizes 224 448 1136 \
  --lightings flat darker brighter

# GPU 1: plane symmetry
CUDA_VISIBLE_DEVICES=1 python3 -m ImagesGenerator.data_render \
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

## Alternate pipeline: mesh-free (no ray-casting against the mesh)

Steps 1–2 (render + Molmo2 inference) are identical. From there, this variant
skips `map_to_3d.py`/`estimate_symmetry.py` entirely and estimates the
axis/plane directly from multi-view ray triangulation — see
[docs/pipeline_sin_malla.md](docs/pipeline_sin_malla.md) and
[docs/implementacion_pipeline_sin_malla.md](docs/implementacion_pipeline_sin_malla.md)
for the full design rationale.

### Step 3' — Fit symmetry without the mesh

```bash
# axis_sym
python Mapping/estimate_symmetry_no_mesh.py \
    --renders-root ../data/renders \
    --symmetry-type axis_sym \
    --sizes 224 --lightings flat \
    --overwrite

# plane_sym, single best-fit plane (method "triangulation")
python Mapping/estimate_symmetry_no_mesh.py \
    --renders-root ../data/renders \
    --symmetry-type plane_sym \
    --sizes 224 --lightings flat \
    --max-planes 1 --overwrite

# plane_sym, up to 3 candidate planes (method "triangulation_multiplane")
python Mapping/estimate_symmetry_no_mesh.py \
    --renders-root ../data/renders \
    --symmetry-type plane_sym \
    --sizes 224 --lightings flat \
    --max-planes 3 --overwrite
```

Writes `predicted_symmetry[_<EXP>].json` under the method key `triangulation`
(single axis/plane) or `triangulation_multiplane` (`--max-planes > 1`, a list
of candidate planes) — see [docs/data-schemas.md](docs/data-schemas.md).

### Step 4' — Evaluate

```bash
# axis_sym / plane_sym, single plane — same angular-error metrics as the with-mesh path
python Mapping/evaluate.py \
    --renders-root ../data/renders \
    --objects-root ../data/objects \
    --symmetry-type axis_sym \
    --sizes 224 --lightings flat \
    --method triangulation --with-reference-metrics

# plane_sym, multi-plane — recall/precision over the full GT plane set instead
# of a single angular error, plus per-plane SDE_ref/F1_ref
python Mapping/evaluate.py \
    --renders-root ../data/renders \
    --objects-root ../data/objects \
    --symmetry-type plane_sym \
    --sizes 224 --lightings flat \
    --method triangulation_multiplane --with-reference-metrics
```

`--with-reference-metrics` requires `gpytoolbox` (see Installation below).
`compare_results.py` (Step 6) works the same way against these outputs.

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
bash Mapping/cleanup_all_experiments.sh          # dry-run (shows counts)
bash Mapping/cleanup_all_experiments.sh --delete # execute
```

---

## Experimental matrix: flows, clustering, backprojection

Beyond the base Flow-A prompt experiments above, the methodology defines three
orthogonal axes of variation:

| Axis | Options | Where |
|---|---|---|
| **Flow** (VLM prompting) | `a` direct pointing, `b` pointing + description, `c` description + pointing integrados | `--flow` in `molmo_multiview_runner.py` — see `MolmoPointing/README.md` |
| **Clustering** | `none`, `greedy`, `hdbscan` (`min_samples` 2/3/5) | `--clustering-method` in `estimate_symmetry.py` — see `Mapping/README.md` |
| **Backprojection** | `1` (exact), `3`, `5` (averaged patch) | `--patch-size` in `map_to_3d.py` — see `Mapping/README.md` |

These compose with the existing `--experiment-id` isolation pattern (a distinct
ID per flow keeps artifacts separate; clustering/patch-size append their own
tag to the output filename automatically).

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
| **Description-view selection** (Flow B/C) | Seeded random pick per object (`MD5(object_id) mod 2^31`), filtered to elevation in (-60°, +60°) |
| **Clustering** | Greedy centroid (5% bbox diag, always assigns) or HDBSCAN (`min_samples` 2/3/5, drops noise) |
| **Patch backprojection** | Average 3D hits of a `h x h` sub-ray grid (`h` in {1, 3, 5}) around each point |

---

## Additional tools

Maintenance and diagnostic scripts that sit outside the main step-by-step flow:

```bash
# Sweep entrypoint: chains map_to_3d.py -> estimate_symmetry.py -> evaluate.py
# for every method/n_views combination in one call
python Mapping/run_all_postprocessing.py \
    --renders-root ../data/renders --objects-root ../data/objects \
    --symmetry-type axis_sym --sizes 224 --lightings flat

# Remove n_views keys from accumulated JSON results (e.g. after a bad partial run)
python Mapping/cleanup_experiments.py --renders-root ../data/renders --symmetry-type axis_sym

# Audit render directories vs. source .obj files (find missing/extra renders)
bash Mapping/find_extra_renders.sh
```

`Mapping/audit_view_indices_v2.py`, `Mapping/check_independent_vs_midpoint.py`,
`Mapping/check_sde_vs_angular.py`, and `Mapping/check_origin_full.py` are ad hoc
analysis scripts tied to specific thesis findings (view-index consistency,
`--point-mode` comparison, SDE-vs-angular-error correlation) — see each
script's own docstring for its exact input/output and when to use it.
`Mapping/archive/` holds their now-superseded predecessors, kept for reference
only (see [docs/audits/refactor-log.md](docs/audits/refactor-log.md)).

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
| Mesh-free pipeline design | [docs/pipeline_sin_malla.md](docs/pipeline_sin_malla.md), [docs/implementacion_pipeline_sin_malla.md](docs/implementacion_pipeline_sin_malla.md) |
| Metrics: definitions + history of changes | [docs/metricas_evaluacion.md](docs/metricas_evaluacion.md), [docs/actualizacion_metricas.md](docs/actualizacion_metricas.md) |
| JSON schema of every accumulative pipeline file | [docs/data-schemas.md](docs/data-schemas.md) |
| Per-feature methodology docs (description, key files/functions, inputs/outputs, design decisions, known limitations) | [docs/features/](docs/features/) |
| Python code style/norms | [docs/code-norms.md](docs/code-norms.md) |
| Thesis context | [docs/Contexto.md](docs/Contexto.md) |
| Architecture audits / refactor log | [docs/audits/](docs/audits/) |
| Archived/historical notes (superseded, kept for reference) | [docs/archive/](docs/archive/) |
