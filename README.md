# Symmetry Detection Using Multimodal Vision-Language Models

**Hypothesis:** a VLM with pointing capability (Molmo2-8B, compared against MolmoPoint-8B and Qwen3-VL-8B) can identify structural 3D symmetry (axis or plane) from 2D multi-view renders of an object, without direct access to its 3D geometry.

The model receives rendered views of a 3D object and returns 2D points. In the **final method** those points are triangulated across views without the mesh: the symmetry axis is fitted with **EXP-E** (iterative residual reweighting) and the symmetry plane with the **S score** (`S_dat`, one plane per object). The prediction is compared against the ground-truth ShapeNet labels. Ray casting against the mesh (`Mapping/map_to_3d.py`) is kept as a comparison experiment.

---

## Repository structure

```
.
├── README.md  CLAUDE.md  environment.yml  .gitignore
│
├── ImagesGenerator/            # Render .obj meshes to multi-view images (Fibonacci sphere)
│   ├── export_fibonacci_views.py  # one object; --subset-n-views (partial render), --roll-seed
│   └── data_render.py             # batch renderer over a dataset folder
│
├── MolmoPointing/              # Molmo2-8B pointing: molmo_multiview_runner.py, prompts_registry.py, prompts/
├── MolmoPointPointing/         # MolmoPoint-8B pointing (same output schema)
├── QwenPointing/               # Qwen3-VL-8B pointing (same output schema)
├── pipeline_common/            # camera/NDC math, triangulation primitives, naming, datasets, clustering
│
├── Mapping/
│   ├── estimate_symmetry_no_mesh.py  # mesh-free triangulation (baseline estimator)
│   ├── evaluate.py                   # angular error, AUC, precision@θ, SDE_ref/F1_ref
│   ├── compare_results_no_mesh.py    # comparison CSVs across experiments
│   ├── map_to_3d.py  estimate_symmetry.py  compare_results.py   # with-mesh comparison
│   └── visualize_rays.py  visualize_symmetry.py  export_symmetry_overlay.py  render_symmetry_comparison.py
│
├── Pipeline_Experiments/
│   ├── estimate_symmetry_variants.py   # EXP-E axis estimator (triangulation_variants/)
│   ├── config.py  view_filtering.py
│   └── diagnostics/                    # center bias, repeated points, trivial baselines, missing reasons
│
├── PlaneScoreData/             # S plane estimator, run_*.py of every reported experiment,
│                               # diagnostico_*.py, run audit and tests
├── RandomBaseline/             # uniform random baseline (outputs/random_baseline_reference.csv)
├── ExploratoryDataAnalysis/    # dataset EDA, categories, orientation/visibility analysis
├── InteractiveViewer/          # Open3D viewer for meshes and GT symmetries
├── Examples/                   # 2 example objects + their 448 px renders
├── imgs/vistas_degeneradas/    # source of the degenerate-views figure
│
├── utils/
│   ├── generar_tablas_resultados_cap4.py  verificar_tablas_resultados_cap4.py  generar_anexo_prompts.py
│   ├── fig_camaras.py  copiar_vistas_degeneradas.py  verificar_convencion_ndc.py
│   └── rerun_ndc_fix.sh  muestras_render.sh
│
├── resultados_finales/         # CSVs, logs and paired comparisons read by the thesis tables
│   ├── <one folder per experiment>/
│   ├── diagnosticos/  referencias/
│   └── logs/  (logs/inferencia/<config>/ holds the VLM inference logs)
│
└── docs/                       # design docs, metrics, schemas, features, audits, archive/
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

# HDBSCAN clustering (with-mesh estimate_symmetry.py --clustering-method hdbscan)
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
CUDA_VISIBLE_DEVICES=0 python3 -m ImagesGenerator.data_render \
  --input-folder ../data/objects/curated_axis_sym_obj \
  --output-folder ../data/renders \
  --symmetry-type axis_sym \
  --repo-views 114 \
  --sizes 224 448 1136 \
  --lightings flat darker brighter
```

Same for `plane_sym` with `curated_plane_sym_obj`. Each object produces images under
`<renders_root>/<symmetry_type>/<object_id>/<size>/<lighting>/`.
`--subset-n-views 6 14 26` renders only the 36 views the pointing runners pick for those view
counts (same filenames and indices as the full 114-view sequence); `--roll-seed` rolls each
camera about its optical axis.

### Step 2 — VLM inference (GPU required)

```bash
CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \
    --renders-root ../data/renders \
    --symmetry-type axis_sym \
    --sizes 224 --lightings flat \
    --view-groups 6 14 26 \
    --prompt-id axis_v04 --experiment-id axis_v04 \
    --prompt-mode auto --yes
```

Results accumulate in `molmo_multiview[_<EXP>].json`; already-computed keys are skipped.
`QwenPointing/qwen3vl_multiview_runner.py` and `MolmoPointPointing/molmopoint_multiview_runner.py`
write the same schema. See `MolmoPointing/README.md` and
`docs/features/molmopoint-qwen3vl-pointing.md`.

### Step 3 — Fit symmetry without the mesh (final method)

```bash
# axis_sym: EXP-E
python Pipeline_Experiments/estimate_symmetry_variants.py \
    --renders-root ../data/renders --symmetry-type axis_sym \
    --sizes 224 --lightings flat \
    --experiment-id axis_v04 --variant expE --overwrite

# plane_sym: S score (S_dat), one plane per object
python PlaneScoreData/plane_score_data.py \
    --renders-root ../data/renders \
    --input-id plane_v02 --output-id plane_v02_sdat \
    --criterio datos --size 224 --lighting flat --overwrite
```

The baseline mesh-free estimator (`Mapping/estimate_symmetry_no_mesh.py`, SVD triangulation)
is documented in [docs/pipeline_sin_malla.md](docs/pipeline_sin_malla.md) and
[docs/implementacion_pipeline_sin_malla.md](docs/implementacion_pipeline_sin_malla.md).
All of them write `predicted_symmetry_<EXP>.json` under the method key `triangulation`
(or `triangulation_multiplane` with `--max-planes > 1`) — see [docs/data-schemas.md](docs/data-schemas.md).

### Step 4 — Evaluate vs ground truth

```bash
python Mapping/evaluate.py \
    --renders-root ../data/renders --objects-root ../data/objects \
    --symmetry-type axis_sym --sizes 224 --lightings flat \
    --experiment-id axis_v04_expE_nomesh \
    --method triangulation --with-reference-metrics
```

`--with-reference-metrics` adds SDE_ref/F1_ref and requires `gpytoolbox`.

### Step 5 — Compare experiments

```bash
python Mapping/compare_results_no_mesh.py \
    --renders-root ../data/renders --symmetry-type axis_sym \
    --experiment-id axis_v04_expE_nomesh --csv-dir resultados_finales/<folder> --no-plots
```

### Step 6 — Reproduce the thesis results

```bash
# Re-estimates every final configuration (EXP-E in axis, S in plane) from the points already
# inferred, evaluates and compares; writes to resultados_finales/<folder>/
bash PlaneScoreData/run_resultados_finales.sh

# Each additional experiment has its own runner in PlaneScoreData/ (resolution, lighting,
# flows, sending mode, VLMs, mesh, random silhouette, multi-plane, SO(3)/Aguirre rotations,
# roll, view oracle), most with the subcommands `chequeo` (read-only) and `todo`, e.g.:
python3 PlaneScoreData/run_plane_v02_224.py chequeo

# Thesis tables (chapter 4) and prompt appendix, read from resultados_finales/
python utils/generar_tablas_resultados_cap4.py
python utils/verificar_tablas_resultados_cap4.py
python utils/generar_anexo_prompts.py
```

---

## With-mesh comparison

`Mapping/map_to_3d.py` casts each 2D point onto the mesh and `Mapping/estimate_symmetry.py`
fits the axis/plane on the 3D hits (`svd`, `ransac_svd`, `svd_sde`, `ransac_svd_sde`, with
optional patch backprojection and clustering). The reported comparison is run by
`PlaneScoreData/run_axis_malla.py` and `PlaneScoreData/run_plane_malla.py`; see
`Mapping/README.md` for the individual commands. `--point-mode` must match the prompt strategy:
`independent` (points on the axis/plane) or `midpoint` (bilateral pairs).

---

## Key design choices

| Choice | Detail |
|---|---|
| **View sampling** | Fibonacci sphere, 114 views; groups of 6, 14 and 26 views picked with `round(linspace(0, 113, n))` |
| **VLM output scale** | Coordinates in [0, 1000], independent of image resolution |
| **Camera convention** | PyTorch3D row-vector `p_cam = p_world @ R + T`; Molmo→NDC with +X to the left (fixed 2026-10-05, ids `*_ndcfix`) |
| **Axis estimator** | EXP-E: weighted triangulation of the interpretation planes with iterative residual reweighting |
| **Plane estimator** | S score (`S_dat`): candidate normal chosen by agreement with the data, one plane per object |
| **Evaluation** | Sign-agnostic angular error [0°, 90°]; AUC truncated at 45°; objects without prediction count as 90° |
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
| Molmo2 inference | [MolmoPointing/README.md](MolmoPointing/README.md) |
| MolmoPoint / Qwen3-VL inference | [docs/features/molmopoint-qwen3vl-pointing.md](docs/features/molmopoint-qwen3vl-pointing.md), [docs/instalacion_molmopoint_qwen3vl.md](docs/instalacion_molmopoint_qwen3vl.md) |
| Final experiments (S estimator, runners, diagnostics) | `PlaneScoreData/` (header of each script) |
| Random baseline | [RandomBaseline/README.md](RandomBaseline/README.md) |
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
