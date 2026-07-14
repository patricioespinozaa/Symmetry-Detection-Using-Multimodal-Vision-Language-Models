# Mapping — 2D Predictions to 3D Symmetry Evaluation

This folder contains the pipeline to project Molmo2 pointing predictions onto the 3D mesh, estimate the predicted symmetry element (axis or plane), and evaluate it against ground-truth labels.

---

## Pipeline overview

```
molmo_multiview[_EXP].json         .obj mesh + .txt labels
(Molmo2 2D predictions +           (3D geometry + ground truth)
 camera R, T, fov per view)
         └──────────────────────────┬───────────────────────┘
                                    ▼
                            map_to_3d.py
                     (ray casting: 2D coords → 3D points)
                                    │
                                    ▼
                    mapped_points_3d[_EXP].json
                                    │
                                    ▼
                        estimate_symmetry.py
                    (SVD fit: 3D points → axis or plane)
                    (4 methods: svd, ransac_svd, svd_sde, ransac_svd_sde)
                                    │
                                    ▼
                    predicted_symmetry[_EXP].json
                                    │
                                    ▼
                              evaluate.py
                    (angular error, translation error, AUC, precision@k)
                                    │
                                    ▼
            eval_{sizes}_{lightings}[_EXP]_{method}_results.json
            eval_{sizes}_{lightings}[_EXP]_{method}_summary.csv
                                    │
                                    ▼
                          compare_results.py
                    (table + plots across experiments/methods)
```

Camera parameters (R, T, fov) are embedded inside `molmo_multiview.json` — no separate metadata file is needed.

---

## File structure

```
Mapping/
├── map_to_3d.py             # Ray casting: Molmo 2D coords → 3D mesh surface
├── estimate_symmetry.py     # SVD fit: 3D points → predicted axis or plane (4 methods)
├── evaluate.py              # Metrics: predicted symmetry vs true labels
├── compare_results.py       # Aggregate and plot results across experiments/methods
├── visualize_rays.py        # Debug tool: visualize camera rays and hit points in 3D
├── cleanup_experiments.py   # Remove specific n_views keys from molmo JSON results
└── export_viz_samples.py    # Export best/worst-N objects + ready-to-run viz commands
```

Camera-ray math, mesh loading, `--experiment-id` filename suffixing, and point-cloud
clustering (greedy + HDBSCAN) are shared helpers in `../pipeline_common/`, imported by
every script above.

---

## Input requirements

| Source | Location |
|---|---|
| Molmo2 predictions + camera params | `<renders_root>/<symmetry_type>/<object_id>/<size>/<lighting>/molmo_multiview[_EXP].json` |
| 3D meshes | `<objects_root>/curated_axis_sym_obj/<object_id>.obj` |
| True labels | `<objects_root>/curated_axis_sym_obj/<object_id>.txt` |

### True label format

**axis_sym:**
```
1                                            ← number of axes
axis DX DY DZ  OX OY OZ                     ← direction + origin
N_ANGLES
angles A1 A2 ...                             ← rotational symmetry angles (not used)
```

**plane_sym** (supports 1–3 planes per object):
```
2                                            ← number of planes
plane NX NY NZ  OX OY OZ                    ← normal + origin
plane NX NY NZ  OX OY OZ
```

Direction/normal vectors are normalized automatically on load.

---

## Coordinate conventions

| Space | Origin | Scale |
|---|---|---|
| Molmo2 output | Top-left | 0–1000 (independent of image resolution) |
| NDC (camera) | Center | −1 to +1 |
| World / mesh | Mesh centroid ≈ origin | ShapeNet unit cube (~1.0) |

ShapeNet meshes are pre-normalized to fit within a unit cube centered at the origin.

---

## 1. `map_to_3d.py`

Projects each Molmo2 predicted point onto the 3D mesh surface via ray casting.

**Method:** reconstructs the camera ray from `R`, `T`, and `fov=60°` stored in `molmo_multiview.json`, then intersects it with the mesh using `trimesh`. Only ROT_000 images are used (one view per viewpoint index).

**Patch-based backprojection** (`--patch-size 3`/`5`): instead of a single exact ray, averages the 3D hit points of a `patch_size x patch_size` grid of sub-rays around each point. Stabilizes the result against grazing-angle localization noise (small pixel errors near-tangent to the surface otherwise produce large 3D jumps). Default `--patch-size 1` is the original exact single-ray behavior, unchanged. Output goes to a separate `mapped_points_3d_p{patch_size}[_EXP].json` file.

**Output:** `mapped_points_3d[_EXP].json` alongside the renders. Skips objects that already have a result file (unless `--overwrite`).

### Usage

```bash
python Mapping/map_to_3d.py \
    --renders-root ../data/renders \
    --objects-root ../data/objects \
    --symmetry-type axis_sym \
    --sizes 224 --lightings flat \
    --overwrite --yes

# Experiment variant
python Mapping/map_to_3d.py \
    --renders-root ../data/renders \
    --objects-root ../data/objects \
    --symmetry-type axis_sym \
    --sizes 224 --lightings flat \
    --experiment-id axis_v02 --overwrite --yes

# Patch-based backprojection (h=3)
python Mapping/map_to_3d.py \
    --renders-root ../data/renders \
    --objects-root ../data/objects \
    --symmetry-type axis_sym \
    --sizes 224 --lightings flat \
    --patch-size 3 --overwrite --yes
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--renders-root` | *(required)* | Root folder of renders |
| `--objects-root` | *(required)* | Root folder containing `curated_*_obj/` subfolders |
| `--symmetry-type` | *(required)* | `axis_sym` or `plane_sym` |
| `--sizes` | `224 448 1136` | Image sizes to process |
| `--lightings` | `flat brighter darker` | Illumination modes |
| `--fov` | `60.0` | Field of view in degrees |
| `--patch-size` | `1` | `1` (exact), `3`, or `5` (averaged sub-ray patch) |
| `--experiment-id` | `None` | Reads `molmo_multiview_<ID>.json`, writes `mapped_points_3d_<ID>.json` |
| `--max-objects` | `None` | Process only the first N objects |
| `--overwrite` | `False` | Overwrite existing output files |
| `--yes` / `-y` | `False` | Skip interactive confirmation |
| `--gpu-id` | `0` | Process index for round-robin object splitting |
| `--num-gpus` | `1` | Total parallel processes |

### Output format (`mapped_points_3d.json`)

```json
{
  "object_id": "1a9c1cbf...",
  "symmetry_type": "axis_sym",
  "image_size": 224,
  "illumination": "flat",
  "fov_deg": 60.0,
  "patch_size": 1,
  "n_views_results": {
    "6": {
      "images_sent": [...],
      "points_3d": [
        {
          "img_idx": 0,
          "obj_id": 1,
          "molmo_x": 450.0,
          "molmo_y": 230.0,
          "hit": true,
          "point_3d": [0.12, -0.31, 0.05],
          "face_id": 42
        }
      ],
      "n_hits": 2,
      "n_misses": 0
    }
  }
}
```

> When `--patch-size` is `3` or `5`, each point entry above gains `"patch_size"`,
> `"n_patch_hits"`, and `"n_patch_total"` fields (how many of the `patch_size²` sub-rays hit).

---

## 2. `estimate_symmetry.py`

Fits a symmetry axis or plane from the 3D hit points using four methods: plain SVD, RANSAC+SVD, SVD+SDE selection, and RANSAC+SVD+SDE selection.

**SVD fit:**
- `axis_sym`: first principal component of centered hit points = axis direction; origin = centroid.
- `plane_sym`: last principal component (min-variance direction) = plane normal; origin = centroid.

**Point mode** (`--point-mode`) controls how Molmo2 points are treated before SVD:
- `independent`: each 3D hit point enters SVD directly. Requires obj_id 1 AND 2 to both hit per image (images with only a partial hit are discarded); any obj_id beyond 2 is ignored. Use with Flow A/B prompts, which always return exactly that pair.
- `midpoint`: obj_id=1 and obj_id=2 per image are replaced by their 3D midpoint. Use this for bilateral-pair prompts (e.g., `axis_v01`, `plane_v01`) so that midpoints lie on the axis/plane instead of opposite sides.
- `all`: every hit point enters SVD directly, any obj_id count per image, no pairing requirement — an image contributes whatever subset of points it has, even if others were occluded. Required for **Flow C**, whose pre-pass returns a variable number of labeled points (typically 4-10), not a fixed pair; `independent`/`midpoint` would silently drop everything past obj_id 2.

**SDE variants** (`svd_sde`, `ransac_svd_sde`) compute the Symmetry Distance Error against the mesh and store it in the output. Requires `--objects-root`.

Points from all (size × lighting) configurations are pooled per `n_views` group before fitting.

**Clustering** (`--clustering-method`) consolidates the pooled point cloud before fitting, to reduce spatial redundancy when multiple views observe the same surface region:
- `none` (default) — no clustering.
- `greedy` — centroid-based; every point is always assigned to some cluster (threshold = 5% of the point-cloud bbox diagonal). Output suffix `_cluster`. (`--clustering`, without a value, is a back-compat alias for this.)
- `hdbscan` — density-based (`--hdbscan-min-samples`, sweep 2/3/5); explicitly drops sparse/isolated points as noise before fitting, unlike greedy. Output suffix `_hdbscan_ms{N}`.

**Output:** `predicted_symmetry[_EXP].json` at the object level. Skips objects where the file already exists (unless `--overwrite`).

### Usage

```bash
python Mapping/estimate_symmetry.py \
    --renders-root ../data/renders \
    --objects-root ../data/objects \
    --symmetry-type axis_sym \
    --sizes 224 --lightings flat \
    --point-mode independent \
    --overwrite

# Experiment variant (bilateral-pair prompt)
python Mapping/estimate_symmetry.py \
    --renders-root ../data/renders \
    --objects-root ../data/objects \
    --symmetry-type axis_sym \
    --sizes 224 --lightings flat \
    --experiment-id axis_v01 --point-mode midpoint --overwrite

# HDBSCAN clustering sweep
python Mapping/estimate_symmetry.py \
    --renders-root ../data/renders \
    --objects-root ../data/objects \
    --symmetry-type axis_sym \
    --sizes 224 --lightings flat \
    --clustering-method hdbscan --hdbscan-min-samples 3
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--renders-root` | *(required)* | Root folder of renders |
| `--objects-root` | `None` | Root folder with `.obj` meshes (required for SDE computation) |
| `--symmetry-type` | *(required)* | `axis_sym` or `plane_sym` |
| `--sizes` | `224 448 1136` | Sizes to pool points from |
| `--lightings` | `flat brighter darker` | Lightings to pool points from |
| `--point-mode` | `independent` | `independent`, `midpoint`, or `all` (Flow C) — see prompt table in `Experiments.md` |
| `--clustering-method` | `none` | `none`, `greedy`, or `hdbscan` |
| `--hdbscan-min-samples` | `3` | `min_samples` for `--clustering-method hdbscan` (sweep 2, 3, 5) |
| `--clustering` | `False` | Deprecated alias for `--clustering-method greedy` |
| `--experiment-id` | `None` | Reads `mapped_points_3d_<ID>.json`, writes `predicted_symmetry_<ID>.json` |
| `--max-objects` | `None` | Process only the first N objects |
| `--overwrite` | `False` | Overwrite existing output files |
| `--gpu-id` | `0` | Process index for round-robin splitting |
| `--num-gpus` | `1` | Total parallel processes |

### Output format (`predicted_symmetry.json`)

```json
{
  "object_id": "1a9c1cbf...",
  "symmetry_type": "axis_sym",
  "point_mode": "independent",
  "clustering_method": "none",
  "hdbscan_min_samples": null,
  "n_views_predictions": {
    "6": {
      "svd":            {"direction": [dx, dy, dz], "origin": [ox, oy, oz],
                         "n_points": 12, "n_inliers": null, "sde": null,  "accepted": null},
      "ransac_svd":     {"direction": [...],         "origin": [...],
                         "n_points": 12, "n_inliers": 8,    "sde": null,  "accepted": null},
      "svd_sde":        {"direction": [...],         "origin": [...],
                         "n_points": 12, "n_inliers": null, "sde": 0.031, "accepted": true},
      "ransac_svd_sde": {"direction": [...],         "origin": [...],
                         "n_points": 12, "n_inliers": 8,    "sde": 0.021, "accepted": true}
    }
  }
}
```

For `plane_sym`, `"direction"` is replaced by `"normal"`. When `--objects-root` is omitted, `sde` and `accepted` are `null`.

---

## 3. `evaluate.py`

Compares the predicted symmetry element against the ground-truth label and computes evaluation metrics.

**Metrics:**

| Metric | axis_sym | plane_sym |
|---|---|---|
| Angular error (°) | Angle between predicted and true axis directions | Angle between predicted and true plane normals |
| Translation error | Point-to-line distance: predicted origin → true axis | Point-to-plane distance: predicted origin → true plane |
| AUC angular | Area under precision-vs-threshold curve (0–90°) | Same |
| Precision @ 5°/10°/15° | Fraction of objects with angular error < threshold | Same |
| SDE mean / AUC SDE / Precision @ SDE | — | Symmetry Distance Error metrics |

All metrics are **sign-agnostic**. For `plane_sym` with multiple true planes (up to 3), the predicted plane is matched against the **closest true plane** by angular error.

**Output:** saves results under `<renders_root>/<symmetry_type>/`. Always rewrites.

### Usage

```bash
python Mapping/evaluate.py \
    --renders-root ../data/renders \
    --objects-root ../data/objects \
    --symmetry-type axis_sym \
    --sizes 224 --lightings flat \
    --method svd

# All 4 methods (loop)
for METHOD in svd ransac_svd svd_sde ransac_svd_sde; do
    python Mapping/evaluate.py \
        --renders-root ../data/renders \
        --objects-root ../data/objects \
        --symmetry-type axis_sym \
        --sizes 224 --lightings flat \
        --method $METHOD
done

# Experiment variant
python Mapping/evaluate.py \
    --renders-root ../data/renders \
    --objects-root ../data/objects \
    --symmetry-type axis_sym \
    --sizes 224 --lightings flat \
    --experiment-id axis_v02 --method ransac_svd_sde
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--renders-root` | *(required)* | Root folder of renders |
| `--objects-root` | *(required)* | Root folder with `.txt` labels |
| `--symmetry-type` | *(required)* | `axis_sym` or `plane_sym` |
| `--method` | *(required)* | `svd`, `ransac_svd`, `svd_sde`, or `ransac_svd_sde` |
| `--sizes` | `224 448 1136` | Must match what was used in `estimate_symmetry.py` (affects output filename) |
| `--lightings` | `flat brighter darker` | Must match what was used in `estimate_symmetry.py` |
| `--experiment-id` | `None` | Reads `predicted_symmetry_<ID>.json`, writes `eval_*_<ID>_*` files |
| `--max-objects` | `None` | Process only the first N objects |

### Output files

Saved under `<renders_root>/<symmetry_type>/`:

**`eval_{sizes}_{lightings}[_EXP]_{method}_results.json`** — per-object metrics:
```json
{
  "symmetry_type": "axis_sym",
  "method": "svd",
  "experiment_id": "axis_v02",
  "objects": {
    "1a9c1cbf...": {
      "6":  {"angular_error_deg": 5.1, "translation_error": 0.018,
             "n_points": 6, "status": "ok"},
      "14": { ... }
    }
  }
}
```

**`eval_{sizes}_{lightings}[_EXP]_{method}_summary.csv`** — aggregated per n_views group:

| n_views | n_objects | angular_error_mean | angular_error_median | angular_error_std | translation_error_mean | ... | auc_angular | n_points_mean | precision_5deg | precision_10deg | precision_15deg |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 850 | 11.2 | 8.4 | 9.3 | 0.038 | ... | 0.72 | 5.8 | 0.41 | 0.65 | 0.79 |

For `plane_sym`, additional columns: `sde_mean`, `sde_median`, `sde_std`, `auc_sde`, `precision_sde_010`, `precision_sde_020`.

---

## 4. `compare_results.py`

Reads all evaluation CSVs for a symmetry type and generates a console table, bar charts, line plots, and a heatmap comparing experiments and methods.

### Usage

```bash
# Console table only
python Mapping/compare_results.py \
    --renders-root ../data/renders \
    --symmetry-type axis_sym

# Save plots and CSV
python Mapping/compare_results.py \
    --renders-root ../data/renders \
    --symmetry-type axis_sym \
    --save-dir ../results/plots \
    --csv-dir ../results
```

The CSV is saved as `<csv-dir>/experiments_DD_MM_YYYY/<symmetry_type>_comparison.csv`.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--renders-root` | *(required)* | Root folder of renders |
| `--symmetry-type` | *(required)* | `axis_sym` or `plane_sym` |
| `--sizes` | `224` | Must match what was used to run evaluate.py |
| `--lightings` | `flat` | Must match what was used to run evaluate.py |
| `--save-dir` | `None` | Directory to save plots. If omitted, displays on screen |
| `--csv-dir` | `None` | Base directory to save combined CSV |
| `--no-plots` | `False` | Print table only, skip plots |

### Outputs

- **Console table:** experiment × method × n_views with key metrics
- **`{sym}_n_objects.png`:** bar chart of valid-prediction counts per experiment/method
- **`{sym}_{metric}.png`:** line plots of each metric vs n_views, one subplot per method
- **`{sym}_heatmap_precision5.png`:** heatmap of precision@5° for each experiment × method

---

## 5. `cleanup_experiments.py`

Removes specific `n_views` keys from `molmo_multiview[_EXP].json` files. Use this when iterating on prompts to delete only the results you want to rerun without affecting other keys in the same JSON.

- If the JSON still has other keys after removal → rewritten without the removed keys
- If the JSON becomes empty → file deleted entirely

**Always run with `--dry-run` first** to preview what would be changed.

### Usage

```bash
# Preview (no changes)
python Mapping/cleanup_experiments.py \
    --renders-root ../data/renders \
    --symmetry-type axis_sym \
    --sizes 224 --lightings flat \
    --view-groups 1 6 14 26 \
    --dry-run

# Execute (production file)
python Mapping/cleanup_experiments.py \
    --renders-root ../data/renders \
    --symmetry-type axis_sym \
    --sizes 224 --lightings flat \
    --view-groups 1 6 14 26

# Experiment file
python Mapping/cleanup_experiments.py \
    --renders-root ../data/renders \
    --symmetry-type axis_sym \
    --sizes 224 --lightings flat \
    --view-groups 1 6 \
    --experiment-id axis_v02
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--renders-root` | *(required)* | Root folder of renders |
| `--symmetry-type` | *(required)* | `axis_sym` or `plane_sym` |
| `--view-groups` | *(required)* | Keys to remove (e.g. `1 6 14 26`) |
| `--sizes` | `224 448 1136` | Sizes to clean |
| `--lightings` | `flat brighter darker` | Lightings to clean |
| `--experiment-id` | `None` | Targets `molmo_multiview_<ID>.json` instead of `molmo_multiview.json` |
| `--dry-run` | `False` | Preview without making changes |

> **Note:** this script only cleans `molmo_multiview[_EXP].json`. Downstream files (`mapped_points_3d`, `predicted_symmetry`, `eval_*`) must be removed separately if needed.

---

## 6. `visualize_rays.py`

Debugging tool that recomputes all camera rays from scratch and shows them in a 3D Polyscope viewer.

**Requires:** `polyscope` (`pip install polyscope`)

**Shows:**
- Gray mesh (3D object)
- Blue spheres (camera positions, one per view)
- Green lines (hit rays: camera → mesh intersection)
- Orange lines (miss rays: camera → extended endpoint)
- Yellow spheres (hit points on mesh surface)
- Ground-truth axis or plane when `--show-gt` is given

### Usage

```bash
python Mapping/visualize_rays.py \
    --object-id 1a9c1cbf1ca9ca24274623f5a5d0bcdc \
    --renders-root ../data/renders \
    --objects-root ../data/objects \
    --symmetry-type axis_sym \
    --n-views 6 \
    --show-gt

# Experiment variant
python Mapping/visualize_rays.py \
    --object-id 1a9c1cbf1ca9ca24274623f5a5d0bcdc \
    --renders-root ../data/renders \
    --objects-root ../data/objects \
    --symmetry-type axis_sym \
    --n-views 6 \
    --experiment-id axis_v02 \
    --show-gt

# Inspect a patch-backprojected / HDBSCAN-clustered run (flags must match
# whatever produced that experiment via map_to_3d.py / estimate_symmetry.py)
python Mapping/visualize_rays.py \
    --object-id 1a9c1cbf1ca9ca24274623f5a5d0bcdc \
    --renders-root ../data/renders \
    --objects-root ../data/objects \
    --symmetry-type axis_sym \
    --n-views 26 \
    --patch-size 3 \
    --show-clusters --clustering-method hdbscan --hdbscan-min-samples 3
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--object-id` | *(required)* | Object ID to visualize |
| `--renders-root` | *(required)* | Root folder of renders |
| `--objects-root` | *(required)* | Root folder with `.obj` and `.txt` files |
| `--symmetry-type` | *(required)* | `axis_sym` or `plane_sym` |
| `--n-views` | *(required)* | n_views group to visualize (e.g. `6`) |
| `--size` | `224` | Image size |
| `--lighting` | `flat` | Illumination mode |
| `--experiment-id` | `None` | Reads `molmo_multiview_<ID>.json` |
| `--patch-size` | `1` | `1`, `3`, or `5` — must match `map_to_3d.py`'s `--patch-size` to inspect the same output |
| `--show-gt` | `False` | Overlay ground-truth symmetry element |
| `--show-predicted` | `False` | Overlay predicted axis/plane from `predicted_symmetry[_EXP].json` |
| `--pred-method` | `svd` | Which of the 4 fitting methods to visualize with `--show-predicted` |
| `--show-clusters` | `False` | Overlay cluster centroids from `mapped_points_3d[_EXP].json` (requires `map_to_3d.py` to have run) |
| `--clustering-method` | `greedy` | `greedy` or `hdbscan` — clustering method for `--show-clusters` |
| `--hdbscan-min-samples` | `3` | `min_samples` for `--clustering-method hdbscan` |
| `--point-mode` | `independent` | `independent`, `midpoint`, or `all` — must match `estimate_symmetry.py`'s point mode |
| `--ray-length` | `2 × bbox diag` | Length of miss rays |
| `--ray-radius` | `0.004` | Tube radius for all rays |
| `--hit-radius` | `0.015` | Sphere radius for hit points |
| `--cam-radius` | `0.020` | Sphere radius for camera positions |

---

## 7. `export_viz_samples.py`

Selects the N objects with the best and worst angular error for a given experiment/method,
copies their pipeline JSONs into `<results_dir>/<experiment_id>/viz_samples/{good,bad}/`, and
generates a `README.md` with ready-to-run `visualize_rays.py` commands per object (including a
before/after comparison table when `--base-experiment-id` is given, e.g. comparing a clustered
run against its non-clustered base).

```bash
python Mapping/export_viz_samples.py \
    --renders-root ../data/renders \
    --objects-root ../data/objects \
    --symmetry-type axis_sym \
    --experiment-id axis_v00 \
    --method svd \
    --results-dir ../results \
    --sizes 224 --lightings flat \
    --n-samples 10
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--renders-root` | *(required)* | Root folder of renders |
| `--objects-root` | *(required)* | Root folder with `.obj` meshes |
| `--symmetry-type` | *(required)* | `axis_sym` or `plane_sym` |
| `--results-dir` | *(required)* | Output root; writes to `<results_dir>/<experiment_id>/` |
| `--method` | `svd` | Which fitting method to rank good/bad objects by |
| `--experiment-id` | `None` | Reads `eval_*_<ID>_*_results.json` |
| `--base-experiment-id` | `None` | Base experiment to compare against (e.g. `axis_v00` when exporting `axis_v00_cluster`) |
| `--sizes` | `224` | Sizes to copy JSONs for |
| `--lightings` | `flat` | Lightings to copy JSONs for |
| `--n-views` | max available | n_views group used for ranking |
| `--n-samples` | `10` | Number of good + bad objects to export |

---

## Full pipeline example

```bash
EXP=axis_v02
MODE=independent   # see Experiments.md for correct value per prompt

# 1. Molmo2 inference
CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \
    --renders-root ../data/renders \
    --symmetry-type axis_sym \
    --sizes 224 --lightings flat \
    --view-groups 1 6 14 26 \
    --experiment-id $EXP --prompt-id $EXP \
    --prompt-mode auto --yes

# 2. Map 2D predictions to 3D mesh surface
python Mapping/map_to_3d.py \
    --renders-root ../data/renders \
    --objects-root ../data/objects \
    --symmetry-type axis_sym \
    --sizes 224 --lightings flat \
    --experiment-id $EXP --overwrite --yes

# 3. Fit symmetry from 3D points (4 methods)
python Mapping/estimate_symmetry.py \
    --renders-root ../data/renders \
    --objects-root ../data/objects \
    --symmetry-type axis_sym \
    --sizes 224 --lightings flat \
    --experiment-id $EXP --point-mode $MODE --overwrite

# 4. Evaluate vs ground truth (one call per method)
for METHOD in svd ransac_svd svd_sde ransac_svd_sde; do
    python Mapping/evaluate.py \
        --renders-root ../data/renders \
        --objects-root ../data/objects \
        --symmetry-type axis_sym \
        --sizes 224 --lightings flat \
        --experiment-id $EXP --method $METHOD
done

# 5. Compare all experiments
python Mapping/compare_results.py \
    --renders-root ../data/renders \
    --symmetry-type axis_sym \
    --save-dir ../results/plots \
    --csv-dir ../results
```

---

## Resumability

| Script | Skip condition |
|---|---|
| `map_to_3d.py` | Skips objects where `mapped_points_3d[_EXP].json` already exists (unless `--overwrite`) |
| `estimate_symmetry.py` | Skips objects where `predicted_symmetry[_EXP].json` already exists (unless `--overwrite`) |
| `evaluate.py` | Always rewrites (fast, CPU-only) |
| `compare_results.py` | Always rewrites |
| `cleanup_experiments.py` | Skips configs with no matching JSON or keys |

To rerun a specific object, delete its output file:

```bash
rm ../data/renders/axis_sym/<object_id>/224/flat/mapped_points_3d_axis_v02.json
rm ../data/renders/axis_sym/<object_id>/predicted_symmetry_axis_v02.json
```
