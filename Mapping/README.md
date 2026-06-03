# Mapping — 2D Predictions to 3D Symmetry Evaluation

This folder contains the pipeline to project Molmo2 pointing predictions onto the 3D mesh, estimate the predicted symmetry element (axis or plane), and evaluate it against ground-truth labels.

---

## Pipeline overview

```
molmo_multiview.json          metadata_all.json        .obj mesh
(Molmo2 2D predictions)       (camera parameters)      (3D geometry)
         └──────────────────────────┬───────────────────────┘
                                    ▼
                            map_to_3d.py
                     (ray casting: 2D coords → 3D points)
                                    │
                                    ▼
                         mapped_points_3d.json
                                    │
                                    ▼
                        estimate_symmetry.py
                    (SVD fit: 3D points → axis or plane)
                                    │
                                    ▼
                       predicted_symmetry.json
                                    │
                    ┌───────────────┴──────────────────┐
                    ▼                                  ▼
              .txt true labels               predicted_symmetry.json
                    └───────────────┬──────────────────┘
                                    ▼
                              evaluate.py
                    (angular error, origin distance, ...)
                                    │
                                    ▼
                evaluation_results.json + evaluation_summary.csv
```

---

## File structure

```
Mapping/
├── map_to_3d.py             # Ray casting: Molmo 2D coords → 3D mesh surface
├── estimate_symmetry.py     # SVD fit: 3D points → predicted axis or plane
├── evaluate.py              # Metrics: predicted symmetry vs true labels
└── cleanup_experiments.py   # Remove specific n_views keys from JSON results
```

---

## Input requirements

| Source | Location |
|---|---|
| Molmo2 predictions | `<renders_root>/<symmetry_type>/<object_id>/<size>/<illumination>/molmo_multiview.json` |
| Camera parameters | `<renders_root>/<symmetry_type>/<object_id>/<size>/<illumination>/metadata_all.json` |
| 3D meshes | `<objects_root>/curated_axis_sym_obj/<object_id>.obj` |
| True labels | `<objects_root>/curated_axis_sym_obj/<object_id>.txt` |

### True label format

**axis_sym:**
```
1                                            ← number of axes
axis DX DY DZ  OX OY OZ                     ← direction + origin
N_ANGLES
angles A1 A2 ...                             ← rotational symmetry angles (not used for evaluation)
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
| Molmo2 output | Top-left | 0–1000 |
| NDC (camera) | Center | −1 to +1 |
| World / mesh | Mesh centroid ≈ origin | ShapeNet unit cube (~1.0) |

ShapeNet meshes are pre-normalized to fit within a unit cube centered at the origin — no additional centering or scaling is needed.

---

## 1. `map_to_3d.py`

Projects each Molmo2 predicted point onto the 3D mesh surface via ray casting.

**Method:** reconstructs the camera ray from `R`, `T`, and `fov=60°` stored in `metadata_all.json`, then intersects it with the mesh using `trimesh`. Only ROT_000 images are used (one view per viewpoint index, no 2D rotations).

**Output:** `mapped_points_3d.json` alongside the renders. Skips configs where the file already exists.

### Usage

```bash
# Single process
python Mapping/map_to_3d.py \
    --renders-root ../data/renders \
    --objects-root ../data/objects \
    --symmetry-type axis_sym \
    --sizes 224 \
    --lightings flat

# Two parallel processes (CPU-bound, splits objects round-robin)
python Mapping/map_to_3d.py \
    --renders-root ../data/renders \
    --objects-root ../data/objects \
    --symmetry-type axis_sym \
    --gpu-id 0 --num-gpus 2

python Mapping/map_to_3d.py \
    --renders-root ../data/renders \
    --objects-root ../data/objects \
    --symmetry-type axis_sym \
    --gpu-id 1 --num-gpus 2
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
  "n_views_results": {
    "1": {
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
    },
    "6":  { ... },
    "14": { ... }
  }
}
```

---

## 2. `estimate_symmetry.py`

Fits a symmetry axis or plane from the 3D hit points using SVD (PCA).

**Method:**
- `axis_sym`: first principal component of the centered hit points = axis direction. Origin = centroid.
- `plane_sym`: last principal component (minimum variance direction) = plane normal. Origin = centroid.

Points from all (size × illumination) configurations are pooled per `n_views` group before fitting — more configurations give a more robust estimate.

**Output:** `predicted_symmetry.json` at the object level (one file per object, not per size/illumination). Skips objects where the file already exists.

### Usage

```bash
python Mapping/estimate_symmetry.py \
    --renders-root ../data/renders \
    --symmetry-type axis_sym \
    --sizes 224 \
    --lightings flat
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--renders-root` | *(required)* | Root folder of renders |
| `--symmetry-type` | *(required)* | `axis_sym` or `plane_sym` |
| `--sizes` | `224 448 1136` | Sizes to pool points from |
| `--lightings` | `flat brighter darker` | Lightings to pool points from |
| `--gpu-id` | `0` | Process index |
| `--num-gpus` | `1` | Total parallel processes |

### Output format (`predicted_symmetry.json`)

**axis_sym:**
```json
{
  "object_id": "1a9c1cbf...",
  "symmetry_type": "axis_sym",
  "n_views_predictions": {
    "1":  {"direction": [dx, dy, dz], "origin": [ox, oy, oz], "n_points": 2},
    "6":  {"direction": [...], "origin": [...], "n_points": 6},
    "14": { ... },
    "26": { ... }
  }
}
```

**plane_sym:**
```json
{
  "object_id": "...",
  "symmetry_type": "plane_sym",
  "n_views_predictions": {
    "1": {"normal": [nx, ny, nz], "origin": [ox, oy, oz], "n_points": 2}
  }
}
```

---

## 3. `evaluate.py`

Compares the predicted symmetry element against the ground-truth label and computes evaluation metrics.

**Metrics:**

| Metric | axis_sym | plane_sym |
|---|---|---|
| Angular error (°) | Angle between predicted and true axis directions | Angle between predicted and true plane normals |
| Origin distance | Point-to-line distance from predicted origin to true axis | Point-to-plane distance from predicted origin to true plane |

Both metrics are **sign-agnostic** (axis direction and plane normal are undetermined up to sign).

For `plane_sym` with multiple true planes (up to 3), the predicted plane is matched against the **closest true plane** (minimum angular error) — best-match strategy.

**Output:** saves results at the symmetry type level, not per object. Skips nothing — always rewrites.

### Usage

```bash
python Mapping/evaluate.py \
    --renders-root ../data/renders \
    --objects-root ../data/objects \
    --symmetry-type axis_sym

python Mapping/evaluate.py \
    --renders-root ../data/renders \
    --objects-root ../data/objects \
    --symmetry-type plane_sym
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--renders-root` | *(required)* | Root folder of renders |
| `--objects-root` | *(required)* | Root folder with `.txt` labels |
| `--symmetry-type` | *(required)* | `axis_sym` or `plane_sym` |

### Output files

Both saved under `<renders_root>/<symmetry_type>/`:

**`evaluation_results.json`** — per-object, per-n_views metrics:
```json
{
  "symmetry_type": "axis_sym",
  "objects": {
    "1a9c1cbf...": {
      "1":  {"angular_error_deg": 12.4, "origin_dist": 0.041, "n_points": 2, "status": "ok"},
      "6":  {"angular_error_deg":  5.1, "origin_dist": 0.018, "n_points": 6, "status": "ok"},
      "14": { ... },
      "26": { ... }
    }
  }
}
```

**`evaluation_summary.csv`** — aggregated statistics per n_views group:

| n_views | n_objects | angular_error_mean | angular_error_median | angular_error_std | origin_dist_mean | origin_dist_median | origin_dist_std | n_points_mean |
|---|---|---|---|---|---|---|---|---|
| 1 | 850 | 18.3 | 14.1 | 12.5 | 0.062 | 0.041 | 0.055 | 2.0 |
| 6 | 850 | 11.2 | 8.4 | 9.3 | 0.038 | 0.024 | 0.041 | 5.8 |
| ... | | | | | | | | |

---

## 4. `cleanup_experiments.py`

Removes specific `n_views` keys from `molmo_multiview.json` files. Use this when iterating on prompts to delete only the results you want to rerun without affecting other keys.

- If the JSON still has other keys after removal → rewritten without the removed keys
- If the JSON becomes empty → file deleted entirely

**Always run with `--dry-run` first** to preview what would be changed.

### Usage

```bash
# Preview (no changes)
python Mapping/cleanup_experiments.py \
    --renders-root ../data/renders \
    --symmetry-type axis_sym \
    --sizes 224 \
    --lightings flat \
    --view-groups 1 6 14 26 \
    --dry-run

# Execute
python Mapping/cleanup_experiments.py \
    --renders-root ../data/renders \
    --symmetry-type axis_sym \
    --sizes 224 \
    --lightings flat \
    --view-groups 1 6 14 26
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--renders-root` | *(required)* | Root folder of renders |
| `--symmetry-type` | *(required)* | `axis_sym` or `plane_sym` |
| `--view-groups` | *(required)* | Keys to remove (e.g. `1 6 14 26`) |
| `--sizes` | `224 448 1136` | Sizes to clean |
| `--lightings` | `flat brighter darker` | Lightings to clean |
| `--dry-run` | `False` | Preview without making changes |

---

## Full pipeline example

```bash
# 1. Generate Molmo2 predictions (molmo_multiview_runner.py)
CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \
    --renders-root ../data/renders \
    --symmetry-type axis_sym \
    --sizes 224 --lightings flat \
    --view-groups 1 6 14 26

# 2. Map 2D predictions to 3D mesh surface
python Mapping/map_to_3d.py \
    --renders-root ../data/renders \
    --objects-root ../data/objects \
    --symmetry-type axis_sym \
    --sizes 224 --lightings flat \
    --overwrite

# 3. Fit predicted axis/plane from 3D points (RANSAC + SVD + SDE)
python Mapping/estimate_symmetry.py \
    --renders-root ../data/renders \
    --objects-root ../data/objects \
    --symmetry-type axis_sym \
    --sizes 224 --lightings flat \
    --overwrite

# 4. Evaluate vs ground truth
python Mapping/evaluate.py \
    --renders-root ../data/renders \
    --objects-root ../data/objects \
    --symmetry-type axis_sym \
    --sizes 224 --lightings flat

# --- Iterate on prompt ---

# 5. Clean results for a specific experiment
python Mapping/cleanup_experiments.py \
    --renders-root ../data/renders \
    --symmetry-type axis_sym \
    --sizes 224 --lightings flat \
    --view-groups 1 6 14 26 \
    --dry-run   # remove --dry-run to execute

# 6. Rerun from step 1 with new prompt
```

---

## Resumability

| Script | Skip condition |
|---|---|
| `map_to_3d.py` | Skips if `mapped_points_3d.json` already exists |
| `estimate_symmetry.py` | Skips if `predicted_symmetry.json` already exists |
| `evaluate.py` | Always rewrites (fast, CPU-only) |
| `cleanup_experiments.py` | Skips configs with no JSON or no matching keys |

To rerun a specific object in `map_to_3d.py` or `estimate_symmetry.py`, delete the corresponding output file:

```bash
rm ../data/renders/axis_sym/<object_id>/224/flat/mapped_points_3d.json
rm ../data/renders/axis_sym/<object_id>/predicted_symmetry.json
```
