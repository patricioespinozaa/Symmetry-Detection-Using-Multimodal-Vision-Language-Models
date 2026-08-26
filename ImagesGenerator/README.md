# Mesh View Sampling - Fibonacci (FI Protocol)

This tool exports multi-view images of a 3D mesh using the Fibonacci sphere sampling method.

## Workflow

- Loads a `.obj` mesh file (assumed pre-centered at the origin and scaled to a unit bounding box).
- Generates camera viewpoints distributed uniformly on the sphere using the Fibonacci method.
- For each viewpoint, renders the mesh and saves one image.
- Saves metadata for each image, including camera parameters.

## Scripts

| Script | Purpose |
|---|---|
| `export_fibonacci_views.py` | Renders one object from a given set of Fibonacci viewpoints |
| `data_render.py` | Batch runner — applies `export_fibonacci_views.py` sequentially to all objects in a dataset folder (one `subprocess` call per object/size/lighting combo; no GPU-distribution logic of its own — run two instances with different `CUDA_VISIBLE_DEVICES` for manual multi-GPU parallelism, as in the root `README.md`'s Step 1) |

## Usage

```bash
python ImagesGenerator/export_fibonacci_views.py \
  --mesh path/to/model.obj \
  --output output_folder \
  --symmetry-type axis_sym \
  --repo-views 26 \
  --illumination flat
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--mesh` | *(required)* | Path to the input `.obj` mesh file |
| `--symmetry-type` | *(required)* | Symmetry type: `axis_sym` or `plane_sym` |
| `--output` | `output` | Output base folder |
| `--repo-views` | `26` | Number of Fibonacci viewpoints (recommended: 1, 6, 14, 26, 42, 62, 86, 114, …) |
| `--image-size` | `512` | Output image size in pixels (square) |
| `--fov` | `60.0` | Field of view in degrees |
| `--device` | `cuda:0` | Device to use (`cuda:0` or `cpu`) |
| `--camera-distance-factor` | `1.2` | Camera distance multiplier |
| `--illumination` | `flat` | Illumination mode: `flat`, `darker`, or `brighter` |

#### `--illumination` options

| Mode | Gray value | Description |
|---|---|---|
| `flat` | 0.7 | Uniform mid-gray (default) |
| `darker` | 0.3 | Uniform dark gray |
| `brighter` | 0.95 | Uniform light gray |

## Output structure

Images and metadata are saved under the following directory hierarchy:

```
<output>/<symmetry_type>/<object_id>/<image_size>/<illumination>/
```

For example, running with `--mesh Examples/objects/axis_sym_obj.obj --output Examples/renders --image-size 224 --illumination flat` produces:

```
Examples/renders/axis_sym/axis_sym_obj/224/flat/
├── IND_00_AZ_090_EL_+89.png
├── IND_01_AZ_243_EL_+67.png
├── ...
├── metadata_all.json
└── manifest.json
```

### Output files

- **Images:** `IND_{i:02d}_AZ_{az:03d}_EL_{el:+03d}.png`
- **Metadata:** `metadata_all.json` — list of all images and their camera parameters
- **Manifest:** `manifest.json` — processing summary (see fields below)

### Filename convention

| Token | Format | Description |
|---|---|---|
| `IND_{i}` | 2-digit, zero-padded | Viewpoint index (0-based) |
| `AZ_{az}` | 3-digit, zero-padded | Azimuth angle (integer degrees, 0–359) |
| `EL_{el}` | 3-digit with sign | Elevation angle (integer degrees, can be negative) |

**Example:** `IND_03_AZ_120_EL_+45.png` → 4th viewpoint, azimuth 120°, elevation +45°.

### Metadata fields

Each entry in `metadata_all.json` contains:

| Field | Type | Description |
|---|---|---|
| `index` | int | Viewpoint index |
| `filename` | str | Image filename |
| `azimuth` | int | Azimuth angle (degrees, 0–359) |
| `elevation` | int | Elevation angle (degrees) |
| `angle_info` | dict | Dict with `azimuth`, `elevation`, and `radius` |
| `eye` | list[float] | Camera position `[x, y, z]` in world space |
| `R` | list[list[float]] | 3×3 rotation matrix (world → camera). PyTorch3D row-vector convention: `p_cam = p_world @ R + T`. Camera centre recoverable as `-(R @ T)`. |
| `T` | list[float] | Translation vector (world → camera), same convention as `R`. |

### Manifest fields

`manifest.json` contains a processing summary for the rendered batch:

| Field | Description |
|---|---|
| `mesh` | Path to the source `.obj` file |
| `device` | Device used for rendering (`cuda:0`, `cpu`, etc.) |
| `fov` | Field of view in degrees |
| `image_size` | Output image size in pixels |
| `camera_distance` | Computed camera distance (mesh radius × `--camera-distance-factor`) |
| `total_images` | Number of images rendered |
| `processing_time_seconds` | Total rendering time in seconds |
| `processing_time_human` | Same, formatted as `Xm Ys` |
| `illumination` | Illumination mode used |
| `illumination_value` | Gray value applied to mesh vertices |

## Examples

The following commands render the bundled example objects from `Examples/objects/` and write results to `Examples/renders/`.

```bash
# Axial symmetry example
python ImagesGenerator/export_fibonacci_views.py \
  --mesh Examples/objects/axis_sym_obj.obj \
  --output Examples/renders \
  --symmetry-type axis_sym \
  --repo-views 26 \
  --image-size 224 \
  --illumination flat

# Planar symmetry example
python ImagesGenerator/export_fibonacci_views.py \
  --mesh Examples/objects/plane_sym_obj.obj \
  --output Examples/renders \
  --symmetry-type plane_sym \
  --repo-views 26 \
  --image-size 224 \
  --illumination flat
```

Each command generates **26 images** (one per viewpoint), saved under `Examples/renders/<symmetry_type>/`.