# Mesh View Sampling - Fibonacci (FI Protocol)

This tool exports multi-view images of a 3D mesh using the Fibonacci sphere sampling method.

## Workflow

- Loads a `.obj` mesh file.
- Generates camera viewpoints distributed uniformly on the sphere using the Fibonacci method.
- For each viewpoint, renders the mesh and saves one image.
- Saves metadata for each image, including camera parameters.

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
| `--repo-views` | `26` | Number of Fibonacci viewpoints (recommended: 6, 14, 26, 42, 62, 86, 114, …) |
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

- **Images:** `IND_{i}_AZ_{az}_EL_{el}.png`
- **Metadata:** `metadata_all.json` — list of all images and their camera parameters
- **Manifest:** `manifest.json` — processing summary

### Filename convention

| Token | Description |
|---|---|
| `IND_{i}` | Viewpoint index (0-based) |
| `AZ_{az}` | Azimuth angle (integer degrees, 0–359) |
| `EL_{el}` | Elevation angle (integer degrees, can be negative) |

**Example:** `IND_03_AZ_120_EL_+45.png` → 4th viewpoint, azimuth 120°, elevation +45°.

### Metadata fields

Each entry in `metadata_all.json` contains:

| Field | Description |
|---|---|
| `index` | Viewpoint index |
| `filename` | Image filename |
| `azimuth` | Azimuth angle (degrees) |
| `elevation` | Elevation angle (degrees) |
| `angle_info` | Dict with `azimuth`, `elevation`, and `radius` |
| `eye` | Camera position `[x, y, z]` |
| `R` | Camera rotation matrix |
| `T` | Camera translation vector |

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