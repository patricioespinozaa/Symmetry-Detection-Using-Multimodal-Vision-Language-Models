# Mesh View Sampling - Fibonacci (4R-FI Protocol)

This tool exports multi-view images of a 3D mesh using the Fibonacci sphere sampling method, with 4 in-plane rotations per viewpoint (4R-FI).

## Workflow
- Loads a `.obj` mesh file.
- Generates camera viewpoints distributed uniformly on the sphere using the Fibonacci method.
- For each viewpoint, renders the mesh and saves 4 images, each rotated by 0°, 90°, 180°, and 270° (2D rotation of the rendered image).
- Saves metadata for each image, including camera parameters and rotation info.

## Usage

```bash
python ImagesGenerator/export_fibonacci_views.py \
  --mesh path/to/model.obj \
  --output output_folder \
  --symmetry-type axis_sym \
  --repo-views 114 \
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

For example:

```
renders/plane_example/axis_sym/plane_example/224/flat/
```

### Output files

- **Images:** `IND_{i}_AZ_{az}_EL_{el}_ROT_{rot}.png`
- **Metadata:** `metadata_all.json` — list of all images and their parameters
- **Manifest:** `manifest.json` — processing summary

### Filename convention

| Token | Description |
|---|---|
| `IND_{i}` | Viewpoint index (0-based) |
| `AZ_{az}` | Azimuth angle (integer degrees, 0–359) |
| `EL_{el}` | Elevation angle (integer degrees, can be negative) |
| `ROT_{rot}` | 2D rotation applied to the image (0, 90, 180, or 270) |

**Example:** `IND_03_AZ_120_EL_+45_ROT_090.png` → 4th viewpoint, azimuth 120°, elevation +45°, rotated 90°.

### Metadata fields

Each entry in `metadata_all.json` contains:

| Field | Description |
|---|---|
| `index` | Viewpoint index |
| `azimuth` | Azimuth angle (degrees) |
| `elevation` | Elevation angle (degrees) |
| `rotation_deg` | 2D rotation applied (0, 90, 180, or 270) |
| `rotation_index` | Rotation index (0–3) |
| `filename` | Image filename |
| `angle_info` | Dict with `azimuth`, `elevation`, and `radius` |
| `eye` | Camera position `[x, y, z]` |
| `R` | Camera rotation matrix |
| `T` | Camera translation vector |

## Examples

```bash
# axis_sym — flat illumination
python ImagesGenerator/export_fibonacci_views.py \
  --mesh ./objects/plane_example.obj \
  --output renders/plane_example \
  --symmetry-type axis_sym \
  --repo-views 114 \
  --image-size 224 \
  --illumination flat

# plane_sym — darker illumination
python ImagesGenerator/export_fibonacci_views.py \
  --mesh ./objects/plane_example.obj \
  --output renders/plane_example \
  --symmetry-type plane_sym \
  --repo-views 114 \
  --image-size 224 \
  --illumination darker

# plane_sym — brighter illumination
python ImagesGenerator/export_fibonacci_views.py \
  --mesh ./objects/plane_example.obj \
  --output renders/plane_example \
  --symmetry-type plane_sym \
  --repo-views 114 \
  --image-size 224 \
  --illumination brighter
```

The commands above generate 114 viewpoints × 4 rotations = **456 images** each.