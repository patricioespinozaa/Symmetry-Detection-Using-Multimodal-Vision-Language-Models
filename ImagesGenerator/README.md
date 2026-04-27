# Mesh View Sampling - Fibonacci (4R-FI Protocol)

This tool exports multi-view images of a 3D mesh using the Fibonacci sphere sampling method, with 4 in-plane rotations per viewpoint (4R-FI).

## Workflow
- Loads a `.obj` mesh file.
- Generates camera viewpoints distributed uniformly on the sphere using the Fibonacci method.
- For each viewpoint, renders the mesh and saves 4 images, each rotated by 0°, 90°, 180°, and 270° (2D rotation of the rendered image).
- Saves metadata for each image, including camera parameters and rotation info.

## Usage


```bash
python ImagesGenerator/export_fibonacci_views.py --mesh path/to/model.obj --output output_folder --repo-views 114 --illumination flat
```


- `--mesh`: Path to the input .obj mesh file (required)
- `--output`: Output folder (default: output)
- `--repo-views`: Number of Fibonacci viewpoints (default: 26; recommended: 6, 14, 26, 42, 62, 86, 114, ...)
- `--image-size`: Output image size (default: 512)
- `--fov`: Field of view in degrees (default: 60.0)
- `--device`: Device to use (default: cuda:0)
- `--camera-distance-factor`: Camera distance multiplier (default: 1.2)
- `--illumination`: Illumination mode for the mesh material. Options:
	- `flat` (default): uniform gray (0.7)
	- `darker`: uniform dark gray (0.3)
	- `brighter`: uniform light gray (0.95)

### Example


```bash
# Flat (default)
python ImagesGenerator/export_fibonacci_views.py --mesh ./objects/plane_example.obj --output renders/plane_example --repo-views 114 --image-size 224 --illumination flat
# Darker
python ImagesGenerator/export_fibonacci_views.py --mesh ./objects/plane_example.obj --output renders/plane_example --repo-views 114 --image-size 224 --illumination darker
# Brighter
python ImagesGenerator/export_fibonacci_views.py --mesh ./objects/plane_example.obj --output renders/plane_example --repo-views 114 --image-size 224 --illumination brighter
```
This will generate 114 viewpoints × 4 rotations = 456 images in `renders/plane_example/`.

## Output files
- Images: `IND_{i}_AZ_{az}_EL_{el}_ROT_{rot}.png`
- Metadata: `metadata_all.json` (list of all images and their parameters)
- Manifest: `manifest.json` (summary)

### Illumination modes

The `--illumination` argument controls the gray value of the mesh material:

| Mode     | Value |
|----------|-------|
| flat     | 0.7   |
| darker   | 0.3   |
| brighter | 0.95  |

---

### Filename convention
- `IND_{i}`: Index of the viewpoint (0-based)
- `AZ_{az}`: Azimuth angle (degrees, integer, 0–359)
- `EL_{el}`: Elevation angle (degrees, integer, can be negative)
- `ROT_{rot}`: 2D rotation applied to the image (degrees, one of 0, 90, 180, 270)

**Example:** `IND_03_AZ_120_EL_+45_ROT_090.png` is the 4th viewpoint, azimuth 120°, elevation +45°, rotated 90°.

### Metadata fields
Each entry in `metadata_all.json` contains:
- `index`: Viewpoint index
- `azimuth`: Azimuth angle
- `elevation`: Elevation angle
- `rotation_deg`: 2D rotation applied (0, 90, 180, 270)
- `rotation_index`: Index of the rotation (0–3)
- `filename`: Image filename
- `angle_info`: Dictionary with azimuth, elevation, radius
- `eye`: Camera position
- `R`, `T`: Camera extrinsics