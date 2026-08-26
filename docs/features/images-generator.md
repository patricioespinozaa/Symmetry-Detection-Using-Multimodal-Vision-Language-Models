# ImagesGenerator

## Description

`ImagesGenerator/` is the first stage of the thesis pipeline. It takes a 3D mesh
(`.obj`, pre-centered at the origin and scaled to a unit bounding box) and renders a
set of 2D multi-view images of it, distributing camera viewpoints uniformly over a
sphere using the Fibonacci sphere sampling method. Each render is a flat-shaded,
uniform-gray view of the mesh (no texture), lit by a directional light placed at the
camera position, produced with PyTorch3D's differentiable rasterizer (used here only
in inference mode, no gradients).

These renders are the only input Molmo2-8B ever sees of the 3D object: the VLM never
has direct access to the mesh geometry, only to these 2D images plus the camera
parameters saved alongside them. `MolmoPointing/` consumes the images to produce 2D
point predictions per view, and `Mapping/map_to_3d.py` later uses the exact camera
`R`/`T`/`fov`/`eye` metadata saved by this stage to ray-cast those 2D points back onto
the mesh surface in 3D. Because the whole downstream pipeline (retro-projection →
symmetry fit → evaluation) depends on the camera parameters being both correct and
saved in a specific convention, this stage is also responsible for persisting exact,
reproducible per-view camera metadata (`metadata_all.json`) and a run summary
(`manifest.json`), not just the pixels.

## Key Files & Functions

| File | Function / Class | Responsibility |
|---|---|---|
| `ImagesGenerator/export_fibonacci_views.py` | `sample_fibonacci_viewpoints` | Generates `num_points` camera positions on a sphere of given radius using the Fibonacci/golden-angle method; also returns spherical (azimuth/elevation/radius) angle info. |
| `ImagesGenerator/export_fibonacci_views.py` | `rotate_points` | Applies a small Rodrigues-formula rotation to the sampled points (default: 0.001 rad around X) to avoid the pole singularity of the Fibonacci sampling. |
| `ImagesGenerator/export_fibonacci_views.py` | `cartesian_to_spherical` | Converts camera XYZ positions to azimuth/elevation/radius, used both for filenames and metadata. |
| `ImagesGenerator/export_fibonacci_views.py` | `get_min_camera_distance` | Computes the minimum camera distance so the mesh fits inside the field of view, given mesh radius and FOV. |
| `ImagesGenerator/export_fibonacci_views.py` | `stabilize_eye_positions` | Nudges camera positions that fall too close to the Y-axis (near-degenerate look-at) by a small epsilon offset. |
| `ImagesGenerator/export_fibonacci_views.py` | `load_mesh` | Loads a `.obj` file as a PyTorch3D `Meshes` object (no textures loaded from file). |
| `ImagesGenerator/export_fibonacci_views.py` | `mesh_radius` | Estimates mesh radius by sampling 1000 points from the mesh surface and taking the max norm. |
| `ImagesGenerator/export_fibonacci_views.py` | `render_fibonacci` | Core rendering loop: builds cameras/lights per viewpoint, rasterizes the mesh in batches of 4, saves one PNG per view, and builds the per-image metadata records. |
| `ImagesGenerator/export_fibonacci_views.py` | `save_image_tensor` | Converts a rendered `(H, W, 3)` float tensor in `[0, 1]` to a `uint8` PNG. |
| `ImagesGenerator/export_fibonacci_views.py` | `write_metadata` | Dumps the full list of per-image metadata records to `metadata_all.json`. |
| `ImagesGenerator/export_fibonacci_views.py` | `main` | CLI entry point: parses arguments, sets up illumination/gray value, computes camera distance, renders, and writes `manifest.json`. |
| `ImagesGenerator/data_render.py` | `preview_execution` | Prints a summary of the planned batch run (object/view/size/lighting counts) and asks for interactive `'OK'` confirmation before starting. |
| `ImagesGenerator/data_render.py` | `main` | Batch runner: globs all `.obj` files in an input folder and calls `export_fibonacci_views.py` as a subprocess once per object × size × lighting combination, with a `tqdm` progress bar. |

## Inputs & Outputs

**Inputs:**
- A single `.obj` mesh file (`export_fibonacci_views.py --mesh`), assumed
  pre-centered at the origin and scaled to a unit bounding box (per `README.md`); no
  texture is read from the file — a uniform gray vertex color is applied instead.
- A folder of `.obj` files (`data_render.py --input-folder`), non-recursively globbed
  (`*.obj`).
- CLI parameters controlling the render: number of Fibonacci views (`--repo-views`),
  output image size (`--image-size`), field of view (`--fov`), camera distance
  multiplier (`--camera-distance-factor`), illumination mode (`--illumination`:
  `flat`/`darker`/`brighter`), symmetry type label (`--symmetry-type`:
  `axis_sym`/`plane_sym`), and device (`--device`: `cuda:0` or `cpu`).

**Outputs:** written under `<output>/<symmetry_type>/<object_id>/<image_size>/<illumination>/`:
- PNG images, one per viewpoint, named `IND_{i:02d}_AZ_{az:03d}_EL_{el:+03d}.png`.
- `metadata_all.json` — per-image record with `index`, `filename`, `azimuth`,
  `elevation`, `angle_info` (dict with azimuth/elevation/radius), `eye` (camera
  position in world space), `R` (3×3 rotation matrix, world→camera, row-vector
  convention `p_cam = p_world @ R + T`), and `T` (translation vector). This is the
  exact camera information `Mapping/map_to_3d.py` needs to ray-cast Molmo's 2D
  predictions back to 3D.
- `manifest.json` — run summary: mesh path, device, fov, image size, computed camera
  distance, total images rendered, processing time (seconds and human-readable),
  illumination mode and its numeric gray value.

## Results & Observations

No metrics, plots, or evaluation results are produced or logged by this stage — it is
purely a data-generation step. The only recorded outputs are the processing time
fields in `manifest.json` (`processing_time_seconds` / `processing_time_human`),
which give a per-object rendering-time record but are not aggregated or analyzed
anywhere in this module.

## Key Decisions

- **Fibonacci sphere sampling** for camera placement, to distribute viewpoints
  approximately uniformly over the sphere for a given view count (recommended counts
  in `README.md`: 1, 6, 14, 26, 42, 62, 86, 114, …, i.e. the standard Fibonacci-sphere
  progression).
  > Decision found but reason not documented beyond "uniform distribution" in code comments.
- **Small fixed rotation (0.001 rad around X)** applied to all sampled points
  before angle computation, specifically to avoid the pole singularity inherent to
  the Fibonacci sampling formula (both poles map to well-defined but degenerate
  camera orientations otherwise).
- **`stabilize_eye_positions`** independently guards against any camera position
  landing too close to the Y-axis (`|y_unit| > 0.999`) by adding a `1e-4` offset to X
  and Z — a second, defense-in-depth safeguard against the same look-at
  degeneracy (straight up/down view has an ill-defined "up" vector for
  `look_at_view_transform`).
- **Camera distance derived from mesh radius and FOV**, not fixed, so meshes of
  different scale are always framed to fit the field of view; `--camera-distance-factor`
  (default `1.2`) adds a margin beyond the tightest fit.
- **Flat/uniform gray shading with no texture**, controlled only by an
  `--illumination` gray-value choice (`flat`=0.7, `darker`=0.3, `brighter`=0.95)
  applied as vertex colors, plus a directional light from the camera position.
  > Decision found but reason not documented — likely to remove texture/material as a
  > confound and isolate shape-only cues for the VLM's pointing task, consistent with
  > the thesis's stated aim of testing whether the VLM infers symmetry from geometry
  > (renders), not appearance, but this rationale is not stated in the code itself.
- **Batched rendering in groups of 4** views at a time inside `render_fibonacci`,
  presumably to bound GPU memory usage during rasterization.
  > Decision found but reason not documented (batch size of 4 is a hardcoded literal,
  > no comment justifying that specific value).
- **Output directory structure** `<output>/<symmetry_type>/<object_id>/<size>/<lighting>/`
  is fixed and matches the convention documented in `CLAUDE.md`/`docs/Contexto.md`
  for every later pipeline stage.

## Known Limitations

- **`data_render.py`'s batch loop is fully sequential** — it calls
  `export_fibonacci_views.py` as one subprocess per object/size/lighting combination,
  one at a time, with no parallelism or GPU distribution in the code (no device
  selection across multiple GPUs, multiprocessing, or job splitting). Multi-GPU use
  is manual: run two separate `data_render.py` instances with different
  `CUDA_VISIBLE_DEVICES` (as documented in the root `README.md`'s Step 1) and split
  the object list yourself. `ImagesGenerator/README.md` previously claimed the
  script itself has "multi-GPU support" — corrected during the 2026-08-25 doc pass.
- **`data_render.py` requires interactive confirmation** (`preview_execution` calls
  `input("Type 'OK' to start: ")`), which will hang or fail in a fully non-interactive
  / scripted environment (e.g. inside another automation pipeline) unless stdin is
  piped.
- **No overwrite/skip check**: unlike later pipeline stages (per `CLAUDE.md`, which
  says later stages skip already-computed runs), `export_fibonacci_views.py` always
  re-renders and overwrites, and `data_render.py` has no flag to skip
  object/size/lighting combinations that already have output on disk.
- **`.obj` texture data is never loaded** (`load_objs_as_meshes(..., load_textures=False, ...)`)
  — any material/texture information in the source mesh is discarded; only geometry
  is used.
- **No validation that the input mesh is actually centered/unit-scaled** — the
  README states this is an assumption, but the code does not check or enforce it; a
  mesh violating the assumption would silently get an ill-fitted camera distance
  (via `get_min_camera_distance`/`mesh_radius`, which only account for radius, not
  centering).
- **Mesh radius estimated by point sampling, not exact bounding geometry** —
  `mesh_radius` samples only 1000 points from the mesh surface and takes the max
  norm, which can slightly underestimate the true bounding radius for meshes with
  sparse geometric extrema (thin spikes, etc.) between sampled points.
- **`--device` argument silently falls back**: `torch.device(args.device if
  torch.cuda.is_available() or "cpu" in args.device else "cpu")` — if CUDA is
  unavailable and the user did not literally pass a string containing `"cpu"`, this
  still resolves to `"cpu"`, but the logic is easy to misread as validating the
  requested device string rather than just checking for the substring `"cpu"`.
- **Angles in filenames/metadata are truncated to `int`** (`cartesian_to_spherical`
  uses `int(azimuth)`, `int(elevation)`), losing sub-degree precision in the
  human-readable filename/metadata fields — the full-precision camera pose is still
  preserved separately via `R`/`T`/`eye` in `metadata_all.json`, so this only affects
  the descriptive angle fields, not the geometry used by downstream ray casting.
- No automated tests are present for this module (no `test_*` files found under
  `ImagesGenerator/`).
