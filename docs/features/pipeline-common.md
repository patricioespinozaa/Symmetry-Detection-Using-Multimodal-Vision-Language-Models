# pipeline_common

## Description

`pipeline_common/` is the shared-utilities package used across the whole
pipeline (`ImagesGenerator/`, `MolmoPointing/`, `Mapping/`, `InteractiveViewer/`).
It is not itself a pipeline stage — it does not read a stage's input JSON or
write a stage's output JSON — but a library of small, stateless helpers that
several stages need identically, factored out to avoid duplicated (and
potentially drifting) implementations of the same math or convention.

The package covers four concerns that recur throughout the methodology:
camera/ray geometry shared between the ray-casting stage and the debug
viewers (`camera.py`, `triangulation.py`); point-cloud consolidation used
before axis/plane fitting (`clustering.py`); filesystem/dataset conventions
for locating curated ShapeNet objects and loading meshes (`datasets.py`);
and small cross-cutting conveniences — experiment-id filename suffixing
(`naming.py`), deterministic description-view sampling for VLM prompting
(`view_selection.py`), and a single source of truth for the RGB colors used
by every visualization script (`viz_colors.py`). Centralizing this logic
means a correction to, e.g., the Molmo-coordinate-to-NDC conversion or the
GT/predicted color convention only needs to happen once and is guaranteed
consistent across every script that consumes it.

## Key Files & Functions

| File | Function / Class | Responsibility |
|---|---|---|
| `pipeline_common/__init__.py` | — | Empty; makes `pipeline_common` an importable package. No re-exports. |
| `pipeline_common/camera.py` | `molmo_to_ndc(x, y)` | Converts Molmo2 pixel coords (0-1000, top-left origin) to NDC (`[-1, 1]`, OpenGL-style). |
| `pipeline_common/camera.py` | `build_camera_rays(ndc_x, ndc_y, R, T, fov_deg, image_size)` | Builds a world-space camera ray (origin + normalized direction) for a given NDC point, using the PyTorch3D row-vector convention. |
| `pipeline_common/camera.py` | `project_point(p_world, R, T, fov_deg, image_size)` | Forward-projects a 3D world point to pixel coordinates; exact inverse of `build_camera_rays`. Flags points behind the camera. |
| `pipeline_common/camera.py` | `cast_ray(mesh, ray_origin, ray_direction)` | Casts a single ray against a `trimesh.Trimesh`, keeping the closest intersection (hit flag, 3D point, face id). |
| `pipeline_common/camera.py` | `cast_ray_patch(mesh, R, T, x, y, patch_size, image_width, image_height, fov_deg)` | Casts a `patch_size x patch_size` grid of sub-rays around a Molmo point and averages the hits, to stabilize backprojection at grazing angles. |
| `pipeline_common/clustering.py` | `cluster_points(points, bbox_diag, threshold_fraction=0.05)` | Greedy incremental-centroid spatial clustering; every point is assigned to a cluster (no outlier concept). |
| `pipeline_common/clustering.py` | `cluster_hdbscan(points, min_samples=3, min_cluster_size=2)` | Density-based clustering via `sklearn.cluster.HDBSCAN`; explicitly drops noise points (label -1), with graceful fallbacks to `cluster_points`-like behavior on degenerate input. |
| `pipeline_common/datasets.py` | `OBJECTS_SUBDIR` (dict) | Maps `"axis_sym"` / `"plane_sym"` to their curated ShapeNet object subdirectory names. |
| `pipeline_common/datasets.py` | `load_mesh(obj_path)` | Loads a `.obj` as a single `trimesh.Trimesh`, concatenating geometry if the file is a multi-mesh `Scene`. |
| `pipeline_common/datasets.py` | `load_mesh_vertices(obj_path)` | Loads a mesh and returns just its `(N, 3)` vertex array; returns `None` on any failure instead of raising. |
| `pipeline_common/naming.py` | `exp_filename(base, experiment_id)` | Suffixes a base filename with `_<experiment_id>` before the extension, or returns it unchanged when `experiment_id` is falsy. |
| `pipeline_common/triangulation.py` | `ray_dir_for_point(x, y, R, T, fov_deg, image_size)` | Convenience wrapper: Molmo2 `(x, y)` → world-space `(camera_center, ray_direction)`. |
| `pipeline_common/triangulation.py` | `view_forward_direction(R, T, fov_deg, image_size)` | World-space direction the camera is looking (its optical axis), via a ray through NDC origin. |
| `pipeline_common/triangulation.py` | `interpretation_plane_normal(dir_a, dir_b)` | Normal of the plane through the camera center and two rays from it (Bartoli & Sturm 2005); `None` if the rays are numerically parallel. |
| `pipeline_common/triangulation.py` | `triangulate_line(camera_centers, plane_normals)` | Intersects ≥2 interpretation planes via least squares/SVD to recover a 3D line (point + direction) — the mesh-free axis/plane-line recovery step. |
| `pipeline_common/triangulation.py` | `get_point_by_obj_id(pts, obj_id)` | Looks up a point by its fixed `obj_id` in one view's point list. |
| `pipeline_common/triangulation.py` | `widest_pair(pts)` | Returns the most pixel-separated pair of points in a view's point list (generalizes the fixed-`obj_id` pairing convention to free-form point counts). |
| `pipeline_common/view_selection.py` | `select_description_view(object_id, metadata_entries, elevation_range=(-60.0, 60.0))` | Deterministically (per-object MD5-seeded RNG) picks one render view outside near-pole elevations, for free-form VLM description prompts. |
| `pipeline_common/viz_colors.py` | `COLOR_MESH`, `COLOR_CAMERA`, `COLOR_HIT_RAY`, `COLOR_MISS_RAY`, `COLOR_HIT_PT`, `COLOR_CLUSTER_PT`, `COLOR_GT_AXIS`, `COLOR_GT_PLANE`, `COLOR_PRED_AXIS`, `COLOR_PRED_PLANE` | RGB tuple constants (`[0, 1]` range) shared by every Polyscope/matplotlib visualization script under `Mapping/`. |

## Inputs & Outputs

This is a shared-utilities module, not a pipeline stage with data flowing
through files — there is no "reads file X, writes file Y" here. Instead,
each utility group is a pure function (or, for `viz_colors.py`, a set of
constants) operating on in-memory arguments and return values:

**Inputs:**
- `camera.py` / `triangulation.py`: Molmo2 pixel coordinates (`x, y` in
  `[0, 1000]`), per-view camera extrinsics (`R` 3x3 row-major rotation,
  `T` translation vector, both world↔camera per the PyTorch3D row-vector
  convention `p_cam = p_world @ R + T`), `fov_deg`, `image_size`/
  `image_width`/`image_height`, and (for ray casting) a loaded
  `trimesh.Trimesh`.
- `clustering.py`: an `(N, 3)` NumPy array of 3D points plus a scale
  reference (`bbox_diag`) or density parameters (`min_samples`,
  `min_cluster_size`).
- `datasets.py`: a `Path` to a `.obj` file, or a symmetry-type string
  (`"axis_sym"`/`"plane_sym"`) for `OBJECTS_SUBDIR` lookups.
- `naming.py`: a base filename string and an optional experiment-id string.
- `view_selection.py`: an object id string and a list of render-metadata
  dicts (each with at least `"index"` and `"elevation"`), as produced by
  stage-1 rendering metadata (`metadata_all.json`).
- `viz_colors.py`: no inputs — imported as constants.

**Outputs:**
- `camera.py` / `triangulation.py`: NDC coordinates, world-space ray
  origins/directions, pixel coordinates with a behind-camera flag, ray-hit
  dicts (`hit`, `point_3d`, `face_id`, patch statistics), interpretation-plane
  normals, and triangulated 3D lines (point + direction) — all plain
  `float`/`np.ndarray`/`dict` values, not files.
- `clustering.py`: an `(M, 3)` array of cluster centroids (`M <= N`).
- `datasets.py`: a `trimesh.Trimesh` or an `(N, 3)` vertex array (`None` on
  load failure for the vertices-only variant).
- `naming.py`: a filename string.
- `view_selection.py`: a single selected metadata entry (dict).
- `viz_colors.py`: RGB tuple constants consumed directly by caller plotting
  code.

## Results & Observations

> Not found in codebase — needs manual input. (This package contains no
logging, printed summaries, or saved artifacts of its own; any observable
results belong to the calling scripts in `Mapping/`, `MolmoPointing/`, etc.)

## Key Decisions

- **Patch-based backprojection averaging (`cast_ray_patch`)**: exact
  single-pixel ray casting is sensitive to grazing angles, where small
  pixel-localization errors produce large jumps in the resulting 3D point;
  averaging a small neighborhood of sub-rays stabilizes the result against
  local curvature and VLM localization noise (docstring in `camera.py`).
- **Two clustering strategies kept separate rather than unified**:
  `cluster_points` assigns every point to some cluster (no outlier concept),
  while `cluster_hdbscan` explicitly discards sparse/isolated points as noise
  — useful for dropping Molmo hallucinations that happen to hit the mesh but
  land far from the rest of the cloud (`clustering.py` docstring).
- **HDBSCAN's `min_cluster_size` is fixed, not swept**: point clouds here are
  small (tens of points), and scikit-learn's own default of 5 would
  over-aggressively call everything noise at low `n_views`; `min_samples` is
  the parameter the methodology actually sweeps over (2, 3, 5).
- **`cluster_hdbscan` falls back to returning input points unchanged** in
  several degenerate cases (too few points, `min_samples`/`min_cluster_size`
  unsatisfiable, or every point labeled noise) rather than raising or
  returning an empty cloud — deliberately mirrors `cluster_points`' own
  fallback so a small-`n_views` group doesn't get silently skipped downstream.
- **Per-object deterministic seeding in `select_description_view`**
  (`seed = int(MD5(object_id), 16) mod 2**31`) rather than one global seed:
  keeps the choice reproducible per object while ensuring different objects
  are described from different viewpoints.
- **Elevation filtering in `select_description_view`** excludes near-pole
  views because an axis or plane projects to a degenerate point/line for most
  objects from those viewpoints, making them poor choices for a description
  prompt.
- **`triangulation.py`'s `widest_pair`** generalizes the older fixed-`obj_id`
  (`"top"`/`"bottom"`) pairing convention to free-form point counts, picking
  only the single most-separated pair (not all combinations) to avoid
  injecting multiple near-duplicate interpretation planes from the same view;
  documented as a strict generalization that doesn't change behavior for the
  already-validated 2-point prompts.
- **`viz_colors.py` was extracted during a recent refactor** to consolidate
  RGB constants that were previously copy-pasted verbatim across
  `Mapping/visualize_rays.py`, `visualize_symmetry.py`,
  `export_symmetry_overlay.py`, and `render_symmetry_comparison.py`, so the
  GT/predicted color convention only needs to change in one place.
- **`naming.py`'s suffix-before-extension convention** keeps experiment
  output files separate from production output files without altering the
  production path when `experiment_id` is `None`.

## Known Limitations

- `pipeline_common/__init__.py` is empty — no package-level `__all__` or
  re-exports; every caller imports directly from the submodule.
- `cast_ray_patch` documents itself as only meant for `patch_size > 1`;
  `patch_size == 1` should go through `cast_ray` directly for an identical
  result without the averaging overhead, but this is a usage convention, not
  an enforced check inside the function.
- `cluster_hdbscan` raises `ImportError` with a custom message if
  `scikit-learn` (specifically `sklearn.cluster.HDBSCAN`, available since
  scikit-learn>=1.3) is not installed — a hard dependency for that one
  function only.
- `widest_pair`'s O(n^2) pairwise comparison is fine for the small point
  counts involved (tens of points per view) but is a known-limits case
  explicitly called out in the docstring: a view with exactly 1 point per
  image (observed for some Flow C prompts) still cannot yield a pair
  regardless of selection strategy.
- `select_description_view` falls back to the full (unfiltered) entry list
  if no metadata entries fall inside `elevation_range` — documented as
  "should not happen with the standard 114-view Fibonacci sampling" but not
  otherwise guarded against.
- `load_mesh_vertices` swallows all exceptions and returns `None` on any
  load failure, which means callers must explicitly check for `None`; the
  underlying error is not surfaced or logged.
- `triangulation.py`'s functions require `fov_deg`/`image_size` to be passed
  explicitly per call (rather than assumed constant), noted in the module
  docstring as a deliberate change from the original notebook prototype
  (`test_pipeline_sin_malla.ipynb`) where these were notebook-global
  constants for a single test object — a real dataset has objects at
  different sizes/FOV.
