# Mapping (2D → 3D → Symmetry Fitting)

## Description

The Mapping stage turns Molmo2's 2D pointing predictions into a fitted 3D
symmetry axis (`axis_sym`) or plane (`plane_sym`) for each object. It sits
between the VLM inference stage (`MolmoPointing/`) and the evaluation stage
(`Mapping/evaluate.py`, covered separately), and is the core geometric
contribution of the pipeline: since the VLM never sees or reasons about 3D
geometry directly, all 3D structure must be reconstructed from calibrated
camera rays built from the known render camera parameters (`R`, `T`, FOV per
view, stored in `metadata_all.json` / `images_sent` blocks).

Two parallel, independent implementations exist. The **with-mesh pipeline**
(`Mapping/map_to_3d.py` → `Mapping/estimate_symmetry.py`) ray-casts each
predicted 2D point against the object's actual `.obj` mesh to get an exact 3D
surface hit, then fits an axis/plane via SVD, optionally after RANSAC
outlier rejection and/or spatial clustering, and optionally scores the fit
with an internal self-consistency metric (SDE, via `--objects-root`). The
**mesh-free pipeline** (`Mapping/estimate_symmetry_no_mesh.py`, newer,
described in `docs/pipeline_sin_malla.md` and
`docs/implementacion_pipeline_sin_malla.md`) never touches the mesh at all:
it triangulates "interpretation planes" from pairs of 2D points within each
view and intersects them across views to recover the 3D axis/plane directly
from camera geometry, which better matches the "no privileged 3D access"
framing of the thesis question. Both pipelines write to the same
`predicted_symmetry.json` output schema (under different method keys) so
`Mapping/evaluate.py` can score them uniformly.

## Key Files & Functions

| File | Function / Class | Responsibility |
|---|---|---|
| `Mapping/map_to_3d.py` | `process_object` | Per-object: loads mesh once, ray-casts every Molmo 2D point (all n_views groups, all size/lighting configs) to 3D, writes `mapped_points_3d[_EXP].json`. |
| `Mapping/map_to_3d.py` | `main` / `parse_args` / `preview` | CLI: object slicing across GPUs, `--patch-size` selection, confirmation prompt. |
| `Mapping/estimate_symmetry.py` | `fit_axis`, `fit_plane` | SVD fit: first PC (axis) or last PC (plane normal) of centered points. |
| `Mapping/estimate_symmetry.py` | `ransac_axis`, `ransac_plane` | RANSAC inlier selection (line/plane) with a bbox-diagonal-relative threshold. |
| `Mapping/estimate_symmetry.py` | `sde_axis`, `sde_plane` | Internal Symmetry Distance Error: reflect sampled mesh vertices through the fitted axis/plane, mean nearest-neighbor distance (KDTree) / bbox diagonal. |
| `Mapping/estimate_symmetry.py` | `collect_hit_points` | Builds the point cloud per n_views group according to `point_mode` (`independent`/`midpoint`/`all`). |
| `Mapping/estimate_symmetry.py` | `_make_entry`, `process_object` | Assembles the 2×2 method grid (`svd`/`ransac_svd`/`svd_sde`/`ransac_svd_sde`) and applies clustering before fitting. |
| `Mapping/estimate_symmetry_no_mesh.py` | `estimate_axis_no_mesh` | Axis: builds one interpretation plane per view via `widest_pair`, intersects ≥2 planes into a 3D line. |
| `Mapping/estimate_symmetry_no_mesh.py` | `estimate_plane_no_mesh` | Plane: triangulates a candidate line per view-pair, cross-products independent pairs into candidate normals, scores by "edge-on-ness", refits via SVD on the best candidate's views. |
| `Mapping/estimate_symmetry_no_mesh.py` | `detect_planes_no_mesh` | Sequential multi-plane consolidation: repeats `estimate_plane_no_mesh` on the remaining view pool, dedupes near-identical normals; only exercised with `--max-planes > 1`. |
| `Mapping/estimate_symmetry_no_mesh.py` | `process_object` | Per-object orchestration, pools axis interpretation planes across all size/lighting configs; for plane_sym processes each config independently and keeps the one using the most views (documented simplification). |
| `pipeline_common/camera.py` | `molmo_to_ndc`, `build_camera_rays`, `project_point` | Molmo [0,1000] → NDC conversion; NDC + `R`/`T`/FOV → world-space ray; and the exact forward-projection inverse (used for debugging/visualization). |
| `pipeline_common/camera.py` | `cast_ray`, `cast_ray_patch` | Single-ray mesh intersection (closest hit); patch-averaged multi-ray variant to stabilize grazing-angle noise. |
| `pipeline_common/triangulation.py` | `ray_dir_for_point`, `view_forward_direction` | Per-point and per-view (optical axis) world-space ray directions, without mesh intersection. |
| `pipeline_common/triangulation.py` | `interpretation_plane_normal` | Normal of the plane through the camera center and two rays (Bartoli & Sturm 2005) — any 3D line whose 2D projection passes through both points lies in this plane. |
| `pipeline_common/triangulation.py` | `triangulate_line` | Least-squares intersection of ≥2 interpretation planes into a 3D line (direction + point). |
| `pipeline_common/triangulation.py` | `widest_pair`, `get_point_by_obj_id` | Point-pairing strategies: most-separated pair (axis, obj_id-agnostic) vs. fixed obj_id 1/2 lookup (plane, bilateral-pair prompts). |
| `pipeline_common/clustering.py` | `cluster_points` | Greedy, order-dependent centroid clustering (threshold = 5% bbox diagonal); every point always assigned, no outlier concept. |
| `pipeline_common/clustering.py` | `cluster_hdbscan` | Density-based clustering (scikit-learn `HDBSCAN`) that explicitly drops noise points; `min_samples` is the swept parameter (2/3/5), `min_cluster_size=2` fixed. |
| `pipeline_common/datasets.py` | `load_mesh`, `load_mesh_vertices`, `OBJECTS_SUBDIR` | `.obj` loading (merges multi-geometry scenes) and the `axis_sym`/`plane_sym` → curated-object-folder mapping. |
| `pipeline_common/naming.py` | `exp_filename` | Appends `_<experiment_id>` before the extension so experimental runs never collide with production output files. |

## Inputs & Outputs

**Inputs:**
- `molmo_multiview[_<EXP>].json` per `(object_id, size, lighting)` — Molmo2's 2D points per `n_views` group, each view's camera `R`/`T`/FOV (`images_sent`), and `obj_id` tags per point. Written by `MolmoPointing/molmo_multiview_runner.py`.
- `manifest.json` per render folder — authoritative `image_size`/`fov` (falls back to the folder-name size / CLI `--fov` if absent).
- `.obj` mesh files under `<objects_root>/curated_axis_sym_obj|curated_plane_sym_obj/<object_id>.obj` — required by `map_to_3d.py` (ray casting) and optionally by `estimate_symmetry.py` (SDE scoring via `--objects-root`). Not read at all by `estimate_symmetry_no_mesh.py`.

**Outputs:**
- `mapped_points_3d[_<EXP>][_p{3,5}].json` (with-mesh only) — one 3D hit point (or `null`) per Molmo 2D point, per n_views group, with hit/miss counts. Written by `map_to_3d.py`.
- `predicted_symmetry[_<EXP>].json` — one fitted axis (`direction`+`origin`) or plane (`normal`+`origin`) per method per n_views group, under `n_views_predictions`. With-mesh methods: `svd`, `ransac_svd`, `svd_sde`, `ransac_svd_sde`. Mesh-free methods: `triangulation` (single axis/plane), or `triangulation_multiplane` (list of planes, plane_sym + `--max-planes > 1` only). Written by both `estimate_symmetry.py` and `estimate_symmetry_no_mesh.py` to the *same filename* (they must be run with distinct `--experiment-id`s to avoid overwriting each other, since neither script merges into an existing file's methods — each overwrites the whole file unless `--overwrite` is unset and it already exists).

## Results & Observations

No plots or aggregate metrics are computed in this stage itself — quantitative
results (angular error, AUC, SDE_ref/F1_ref, precision@θ) belong to
`Mapping/evaluate.py`, out of scope here. What this stage does surface:
- `n_hits`/`n_misses` per n_views group in `mapped_points_3d.json`, letting misses (rays that don't hit the mesh) be inspected directly.
- The internal `sde`/`accepted` field on `svd_sde`/`ransac_svd_sde` entries (self-consistency heuristic against `SDE_THRESHOLD = 0.05`) — informative only, not used to filter or select among methods automatically (per `CLAUDE.md`).
- `n_points_raw` vs `n_points_fit` per n_views group in `estimate_symmetry.py`'s output, exposing how much clustering (if enabled) reduced the point count before fitting.
- `estimate_symmetry_no_mesh.py`'s plane path records `n_candidates` and `good_views` per fitted plane, showing how many candidate normals were generated and which views were judged non-edge-on to the winning candidate.

## Key Decisions

- **RANSAC threshold and inlier iteration count are fixed constants**
  (`RANSAC_ITERS = 1000`, `RANSAC_THRESH = 0.05` i.e. 5% of point-cloud bbox
  diagonal) rather than CLI-configurable — keeps the 2×2 method grid
  comparable across all runs.
- **Two independent symmetry-fitting pipelines run in parallel rather than
  one replacing the other** — the mesh-free pipeline is explicitly designed
  to "run in parallel to the existing with-mesh pipeline... does not replace
  it, does not modify its output files" (module docstring), so both remain
  directly comparable via `--experiment-id`.
- **Axis point-pairing uses `widest_pair` (obj_id-agnostic) while plane
  point-pairing uses fixed `obj_id` 1/2 lookup** — axis "top/bottom" framing
  has no fixed geometric role per point, so the most-separated pair is a
  strict generalization; bilateral plane prompts give obj_id 1/2 a real
  left/right mirror-pair meaning, so no generalization is applied there
  (see `pipeline_common/triangulation.py` module comments and
  `docs/implementacion_pipeline_sin_malla.md` S2.1).
- **`point_mode` (`independent`/`midpoint`/`all`) must match the prompting
  strategy used upstream in `MolmoPointing/`** (per `CLAUDE.md`): `independent`
  assumes points already lie on the axis/plane; `midpoint` assumes bilateral
  pairs and requires ≥2 views structurally; `all` drops the pairing
  requirement entirely for variable-point-count prompts (Flow C).
- **Patch-based backprojection (`--patch-size 3/5`) writes to a separate
  output file rather than overwriting the exact single-ray result** — keeps
  exact-mode behavior byte-for-byte unchanged and allows direct comparison.
- **HDBSCAN's `min_cluster_size` is hardcoded to 2, not exposed as a sweep
  parameter** — scikit-learn's own default of 5 would over-aggressively
  label everything as noise given how small these point clouds are (tens of
  points); `min_samples` (2/3/5) is the parameter actually swept.
- **Plane-sym in the mesh-free pipeline picks one (size, lighting) config
  per n_views group rather than pooling all configs like the with-mesh
  pipeline and the axis mesh-free path do** — documented in the code as "a
  known simplification" because plane candidate generation needs
  index-consistent view pairing within a single `points_by_image` set.
- **`estimate_symmetry_no_mesh.py` wraps each n_views group's fitting in a
  broad `except Exception: continue`** — a failure fitting one n_views group
  (e.g. too few valid views) does not abort the whole object; only that
  group is skipped.

## Known Limitations

- **`triangulation_multiplane` output (plane_sym, `--max-planes > 1`) is not
  consumed by the main evaluation flow** — `docs/data-schemas.md` and the
  module docstring both note it requires the separate
  `evaluate_plane_multiset`/`evaluate_plane_multi_from_pred` path in
  `evaluate.py`, not yet wired into the default per-object scoring.
- **No axis equivalent of multi-plane detection** — `detect_planes_no_mesh`
  and `--max-planes` are plane_sym-only; axis_sym always fits a single line.
- **RANSAC falls back to using all points as "inliers" if fewer than 2
  inliers are found** in any of the 1000 iterations (`ransac_axis`/
  `ransac_plane`), silently degrading to the same result as plain SVD in
  that edge case rather than failing or flagging it.
- **`cluster_points` (greedy) is order-dependent** — merging is done in
  input point order with incremental centroid updates, so a different point
  order could produce different clusters given the same threshold; not
  addressed since it's a documented behavior, not a bug.
- **`estimate_symmetry_no_mesh.py`'s plane detection requires ≥4 valid views
  (2 independent pairs)** to run at all (`ValueError` otherwise), a
  structurally higher bar than the with-mesh pipeline's ≥2-point minimum.
- **Mesh-free `process_object` for plane_sym only keeps a single "best"
  config per n_views group (by view count), discarding the rest** even when
  multiple (size, lighting) configs have usable data — see the "Key
  Decisions" note above; a documented simplification, not a bug, but it
  means some valid data is silently unused.
- **No cross-validation that `--point-mode` matches the actual upstream
  prompt strategy** — `collect_hit_points`/`estimate_symmetry_no_mesh.py`
  will silently produce a point cloud under whatever mode is passed, even if
  it mismatches the prompt convention used when generating
  `molmo_multiview.json`; this is a documented user responsibility, not a
  runtime check (see `CLAUDE.md`).
- **`edge_on_thresh` and `dup_angle_thresh_deg` in the mesh-free plane
  pipeline are magic-number defaults (`0.5`, `15.0` degrees)** exposed only
  partially via CLI (`--edge-on-thresh`; `dup_angle_thresh_deg` is not a CLI
  flag at all, hardcoded as `DUP_ANGLE_THRESH_DEFAULT`).
