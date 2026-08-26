# Interactive / Visualization Viewers

## Description

This area covers the repo's tools for looking at 3D symmetry results with the
naked eye rather than through aggregate metrics. Two genuinely different
things live here, and — contrary to an open question raised in a recent
architecture audit (`docs/audits/architecture-audit.md`, §5 and item 12) —
they are not duplicates of each other in their current form.

`InteractiveViewer/view_symmetries.py` is a small, standalone, Open3D-based
viewer that takes any `.obj` mesh plus a simple hand-written `.txt` symmetry
file (`plane`/`axis` entries with a normal/direction, a point, and an
optional confidence) and draws the mesh with the symmetry elements overlaid.
It has no knowledge of this thesis's render/prediction pipeline — it does not
read `molmo_multiview.json`, `mapped_points_3d.json`, or
`predicted_symmetry.json`, and does not distinguish ground truth from
predictions (both are drawn from the same generic `.txt` format, colored by
list order). It is a general-purpose "look at a mesh and some planes/axes"
utility.

The `Mapping/` visualization scripts (`visualize_symmetry.py`,
`visualize_rays.py`, `render_symmetry_comparison.py`,
`export_symmetry_overlay.py`) are, by contrast, deeply integrated debugging
and figure-generation tools for the actual pipeline: they read the
experiment's own JSON artifacts (`molmo_multiview[_EXP].json`,
`mapped_points_3d[_EXP].json`, `predicted_symmetry[_EXP].json`) and the
curated ShapeNet GT annotation (`<object_id>.txt`), and render them with
Polyscope (three of the four) or Matplotlib-over-photograph
(`export_symmetry_overlay.py`). They exist to answer pipeline-specific
questions — "did the 2D→3D ray casting hit the mesh correctly?", "how far is
the fitted axis/plane from ground truth for this object?", "what did Molmo2's
raw points look like before/after clustering?" — that the generic
`InteractiveViewer` cannot answer because it has no notion of an experiment,
an `n_views` group, a fitting method, or a point-collection mode. Whether
`InteractiveViewer/` should eventually be folded into `Mapping/` or merged
with `visualize_symmetry.py`'s feature set is exactly the open question the
architecture audit raises — this document describes what exists today, not a
recommended resolution.

## Key Files & Functions

| File | Function / Class | Responsibility |
|---|---|---|
| `InteractiveViewer/view_symmetries.py` | `load_mesh` | Loads a `.obj` via Open3D, computes normals, paints it gray. |
| `InteractiveViewer/view_symmetries.py` | `load_symmetries` | Parses the generic `N` / `plane ...` / `axis ...` `.txt` format into dicts. |
| `InteractiveViewer/view_symmetries.py` | `create_plane_geometry` / `create_axis_geometry` | Builds Open3D mesh/line/sphere/cone primitives for one symmetry element. |
| `InteractiveViewer/view_symmetries.py` | `main` | CLI entrypoint: loads mesh + `.txt`, launches an Open3D `Visualizer` window. |
| `InteractiveViewer/README.md` | — | Usage docs, controls, color legend, input format spec, "Core API" reference. |
| `Mapping/visualize_symmetry.py` | `load_gt`, `build_plane_quad`, `axis_endpoints` | Parses ShapeNet GT `.txt`; builds plane-quad / axis-segment geometry for Polyscope. |
| `Mapping/visualize_symmetry.py` | `main` | Loads mesh, `predicted_symmetry.json`, `mapped_points_3d.json`, and (optionally) GT; renders mesh + per-`n_views` hit-point clouds + predicted axis/plane + GT overlay in Polyscope. |
| `Mapping/visualize_rays.py` | `_collect_hit_points` | Mirrors `estimate_symmetry.collect_hit_points` — extracts 3D hit points per `point_mode` (`independent`/`midpoint`/`all`). |
| `Mapping/visualize_rays.py` | `main` | Recomputes rays from `molmo_multiview[_EXP].json` camera params (no dependency on `map_to_3d.py` having run); shows cameras, hit/miss rays, hit points, optional clusters, optional predicted and GT symmetry. |
| `Mapping/render_symmetry_comparison.py` | `render_panel` | Renders one Polyscope panel (mesh + one symmetry element, optional hit points) to a PNG. |
| `Mapping/render_symmetry_comparison.py` | `crop_pair_to_shared_content`, `_content_bbox` | Crops GT/predicted screenshots to a shared content bounding box so panel figures align. |
| `Mapping/render_symmetry_comparison.py` | `main` | Builds a print-ready GT-vs-predicted side-by-side figure from mesh + `predicted_symmetry[_EXP].json` only (no rendered photos needed). |
| `Mapping/export_symmetry_overlay.py` | `choose_view` | Picks the rendered photo view Molmo2 returned the most points for (unless `--view-index` is given). |
| `Mapping/export_symmetry_overlay.py` | `main` | Projects 3D GT/predicted elements onto one of the object's actual rendered photographs and composites a GT-vs-predicted Matplotlib figure. |
| `pipeline_common/viz_colors.py` | `COLOR_GT_AXIS`, `COLOR_GT_PLANE`, `COLOR_PRED_AXIS`, `COLOR_PRED_PLANE`, `COLOR_MESH`, `COLOR_CAMERA`, `COLOR_HIT_RAY`, `COLOR_MISS_RAY`, `COLOR_HIT_PT`, `COLOR_CLUSTER_PT` | Shared RGB color constants, consolidating what used to be copy-pasted across the `Mapping/` scripts. |

## Inputs & Outputs

**Inputs:**

- `InteractiveViewer/view_symmetries.py`:
  - `--mesh`: any `.obj` file.
  - `--symmetries`: a `.txt` file in the generic format `N` followed by `plane <nx ny nz px py pz [confidence]>` / `axis <...>` lines (defaults to same basename as the mesh). Not the ShapeNet GT format used elsewhere in the repo — this is a self-contained, ad hoc text spec documented only in the tool's own README.
- `Mapping/visualize_symmetry.py`, `Mapping/visualize_rays.py`, `Mapping/render_symmetry_comparison.py`, `Mapping/export_symmetry_overlay.py`:
  - Curated ShapeNet mesh + GT annotation: `<objects_root>/curated_{axis,plane}_sym_obj/<object_id>.obj` / `.txt` (ShapeNet symmetry-annotation format: `N` then `axis`/`plane` lines, no confidence field).
  - Pipeline JSON artifacts under `<renders_root or json_root>/<symmetry_type>/<object_id>/...`: `molmo_multiview[_EXP].json` (Molmo2 raw 2D points + camera params), `mapped_points_3d[_EXP].json` (ray-cast 3D hit points), `predicted_symmetry[_EXP].json` (fitted axis/plane per `n_views`/method).
  - `export_symmetry_overlay.py` additionally needs the actual rendered PNG photographs (`--photos-root`), which are not copied by `export_viz_samples.py` and may need to be fetched separately from the render machine.
  - CLI flags select `--n-views`, `--pred-method` (`svd`/`ransac_svd`/`svd_sde`/`ransac_svd_sde`), `--point-mode` (`independent`/`midpoint`/`all`), clustering method, experiment ID, etc.

**Outputs:**

- `InteractiveViewer/view_symmetries.py`: an interactive Open3D window only (no file output); prints a text summary of loaded symmetries to stdout.
- `Mapping/visualize_symmetry.py`, `Mapping/visualize_rays.py`: interactive Polyscope windows only (no file output); print text summaries (bbox diagonal, hit/miss ray counts, cluster counts, etc.) to stdout.
- `Mapping/render_symmetry_comparison.py`: a static composited image file (`--out`, `.png`/`.pdf`) — GT panel vs. predicted panel, screenshotted from Polyscope and assembled with Matplotlib; also prints the angular error between GT and the closest-matching prediction.
- `Mapping/export_symmetry_overlay.py`: a static composited image file (`--out`) with GT/predicted symmetry projected onto an actual rendered photo of the object; prints the chosen view index and angular error.

## Results & Observations

- Both figure-generating scripts (`render_symmetry_comparison.py`,
  `export_symmetry_overlay.py`) print the angular error (in degrees) between
  the predicted axis/plane and whichever ground-truth candidate it is
  closest to, mirroring `evaluate.py`'s min-over-GT convention for objects
  with more than one valid GT symmetry.
- `visualize_rays.py` prints ray-casting diagnostics (unique camera count,
  hit-ray count, miss-ray count) and, when `--show-clusters` is set, the
  raw-point-to-cluster-centroid reduction count.
- No persisted metrics, logs, or plots from these tools are checked into the
  repo (they are interactive/on-demand debugging and figure-export tools,
  not part of the automated evaluation pipeline) — anything they would
  "report" only appears at run time in stdout or the requested `--out` file.

## Key Decisions

- `InteractiveViewer/view_symmetries.py` uses Open3D while all four
  `Mapping/` visualization scripts use Polyscope (plus Matplotlib for
  compositing). `> Decision found but reason not documented.`
- The GT/predicted color convention (blue = GT axis, green = GT plane, red =
  predicted axis, magenta = predicted plane) is stated verbatim across
  multiple `Mapping/` script docstrings as a deliberate choice "so figures
  from either tool read consistently," and was consolidated into
  `pipeline_common/viz_colors.py` specifically to stop that convention from
  being copy-pasted in four places (per the file's own module docstring).
  `visualize_symmetry.py` predates/was not migrated to this shared module —
  it still defines its own local `PRED_COLOR`, `GT_AXIS_COLOR`,
  `GT_PLANE_COLOR` constants rather than importing from
  `pipeline_common/viz_colors.py`, even though its color values are
  functionally equivalent. Currently only `visualize_rays.py`,
  `render_symmetry_comparison.py`, and `export_symmetry_overlay.py` import
  from `viz_colors.py`.
- `render_symmetry_comparison.py` and `export_symmetry_overlay.py` both
  resolve ambiguity for objects with multiple GT symmetry candidates by
  picking whichever GT element is angularly closest to the prediction,
  explicitly said in-code to mirror `evaluate.py`'s own min-over-GT scoring
  so the figure's reported error matches what the evaluation pipeline would
  report.
- `export_symmetry_overlay.py` auto-selects the rendered view to overlay on
  (`choose_view`) by picking whichever image Molmo2 returned the most 2D
  points for, on the reasoning (implicit in the code, not spelled out in
  comments) that a view with more returned points is more likely to show the
  symmetry element clearly. `> Decision found but reason not documented.`
- `visualize_rays.py` recomputes rays directly from
  `molmo_multiview[_EXP].json` camera parameters rather than reading
  `mapped_points_3d.json`, explicitly so it works even before
  `map_to_3d.py` has been run for that experiment (stated in its docstring).

## Known Limitations

- `InteractiveViewer/view_symmetries.py` has no concept of ground truth vs.
  prediction, no `n_views`/experiment/method selection, and cannot read any
  of the pipeline's JSON artifacts — it only understands a hand-authored
  `.txt` symmetry file. It cannot be used to inspect an actual experiment's
  results without first converting that experiment's output into its
  bespoke `.txt` format (no such converter exists in the repo, as far as
  this review found).
- `export_symmetry_overlay.py` depends on the raw rendered PNGs being
  present at `--photos-root`, which are explicitly documented as NOT copied
  by `export_viz_samples.py` (too heavy) — the docstring notes they may need
  to be manually copied from wherever rendering was originally run.
- All four `Mapping/` visualization scripts require `polyscope` (interactive
  ones) and/or `matplotlib`/`pillow` (figure-export ones); none of these are
  listed as required by the core pipeline scripts, so they are optional,
  debug-only dependencies.
- `visualize_symmetry.py` does not import the shared
  `pipeline_common/viz_colors.py` module even though three sibling scripts
  now do — its local color constants must be kept in sync by hand if the
  shared palette changes.
- `render_symmetry_comparison.py` and `export_symmetry_overlay.py` both exit
  hard (`sys.exit`) if the requested `--pred-method` or GT element is
  missing, rather than falling back to another method or skipping — no
  partial-figure output is attempted.
- Per the architecture audit (`docs/audits/architecture-audit.md`), whether
  `InteractiveViewer/` should be folded into `Mapping/` or kept as a
  deliberately lightweight standalone alternative is an open, unresolved
  question — not something the code itself answers.
