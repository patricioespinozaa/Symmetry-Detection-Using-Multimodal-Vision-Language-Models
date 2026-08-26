# Results Consolidation, Metrics Computation & Performance Evaluation

## Description

This component is the final stage of the pipeline: it compares the symmetry
(axis or plane) predicted by `Mapping/estimate_symmetry.py` against the
ground-truth annotation for each ShapeNet object, computes a fixed set of
geometric error metrics per object, and aggregates them into per-`n_views`
summary tables that drive the thesis's ranking and ablation analysis.

The engine (`Mapping/evaluate.py`) reports two independent metric families
side by side without mixing their conventions: (1) the project's own
ground-truth-based metrics (`angular_error_deg`, `translation_error`,
`precision@θ`, `auc_angular`) plus an internal, opt-in-free "SDE" acceptance
heuristic computed earlier in `estimate_symmetry.py`, and (2) an opt-in
`--with-reference-metrics`/`--all` family (`SDE_ref`, `F1_ref`,
`f1_ref_hungarian`) that is a verbatim (or, for axis SDE, a deliberate
extension of a) port of an external reference paper's evaluation code, kept
separate so results can be compared against that paper's own convention
without polluting the project's primary metrics. A historical third family —
a broken in-house "SDE" formula that never verified the reflected point
landed on real geometry — was found to be invalid and was deleted outright
from `evaluate.py`; it is documented here only so it is not mistaken for a
currently-reported metric. `Mapping/compare_results.py` then consumes the
CSVs this stage produces to build cross-experiment comparison tables and
plots, `Mapping/cleanup_experiments.py` lets prompt iteration re-run just
part of the sweep, `Mapping/run_all_postprocessing.py` chains the whole
post-Molmo pipeline (map_to_3d → estimate_symmetry → evaluate → compare)
unattended over every discovered experiment, and a handful of ad hoc
`Mapping/check_*.py`/`Mapping/audit_view_indices*.py` scripts were written to
diagnose specific anomalies found while consolidating results (SDE/angular
discordance, independent-vs-midpoint pointing strategy, view-index sampling
bugs).

## Key Files & Functions

| File | Function / Class | Responsibility |
|---|---|---|
| `Mapping/evaluate.py` | `angular_error_deg`, `point_to_line_distance`, `point_to_plane_distance` | Core geometric error primitives (sign-agnostic angle in [0°,90°]; point-to-line / point-to-plane distance) |
| `Mapping/evaluate.py` | `auc_from_errors` | AUC of the precision-vs-threshold curve, truncated at `AUC_ANGULAR_MAX = 45.0` degrees |
| `Mapping/evaluate.py` | `evaluate_axis`, `evaluate_plane` | Per-object metrics for a single predicted axis/plane vs. ground truth (best-match against all GT planes for `plane_sym`) |
| `Mapping/evaluate.py` | `evaluate_plane_multiset`, `evaluate_plane_multi_from_pred` | Recall/precision over the *full* GT plane set for multi-plane predictions (`triangulation_multiplane` method) — scaffolding, not the single-plane default path |
| `Mapping/evaluate.py` | `evaluate_object` | Dispatches per-object evaluation across all `n_views` groups in `predicted_symmetry*.json`, selecting axis/single-plane/multi-plane logic |
| `Mapping/evaluate.py` | `compute_summary` | Aggregates per-object metrics into per-`n_views` summary stats; imputes `angular_error=90°`/`precision=0` for objects with no valid prediction; accumulates F1_ref TP/FP/FN globally (not per-object average) |
| `Mapping/evaluate.py` | `write_csv` | Writes the aggregated summary CSV, branching on single-plane/axis vs. multi-plane column schema |
| `Mapping/evaluate.py` | `calplaneloss`, `calaxisloss` | `SDE_ref`: area-weighted surface sample, squared distance to the real triangulated mesh (AABB tree via `gpytoolbox`) after reflecting through the predicted plane/axis. `calplaneloss` is a verbatim port of the reference paper; `calaxisloss` is this project's own extension (no axis SDE exists in the reference repo or, per literature review, in the field) |
| `Mapping/evaluate.py` | `f1_match_counts`, `f1_match_counts_hungarian`, `_f1_from_counts_by_threshold` | `F1_ref` (plane_sym only): greedy list-order matching (verbatim port, PRS-Net/E3Sym convention) vs. optimal bipartite assignment via `scipy.optimize.linear_sum_assignment` (Reflect3D/ArchSym 2025-2026 convention); F1 computed from dataset-level accumulated TP/FP/FN per threshold |
| `Mapping/evaluate.py` | `score_experiment_bulk`, `run_bulk_rescore`, `discover_experiment_ids`, `_MeshCache` | Bulk re-scoring mode (`--all`/`--experiment-ids`): re-computes `SDE_ref`/`F1_ref` for many already-fitted experiments × methods from disk without re-running ray-casting/SVD/RANSAC, caching shared `(direction/normal, origin)` fits |
| `Mapping/evaluate.py` | `parse_true_label`, `bbox_diagonal` | Parses `.txt` ground-truth label files; computes bounding-box diagonal used to normalize `translation_error` |
| `Mapping/compare_results.py` | `load_csvs` | Loads and concatenates all `eval_*_summary.csv` files for a symmetry type, exact-matching `--experiment-id` (no filename-prefix collisions) |
| `Mapping/compare_results.py` | `plot_metrics_by_nviews`, `plot_precision_curve`, `plot_acceptance_rate`, `plot_valid_by_prompt` | Generate comparison plots: metrics vs. `n_views` per method/experiment; continuous precision-vs-threshold curve at max `n_views`; SDE acceptance rate (`svd_sde`/`ransac_svd_sde` only); % objects with a valid prediction per prompt |
| `Mapping/compare_results.py` | `print_table` | Console table of `experiment × method × n_views` with core metrics |
| `Mapping/cleanup_experiments.py` | `cleanup_object`, `preview` | Removes specific `n_views` keys from `molmo_multiview*.json` files (per object/size/lighting) to allow re-running only part of a prompt experiment |
| `Mapping/run_all_postprocessing.py` | `discover_experiments`, `process_experiment`, `run_core_stages`, `run_clustering_sweep`, `run_patch_sweep`, `final_consolidation` | Unattended sweep entrypoint: chains map_to_3d → estimate_symmetry → evaluate (×4 methods) → compare_results for every discovered Molmo experiment and both symmetry types, plus a clustering (C2/C3) and patch-size (p3/p5) variant sweep per experiment; resumable by skipping already-completed stages |
| `Mapping/check_independent_vs_midpoint.py` | `load_angular_errors` | Ad hoc diagnostic: compares angular error between `--point-mode independent` vs. `midpoint` axis-prompt variants at fixed `n_views=26`, `method=svd` |
| `Mapping/check_sde_vs_angular.py` | `pearson`, `spearman_via_ranks`, `load_pairs` | Ad hoc diagnostic: correlates `angular_error_deg` vs. the internal `sde` field and flags "discordant" objects (high angular error, low SDE) from already-produced `eval_*_results.json` files |
| `Mapping/check_origin_full.py` | `sphericity`, `load_eval_pairs` | Ad hoc diagnostic (successor to archived `check_origin_compactness.py`): population-wide comparison of `dist_origin_centroid_norm`/`sphericity` vs. `angular_error_deg`/`sde` across an entire experiment, not just a discordant-tail sample |
| `Mapping/archive/check_origin_compactness.py` | — | Archived predecessor of `check_origin_full.py`: compared only the discordant tail (from `check_sde_vs_angular.py`) against a random control sample, rather than the full population |
| `Mapping/audit_view_indices_v2.py` | `expected_indices_for`, `classify`, `infer_standard_path_info` | Audits every `molmo_multiview*.json` found recursively under a search root for a view-sampling bug (sequential `[0..n_views-1]` indices instead of the correct Fibonacci-spaced linspace indices), including files outside the standard directory convention |
| `Mapping/archive/audit_view_indices.py` | `expected_indices_for`, `classify` | Archived predecessor of `audit_view_indices_v2.py`: same sequential-vs-spaced audit, but only over the fixed `<renders_root>/<symmetry_type>/...` directory convention (no recursive search) |

## Inputs & Outputs

**Inputs:**
- `predicted_symmetry[_<experiment_id>].json` per object (from
  `Mapping/estimate_symmetry.py`) — up to four fitted estimates
  (`svd`, `ransac_svd`, `svd_sde`, `ransac_svd_sde`, or
  `triangulation`/`triangulation_multiplane`) per `n_views` group.
- Ground-truth `.txt` label files per object (axis: direction + origin;
  plane: one or more normal + origin entries), under
  `<objects_root>/<OBJECTS_SUBDIR[symmetry_type]>/`.
- The object's `.obj` mesh (vertices always; vertices+faces when
  `--with-reference-metrics` is used, for `SDE_ref`'s AABB-tree surface
  sampling via `gpytoolbox`).
- For bulk re-scoring (`--all`/`--experiment-ids`): every
  `predicted_symmetry_<EXP>.json` already on disk under
  `<renders_root>/<symmetry_type>/`, discovered by filename pattern.
- For `compare_results.py`/`run_all_postprocessing.py`: the `eval_*_summary.csv`
  and `eval_*_results.json` files produced by `evaluate.py` itself, plus
  `predicted_symmetry_<EXP>.json` (for the SDE acceptance-rate plot).

**Outputs:**
- `eval_{sizes}_{lightings}[_{experiment_id}]_{method}_results.json` — per-object,
  per-`n_views` metrics, under `<renders_root>/<symmetry_type>/`.
- `eval_{sizes}_{lightings}[_{experiment_id}]_{method}_summary.csv` — aggregated
  per-`n_views` metrics, same directory.
- `reference_metrics_{axis,plane}.csv` (or `--out` path) — bulk re-scoring mode
  output: one row per `(experiment, method, n_views)` with `sde_ref_mean/min/max`,
  `f1_ref`, `f1_ref_hungarian`.
- `compare_results.py`: a combined comparison CSV
  (`<csv-dir>/experiments_<DD_MM_YYYY>/{prefix}_comparison.csv`) and PNG plots
  under `--save-dir` (`{prefix}_<metric>.png`, `{prefix}_precision_curve_nv<N>.png`,
  `{prefix}_acceptance_rate.png`, `valid/{prefix}_valid.png`).
- `run_all_postprocessing.py`: a greppable stage log (`log_postprocesos.txt`)
  and a full stdout/stderr transcript (`log_postprocesos_full.txt`), plus
  `<results_root>/all_experiments_comparison.csv` merging both symmetry types.
- `cleanup_experiments.py`: rewrites/deletes `molmo_multiview[_<EXP>].json`
  files in place (no new output files).
- The diagnostic `check_*`/`audit_view_indices*` scripts write one-off CSVs
  under `scratch/` (now removed from git tracking, see Known Limitations) —
  not part of the production pipeline output.

## Results & Observations

- Console tables (`evaluate.py::main`, `compare_results.py::print_table`)
  print per-`n_views` rows of `angular_error_mean/median`, `AUC_angular`,
  `precision@{5,10}°`, `translation_error_mean`, and (opt-in)
  `SDE_ref`/`F1_ref`/`F1_ref_hungarian`.
- `compare_results.py::plot_acceptance_rate` reports the fraction of
  predictions with the internal `accepted=True` flag (`SDE_THRESHOLD = 0.05`)
  for the `svd_sde`/`ransac_svd_sde` methods only.
- Measured cost (`docs/metricas_evaluacion.md` §5.5): `SDE_ref`/`F1_ref`
  re-scoring is ~21 ms per prediction; the full sweep (~220 experiments ×
  both symmetry types, ~2M predictions) takes ~6 hours sequential CPU time —
  the reason `--with-reference-metrics`/`--all` are opt-in rather than run by
  default.
- `docs/metricas_evaluacion.md` §5.2–5.3 records two methodological findings
  from comparing against the external reference repo (`EnhancedBackProjection`):
  (1) with a best-match framing and a single prediction per object,
  `AUC_angular` can look near-perfect while `F1_ref` (full-set recall) is low
  on objects with multiple GT planes — not a contradiction, different
  questions; (2) `F1_ref` structurally favors multi-candidate methods (the
  reference method predicts up to 10 plane candidates per object vs. this
  pipeline's 1), so it should never be reported as a like-for-like comparison
  without stating that difference.
- The ad hoc diagnostics found concrete anomalies during the sweep: a
  view-index sampling bug (sequential vs. correctly Fibonacci-spaced
  `n_views` sub-sampling, audited by `audit_view_indices_v2.py`), and
  discordant objects where low internal SDE co-occurred with high angular
  error, investigated via mesh compactness/sphericity
  (`check_sde_vs_angular.py` → `check_origin_full.py`) to check whether a
  plane through a compact object's centroid was scoring artificially well —
  motivating part of the decision to promote `SDE_ref` (real-surface,
  squared distance) over any in-house plane-distance formula.

## Key Decisions

- **`AUC_ANGULAR_MAX = 45.0`** (not 90°): every angular error ≥45° is treated
  as total failure regardless of magnitude. No benchmark canonizes this
  value; `docs/metricas_evaluacion.md` explicitly documents it as a
  defensible-but-own choice (>45° means "more orthogonal than parallel to
  GT") rather than a literature standard.
- **`ANGULAR_THRESHOLDS = [5, 10, 15]` degrees** for `precision@θ`: not
  canonical thresholds either — documented in `docs/metricas_evaluacion.md`
  as the project's own reasonable choice, to be stated as such in the thesis.
- **The old `symmetry_distance_error` formula was deleted, not fixed
  in place**: it measured 2× mean vertex-to-predicted-plane distance without
  ever checking that the *reflected* point landed on real geometry (a plane
  through a sphere's centroid would score ~0 regardless of whether the
  sphere is actually symmetric about it). Confirmed via literature review to
  have no real basis, so it was removed outright along with the
  `sde`/`sde_mean`/`auc_sde`/`precision_sde_*` summary fields it populated —
  see `docs/actualizacion_metricas.md` §0–2 and `docs/metricas_evaluacion.md`
  §2.1 note. `SDE_ref` is the only SDE `evaluate.py` reports today, for both
  symmetry types equally.
- **Objects without a valid prediction are imputed with `angular_error=90°`**
  (worst case) rather than excluded from `compute_summary`, so that objects
  where Molmo produced no usable points penalize the aggregate metrics
  instead of being silently dropped (which would inflate them).
  Translation/`SDE_ref` metrics, by contrast, are computed only over valid
  predictions — there is no principled worst-case value to impute for a
  continuous distance.
- **`F1_ref`/`f1_ref_hungarian` accumulate TP/FP/FN globally across the whole
  dataset per threshold before computing F1**, not a per-object F1 averaged
  afterward — those give different numbers whenever `n_true_planes` varies
  across objects. This was a real bug caught during implementation (an
  earlier version averaged per-object F1) by comparing the normal mode
  against the bulk mode, which already replicated the reference paper's
  accumulation correctly.
- **`Mapping/reference_metrics.py` was merged into `Mapping/evaluate.py`**
  rather than kept as a separate module, so the two metric families share one
  entry point and one CLI; the merge preserved every function name
  (`calplaneloss`, `calaxisloss`, `f1_match_counts`, `THRESHOLDS_INLIER`,
  `N_SAMPLES_DEFAULT`) so any code importing the old module needs only a
  changed import path.
- **`--with-reference-metrics`/`--all` are opt-in, not default**: they
  require `gpytoolbox` (an extra dependency) and are meaningfully more
  expensive (mesh load + AABB tree + surface sampling per object) —
  intended to be pointed at a short list of "winner" experiments, not the
  full sweep, mirroring the original standalone script's design.
- **Greedy (`f1_match_counts`) and Hungarian (`f1_match_counts_hungarian`)
  F1 matching are both kept, side by side**, rather than replacing the
  historical greedy convention: greedy preserves comparability with the
  PRS-Net/E3Sym-based reference work already used in the thesis; Hungarian
  tracks the newer optimal-assignment convention (Reflect3D/ArchSym
  2025-2026). They are mathematically identical with exactly one predicted
  plane per object (today's case) and are expected to diverge only once a
  multi-plane detector is wired in.
- **`translation_error_normalized` (by bbox diagonal) was added for both
  axis and plane** so distances are comparable across objects of different
  scales; mesh vertices are now loaded unconditionally in `evaluate.py::main`
  for this purpose (previously mesh loading was plane-only).
- **`evaluate_plane_multiset`/`evaluate_plane_multi_from_pred` are
  implemented but intentionally not wired into the default single-plane
  path** — see Known Limitations.

## Known Limitations

- **Multi-plane recall/precision (`evaluate_plane_multiset`) is scaffolding,
  not active on the main path**: `predicted_symmetry.json` today still stores
  exactly one plane per `(object, method, n_views)`, so `evaluate_object`
  only takes the multi-plane branch when it sees the
  `triangulation_multiplane` method's `{"planes": [...]}` shape. It has no
  effect on the ~220 already-run sweep experiments.
- **No F1 metric exists for `axis_sym`** — confirmed as an intentional
  omission (no established convention in the reference repo or, per
  literature review recorded in `docs/metricas_evaluacion.md` §4, in the
  field generally), not a gap to fill.
- **`SDE_ref`'s axis variant (`calaxisloss`) is this project's own
  extension**, not a verbatim port — there is no ground-truth-based axis SDE
  in the reference repository to port from.
- **`F1_ref` structurally favors multi-candidate detectors**: with exactly
  one predicted plane per object, its recall over multiple GT planes has a
  mathematical ceiling of `1/n_true_planes`. Documented as a limitation to
  state explicitly whenever `F1_ref` is compared against an external
  multi-candidate method, not something the metric itself corrects for.
- **`--ref-seed`/reproducibility**: `SDE_ref` surface sampling seeds via
  `np.random.seed`, which mutates the shared NumPy global RNG state — a
  caller running other randomized code around `evaluate.py` calls in the
  same process could get seed-order-dependent results (only relevant to
  library/notebook use, not the CLI entrypoint).
- **Bulk re-scoring mode assumes `svd`/`svd_sde` (and
  `ransac_svd`/`ransac_svd_sde`) share the exact same fit** to justify its
  ~50% cache savings — correct today because the `_sde` variants only add a
  field without re-fitting, but this is an implicit coupling to
  `estimate_symmetry.py`'s current behavior, not something `evaluate.py`
  itself verifies.
- **`compare_results.py::print_table` and `plot_metrics_by_nviews` still
  reference legacy `sde_mean`/`auc_sde`/`precision_sde_010` columns** via
  `if c in df.columns` guards — harmless (they simply won't render for
  current CSVs, which no longer contain those columns) but the code paths
  referencing the removed metric were not cleaned out of this file, only
  the CSV writer that used to produce them.
- **Ad hoc diagnostic scripts (`check_*.py`, `audit_view_indices*.py`) are
  not part of the production pipeline** — they were written to investigate
  specific anomalies (view-index sampling bug, SDE/angular discordance,
  pointing-strategy comparison) and read already-produced `eval_*`/
  `predicted_symmetry_*` files rather than being invoked by
  `run_all_postprocessing.py`. Their one-off CSV outputs under `scratch/`
  were removed from git tracking (per the working tree's current status),
  so results from past runs are not preserved in the repo.
- **`EnhancedBackProjection` paper citation is unverified**: the reference
  repo's own README attributes it to "Aguirre & Sipiran, WACV 2026," but
  `docs/metricas_evaluacion.md` §4.1 explicitly flags this as read from the
  repo only, not confirmed via independent bibliographic search — needs
  verification before citing formally in the thesis.
- **Literature-review claims behind the metric decisions are one level
  removed from primary sources**: `docs/metricas_evaluacion.md` §4.2 notes
  its literature summary (PRS-Net as the SDE convention's origin, F1
  multi-candidate precedents in E3Sym/Je et al., absence of an axis-F1
  standard) came from an external web-search session pasted back in, not
  independently verified paper-by-paper.
