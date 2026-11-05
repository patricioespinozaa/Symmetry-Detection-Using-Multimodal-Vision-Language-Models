# Random Baseline (uniform random direction)

## Description

`RandomBaseline/` answers "is the method better than guessing?" for both
symmetry types. The reference predictor ignores the renders and draws the axis
direction (axis_sym) or plane normal (plane_sym) uniformly on the sphere; a
random plane passes through the origin (`d = 0`), i.e. the centre of the
normalized, centred object. It is scored with exactly the rules of
`Mapping/evaluate.py` and compared against a method's results. By default it
predicts every object; `--coverage-mode matched` gives it the method's
coverage instead. Theory, closed-form values and the statistical options come from
`docs/literature/baseline_aleatorio.md`. This module is the implementation
of that document. It is independent of the pipeline: it only reads GT `.txt`
labels and the outputs of `evaluate.py`/`compare_results.py`, and it writes
its own CSVs and LaTeX tables. Only option A of the literature doc (§5) is
implemented; the paired Wilcoxon test (option B) was removed.

## Key Files & Functions

| File | Function / Class | Responsibility |
|---|---|---|
| `RandomBaseline/chance.py` | `mean_error_deg`, `median_error_deg`, `std_error_deg`, `precision_at`, `auc_angular`, `analytic_summary` | Closed-form chance values against ONE GT element, generalized to coverage `c` (mixture with a point mass at 90°) |
| `RandomBaseline/simulation.py` | `GroundTruth`, `load_ground_truth` | GT labels parsed with `evaluate.parse_true_label`, NaN-padded to `(N, K, 3)` vectors and `(N, K, 4)` reference-convention planes |
| `RandomBaseline/simulation.py` | `random_directions`, `best_match_errors`, `f1_counts`, `summarize_errors` | Vectorized scoring identical to `evaluate_axis`/`evaluate_plane` (best match, rounded to 4 dp), `f1_match_counts`/`f1_match_counts_hungarian` (single predicted plane) and `compute_summary` (mean/median/std, `auc_from_errors`, precision with strict `<`) |
| `RandomBaseline/simulation.py` | `simulate_random_baseline` | R simulated datasets: all objects, a fixed coverage mask, or a random subset of `n_objects` per repetition |
| `RandomBaseline/multiplane.py` | `random_plane_sets`, `draw_plane_counts`, `multiset_scores`, `multiset_f1_counts`, `adjusted_metrics`, `simulate_random_multiplane` | Random baseline for `triangulation_multiplane`: a set of separated random planes per object, scored like `evaluate_plane_multiset` and `f1_match_counts`/`f1_match_counts_hungarian` |
| `RandomBaseline/significance.py` | `compare_metric` | Option A: 95% interval, one-sided empirical p, z, normal-approximation p |
| `RandomBaseline/latex_tables.py` | `axis_table`, `plane_table` | LaTeX tables (booktabs + adjustbox, same style as `utils/generar_tablas_latex.py`) of the random baseline on the full dataset |
| `RandomBaseline/run_baseline.py` | `run_reference`, `run_compare`, `run_latex`, `compare_summary_csv`, `compare_eval_results` | CLI (`reference` / `compare` / `latex`) |
| `RandomBaseline/tests/test_random_baseline.py` | — | Equivalence against `evaluate.py` functions + agreement with the closed form |

## Inputs & Outputs

**Inputs:**
- `<objects_root>/curated_{axis,plane}_sym_obj/<id>.txt` (GT labels).
- `compare --summary-csv`: `eval_*_summary.csv` (evaluate.py) or
  `*_comparison.csv` (compare_results.py). Each row needs `n_views`,
  `n_total`, `n_objects` and the metric columns.
- `compare --eval-results`: `eval_*_results.json` (evaluate.py per-object
  results). Method metrics are recomputed with `evaluate.compute_summary`.
  Only needed for `--coverage-mode matched` with the exact per-object mask;
  with the default `all` it gives the same random baseline as the CSVs.

**Outputs:**
- `reference --out`: per `(symmetry_type, coverage, metric)`, the closed form
  for a single GT, the simulated mean/std and the 95% interval.
- `compare --out` (default `RandomBaseline/outputs/random_baseline_comparison.csv`):
  per `(experiment, method, n_views, metric)`: the method and random
  coverage, the method value, the random mean/std/interval, `z`,
  `p_empirical`, `p_normal` and `better_than_random`.
- `latex --out-dir` (default `docs/resultados/tablas_azar/`): `azar_axis.tex`
  (`tab:axis_azar`: closed form, simulation, 95% interval) and
  `azar_plane.tex` (`tab:plane_azar`: simulation and 95% interval; F1_ref
  rows included). Full coverage. `azar_plane_multi.tex` (`tab:plane_azar_multi`):
  recall, precision and F1_ref with 1, 2 and 3 random planes per object.
- `cap4 --out` (default `Tesis_DCC_MDS/cap4_baselines.tex`): one file with six
  tables and `tab:base_*` labels, the three random tables above plus the fixed
  world-axis predictors of `Pipeline_Experiments/diagnostics/trivial_baselines.py`
  (`tab:base_axis_fijos`, `tab:base_plane_fijos`, `tab:base_plane_fijos_multi`).

## Key Decisions

- **Reuse `evaluate.py` instead of reimplementing metrics.** The module
  imports the constants and the aggregate functions (`ANGULAR_THRESHOLDS`,
  `AUC_ANGULAR_MAX`, `THRESHOLDS_INLIER`, `auc_from_errors`,
  `_f1_from_counts_by_threshold`, `parse_true_label`, `compute_summary`). Only
  the per-object loops are vectorized, and the tests check that they match
  `angular_error_deg`, `f1_match_counts` and `f1_match_counts_hungarian`
  exactly. If a metric rule changes in `evaluate.py`, the baseline follows it.
- **The random predictor answers every object by default (`--coverage-mode
  all`).** The method keeps its 90° imputation for the objects it missed, so
  this is the strictest comparison for the method: missing predictions count
  against it and the random predictor never abstains. On ndcfix 448 this
  matters for low-coverage runs: plane_v05_1 and plane_v06 at nv=14
  (63% and 53% coverage) beat random with matched coverage but not with
  full coverage. F1_ref is the exception: `compute_summary` leaves missing
  objects out of the method's F1, while the random F1 then covers all
  objects.
- **`--coverage-mode matched`** follows `docs/literature/baseline_aleatorio.md`
  §3.1: the random predictor gets the method's coverage (90° and no F1
  contribution for the rest). From summary CSVs only the count is known, so
  the covered subset is redrawn every repetition; from `eval_*_results.json`
  the exact mask is used.
- **Simulation is the reported number; the closed form is a check.** For
  plane_sym the best match over 1–3 GT planes lowers the chance error from
  57.30° to 54.59° on the full dataset, so the single-GT formulas are not
  valid there.
- **The planar LaTeX table has no closed-form column.** The single-plane
  value (57.3°) does not hold with 1–3 GT planes per object, so the caption
  mentions it only as the reason the simulated value is lower.
- **Common random numbers.** All rows use the same `--seed`, so differences
  between experiments are not caused by different random draws.
- **The multi-plane random predictor uses the method's number of planes.**
  Plane count drives recall and precision, so a fixed count would not be a fair
  reference. `--eval-results` gives the exact count per object; objects the
  method missed draw a count from the method's own counts under
  `--coverage-mode all`. `--summary-csv` only knows the mean.
- **Planes of one object stay at least 15 deg apart**
  (`DUP_ANGLE_THRESH_DEFAULT`), because `detect_planes_no_mesh` never keeps two
  planes closer than that. Without this the random predictor would waste planes
  on near-duplicates that the method cannot produce, and be easier to beat.
- **Coverage-adjusted metrics are compared too.** `recall_adj` and `f1_adj` use
  the thesis definition (`R_adj = R * N_obj / 850`, `F1_adj` from `R_adj` and
  `P`). Raw recall, precision and F1_ref leave missing objects out of the method,
  so these two are the fair comparison when coverage is low.

## Known Limitations

- **Multi-plane rows from a summary CSV are approximate.** With only the mean
  plane count, the random predictor gets counts as even as possible (a mean of
  2.36 gives only 2s and 3s), while the method may mix 1, 2 and 3. Use
  `--eval-results` for exact counts.
- **The separation rule assumes the default duplicate threshold.** Runs with
  another `dup_angle_thresh` need a different `min_separation_deg`.
- **SDE_ref and translation errors** have no random baseline here. SDE_ref
  would need meshes and gpytoolbox; translation errors depend on the `d = 0`
  choice and say nothing about information content.
- **`p_empirical` has a floor of `1/(R+1)`.** Use `z`/`p_normal`, or raise
  `--repetitions`, to tell very large effects apart.
- **The random subset in CSV `matched` mode assumes coverage is unrelated to
  the GT plane count.** If missing predictions concentrate on 1-plane objects, the
  exact-mask mode (`--eval-results`) gives a slightly different chance level.
- **The literature doc's σ = 21.57°** is a rounding slip:
  √(π − 3) rad = 21.56°. The z denominator for n = 850 is still 0.74°.
