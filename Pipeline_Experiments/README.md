# Pipeline_Experiments

Axis estimator of the thesis' final method (**EXP-E**, iterative residual reweighting) and
the read-only diagnostics that use the same experiment-id conventions.

This module reads existing `molmo_multiview_<EXP>.json` files and never touches
`Mapping/estimate_symmetry_no_mesh.py`, `Mapping/evaluate.py`, or their production output
files. Each run writes its own `predicted_symmetry_<NEW_ID>.json` under a new
`--experiment-id`, in the schema `Mapping/evaluate.py` already understands (`triangulation` /
`triangulation_multiplane` method keys), so evaluation and comparison work unmodified.

The ablations EXP-A..D and EXP-F, the EXP-LIT-1 candidates runner and the YAML-driven batch
runner (`run_batch.py`, `configs/`) were not reported in the thesis and were removed in the
2026-11 restructuring (`docs/plan_reestructuracion_repo.md`); they remain in the git history.
Their design and results are described in `docs/archive/experimentos_pipeline_sin_malla.md`.

## What's here

| What | Source | Needs GPU | Needs mesh |
|---|---|---|---|
| `triangulation_variants/exp_e_iterative_reweight.py` — iterative residual reweighting (axis only) | Hess-Flores et al.; Zhang et al., arXiv:2008.01258 | no | no |
| `view_filtering.py` — `none`/`per_point`/`any`/`both` view-discard policies | sandbox of August 2026 | no | no |
| `diagnostics/diagnose_center_bias.py` — X≈500 collapse diagnostic | sandbox of August 2026 | no | no |
| `diagnostics/diagnose_repeated_points.py` — points copied between views | — | no | no |
| `diagnostics/trivial_baselines.py` — fixed world-axis predictors | — | no | no |
| `diagnostics/diagnose_missing_reason.py` — why an object has no prediction | — | no | no |

## Directory layout

```
Pipeline_Experiments/
├── config.py                        # experiment_id auto-discovery, VariantConfig, output ids
├── view_filtering.py                # none/per_point/any/both view-discard policies
├── estimate_symmetry_variants.py    # single (experiment_id, variant) CLI -- sibling of
│                                     #   Mapping/estimate_symmetry_no_mesh.py
├── triangulation_variants/          # EXP-E, registered in __init__.py::VARIANTS
└── diagnostics/
```

## Running EXP-E

The final runs call it from `PlaneScoreData/run_resultados_finales.sh` and the
`PlaneScoreData/run_*.py` runners. Standalone:

```bash
python Pipeline_Experiments/estimate_symmetry_variants.py \
    --renders-root ../data/renders --symmetry-type axis_sym \
    --experiment-id axis_v04 --variant expE \
    --sizes 224 --lightings flat --overwrite

# Then evaluate/compare like any other no-mesh run:
python Mapping/evaluate.py --renders-root ../data/renders --objects-root ../data/objects \
    --symmetry-type axis_sym --experiment-id axis_v04_expE_nomesh --method triangulation
```

## Center-bias diagnostic standalone

```bash
python Pipeline_Experiments/diagnostics/diagnose_center_bias.py \
    --renders-root ../data/renders --symmetry-type axis_sym --all \
    --out-detail ../results/diagnostics/axis_center_bias_detail.csv \
    --out-summary ../results/diagnostics/axis_center_bias_summary.csv
```

## Notes / known limitations

- `expE` is axis-only: its point-then-direction iterative framing has no documented plane
  counterpart. The plane estimator of the final method is `PlaneScoreData/plane_score_data.py`.
- The axis variant processes each `(size, lighting)` config independently and keeps the one
  using the most views, instead of pooling raw rays across configs the way
  `Mapping/estimate_symmetry_no_mesh.py`'s axis branch does — see the comment in
  `estimate_symmetry_variants.py::process_object`. In practice this only matters when a single
  prompt run spans more than one size/lighting combination.
