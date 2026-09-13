# Pipeline_Experiments

Production-scale (850-object) implementation of the experiments prototyped in
`Experiments/*.ipynb` (sandbox notebooks on ~30 curated objects) and in
`docs/verificacion_metricas_literatura.md`. Everything here is **config-driven**:
instead of hardcoding one prompt/experiment_id the way the sandbox notebooks
did, every script accepts an explicit list of `--experiment-id` values, and
`run_batch.py` can auto-discover every prompt run you've already executed
(`molmo_multiview_<ID>.json` already present under `<renders_root>/<symmetry_type>/`)
and apply the configured experiments to all of them in one call.

This module reads existing `molmo_multiview_<EXP>.json` files and never
touches `Mapping/estimate_symmetry_no_mesh.py`, `Mapping/evaluate.py`, or
their production output files — every ablation writes its own
`predicted_symmetry_<NEW_ID>.json` under a new `--experiment-id`, in the
exact schema `Mapping/evaluate.py` already understands (`triangulation` /
`triangulation_multiplane` method keys), so the existing evaluation and
comparison stages work against it unmodified.

## What's here, and what it maps to

| What | Source | Needs GPU | Needs mesh |
|---|---|---|---|
| `triangulation_variants/exp_a_weighted.py` — pixel-separation weighted SVD | Bartoli & Sturm, CVIU 2005 | no | no |
| `triangulation_variants/exp_b_point_then_direction.py` — anchor/direction split (axis only) | Wu et al. 2021 | no | no |
| `triangulation_variants/exp_c_ransac2d.py` — RANSAC 2D point-pair selection | Recker et al., WACV 2013 | no | no |
| `triangulation_variants/exp_d_confidence.py` — edge-penalized weight | AssemblyHands-X, arXiv:2509.23888 | no | no |
| `triangulation_variants/exp_e_iterative_reweight.py` — iterative residual reweighting (axis only) | Hess-Flores et al.; Zhang et al., arXiv:2008.01258 | no | no |
| `triangulation_variants/exp_f_sde_gate.py` — real-SDE_ref plane acceptance gate (plane only) | PRS-Net, Gao et al., IEEE TVCG 2021 | no | **yes** (`--objects-root`) |
| `view_filtering.py` — `none`/`per_point`/`any`/`both` view-discard policies | `Experiments/sandbox_pipeline_server_final.ipynb` §14 | no | no |
| `candidates_then_select/` — EXP-LIT-1, real Molmo2 candidate selection | CVPC arXiv:2512.04686; ZeroDex arXiv:2606.19340 | **yes** | no |
| `diagnostics/diagnose_center_bias.py` — X≈500 collapse diagnostic | `Experiments/sandbox_pipeline_server_updated_6_pts_v2.ipynb` | no | no |

`axis_lit2_grid` (Grid Overlay, MOKA arXiv:2403.03174) and `axis_lit3_cot`
(Structured CoT, arXiv:2507.13362) needed no new code — they were already
real, runnable prompts in `MolmoPointing/prompts_registry.py` /
`--grid-overlay`; run them with `MolmoPointing/molmo_multiview_runner.py`
like any other prompt, then feed the resulting `--experiment-id` into this
module like any other source experiment.

`estimate_symmetry_variants.py` also exposes `--dup-angle-thresh` and
`--sde-gate` as real CLI flags (previously hardcoded constants in
`Mapping/estimate_symmetry_no_mesh.py`), so both can be swept.

## Directory layout

```
Pipeline_Experiments/
├── config.py                        # YAML config schema + experiment_id auto-discovery
├── view_filtering.py                 # none/per_point/any/both view-discard policies
├── estimate_symmetry_variants.py     # single (experiment_id, variant) CLI -- sibling of
│                                      #   Mapping/estimate_symmetry_no_mesh.py
├── run_batch.py                      # config-driven orchestrator (the "run it on
│                                      #   ALL my prompt executions" entrypoint)
├── triangulation_variants/           # EXP-A..F, registered in __init__.py::VARIANTS
├── candidates_then_select/           # EXP-LIT-1 real inference (needs GPU)
├── diagnostics/
│   └── diagnose_center_bias.py
└── configs/
    └── experiments.example.yaml
```

## Quick start: run everything on every prompt you've already tried

```bash
conda activate tesis_env
cp Pipeline_Experiments/configs/experiments.example.yaml Pipeline_Experiments/configs/my_run.yaml
# edit renders_root/objects_root if needed -- defaults assume you run this
# from the repo root with data/ one level up, same convention as every other
# Mapping/ script.
python Pipeline_Experiments/run_batch.py --config Pipeline_Experiments/configs/my_run.yaml
```

With `experiment_ids: {axis_sym: auto, plane_sym: auto}` (the example
config's default), this discovers every `molmo_multiview_<ID>.json` prompt
run already present under `data/renders/` — whatever naming convention its
`--experiment-id` used (e.g. plain `axis_v06`, the prompt_id itself, or an
older sweep's `axis_v06_nomesh`) — runs every
configured variant on top of each one, runs the `center_bias` diagnostic on
each, then chains `Mapping/evaluate.py` (optionally with
`--with-reference-metrics`) and `Mapping/compare_results_no_mesh.py` so the
existing comparison CSVs (and `Experiments/analisis_prompts_no_mesh.ipynb`,
which reads them) pick up the new experiment_ids automatically — no changes
needed there.

Run only part of the pipeline with `--skip-variants` / `--skip-diagnostics` /
`--skip-evaluate` / `--skip-compare`.

## Parallelizing across CPU cores (not GPUs)

Nothing in `Pipeline_Experiments` touches a GPU — the triangulation variants
are pure numpy, and `expF`/`--with-reference-metrics` use `gpytoolbox`
(CPU AABB tree), same as `Mapping/evaluate.py` always has. Only
`candidates_then_select/molmo_candidates_runner.py` (EXP-LIT-1's real
inference) needs one. So on a multi-GPU box those GPUs are irrelevant to
`run_batch.py`; if you want it to run faster, split the CPU-bound object
list across parallel processes with `--shard-id`/`--num-shards` instead:

```bash
# 2 shards in parallel, each doing half the objects -- must skip
# diagnostics/evaluate/compare (they each need EVERY object's output)
python Pipeline_Experiments/run_batch.py --config my_run.yaml \
    --shard-id 0 --num-shards 2 --skip-diagnostics --skip-evaluate --skip-compare &
python Pipeline_Experiments/run_batch.py --config my_run.yaml \
    --shard-id 1 --num-shards 2 --skip-diagnostics --skip-evaluate --skip-compare &
wait

# then once, unsharded, over the now-complete object set:
python Pipeline_Experiments/run_batch.py --config my_run.yaml --skip-variants
```

`run_batch.py` refuses to run diagnostics/evaluate/compare together with
`--num-shards > 1` (it would score a partial object set and race two
processes writing the same CSV) — pass the `--skip-*` flags shown above on
every sharded invocation.

## Config schema (`configs/*.yaml`)

```yaml
dataset:
  renders_root: ../data/renders
  objects_root: ../data/objects
  symmetry_types: [axis_sym, plane_sym]
  sizes: [224]
  lightings: [flat]
  max_objects: null        # null = every object; an int for a quick smoke test
  overwrite: false

experiment_ids:
  axis_sym: auto           # or an explicit list: [axis_v06_nomesh, axis_v08_nomesh]
  plane_sym: auto

variants:                  # one entry per (variant, hyperparameter) combination to run
  - variant: expA           # expA | expB | expC | expD | expE | expF
    filter_policy: none     # none | per_point | any | both
    max_planes: 1           # plane_sym only
    edge_on_thresh: 0.5      # plane_sym only
    dup_angle_thresh: 15.0   # plane_sym only
    sde_gate: 0.02           # plane_sym + expF only

diagnostics:
  - center_bias

evaluation:
  run_evaluate: true
  with_reference_metrics: true
  run_compare: true
  extra_args: []            # extra Mapping/evaluate.py flags, passed through verbatim

results_dir: ../results
```

Each `variants` entry becomes its own output `--experiment-id` via
`config.build_output_experiment_id`, which always appends a trailing
`_nomesh` (added if the source didn't already have one, never doubled if it
did) because `Mapping/compare_results_no_mesh.py` only discovers
experiment_ids ending in `_nomesh`: `"axis_v06"` + `expA` →
`"axis_v06_expA_nomesh"`; `"axis_v06_nomesh"` + `expA` →
`"axis_v06_expA_nomesh"` too. A variant that doesn't apply to a given
`symmetry_type` (e.g. `expB`/`expE` for `plane_sym`, `expF` for `axis_sym`)
is silently skipped for that symmetry type.

## Running one variant standalone (no config file)

Useful for a quick one-off test, or wiring into your own shell loop:

```bash
python Pipeline_Experiments/estimate_symmetry_variants.py \
    --renders-root ../data/renders --symmetry-type axis_sym \
    --experiment-id axis_v06_nomesh --variant expA \
    --sizes 224 --lightings flat

# EXP-F needs the real mesh:
python Pipeline_Experiments/estimate_symmetry_variants.py \
    --renders-root ../data/renders --objects-root ../data/objects \
    --symmetry-type plane_sym --experiment-id plane_v04_1_nomesh \
    --variant expF --max-planes 3 --sde-gate 0.02

# Then evaluate/compare exactly like any other no-mesh run:
python Mapping/evaluate.py --renders-root ../data/renders --objects-root ../data/objects \
    --symmetry-type axis_sym --experiment-id axis_v06_expA_nomesh --method triangulation
```

## EXP-LIT-1 (Candidates-then-Select) — needs a GPU

Unlike every triangulation variant above (pure post-hoc re-estimation from
already-collected points), this one makes *new* Molmo2 calls: it reads an
existing prompt run's points as "pass 1", marks numbered candidates around
each point on the real render, and asks Molmo2 to pick one ("pass 2"). Run it
on the server, exactly like `MolmoPointing/molmo_multiview_runner.py`:

```bash
CUDA_VISIBLE_DEVICES=0 python Pipeline_Experiments/candidates_then_select/molmo_candidates_runner.py \
    --renders-root ../data/renders --symmetry-type axis_sym \
    --base-experiment-id axis_v06_nomesh \
    --sizes 224 --lightings flat
```

It processes every `n_views` group already present in `--base-experiment-id`'s
JSON (no separate `--view-groups` flag — there is nothing new to select
views for, pass 1 already did that).

Its output (`molmo_multiview_axis_v06_nomesh_candidates.json`) is a normal
prompt-run JSON — feed it into `Mapping/estimate_symmetry_no_mesh.py` or
`estimate_symmetry_variants.py` with `--experiment-id
axis_v06_nomesh_candidates` like any other prompt, including through
`run_batch.py`'s `experiment_ids: auto` discovery.

## Center-bias diagnostic standalone

```bash
python Pipeline_Experiments/diagnostics/diagnose_center_bias.py \
    --renders-root ../data/renders --symmetry-type axis_sym --all \
    --out-detail ../results/diagnostics/axis_center_bias_detail.csv \
    --out-summary ../results/diagnostics/axis_center_bias_summary.csv
```

## Notes / known limitations

- `expB`/`expE` are axis-only and `expF` is plane-only, matching how they
  were scoped in `Experiments/sandbox_pipeline_server_extended.ipynb` — see
  the module docstrings for why (EXP-B/E's "point-then-direction"/iterative
  framing doesn't have a documented plane counterpart in the sandbox; EXP-F's
  gate is inherently a plane-consolidation concept).
- Axis variants process each `(size, lighting)` config independently and
  keep the one using the most views, instead of pooling raw rays across
  configs the way `Mapping/estimate_symmetry_no_mesh.py`'s axis branch does —
  see the comment in `estimate_symmetry_variants.py::process_object`. In
  practice this only matters when a single prompt run spans more than one
  size/lighting combination.
- `expF` needs `gpytoolbox` (same dependency `Mapping/evaluate.py --with-reference-metrics`
  already needs) and `--objects-root`.
