# Post-MolmoPointing Pipeline — Per Prompt/Flow Commands + Compare Results

Command reference for three things, in order:

1. Running the **post-Molmo** stage (`map_to_3d.py` → `estimate_symmetry.py` → `evaluate.py`)
   for **one prompt-flow experiment at a time**, independently of any other experiment.
2. Generating a **compare_results specific to that single prompt-flow** right after it finishes.
3. Once **every** planned experiment has been processed, generating **one consolidated
   comparison CSV across all of them**.

This assumes Step 2 of the main pipeline (Molmo2 inference,
`MolmoPointing/molmo_multiview_runner.py`) has already produced
`molmo_multiview_<EXP_ID>.json` for the experiment you're processing. See
[`MolmoPointing/Experiments.md`](MolmoPointing/Experiments.md) for the full registered-prompt
table and Molmo inference commands, and
[`EXPERIMENT_ROADMAP.md`](EXPERIMENT_ROADMAP.md) for Flow B/C, clustering, and
patch-size command sequences.

> `Mapping/compare_results.py` now supports `--experiment-id EXP_ID [EXP_ID ...]` to filter to
> one or more specific experiments even while other experiments' eval files already sit on
> disk in the same `<renders_root>/<symmetry_type>/`. The match is against the fully-parsed
> experiment id, **never a filename prefix** — `--experiment-id axis_v00` never accidentally
> pulls in `axis_v00_1`, and `axis_v05_1` never pulls in `axis_v05_1_flowB`. Output filenames
> (CSV + plots) are automatically tagged with the filtered experiment id(s) so they never
> overwrite a full "all experiments" comparison saved to the same `--save-dir`/`--csv-dir`.

---

## 0. Variables used throughout

```bash
RENDERS=../data/renders
OBJECTS=../data/objects
RESULTS=../results
```

---

## 1. Post-Molmo processing for ONE prompt-flow (independent)

```bash
run_post_molmo() {
    local SYM=$1     # axis_sym | plane_sym
    local EXP=$2     # e.g. axis_v01, axis_v05_1_flowB, axis_v05_1_flowC
    local MODE=$3    # independent | midpoint | all  (see table in Experiments.md; Flow C = all)

    echo ""
    echo "================================================================"
    echo "  post-molmo: $EXP  ($SYM, point-mode=$MODE)"
    echo "================================================================"

    python Mapping/map_to_3d.py \
        --renders-root $RENDERS --objects-root $OBJECTS \
        --symmetry-type $SYM --sizes 224 --lightings flat \
        --experiment-id $EXP --overwrite --yes

    python Mapping/estimate_symmetry.py \
        --renders-root $RENDERS --objects-root $OBJECTS \
        --symmetry-type $SYM --sizes 224 --lightings flat \
        --experiment-id $EXP --point-mode $MODE --overwrite

    for METHOD in svd ransac_svd svd_sde ransac_svd_sde; do
        python Mapping/evaluate.py \
            --renders-root $RENDERS --objects-root $OBJECTS \
            --symmetry-type $SYM --sizes 224 --lightings flat \
            --experiment-id $EXP --method $METHOD
    done
}
```

**Point-mode cheat sheet** (full detail in `MolmoPointing/Experiments.md`):

| Symmetry | `independent` | `midpoint` |
|---|---|---|
| axis_sym | `v00, v04, v05, v00_1, v02_1, v04_1, v05_1` | `v01, v02, v03, v01_1, v03_1` |
| plane_sym | `v00, v02, v04, v05, v00_1, v02_1, v04_1, v05_1` | `v01, v03, v01_1, v03_1` |

- **Flow A/B** — use the base prompt's mode from the table above, regardless of the
  `_flowB` suffix on `--experiment-id`.
- **Flow C** — always `--point-mode all`, regardless of base prompt (variable point count
  per image; `independent`/`midpoint` would silently drop points beyond obj_id 2).

Example — one Flow-A axis experiment:

```bash
run_post_molmo axis_sym axis_v01 midpoint
```

Example — a Flow-B / Flow-C experiment:

```bash
run_post_molmo axis_sym axis_v05_1_flowB independent   # Flow B keeps base prompt's mode
run_post_molmo axis_sym axis_v05_1_flowC all            # Flow C always uses "all"
```

---

## 2. Compare results specific to that ONE prompt-flow

`compare_results.py` accepts `--experiment-id` to filter to one (or a few) specific
experiments instead of aggregating everything found under `renders-root`:

```bash
compare_one_experiment() {
    local SYM=$1     # axis_sym | plane_sym
    local EXP=$2

    python Mapping/compare_results.py \
        --renders-root $RENDERS --symmetry-type $SYM \
        --sizes 224 --lightings flat --total-objects 850 \
        --experiment-id $EXP \
        --save-dir $RESULTS/$SYM/per_experiment/$EXP/plots \
        --csv-dir  $RESULTS/$SYM/per_experiment/$EXP
}
```

Example:

```bash
compare_one_experiment axis_sym axis_v01
```

Output for that single experiment lands in:
- `../results/axis_sym/per_experiment/axis_v01/plots/axis_sym_axis_v01_*.png`
- `../results/axis_sym/per_experiment/axis_v01/experiments_DD_MM_YYYY/axis_sym_axis_v01_comparison.csv`

`--experiment-id` also accepts multiple values — handy for a Flow A/B/C comparison of one
base prompt:

```bash
python Mapping/compare_results.py \
    --renders-root $RENDERS --symmetry-type axis_sym \
    --sizes 224 --lightings flat --total-objects 850 \
    --experiment-id axis_v05_1 axis_v05_1_flowB axis_v05_1_flowC \
    --save-dir $RESULTS/axis_sym/plots \
    --csv-dir  $RESULTS
```

---

## 2b. Export best/worst example objects for that ONE prompt-flow (optional)

A **different command**, not `compare_results.py`: `Mapping/export_viz_samples.py` picks the
N objects with the best and worst angular error for one experiment + method, copies their
pipeline JSONs into `<results_dir>/<experiment_id>/viz_samples/{good,bad}/`, and writes a
`README.md` with ready-to-run `visualize_rays.py` commands per object.

```bash
export_examples() {
    local SYM=$1     # axis_sym | plane_sym
    local EXP=$2
    local METHOD=${3:-svd}

    python Mapping/export_viz_samples.py \
        --renders-root $RENDERS --objects-root $OBJECTS \
        --symmetry-type $SYM --sizes 224 --lightings flat \
        --experiment-id $EXP --method $METHOD --n-views 14 \
        --n-samples 10 --results-dir $RESULTS
}
```

Example:

```bash
export_examples axis_sym axis_v01 svd
```

Output: `../results/axis_v01/viz_samples/good/`, `.../bad/`, `.../README.md`. Requires
`evaluate.py` to have already run for that `--method` (reads `eval_*_<EXP>_<method>_results.json`).

---

## 3. Full per-experiment cycle (process + compare + examples in one call)

```bash
run_experiment_and_compare() {
    local SYM=$1
    local EXP=$2
    local MODE=$3

    run_post_molmo "$SYM" "$EXP" "$MODE"
    compare_one_experiment "$SYM" "$EXP"
    export_examples "$SYM" "$EXP" svd
}

# Examples
run_experiment_and_compare axis_sym  axis_v00              independent
run_experiment_and_compare axis_sym  axis_v01               midpoint
run_experiment_and_compare axis_sym  axis_v05_1_flowB       independent
run_experiment_and_compare axis_sym  axis_v05_1_flowC       all
run_experiment_and_compare plane_sym plane_v00              independent
run_experiment_and_compare plane_sym plane_v01               midpoint
```

Run this once per finished Molmo experiment — it never touches any other experiment's
files (`map_to_3d.py`/`estimate_symmetry.py`/`evaluate.py` are isolated by `--experiment-id`,
`compare_one_experiment` only reads that one experiment's rows via `--experiment-id`, and
`export_examples` only reads that one experiment's eval results).

---

## 4. Once ALL experiments are finished — one consolidated CSV

Once every planned experiment (Flow A/B/C, all prompts, both symmetry types) has gone
through §1–§3, run `compare_results.py` **without** `--experiment-id` — it will pick up
every `eval_s224_flat_*_summary.csv` file that exists under `$RENDERS`:

```bash
python Mapping/compare_results.py \
    --renders-root $RENDERS --symmetry-type axis_sym \
    --sizes 224 --lightings flat --total-objects 850 \
    --save-dir $RESULTS/axis_sym/plots \
    --csv-dir  $RESULTS

python Mapping/compare_results.py \
    --renders-root $RENDERS --symmetry-type plane_sym \
    --sizes 224 --lightings flat --total-objects 850 \
    --save-dir $RESULTS/plane_sym/plots \
    --csv-dir  $RESULTS
```

This writes:
- `../results/experiments_DD_MM_YYYY/axis_sym_comparison.csv`
- `../results/experiments_DD_MM_YYYY/plane_sym_comparison.csv`

(`DD_MM_YYYY` = today's date, set by `compare_results.py` itself.) Each CSV already has an
`experiment` and `method` column covering every experiment found on disk — this is the
consolidated result table for that symmetry type.

### Optional — merge axis_sym + plane_sym into a single master CSV

`compare_results.py` requires `--symmetry-type` per call, so it always produces two files.
To combine them into one CSV (adding a `symmetry_type` column to tell the rows apart):

```bash
python - <<'PY'
import pandas as pd, glob, os

today_dir = sorted(glob.glob("../results/experiments_*"))[-1]  # most recent run
frames = []
for sym in ["axis_sym", "plane_sym"]:
    path = os.path.join(today_dir, f"{sym}_comparison.csv")
    df = pd.read_csv(path)
    df.insert(0, "symmetry_type", sym)
    frames.append(df)

combined = pd.concat(frames, ignore_index=True)
out_path = os.path.join(today_dir, "all_experiments_comparison.csv")
combined.to_csv(out_path, index=False)
print(f"Wrote {out_path} ({len(combined)} rows)")
PY
```

---

## 5. Related docs

| Doc | What it covers |
|---|---|
| [`MolmoPointing/Experiments.md`](MolmoPointing/Experiments.md) | Registered prompt table, `--point-mode` per prompt, Molmo inference commands (step before this doc) |
| [`EXPERIMENT_ROADMAP.md`](EXPERIMENT_ROADMAP.md) | Ordered checklist for Flow B/C, HDBSCAN clustering sweep, patch backprojection |
| [`Mapping/README.md`](Mapping/README.md) | Full flag reference for every `Mapping/*.py` script |
| [`MolmoPointing/README.md`](MolmoPointing/README.md) | Flow A/B/C semantics, prompt modes, resumability |
