# Experiment Roadmap

A single, ordered checklist for the experimental work that's still pending, with
copy-paste commands and a clear `--experiment-id` naming convention so every
result file is traceable back to exactly which experiment produced it.

This complements, but doesn't replace, the existing docs:
- **`MolmoPointing/Experiments.md`** — the full command reference for the 24
  Flow-A prompt variants (already completed) and a quick-reference for
  Flow B/C, clustering, and patch-size flags.
- **`Mapping/README.md`** / **`MolmoPointing/README.md`** — full flag reference
  for every script.
- **`metodologia_latex.tex` §3.7 / `PROJECT_CONTEXT_REPORT.md` §10** — the
  authoritative definition of what's done vs. pending.

---

## 0. What's already done

Per `metodologia_latex.tex` §3.7: **Flow A is fully completed** — both original
(v00–v05) and improved (v00_1–v05_1) prompts, for both symmetry types, across
`N ∈ {1, 6, 14, 26}` views, all 4 fitting methods, with clustering variants C1
(none) and C2 (greedy).

**Pending** (this roadmap): Flow B, Flow C, clustering variant C3 (HDBSCAN),
and patch-based backprojection (R2, `h ∈ {3, 5}`).

Before starting, confirm your current best Flow-A prompt per symmetry type
(the commands below use `axis_v05_1` / `plane_v04_1` as placeholders — these
are cited in `PROJECT_CONTEXT_REPORT.md` as the current leaders, but that
citation points at a `results/experiments_20_06_2026/` folder that isn't in
this repo, so double-check against your own latest CSVs before committing to
them as the base for Flow B/C).

---

## 1. Prerequisites

None of the commands in this roadmap pass `--max-objects`, so every script
processes all objects it finds under `<renders_root>/<symmetry_type>/` — this
roadmap targets the full 850-object dataset per symmetry type, not the
100-object subset used for the original prompt experiments. Confirm both are
fully rendered before starting (a partial `renders_root` just silently gives a
partial run, not an error):

```bash
find ../data/renders/axis_sym  -maxdepth 1 -mindepth 1 -type d | wc -l   # expect 850
find ../data/renders/plane_sym -maxdepth 1 -mindepth 1 -type d | wc -l   # expect 850

# HDBSCAN clustering needs scikit-learn (not required for Flow B/C themselves)
pip install scikit-learn

CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(torch.cuda.is_available())"
```

Pick your base prompts once and reuse them through the rest of this roadmap:

```bash
AXIS_BASE=axis_v05_1     # <- replace with your confirmed best Flow-A axis prompt
PLANE_BASE=plane_v04_1   # <- replace with your confirmed best Flow-A plane prompt
AXIS_MODE=independent    # point-mode used by $AXIS_BASE  (see Experiments.md table)
PLANE_MODE=independent   # point-mode used by $PLANE_BASE
```

---

## 2. Naming convention (read this before running anything)

Each stage's output filename gets an experiment-id suffix. **Clustering and
patch-size tags are appended automatically to that stage's own output** — but
the *next* stage must be given the full resulting string as its
`--experiment-id`, because each script reads its input using the
`--experiment-id` you pass it literally (it doesn't know about tags a
different script appended upstream).

Worked example — Flow B, axis, patch-size 3, HDBSCAN min_samples=3:

```bash
EXP=${AXIS_BASE}_flowB                     # your chosen tag for this flow run

# Stage 1: Molmo — writes molmo_multiview_axis_v05_1_flowB.json
molmo_multiview_runner.py --experiment-id $EXP --flow b ...

# Stage 2: map_to_3d — READS using $EXP, WRITES with "_p3" appended
#   -> reads  molmo_multiview_axis_v05_1_flowB.json
#   -> writes mapped_points_3d_axis_v05_1_flowB_p3.json
map_to_3d.py --experiment-id $EXP --patch-size 3 ...

# Stage 3: estimate_symmetry — must pass the GROWN string to read stage 2's output,
# then it appends its own "_hdbscan_ms3" on top
EXP2=${EXP}_p3                             # carry the patch tag forward
#   -> reads  mapped_points_3d_axis_v05_1_flowB_p3.json
#   -> writes predicted_symmetry_axis_v05_1_flowB_p3_hdbscan_ms3.json
estimate_symmetry.py --experiment-id $EXP2 --clustering-method hdbscan --hdbscan-min-samples 3 ...

# Stage 4: evaluate — must pass the FULLY GROWN string
EXP3=${EXP2}_hdbscan_ms3
#   -> reads eval_s224_flat_axis_v05_1_flowB_p3_hdbscan_ms3_<method>_results.json
evaluate.py --experiment-id $EXP3 --method ransac_svd_sde ...
```

If you only run one variant (e.g. Flow B with no clustering/patch changes), the
suffix never grows and `--experiment-id $EXP` is all you need at every stage.
`compare_results.py` doesn't need an `--experiment-id` at all — it aggregates
every `eval_*_summary.csv` under the symmetry-type folder automatically, and
the `experiment` column in its output tells you which is which.

---

## 3. Flow B — pointing con descripción

Run once per symmetry type, sweeping `N ∈ {1, 6, 14, 26}` and clustering C1/C2/C3.

### 3a. Molmo inference (GPU)

```bash
CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \
    --renders-root ../data/renders --symmetry-type axis_sym \
    --sizes 224 --lightings flat --view-groups 1 6 14 26 \
    --prompt-id $AXIS_BASE --flow b \
    --experiment-id ${AXIS_BASE}_flowB \
    --prompt-mode auto --yes

CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \
    --renders-root ../data/renders --symmetry-type plane_sym \
    --sizes 224 --lightings flat --view-groups 1 6 14 26 \
    --prompt-id $PLANE_BASE --flow b \
    --experiment-id ${PLANE_BASE}_flowB \
    --prompt-mode auto --yes
```

### 3b. Mapping + estimation + evaluation (CPU) — C1 (none) and C2 (greedy)

```bash
for SYM in axis plane; do
    BASE_VAR=${SYM^^}_BASE; BASE=${!BASE_VAR}
    MODE_VAR=${SYM^^}_MODE; MODE=${!MODE_VAR}
    EXP=${BASE}_flowB
    SYMTYPE=${SYM}_sym

    python Mapping/map_to_3d.py \
        --renders-root ../data/renders --objects-root ../data/objects \
        --symmetry-type $SYMTYPE --sizes 224 --lightings flat \
        --experiment-id $EXP --overwrite --yes

    for CLUSTER_ARGS in "" "--clustering-method greedy"; do
        python Mapping/estimate_symmetry.py \
            --renders-root ../data/renders --objects-root ../data/objects \
            --symmetry-type $SYMTYPE --sizes 224 --lightings flat \
            --experiment-id $EXP --point-mode $MODE --overwrite $CLUSTER_ARGS
    done

    for METHOD in svd ransac_svd svd_sde ransac_svd_sde; do
        python Mapping/evaluate.py \
            --renders-root ../data/renders --objects-root ../data/objects \
            --symmetry-type $SYMTYPE --sizes 224 --lightings flat \
            --experiment-id $EXP --method $METHOD
        python Mapping/evaluate.py \
            --renders-root ../data/renders --objects-root ../data/objects \
            --symmetry-type $SYMTYPE --sizes 224 --lightings flat \
            --experiment-id ${EXP}_cluster --method $METHOD
    done
done
```

### 3c. C3 — HDBSCAN sweep (reuses stage 3a/3b's mapped points, no new Molmo calls)

```bash
for SYM in axis plane; do
    BASE_VAR=${SYM^^}_BASE; BASE=${!BASE_VAR}
    MODE_VAR=${SYM^^}_MODE; MODE=${!MODE_VAR}
    EXP=${BASE}_flowB
    SYMTYPE=${SYM}_sym

    for MS in 2 3 5; do
        python Mapping/estimate_symmetry.py \
            --renders-root ../data/renders --objects-root ../data/objects \
            --symmetry-type $SYMTYPE --sizes 224 --lightings flat \
            --experiment-id $EXP --point-mode $MODE \
            --clustering-method hdbscan --hdbscan-min-samples $MS

        for METHOD in svd ransac_svd svd_sde ransac_svd_sde; do
            python Mapping/evaluate.py \
                --renders-root ../data/renders --objects-root ../data/objects \
                --symmetry-type $SYMTYPE --sizes 224 --lightings flat \
                --experiment-id ${EXP}_hdbscan_ms${MS} --method $METHOD
        done
    done
done
```

- [ ] Flow B — axis — Molmo inference
- [ ] Flow B — axis — map_to_3d + estimate (C1, C2) + evaluate
- [ ] Flow B — axis — HDBSCAN sweep (C3, ms=2/3/5) + evaluate
- [ ] Flow B — plane — Molmo inference
- [ ] Flow B — plane — map_to_3d + estimate (C1, C2) + evaluate
- [ ] Flow B — plane — HDBSCAN sweep (C3, ms=2/3/5) + evaluate

---

## 4. Flow C — descripción y pointing integrados

**Different point geometry from Flow B — read before copying §3's loops.**
Flow C's single-view pre-pass (`describe_and_point_axis.txt` /
`describe_and_point_plane.txt`) asks Molmo to *name* every distinctive
landmark it can find near the axis/plane, in plain text — it decides how
many itself and returns **no pixel coordinates at all** in this step.
Localization happens entirely in the second request: those labels replace
the `--prompt-id` base prompt for the N-view call, where the model is asked
to actually locate each named landmark, by name, in every view — see
`build_flow_c_prompts` in `molmo_multiview_runner.py`. `--prompt-id` is still
required as the fallback prompt for the (rare) case where the pre-pass
produces no parseable labels for a given object.

Because each image can now yield anywhere from 0 to ~10+ points instead of
exactly 2, `estimate_symmetry.py` needs **`--point-mode all`** (not
`independent`) — `independent` requires obj_id 1 AND 2 to both hit and
silently discards every other obj_id, which would drop most of Flow C's
points. Given the higher, noisier point count per object (Molmo can return
inconsistent points across views for the same named landmark, same failure
mode reported for multi-point prompting in the ZeroKey paper), the C3
(HDBSCAN) sweep in §4c is more likely to matter for Flow C than it was for
Flow A/B — don't skip it.

### 4a. Molmo inference (GPU)

```bash
CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \
    --renders-root ../data/renders --symmetry-type axis_sym \
    --sizes 224 --lightings flat --view-groups 1 6 14 26 \
    --prompt-id $AXIS_BASE --flow c \
    --experiment-id ${AXIS_BASE}_flowC \
    --prompt-mode auto --yes

CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \
    --renders-root ../data/renders --symmetry-type plane_sym \
    --sizes 224 --lightings flat --view-groups 1 6 14 26 \
    --prompt-id $PLANE_BASE --flow c \
    --experiment-id ${PLANE_BASE}_flowC \
    --prompt-mode auto --yes
```

### 4b. Mapping + estimation + evaluation (CPU) — C1 (none) and C2 (greedy)

```bash
for SYM in axis plane; do
    BASE_VAR=${SYM^^}_BASE; BASE=${!BASE_VAR}
    EXP=${BASE}_flowC
    SYMTYPE=${SYM}_sym

    python Mapping/map_to_3d.py \
        --renders-root ../data/renders --objects-root ../data/objects \
        --symmetry-type $SYMTYPE --sizes 224 --lightings flat \
        --experiment-id $EXP --overwrite --yes

    for CLUSTER_ARGS in "" "--clustering-method greedy"; do
        python Mapping/estimate_symmetry.py \
            --renders-root ../data/renders --objects-root ../data/objects \
            --symmetry-type $SYMTYPE --sizes 224 --lightings flat \
            --experiment-id $EXP --point-mode all --overwrite $CLUSTER_ARGS
    done

    for METHOD in svd ransac_svd svd_sde ransac_svd_sde; do
        python Mapping/evaluate.py \
            --renders-root ../data/renders --objects-root ../data/objects \
            --symmetry-type $SYMTYPE --sizes 224 --lightings flat \
            --experiment-id $EXP --method $METHOD
        python Mapping/evaluate.py \
            --renders-root ../data/renders --objects-root ../data/objects \
            --symmetry-type $SYMTYPE --sizes 224 --lightings flat \
            --experiment-id ${EXP}_cluster --method $METHOD
    done
done
```

### 4c. C3 — HDBSCAN sweep (reuses stage 4a/4b's mapped points, no new Molmo calls)

```bash
for SYM in axis plane; do
    BASE_VAR=${SYM^^}_BASE; BASE=${!BASE_VAR}
    EXP=${BASE}_flowC
    SYMTYPE=${SYM}_sym

    for MS in 2 3 5; do
        python Mapping/estimate_symmetry.py \
            --renders-root ../data/renders --objects-root ../data/objects \
            --symmetry-type $SYMTYPE --sizes 224 --lightings flat \
            --experiment-id $EXP --point-mode all \
            --clustering-method hdbscan --hdbscan-min-samples $MS

        for METHOD in svd ransac_svd svd_sde ransac_svd_sde; do
            python Mapping/evaluate.py \
                --renders-root ../data/renders --objects-root ../data/objects \
                --symmetry-type $SYMTYPE --sizes 224 --lightings flat \
                --experiment-id ${EXP}_hdbscan_ms${MS} --method $METHOD
        done
    done
done
```

- [ ] Flow C — axis — Molmo inference
- [ ] Flow C — axis — map_to_3d + estimate (C1, C2, `--point-mode all`) + evaluate
- [ ] Flow C — axis — HDBSCAN sweep (C3, ms=2/3/5) + evaluate
- [ ] Flow C — plane — Molmo inference
- [ ] Flow C — plane — map_to_3d + estimate (C1, C2, `--point-mode all`) + evaluate
- [ ] Flow C — plane — HDBSCAN sweep (C3, ms=2/3/5) + evaluate

---

## 5. Compare A vs B vs C, pick the winner

```bash
python Mapping/compare_results.py \
    --renders-root ../data/renders --symmetry-type axis_sym \
    --sizes 224 --lightings flat --total-objects 850 \
    --save-dir ../results/plots --csv-dir ../results

python Mapping/compare_results.py \
    --renders-root ../data/renders --symmetry-type plane_sym \
    --sizes 224 --lightings flat --total-objects 850 \
    --save-dir ../results/plots --csv-dir ../results
```

Open the generated `<symmetry>_comparison.csv` and, per symmetry type, identify
across **all** experiment rows so far (Flow A's completed sweep plus the new
Flow B/C rows above) the combination with the best `auc_angular` (axis) /
`auc_sde` (plane): which **flow** (A vs B vs C), which **n_views**, and which
**clustering method** (none / greedy / hdbscan + min_samples) wins. Write these
down — step 6 only tests patch backprojection on this single winning
combination per symmetry type (per the methodology's experimental design
table, patch backprojection is evaluated only against `best(A/B)`, i.e. only
if Flow A or Flow B wins — if Flow C wins for a symmetry type, run this step
against the best Flow A/B alternative for that type instead, and note the gap
to Flow C in your writeup).

- [ ] Axis — winning combo identified: flow=____, N=____, clustering=____
- [ ] Plane — winning combo identified: flow=____, N=____, clustering=____

---

## 6. Patch-based backprojection (R2), on the winning combo only

Substitute your step-5 winners for `$WIN_EXP` (the experiment-id of the winning
combination — e.g. `${AXIS_BASE}` for plain Flow A, or `${AXIS_BASE}_flowB` for
Flow B; append `_cluster` or `_hdbscan_ms{N}` if the winning clustering variant
wasn't "none") and `$WIN_N` (the single best n_views value).

```bash
for SYM in axis plane; do
    SYMTYPE=${SYM}_sym
    WIN_EXP=___your_winning_experiment_id___
    WIN_MODE=___independent_or_midpoint___

    for PATCH in 3 5; do
        python Mapping/map_to_3d.py \
            --renders-root ../data/renders --objects-root ../data/objects \
            --symmetry-type $SYMTYPE --sizes 224 --lightings flat \
            --experiment-id $WIN_EXP --patch-size $PATCH --overwrite --yes

        python Mapping/estimate_symmetry.py \
            --renders-root ../data/renders --objects-root ../data/objects \
            --symmetry-type $SYMTYPE --sizes 224 --lightings flat \
            --experiment-id ${WIN_EXP}_p${PATCH} --point-mode $WIN_MODE --overwrite

        for METHOD in svd ransac_svd svd_sde ransac_svd_sde; do
            python Mapping/evaluate.py \
                --renders-root ../data/renders --objects-root ../data/objects \
                --symmetry-type $SYMTYPE --sizes 224 --lightings flat \
                --experiment-id ${WIN_EXP}_p${PATCH} --method $METHOD
        done
    done
done
```

- [ ] Axis — patch-size 3 mapped + estimated + evaluated
- [ ] Axis — patch-size 5 mapped + estimated + evaluated
- [ ] Plane — patch-size 3 mapped + estimated + evaluated
- [ ] Plane — patch-size 5 mapped + estimated + evaluated

---

## 7. Final consolidation

```bash
python Mapping/compare_results.py --renders-root ../data/renders --symmetry-type axis_sym  --sizes 224 --lightings flat --total-objects 850 --save-dir ../results/plots --csv-dir ../results
python Mapping/compare_results.py --renders-root ../data/renders --symmetry-type plane_sym --sizes 224 --lightings flat --total-objects 850 --save-dir ../results/plots --csv-dir ../results
```

At this point `<symmetry>_comparison.csv` has every row needed for the thesis
comparison table: Flow A (done previously), Flow B, Flow C, the HDBSCAN sweep,
and the patch-backprojection variants on the overall winner. Optionally, export
visualization samples for the final winning combination:

```bash
python Mapping/export_viz_samples.py \
    --renders-root ../data/renders --objects-root ../data/objects \
    --symmetry-type axis_sym --experiment-id $WIN_EXP --method ransac_svd_sde \
    --results-dir ../results --sizes 224 --lightings flat --n-samples 10
```

- [ ] Final comparison CSVs regenerated for both symmetry types
- [ ] Viz samples exported for the final winning combination (optional)
