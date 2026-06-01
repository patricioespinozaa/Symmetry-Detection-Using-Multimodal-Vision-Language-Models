# Prompt Experiments

This document describes the workflow for testing prompt variants on a subset of objects before running the full dataset.

---

## Overview

The pipeline supports isolated experiments via three flags available in all four scripts:

| Flag | Effect |
|---|---|
| `--experiment-id EXP_ID` | All output files get `_EXP_ID` appended. Production files are never touched. |
| `--prompt-id PROMPT_ID` | Loads prompt texts from `prompts_registry.py` instead of the hardcoded defaults. |
| `--max-objects N` | Processes only the first N objects (same sorted order across all scripts). |

The flags propagate through the full pipeline:

```
molmo_multiview_<EXP_ID>.json
  → mapped_points_3d_<EXP_ID>.json
  → predicted_symmetry_<EXP_ID>.json
  → eval_s224_flat_<EXP_ID>_results.json
  → eval_s224_flat_<EXP_ID>_summary.csv
```

Production files (no suffix) are never modified.

---

## Adding a new prompt

Open `MolmoPointing/prompts_registry.py` and add a new entry to the `PROMPTS` dict:

```python
"axis_v01": {
    "symmetry_type": "axis_sym",
    "description":   "What this variant tests (one line)",
    "single":        """...""",   # prompt for n_views == 1
    "multi":         """...""",   # prompt for n_views  > 1
},
```

List all registered prompts:

```bash
python MolmoPointing/molmo_multiview_runner.py --list-prompts
# or
python MolmoPointing/prompts_registry.py
```

---

## Running one experiment (single prompt)

Replace `axis_v01` with your prompt ID and run all four pipeline steps:

```bash
EXP=axis_v01

# Step 1 — Molmo inference
CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \
    --renders-root ../data/renders \
    --objects-root ../data/objects \
    --symmetry-type axis_sym \
    --sizes 224 --lightings flat \
    --view-groups 1 6 14 26 \
    --max-objects 50 \
    --prompt-id  $EXP \
    --experiment-id $EXP \
    --prompt-mode auto

# Step 2 — Ray casting 2D→3D
python Mapping/map_to_3d.py \
    --renders-root ../data/renders \
    --objects-root ../data/objects \
    --symmetry-type axis_sym \
    --sizes 224 --lightings flat \
    --max-objects 50 \
    --experiment-id $EXP \
    --overwrite

# Step 3 — RANSAC + SVD
python Mapping/estimate_symmetry.py \
    --renders-root ../data/renders \
    --objects-root ../data/objects \
    --symmetry-type axis_sym \
    --sizes 224 --lightings flat \
    --max-objects 50 \
    --experiment-id $EXP \
    --overwrite

# Step 4 — Evaluation
python Mapping/evaluate.py \
    --renders-root ../data/renders \
    --objects-root ../data/objects \
    --symmetry-type axis_sym \
    --sizes 224 --lightings flat \
    --max-objects 50 \
    --experiment-id $EXP
```

---

## Running all experiments in a loop

```bash
# Axis experiments
for EXP in axis_v00 axis_v01 axis_v02 axis_v03 axis_v04; do
    echo "===== $EXP ====="

    CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \
        --renders-root ../data/renders --objects-root ../data/objects \
        --symmetry-type axis_sym --sizes 224 --lightings flat \
        --view-groups 1 6 14 26 --max-objects 50 \
        --prompt-id $EXP --experiment-id $EXP --prompt-mode auto

    python Mapping/map_to_3d.py \
        --renders-root ../data/renders --objects-root ../data/objects \
        --symmetry-type axis_sym --sizes 224 --lightings flat \
        --max-objects 50 --experiment-id $EXP --overwrite

    python Mapping/estimate_symmetry.py \
        --renders-root ../data/renders --objects-root ../data/objects \
        --symmetry-type axis_sym --sizes 224 --lightings flat \
        --max-objects 50 --experiment-id $EXP --overwrite

    python Mapping/evaluate.py \
        --renders-root ../data/renders --objects-root ../data/objects \
        --symmetry-type axis_sym --sizes 224 --lightings flat \
        --max-objects 50 --experiment-id $EXP
done

# Plane experiments (same structure, different prompt IDs)
for EXP in plane_v00 plane_v01 plane_v02; do
    echo "===== $EXP ====="

    CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \
        --renders-root ../data/renders --objects-root ../data/objects \
        --symmetry-type plane_sym --sizes 224 --lightings flat \
        --view-groups 1 6 14 26 --max-objects 50 \
        --prompt-id $EXP --experiment-id $EXP --prompt-mode auto

    python Mapping/map_to_3d.py \
        --renders-root ../data/renders --objects-root ../data/objects \
        --symmetry-type plane_sym --sizes 224 --lightings flat \
        --max-objects 50 --experiment-id $EXP --overwrite

    python Mapping/estimate_symmetry.py \
        --renders-root ../data/renders --objects-root ../data/objects \
        --symmetry-type plane_sym --sizes 224 --lightings flat \
        --max-objects 50 --experiment-id $EXP --overwrite

    python Mapping/evaluate.py \
        --renders-root ../data/renders --objects-root ../data/objects \
        --symmetry-type plane_sym --sizes 224 --lightings flat \
        --max-objects 50 --experiment-id $EXP
done
```

---

## Comparing results across experiments

Each experiment produces a CSV at:
```
../data/renders/axis_sym/eval_s224_flat_<EXP_ID>_summary.csv
```

To compare all axis experiments side by side:

```bash
python - <<'EOF'
import pandas as pd
from pathlib import Path

root    = Path("../data/renders/axis_sym")
csvs    = sorted(root.glob("eval_s224_flat_axis_v*_summary.csv"))
frames  = []

for csv in csvs:
    df = pd.read_csv(csv)
    df.insert(0, "experiment", csv.stem.replace("eval_s224_flat_", "").replace("_summary", ""))
    frames.append(df)

combined = pd.concat(frames, ignore_index=True)
cols = ["experiment", "n_views", "n_objects",
        "angular_error_mean", "angular_error_median",
        "auc_angular", "precision_5deg", "precision_10deg"]
print(combined[cols].to_string(index=False))
EOF
```

---

## File isolation guarantee

| EXP_ID | Files created | Production files |
|---|---|---|
| `axis_v01` | `molmo_multiview_axis_v01.json` | `molmo_multiview.json` — untouched |
| `axis_v01` | `mapped_points_3d_axis_v01.json` | `mapped_points_3d.json` — untouched |
| `axis_v01` | `predicted_symmetry_axis_v01.json` | `predicted_symmetry.json` — untouched |
| `axis_v01` | `eval_s224_flat_axis_v01_results.json` | `eval_s224_flat_results.json` — untouched |

Running any experiment script **without** `--experiment-id` always uses the production filenames and is unaffected by any experiment runs.

---

## Registered prompts

| ID | Type | Description |
|---|---|---|
| `axis_v00` | axis_sym | Baseline: current production prompts |
| `plane_v00` | plane_sym | Baseline: first plane symmetry prompts |

*(Add new rows here as you register prompts in `prompts_registry.py`.)*
