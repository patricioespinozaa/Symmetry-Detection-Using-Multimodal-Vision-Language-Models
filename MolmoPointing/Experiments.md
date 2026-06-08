# Prompt Experiments

This document describes the workflow for testing prompt variants on a subset of objects before running the full dataset.

---

## Registered prompts

Complete reference table — use this to decide which `--point-mode` to pass to `estimate_symmetry.py`.

### Axis symmetry (`--symmetry-type axis_sym`)

| ID | Strategy | What Molmo returns | `--point-mode` |
|---|---|---|---|
| `axis_v00` | Axis projection | Two far-apart points **directly on** the projected axis | `independent` |
| `axis_v01` | Bilateral pair | Left point + its **right mirror** equidistant from the axis | `midpoint` |
| `axis_v02` | Widest silhouette | Leftmost + rightmost surface points at the widest cross-section | `midpoint` |
| `axis_v03` | Structural feature pairs | Symmetric structural elements (handles, holes, ribs) on each side | `midpoint` |
| `axis_v04` | Polar extremes | Topmost + bottommost points where the axis **exits** the surface | `independent` |
| `axis_v05` | Axis centerline | One point on the centerline in the **upper half** + one in the **lower half** | `independent` |

### Plane symmetry (`--symmetry-type plane_sym`)

| ID | Strategy | What Molmo returns | `--point-mode` |
|---|---|---|---|
| `plane_v00` | Plane trace | Top + bottom points **on** the plane's surface intersection | `independent` |
| `plane_v01` | Bilateral pair | Left point + its **right mirror** equidistant from the plane | `midpoint` |
| `plane_v02` | Plane seam | Two points **on** the visible seam dividing the two mirror halves | `independent` |
| `plane_v03` | Structural feature pairs | Corresponding symmetric elements (legs, wheels, arms) across the plane | `midpoint` |
| `plane_v04` | Silhouette midpoints | Horizontal center of the object's width near the **top** + near the **bottom** | `independent` |
| `plane_v05` | Plane trace extremes | Two most **distant** points along the visible plane trace | `independent` |

### Why `--point-mode` matters

```
independent  →  each 3D point from ray casting enters SVD directly
midpoint     →  obj_id=1 and obj_id=2 per image are replaced by their
                3D midpoint before SVD
```

SVD for **axis detection** needs the cloud to have high variance *along* the axis. Bilateral pair prompts return points on *opposite sides* of the axis — their midpoints lie on the axis, but the individual points do not. `--point-mode midpoint` corrects this by computing the midpoint in 3D before SVD.

---

## Pipeline flags reference

| Flag | Script(s) | Effect |
|---|---|---|
| `--experiment-id EXP_ID` | all 4 | Output files get `_EXP_ID` suffix. Production files never touched. |
| `--prompt-id PROMPT_ID` | runner only | Loads prompt text from `prompts_registry.py`. |
| `--point-mode MODE` | estimate_symmetry | `independent` or `midpoint` (see table above). |
| `--method METHOD` | evaluate | `svd`, `ransac_svd`, `svd_sde`, or `ransac_svd_sde` (required). |
| `--max-objects N` | all 4 | Processes only the first N objects (same sorted order). |
| `--yes` / `-y` | runner, map_to_3d | Skips interactive confirmation. Required for automated loops. |

File chain per experiment (evaluate genera 4 pares, uno por método):
```
molmo_multiview_<EXP_ID>.json
  → mapped_points_3d_<EXP_ID>.json
  → predicted_symmetry_<EXP_ID>.json  (contiene los 4 métodos)
  → eval_s224_flat_<EXP_ID>_svd_results.json
  → eval_s224_flat_<EXP_ID>_svd_summary.csv
  → eval_s224_flat_<EXP_ID>_ransac_svd_results.json
  → eval_s224_flat_<EXP_ID>_ransac_svd_summary.csv
  → eval_s224_flat_<EXP_ID>_svd_sde_results.json
  → eval_s224_flat_<EXP_ID>_svd_sde_summary.csv
  → eval_s224_flat_<EXP_ID>_ransac_svd_sde_results.json
  → eval_s224_flat_<EXP_ID>_ransac_svd_sde_summary.csv
```

---

## Run ALL axis experiments (two-phase workflow)

### Phase 1 — Molmo (GPU)

```bash
RENDERS=../data/renders

# Todos los prompts axiales — solo inferencia Molmo
for EXP in axis_v00 axis_v01 axis_v02 axis_v03 axis_v04 axis_v05; do
    echo "===== Molmo: $EXP ====="
    CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \
        --renders-root $RENDERS --symmetry-type axis_sym \
        --sizes 224 --lightings flat --view-groups 1 6 14 26 \
        --max-objects 50 --prompt-id $EXP --experiment-id $EXP \
        --prompt-mode auto --yes
done
```

### Fase 2 — Mapeo + estimación + evaluación (CPU) · ejecutar después

```bash
RENDERS=../data/renders
OBJECTS=../data/objects

run_axis_mapping() {
    local EXP=$1
    local MODE=$2
    echo ""
    echo "============================================="
    echo "  axis mapping: $EXP  (point-mode=$MODE)"
    echo "============================================="

    python Mapping/map_to_3d.py \
        --renders-root $RENDERS --objects-root $OBJECTS \
        --symmetry-type axis_sym --sizes 224 --lightings flat \
        --max-objects 50 --experiment-id $EXP --overwrite --yes

    python Mapping/estimate_symmetry.py \
        --renders-root $RENDERS --objects-root $OBJECTS \
        --symmetry-type axis_sym --sizes 224 --lightings flat \
        --max-objects 50 --experiment-id $EXP --point-mode $MODE --overwrite

    for METHOD in svd ransac_svd svd_sde ransac_svd_sde; do
        python Mapping/evaluate.py \
            --renders-root $RENDERS --objects-root $OBJECTS \
            --symmetry-type axis_sym --sizes 224 --lightings flat \
            --max-objects 50 --experiment-id $EXP --method $METHOD
    done
}

# Points directly on the axis → independent
for EXP in axis_v00 axis_v04 axis_v05; do run_axis_mapping $EXP independent; done

# Bilateral symmetric pairs → midpoint
for EXP in axis_v01 axis_v02 axis_v03; do run_axis_mapping $EXP midpoint; done
```

### Todo en un solo bloque (sin separar fases)

```bash
RENDERS=../data/renders
OBJECTS=../data/objects

run_axis_exp() {
    local EXP=$1
    local MODE=$2
    echo ""
    echo "============================================="
    echo "  axis experiment: $EXP  (point-mode=$MODE)"
    echo "============================================="

    CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \
        --renders-root $RENDERS --symmetry-type axis_sym \
        --sizes 224 --lightings flat --view-groups 1 6 14 26 \
        --max-objects 50 --prompt-id $EXP --experiment-id $EXP \
        --prompt-mode auto --yes

    python Mapping/map_to_3d.py \
        --renders-root $RENDERS --objects-root $OBJECTS \
        --symmetry-type axis_sym --sizes 224 --lightings flat \
        --max-objects 50 --experiment-id $EXP --overwrite --yes

    python Mapping/estimate_symmetry.py \
        --renders-root $RENDERS --objects-root $OBJECTS \
        --symmetry-type axis_sym --sizes 224 --lightings flat \
        --max-objects 50 --experiment-id $EXP --point-mode $MODE --overwrite

    for METHOD in svd ransac_svd svd_sde ransac_svd_sde; do
        python Mapping/evaluate.py \
            --renders-root $RENDERS --objects-root $OBJECTS \
            --symmetry-type axis_sym --sizes 224 --lightings flat \
            --max-objects 50 --experiment-id $EXP --method $METHOD
    done
}

for EXP in axis_v00 axis_v04 axis_v05; do run_axis_exp $EXP independent; done
for EXP in axis_v01 axis_v02 axis_v03; do run_axis_exp $EXP midpoint; done
```

---

## Run ALL plane experiments (two-phase workflow)

### Fase 1 — Molmo (GPU)

```bash
RENDERS=../data/renders

for EXP in plane_v00 plane_v01 plane_v02 plane_v03 plane_v04 plane_v05; do
    echo "===== Molmo: $EXP ====="
    CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \
        --renders-root $RENDERS --symmetry-type plane_sym \
        --sizes 224 --lightings flat --view-groups 1 6 14 26 \
        --max-objects 50 --prompt-id $EXP --experiment-id $EXP \
        --prompt-mode auto --yes
done
```

### Fase 2 — Mapeo + estimación + evaluación (CPU)

```bash
RENDERS=../data/renders
OBJECTS=../data/objects

run_plane_mapping() {
    local EXP=$1
    local MODE=$2
    echo ""
    echo "=============================================="
    echo "  plane mapping: $EXP  (point-mode=$MODE)"
    echo "=============================================="

    python Mapping/map_to_3d.py \
        --renders-root $RENDERS --objects-root $OBJECTS \
        --symmetry-type plane_sym --sizes 224 --lightings flat \
        --max-objects 50 --experiment-id $EXP --overwrite --yes

    python Mapping/estimate_symmetry.py \
        --renders-root $RENDERS --objects-root $OBJECTS \
        --symmetry-type plane_sym --sizes 224 --lightings flat \
        --max-objects 50 --experiment-id $EXP --point-mode $MODE --overwrite

    for METHOD in svd ransac_svd svd_sde ransac_svd_sde; do
        python Mapping/evaluate.py \
            --renders-root $RENDERS --objects-root $OBJECTS \
            --symmetry-type plane_sym --sizes 224 --lightings flat \
            --max-objects 50 --experiment-id $EXP --method $METHOD
    done
}

# Points directly on the plane → independent
for EXP in plane_v00 plane_v02 plane_v04 plane_v05; do run_plane_mapping $EXP independent; done

# Bilateral symmetric pairs → midpoint
for EXP in plane_v01 plane_v03; do run_plane_mapping $EXP midpoint; done
```

---

## Running one experiment manually

```bash
EXP=axis_v01
MODE=midpoint   # see table above for correct value per prompt

CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \
    --renders-root ../data/renders --symmetry-type axis_sym \
    --sizes 224 --lightings flat --view-groups 1 6 14 26 \
    --max-objects 50 --prompt-id $EXP --experiment-id $EXP \
    --prompt-mode auto --yes

python Mapping/map_to_3d.py \
    --renders-root ../data/renders --objects-root ../data/objects \
    --symmetry-type axis_sym --sizes 224 --lightings flat \
    --max-objects 50 --experiment-id $EXP --overwrite --yes

python Mapping/estimate_symmetry.py \
    --renders-root ../data/renders --objects-root ../data/objects \
    --symmetry-type axis_sym --sizes 224 --lightings flat \
    --max-objects 50 --experiment-id $EXP --point-mode $MODE --overwrite

for METHOD in svd ransac_svd svd_sde ransac_svd_sde; do
    python Mapping/evaluate.py \
        --renders-root ../data/renders --objects-root ../data/objects \
        --symmetry-type axis_sym --sizes 224 --lightings flat \
        --max-objects 50 --experiment-id $EXP --method $METHOD
done
```

---

## Comparing results across experiments

Cada experimento produce 4 CSVs (uno por método):
```
../data/renders/axis_sym/eval_s224_flat_<EXP_ID>_svd_summary.csv
../data/renders/axis_sym/eval_s224_flat_<EXP_ID>_ransac_svd_summary.csv
../data/renders/axis_sym/eval_s224_flat_<EXP_ID>_svd_sde_summary.csv
../data/renders/axis_sym/eval_s224_flat_<EXP_ID>_ransac_svd_sde_summary.csv
```

Comparar todos los experimentos y métodos a la vez:

```bash
clear
```

Same for plane (replace `axis_sym` and `axis_v` with `plane_sym` and `plane_v`).

---

## Adding a new prompt

1. Create two `.txt` files in the appropriate folder:
   ```
   MolmoPointing/prompts/axis/v06_single.txt   ← prompt for n_views == 1
   MolmoPointing/prompts/axis/v06_multi.txt    ← prompt for n_views  > 1
   ```
2. Optionally add a description in `DESCRIPTIONS` in `prompts_registry.py`.
3. Add the new prompt to the registered prompts table above with its `--point-mode`.
4. Add it to the appropriate loop in the run-all commands above.

Verify it was detected:
```bash
python MolmoPointing/molmo_multiview_runner.py --list-prompts
```

---

## File isolation guarantee

| EXP_ID | Files created | Production files |
|---|---|---|
| `axis_v01` | `molmo_multiview_axis_v01.json` | `molmo_multiview.json` — untouched |
| `axis_v01` | `mapped_points_3d_axis_v01.json` | `mapped_points_3d.json` — untouched |
| `axis_v01` | `predicted_symmetry_axis_v01.json` | `predicted_symmetry.json` — untouched |
| `axis_v01` | `eval_s224_flat_axis_v01_svd_results.json` | `eval_s224_flat_svd_results.json` — untouched |
| `axis_v01` | `eval_s224_flat_axis_v01_ransac_svd_results.json` | `eval_s224_flat_ransac_svd_results.json` — untouched |
| `axis_v01` | `eval_s224_flat_axis_v01_svd_sde_results.json` | `eval_s224_flat_svd_sde_results.json` — untouched |
| `axis_v01` | `eval_s224_flat_axis_v01_ransac_svd_sde_results.json` | `eval_s224_flat_ransac_svd_sde_results.json` — untouched |

Running any script **without** `--experiment-id` always uses production filenames, unaffected by any experiment.
