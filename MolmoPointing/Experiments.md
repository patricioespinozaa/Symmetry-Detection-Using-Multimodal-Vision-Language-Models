# Prompt Experiments

This document describes the workflow for testing prompt variants across the full curated dataset (850 objects per symmetry type). None of the commands below pass `--max-objects`, so every script processes all objects it finds under `<renders_root>/<symmetry_type>/` — make sure that folder actually has all 850 rendered before running these loops (a partial `renders_root` just means a partial run, not an error).

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
| `axis_v00_1` | Axis projection v1 | Upper/lower split enforced + centerline definition + no silhouette edges | `independent` |
| `axis_v01_1` | Bilateral pair v1 | Bilateral pair + midpoint-on-centerline verification + height diversity across views | `midpoint` |
| `axis_v02_1` | Widest cross-section v1 | Centerline midpoint at widest height + centerline midpoint at top/bottom (**changed: independent**) | `independent` |
| `axis_v03_1` | Structural pairs v1 | Structural pair + midpoint verification + widest cross-section fallback | `midpoint` |
| `axis_v04_1` | Polar extremes v1 | Polar extremes + horizontal center verification + flat-surface fallback | `independent` |
| `axis_v05_1` | Axis centerline v1 | Explicit step-0 global axis identification + cross-view consistency check | `independent` |

### Plane symmetry (`--symmetry-type plane_sym`)

| ID | Strategy | What Molmo returns | `--point-mode` |
|---|---|---|---|
| `plane_v00` | Plane trace | Top + bottom points **on** the plane's surface intersection | `independent` |
| `plane_v01` | Bilateral pair | Left point + its **right mirror** equidistant from the plane | `midpoint` |
| `plane_v02` | Plane seam | Two points **on** the visible seam dividing the two mirror halves | `independent` |
| `plane_v03` | Structural feature pairs | Corresponding symmetric elements (legs, wheels, arms) across the plane | `midpoint` |
| `plane_v04` | Silhouette midpoints | Horizontal center of the object's width near the **top** + near the **bottom** | `independent` |
| `plane_v05` | Plane trace extremes | Two most **distant** points along the visible plane trace | `independent` |
| `plane_v00_1` | Plane trace v1 | Top/bottom split enforced + horizontal center guidance | `independent` |
| `plane_v01_1` | Bilateral pair v1 | Bilateral pair + midpoint-on-trace verification + height diversity across views | `midpoint` |
| `plane_v02_1` | Plane seam v1 | Step-0 global plane ID + seam consistency across views + top/bottom enforced | `independent` |
| `plane_v03_1` | Structural pairs v1 | Structural pair + midpoint verification + widest cross-section fallback | `midpoint` |
| `plane_v04_1` | Silhouette midpoints v1 | Step-0 global plane ID + explicit midpoint formula + cross-view consistency | `independent` |
| `plane_v05_1` | Plane trace v1 | Topmost/bottommost on trace (vertical separation explicit, not diagonal) | `independent` |

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
| `--max-objects N` | all 4 | Processes only the first N objects (same sorted order). **Not used in this document** — omitting it processes every object found in `renders_root` (all 850). Useful for a quick sanity check on a handful of objects before committing to a full run. |
| `--yes` / `-y` | runner, map_to_3d | Skips interactive confirmation. Required for automated loops. |
| `--flow {a,b,c}` | runner only | See "Flows B/C" below. Default `a` reproduces this document's experiments exactly. |
| `--patch-size {1,3,5}` | map_to_3d | See "Backprojection variants" below. |
| `--clustering-method {none,greedy,hdbscan}` | estimate_symmetry | See "Clustering variants" below. |

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

## Flows B/C, clustering variants, backprojection variants

Everything below this point (v00–v05, v00_1–v05_1) exercises **Flow A** (direct
pointing) against the greedy-clustering / exact-backprojection defaults. The
methodology defines two more experimental axes, layered on top of the best
Flow-A prompt rather than replacing these loops.

**For the full ordered checklist of what's still pending (Flow B, Flow C,
HDBSCAN sweep, patch backprojection) with a worked `--experiment-id` chaining
example, see [`EXPERIMENT_ROADMAP.md`](../EXPERIMENT_ROADMAP.md) at the repo
root.** The quick-reference below is just the raw flag syntax.

**Flows B/C** — run the best-performing Flow-A prompt (`--prompt-id`) under
`--flow b` or `--flow c` instead of the default `--flow a`:
```bash
CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \
    --renders-root ../data/renders --symmetry-type axis_sym \
    --sizes 224 --lightings flat --view-groups 1 6 14 26 \
    --prompt-id axis_v05_1 --flow b --experiment-id axis_v05_1_flowB \
    --prompt-mode auto --yes
```
Then run `map_to_3d.py` / `estimate_symmetry.py` / `evaluate.py` exactly as in the
loops below, substituting `axis_v05_1_flowB` for `--experiment-id`. See
`MolmoPointing/README.md` § Flows.

**Clustering variants** — sweep `--clustering-method hdbscan --hdbscan-min-samples {2,3,5}`
in `estimate_symmetry.py` against an existing `mapped_points_3d[_EXP].json` (no
re-run of Molmo/map_to_3d needed):
```bash
for MS in 2 3 5; do
    python Mapping/estimate_symmetry.py \
        --renders-root ../data/renders --objects-root ../data/objects \
        --symmetry-type axis_sym --sizes 224 --lightings flat \
        --experiment-id axis_v05_1 --clustering-method hdbscan --hdbscan-min-samples $MS
done
```
See `Mapping/README.md` § 2.

**Backprojection variants** — sweep `--patch-size {3,5}` in `map_to_3d.py` for the
best Flow-A/B prompt and best `N` (re-run `estimate_symmetry.py`/`evaluate.py`
afterward, reading the `_p3`/`_p5`-tagged `mapped_points_3d` file). See
`Mapping/README.md` § 1.

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
        --prompt-id $EXP --experiment-id $EXP \
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
        --experiment-id $EXP --overwrite --yes

    python Mapping/estimate_symmetry.py \
        --renders-root $RENDERS --objects-root $OBJECTS \
        --symmetry-type axis_sym --sizes 224 --lightings flat \
        --experiment-id $EXP --point-mode $MODE --overwrite

    for METHOD in svd ransac_svd svd_sde ransac_svd_sde; do
        python Mapping/evaluate.py \
            --renders-root $RENDERS --objects-root $OBJECTS \
            --symmetry-type axis_sym --sizes 224 --lightings flat \
            --experiment-id $EXP --method $METHOD
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
        --prompt-id $EXP --experiment-id $EXP \
        --prompt-mode auto --yes

    python Mapping/map_to_3d.py \
        --renders-root $RENDERS --objects-root $OBJECTS \
        --symmetry-type axis_sym --sizes 224 --lightings flat \
        --experiment-id $EXP --overwrite --yes

    python Mapping/estimate_symmetry.py \
        --renders-root $RENDERS --objects-root $OBJECTS \
        --symmetry-type axis_sym --sizes 224 --lightings flat \
        --experiment-id $EXP --point-mode $MODE --overwrite

    for METHOD in svd ransac_svd svd_sde ransac_svd_sde; do
        python Mapping/evaluate.py \
            --renders-root $RENDERS --objects-root $OBJECTS \
            --symmetry-type axis_sym --sizes 224 --lightings flat \
            --experiment-id $EXP --method $METHOD
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
        --prompt-id $EXP --experiment-id $EXP \
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
        --experiment-id $EXP --overwrite --yes

    python Mapping/estimate_symmetry.py \
        --renders-root $RENDERS --objects-root $OBJECTS \
        --symmetry-type plane_sym --sizes 224 --lightings flat \
        --experiment-id $EXP --point-mode $MODE --overwrite

    for METHOD in svd ransac_svd svd_sde ransac_svd_sde; do
        python Mapping/evaluate.py \
            --renders-root $RENDERS --objects-root $OBJECTS \
            --symmetry-type plane_sym --sizes 224 --lightings flat \
            --experiment-id $EXP --method $METHOD
    done
}

# Points directly on the plane → independent
for EXP in plane_v00 plane_v02 plane_v04 plane_v05; do run_plane_mapping $EXP independent; done

# Bilateral symmetric pairs → midpoint
for EXP in plane_v01 plane_v03; do run_plane_mapping $EXP midpoint; done
```

---

## Run ALL v1 experiments (improved prompts batch)

> Nota: `axis_v02_1` usa `independent` (cambio respecto a `axis_v02` que usaba `midpoint`), porque v02_1 ya devuelve puntos directamente en el centerline.

### Fase 1 — Molmo (GPU)

```bash
RENDERS=../data/renders

# Axis v1
for EXP in axis_v00_1 axis_v01_1 axis_v02_1 axis_v03_1 axis_v04_1 axis_v05_1; do
    echo "===== Molmo: $EXP ====="
    CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \
        --renders-root $RENDERS --symmetry-type axis_sym \
        --sizes 224 --lightings flat --view-groups 1 6 14 26 \
        --prompt-id $EXP --experiment-id $EXP \
        --prompt-mode auto --yes
done

# Plane v1
for EXP in plane_v00_1 plane_v01_1 plane_v02_1 plane_v03_1 plane_v04_1 plane_v05_1; do
    echo "===== Molmo: $EXP ====="
    CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \
        --renders-root $RENDERS --symmetry-type plane_sym \
        --sizes 224 --lightings flat --view-groups 1 6 14 26 \
        --prompt-id $EXP --experiment-id $EXP \
        --prompt-mode auto --yes
done
```

### Fase 2 — Mapeo + estimación + evaluación (CPU)

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
        --experiment-id $EXP --overwrite --yes

    python Mapping/estimate_symmetry.py \
        --renders-root $RENDERS --objects-root $OBJECTS \
        --symmetry-type axis_sym --sizes 224 --lightings flat \
        --experiment-id $EXP --point-mode $MODE --overwrite

    for METHOD in svd ransac_svd svd_sde ransac_svd_sde; do
        python Mapping/evaluate.py \
            --renders-root $RENDERS --objects-root $OBJECTS \
            --symmetry-type axis_sym --sizes 224 --lightings flat \
            --experiment-id $EXP --method $METHOD
    done
}

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
        --experiment-id $EXP --overwrite --yes

    python Mapping/estimate_symmetry.py \
        --renders-root $RENDERS --objects-root $OBJECTS \
        --symmetry-type plane_sym --sizes 224 --lightings flat \
        --experiment-id $EXP --point-mode $MODE --overwrite

    for METHOD in svd ransac_svd svd_sde ransac_svd_sde; do
        python Mapping/evaluate.py \
            --renders-root $RENDERS --objects-root $OBJECTS \
            --symmetry-type plane_sym --sizes 224 --lightings flat \
            --experiment-id $EXP --method $METHOD
    done
}

# Axis v1 — point modes
for EXP in axis_v00_1 axis_v02_1 axis_v04_1 axis_v05_1; do run_axis_mapping $EXP independent; done
for EXP in axis_v01_1 axis_v03_1;                        do run_axis_mapping $EXP midpoint;   done

# Plane v1 — point modes
for EXP in plane_v00_1 plane_v02_1 plane_v04_1 plane_v05_1; do run_plane_mapping $EXP independent; done
for EXP in plane_v01_1 plane_v03_1;                          do run_plane_mapping $EXP midpoint;   done
```

### Guardar resultados en results/

```bash
RENDERS=../data/renders
OBJECTS=../data/objects
RESULTS=../results

python Mapping/compare_results.py \
    --renders-root $RENDERS --symmetry-type axis_sym \
    --sizes 224 --lightings flat --total-objects 850 \
    --save-dir $RESULTS/axis_sym/plots --csv-dir $RESULTS

python Mapping/compare_results.py \
    --renders-root $RENDERS --symmetry-type plane_sym \
    --sizes 224 --lightings flat --total-objects 850 \
    --save-dir $RESULTS/plane_sym/plots --csv-dir $RESULTS

for EXP in axis_v00_1 axis_v01_1 axis_v02_1 axis_v03_1 axis_v04_1 axis_v05_1; do
    python Mapping/export_viz_samples.py \
        --renders-root $RENDERS --objects-root $OBJECTS \
        --symmetry-type axis_sym --sizes 224 --lightings flat \
        --experiment-id $EXP --method svd --n-views 14 \
        --n-samples 10 --results-dir $RESULTS
done

for EXP in plane_v00_1 plane_v01_1 plane_v02_1 plane_v03_1 plane_v04_1 plane_v05_1; do
    python Mapping/export_viz_samples.py \
        --renders-root $RENDERS --objects-root $OBJECTS \
        --symmetry-type plane_sym --sizes 224 --lightings flat \
        --experiment-id $EXP --method svd --n-views 14 \
        --n-samples 10 --results-dir $RESULTS
done
```

### Limpiar experimentos v1 si necesitas re-ejecutar

```bash
RENDERS=../data/renders

# Axis v1
for EXP in axis_v00_1 axis_v01_1 axis_v02_1 axis_v03_1 axis_v04_1 axis_v05_1; do
    find $RENDERS/axis_sym -name "*_${EXP}.json" -delete
    rm -f $RENDERS/axis_sym/eval_*_${EXP}_*.{json,csv}
    echo "Limpiado: $EXP"
done

# Plane v1
for EXP in plane_v00_1 plane_v01_1 plane_v02_1 plane_v03_1 plane_v04_1 plane_v05_1; do
    find $RENDERS/plane_sym -name "*_${EXP}.json" -delete
    rm -f $RENDERS/plane_sym/eval_*_${EXP}_*.{json,csv}
    echo "Limpiado: $EXP"
done
```

---

## Running one experiment manually

```bash
EXP=axis_v01
MODE=midpoint   # see table above for correct value per prompt

CUDA_VISIBLE_DEVICES=0 python MolmoPointing/molmo_multiview_runner.py \
    --renders-root ../data/renders --symmetry-type axis_sym \
    --sizes 224 --lightings flat --view-groups 1 6 14 26 \
    --prompt-id $EXP --experiment-id $EXP \
    --prompt-mode auto --yes

python Mapping/map_to_3d.py \
    --renders-root ../data/renders --objects-root ../data/objects \
    --symmetry-type axis_sym --sizes 224 --lightings flat \
    --experiment-id $EXP --overwrite --yes

python Mapping/estimate_symmetry.py \
    --renders-root ../data/renders --objects-root ../data/objects \
    --symmetry-type axis_sym --sizes 224 --lightings flat \
    --experiment-id $EXP --point-mode $MODE --overwrite

for METHOD in svd ransac_svd svd_sde ransac_svd_sde; do
    python Mapping/evaluate.py \
        --renders-root ../data/renders --objects-root ../data/objects \
        --symmetry-type axis_sym --sizes 224 --lightings flat \
        --experiment-id $EXP --method $METHOD
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
# Axis
python Mapping/compare_results.py \
    --renders-root ../data/renders \
    --symmetry-type axis_sym

# Plane
python Mapping/compare_results.py \
    --renders-root ../data/renders \
    --symmetry-type plane_sym
```

Guardar gráficos con `--save-dir` y CSV con `--csv-dir`:

```bash
python Mapping/compare_results.py \
    --renders-root ../data/renders \
    --symmetry-type axis_sym \
    --save-dir ./results/plots \
    --csv-dir ./results
```

El CSV se guarda en `../results/experiments_DD_MM_YYYY/<symmetry_type>_comparison.csv`.

---

## Cleaning past experiments

El runner omite objetos cuya clave ya existe en `molmo_multiview_<EXP_ID>.json`:

```
(Existing JSON keys are skipped automatically)
GPU 0  ████ 850/850 [00:02<00:00]   ← 2 segundos = no corrió nada
```

Si ves esto, los archivos del experimento anterior siguen en disco. Bórralos antes de re-ejecutar (por ejemplo, tras corregir un bug o cambiar el prompt).

### Verificar qué archivos existen para un experimento

```bash
EXP=axis_v00
RENDERS=../data/renders

# Archivos por objeto (molmo, mapped, predicted)
find $RENDERS/axis_sym -name "*_${EXP}.json" | head -20
find $RENDERS/axis_sym -name "*_${EXP}.json" | wc -l

# Archivos de evaluación en la raíz del tipo
ls $RENDERS/axis_sym/eval_*_${EXP}_*.{json,csv} 2>/dev/null
```

### Limpiar un experimento concreto

```bash
EXP=axis_v00
RENDERS=../data/renders

# Archivos por objeto: molmo_multiview, mapped_points_3d, predicted_symmetry
find $RENDERS/axis_sym -name "*_${EXP}.json" -delete

# Archivos de evaluación en la raíz
rm -f $RENDERS/axis_sym/eval_*_${EXP}_*.json
rm -f $RENDERS/axis_sym/eval_*_${EXP}_*.csv

echo "Limpiado: $EXP"
```

### Limpiar todos los experimentos de un tipo de simetría

```bash
RENDERS=../data/renders
SYM=axis_sym   # o plane_sym

# Todos los archivos con sufijo _vXX
find $RENDERS/$SYM -name "*_v[0-9][0-9].json" -delete
rm -f $RENDERS/$SYM/eval_*_v[0-9][0-9]_*.json
rm -f $RENDERS/$SYM/eval_*_v[0-9][0-9]_*.csv

echo "Limpiados todos los experimentos de $SYM"
```

### No borrar

| Archivo | Motivo |
|---|---|
| `metadata_all.json` | Índice de renders generado por ImagesGenerator — costoso de regenerar |
| `manifest.json` | Lista de objetos curados — se usa en todos los scripts |
| `molmo_multiview.json` (sin sufijo) | Archivos de producción — aislados de los experimentos |
| `mapped_points_3d.json` (sin sufijo) | Ídem |
| `predicted_symmetry.json` (sin sufijo) | Ídem |

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

### Guardar resultados en results/ (prompts originales v00–v05)

```bash
RENDERS=../data/renders
OBJECTS=../data/objects
RESULTS=../results

# ── Gráficos de comparación (todos los experimentos juntos) ───────────────────

python Mapping/compare_results.py \
    --renders-root $RENDERS --symmetry-type axis_sym \
    --sizes 224 --lightings flat --total-objects 850 \
    --save-dir $RESULTS/axis_sym/plots \
    --csv-dir $RESULTS

python Mapping/compare_results.py \
    --renders-root $RENDERS --symmetry-type plane_sym \
    --sizes 224 --lightings flat --total-objects 850 \
    --save-dir $RESULTS/plane_sym/plots \
    --csv-dir $RESULTS

# ── JSONs de objetos buenos/malos por experimento ─────────────────────────────

for EXP in axis_v00 axis_v01 axis_v02 axis_v03 axis_v04 axis_v05; do
    python Mapping/export_viz_samples.py \
        --renders-root $RENDERS --objects-root $OBJECTS \
        --symmetry-type axis_sym --sizes 224 --lightings flat \
        --experiment-id $EXP --method svd --n-views 14 \
        --n-samples 10 --results-dir $RESULTS
done

for EXP in plane_v00 plane_v01 plane_v02 plane_v03 plane_v04 plane_v05; do
    python Mapping/export_viz_samples.py \
        --renders-root $RENDERS --objects-root $OBJECTS \
        --symmetry-type plane_sym --sizes 224 --lightings flat \
        --experiment-id $EXP --method svd --n-views 14 \
        --n-samples 10 --results-dir $RESULTS
done
```
