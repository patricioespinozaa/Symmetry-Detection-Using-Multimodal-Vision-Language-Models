# Reporte de resultados — pipeline sin malla (triangulación multivista)

> Análisis de todos los archivos de `results/` que pertenecen al pipeline que
> estima eje/plano por triangulación de los puntos 2D de Molmo2, sin
> ray-casting contra la malla (`Mapping/estimate_symmetry_no_mesh.py` y las
> ablaciones de `Pipeline_Experiments/`). Fecha del análisis: 2026-09-26.
> Resumen corto: `docs/resumen_ejecutivo_sin_malla.md`. Contexto de los
> prompts y de la metodología: `docs/experimentos_pipeline_sin_malla.md`.
>
> Todas las tablas numéricas de este documento se generaron por script
> directamente desde los CSV (no se transcribieron a mano). Las
> verificaciones mecanísticas (§2, §3.3, §4.3) se hicieron re-ejecutando el
> código de producción (`pipeline_common/triangulation.py`,
> `Mapping/estimate_symmetry_no_mesh.py`, `Mapping/evaluate.py`) sobre los 30
> objetos del sandbox que existen localmente (`Experiments/objects/`,
> `Experiments/renders/`); donde una conclusión depende de esa muestra se
> indica explícitamente.

---

## 0. Hallazgos principales

1. **Eje: el pipeline sin malla está al nivel del azar.** Una dirección
   elegida uniformemente al azar da, analíticamente, error medio 57.30°,
   mediana 60°, AUC@45 = 0.0997, P@10° = 0.0152. El mejor baseline
   (`axis_v05_1`, n=26) da 57.36° / 60.53° / 0.104 / 0.013. **13 de los 14
   prompts con baseline son estadísticamente indistinguibles del azar o
   peores** (la excepción, `v02_1` a n=26, solo supera al azar sobre el 68%
   de objetos que logra predecir; su promedio total, 66°, es peor que el
   azar); ninguna de las 5 ablaciones cambia eso.
2. **El mecanismo está identificado**: los ejes GT del dataset están
   orientados al azar (los objetos axiales fueron rotados), las cámaras
   miran al origen con "arriba" = +Y del mundo, y Molmo2 marca
   mayoritariamente la columna central de la imagen (X≈500). Con eso la
   triangulación recupera el **eje vertical del render**, no el del objeto:
   en la muestra local, 12/29 predicciones son exactamente +Y (0.0°–1.2°) y
   48% caen a <5° de un eje del mundo. Los casos "BUENO" del sandbox son
   objetos cuyo GT *casualmente* es vertical.
3. **Único prompt de eje sobre el azar: `axis_v08`** (el diseñado contra el
   colapso a X≈500): 55.5°, z = −3.6, P@5° = 3.8× azar. Efecto real pero
   chico en términos absolutos.
4. **Prompts bilaterales (`v01`, `v02`, `v03` y `_1`) son incompatibles con
   los estimadores sin malla**: en eje dan resultados *significativamente
   peores que el azar* (z hasta +3.7) y en plano colapsan (recall 0.01–0.15).
   Los estimadores suponen que los dos puntos de una vista están **sobre**
   el eje/traza; un par bilateral los pone a ambos lados, y la línea que los
   une queda perpendicular a lo que se quiere estimar. No existe modo
   "midpoint" en el pipeline sin malla.
5. **Plano: los números altos (recall ≈ 0.85, `f1_ref` ≈ 0.48) no se pueden
   atribuir todavía a Molmo2.** Los planos GT están en pose canónica
   ShapeNet (100% de las normales de la muestra local a <1.4° de un eje del
   mundo; 23/31 son X). Un predictor trivial "siempre el plano X por el
   origen" obtiene en esa muestra recall 0.767, `f1_ref` 0.742; el pipeline
   sobre esos mismos objetos obtiene recall 0.667 (mismo promedio por objeto
   que `evaluate.py`). En el dataset completo,
   el sesgo a X≈500 correlaciona **positivamente** con el desempeño
   (Pearson +0.81 con F1 ajustado). Falta confirmar en los 850 con
   `Pipeline_Experiments/diagnostics/trivial_baselines.py` (§6.1).
6. **Las métricas de plano excluyen los objetos sin predicción**
   (`evaluate.py::compute_summary`), a diferencia de las de eje, que imputan
   90°. La cobertura varía entre 19% y 100%, así que las comparaciones
   crudas están sesgadas. Con recall ajustado por cobertura, el ranking y la
   conclusión sobre EXP-F cambian.
7. **Ablaciones**: con la comparación justa en modo multiplano (corrida
   `plane_maxplanes3_rerun`, `experiments_22_09_2026`), EXP-A/C/D no tienen
   efecto en plano (|Δ| ≤ 0.02; EXP-C es idéntica al baseline). **EXP-F**
   mejora `f1_ref` en 35/42 combinaciones (+0.060) y precisión (+0.118), pero
   lo hace **absteniéndose en ~1/3 de los objetos**: su recall ajustado por
   cobertura *baja* 0.078 (0 mejoras, 23 empeoramientos). En eje, A/B/D
   empeoran el ángulo ~0.6°, C es un no-op y E empeora 0.26°; la mejora de
   traslación de EXP-B se concentra en outliers (mediana sin cambio).

---

## 1. Qué archivos se analizaron

| Archivo / carpeta | Pipeline | Uso en este reporte |
|---|---|---|
| `results/experiments_15_09_2026/axis_sym_nomesh_comparison.csv` | sin malla | **Fuente definitiva de eje**: baseline (14 prompts) + EXP-A..E (20 prompts) |
| `results/experiments_22_09_2026/plane_sym_nomesh_comparison.csv` | sin malla | **Fuente definitiva de plano**: baseline + EXP-A/C/D (un plano y `mp3`) + EXP-F, con `SDE_ref`/`f1_ref` en todas las filas multiplano |
| `results/experiments_15_09_2026/plane_sym_nomesh_comparison.csv` | sin malla | Reemplazado por `22_09` (subconjunto; mismas cifras salvo EXP-F, ver nota) |
| `results/experiments_26_08_2026/*_nomesh_*`, `experiments_27_08_2026/*_nomesh_*` | sin malla | Corridas anteriores: **subconjuntos exactos** de `15_09` (0 celdas distintas en 36 y 42 filas comunes). No aportan información nueva |
| `results/experiments_27_08_2026_v2/` | sin malla | **Copia byte-idéntica** de `27_08` (mismo md5) |
| `results/diagnostics/{axis,plane}_sym_center_bias_{summary,detail}.csv` | sin malla | Diagnóstico de colapso a X≈500 (§5.1) |
| `results/diagnostics/plane_v04_1_flowC_expF_missing.csv` | sin malla | Razón de predicciones faltantes (§5.2) |
| `results/plots_nomesh/` | sin malla | Gráficos de la corrida `26_08` (12 prompts, desactualizados); no se usaron |
| `results/experiments_17_08_2026/`, `experiments_20_06_2026/` | **con malla** | Excluidos (métodos `svd`/`ransac_svd*`) |
| `results/{axis,plane}_sym/per_experiment/`, `results/{axis,plane}_sym/plots/` | **con malla** | Excluidos (variantes `cluster`, `hdbscan`, `p3`/`p5`, `acceptance_rate`, `sde_mean`) |
| `results/axis_v0X_1/viz_samples/` | **con malla** | Excluidos (contienen `mapped_points_3d`, es decir, ray-casting) |

**Nota sobre `15_09` vs `22_09` en plano**: las filas EXP-F difieren en
hasta 4 objetos y ≤0.0035 en recall/precisión entre ambas corridas
(la re-evaluación con `--with-reference-metrics` recomputó algunos objetos).
Es despreciable; se usa `22_09`.

**Lo que no está en ningún CSV**: `axis_lit2_grid` y `axis_lit3_cot` sí
corrieron las ablaciones en el servidor (aparecen en `audit_results.py`),
pero no figuran en `15_09` — no se pueden analizar hasta re-correr
`Mapping/compare_results_no_mesh.py` (§6). Sí figuran en el diagnóstico de
centrado.

---

## 2. Referencias necesarias para leer los números

### 2.1 Nivel de azar para el eje

Para una dirección uniforme en la esfera, el ángulo sin signo φ ∈ [0°, 90°]
contra un eje fijo tiene densidad sin φ, de donde:

| Métrica | Valor de azar | Derivación |
|---|---|---|
| error medio | **57.30°** | ∫ φ sin φ dφ = 1 rad |
| mediana | **60.00°** | 1 − cos φ = 0.5 |
| desvío estándar | 21.56° | √(π − 3) rad |
| AUC@45 (`evaluate.py`) | **0.0997** | 1 − sin(π/4)/(π/4) |
| P@5° / P@10° / P@15° | 0.0038 / 0.0152 / 0.0341 | 1 − cos θ |

Como `evaluate.py` imputa 90° a los objetos sin predicción, el azar
esperado para una fila con cobertura n/850 es (n·57.30 + (850−n)·90)/850;
la columna "z vs azar" de §3.1 usa esa referencia con error estándar
21.56·√n/850. z < −2 = mejor que azar; z > +2 = peor que azar.

### 2.2 Orientación del ground truth (muestra local de 30+30 objetos)

| | Eje (`curated_axis_sym_obj`) | Plano (`curated_plane_sym_obj`) |
|---|---|---|
| Vectores GT a <5° de un eje del mundo | 10% | **100%** |
| Vectores GT a <15° de un eje del mundo | 27% | **100%** |
| Distribución | aleatoria (media \|dir·Y\| = 0.45; uniforme = 0.50) | 23 X, 8 Z (pose canónica ShapeNet) |
| Origen GT | (0,0,0) en los 30 | (0,0,0) en los 30 |

Además, `ImagesGenerator/export_fibonacci_views.py` usa
`look_at_view_transform(eye=…)`: todas las cámaras miran al origen con
"arriba" = +Y, y para un `n_views` dado **el conjunto de cámaras es el mismo
para todos los objetos** (misma secuencia Fibonacci). Consecuencia: una
predicción que no depende del objeto (p. ej. "siempre +Y") es posible y,
en plano, puntúa bien por la pose canónica.

**Verificación pendiente**: esto está medido sobre la muestra local.
`trivial_baselines.py` (§6.1) lo mide sobre los 850+850.

### 2.3 Objetos sin predicción (cobertura)

`Mapping/evaluate.py::compute_summary` trata distinto a eje y plano:

- **Eje y plano-un-solo-plano** (`method="triangulation"`): métricas
  angulares sobre los 850, imputando 90° y precisión 0 a los faltantes.
  Traslación y `SDE_ref` solo sobre los válidos.
- **Plano multiplano** (`triangulation_multiplane`): `recall`, `precision`
  **y `f1_ref`** solo sobre objetos con predicción. Un objeto sin predicción
  no suma falsos negativos.

Por eso este reporte agrega, para plano, **recall ajustado** = recall ×
n_obj/850 (un faltante encontró 0 de sus planos GT; la precisión no cambia)
y **F1 ajustado** = media armónica de recall ajustado y precisión. No se
puede ajustar `f1_ref` sin los conteos TP/FP/FN por umbral, que no están en
el CSV (§6.3).

### 2.4 Qué mide `translation_error`

Distancia del **punto ancla** predicho a la **recta GT** (no distancia
entre rectas). Como el GT pasa por el origen y el origen es el centro al que
miran todas las cámaras, un ancla cerca del centro del objeto da error bajo
**independientemente de si la dirección es correcta**. Mide, sobre todo, qué
tan bien condicionado está el sistema de mínimos cuadrados que ubica el
ancla.

### 2.5 `f1_ref`

Umbrales {0.05, 0.10, 0.15, 0.20} sobre ‖[n, d] − [n', d']‖ (con signo
opuesto), TP/FP/FN acumulados sobre todo el dataset, F1 promediado sobre los
4 umbrales. Un error de normal de 15° equivale a 0.26 → no pasa ningún
umbral; se necesita < ~11.5° para el más laxo y < ~3° para el más estricto,
además de un offset `d` parecido. Es **mucho más estricta** que el recall a
15°. La variante greedy (puerto verbatim de PRS-Net) cuenta un FP por cada
comparación predicción–GT fallida, no por predicción: penaliza fuerte la
sobre-predicción de planos.

---

## 3. Eje (`axis_sym`)

### 3.1 Baseline — métricas de dirección (14 prompts × 3 `n_views`)

| Prompt | n_views | n_obj (cob.) | error medio | mediana | std | AUC@45 | P@5° | P@10° | P@15° | z vs azar |
|---|---|---|---|---|---|---|---|---|---|---|
| *Azar uniforme (analítico)* | — | 850 (100%) | 57.30 | 60.00 | 21.56 | 0.0997 | 0.0038 | 0.0152 | 0.0341 | 0 |
| `axis_v00` | 6 | 793 (93%) | 59.09 | 61.88 | 22.2 | 0.0935 | 0.0059 | 0.0153 | 0.0365 | -0.56 |
| `axis_v00` | 14 | 745 (88%) | 61.72 | 66.47 | 23.4 | 0.0905 | 0.0047 | 0.0224 | 0.0459 | +0.55 |
| `axis_v00` | 26 | 781 (92%) | 59.91 | 62.70 | 22.9 | 0.0952 | 0.0035 | 0.0141 | 0.0365 | -0.06 |
| `axis_v00_1` | 6 | 791 (93%) | 59.59 | 62.48 | 22.3 | 0.0921 | 0.0035 | 0.0129 | 0.0376 | +0.03 |
| `axis_v00_1` | 14 | 789 (93%) | 59.30 | 62.56 | 22.5 | 0.0943 | 0.0035 | 0.0141 | 0.0329 | -0.48 |
| `axis_v00_1` | 26 | 828 (97%) | 57.78 | 61.15 | 22.0 | 0.1002 | 0.0035 | 0.0129 | 0.0365 | -0.50 |
| `axis_v01` | 6 | 818 (96%) | 59.89 | 64.62 | 23.2 | 0.1022 | 0.0035 | 0.0188 | 0.0400 | +1.88 |
| `axis_v01` | 14 | 811 (95%) | 60.13 | 64.87 | 23.1 | 0.1001 | 0.0035 | 0.0200 | 0.0471 | +1.85 |
| `axis_v01` | 26 | 823 (97%) | 60.42 | 63.72 | 21.5 | 0.0834 | 0.0024 | 0.0153 | 0.0259 | +2.87 |
| `axis_v01_1` | 6 | 827 (97%) | 60.20 | 64.52 | 22.4 | 0.0933 | 0.0035 | 0.0153 | 0.0353 | +2.76 |
| `axis_v01_1` | 14 | 821 (97%) | 60.41 | 65.12 | 22.8 | 0.0963 | 0.0024 | 0.0141 | 0.0376 | +2.75 |
| `axis_v01_1` | 26 | 832 (98%) | 60.68 | 64.41 | 21.1 | 0.0792 | 0.0000 | 0.0129 | 0.0259 | +3.68 |
| `axis_v02` | 6 | 816 (96%) | 60.53 | 65.14 | 22.9 | 0.0979 | 0.0059 | 0.0176 | 0.0400 | +2.65 |
| `axis_v02` | 14 | 773 (91%) | 62.37 | 67.34 | 23.2 | 0.0913 | 0.0047 | 0.0176 | 0.0424 | +2.99 |
| `axis_v02` | 26 | 782 (92%) | 62.35 | 66.45 | 21.6 | 0.0756 | 0.0012 | 0.0094 | 0.0247 | +3.44 |
| `axis_v02_1` | 6 | 678 (80%) | 64.19 | 68.11 | 22.6 | 0.0710 | 0.0059 | 0.0153 | 0.0271 | +0.42 |
| `axis_v02_1` | 14 | 675 (79%) | 63.62 | 68.79 | 23.9 | 0.0871 | 0.0035 | 0.0129 | 0.0306 | -0.63 |
| `axis_v02_1` | 26 | 576 (68%) | 66.11 | 73.68 | 24.9 | 0.0861 | 0.0012 | 0.0106 | 0.0318 | -2.84 |
| `axis_v03` | 6 | 805 (95%) | 59.35 | 63.45 | 23.3 | 0.1051 | 0.0035 | 0.0188 | 0.0482 | +0.44 |
| `axis_v03` | 14 | 798 (94%) | 59.83 | 64.39 | 23.3 | 0.1012 | 0.0071 | 0.0200 | 0.0424 | +0.74 |
| `axis_v03` | 26 | 764 (90%) | 62.16 | 66.65 | 22.7 | 0.0873 | 0.0024 | 0.0118 | 0.0318 | +2.22 |
| `axis_v03_1` | 6 | 814 (96%) | 59.74 | 63.49 | 23.1 | 0.1018 | 0.0024 | 0.0224 | 0.0435 | +1.46 |
| `axis_v03_1` | 14 | 800 (94%) | 60.72 | 65.36 | 23.1 | 0.0956 | 0.0012 | 0.0118 | 0.0306 | +2.09 |
| `axis_v03_1` | 26 | 782 (92%) | 61.43 | 65.31 | 22.1 | 0.0837 | 0.0012 | 0.0129 | 0.0282 | +2.15 |
| `axis_v04` | 6 | 831 (98%) | 57.68 | 59.84 | 22.0 | 0.1001 | 0.0059 | 0.0165 | 0.0424 | -0.48 |
| `axis_v04` | 14 | 816 (96%) | 58.51 | 61.82 | 22.8 | 0.1043 | 0.0047 | 0.0141 | 0.0447 | -0.13 |
| `axis_v04` | 26 | 806 (95%) | 58.66 | 61.53 | 22.6 | 0.1014 | 0.0047 | 0.0165 | 0.0400 | -0.46 |
| `axis_v04_1` | 6 | 824 (97%) | 58.50 | 60.54 | 21.9 | 0.0935 | 0.0059 | 0.0141 | 0.0435 | +0.28 |
| `axis_v04_1` | 14 | 809 (95%) | 59.39 | 62.16 | 22.5 | 0.0940 | 0.0059 | 0.0153 | 0.0435 | +0.71 |
| `axis_v04_1` | 26 | 829 (98%) | 57.84 | 60.06 | 22.1 | 0.0989 | 0.0035 | 0.0176 | 0.0424 | -0.37 |
| `axis_v05` | 6 | 818 (96%) | 58.68 | 61.29 | 22.3 | 0.0965 | 0.0035 | 0.0165 | 0.0424 | +0.20 |
| `axis_v05` | 14 | 801 (94%) | 58.32 | 60.52 | 23.0 | 0.1050 | 0.0071 | 0.0188 | 0.0424 | -1.20 |
| `axis_v05` | 26 | 830 (98%) | 57.67 | 60.85 | 22.1 | 0.1021 | 0.0035 | 0.0141 | 0.0353 | -0.53 |
| `axis_v05_1` | 6 | 804 (95%) | 58.64 | 61.81 | 22.4 | 0.0982 | 0.0035 | 0.0141 | 0.0329 | -0.59 |
| `axis_v05_1` | 14 | 793 (93%) | 58.95 | 61.93 | 22.7 | 0.0994 | 0.0059 | 0.0165 | 0.0365 | -0.75 |
| `axis_v05_1` | 26 | 831 (98%) | 57.36 | 60.53 | 22.1 | 0.1039 | 0.0035 | 0.0129 | 0.0353 | -0.92 |
| `axis_v06` | 6 | 824 (97%) | 57.60 | 59.62 | 22.1 | 0.1011 | 0.0071 | 0.0153 | 0.0471 | -0.95 |
| `axis_v06` | 14 | 812 (96%) | 58.90 | 61.60 | 22.2 | 0.0958 | 0.0047 | 0.0188 | 0.0447 | +0.20 |
| `axis_v06` | 26 | 821 (97%) | 57.37 | 58.68 | 22.4 | 0.1034 | 0.0035 | 0.0176 | 0.0435 | -1.44 |
| `axis_v07` | 6 | 502 (59%) | 70.92 | 79.62 | 22.3 | 0.0514 | 0.0024 | 0.0094 | 0.0141 | +0.42 |
| `axis_v07` | 14 | 493 (58%) | 71.86 | 82.74 | 22.4 | 0.0535 | 0.0024 | 0.0094 | 0.0188 | +1.46 |
| `axis_v07` | 26 | 500 (59%) | 70.10 | 80.70 | 23.6 | 0.0653 | 0.0012 | 0.0082 | 0.0235 | -1.16 |

### 3.2 Baseline — traslación y `SDE_ref`

| Prompt | n_views | trasl. norm. media | trasl. norm. mediana | trasl. cruda media | SDE_ref media | SDE_ref mín | SDE_ref máx |
|---|---|---|---|---|---|---|---|
| `axis_v00` | 6 | 0.467 | 0.072 | 0.504 | 0.0566 | 0.000174 | 1.751 |
| `axis_v00` | 14 | 0.287 | 0.058 | 0.309 | 0.0601 | 0.000172 | 2.138 |
| `axis_v00` | 26 | 0.264 | 0.086 | 0.284 | 0.0767 | 0.000170 | 2.711 |
| `axis_v00_1` | 6 | 0.969 | 0.343 | 1.049 | 0.0492 | 0.000013 | 1.537 |
| `axis_v00_1` | 14 | 0.621 | 0.317 | 0.670 | 0.0507 | 0.000014 | 1.345 |
| `axis_v00_1` | 26 | 0.558 | 0.287 | 0.603 | 0.0560 | 0.000014 | 2.211 |
| `axis_v01` | 6 | 0.110 | 0.062 | 0.120 | 0.0432 | 0.000161 | 0.688 |
| `axis_v01` | 14 | 0.075 | 0.047 | 0.083 | 0.0619 | 0.000442 | 6.321 |
| `axis_v01` | 26 | 0.085 | 0.063 | 0.093 | 0.0726 | 0.000707 | 2.873 |
| `axis_v01_1` | 6 | 0.107 | 0.059 | 0.116 | 0.0412 | 0.000536 | 0.717 |
| `axis_v01_1` | 14 | 0.063 | 0.041 | 0.069 | 0.0461 | 0.000474 | 0.745 |
| `axis_v01_1` | 26 | 0.089 | 0.062 | 0.096 | 0.0802 | 0.000384 | 3.570 |
| `axis_v02` | 6 | 0.116 | 0.068 | 0.127 | 0.0589 | 0.000041 | 1.922 |
| `axis_v02` | 14 | 0.086 | 0.048 | 0.094 | 0.0672 | 0.000042 | 2.416 |
| `axis_v02` | 26 | 0.099 | 0.075 | 0.108 | 0.0881 | 0.000482 | 2.758 |
| `axis_v02_1` | 6 | 0.361 | 0.032 | 0.390 | 0.0433 | 0.000034 | 0.842 |
| `axis_v02_1` | 14 | 0.178 | 0.025 | 0.193 | 0.0386 | 0.000014 | 0.362 |
| `axis_v02_1` | 26 | 0.201 | 0.041 | 0.217 | 0.0491 | 0.000014 | 0.519 |
| `axis_v03` | 6 | 0.099 | 0.051 | 0.108 | 0.0583 | 0.000151 | 2.402 |
| `axis_v03` | 14 | 0.067 | 0.040 | 0.073 | 0.0600 | 0.000315 | 1.182 |
| `axis_v03` | 26 | 0.086 | 0.059 | 0.094 | 0.0839 | 0.000244 | 2.406 |
| `axis_v03_1` | 6 | 0.105 | 0.053 | 0.115 | 0.0672 | 0.000110 | 9.963 |
| `axis_v03_1` | 14 | 0.067 | 0.039 | 0.073 | 0.0605 | 0.000502 | 1.694 |
| `axis_v03_1` | 26 | 0.084 | 0.051 | 0.092 | 0.0870 | 0.000203 | 2.130 |
| `axis_v04` | 6 | 0.278 | 0.036 | 0.295 | 0.0438 | 0.000176 | 2.024 |
| `axis_v04` | 14 | 0.147 | 0.030 | 0.155 | 0.0451 | 0.000302 | 0.584 |
| `axis_v04` | 26 | 0.193 | 0.059 | 0.206 | 0.0606 | 0.000173 | 1.379 |
| `axis_v04_1` | 6 | 0.429 | 0.038 | 0.452 | 0.0461 | 0.000160 | 1.889 |
| `axis_v04_1` | 14 | 0.204 | 0.034 | 0.217 | 0.0482 | 0.000159 | 1.998 |
| `axis_v04_1` | 26 | 0.218 | 0.060 | 0.237 | 0.0548 | 0.000160 | 1.083 |
| `axis_v05` | 6 | 0.603 | 0.107 | 0.643 | 0.0482 | 0.000179 | 1.629 |
| `axis_v05` | 14 | 0.400 | 0.117 | 0.428 | 0.0494 | 0.000178 | 1.913 |
| `axis_v05` | 26 | 0.419 | 0.158 | 0.451 | 0.0524 | 0.000176 | 1.196 |
| `axis_v05_1` | 6 | 1.305 | 0.305 | 1.386 | 0.0471 | 0.000013 | 1.315 |
| `axis_v05_1` | 14 | 0.567 | 0.239 | 0.615 | 0.0420 | 0.000170 | 0.763 |
| `axis_v05_1` | 26 | 0.424 | 0.151 | 0.461 | 0.0405 | 0.000014 | 0.875 |
| `axis_v06` | 6 | 0.336 | 0.032 | 0.356 | 0.0431 | 0.000160 | 1.872 |
| `axis_v06` | 14 | 0.180 | 0.031 | 0.192 | 0.0445 | 0.000190 | 0.797 |
| `axis_v06` | 26 | 0.165 | 0.060 | 0.177 | 0.0609 | 0.000178 | 1.535 |
| `axis_v07` | 6 | 0.739 | 0.070 | 0.805 | 0.0499 | 0.000033 | 0.917 |
| `axis_v07` | 14 | 0.274 | 0.034 | 0.294 | 0.0460 | 0.000013 | 0.330 |
| `axis_v07` | 26 | 0.220 | 0.052 | 0.236 | 0.0650 | 0.000013 | 1.212 |

`SDE_ref` solo está calculado para las filas baseline (42/342); ninguna
ablación de eje lo tiene.

### 3.3 Prompts sin corrida baseline propia (EXP-C como proxy)

Estos prompts solo corrieron las ablaciones. Para prompts de **2 puntos por
vista**, EXP-C es **numéricamente idéntico al baseline** (verificado: Δ =
0.000 en las 42 combinaciones que sí tienen ambos). Para los `_6pts` no es
idéntico (el RANSAC 2D sí actúa con ≥3 puntos).

| Prompt | n_views | n_obj (cob.) | error medio | mediana | AUC@45 | P@5° | P@10° | P@15° | z vs azar | trasl. norm. media |
|---|---|---|---|---|---|---|---|---|---|---|
| `axis_v08` | 6 | 829 (98%) | 55.51 | 58.14 | 0.1186 | 0.0141 | 0.0224 | 0.0459 | -3.55 | 0.064 |
| `axis_v08` | 14 | 813 (96%) | 56.97 | 59.51 | 0.1132 | 0.0082 | 0.0271 | 0.0471 | -2.42 | 0.056 |
| `axis_v08` | 26 | 829 (98%) | 57.11 | 58.93 | 0.1021 | 0.0024 | 0.0129 | 0.0329 | -1.36 | 0.092 |
| `axis_v05_1_flowB` | 6 | 809 (95%) | 58.48 | 60.78 | 0.0974 | 0.0035 | 0.0141 | 0.0365 | -0.54 | 0.877 |
| `axis_v05_1_flowB` | 14 | 817 (96%) | 58.31 | 61.28 | 0.1018 | 0.0047 | 0.0141 | 0.0318 | -0.35 | 0.627 |
| `axis_v05_1_flowB` | 26 | 838 (99%) | 57.29 | 60.20 | 0.1048 | 0.0035 | 0.0129 | 0.0353 | -0.64 | 0.360 |
| `axis_v05_1_flowC` | 6 | 374 (44%) | 76.39 | 90.00 | 0.0344 | 0.0024 | 0.0059 | 0.0106 | +1.59 | 0.102 |
| `axis_v05_1_flowC` | 14 | 244 (29%) | 81.78 | 90.00 | 0.0221 | 0.0024 | 0.0035 | 0.0082 | +2.96 | 0.082 |
| `axis_v05_1_flowC` | 26 | 168 (20%) | 83.40 | 90.00 | 0.0209 | 0.0012 | 0.0024 | 0.0059 | -0.41 | 0.105 |
| `axis_v06_6pts` | 6 | 797 (94%) | 59.12 | 62.18 | 0.1033 | 0.0047 | 0.0153 | 0.0400 | -0.30 | 0.795 |
| `axis_v06_6pts` | 14 | 682 (80%) | 63.62 | 68.35 | 0.0888 | 0.0047 | 0.0129 | 0.0412 | -0.21 | 0.368 |
| `axis_v06_6pts` | 26 | 806 (95%) | 57.72 | 60.06 | 0.1084 | 0.0047 | 0.0141 | 0.0365 | -1.77 | 0.402 |
| `axis_v07_6pts` | 6 | 797 (94%) | 58.98 | 62.52 | 0.1048 | 0.0035 | 0.0141 | 0.0412 | -0.50 | 0.635 |
| `axis_v07_6pts` | 14 | 735 (86%) | 60.90 | 65.99 | 0.1008 | 0.0047 | 0.0188 | 0.0376 | -1.20 | 0.444 |
| `axis_v07_6pts` | 26 | 731 (86%) | 60.01 | 63.22 | 0.1056 | 0.0035 | 0.0141 | 0.0365 | -2.71 | 0.369 |

### 3.4 Por qué sube o baja cada prompt

Todo lo que sigue se lee contra el azar (57.3°): una diferencia de ±1–2°
entre prompts con z entre −2 y +2 **no es una diferencia real**.

| Prompt | Qué pide | Resultado | Por qué |
|---|---|---|---|
| `v00` | 2 puntos sobre el eje proyectado | 59.1–61.7°, ≈ azar | ~52% de los puntos no cenitales en X≈500. Sin saber hacia dónde apunta el eje del objeto, "el eje en la imagen" se vuelve la vertical de la imagen → la triangulación da +Y del mundo, que no guarda relación con un GT rotado al azar |
| `v00_1` | + split arriba/abajo + definición de centerline | 57.8–59.6°, ≈ azar; **peor traslación** (0.56–0.97) | La definición de centerline empuja al 77–80% a X≈500. Todas las vistas generan planos de interpretación que contienen la misma vertical → el sistema que ubica el ancla queda casi degenerado (ver correlación centrado↔traslación +0.79, §5.1). La mejora de v1 medida con malla (+0.121 AUC) no se transfiere |
| `v01`, `v01_1` | par bilateral izquierda/derecha | 59.9–60.7°, **peor que azar** (z +1.9 a +3.7) | `widest_pair` asume que los dos puntos están **sobre** el eje; un par bilateral los pone a ambos lados, así que la línea que los une es perpendicular a la proyección del eje. El pipeline con malla resolvía esto con `--point-mode midpoint`; el sin malla no tiene equivalente. Traslación buena (0.06–0.11) porque el par rodea el centro |
| `v02` | extremos de la silueta más ancha | 60.5–62.4°, **peor que azar** (z +2.7 a +3.4) | Misma incompatibilidad que `v01`, más marcada: los extremos izquierdo/derecho son perpendiculares al eje por construcción |
| `v02_1` | rediseño a midpoints sobre la centerline | 63.6–66.1°; cobertura 68–80% | Vuelve a los puntos sobre la vertical (66–70% en X≈500), pero con 170–270 objetos sin predicción imputados a 90°. Sobre los que sí predice es ≈ azar a n=6/14 y levemente mejor a n=26 (z −2.8), es decir, falla justamente en objetos que habría errado igual |
| `v03`, `v03_1` | pares de elementos estructurales | 59.4–62.2°, peor que azar a n=26 (z +2.1 a +2.2) | Son pares bilaterales: misma incompatibilidad que `v01` |
| `v04`, `v04_1` | polos (donde el eje sale de la superficie) | 57.7–59.4°, ≈ azar | 39–47% en X≈500. Los "polos" en la imagen tienden a quedar en la vertical central |
| `v05`, `v05_1` | centerline, mitad superior + mitad inferior (+ paso 0 global) | 57.4–58.9° (los mejores crudos), ≈ azar | El más centrado: 63–86% en X≈500 → colapso a +Y casi total. "Mejor" solo por estar más cerca de 57.3° que los bilaterales. Traslación mala (0.40–1.30) por el mismo motivo que `v00_1` |
| `v06` | polos + regla de centro de curvatura | 57.4–58.9°, ≈ azar | 43–45% en X≈500 |
| `v07` | corte más angosto (cuello/cintura) | 70.1–71.9° | Cobertura 58–59%: ~350 objetos imputados a 90°. Sobre los predichos, ≈ azar (z −1.2 a +1.5) |
| `v08` | polos + auto-verificación "si ambos puntos caen en X≈500, la cámara no mira de frente al eje" | **55.5° a n=6, z = −3.6; P@5° = 0.014 (3.8× azar); AUC 0.119** | **El único sobre el azar**, y el único diseñado contra el colapso: 18–20% en X≈500. La señal se diluye con más vistas (z −2.4 a n=14, −1.4 a n=26). Mejor traslación de los prompts "sobre el eje" (0.06–0.09) |
| `v05_1_flowB` | `v05_1` + descripción previa | 57.3–58.5°, ≈ azar | Idéntico a `v05_1`; la descripción aumenta el centrado (91% a n=26). No aporta |
| `v05_1_flowC` | descripción + pista de ubicación, sin identidad entre vistas | 76.4–83.4°; cobertura 20–44% | Molmo2 devuelve <2 puntos útiles por vista (§5.2); la imputación domina el promedio |
| `v06_6pts`, `v07_6pts` | 6 puntos a lo largo del eje | 57.7–63.6°, ≈ azar | Más puntos por vista no cambian la dirección de la línea (siguen en la vertical: 62–70% centrados). Cobertura cae a 80–86% en n=14 |

### 3.5 Efecto de `n_views`

- **Dirección**: sin tendencia consistente; entre 6, 14 y 26 vistas los
  cambios son de 1–3° y en ambos sentidos. Es lo esperado si la salida no
  depende del objeto.
- **Traslación**: mejora con más vistas en los prompts "sobre el eje"
  (`v05_1` 1.30 → 0.57 → 0.42; `v00_1` 0.97 → 0.62 → 0.56): más planos de
  interpretación no paralelos condicionan mejor el ancla. En los bilaterales
  ya es baja con 6 vistas.
- **Cobertura**: estable salvo `v02_1` (80% → 68%) y `v07`/`flowC`.

### 3.6 Ablaciones EXP-A..E (pareadas, 42 combinaciones prompt × `n_views`)

| Variante | Δ error medio | Δ mediana | Δ AUC | Δ P@10° | Δ P@15° | Δ trasl. norm. media | Δ trasl. norm. mediana | mejor / peor (error medio) |
|---|---|---|---|---|---|---|---|---|
| EXP-A | +0.615° | +0.772° | -0.0053 | -0.0010 | -0.0029 | +0.0044 | +0.0010 | 2 / 40 |
| EXP-B | +0.615° | +0.772° | -0.0053 | -0.0010 | -0.0029 | -0.2200 | -0.0378 | 2 / 40 |
| EXP-C | +0.000° | +0.000° | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0 / 0 |
| EXP-D | +0.614° | +0.767° | -0.0053 | -0.0010 | -0.0029 | +0.0052 | +0.0010 | 2 / 40 |
| EXP-E | +0.264° | +0.350° | -0.0009 | +0.0006 | +0.0010 | +0.0014 | +0.0003 | 3 / 39 |

- **EXP-C = baseline exacto** (con 2 puntos por vista, el RANSAC 2D devuelve
  el único par posible).
- **EXP-A y EXP-B tienen el mismo error angular** en las 42: calculan la
  dirección con la misma fórmula ponderada; solo difieren en el ancla.
  EXP-D ≈ EXP-A (la penalización de borde casi nunca actúa: los puntos
  están cerca del centro).
- **Ponderar empeora (~0.6°, 40/42)**: repesar vistas por separación en
  píxeles no puede recuperar una dirección que la entrada no contiene; solo
  cambia cuál de las vistas (todas colapsadas) domina.
- **EXP-B mejora la traslación media (−0.22) pero casi no la mediana (Δ
  típico −0.003; 22 combinaciones mejoran y 20 empeoran)**: su ancla (punto más cercano a todos
  los rayos) cae en el centro al que miran las cámaras, que está sobre el GT.
  Elimina los outliers del ancla del baseline; no localiza mejor el eje.
  **Esto corrige** lo afirmado en `docs/experimentos_pipeline_sin_malla.md`
  §5.2bis.

---

## 4. Plano (`plane_sym`)

### 4.1 Baseline multiplano (`max_planes=3`) — todas las métricas

| Prompt | n_views | n_obj (cob.) | planos pred. | planos GT | matcheados | recall | precisión | recall ajust. | F1 ajust. | SDE_ref | f1_ref | f1_ref_hung. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `plane_v00` | 6 | 833 (98%) | 1.00 | 1.19 | 0.412 | 0.368 | 0.412 | 0.361 | 0.385 | 0.0729 | 0.144 | 0.157 |
| `plane_v00` | 14 | 777 (91%) | 1.97 | 1.19 | 0.687 | 0.635 | 0.351 | 0.580 | 0.438 | 0.0296 | 0.281 | 0.277 |
| `plane_v00` | 26 | 810 (95%) | 2.35 | 1.20 | 0.675 | 0.595 | 0.331 | 0.567 | 0.418 | 0.0263 | 0.133 | 0.144 |
| `plane_v00_1` | 6 | 737 (87%) | 1.00 | 1.18 | 0.316 | 0.293 | 0.316 | 0.254 | 0.282 | 0.0212 | 0.148 | 0.161 |
| `plane_v00_1` | 14 | 582 (68%) | 1.94 | 1.20 | 0.902 | 0.816 | 0.477 | 0.559 | 0.514 | 0.0142 | 0.442 | 0.416 |
| `plane_v00_1` | 26 | 837 (98%) | 2.36 | 1.19 | 0.876 | 0.763 | 0.424 | 0.751 | 0.542 | 0.0145 | 0.171 | 0.184 |
| `plane_v01` | 6 | 846 (100%) | 1.00 | 1.19 | 0.128 | 0.077 | 0.128 | 0.077 | 0.096 | 0.0182 | 0.067 | 0.068 |
| `plane_v01` | 14 | 846 (100%) | 1.96 | 1.19 | 0.018 | 0.014 | 0.009 | 0.014 | 0.011 | 0.0440 | 0.002 | 0.002 |
| `plane_v01` | 26 | 840 (99%) | 2.00 | 1.20 | 0.020 | 0.016 | 0.010 | 0.016 | 0.012 | 0.0519 | 0.001 | 0.002 |
| `plane_v01_1` | 6 | 847 (100%) | 1.00 | 1.19 | 0.125 | 0.073 | 0.125 | 0.072 | 0.092 | 0.0162 | 0.072 | 0.071 |
| `plane_v01_1` | 14 | 847 (100%) | 1.96 | 1.19 | 0.022 | 0.016 | 0.011 | 0.016 | 0.013 | 0.0395 | 0.001 | 0.002 |
| `plane_v01_1` | 26 | 844 (99%) | 2.00 | 1.19 | 0.017 | 0.012 | 0.008 | 0.012 | 0.010 | 0.0454 | 0.001 | 0.001 |
| `plane_v02` | 6 | 792 (93%) | 1.00 | 1.19 | 0.516 | 0.464 | 0.516 | 0.433 | 0.471 | 0.0135 | 0.259 | 0.281 |
| `plane_v02` | 14 | 658 (77%) | 1.93 | 1.18 | 0.898 | 0.821 | 0.476 | 0.635 | 0.545 | 0.0142 | 0.428 | 0.401 |
| `plane_v02` | 26 | 828 (97%) | 2.13 | 1.19 | 0.855 | 0.753 | 0.484 | 0.734 | 0.583 | 0.0131 | 0.217 | 0.235 |
| `plane_v02_1` | 6 | 717 (84%) | 1.00 | 1.18 | 0.407 | 0.369 | 0.407 | 0.311 | 0.353 | 0.0170 | 0.203 | 0.221 |
| `plane_v02_1` | 14 | 544 (64%) | 1.95 | 1.19 | 0.888 | 0.808 | 0.467 | 0.517 | 0.491 | 0.0150 | 0.414 | 0.388 |
| `plane_v02_1` | 26 | 779 (92%) | 2.18 | 1.19 | 0.843 | 0.737 | 0.455 | 0.676 | 0.544 | 0.0138 | 0.207 | 0.223 |
| `plane_v03` | 6 | 843 (99%) | 1.00 | 1.19 | 0.173 | 0.109 | 0.173 | 0.108 | 0.133 | 0.0201 | 0.083 | 0.086 |
| `plane_v03` | 14 | 840 (99%) | 1.95 | 1.20 | 0.024 | 0.019 | 0.013 | 0.019 | 0.015 | 0.0483 | 0.003 | 0.003 |
| `plane_v03` | 26 | 823 (97%) | 2.00 | 1.20 | 0.017 | 0.013 | 0.009 | 0.012 | 0.010 | 0.0583 | 0.003 | 0.003 |
| `plane_v03_1` | 6 | 838 (99%) | 1.00 | 1.19 | 0.166 | 0.103 | 0.166 | 0.102 | 0.126 | 0.0223 | 0.077 | 0.081 |
| `plane_v03_1` | 14 | 841 (99%) | 1.95 | 1.19 | 0.031 | 0.025 | 0.015 | 0.025 | 0.019 | 0.0468 | 0.007 | 0.008 |
| `plane_v03_1` | 26 | 826 (97%) | 2.00 | 1.20 | 0.013 | 0.011 | 0.007 | 0.010 | 0.009 | 0.0523 | 0.002 | 0.002 |
| `plane_v04` | 6 | 648 (76%) | 1.00 | 1.16 | 0.299 | 0.280 | 0.299 | 0.213 | 0.249 | 0.0204 | 0.153 | 0.164 |
| `plane_v04` | 14 | 641 (75%) | 1.94 | 1.17 | 0.924 | 0.845 | 0.487 | 0.637 | 0.552 | 0.0143 | 0.475 | 0.426 |
| `plane_v04` | 26 | 756 (89%) | 2.36 | 1.18 | 0.939 | 0.819 | 0.438 | 0.729 | 0.547 | 0.0121 | 0.163 | 0.177 |
| `plane_v04_1` | 6 | 707 (83%) | 1.00 | 1.18 | 0.229 | 0.214 | 0.229 | 0.178 | 0.200 | 0.0211 | 0.115 | 0.124 |
| `plane_v04_1` | 14 | 553 (65%) | 1.89 | 1.17 | 0.931 | 0.854 | 0.518 | 0.556 | 0.536 | 0.0119 | 0.482 | 0.435 |
| `plane_v04_1` | 26 | 687 (81%) | 2.39 | 1.16 | 0.937 | 0.830 | 0.426 | 0.671 | 0.521 | 0.0130 | 0.162 | 0.174 |
| `plane_v05` | 6 | 839 (99%) | 1.00 | 1.18 | 0.198 | 0.153 | 0.198 | 0.151 | 0.171 | 0.0764 | 0.079 | 0.082 |
| `plane_v05` | 14 | 812 (96%) | 1.97 | 1.19 | 0.108 | 0.098 | 0.054 | 0.094 | 0.069 | 0.0566 | 0.021 | 0.023 |
| `plane_v05` | 26 | 848 (100%) | 2.01 | 1.19 | 0.092 | 0.082 | 0.045 | 0.082 | 0.058 | 0.0555 | 0.026 | 0.029 |
| `plane_v05_1` | 6 | 796 (94%) | 1.00 | 1.19 | 0.563 | 0.502 | 0.563 | 0.470 | 0.512 | 0.0159 | 0.271 | 0.294 |
| `plane_v05_1` | 14 | 684 (80%) | 1.97 | 1.18 | 0.738 | 0.679 | 0.380 | 0.546 | 0.448 | 0.0223 | 0.324 | 0.314 |
| `plane_v05_1` | 26 | 827 (97%) | 2.23 | 1.19 | 0.573 | 0.505 | 0.292 | 0.491 | 0.367 | 0.0271 | 0.118 | 0.127 |
| `plane_v06` | 6 | 730 (86%) | 1.00 | 1.18 | 0.444 | 0.402 | 0.444 | 0.345 | 0.388 | 0.0167 | 0.218 | 0.236 |
| `plane_v06` | 14 | 531 (62%) | 1.93 | 1.18 | 0.908 | 0.830 | 0.487 | 0.518 | 0.502 | 0.0145 | 0.435 | 0.403 |
| `plane_v06` | 26 | 822 (97%) | 2.17 | 1.19 | 0.865 | 0.751 | 0.464 | 0.726 | 0.566 | 0.0136 | 0.205 | 0.223 |
| `plane_v07` | 6 | 652 (77%) | 1.00 | 1.17 | 0.233 | 0.216 | 0.233 | 0.166 | 0.194 | 0.0220 | 0.112 | 0.121 |
| `plane_v07` | 14 | 501 (59%) | 1.92 | 1.17 | 0.922 | 0.846 | 0.499 | 0.499 | 0.499 | 0.0124 | 0.465 | 0.430 |
| `plane_v07` | 26 | 709 (83%) | 2.49 | 1.19 | 0.985 | 0.858 | 0.443 | 0.716 | 0.547 | 0.0136 | 0.164 | 0.176 |

### 4.2 Prompts sin corrida baseline propia (EXP-C `mp3` como proxy)

EXP-C en modo multiplano es **idéntica al baseline** (Δ = 0 en las 42
combinaciones con ambos).

| Prompt | n_views | n_obj (cob.) | planos pred. | planos GT | matcheados | recall | precisión | recall ajust. | F1 ajust. | SDE_ref | f1_ref | f1_ref_hung. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `plane_v08` | 6 | 841 (99%) | 1.00 | 1.19 | 0.165 | 0.120 | 0.165 | 0.119 | 0.139 | 0.0352 | 0.062 | 0.066 |
| `plane_v08` | 14 | 844 (99%) | 1.96 | 1.19 | 0.094 | 0.086 | 0.047 | 0.085 | 0.060 | 0.0570 | 0.019 | 0.021 |
| `plane_v08` | 26 | 844 (99%) | 2.02 | 1.19 | 0.071 | 0.065 | 0.035 | 0.064 | 0.045 | 0.0569 | 0.017 | 0.019 |
| `plane_v04_1_flowB` | 6 | 668 (79%) | 1.00 | 1.17 | 0.295 | 0.275 | 0.295 | 0.216 | 0.249 | 0.0200 | 0.143 | 0.154 |
| `plane_v04_1_flowB` | 14 | 578 (68%) | 1.90 | 1.19 | 0.941 | 0.854 | 0.516 | 0.580 | 0.547 | 0.0116 | 0.464 | 0.427 |
| `plane_v04_1_flowB` | 26 | 755 (89%) | 2.53 | 1.17 | 0.976 | 0.844 | 0.400 | 0.750 | 0.521 | 0.0132 | 0.140 | 0.150 |
| `plane_v04_1_flowC` | 6 | 317 (37%) | 1.00 | 1.26 | 0.227 | 0.158 | 0.227 | 0.059 | 0.094 | 0.0766 | 0.089 | 0.095 |
| `plane_v04_1_flowC` | 14 | 163 (19%) | 1.90 | 1.31 | 0.135 | 0.114 | 0.074 | 0.022 | 0.034 | 0.0666 | 0.024 | 0.029 |
| `plane_v04_1_flowC` | 26 | 169 (20%) | 2.17 | 1.24 | 0.136 | 0.118 | 0.053 | 0.024 | 0.033 | 0.0559 | 0.024 | 0.024 |
| `plane_v04_1_6pts` | 6 | 649 (76%) | 1.00 | 1.17 | 0.279 | 0.254 | 0.279 | 0.194 | 0.229 | 0.0210 | 0.147 | 0.158 |
| `plane_v04_1_6pts` | 14 | 425 (50%) | 1.94 | 1.17 | 0.864 | 0.792 | 0.458 | 0.396 | 0.425 | 0.0154 | 0.436 | 0.390 |
| `plane_v04_1_6pts` | 26 | 294 (35%) | 2.30 | 1.14 | 0.694 | 0.638 | 0.285 | 0.221 | 0.249 | 0.0214 | 0.098 | 0.103 |
| `plane_v05_6pts` | 6 | 806 (95%) | 1.00 | 1.19 | 0.454 | 0.416 | 0.454 | 0.394 | 0.422 | 0.0253 | 0.239 | 0.260 |
| `plane_v05_6pts` | 14 | 724 (85%) | 1.92 | 1.20 | 0.930 | 0.837 | 0.505 | 0.713 | 0.591 | 0.0115 | 0.477 | 0.433 |
| `plane_v05_6pts` | 26 | 771 (91%) | 2.65 | 1.18 | 0.982 | 0.846 | 0.392 | 0.768 | 0.519 | 0.0156 | 0.129 | 0.139 |

### 4.3 Por qué sube o baja cada prompt

| Prompt | Qué pide | Resultado (mejor fila) | Por qué |
|---|---|---|---|
| `v00` | traza del plano, punto arriba y abajo | `f1_ref` 0.28; recall ajust. 0.58 | Centrado moderado y creciente con n (41% → 60%) |
| `v00_1` | + split + guía de centro horizontal | `f1_ref` 0.44 (n=14); F1 ajust. 0.54 (n=26) | La guía de "centro horizontal" lleva el centrado a 91–93% → más aciertos |
| `v01`, `v01_1` | par bilateral | `f1_ref` ≤ 0.07; recall 0.01–0.08 | **Incompatibilidad estructural**: el segmento P–P' de un par espejo es **paralelo a la normal**, pero el estimador lo trata como una recta **contenida** en el plano. Las normales candidatas salen mal por construcción. (`docs/pipeline_sin_malla.md` §6 recomendaba pares bilaterales justamente por P−P' ∥ n, pero el estimador implementado no usa esa relación.) Centrado 4–7% |
| `v02`, `v02_1` | costura (seam) visible | `f1_ref` 0.41–0.43 (n=14); **mejor F1 ajust.: `v02` n=26 = 0.583** | Puntos sobre la línea central: 84–94% centrados. Alta cobertura a n=26 |
| `v03`, `v03_1` | pares estructurales | `f1_ref` ≤ 0.08 | Bilateral: misma incompatibilidad que `v01` |
| `v04`, `v04_1` | punto medio horizontal de la silueta | **mejor `f1_ref` crudo: 0.475 / 0.482 (n=14)** | El más centrado (94–99%). Con pose canónica, el punto medio de la silueta coincide con la traza del plano X. Pero la cobertura a n=14 es 65–75%: parte del `f1_ref` alto es de objetos "fáciles" |
| `v05` | los dos puntos más distantes de la traza | `f1_ref` ≤ 0.08 | Molmo2 elige puntos diagonales (4–6% centrados) |
| `v05_1` | "máxima separación **vertical**, no diagonal" | `f1_ref` 0.32 (n=14); recall 0.50 a n=6 | El reencuadre mueve los puntos a la columna central (62–73%): mejora grande sobre `v05` (+0.25 `f1_ref`) |
| `v05_6pts` | 6 puntos a lo largo de la traza + auto-chequeo de colinealidad | `f1_ref` 0.477 (n=14); **recall ajust. 0.713, F1 ajust. 0.591** — el más alto de todos | 92–95% centrados y buena cobertura (85% a n=14): 6 puntos alineados sobre la vertical dan pares robustos |
| `v04_1_6pts` | `v04_1` con 6 midpoints | `f1_ref` 0.436 (n=14) | Similar a `v04_1`, pero la cobertura colapsa a n=26 (35%) |
| `v06` | costura reforzada + fallback de punto medio | `f1_ref` 0.435; F1 ajust. 0.566 (n=26) | 88–93% centrados |
| `v07` | punto medio anclado a la orientación propia del plano | `f1_ref` 0.465 (n=14); recall 0.858 (n=26) | 95–97% centrados |
| `v08` | costura + auto-verificación "no defaults a X=500" | `f1_ref` ≤ 0.06 | **La misma instrucción que ayuda en eje hunde el plano** (4–6% centrados). Consistente con que, en plano, el "acierto" venga de la columna central + pose canónica |
| `v04_1_flowB` | `v04_1` + descripción previa | = `v04_1` | La descripción no aporta |
| `v04_1_flowC` | descripción + pista, sin identidad | `f1_ref` ≤ 0.09; cobertura 19–37% | Pocos puntos útiles por vista (§5.2) |

**Patrón general**: en plano, **cuanto más centrado el prompt, mejor
puntúa** (§5.1: Pearson +0.81 con F1 ajustado). Todo lo que aleja los puntos
de la columna central (bilaterales, "más distantes", `v08`) hunde el
desempeño.

### 4.4 Efecto de `n_views`

- **n=6**: como máximo 1 plano por objeto (cada plano requiere ≥4 vistas
  independientes). Recall limitado (0.08–0.50).
- **n=14**: `f1_ref` crudo más alto (0.41–0.48 en los prompts centrados),
  pero es el `n_views` con **menor cobertura** (59–77%): los objetos
  difíciles fallan y salen del promedio.
- **n=26**: más planos predichos (2.1–2.65 contra 1.19 reales) → precisión
  baja y **`f1_ref` cae a 0.13–0.22**, aunque el recall a 15° se mantiene.
  La caída de `f1_ref` refleja la penalización greedy por sobre-predicción y
  su exigencia de normales/offsets muy precisos, no una pérdida de
  orientación gruesa. Ajustado por cobertura, **n=26 es el mejor `n_views`**
  en 5 de los 9 prompts centrados.

### 4.5 Ablaciones sobre plano (comparación justa, modo multiplano)

| Variante | Δ recall | Δ recall ajust. | Δ precisión | Δ planos pred. | Δ f1_ref | Δ f1_ref_hung. | Δ SDE_ref | Δ n_obj | Δ F1 ajust. | f1_ref mejor / peor |
|---|---|---|---|---|---|---|---|---|---|---|
| EXP-A (mp3) | +0.002 | +0.002 | +0.002 | +0.000 | +0.001 | +0.001 | -0.0006 | +0 | +0.002 | 0 / 1 |
| EXP-C (mp3) | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.0000 | +0 | +0.000 | 0 / 0 |
| EXP-D (mp3) | +0.002 | +0.002 | +0.002 | +0.000 | +0.001 | +0.001 | -0.0006 | +0 | +0.002 | 0 / 1 |
| EXP-F (mp3) | +0.053 | -0.078 | +0.118 | -0.283 | +0.060 | +0.069 | -0.0217 | -281 | +0.011 | 35 / 0 |

- **EXP-A/C/D**: sin efecto (|Δ| ≤ 0.02 en todo; EXP-C idéntico).
  Confirmado ahora con la comparación justa que antes no era posible.
- **EXP-F**: mejora `f1_ref` crudo en 35/42 (0 peores), precisión +0.118,
  `SDE_ref` −0.022. Pero:
  1. **Cobertura −281 objetos en promedio** (−75 a −531). Recall ajustado
     **−0.078** (0 mejores, 23 peores); F1 ajustado +0.011 (neto casi nulo).
     EXP-F cambia cobertura por precisión.
  2. **La mejora de `SDE_ref` es circular**: el gate rechaza justamente los
     candidatos con `SDE_ref` > 0.02.
  3. **Usa la malla para predecir**, así que no es una variante "sin malla".

  **Esto corrige** lo afirmado en `docs/experimentos_pipeline_sin_malla.md`
  §5.4 ("la ablación con evidencia más sólida").

Detalle por prompt de EXP-F:

| Prompt | n_views | cob. base → EXP-F | recall base → EXP-F | recall ajust. base → EXP-F | precisión base → EXP-F | f1_ref base → EXP-F |
|---|---|---|---|---|---|---|
| `plane_v00` | 6 | 98% → 57% | 0.368 → 0.458 | 0.361 → 0.259 | 0.412 → 0.509 | 0.144 → 0.243 |
| `plane_v00` | 14 | 91% → 63% | 0.635 → 0.733 | 0.580 → 0.460 | 0.351 → 0.535 | 0.281 → 0.403 |
| `plane_v00` | 26 | 95% → 60% | 0.595 → 0.625 | 0.567 → 0.378 | 0.331 → 0.429 | 0.133 → 0.206 |
| `plane_v00_1` | 6 | 87% → 49% | 0.293 → 0.512 | 0.254 → 0.249 | 0.316 → 0.550 | 0.148 → 0.261 |
| `plane_v00_1` | 14 | 68% → 55% | 0.816 → 0.814 | 0.559 → 0.447 | 0.477 → 0.671 | 0.442 → 0.480 |
| `plane_v00_1` | 26 | 98% → 65% | 0.763 → 0.742 | 0.751 → 0.479 | 0.424 → 0.519 | 0.171 → 0.262 |
| `plane_v01` | 6 | 100% → 73% | 0.077 → 0.091 | 0.077 → 0.067 | 0.128 → 0.151 | 0.067 → 0.087 |
| `plane_v01` | 14 | 100% → 43% | 0.014 → 0.017 | 0.014 → 0.007 | 0.009 → 0.014 | 0.002 → 0.005 |
| `plane_v01` | 26 | 99% → 59% | 0.016 → 0.018 | 0.016 → 0.011 | 0.010 → 0.015 | 0.001 → 0.002 |
| `plane_v01_1` | 6 | 100% → 74% | 0.073 → 0.090 | 0.072 → 0.067 | 0.125 → 0.157 | 0.072 → 0.092 |
| `plane_v01_1` | 14 | 100% → 45% | 0.016 → 0.018 | 0.016 → 0.008 | 0.011 → 0.013 | 0.001 → 0.003 |
| `plane_v01_1` | 26 | 99% → 60% | 0.012 → 0.015 | 0.012 → 0.009 | 0.008 → 0.013 | 0.001 → 0.001 |
| `plane_v02` | 6 | 93% → 67% | 0.464 → 0.646 | 0.433 → 0.430 | 0.516 → 0.717 | 0.259 → 0.357 |
| `plane_v02` | 14 | 77% → 64% | 0.821 → 0.830 | 0.635 → 0.530 | 0.476 → 0.675 | 0.428 → 0.469 |
| `plane_v02` | 26 | 97% → 70% | 0.753 → 0.764 | 0.734 → 0.538 | 0.484 → 0.586 | 0.217 → 0.304 |
| `plane_v02_1` | 6 | 84% → 52% | 0.369 → 0.591 | 0.311 → 0.309 | 0.407 → 0.649 | 0.203 → 0.327 |
| `plane_v02_1` | 14 | 64% → 54% | 0.808 → 0.822 | 0.517 → 0.441 | 0.467 → 0.674 | 0.414 → 0.457 |
| `plane_v02_1` | 26 | 92% → 63% | 0.737 → 0.732 | 0.676 → 0.464 | 0.455 → 0.553 | 0.207 → 0.301 |
| `plane_v03` | 6 | 99% → 73% | 0.109 → 0.136 | 0.108 → 0.100 | 0.173 → 0.215 | 0.083 → 0.108 |
| `plane_v03` | 14 | 99% → 36% | 0.019 → 0.033 | 0.019 → 0.012 | 0.013 → 0.024 | 0.003 → 0.008 |
| `plane_v03` | 26 | 97% → 47% | 0.013 → 0.018 | 0.012 → 0.008 | 0.009 → 0.013 | 0.003 → 0.004 |
| `plane_v03_1` | 6 | 99% → 70% | 0.103 → 0.131 | 0.102 → 0.092 | 0.166 → 0.209 | 0.077 → 0.103 |
| `plane_v03_1` | 14 | 99% → 38% | 0.025 → 0.053 | 0.025 → 0.020 | 0.015 → 0.046 | 0.007 → 0.020 |
| `plane_v03_1` | 26 | 97% → 50% | 0.011 → 0.013 | 0.010 → 0.007 | 0.007 → 0.011 | 0.002 → 0.002 |
| `plane_v04` | 6 | 76% → 42% | 0.280 → 0.509 | 0.213 → 0.213 | 0.299 → 0.545 | 0.153 → 0.276 |
| `plane_v04` | 14 | 75% → 66% | 0.845 → 0.851 | 0.637 → 0.565 | 0.487 → 0.700 | 0.475 → 0.517 |
| `plane_v04` | 26 | 89% → 52% | 0.819 → 0.754 | 0.729 → 0.395 | 0.438 → 0.515 | 0.163 → 0.259 |
| `plane_v04_1` | 6 | 83% → 43% | 0.214 → 0.410 | 0.178 → 0.178 | 0.229 → 0.440 | 0.115 → 0.217 |
| `plane_v04_1` | 14 | 65% → 56% | 0.854 → 0.853 | 0.556 → 0.474 | 0.518 → 0.719 | 0.482 → 0.521 |
| `plane_v04_1` | 26 | 81% → 47% | 0.830 → 0.776 | 0.671 → 0.362 | 0.426 → 0.508 | 0.162 → 0.262 |
| `plane_v05` | 6 | 99% → 61% | 0.153 → 0.227 | 0.151 → 0.139 | 0.198 → 0.290 | 0.079 → 0.120 |
| `plane_v05` | 14 | 96% → 43% | 0.098 → 0.205 | 0.094 → 0.089 | 0.054 → 0.141 | 0.021 → 0.057 |
| `plane_v05` | 26 | 100% → 43% | 0.082 → 0.153 | 0.082 → 0.066 | 0.045 → 0.086 | 0.026 → 0.054 |
| `plane_v05_1` | 6 | 94% → 72% | 0.502 → 0.644 | 0.470 → 0.460 | 0.563 → 0.712 | 0.271 → 0.345 |
| `plane_v05_1` | 14 | 80% → 63% | 0.679 → 0.765 | 0.546 → 0.483 | 0.380 → 0.598 | 0.324 → 0.415 |
| `plane_v05_1` | 26 | 97% → 60% | 0.505 → 0.546 | 0.491 → 0.329 | 0.292 → 0.395 | 0.118 → 0.192 |
| `plane_v06` | 6 | 86% → 55% | 0.402 → 0.627 | 0.345 → 0.345 | 0.444 → 0.692 | 0.218 → 0.340 |
| `plane_v06` | 14 | 62% → 54% | 0.830 → 0.841 | 0.518 → 0.451 | 0.487 → 0.691 | 0.435 → 0.484 |
| `plane_v06` | 26 | 97% → 67% | 0.751 → 0.739 | 0.726 → 0.492 | 0.464 → 0.556 | 0.205 → 0.294 |
| `plane_v07` | 6 | 77% → 39% | 0.216 → 0.428 | 0.166 → 0.166 | 0.233 → 0.462 | 0.112 → 0.220 |
| `plane_v07` | 14 | 59% → 47% | 0.846 → 0.857 | 0.499 → 0.404 | 0.499 → 0.696 | 0.465 → 0.508 |
| `plane_v07` | 26 | 83% → 47% | 0.858 → 0.808 | 0.716 → 0.378 | 0.443 → 0.541 | 0.164 → 0.269 |

### 4.6 Modo un-solo-plano (`max_planes=1`) — métricas angulares

Salida de EXP-C con `max_planes=1`, equivalente al baseline de un plano
(este modo sí imputa 90° a los faltantes, como el eje).

| Prompt | n_views | n_obj | error medio | mediana | AUC@45 | P@10° | P@15° |
|---|---|---|---|---|---|---|---|
| `plane_v00` | 6 | 833 | 36.3 | 27.9 | 0.418 | 0.298 | 0.404 |
| `plane_v00` | 14 | 777 | 38.1 | 29.3 | 0.429 | 0.421 | 0.444 |
| `plane_v00` | 26 | 810 | 42.3 | 41.9 | 0.286 | 0.171 | 0.207 |
| `plane_v00_1` | 6 | 737 | 39.7 | 40.0 | 0.301 | 0.254 | 0.274 |
| `plane_v00_1` | 14 | 582 | 43.8 | 40.9 | 0.425 | 0.438 | 0.452 |
| `plane_v00_1` | 26 | 837 | 34.5 | 41.8 | 0.313 | 0.252 | 0.271 |
| `plane_v01` | 6 | 846 | 71.1 | 85.9 | 0.135 | 0.114 | 0.127 |
| `plane_v01` | 14 | 846 | 69.2 | 70.9 | 0.024 | 0.007 | 0.008 |
| `plane_v01` | 26 | 840 | 81.2 | 83.5 | 0.012 | 0.000 | 0.014 |
| `plane_v01_1` | 6 | 847 | 71.1 | 85.7 | 0.135 | 0.111 | 0.125 |
| `plane_v01_1` | 14 | 847 | 68.3 | 69.3 | 0.027 | 0.005 | 0.009 |
| `plane_v01_1` | 26 | 844 | 81.2 | 83.4 | 0.011 | 0.000 | 0.015 |
| `plane_v02` | 6 | 792 | 30.4 | 33.3 | 0.464 | 0.473 | 0.481 |
| `plane_v02` | 14 | 658 | 37.7 | 12.7 | 0.482 | 0.485 | 0.520 |
| `plane_v02` | 26 | 828 | 31.6 | 39.5 | 0.390 | 0.319 | 0.353 |
| `plane_v02_1` | 6 | 717 | 38.7 | 40.0 | 0.352 | 0.333 | 0.344 |
| `plane_v02_1` | 14 | 544 | 46.2 | 58.8 | 0.411 | 0.420 | 0.446 |
| `plane_v02_1` | 26 | 779 | 37.6 | 41.9 | 0.329 | 0.269 | 0.291 |
| `plane_v03` | 6 | 843 | 68.0 | 85.3 | 0.170 | 0.140 | 0.172 |
| `plane_v03` | 14 | 840 | 67.0 | 69.1 | 0.028 | 0.005 | 0.009 |
| `plane_v03` | 26 | 823 | 81.0 | 83.0 | 0.010 | 0.001 | 0.009 |
| `plane_v03_1` | 6 | 838 | 68.6 | 85.6 | 0.163 | 0.138 | 0.164 |
| `plane_v03_1` | 14 | 841 | 66.4 | 68.5 | 0.035 | 0.014 | 0.020 |
| `plane_v03_1` | 26 | 826 | 80.9 | 83.1 | 0.011 | 0.001 | 0.008 |
| `plane_v04` | 6 | 648 | 45.7 | 40.0 | 0.257 | 0.224 | 0.228 |
| `plane_v04` | 14 | 641 | 34.7 | 4.9 | 0.539 | 0.567 | 0.586 |
| `plane_v04` | 26 | 756 | 41.2 | 43.3 | 0.237 | 0.181 | 0.192 |
| `plane_v04_1` | 6 | 707 | 44.2 | 40.0 | 0.233 | 0.186 | 0.191 |
| `plane_v04_1` | 14 | 553 | 43.3 | 26.1 | 0.450 | 0.481 | 0.494 |
| `plane_v04_1` | 26 | 687 | 44.9 | 43.6 | 0.218 | 0.173 | 0.180 |
| `plane_v05` | 6 | 839 | 64.2 | 81.7 | 0.198 | 0.151 | 0.195 |
| `plane_v05` | 14 | 812 | 62.4 | 68.1 | 0.108 | 0.059 | 0.094 |
| `plane_v05` | 26 | 848 | 77.9 | 79.8 | 0.015 | 0.000 | 0.012 |
| `plane_v05_1` | 6 | 796 | 34.3 | 10.0 | 0.490 | 0.500 | 0.527 |
| `plane_v05_1` | 14 | 684 | 40.0 | 24.0 | 0.445 | 0.438 | 0.472 |
| `plane_v05_1` | 26 | 827 | 40.1 | 41.5 | 0.281 | 0.162 | 0.194 |
| `plane_v06` | 6 | 730 | 36.7 | 40.0 | 0.382 | 0.372 | 0.381 |
| `plane_v06` | 14 | 531 | 46.4 | 56.1 | 0.414 | 0.426 | 0.452 |
| `plane_v06` | 26 | 822 | 33.7 | 41.4 | 0.346 | 0.285 | 0.302 |
| `plane_v07` | 6 | 652 | 47.1 | 40.0 | 0.218 | 0.162 | 0.179 |
| `plane_v07` | 14 | 501 | 49.7 | 60.7 | 0.373 | 0.395 | 0.401 |
| `plane_v07` | 26 | 709 | 44.0 | 43.7 | 0.226 | 0.187 | 0.195 |
| `plane_v08` | 6 | 841 | 66.1 | 81.6 | 0.170 | 0.120 | 0.164 |
| `plane_v08` | 14 | 844 | 61.7 | 67.1 | 0.100 | 0.054 | 0.080 |
| `plane_v08` | 26 | 844 | 76.8 | 79.6 | 0.023 | 0.002 | 0.009 |
| `plane_v04_1_flowB` | 6 | 668 | 44.8 | 40.0 | 0.260 | 0.224 | 0.232 |
| `plane_v04_1_flowB` | 14 | 578 | 41.0 | 11.8 | 0.473 | 0.494 | 0.518 |
| `plane_v04_1_flowB` | 26 | 755 | 42.8 | 43.5 | 0.182 | 0.128 | 0.135 |
| `plane_v04_1_flowC` | 6 | 317 | 78.3 | 90.0 | 0.089 | 0.065 | 0.085 |
| `plane_v04_1_flowC` | 14 | 163 | 84.1 | 90.0 | 0.025 | 0.011 | 0.017 |
| `plane_v04_1_flowC` | 26 | 169 | 85.5 | 90.0 | 0.014 | 0.001 | 0.004 |
| `plane_v04_1_6pts` | 6 | 649 | 46.2 | 40.0 | 0.247 | 0.213 | 0.213 |
| `plane_v04_1_6pts` | 14 | 425 | 52.9 | 90.0 | 0.364 | 0.374 | 0.394 |
| `plane_v04_1_6pts` | 26 | 294 | 72.2 | 90.0 | 0.078 | 0.042 | 0.048 |
| `plane_v05_6pts` | 6 | 806 | 31.1 | 39.9 | 0.430 | 0.426 | 0.431 |
| `plane_v05_6pts` | 14 | 724 | 27.5 | 4.6 | 0.615 | 0.662 | 0.672 |
| `plane_v05_6pts` | 26 | 771 | 42.8 | 43.5 | 0.174 | 0.120 | 0.123 |

- **Distribución bimodal**: medianas muy por debajo de la media (p. ej.
  `plane_v05_6pts` n=14: mediana 4.6°, media 27.5°; `plane_v04` n=14: 4.9°
  vs 34.7°). Una masa de objetos casi exactos (normal ≈ eje X del mundo, que
  coincide con el GT canónico) y otra cerca de 90°.
- **Huella de una predicción que no depende del objeto**: a n=6, la
  **mediana es exactamente 40.0°** en 8 prompts distintos (`v00_1`, `v02_1`,
  `v04`, `v04_1`, `v06`, `v07`, `flowB`, `v04_1_6pts`; 39.9° en `v05_6pts`).
  Con los mismos 6 cámaras para todos los objetos y puntos en la columna
  central, la mayoría de los objetos recibe el mismo plano predicho. Errores
  continuos que dependan del objeto no producirían la misma mediana exacta
  en 8 configuraciones. (Inferencia desde el CSV agregado; se confirmaría
  mirando las normales predichas por objeto en el servidor.)

### 4.7 Otras estadísticas de plano

- **Sobre-predicción de planos**: 1.9–2.65 predichos contra 1.19 reales a
  n ≥ 14 en todos los prompts.
- **`SDE_ref`**: 0.012–0.015 en los prompts centrados, 0.04–0.08 en los
  bilaterales y en `v05`. No usa GT (es autoconsistencia contra la malla) y
  sigue el mismo patrón que el centrado: bajo donde el desempeño es alto.
- **Sanity check del GT**: `n_true_planes_mean` ≈ 1.16–1.20 en todas las
  filas, consistente con 692×1 + 152×2 + 6×3 = 1.193 planos/objeto.

---

## 5. Diagnósticos

### 5.1 Colapso a X≈500 (`center_bias`)

% de puntos no cenitales con |x − 500| < 30 (en n=1 la única vista es
cenital, por lo que esa columna muestra el % total). Veredicto del script:
>50% "PROBLEMA CONFIRMADO", >20% "problema parcial".

**Eje:**

| Prompt | n=1 | n=6 | n=14 | n=26 | veredicto (n=26) |
|---|---|---|---|---|---|
| `axis_lit2_grid` | 8% | 32% | 33% | 34% | problema parcial |
| `axis_lit3_cot` | 27% | 27% | 33% | 30% | problema parcial |
| `axis_v00` | 13% | 52% | 53% | 50% | PROBLEMA CONFIRMADO |
| `axis_v00_1` | 70% | 77% | 78% | 80% | PROBLEMA CONFIRMADO |
| `axis_v01` | 10% | 18% | 19% | 14% | sin problema generalizado |
| `axis_v01_1` | 9% | 19% | 20% | 14% | sin problema generalizado |
| `axis_v02` | 7% | 11% | 13% | 11% | sin problema generalizado |
| `axis_v02_1` | 65% | 66% | 68% | 70% | PROBLEMA CONFIRMADO |
| `axis_v03` | 11% | 23% | 25% | 22% | problema parcial |
| `axis_v03_1` | 10% | 23% | 26% | 23% | problema parcial |
| `axis_v04` | 44% | 39% | 39% | 39% | problema parcial |
| `axis_v04_1` | 50% | 46% | 47% | 46% | problema parcial |
| `axis_v05` | 52% | 63% | 66% | 69% | PROBLEMA CONFIRMADO |
| `axis_v05_1` | 65% | 76% | 81% | 86% | PROBLEMA CONFIRMADO |
| `axis_v05_1_flowB` | 69% | 77% | 84% | 91% | PROBLEMA CONFIRMADO |
| `axis_v05_1_flowC` | 49% | 51% | 55% | 65% | PROBLEMA CONFIRMADO |
| `axis_v06` | — | 44% | 45% | 43% | problema parcial |
| `axis_v06_6pts` | 60% | 67% | 68% | 70% | PROBLEMA CONFIRMADO |
| `axis_v06_nomesh_filtered` | — | 44% | 45% | 43% | problema parcial |
| `axis_v07` | — | 72% | 68% | 68% | PROBLEMA CONFIRMADO |
| `axis_v07_6pts` | 57% | 63% | 65% | 62% | PROBLEMA CONFIRMADO |
| `axis_v08` | 6% | 18% | 20% | 18% | sin problema generalizado |

**Plano:**

| Prompt | n=1 | n=6 | n=14 | n=26 | veredicto (n=26) |
|---|---|---|---|---|---|
| `plane_v00` | 4% | 41% | 56% | 60% | PROBLEMA CONFIRMADO |
| `plane_v00_1` | 91% | 91% | 91% | 93% | PROBLEMA CONFIRMADO |
| `plane_v01` | 3% | 5% | 6% | 4% | sin problema generalizado |
| `plane_v01_1` | 3% | 6% | 7% | 4% | sin problema generalizado |
| `plane_v02` | 93% | 89% | 91% | 94% | PROBLEMA CONFIRMADO |
| `plane_v02_1` | 95% | 87% | 84% | 89% | PROBLEMA CONFIRMADO |
| `plane_v03` | 5% | 10% | 10% | 7% | sin problema generalizado |
| `plane_v03_1` | 5% | 10% | 11% | 7% | sin problema generalizado |
| `plane_v04` | 99% | 94% | 94% | 98% | PROBLEMA CONFIRMADO |
| `plane_v04_1` | 99% | 96% | 97% | 99% | PROBLEMA CONFIRMADO |
| `plane_v04_1_6pts` | 93% | 94% | 91% | 94% | PROBLEMA CONFIRMADO |
| `plane_v04_1_flowB` | 99% | 96% | 97% | 99% | PROBLEMA CONFIRMADO |
| `plane_v04_1_flowC` | 31% | 47% | 66% | 66% | PROBLEMA CONFIRMADO |
| `plane_v04_1_nomesh_filtered` | 99% | 96% | 97% | 99% | PROBLEMA CONFIRMADO |
| `plane_v05` | 5% | 4% | 6% | 4% | sin problema generalizado |
| `plane_v05_1` | 81% | 62% | 72% | 73% | PROBLEMA CONFIRMADO |
| `plane_v05_6pts` | 99% | 92% | 95% | 93% | PROBLEMA CONFIRMADO |
| `plane_v06` | — | 90% | 88% | 93% | PROBLEMA CONFIRMADO |
| `plane_v07` | — | 96% | 95% | 97% | PROBLEMA CONFIRMADO |
| `plane_v08` | 4% | 5% | 6% | 4% | sin problema generalizado |

**Correlación con el desempeño** (57 filas prompt × `n_views` por tipo,
baseline o proxy EXP-C):

| Tipo | Métrica | Pearson | Spearman | Lectura |
|---|---|---|---|---|
| Eje | error angular medio | +0.13 | −0.10 | sin relación (todo está al azar) |
| Eje | AUC | −0.12 | 0.00 | sin relación |
| Eje | traslación normalizada | **+0.72** | **+0.79** | más centrado → ancla peor condicionada |
| Plano | recall ajustado | **+0.75** | **+0.72** | más centrado → mejor |
| Plano | F1 ajustado | **+0.81** | **+0.74** | más centrado → mejor |
| Plano | `f1_ref` | **+0.67** | **+0.67** | más centrado → mejor |

El mismo comportamiento de Molmo2 (marcar la columna central) es **inútil
en eje y "exitoso" en plano**. La explicación que reconcilia ambos
resultados es la orientación del GT (§2.2), no una diferencia en la
capacidad del modelo entre tareas.

### 5.2 Predicciones faltantes (`plane_v04_1_flowC_expF_missing.csv`)

2415 filas (604 objetos × 4 `n_views`), **todas** con
`detect_planes returned zero accepted planes` y categoría
`insufficient_data`; 0 excepciones inesperadas. Los faltantes son escasez
real de pares de puntos válidos (Flujo C sin identidad entre vistas), no
errores de código. (Generado con
`Pipeline_Experiments/diagnostics/diagnose_missing_reason.py`.)

### 5.3 Referencias triviales (muestra local; ejecutar en el servidor para el dataset completo)

`Pipeline_Experiments/diagnostics/trivial_baselines.py` sobre los 30+30
objetos locales, con las funciones de `evaluate.py`:

| Predictor trivial | Eje: error medio / AUC / P@10° | Plano: recall / precisión / `f1_ref` / `f1_ref_hung.` |
|---|---|---|
| azar uniforme (analítico) | 57.30° / 0.0997 / 0.0152 | — |
| siempre X | 55.10° / 0.128 / 0.033 | 0.767 / 0.767 / **0.742** / 0.754 |
| siempre Y | 58.53° / 0.163 / 0.167 | 0.000 / 0.000 / 0.000 / 0.000 |
| siempre Z | 60.07° / 0.094 / 0.000 | 0.217 / 0.233 / 0.258 / 0.230 |
| siempre {X, Z} | — | 0.983 / 0.500 / 0.886 / 0.659 |
| siempre {X, Y, Z} | — | 0.983 / 0.333 / 0.795 / 0.496 |

Y el pipeline real sobre esos mismos 30 objetos de plano (`plane_v04_1`,
re-ejecutado localmente): recall 0.645 / precisión 0.400 a n=14 agregando
TP/GT/predichos globalmente; `sandbox_pipeline_server_final.ipynb` reporta
0.667 / 0.483 promediando por objeto (la misma convención que `evaluate.py`
y que la tabla de arriba). **Con cualquiera de las dos convenciones, el
pipeline de plano queda por debajo de "siempre X" en la muestra**.

---

## 6. Recomendaciones (en orden de prioridad)

1. **Correr `trivial_baselines.py` sobre el dataset completo** (segundos, sin
   GPU):
   ```bash
   python Pipeline_Experiments/diagnostics/trivial_baselines.py --objects-root ../data/objects
   ```
   Decide si los resultados de plano superan a un predictor sin información
   en los 850 objetos. Es el número de referencia que falta para cualquier
   afirmación de la tesis sobre plano.
2. **Reportar siempre contra el azar (eje) y contra el trivial (plano)**,
   no solo entre prompts.
3. **Imputar los faltantes en el resumen multiplano** de
   `Mapping/evaluate.py::compute_summary` (recall 0; FN = planos GT del
   objeto en `f1_counts_ref`), o como mínimo reportar la cobertura junto a
   cada número. Hoy comparar filas con 19% y 100% de cobertura no es válido.
4. **Quitar el atajo de pose canónica en plano**: evaluar sobre objetos de
   plano rotados al azar (como ya están los de eje), o con un "arriba" de
   cámara aleatorio por vista. Sin eso, un buen resultado de plano no
   distingue entre "entiende la simetría" y "marca el centro de la imagen".
5. **Dar a los prompts bilaterales un estimador compatible**: en eje, un
   modo "midpoint" (el punto medio del par sí está sobre el eje, y hacen
   falta ≥2 pares por vista para una recta); en plano, usar P−P' como
   estimación directa de la normal en vez de tratarlo como una recta
   contenida en el plano.
6. **Seguir la única pista positiva en eje: `axis_v08`**. Correr su baseline
   propio (hoy solo hay proxy EXP-C) y explorar variantes que reduzcan aún
   más el colapso a X≈500.
7. **Pausar nuevas ablaciones de triangulación (tipo A–E)**: no pueden
   mejorar una entrada sin señal direccional. El cuello de botella está en
   los puntos de Molmo2, no en la triangulación.
8. Menores: re-correr `Mapping/compare_results_no_mesh.py` para incluir
   `axis_lit2_grid`/`axis_lit3_cot`; deduplicar `*_nomesh_filtered`
   (confirmado de nuevo: números idénticos a `axis_v06`/`plane_v04_1`);
   borrar `experiments_27_08_2026_v2` (copia exacta).

---

## Apéndice — cómo se calculó cada cosa

| Qué | Cómo |
|---|---|
| Tablas de métricas | Script sobre `experiments_15_09_2026/axis_sym_nomesh_comparison.csv` y `experiments_22_09_2026/plane_sym_nomesh_comparison.csv` |
| Deltas de ablaciones | Pareadas por (prompt, `n_views`) contra la fila baseline del mismo prompt; 14 prompts × 3 `n_views` = 42 pares |
| Recall/F1 ajustados | recall × n_obj/850; F1 = media armónica con la precisión sin ajustar |
| z vs azar | (media − azar imputado) / (21.56·√n/850) |
| Colapso a +Y y a ejes del mundo | Re-ejecución de `widest_pair` + `interpretation_plane_normal` + `triangulate_line` sobre `molmo_multiview_axis_v06.json` de los 30 objetos de `Experiments/renders/axis_sym/` |
| Pipeline de plano en la muestra | `estimate_symmetry_no_mesh.detect_planes_no_mesh(max_planes=3)` sobre `molmo_multiview_plane_v04_1.json` de los 30 objetos de `Experiments/renders/plane_sym/` |
| Predictores triviales | `Pipeline_Experiments/diagnostics/trivial_baselines.py` (usa `angular_error_deg`, `auc_from_errors`, `evaluate_plane_multiset`, `f1_match_counts[_hungarian]` de `evaluate.py`) |
| Correlaciones centrado ↔ desempeño | `pct_noncenital_centered` de `results/diagnostics/*_center_bias_summary.csv` contra la fila baseline (o proxy EXP-C) del mismo prompt y `n_views` |
