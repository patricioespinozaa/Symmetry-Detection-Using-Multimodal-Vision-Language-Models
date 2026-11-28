# Plan de reestructuración del repositorio

Fecha: 2026-11-27. Rama de trabajo sugerida: `refactor`.

Este plan deja en el repositorio solo lo que hace falta para reproducir los resultados de la
tesis. Eso incluye el código del método final y de cada experimento reportado, los CSV que leen
los generadores de tablas, la documentación vigente y el ejemplo mínimo. Se revisaron los 130
`.py` (docstring, imports locales y quién los llama) y las carpetas de resultados solo por nombre.
`Tesis_DCC_MDS/` y `Prompted_Geometry__.../` quedan fuera de la revisión.

**Tamaño actual:** 17 165 archivos versionados. Unos 15 400 (≈90 %) se pueden borrar sin
afectar ninguna tabla del capítulo 4 ni ningún anexo. Casi todo ese volumen sale de
`Experiments/` (14 437 archivos de renders y notebooks del sandbox de agosto).

---

## 1. Criterios

1. **Se conserva**:
   - el código del método final: renders, inferencia con los 3 VLMs, triangulación, EXP-E en
     eje, puntaje $S$ en plano, evaluación y comparación;
   - el código de cada experimento con tabla en la tesis: resolución, iluminación, flujos, modo de
     envío, VLMs, malla, silueta aleatoria, multiplano, rotaciones SO(3) y Aguirre, oráculo,
     triviales, azar, centro, repetidos, categorías y vistas de canto;
   - todo lo que leen `utils/generar_tablas_resultados_cap4.py`,
     `utils/verificar_tablas_resultados_cap4.py` y `utils/generar_anexo_prompts.py`.
2. **Se borra**:
   - lo que corresponde a etapas superadas: resultados anteriores a la corrección NDC, el
     sandbox de agosto, generadores de tablas de borradores anteriores y ablaciones no
     reportadas;
   - copias duplicadas;
   - logs sueltos en la raíz;
   - fragmentos de la tesis que quedaron versionados por error.
3. **Los nombres de las carpetas de código no cambian** en esta reestructuración, por tres
   razones:
   - hay 83 `sys.path.insert(...)` que dependen de la profundidad y del nombre de cada carpeta,
     por ejemplo `REPO / "Mapping"` o `parent.parent`;
   - los scripts del servidor llaman rutas como `Mapping/evaluate.py` o
     `PlaneScoreData/run_plane_malla.py`;
   - el anexo de prompts de la tesis cita rutas como `MolmoPointing/prompts/axis/v06_multi.txt`.

   Renombrar o anidar en un `src/` es posible, pero cuesta mucho más de lo que aporta. Queda como
   fase opcional en la sección 7.

---

## 2. Estructura final propuesta

```
.
├── README.md                     # actualizar (sección 6)
├── CLAUDE.md                     # actualizar (sección 6)
├── environment.yml
├── .gitignore                    # ampliar (sección 5)
│
├── docs/
│   ├── Contexto.md  metricas_evaluacion.md  data-schemas.md  code-norms.md
│   ├── actualizacion_metricas.md  hallazgo_seleccion_normal_plano.md  plan_rotacion_aguirre.md
│   ├── pipeline_sin_malla.md  implementacion_pipeline_sin_malla.md
│   ├── implementacion_pipeline_experiments.md
│   ├── instalacion_molmopoint_qwen3vl.md        ← guia_instalacion_molmopoint_qwen3vl.md (raíz)
│   ├── plan_reestructuracion_repo.md            (este archivo; borrar al terminar)
│   ├── features/   audits/
│   ├── literature/ (+ metricas_enhanced_backprojection_3D_symmetry.md, desde la raíz)
│   └── archive/    (+ informes de etapas anteriores, sección 4)
│
├── Examples/                     # 2 objetos de ejemplo, solo renders 448 px (los usa verificar_convencion_ndc.py)
│   ├── objects/  axis_sym_obj.{obj,txt}  plane_sym_obj.{obj,txt}
│   └── renders/{axis_sym,plane_sym}/<obj>/448/flat/
│
├── ImagesGenerator/              # data_render.py, export_fibonacci_views.py (fusionados con _subset), README.md
├── MolmoPointing/                # molmo_multiview_runner.py (fusionado), prompts_registry.py, prompts/, README.md
├── QwenPointing/                 # qwen3vl_multiview_runner.py (fusionado), prompts_registry.py, prompts/, experiments_04_11_2026/
├── MolmoPointPointing/           # molmopoint_multiview_runner.py (fusionado), prompts_registry.py, prompts/, experiments_04_11_2026/
├── pipeline_common/              # sin cambios
│
├── Mapping/
│   ├── estimate_symmetry_no_mesh.py  evaluate.py  compare_results_no_mesh.py      # núcleo sin malla
│   ├── map_to_3d.py  estimate_symmetry.py  compare_results.py                     # con malla (run_*_malla.py)
│   ├── visualize_rays.py  visualize_symmetry.py  export_symmetry_overlay.py  render_symmetry_comparison.py
│   └── README.md
│
├── Pipeline_Experiments/
│   ├── estimate_symmetry_variants.py  config.py  view_filtering.py
│   ├── triangulation_variants/  __init__.py  _shared.py  exp_e_iterative_reweight.py
│   ├── diagnostics/  diagnose_center_bias.py  diagnose_repeated_points.py  trivial_baselines.py  diagnose_missing_reason.py
│   └── README.md
│
├── PlaneScoreData/               # estimador S, run_*.py de cada experimento, diagnostico_*.py, auditoría, pruebas
├── RandomBaseline/               # sin latex_tables.py; outputs/random_baseline_reference.csv
├── ExploratoryDataAnalysis/      # scripts, EDA.ipynb, categorias_objetos*.csv, revision_ambiguos.csv, figuras/
├── InteractiveViewer/            # sin cambios
│
├── utils/
│   ├── generar_tablas_resultados_cap4.py  verificar_tablas_resultados_cap4.py  generar_anexo_prompts.py
│   ├── fig_camaras.py  copiar_vistas_degeneradas.py  verificar_convencion_ndc.py
│   ├── rerun_ndc_fix.sh          # documenta cómo se crearon los ids *_ndcfix en el servidor
│   └── muestras_render.sh
│
├── imgs/vistas_degeneradas/      # fuente de la figura 3.4
│
└── resultados_finales/           # lo que leen las tablas; sin planar_sdat/
    ├── <una carpeta por experimento>/
    ├── diagnosticos/  referencias/
    └── logs/
        ├── (logs de post-proceso actuales)
        ├── auditoria_noviembre.log           ← raíz
        └── inferencia/                       ← logs de Molmo2/Qwen/MolmoPoint hoy sueltos en la raíz
            ├── 224_flat_vlms/                ← logs/{molmopoint,qwen3vl}_{axis,plane}.log, render_rot_so3_224_flat.log
            ├── 224_flat_v06/                 ← axis_v06_gpu0.log, axis_v06_gpu1.log
            ├── 448_flat/                     ← logs_448_flat/molmo_*.log
            ├── 1136_flat/                    ← logs_1136_flat/molmo_*.log
            ├── 224_iluminacion/              ← logs_224_iluminacion/*
            ├── 224_single/                   ← logs_single_oct26/*
            ├── 224_flujoB/                   ← logs_flujoB/*
            └── silueta/                      ← logs_silueta/*.log
```

---

## 3. Listado de cosas a borrar

Columnas:
- **Ruta**: el archivo o la carpeta a borrar.
- **Archivos**: cuántos archivos versionados contiene.
- **Motivo**: por qué ya no hace falta.
- **Antes de borrar**: lo que hay que ajustar primero, cuando corresponde.

### 3.1 Raíz del repositorio

| Ruta | Archivos | Motivo | Antes de borrar |
|---|---|---|---|
| `cap4_originañ.tex` | 1 | Borrador del capítulo 4 versionado en un repo público | Ver sección 8 (historial) |
| `cap4_baselines.tex` | 1 | Fragmento de la tesis versionado en un repo público | Ver sección 8 |
| `analisis_resultados_22_07.ipynb` | 1 | Análisis de julio, anterior a la corrección NDC y al método final | — |
| `plane_v04_1_flowC_expF_missing.csv` | 1 | Diagnóstico puntual de EXP-F, que no se reporta | — |
| `logs_multiplano.txt` | 1 | Copia de consola; el log vigente es `resultados_finales/logs/multiplano_v02.log` | — |
| `experiments_27_08_2026_v1/` | 2 | CSV de agosto, anteriores a la corrección NDC | — |
| `experiments_27_08_2026_v2/` | 2 | Ídem | — |
| `experiments_15_09_2026/` | 2 | Ídem | — |
| `logs_robustez/` | 3 | Experimento de robustez (v42/v62/v86) sin resultados en la tesis | — |
| `logs/eval_*.log`, `logs/nomesh_*.log` | 9 | Post-proceso anterior al método final, rehecho en `resultados_finales/*/logs` | Mover antes los logs de inferencia (sección 4) |
| `logs_448_flat/postproceso_*.log`, `logs_1136_flat/postproceso.log` | 3 | Ídem | Ídem |
| `logs_silueta/silueta.pid` | 1 | Archivo de proceso, no es un resultado | — |

### 3.2 Carpetas completas de etapas superadas

| Ruta | Archivos | Motivo | Antes de borrar |
|---|---|---|---|
| `Experiments/` | 14 437 | Sandbox de agosto:<br>- 60+60 objetos con sus renders;<br>- 8 notebooks de prueba;<br>- PNG de ablaciones (`lit1`, `lit2`, `sweep_*`, `heatmap_*`).<br>Ningún `.py` lo lee. | Borrar también `docs/inventario_Experiments.md` (o pasarlo a `docs/archive/`) |
| `ndc_fix/` | 183 | Primera corrida tras la corrección NDC (octubre), rehecha con EXP-E y $S$ en `resultados_finales/`. Los pareados contra esa base ya están guardados en `resultados_finales/*/pareado/`. | Ajustar `csv_base()` en `PlaneScoreData/run_resultados_finales.sh`, que todavía lista `ndc_fix/<carpeta>/experiments_*` (sección 5.4) |
| `ndc_fix_axial_expE/` | 6 | Prueba inicial de EXP-E, superada por `resultados_finales/` | — |
| `historial/` | 13 | `reference_metrics_*_all.csv` antiguos, `scratch/`, auditoría de índices de vistas de agosto y su notebook | Quitar la mención a `analisis_tiempos_postprocesamiento.ipynb` en `CLAUDE.md` |
| `resultados_finales/planar_sdat/` | 60 | Copia de los CSV planares hecha por `armar_planar_sdat.py` para tablas de un borrador anterior. Las tablas finales leen las carpetas originales. | Borrar `utils/armar_planar_sdat.py` en el mismo commit |

### 3.3 Código obsoleto

**ImagesGenerator, MolmoPointing, QwenPointing, MolmoPointPointing**

| Ruta | Motivo | Antes de borrar |
|---|---|---|
| `ImagesGenerator/data_render_subset.py`, `export_fibonacci_views_subset.py` | Copias del original con `--subset-n-views` y `--roll-seed` | Fusionar con el original (sección 5.1) y cambiar las llamadas en `run_plane_v02_224_roll.py`, `_rot_aguirre.py` y `_rot_so3.py` |
| `MolmoPointing/molmo_multiview_runner_subset.py` | Copia con 24 líneas distintas. Si el render no tiene `fibonacci_total`, el cambio cae al comportamiento original. | Fusionar (sección 5.1) y cambiar las llamadas en `run_plane_v02_224_rot_so3.py` y `_rot_so3_vlms.py` |
| `QwenPointing/qwen3vl_multiview_runner_subset.py` | Ídem | Ídem en `run_plane_v02_224_rot_so3_vlms.py` |
| `MolmoPointPointing/molmopoint_multiview_runner_subset.py` | Ídem | Ídem |
| `MolmoPointing/molmo2_multiview_experiment.ipynb` | Notebook de prueba inicial de Molmo2 | — |
| `MolmoPointing/Experiments.md`, `PROMPT_IMPROVEMENTS_v1.md` | Notas del sweep con malla de julio y agosto | Pasar a `docs/archive/` si se quieren conservar |
| `MolmoPointing/ContextLimits/` (2 `.py`, 1 `.md`, `resultados/`) | Análisis de largo de contexto. Sus tablas no están en la tesis y `verificar_formato_tracks.py` no tiene referencias. | Borrar junto con `utils/generar_tablas_tokens.py` y `utils/rerun_ndcfix_completo.py` |
| `MolmoPointing/prompts/axis/lit2_grid_*`, `lit3_cot_*` | Prompts de los experimentos de literatura que no se reportan ni aparecen en el anexo C | — |
| `MolmoPointing/prompts/{axis,plane}/*_6pts_*` (10 archivos) | Prompts de 6 puntos, quitados del anexo de prompts | — |
| `QwenPointing/verificar_puntos_qwen.ipynb` | Verificación puntual del formato de salida de Qwen | — |

**Mapping**

| Ruta | Motivo | Antes de borrar |
|---|---|---|
| `Mapping/archive/` | Ya archivado | — |
| `Mapping/check_independent_vs_midpoint.py`, `check_origin_full.py`, `check_sde_vs_angular.py` | Comprobaciones de agosto sobre el modo midpoint y la SDE vieja (eliminada) | — |
| `Mapping/diagnose_axis_conditioning.py`, `diagnose_localization_vs_error.py`, `diagnose_point_localization.py`, `diagnose_view_geometry.py` | Diagnósticos del pipeline con malla de agosto. Los diagnósticos reportados están en `PlaneScoreData/diagnostico_*.py`. | Pasar `docs/diagnostico_conditioning_axis.md` a `docs/archive/` |
| `Mapping/select_sandbox_objects.py`, `export_viz_samples.py` | Arman el sandbox de `Experiments/` | `export_symmetry_overlay.py` solo lo nombra en su docstring; ajustar ese texto |
| `Mapping/audit_view_indices_v2.py` | Auditoría de agosto, cuyo resultado está en `historial/` (que también se borra) | — |
| `Mapping/run_all_postprocessing.py` | Orquestador del sweep con malla de agosto, reemplazado por `PlaneScoreData/run_{axis,plane}_malla.py` | Actualizar `README.md` |
| `Mapping/cleanup_experiments.py`, `cleanup_all_experiments.sh`, `find_extra_renders.sh` | Limpieza del servidor durante el sweep. No producen resultados. | Actualizar `README.md` y `Mapping/README.md` |

**Pipeline_Experiments**

| Ruta | Motivo | Antes de borrar |
|---|---|---|
| `Pipeline_Experiments/triangulation_variants/exp_a_weighted.py`, `exp_b_*`, `exp_c_*`, `exp_d_*`, `exp_f_*` | Ablaciones no reportadas. La tesis solo compara SVD frente a EXP-E (`tab:res_estimador_eje`). | Dejar solo `expE` en `VARIANTS` de `triangulation_variants/__init__.py` (sección 5.2) |
| `Pipeline_Experiments/candidates_then_select/` | EXP-LIT-1, no reportado | — |
| `Pipeline_Experiments/run_batch.py`, `configs/*.yaml`, `list_prompts.py`, `audit_results.py` | Orquestan y auditan el sweep de ablaciones. Los experimentos finales se corren con `PlaneScoreData/run_*.py` y `run_resultados_finales.sh`. | Quitar `RunConfig` y sus helpers de `config.py` si solo los usa `run_batch.py`. `VariantConfig`, `build_output_experiment_id` y `discover_experiment_ids` se quedan porque los usan `estimate_symmetry_variants.py` y `diagnostics/`. |

**utils, RandomBaseline y ExploratoryDataAnalysis**

| Ruta | Motivo | Antes de borrar |
|---|---|---|
| `utils/generar_tablas_latex.py`, `_noviembre.py`, `_iluminacion.py`, `_modo_envio.py`, `_planar_sdat_v02.py`, `_seleccion.py`, `_tokens.py`, `_vlms.py` | Generadores de tablas de borradores anteriores, que leen `ndc_fix/` o `planar_sdat/`. Las tablas actuales las genera `generar_tablas_resultados_cap4.py`. | Borrar todos juntos porque se importan entre sí |
| `utils/generar_resumen_resultados.py` | Resúmenes en Markdown desde `ndc_fix/` para `docs/resultados/` | Pasar `docs/resultados/` a `docs/archive/` |
| `utils/armar_planar_sdat.py` | Arma `resultados_finales/planar_sdat/`, que se borra | — |
| `utils/rerun_ndcfix_completo.py` | Re-proceso de octubre hacia `ndc_fix/`, reemplazado por `run_resultados_finales.sh` | — |
| `RandomBaseline/latex_tables.py` | Tablas LaTeX del azar con el estilo del generador viejo. La tesis lee `outputs/random_baseline_reference.csv`. | Quitar `import latex_tables` y el subcomando que lo usa en `run_baseline.py` (sección 5.3) |
| `RandomBaseline/outputs/ndcfix_448_*_vs_random.csv` | Comparaciones contra corridas de `ndc_fix/` | — |
| `ExploratoryDataAnalysis/revision_ambiguos/` (≈195 PNG) | Imágenes para la revisión manual de los 65 objetos ambiguos, ya aplicada (`categorias_objetos_final.csv`) | — |
| `ExploratoryDataAnalysis/figuras/orientacion_visibilidad_rot_y/` | Variante con giro en Y, que no está en la tesis | — |

### 3.4 Reducir sin borrar la carpeta

| Ruta | Qué se borra | Motivo |
|---|---|---|
| `Examples/renders/*/*/224/` y `*/1134/` | ≈465 archivos | `verificar_convencion_ndc.py` usa solo 448 px. Con una resolución basta para probar el pipeline. |
| `ExploratoryDataAnalysis/pointing2d_*.csv`, `pointing2d_*.log` | Se mueven, no se borran | Son salidas de `pointing_2d_canto.py`. Van a `resultados_finales/diagnosticos/`. |

### 3.5 Fuera del control de versiones (solo locales)

Estas carpetas no están versionadas, así que no afectan al repo público. Se pueden borrar en
local para ahorrar espacio:

| Ruta | Tamaño | Sugerencia |
|---|---|---|
| `venv/` | 8.5 GB | Borrar si se trabaja con `conda activate tesis_env` |
| `results/`, `results_25_08_2026/` | 183 MB | Salidas locales de agosto. Borrar. |
| `imgui.ini`, `.polyscope.ini` | — | Estado del visor. Ya ignorados. |
| `ECCV GeoInt poster/`, `bibliografia/` | 1.1 GB | Material personal. Mantener ignorados. Si se quiere versionar `bibliografia/descargar_bibliografia.py`, moverlo a `utils/`. |

### 3.6 Documentación a `docs/archive/`

No se borran porque registran decisiones que explican el método, pero describen etapas
superadas:

- `docs/experimentos_pipeline_sin_malla.md`
- `docs/reporte_resultados_sin_malla.md`
- `docs/resumen_ejecutivo_sin_malla.md`
- `docs/diagnostico_conditioning_axis.md`
- `docs/verificacion_metricas_literatura.md`
- `docs/inventario_Experiments.md`
- `docs/resultados/` (Flujo_A/B/C, Prompts_6pts)

---

## 4. Movimientos

| Origen | Destino |
|---|---|
| `guia_instalacion_molmopoint_qwen3vl.md` | `docs/instalacion_molmopoint_qwen3vl.md` |
| `metricas_enhanced_backprojection_3D_symmetry.md` | `docs/literature/metricas_enhanced_backprojection_3D_symmetry.md` |
| `auditoria_noviembre.log` | `resultados_finales/logs/auditoria_noviembre.log` |
| `axis_v06_gpu0.log`, `axis_v06_gpu1.log` | `resultados_finales/logs/inferencia/224_flat_v06/` |
| `logs/{molmopoint,qwen3vl}_{axis,plane}.log`, `logs/render_rot_so3_224_flat.log` | `resultados_finales/logs/inferencia/224_flat_vlms/` |
| `logs_448_flat/molmo_*.log` | `resultados_finales/logs/inferencia/448_flat/` |
| `logs_1136_flat/molmo_*.log` | `resultados_finales/logs/inferencia/1136_flat/` |
| `logs_224_iluminacion/*` | `resultados_finales/logs/inferencia/224_iluminacion/` |
| `logs_single_oct26/*` | `resultados_finales/logs/inferencia/224_single/` |
| `logs_flujoB/*` | `resultados_finales/logs/inferencia/224_flujoB/` |
| `logs_silueta/*.log` | `resultados_finales/logs/inferencia/silueta/` |
| `ExploratoryDataAnalysis/pointing2d_*.{csv,log}` | `resultados_finales/diagnosticos/` |
| informes de la sección 3.6 | `docs/archive/` |

Ningún script lee `ExploratoryDataAnalysis/pointing2d_*`. Son salidas a mano de
`pointing_2d_canto.py` (`--out`). Los runners de roll y Aguirre ya escriben las suyas en
`resultados_finales/<exp>/diag/`, así que moverlos no rompe nada.

---

## 5. Cambios de código necesarios

### 5.1 Fusionar las copias `_subset`

Cada `_subset` es su original más un cambio compatible hacia atrás:
- **runners:** `total = max(fibonacci_total, len(metadata))` y un error si faltan vistas;
- **render:** `--subset-n-views` y `--roll-seed`.

Con un render completo, el `_subset` hace exactamente lo mismo que el original. Pasos:

1. Copiar `X_subset.py` sobre `X.py` y borrar el comentario «Copia deliberada…».
2. En `data_render.py`, apuntar `EXPORT_SCRIPT` a `export_fibonacci_views.py`.
3. Reemplazar `_subset.py` por `.py` en
   - `PlaneScoreData/run_plane_v02_224_rot_so3.py`
   - `PlaneScoreData/run_plane_v02_224_rot_so3_vlms.py`
   - `PlaneScoreData/run_plane_v02_224_rot_aguirre.py`
   - `PlaneScoreData/run_plane_v02_224_roll.py`

   y en los docstrings que los citan.
4. Prueba rápida en el servidor, sin GPU: `--help` de cada runner. Después, un objeto con
   `--n-views 6` sobre un render completo y otro sobre un render parcial de `renders_rot_so3`.

### 5.2 Dejar solo EXP-E en `triangulation_variants`

En `Pipeline_Experiments/triangulation_variants/__init__.py`:

```python
from . import exp_e_iterative_reweight

VARIANTS = {"expE": exp_e_iterative_reweight}
```

Revisar si `_shared.py` tiene funciones que solo usaban A–D o F. Las que use
`exp_e_iterative_reweight.py`, por ejemplo `weighted_triangulate_line`, se quedan.

Revisar también la ayuda de `--variant` en `estimate_symmetry_variants.py`.

`PlaneScoreData/audit_run.py` importa `triangulation_variants`, pero solo usa `expE`.

### 5.3 RandomBaseline

Quitar `import latex_tables` y el subcomando que lo usa en `run_baseline.py`.

Correr `python RandomBaseline/tests/test_random_baseline.py`.

### 5.4 `run_resultados_finales.sh`

`csv_base()` busca la línea base en `ndc_fix/<carpeta>/experiments_*`. Hay dos opciones:

- **(a)** Si el script ya no se va a volver a correr, dejar un comentario de que la base vive
  ahora en `resultados_finales/*/pareado/`, y que la rama `*)` requiere recuperar `ndc_fix/`
  desde el historial de git.
- **(b)** Apuntar la rama `*)` a los CSV finales de `resultados_finales/<carpeta>/experiments_*`.

Se recomienda la (a), porque los resultados ya están cerrados y el pareado guardado no
cambia.

### 5.5 Verificación después de cada fase

```bash
python utils/generar_tablas_resultados_cap4.py
python utils/verificar_tablas_resultados_cap4.py        # debe seguir dando 43 tablas y 4491 celdas
python utils/generar_anexo_prompts.py
python PlaneScoreData/test_plane_score_data.py
python PlaneScoreData/test_audit_run.py
python RandomBaseline/tests/test_random_baseline.py
python -m compileall -q Mapping Pipeline_Experiments PlaneScoreData RandomBaseline utils pipeline_common
```

Si `generar_tablas_resultados_cap4.py` produce los mismos `.tex` (`git diff --stat` en
`Tesis_DCC_MDS/tablas_resultados/` vacío, aunque esa carpeta no se versiona), el borrado no
tocó nada de la tesis.

### 5.6 `.gitignore`

Agregar:

```
*.pid
*.synctex.gz
logs_*/
experiments_[0-9]*/
```

---

## 6. Documentación a actualizar

- **`README.md`**:
  - rehacer el árbol de carpetas con la estructura de la sección 2;
  - quitar `run_all_postprocessing.py`, `cleanup_experiments.py` y el sweep de ablaciones;
  - describir el método final (EXP-E en eje y puntaje $S$ en plano, sin malla);
  - agregar `run_resultados_finales.sh` y `generar_tablas_resultados_cap4.py` como pasos
    finales.
- **`CLAUDE.md`**:
  - el bloque de notebooks cita `ranking_postprocesamiento.ipynb`, que ya no existe en la
    raíz, y `analisis_tiempos_postprocesamiento.ipynb`, que estaba en `historial/`; quitar
    ambos;
  - agregar `PlaneScoreData/` al flujo del pipeline (estimador $S$ y runners de cada
    experimento);
  - mencionar que las copias `_subset` se fusionaron.
- **`Mapping/README.md`, `Pipeline_Experiments/README.md`, `MolmoPointing/README.md`:**
  quitar los scripts borrados.
- **`docs/features/*.md`:**
  - `mapping.md` y `consolidation-evaluation.md` citan los diagnósticos y la limpieza de
    `Mapping/`;
  - `molmo-pointing.md` cita `list_prompts.py` y el notebook;
  - `images-generator.md` debe documentar `--subset-n-views` y `--roll-seed`.
- **`docs/audits/refactor-log.md`:** agregar una entrada que resuma esta reestructuración.

---

## 7. Orden de ejecución

Cada fase cabe en un commit y se puede revertir por separado. Los comandos son los que usaría
el autor. Los commits los hace él.

1. **Fase 1, borrado sin dependencias** (sección 3.1 sin los logs, `Experiments/`,
   `ndc_fix_axial_expE/`, `historial/`, `experiments_*`, `Mapping/archive/`, los diagnósticos
   de `Mapping/`, `candidates_then_select/`, prompts `lit*`/`_6pts`, notebooks, imágenes de
   `revision_ambiguos/`):

   ```bash
   git rm -r --quiet Experiments ndc_fix_axial_expE historial experiments_27_08_2026_v1 \
     experiments_27_08_2026_v2 experiments_15_09_2026 Mapping/archive \
     Pipeline_Experiments/candidates_then_select ExploratoryDataAnalysis/revision_ambiguos \
     ExploratoryDataAnalysis/figuras/orientacion_visibilidad_rot_y
   git rm cap4_originañ.tex cap4_baselines.tex analisis_resultados_22_07.ipynb \
     plane_v04_1_flowC_expF_missing.csv logs_multiplano.txt logs_silueta/silueta.pid
   git rm MolmoPointing/prompts/axis/lit2_grid_* MolmoPointing/prompts/axis/lit3_cot_* \
     MolmoPointing/prompts/*/*_6pts_*
   ```

   Verificar (5.5).
2. **Fase 2, logs y documentación**: `git mv` de la sección 4, borrado de los logs de
   post-proceso viejos (3.1) y de los informes (3.6). Verificar.
3. **Fase 3, código obsoleto**:
   - generadores viejos de `utils/`;
   - `planar_sdat/`;
   - `ndc_fix/` junto con el cambio 5.4;
   - `ContextLimits/`;
   - `latex_tables.py` junto con 5.3;
   - variantes A–D y F junto con 5.2;
   - `run_batch` y sus configuraciones.

   Verificar.
4. **Fase 4, fusión de las copias `_subset`** (5.1). Probar en el servidor antes del commit.
5. **Fase 5, documentación** (sección 6) y reducción de `Examples/` (3.4).
6. **Opcional, renombrado de carpetas**. Por ejemplo:
   - `PlaneScoreData/` → `experimentos/`;
   - `Mapping/` → `estimacion/`;
   - las tres carpetas de pointing bajo `pointing/`.

   Requiere:
   - cambiar los 83 `sys.path.insert`;
   - cambiar las rutas fijas de los `run_*.py`, de `run_resultados_finales.sh` y de los
     comandos documentados;
   - regenerar el anexo C, que cita `MolmoPointing/prompts/...`.

   No se recomienda antes de la entrega de la tesis.

---

## 8. Nota sobre el historial de git

`cap4_originañ.tex` y `cap4_baselines.tex` son fragmentos de la tesis y están versionados
desde los commits `b220a67f` y `485a6bf6`. Borrarlos con `git rm` los saca de la rama, pero
siguen visibles en el historial público.

Si no deben quedar públicos, hay que reescribir el historial. Por ejemplo:

```bash
git filter-repo --path cap4_originañ.tex --path cap4_baselines.tex --invert-paths
```

Después hay que hacer `push --force` de cada rama. Esto obliga a re-clonar en el servidor.
Conviene decidirlo antes de la fase 1.

---

## 9. Resumen de lo que queda

| Área | Antes | Después (aprox.) |
|---|---|---|
| Archivos versionados | 17 165 | ≈1 750 |
| `.py` | 130 | ≈85 |
| Carpetas en la raíz | 30 | 17 |
| Archivos sueltos en la raíz | 14 | 4 (`README.md`, `CLAUDE.md`, `environment.yml`, `.gitignore`) |
