# CLAUDE.md

Contexto persistente para trabajar en este repo. Complementa (no duplica) a:
- `docs/Contexto.md` — contexto completo de la tesis (hipótesis, dataset, metodología).
- `docs/metricas_evaluacion.md` — definición exacta de cada métrica, origen (código y
  literatura), y hallazgos metodológicos ya validados.
- `README.md` — estructura del repo, instalación, pipeline paso a paso con comandos.
- `docs/data-schemas.md` — esquema JSON exacto de cada archivo acumulativo del pipeline
  (`molmo_multiview*.json`, `mapped_points_3d*.json`, `predicted_symmetry*.json`,
  `eval_*_results.json`).
- `docs/features/` — un `.md` por feature/etapa (metodología generada desde el código:
  descripción, archivos/funciones clave, inputs/outputs, decisiones de diseño,
  limitaciones conocidas). Uno por etapa del pipeline de abajo:
  [`exploratory-data-analysis.md`](docs/features/exploratory-data-analysis.md),
  [`images-generator.md`](docs/features/images-generator.md),
  [`molmo-pointing.md`](docs/features/molmo-pointing.md),
  [`mapping.md`](docs/features/mapping.md),
  [`consolidation-evaluation.md`](docs/features/consolidation-evaluation.md),
  [`interactive-viewer.md`](docs/features/interactive-viewer.md),
  [`pipeline-common.md`](docs/features/pipeline-common.md).
- `docs/audits/` — auditoría de arquitectura (`architecture-audit.md`) y el log de la
  refactorización que se ejecutó a partir de ella (`refactor-log.md`).
- `docs/code-norms.md` — normas de estilo Python (docstrings, orden de imports, type
  hints, naming) para código nuevo o modificado en este repo.

Si una pregunta es sobre *qué significa una métrica* o *contexto de la tesis*, lee esos
docs primero en vez de re-derivarlo desde el código. Si es sobre *cómo está implementada
una etapa concreta* (qué archivos/funciones la componen, qué decisiones de diseño tiene),
mirá primero el `.md` correspondiente en `docs/features/` antes de releer el código fuente
completo.

## Qué es este proyecto

Tesis: ¿puede un VLM con capacidad de "pointing" (Molmo2-8B) identificar simetría
estructural 3D (eje o plano) a partir de renders 2D multi-vista de un objeto, sin
acceso directo a la geometría 3D? El modelo recibe imágenes renderizadas y devuelve
coordenadas 2D; esos puntos se retro-proyectan a 3D vía ray casting y se ajusta un
eje/plano de simetría, evaluado contra ground truth de ShapeNet.

## Flujo del pipeline (en orden)

```
ImagesGenerator/  → renders multi-vista de mallas .obj (Fibonacci sphere sampling)
MolmoPointing/    → Molmo2-8B infiere puntos 2D sobre esos renders (requiere GPU)

# Con malla (pipeline original):
Mapping/map_to_3d.py         → ray casting: puntos 2D → puntos 3D sobre la malla
Mapping/estimate_symmetry.py → SVD/RANSAC: puntos 3D → eje o plano
                                (métodos svd/ransac_svd/svd_sde/ransac_svd_sde)

# Sin malla (pipeline alternativo, ver docs/pipeline_sin_malla.md):
Mapping/estimate_symmetry_no_mesh.py → triangulación de rayos multi-vista → eje o
                                        plano, SIN ray-casting contra la malla
                                        (métodos triangulation/triangulation_multiplane,
                                        este último hasta 3 planos por objeto)

Mapping/evaluate.py          → métricas vs. ground truth (angular error, AUC,
                                precisión@θ) + SDE_ref/F1_ref opt-in
                                (--with-reference-metrics una corrida, o
                                --all/--experiment-ids modo bulk sobre muchas).
                                Para triangulation_multiplane, métricas de
                                recall/precision sobre el conjunto completo de
                                planos (evaluate_plane_multiset) en vez de un
                                único ángulo — ver docs/implementacion_pipeline_sin_malla.md.
Mapping/compare_results.py   → tablas + plots agregados
```

`reference_metrics.py` ya no existe como script separado — se fusionó dentro
de `evaluate.py` (ver `docs/actualizacion_metricas.md`). Cualquier código que
haga `from reference_metrics import ...` hay que migrarlo a
`from evaluate import ...` (mismos nombres de función).

Cada etapa lee/escribe JSON acumulativos por objeto bajo
`<renders_root>/<symmetry_type>/<object_id>/<size>/<lighting>/` — las corridas ya
computadas se saltan automáticamente (no hace falta `--overwrite` salvo que quieras
recalcular). Comandos exactos para cada paso: ver "Full pipeline" en `README.md`.

`--point-mode` debe coincidir con la estrategia del prompt usado en `MolmoPointing/`:
`independent` (puntos ya sobre el eje/plano) vs. `midpoint` (pares bilaterales,
midpoint calculado antes del SVD). Con `midpoint`, `n_views=1` es estructuralmente
imposible (se necesitan ≥2 vistas para tener un par).

## Convenciones no obvias (fáciles de repisar sin darse cuenta)

- **`AUC_angular` trunca en 45°, no en 90°** (`evaluate.py::AUC_ANGULAR_MAX = 45.0`).
  Todo error ≥45° cuenta como fallo total sin importar cuán grande sea.
- **`evaluate.py` ya no reporta ningún "SDE propio"** — la fórmula vieja
  (`symmetry_distance_error`, `sde`/`sde_mean`/`auc_sde`/`precision_sde_*` en
  el CSV de resumen, solo para `plane_sym`) resultó ser un bug conceptual, no
  una variante válida (medía distancia al plano promediada, sin verificar que
  el reflejo cayera sobre superficie real) — se eliminó por completo, ver
  `docs/actualizacion_metricas.md`. `SDE_ref` (abajo) es ahora la única SDE
  que reporta este script, **para ambos tipos de simetría por igual** — la
  vieja asimetría eje/plano en el CSV ya no existe. El campo `sde`/`accepted`
  interno de `estimate_symmetry.py` (heurística de aceptación, no lo que
  `evaluate.py` re-puntúa) sigue intacto, sin cambios.
- **`SDE_THRESHOLD = 0.05`** (en `estimate_symmetry.py`) define el campo
  informativo `accepted`, no filtra ninguna métrica.
- **Objetos sin predicción válida** se imputan con `angular_error = 90°` (peor caso),
  no se excluyen — así penalizan las métricas en vez de inflarlas artificialmente.
- **SDE_ref no es data leakage**: no usa el archivo de ground truth (excepto
  `F1_ref`, que sí compara contra el plano GT por diseño — es una métrica de
  detección, no de autoconsistencia).
- **`svd`/`svd_sde` y `ransac_svd`/`ransac_svd_sde` comparten el mismo ajuste** — la
  variante `_sde` solo agrega el campo SDE interno (`accepted`), no re-ajusta nada.
  El modo bulk de `evaluate.py` (`--all`/`--experiment-ids`) explota esto con una
  caché local para ahorrar ~50% del cómputo de SDE_ref/F1_ref.
- **`SDE_ref`/`F1_ref` NO son intercambiables con `angular_error`/`translation_error`
  del pipeline propio** — muestreo, escala y normalización distintos (§3.1/5.4 de
  `metricas_evaluacion.md`, a actualizar tras el refactor). Nunca mezclar en la
  misma tabla sin dejar explícita la diferencia de convención.
- **`F1_ref` tiene dos variantes**: `f1_ref` (greedy, orden de lista — convención
  histórica de PRS-Net/E3Sym) y `f1_ref_hungarian` (asignación óptima —
  convención de Reflect3D/ArchSym 2025-2026). Dan el mismo resultado con un
  solo plano predicho por objeto (el caso actual); divergen recién cuando el
  pipeline prediga varios planos candidato por objeto.
- **No existe F1 de referencia para `axis_sym`** — ni en el repo de referencia ni,
  según revisión de literatura, en el campo en general. No es una omisión.
- **Métricas multi-plano (`evaluate_plane_multiset`, recall/precision sobre el
  conjunto de planos GT) SÍ están conectadas al flujo principal** — vía el
  método `triangulation_multiplane` de `Mapping/estimate_symmetry_no_mesh.py`
  (`--max-planes > 1`) y el wrapper `evaluate_plane_multi_from_pred` en
  `evaluate.py`, incluyendo SDE_ref/F1_ref por plano con
  `--with-reference-metrics`. Los métodos con malla (`svd`/`ransac_svd`/...)
  siguen guardando un solo plano por objeto/n_views — el multi-plano solo
  existe hoy en el pipeline sin malla.

## Entorno

```bash
conda activate tesis_env
```
Ver "Installation" en `README.md` para la lista completa de deps (PyTorch3D con CUDA
12.4, `transformers`/Molmo2, `trimesh`/`scipy`/`pandas` para Mapping, `scikit-learn`
para HDBSCAN, `gpytoolbox`/`rtree` para `evaluate.py --with-reference-metrics`/`--all`).
El entorno de
desarrollo local (Windows) no tiene GPU ni todas las deps pesadas instaladas — los
pasos de Molmo2/renderizado y las corridas largas del sweep se ejecutan en el
servidor remoto, no localmente.

## Notebooks de análisis

- `ranking_postprocesamiento.ipynb` — notebook principal de análisis de resultados:
  rankings por experimento/prompt, ablations, y (secciones 10-11) comparación contra
  `reference_metrics_*.csv` (generado por `evaluate.py --all`, ver arriba — mismo
  nombre/columnas de siempre, ahora fusionado en `evaluate.py`). Reusa helpers ya
  definidos ahí (`enrich`, `inventory`, `best_per_experiment`, `plot_ablation_bar`,
  etc.) — no dupliques esa lógica en un notebook nuevo.
- `analisis_tiempos_postprocesamiento.ipynb` — tiempos de cómputo del postproceso.
- **Los notebooks pueden superar el límite de tokens de lectura de las tools** si
  tienen outputs de celdas cacheados. Si una tool de notebook falla por tamaño, limpiar
  `outputs`/`execution_count` de las celdas de código primero (script Python rápido
  sobre el JSON del notebook) antes de reintentar.
- Evitar `scipy` en código de notebook si es evitable (no está garantizado en
  `tesis_env`) — por ejemplo, Spearman se puede calcular como Pearson sobre rangos
  (`.rank().corr(..., method="pearson")`) en vez de `.corr(method="spearman")`.

## Qué NO poner en este archivo

- Nada que ya esté en `docs/Contexto.md` o `docs/metricas_evaluacion.md` — apuntar ahí,
  no copiar.
- Estado efímero: qué experimento está corriendo ahora, TODOs de corto plazo,
  resultados de una corrida puntual. Eso pertenece a la conversación o a un doc de
  resultados, no aquí — este archivo debe seguir siendo válido dentro de meses.
