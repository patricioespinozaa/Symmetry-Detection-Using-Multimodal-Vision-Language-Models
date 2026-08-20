# CLAUDE.md

Contexto persistente para trabajar en este repo. Complementa (no duplica) a:
- `docs/Contexto.md` — contexto completo de la tesis (hipótesis, dataset, metodología).
- `docs/metricas_evaluacion.md` — definición exacta de cada métrica, origen (código y
  literatura), y hallazgos metodológicos ya validados.
- `README.md` — estructura del repo, instalación, pipeline paso a paso con comandos.

Si una pregunta es sobre *qué significa una métrica* o *contexto de la tesis*, lee esos
docs primero en vez de re-derivarlo desde el código.

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
Mapping/map_to_3d.py         → ray casting: puntos 2D → puntos 3D sobre la malla
Mapping/estimate_symmetry.py → SVD/RANSAC: puntos 3D → eje o plano (4 métodos)
Mapping/evaluate.py          → métricas vs. ground truth (angular error, AUC, SDE)
Mapping/compare_results.py   → tablas + plots agregados
Mapping/reference_metrics.py → re-scoring standalone con fórmulas de un paper externo
```

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
- **El SDE del pipeline propio solo se acumula en el CSV para `plane_sym`**, nunca
  para `axis_sym` — aunque `sde_axis` sí se calcula y guarda por objeto
  (`evaluate.py::compute_summary`). Es una asimetría real del pipeline original, no
  un bug a "arreglar" sin más — ver §5.4 de `docs/metricas_evaluacion.md`.
- **`SDE_THRESHOLD = 0.05`** define el campo informativo `accepted`, no filtra
  ninguna métrica.
- **Objetos sin predicción válida** se imputan con `angular_error = 90°` (peor caso),
  no se excluyen — así penalizan las métricas en vez de inflarlas artificialmente.
- **SDE (propio y de referencia) no es data leakage**: no usa el archivo de ground
  truth, es autoconsistencia geométrica (reflejar contra la propia malla).
- **`svd`/`svd_sde` y `ransac_svd`/`ransac_svd_sde` comparten el mismo ajuste** — la
  variante `_sde` solo agrega el campo SDE, no re-ajusta nada. `reference_metrics.py`
  explota esto con una caché (`plane_cache`) para ahorrar ~50% del cómputo.
- **`SDE_ref`/`F1_ref` (de `reference_metrics.py`) NO son intercambiables con
  `sde_mean`/`auc_angular` del pipeline propio** — muestreo, escala y normalización
  distintos (§3.1/5.4 de `metricas_evaluacion.md`). Nunca mezclar en la misma tabla
  sin dejar explícita la diferencia de convención.
- **No existe F1 de referencia para `axis_sym`** — ni en el repo de referencia ni,
  según revisión de literatura, en el campo en general. No es una omisión.

## Entorno

```bash
conda activate tesis_env
```
Ver "Installation" en `README.md` para la lista completa de deps (PyTorch3D con CUDA
12.4, `transformers`/Molmo2, `trimesh`/`scipy`/`pandas` para Mapping, `scikit-learn`
para HDBSCAN, `gpytoolbox`/`rtree` para `reference_metrics.py`). El entorno de
desarrollo local (Windows) no tiene GPU ni todas las deps pesadas instaladas — los
pasos de Molmo2/renderizado y las corridas largas del sweep se ejecutan en el
servidor remoto, no localmente.

## Notebooks de análisis

- `ranking_postprocesamiento.ipynb` — notebook principal de análisis de resultados:
  rankings por experimento/prompt, ablations, y (secciones 10-11) comparación contra
  `reference_metrics_*.csv`. Reusa helpers ya definidos ahí (`enrich`, `inventory`,
  `best_per_experiment`, `plot_ablation_bar`, etc.) — no dupliques esa lógica en un
  notebook nuevo.
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
