# Métricas de evaluación: pipeline propio vs. métricas de referencia

> Documento de contexto para escribir la sección de metodología de la tesis.
> Resume, con referencias a código exacto (archivo:línea), todas las métricas de
> evaluación usadas en el proyecto: las del pipeline original (`Mapping/evaluate.py`,
> `Mapping/estimate_symmetry.py`) y las agregadas después en
> `Mapping/reference_metrics.py` para poder compararse contra un trabajo de
> referencia externo. Incluye el origen (repo/paper) de estas últimas, y los
> hallazgos metodológicos que surgieron al compararlas.

---

## 1. Contexto general

El pipeline de esta tesis predice **una única** simetría (eje o plano) por objeto,
por combinación de `n_views ∈ {1,6,14,26}` y método de ajuste
(`svd`, `ransac_svd`, `svd_sde`, `ransac_svd_sde`). Esto es relevante para entender
varias de las diferencias metodológicas descritas más abajo: varias métricas de la
literatura (en particular el F1 de detección) están diseñadas para métodos que
predicen **un conjunto** de candidatos con score de confianza, no un solo resultado.

Se comparó contra el repositorio **`EnhancedBackProjection`**
("Enhanced Back-Projection of Vision Features for 3D Symmetry Detection",
Aguirre & Sipiran, WACV 2026 — verificar cita exacta antes de publicar, no se
confirmó con búsqueda bibliográfica independiente, solo se leyó el código y el
`readme.md` del repo), un método geométrico basado en features de DINOv2
retro-proyectadas sobre la malla, que sí predice **múltiples** planos candidatos
por objeto (hasta 10), cada uno con un score de "confianza" geométrico.

---

## 2. Métricas del pipeline propio (ya usadas en todos los experimentos)

Definidas en `Mapping/evaluate.py` y `Mapping/estimate_symmetry.py`. Se calculan
para **todos** los ~220 experimentos del sweep (24 prompts × Flow A + 4 Flow B/C,
× hasta 7-15 variantes de clustering/patch-size cada uno).

### 2.1 Basadas en ground truth (comparan contra el eje/plano anotado)

| Métrica | Definición | Archivo:línea |
|---|---|---|
| `angular_error_deg` | Ángulo mínimo entre la dirección/normal predicha y el ground truth, *sign-agnostic* (rango [0°, 90°]) | `evaluate.py::angular_error_deg` (~167-171) |
| `translation_error` | Distancia punto-a-línea (axis) o punto-a-plano (plane) entre el origen predicho y el GT | `evaluate.py::point_to_line_distance` / `point_to_plane_distance` (~174-188) |
| `auc_angular` | Área bajo la curva de precisión acumulada, integrando la fracción de objetos con `angular_error < θ` para `θ ∈ [0°, 45°]` (100 pasos), normalizada a [0,1]. **Ojo: el rango de integración se trunca en 45°, no en 90°** — todo error ≥45° se trata como "fallo total" indistintamente de cuán grande sea | `evaluate.py::auc_from_errors` (~208-218), `AUC_ANGULAR_MAX = 45.0` (línea 120) |
| `precision_{5,10,15}deg` | Fracción de objetos con `angular_error < θ` para θ ∈ {5°,10°,15°} | `evaluate.py::ANGULAR_THRESHOLDS` |
| **Manejo de objetos sin predicción válida** | Se imputa `angular_error = 90°` (peor caso) — así los objetos donde Molmo no produjo puntos usables penalizan las métricas en vez de excluirse silenciosamente | `evaluate.py::compute_summary`, docstring líneas 326-328 |
| **Manejo de múltiples planos GT (solo plane_sym)** | Se toma el **mejor match** (menor error angular) entre todos los planos GT del objeto — **no** hay concepto de recall sobre el conjunto completo de simetrías; no se penaliza por los GT no encontrados | `evaluate.py::evaluate_plane` (líneas 244-269), campo `n_true_planes` guardado para referencia |

> **Actualización (ver `docs/actualizacion_metricas.md` para el detalle completo)**:
> `evaluate.py` tenía además una fórmula propia y separada de "SDE"
> (`symmetry_distance_error`, poblaba `sde`/`sde_mean`/`auc_sde`/
> `precision_sde_*` en el CSV de resumen, solo para `plane_sym`) que resultó
> **no ser una SDE válida** — medía el doble de la distancia promedio al
> plano predicho, sin verificar que el punto reflejado cayera sobre
> superficie real. Se eliminó por completo (no estaba documentada en esta
> sección, que siempre describió solo la SDE *interna* de abajo). Desde el
> refactor, `evaluate.py` no reporta ningún "SDE propio" — el único SDE que
> reporta es `SDE_ref` (§3), ahora simétrico entre `axis_sym` y `plane_sym`
> (antes solo se acumulaba para plano).

### 2.2 SDE propio (Symmetry Distance Error) — NO usa ground truth

Autoconsistencia geométrica: refleja puntos de la malla a través del eje/plano
**predicho** y mide qué tan lejos caen de la superficie real del mismo objeto. No
compara contra ninguna anotación — por eso no constituye *data leakage* (no usa la
respuesta correcta, solo la geometría propia del objeto).

| Aspecto | Implementación del pipeline | Archivo:línea |
|---|---|---|
| Muestreo | Hasta 1000 **vértices** de la malla, elegidos al azar (no ponderado por área) | `estimate_symmetry.py::process_object`, `N_SDE_SAMPLE=1000` (línea 111), `sde_idx = np.random.default_rng(0).choice(...)` (línea ~439) |
| Reflexión (plane) | `d=(v-origin)·n; reflected = v - 2d·n` | `estimate_symmetry.py::sde_plane` (líneas 216-227) |
| Reflexión (axis, 180°) | `t=(v-origin)·dir; proj=origin+t·dir; reflected=2·proj-v` | `estimate_symmetry.py::sde_axis` (líneas 200-213) |
| Distancia | Vecino más cercano en un **`KDTree` sobre la misma muestra de vértices** (no contra la superficie completa) | `KDTree(v_sample).query(reflected)` |
| Escala | Distancia **lineal** (no al cuadrado), **normalizada** por la diagonal del bounding box del objeto | `dists.mean() / bbox_diag` |
| Umbral de aceptación | `accepted = sde <= 0.05` (5% del tamaño del objeto) — es una etiqueta informativa, no filtra datos de las métricas de arriba | `SDE_THRESHOLD = 0.05` (línea 110), usado en `compare_results.py::plot_acceptance_rate` (línea 295) para el gráfico `*_acceptance_rate.png` |

**Nota**: el SDE se guarda solo para los métodos `svd_sde`/`ransac_svd_sde` — el
ajuste (`direction`/`origin`) es **idéntico** al de `svd`/`ransac_svd` respectivamente;
la variante `_sde` únicamente le agrega este campo, no re-ajusta nada
(`estimate_symmetry.py::_make_entry`, líneas 317-335).

---

## 3. Métricas de referencia — `Mapping/evaluate.py` (`--with-reference-metrics` / `--all`)

**Antes vivían en un script separado, `Mapping/reference_metrics.py`** —
fusionado dentro de `evaluate.py` (ver `docs/actualizacion_metricas.md`),
mismos nombres de función (`calplaneloss`, `calaxisloss`, `f1_match_counts`,
etc.), mismas fórmulas exactas, sin cambios de comportamiento salvo la
adición de `f1_match_counts_hungarian` (§3.2). No re-ejecuta
`map_to_3d`/`estimate_symmetry` — **re-puntúa** predicciones ya guardadas en
`predicted_symmetry_<EXP>.json`, calculando métricas con la **misma fórmula
exacta** que usa el repositorio de referencia — deliberadamente distinta a la
sección 2.2, para poder comparar contra ese trabajo externo o contra la
convención general de la literatura. Es opt-in (`--with-reference-metrics`
para una corrida normal, o `--all`/`--experiment-ids` para el modo bulk sobre
muchas a la vez) porque necesita `gpytoolbox` y es notoriamente más caro que
el resto de las métricas (ver §5.5).

### 3.1 SDE de referencia (`SDE_ref`) — plane y axis

| Aspecto | Referencia (`evaluate.py`) | Diferencia vs. SDE interno de `estimate_symmetry.py` (§2.2) |
|---|---|---|
| Muestreo | 1000 puntos de la **superficie** (área-ponderado, `gpy.random_points_on_mesh`) | Vértices → superficie |
| Distancia | A la **malla triangulada real** (árbol AABB, `gpy.squared_distance`) | Vecino-más-cercano-en-muestra → superficie real |
| Escala | Distancia **cuadrática** (mean squared distance) | Lineal → cuadrática |
| Normalización | **Sin normalizar** — unidades crudas de la malla | Normalizada por bbox → sin normalizar |

- **Plane**: `calplaneloss()` — puerto verbatim de `metric_SDE.py::calplaneloss` del
  repo de referencia. Refleja vía `plane = [nx,ny,nz,d]` (`points - 2·λ·plane`).
- **Axis**: `calaxisloss()` — **extensión propia**, NO existe en el repo de
  referencia (confirmado: `EnhancedBackProjection` no tiene evaluación de ejes
  basada en ground truth, solo un proxy de invarianza de features — ver §4). Aplica
  el mismo principio metodológico (superficie + cuadrática + sin normalizar) a la
  reflexión de 180° sobre el eje, usando la misma geometría de reflexión que
  `sde_axis` del pipeline propio.

### 3.2 F1 de referencia (`F1_ref`) — **solo plane, no existe para axis**

Puerto verbatim de `metric_F1.py::f1_score_calc` (incluyendo su comportamiento
exacto de conteo, con sus particularidades — ver §5.3). Framing de detección:

- Umbrales de distancia entre planos `{0.05, 0.10, 0.15, 0.20}` (representación
  `[nx,ny,nz,d]`), F1 promediado sobre los 4.
- Matching contra **todos** los planos GT del objeto (recall real sobre el
  conjunto de simetrías, a diferencia de §2.1).
- Acumulación de TP/FP/FN **global** sobre todo el dataset, no promedio por objeto.
- Nuestro pipeline predice un solo plano por objeto/método (sin score de
  confianza) → se trata como candidato único siempre-aceptado
  (`f1_match_counts([pred_plane], ...)`, `reference_metrics.py` líneas ~112-140).
  La función acepta una **lista** de candidatos (no solo uno), lista para el día
  que el pipeline emita varios candidatos reales.
- **No implementado para axis**: no hay F1 estándar para simetría axial ni en el
  repo de referencia ni (según revisión de literatura, §4) en el campo en general.

---

## 4. Origen de las métricas de referencia — repo y literatura

### 4.1 Repositorio fuente

- **Repo**: `EnhancedBackProjection` (ruta local:
  `C:\Users\HP\Desktop\Seminario de Tesis I\EnhancedBackProjection`)
- **Paper**: "Enhanced Back-Projection of Vision Features for 3D Symmetry
  Detection" — Aguirre & Sipiran, WACV 2026 (título/venue leídos del `readme.md`
  del repo, **no verificados con una búsqueda bibliográfica independiente**;
  confirmar antes de citar formalmente).
- **Método**: back-projection de features DINOv2 sobre puntos de la malla,
  candidatos de plano generados por pares/tríos de puntos con features similares
  (`implementation/compute/comp_planes.py`), filtrados por distancia Chamfer
  reflectiva (`threshold=0.01`), deduplicados por similitud rotacional, **cap de
  10 candidatos por objeto** (`implementation/planes/planes.py:234-235`).
- **"Confianza" de cada candidato**: `1 - chamfer_distance/threshold` — un score
  geométrico (qué tan reflectivamente simétrico resultó ese candidato), **no** un
  score aprendido/probabilístico (`implementation/planes/planes.py:222`).
- **Dataset**: `datasets/curated_plane_sym_obj.txt` (850 objetos) usa la misma
  convención de hash de ShapeNet y el mismo nombre (`curated_plane_sym_obj`) que
  `pipeline_common/datasets.py::OBJECTS_SUBDIR` de este proyecto — fuerte indicio
  de que es el mismo dataset curado (no confirmado con los archivos `.obj`
  reales, que no vienen incluidos en ese repo).
- **Sin normalización de mallas**: se revisó `implementation/utils/load.py` y no
  hay ningún paso de reescalado (unit-cube/unit-sphere) en todo el repo — las
  mallas se usan a su escala nativa del `.obj`. Esto es relevante: si ambos
  proyectos leen los mismos archivos `.obj` sin re-escalarlos por separado, la
  escala ya es consistente entre ambos sistemas.
- **Axis**: el repo SÍ detecta ejes (`implementation/axis/generator_axis.py`,
  `circle.py`, `compute_axis()`), pero la única evaluación es un proxy
  auto-supervisado (`evaluation/feature_eval_axes.py::feature_distance_eval_axis`)
  que rota la malla 45°/90°/135°/180° sobre el eje GT y mide consistencia de
  *features* DINOv2 entre esas rotaciones — **no compara contra ground truth**,
  no es SDE ni F1, y no es portable a este proyecto (depende de features DINOv2
  por punto que este pipeline, basado en Molmo, no produce).

### 4.2 Revisión de literatura (vía búsqueda web externa — no verificada de primera mano)

> **Advertencia**: esta sección resume una respuesta de búsqueda web hecha en otra
> sesión de Claude (con acceso a internet), pegada de vuelta a este chat. No se
> verificó cada cita de forma independiente — antes de citar formalmente en la
> tesis, revisar cada paper directamente.

- **PRS-Net** (Gao et al., IEEE TVCG — año exacto a confirmar, ~2019/2021) —
  paper que popularizó SDE (y "GTE") como métricas estándar para simetría
  **planar** en ShapeNet. La convención reportada coincide con la de §3.1
  (superficie, cuadrática, sin normalizar, unidades ×10⁻⁴) — es decir,
  `reference_metrics.py` ya sigue la convención general del campo, no solo la de
  `EnhancedBackProjection` puntualmente.
- **F1 multi-candidato para planos**: reportado como estándar en trabajos
  recientes — mencionados "E3Sym" (ICCV 2023) y un trabajo de Je et al.
  ("Robust Symmetry Detection via Riemannian Langevin Dynamics", SIGGRAPH Asia
  2024) — **sin verificar independientemente**.
- **F1 para ejes**: la búsqueda no encontró un estándar consolidado análogo al de
  planos — refuerza la decisión de no implementarlo (§3.2). Se mencionó un
  trabajo de detección de ejes en imágenes 2D (dataset "DENDI") que sí reporta
  F1, pero opera sobre un dominio/dataset distinto, no comparable directamente.
- **Framing "mejor-match" vs. multi-candidato**: según la búsqueda, ambos
  coexisten en la literatura según si el método produce un único candidato o
  varios — el framing de este pipeline (mejor-match, §2.1) es apropiado dado que
  predice un solo resultado por objeto.

---

## 5. Hallazgos metodológicos relevantes para la discusión de la tesis

### 5.1 No hay *data leakage* en el SDE (ninguna de las dos versiones)

Ni el SDE propio ni el `SDE_ref` usan el archivo de ground truth — ambos son
chequeos de autoconsistencia geométrica (reflejar y medir contra la malla real
del propio objeto). Es distinto de `angular_error`/`F1_ref`, que sí comparan
contra la anotación. Ver discusión completa en la conversación de origen —
resumen: SDE podría en teoría aceptar un plano que no es el anotado si el objeto
tiene una simetría real adicional no etiquetada (limitación conocida, no leakage).

### 5.2 Comparabilidad limitada entre `AUC_angular`/`F1_ref` cuando hay múltiples GT

En un objeto con varias simetrías GT, `AUC_angular` (mejor-match) puede dar un
resultado casi perfecto mientras `F1_ref` (recall sobre el conjunto completo) da
bajo — **no es una contradicción**, son preguntas distintas ("¿mi mejor
predicción es correcta?" vs. "¿encontré todas las simetrías del objeto?"). Con
una sola predicción por objeto, el recall de `F1_ref` tiene un techo matemático
de `1/n_true_planes`.

### 5.3 `F1_ref` favorece estructuralmente a `EnhancedBackProjection`

Su método predice hasta 10 candidatos (filtrados a los de confianza ≈1.0, casi-
perfectamente reflectivos) por objeto; este pipeline predice exactamente 1. En
un objeto con múltiples simetrías reales, ellos pueden acumular varios TP: este
pipeline, como máximo, 1. **Esto debe declararse explícitamente** si se reporta
`F1_ref` al lado del de ellos — no es una comparación método-vs-método sobre la
misma tarea, es "un candidato" vs. "hasta diez candidatos", evaluados con una
métrica que premia tener más candidatos.

### 5.4 `SDE_ref` y `sde_mean` (pipeline propio) no son intercambiables

Debido a las 4 diferencias de §3.1 (muestreo, distancia, escala, normalización).
No hay que mezclarlos en la misma tabla sin conversión, y ninguno es "más
correcto" en abstracto — dependen de para qué se usan (ranking interno propio →
usar el del pipeline; comparación con literatura/paper externo → usar `_ref`).

### 5.5 Costo computacional

`reference_metrics.py` NO re-ejecuta ray-casting/SVD/RANSAC — solo re-puntúa
predicciones ya guardadas, cacheando el resultado cuando `svd`/`svd_sde` (o
`ransac_svd`/`ransac_svd_sde`) comparten exactamente el mismo ajuste (ahorra
~50% del cómputo). Medido empíricamente en el servidor: ~21 ms por predicción
individual. Correr sobre **todos** los ~220 experimentos × ambos tipos de
simetría (~2 millones de predicciones) toma ~6 horas secuenciales (CPU, sin
aceleración por GPU — `gpytoolbox`/`trimesh`/`numpy` son CPU-only); correr solo
sobre las configuraciones "ganadoras" del ranking toma minutos.

---

## 6. Resumen ejecutivo para la sección de metodología

| | axis_sym | plane_sym |
|---|---|---|
| Métrica principal (ground truth) | `angular_error` / `AUC_angular` / `precision@θ` | Igual, + framing "mejor-match" sobre múltiples GT |
| SDE propio (interno, sin GT) | `sde_axis` — vértices, lineal, normalizado | `sde_plane` — igual |
| SDE de referencia (`_ref`) | `calaxisloss` — **extensión propia**, sigue convención PRS-Net | `calplaneloss` — puerto verbatim del repo de referencia |
| F1 de referencia | **No existe** (ni en la referencia ni, aparentemente, en la literatura) | `f1_score_calc`, puerto verbatim, framing multi-candidato |
| ¿Es data leakage el SDE? | No — no usa el archivo de ground truth | No — ídem |
| Limitación principal a declarar | Ninguna comparación externa de F1 disponible | `F1_ref` favorece estructuralmente a métodos multi-candidato |
