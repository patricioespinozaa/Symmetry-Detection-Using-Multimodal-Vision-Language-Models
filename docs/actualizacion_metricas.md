# Actualización de métricas: plan de refactor

> Decisiones sobre qué métricas eliminar, renombrar, mantener y agregar,
> tomadas a partir de `docs/verificacion_metricas_literatura.md` y las
> respuestas obtenidas de Claude Web (con websearch) sobre cada una. Motivado
> además por un cambio de alcance: el pipeline pasa de predecir **un solo**
> plano por objeto a detectar **hasta 3 planos** de simetría (el dataset
> curado tiene objetos con 1, 2 y 3 planos GT), lo que activa preguntas de
> matching multi-candidato que antes eran irrelevantes.
>
> Principio general pedido por el autor: **mantener las métricas de
> `Mapping/reference_metrics.py`** (son las usadas históricamente para
> comparar contra el trabajo de referencia) y **agregar las más recientes
> como complementarias**, no reemplazar de entrada.

---

## 0. Qué cambió respecto a `docs/verificacion_metricas_literatura.md`

Claude Web (con websearch) confirmó/aportó tres cosas nuevas que no estaban
en la revisión interna:

1. **`evaluate.py::symmetry_distance_error` no es SDE** — es distancia media
   al plano (×2), no verifica que el punto reflejado caiga sobre superficie
   real. Confirmado como "sin respaldo real", no una variante válida.
2. **`SDE_ref` (`reference_metrics.py::calplaneloss`/`calaxisloss`)** sí es la
   convención estándar (PRS-Net y seguidores) — confirmado como referencia
   correcta a mantener.
3. **El matching de `F1_ref` (greedy, orden de iteración)** es la convención
   histórica de PRS-Net/E3Sym, pero **Reflect3D (CVPR 2025) y ArchSym (2026)
   migraron a matching bipartito óptimo** entre todos los planos predichos y
   todos los GT. Con 1 plano predicho por objeto esto no cambiaba nada
   (greedy y óptimo coinciden con un solo candidato) — pero **al pasar a
   detectar hasta 3 planos, esto sí puede cambiar el resultado**, y conviene
   tenerlo como variante complementaria desde ahora.

---

## 1. Tabla de decisiones

| Métrica | Tipo | Accion | Por qué |
|---|---|---|---|
| `angular_error_deg` | ambos | **Mantener** | Estándar confirmado (PRS-Net, E3Sym, Aguirre & Sipiran, ArchSym) |
| `translation_error` | ambos | **Mantener** | Sin normalizar hoy; agregar variante normalizada por bbox como complementaria (ver §3) |
| `precision_{5,10,15}deg` | ambos | **Mantener, documentar** | Umbrales no canónicos pero razonables — dejar explícito en la tesis que son elección propia, no un estándar de facto |
| `auc_angular` (trunca en 45°) | ambos | **Mantener, documentar** | Ningún benchmark canonizó 45°, pero es defendible (>45° = eje/plano más ortogonal que paralelo al GT). Documentar explícitamente la elección |
| `sde` interno / `accepted` (`estimate_symmetry.py::sde_axis`/`sde_plane`, KDTree en muestra de vértices) | ambos | **Mantener sin cambios, re-etiquetar en docs** | Es una aproximación válida y barata (no un bug) — pero renombrar su rol en la documentación a "heurística interna de aceptación", **no** presentarla como "el SDE" del proyecto. No tocar el campo JSON (rompería compatibilidad con ~220 experimentos ya corridos) |
| `evaluate.py::symmetry_distance_error` → `sde`/`sde_mean`/`sde_median`/`sde_std`/`auc_sde`/`precision_sde_{010,020}` en el CSV de resumen | **plane_sym únicamente** | **ELIMINAR de los resultados reportados** | Confirmado sin respaldo en literatura — no verifica autoconsistencia real. Ver plan de migración en §2 |
| `SDE_ref` (`reference_metrics.py::calplaneloss`/`calaxisloss`) | ambos | **Mantener — PROMOVER a SDE principal reportado** | Estándar confirmado (puerto verbatim de PRS-Net para plano; extensión propia consistente con la misma convención para eje) |
| `F1_ref` (greedy, 4 umbrales `{0.05,0.10,0.15,0.20}`) | **plane_sym únicamente** | **Mantener** (uso histórico pedido explícitamente) | Estándar de PRS-Net/E3Sym — mantener para comparabilidad con el trabajo de referencia ya usado en la tesis |
| F1 con matching óptimo (Hungarian) | **plane_sym únicamente** | **AGREGAR como complementaria** | Convención de Reflect3D/ArchSym (2025-2026). Con ≤1 plano predicho no cambiaba nada; con hasta 3 planos ahora sí puede diferir del greedy — agregar sin quitar el greedy |
| F1/AUC/precisión para eje | axis_sym | **No agregar** | Confirmado: no existe convención establecida en la literatura para esto. La ausencia no es una omisión propia |
| Recall multi-plano propio (contra el pipeline's own GT, no solo `F1_ref`) | plane_sym | **AGREGAR (nueva)** | Hoy `evaluate.py::evaluate_plane` solo guarda el mejor match (`matched_true_plane`) — no existe una métrica propia de recall sobre el conjunto completo de planos GT. Necesaria ahora que se detectan hasta 3 planos por objeto |
| `n_planes_detected` vs `n_planes_gt` | plane_sym | **AGREGAR (nueva, diagnóstico)** | Ya se usó en el notebook de prueba (`test_pipeline_sin_malla.ipynb`, sección de consolidación) y resultó reveladora — cuántos planos se detectan vs. cuántos hay realmente, por objeto y agregado |

---

## 2. Plan de migración para `evaluate.py::symmetry_distance_error`

**No hay que tocar retroactivamente los ~220 experimentos ya corridos** — el
campo `sde` interno de `estimate_symmetry.py` (usado para `accepted`) sigue
intacto en `predicted_symmetry_<exp>.json`, eso no cambia. Lo que cambia es
qué se **reporta como "SDE" en las tablas y gráficos de la tesis**:

1. En `evaluate.py::evaluate_plane` y `compute_summary`: dejar de llamar
   `symmetry_distance_error()` para poblar `sde`/`sde_mean`/`auc_sde`/
   `precision_sde_*`. Opciones, de más a menos preferida:
   - (a) **Dejar de calcular esto en `evaluate.py` por completo** y usar
     directamente `reference_metrics.py::calplaneloss` (ya existe, ya está
     probado) como la única fuente de SDE reportado — evita mantener dos
     implementaciones.
   - (b) Si se quiere mantener un SDE "rápido" sin depender de `gpytoolbox`
     dentro de `evaluate.py`, reemplazar la fórmula por la de
     `estimate_symmetry.py::sde_plane` (reflejar + vecino más cercano vía
     KDTree), ya implementada y correcta — solo habría que exponerla /
     reutilizarla en vez de reimplementar la fórmula rota.
2. Eliminar del `write_csv` los campos derivados de la fórmula vieja
   (`sde_mean`, `sde_median`, `sde_std`, `auc_sde`, `precision_sde_010`,
   `precision_sde_020`) — o, si se prefiere no romper el schema del CSV para
   scripts que ya lo leen (`compare_results.py`), recalcularlos con la
   fórmula correcta antes de escribirlos, dejando el mismo nombre de columna
   pero con el valor correcto (documentar el cambio de fórmula con fecha en
   el propio `docs/metricas_evaluacion.md`).
3. Actualizar `docs/metricas_evaluacion.md` §2.2 para documentar esta
   corrección explícitamente (hoy no menciona `evaluate.py::symmetry_distance_error`
   en absoluto — es un hallazgo nuevo de esta conversación).
4. Re-correr `reference_metrics.py --all` (o al menos sobre los experimentos
   "ganadores" ya identificados) para tener `SDE_ref`/`F1_ref` actualizados
   como fuente principal de reporte, si no se corrió ya para todos.

---

## 3. Métricas nuevas a agregar (complementarias, no reemplazan nada del §1)

### 3.1 F1 con matching óptimo (Hungarian) — plane_sym

Agregar en `reference_metrics.py` una función `f1_match_counts_optimal()`
(o un flag `--matching {greedy,hungarian}` en el CLI) que use
`scipy.optimize.linear_sum_assignment` sobre la matriz de distancias
`‖pred_i − gt_j‖` (considerando también `‖pred_i + gt_j‖` por la ambigüedad
de signo, igual que el greedy) en vez del bucle voraz de `f1_match_counts`.
Reportar como columna nueva `f1_ref_hungarian`, sin tocar `f1_ref` (greedy)
existente. Con predicciones de un solo plano por objeto, ambas deberían dar
resultados idénticos — sirve como test de regresión al implementarla.

### 3.2 Traslación normalizada por bbox — ambos tipos

Agregar `translation_error_normalized = translation_error / bbox_diagonal`
junto al `translation_error` sin normalizar ya existente, para que sea
comparable entre objetos de escalas distintas (crítica de la revisión: el
`translation_error` actual mezcla escalas del objeto sin normalizar, algo
poco citado en la literatura de simetría según Claude Web).

### 3.3 Recall multi-plano propio — plane_sym

Con hasta 3 planos detectados por objeto, agregar (en `evaluate.py`, no en
`reference_metrics.py`, para que use la misma convención angular ya definida
en §2.1 del proyecto en vez de la distancia de vector `[nx,ny,nz,d]`):

- `n_planes_matched`: cuántos planos GT tienen al menos un plano predicho
  dentro de un umbral angular (p. ej. 15°, reusando `ANGULAR_THRESHOLDS`).
- `recall_planes = n_planes_matched / n_true_planes`.
- `precision_planes = n_planes_matched / n_planes_predicted`.

Esto llena el hueco que señalaba la revisión anterior ("no hay concepto de
recall sobre el conjunto completo de simetrías" — B.1 de
`verificacion_metricas_literatura.md`), sin depender de `gpytoolbox` ni de
la convención de distancia de vector de `F1_ref`.

### 3.4 Diagnóstico `n_planes_detected` vs `n_planes_gt` — plane_sym

Agregar como campo simple en el JSON de predicción y en el resumen: cuántos
planos detectó el algoritmo de consolidación vs. cuántos planos GT tiene el
objeto (`n_true_planes`, ya existe). Esto ya demostró ser útil en
`test_pipeline_sin_malla.ipynb` (sección de consolidación S4) para detectar
sobre-detección (planos espurios) o sub-detección.

---

## 4. Resumen ejecutivo (para la sección de metodología de la tesis)

**Eje**: sin cambios de fondo — el conjunto de métricas ya usado
(`angular_error`, `translation_error`, `precision@{5,10,15}°`,
`auc_angular`, `SDE_ref`) es correcto y está alineado con lo que reporta la
literatura para ejes de simetría (que no incluye F1, confirmado dos veces).

**Plano**: se corrige un hallazgo real (SDE mal calculado en el CSV de
resumen propio, nunca documentado, no afecta `SDE_ref` que ya estaba bien),
se mantiene `F1_ref` greedy por continuidad histórica con el trabajo de
referencia, y se agregan 4 métricas nuevas motivadas por el paso a detección
multi-plano: F1 con matching óptimo (comparabilidad con el estado del arte
2025-2026), traslación normalizada, recall/precision propios sobre el
conjunto de planos, y el diagnóstico de conteo de planos detectados vs. GT.

**Nada de esto invalida experimentos ya corridos** — los JSON de predicción
(`predicted_symmetry_<exp>.json`) no cambian de formato; lo que cambia es
qué se calcula/reporta a partir de ellos hacia adelante.


---

## 5. Estado: IMPLEMENTADO — `Mapping/reference_metrics.py` fusionado en `Mapping/evaluate.py`

Todo lo de este documento ya esta implementado, no es solo un plan. Resumen
de lo que cambio realmente en el codigo (no solo en la documentacion):

### 5.1 Consolidacion

`Mapping/reference_metrics.py` **ya no existe como archivo separado**. Todas
sus funciones viven ahora en `Mapping/evaluate.py`, con los mismos nombres
(`calplaneloss`, `calaxisloss`, `normal_origin_to_plane`, `gt_planes_for_object`,
`f1_match_counts`, `THRESHOLDS_INLIER`, `N_SAMPLES_DEFAULT`) -- cualquier
`from reference_metrics import ...` hay que cambiarlo a `from evaluate import ...`.
Ya se actualizo el unico lugar que hacia esto (`test_pipeline_sin_malla.ipynb`).

### 5.2 Fix del SDE roto (seccion 2 del plan) -- HECHO

`evaluate.py::symmetry_distance_error` y los campos que poblaba
(`sde`/`sde_mean`/`sde_median`/`sde_std`/`auc_sde`/`precision_sde_010`/`_020`)
fueron **eliminados por completo** del CSV de resumen y del JSON por objeto.
`evaluate.py` ya no reporta ningun "SDE propio" -- `SDE_ref` (opt-in, ver 5.3)
es la unica SDE que reporta, para **ambos** tipos de simetria por igual (la
vieja asimetria eje/plano en el CSV ya no aplica, porque ya no hay SDE propio
del CSV para ninguno de los dos). El campo interno `sde`/`accepted` de
`estimate_symmetry.py` (heuristica de aceptacion) sigue exactamente igual,
sin tocar.

### 5.3 SDE_ref/F1_ref -- opt-in, no por defecto

Por el costo (necesita `gpytoolbox`, arma AABB tree + muestreo de superficie
por objeto), quedaron **opt-in**, igual que el diseño original de
`reference_metrics.py`:

- `python Mapping/evaluate.py ... --with-reference-metrics` -- agrega
  `sde_ref`/`f1_counts_ref`/`f1_counts_ref_hungarian` al JSON por objeto y
  `sde_ref_mean/min/max`/`f1_ref`/`f1_ref_hungarian` al CSV de resumen, para
  UNA corrida (un experiment_id + un metodo).
- `python Mapping/evaluate.py ... --all` (o `--experiment-ids ID1 ID2 ...`)
  -- modo bulk, equivalente exacto al viejo `reference_metrics.py --all`:
  descubre/re-puntua muchos experimentos x metodos a la vez sin re-ejecutar
  SVD/RANSAC, escribe `reference_metrics_{axis,plane}.csv` (mismo nombre y
  columnas de siempre, mas la columna nueva `f1_ref_hungarian`). Esto NO
  estaba en el plan original de este documento -- se agrego durante la
  implementacion al notar que `ranking_postprocesamiento.ipynb` (secciones
  10-11) depende exactamente de esos archivos generados por ese modo bulk; se
  replico para no romper ese notebook.

### 5.4 F1 con matching optimo (seccion 3.1 del plan) -- HECHO

`f1_match_counts_hungarian()` implementada con `scipy.optimize.linear_sum_assignment`.
Reporta como `f1_ref_hungarian`, sin tocar `f1_ref` (greedy) existente. Con
un solo plano predicho por objeto (caso actual) da identico a `f1_ref` --
verificado con test: ambos dieron 0.3750 sobre un fixture de 2 objetos.

**Correccion importante encontrada durante la implementacion, no prevista en
el plan original**: `F1_ref` (y ahora tambien `f1_ref_hungarian`) es una
metrica de **acumulacion global** de TP/FP/FN sobre TODO el dataset por
umbral, antes de calcular F1 -- **no** el promedio de un F1 por objeto (esas
dos cosas dan numeros distintos cuando `n_true_planes` varia entre objetos).
La primera version de esta implementacion calculaba por error un F1 por
objeto y promediaba -- se detecto comparando el modo normal contra el modo
bulk (que si replicaba la formula original correctamente) y no coincidian.
Se corrigio: `evaluate_plane()` ahora guarda conteos crudos `(tp, fp, fn)`
por objeto y por umbral (`f1_counts_ref`/`f1_counts_ref_hungarian`), y
`compute_summary()` los acumula globalmente antes de calcular F1 -- igual
que el modo bulk. Verificado: ambos modos ya dan el mismo numero.

### 5.5 Traslacion normalizada (seccion 3.2 del plan) -- HECHO

`translation_error_normalized` (= `translation_error / diagonal_bbox`) se
agrego para **ambos** tipos de simetria (antes la malla solo se cargaba para
`plane_sym`; ahora se carga siempre, es barato -- no necesita `gpytoolbox`).

### 5.6 Recall/precision multi-plano y diagnostico de conteo (secciones 3.3/3.4) -- IMPLEMENTADO COMO SCAFFOLDING

`evaluate_plane_multiset(pred_planes, true_elements, angular_threshold_deg)`
esta implementada y probada (`n_planes_matched`, `recall_planes`,
`precision_planes`), pero **no esta conectada** a `evaluate_object`/
`compute_summary` -- el JSON de prediccion sigue guardando un solo plano por
metodo/n_views hoy. Llamarla directamente una vez que exista el detector de
hasta 3 planos; no requiere cambios adicionales en `evaluate.py` para
integrarse, solo pasarle la lista de planos candidatos que el detector
produzca.

### 5.7 Verificacion

Probado end-to-end con fixtures sinteticos (mesh + GT + predicted_symmetry.json
armados a mano, sin depender de datos reales del servidor):
- Modo normal sin `--with-reference-metrics`: igual que antes, menos los
  campos de SDE roto (ya no aparecen).
- Modo normal con `--with-reference-metrics`: `sde_ref`/`f1_ref`/`f1_ref_hungarian`
  calculados correctamente para eje y plano.
- Modo bulk (`--all` y `--experiment-ids`): replica el CSV de
  `reference_metrics.py` original, con la columna nueva `f1_ref_hungarian`.
- Los dos modos (normal y bulk) dan **el mismo** `F1_ref` sobre el mismo
  fixture -- confirma que la correccion de agregacion global (5.4) quedo
  bien aplicada en ambos caminos de codigo.

### 5.8 Pendiente (no incluido en este refactor)

- `docs/metricas_evaluacion.md` fue actualizado para reflejar el merge y el
  fix de SDE, pero no se reescribio integramente -- revisar antes de citarlo
  textualmente en la tesis.
- El detector de hasta 3 planos (mencionado por el autor como proximo paso)
  no se implemento aca -- esta seccion solo dejo listas las metricas para
  cuando exista.
