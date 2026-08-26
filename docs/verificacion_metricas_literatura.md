# Verificación de métricas actuales contra la literatura

> Propósito: insumo para preguntarle a un modelo con websearch (Claude Web u
> otro) cuáles de las métricas que usa hoy este pipeline existen/son estándar
> en la literatura actual de detección de simetría 3D, y cuáles convendría
> mantener, renombrar o descartar. El detalle de cada métrica está sacado
> directo del código (`Mapping/evaluate.py`, `Mapping/estimate_symmetry.py`,
> `Mapping/reference_metrics.py`) y de `docs/metricas_evaluacion.md`, no de
> memoria.

---

## ⚠️ Hallazgo a verificar antes de mandar el prompt

Durante el armado de este documento se encontró algo que **no está
documentado en `docs/metricas_evaluacion.md`**: hay **dos fórmulas de SDE
distintas para plano**, ambas activas en el código, que producen números
diferentes:

1. `estimate_symmetry.py::sde_plane` (documentada en §2.2 de
   `metricas_evaluacion.md`) — vecino-más-cercano en una muestra de 1000
   vértices, vía `KDTree`. Se usa **solo** para el campo `accepted` en
   `predicted_symmetry_<exp>.json`.
2. `evaluate.py::symmetry_distance_error` (NO documentada en
   `metricas_evaluacion.md`) — `mean(2·|dot(v - origin, n)|) / bbox_diag`
   sobre **todos** los vértices de la malla, sin reflejar-y-buscar-vecino.
   Esta es la que efectivamente llena `sde_mean`/`sde_median`/`sde_std`/
   `auc_sde`/`precision_sde_010`/`precision_sde_020` en el CSV de resumen
   (`eval_..._summary.csv`) y lo que grafica `compare_results.py`.

Estas dos fórmulas miden cosas conceptualmente distintas (una compara contra
la geometría real después de reflejar; la otra es directamente distancia al
plano promediada, sin verificar que el reflejo caiga sobre superficie real).
Se documentan ambas por separado más abajo. **Antes de reportar resultados
de tesis basados en `sde_mean`/`auc_sde` del CSV de resumen, conviene decidir
si esto es intencional o un bug** — no se resuelve en este documento, solo se
deja marcado para que se investigue.

---

## Metodología general del pipeline (contexto necesario)

- El pipeline predice **una sola** simetría (eje o plano) por objeto, por
  combinación de `n_views ∈ {1,6,14,26,42,62,86,114}` y método de ajuste.
- 4 métodos de ajuste por combinación: `svd`, `ransac_svd`, `svd_sde`,
  `ransac_svd_sde`. `svd`/`ransac_svd` ajustan sobre los puntos 3D (via
  ray-casting de los puntos 2D de Molmo2 contra la malla real); las
  variantes `_sde` son el mismo ajuste, solo con el campo SDE agregado (no
  re-ajustan nada).
- Ajuste: SVD sobre la nube de puntos 3D — primera componente principal =
  dirección del eje (`fit_axis`); última componente (mínima varianza) =
  normal del plano (`fit_plane`).
- RANSAC (`ransac_axis`/`ransac_plane`): 1000 iteraciones, umbral de inlier =
  5% de la diagonal del bounding box del objeto, semilla fija (42).

---

## A. Simetría axial (`axis_sym`)

### A.1 Basadas en ground truth

| Métrica | Fórmula exacta | Rango / umbral | Código |
|---|---|---|---|
| **Error angular** (`angular_error_deg`) | `arccos(clip(\|dir_pred · dir_GT\|, 0, 1))` en grados, con ambos vectores normalizados | `[0°, 90°]`, sign-agnostic (invariante a que el vector apunte "para el otro lado") | `evaluate.py::angular_error_deg` |
| **Error de traslación** (`translation_error`) | Distancia punto-a-línea: `‖(origen_pred − origen_GT) − ((origen_pred − origen_GT)·dir) · dir‖` | sin normalizar (unidades de la malla) | `evaluate.py::point_to_line_distance` |
| **Precisión@θ** (`precision_5deg`, `precision_10deg`, `precision_15deg`) | Fracción de objetos con `angular_error < θ`, θ ∈ {5°, 10°, 15°} | binario por objeto, luego promedio | `evaluate.py::ANGULAR_THRESHOLDS = [5, 10, 15]` |
| **AUC angular** (`auc_angular`) | Área bajo la curva "fracción de objetos con error < t" para `t` en 101 pasos de `[0°, 45°]`, normalizada dividiendo por 45 | `[0, 1]`. **El truncamiento en 45° (no 90°) es una decisión del pipeline**: todo error ≥45° cuenta como fallo total sin importar cuán grande sea | `evaluate.py::auc_from_errors`, `AUC_ANGULAR_MAX = 45.0` |
| **Imputación de objetos sin predicción válida** | `angular_error = 90°` (peor caso posible), `precision_*deg = 0` — no se excluyen, penalizan las métricas | — | `evaluate.py::compute_summary`, docstring |
| **Manejo de múltiples ejes GT** | No aplica — el dataset de ejes trae **un solo eje** por objeto (no hay "mejor match" ni recall sobre un conjunto) | — | `evaluate.py::evaluate_axis` usa `true_elements[0]` directamente |

### A.2 Sin ground truth (autoconsistencia geométrica)

| Métrica | Fórmula exacta | Uso | Código |
|---|---|---|---|
| **SDE interno** (`sde` en `predicted_symmetry_<exp>.json`, solo métodos `_sde`) | 1. Muestrea hasta 1000 **vértices** al azar (`seed=0`, no ponderado por área). 2. Refleja cada vértice `v` con rotación de 180° sobre el eje predicho: `t=(v−origen)·dir; proj=origen+t·dir; reflejado=2·proj−v`. 3. Busca el vecino más cercano del punto reflejado **dentro de la misma muestra de 1000 vértices** (`KDTree`). 4. `SDE = mean(distancia) / diagonal_bbox` | Determina el flag `accepted = SDE ≤ 0.05` — **no se agrega a ningún CSV de resumen para eje** (ver nota de asimetría más abajo) | `estimate_symmetry.py::sde_axis`, `SDE_THRESHOLD = 0.05` |
| **Asimetría documentada eje vs. plano** | El SDE de eje se calcula y guarda por objeto, pero `evaluate.py::compute_summary` solo acumula SDE cuando `symmetry_type == "plane_sym"` — para eje, esta rama nunca corre. Es una decisión de diseño del pipeline original, no un bug a corregir sin más | — | `evaluate.py::compute_summary`, líneas ~359-363, 405-414 |

### A.3 Métricas de referencia externa (`Mapping/reference_metrics.py`)

| Métrica | Fórmula exacta | Diferencia vs. A.2 | Código |
|---|---|---|---|
| **SDE_ref** (`sde_ref_mean/min/max`) | 1. Muestrea 1000 puntos de la **superficie real** (ponderado por área, `gpytoolbox.random_points_on_mesh`, `seed=0`). 2. Refleja cada punto con la misma geometría de rotación 180° que A.2. 3. Distancia al cuadrado (no lineal) contra la **malla triangulada completa** (árbol AABB, `gpytoolbox.squared_distance`), no contra una muestra de vértices. 4. **Sin normalizar** por bbox (unidades crudas) | Vértices→superficie; vecino-en-muestra→superficie real; lineal→cuadrática; normalizada→sin normalizar | `reference_metrics.py::calaxisloss` |
| **F1 de referencia** | **No existe para eje.** Ni el repo de referencia (`EnhancedBackProjection`) ni, según revisión de literatura ya hecha, el campo en general, tienen una convención establecida de F1 multi-candidato para ejes de simetría — lo que se reporta habitualmente es error angular/AUC | — | `reference_metrics.py`, docstring del módulo |

**Nota**: `calaxisloss` es una **extensión propia** del proyecto (no es un
puerto de código de ningún repo externo) — aplica la misma convención de
`calplaneloss` (ver B.3) a la reflexión de 180° en vez de a la reflexión
planar, porque el repo de referencia no tiene nada de esto para ejes.

---

## B. Simetría planar (`plane_sym`)

### B.1 Basadas en ground truth

| Métrica | Fórmula exacta | Rango / umbral | Código |
|---|---|---|---|
| **Error angular** (`angular_error_deg`) | Igual fórmula que A.1, pero calculado contra **cada** plano GT del objeto y se toma el **mejor match** (menor ángulo) — el dataset trae objetos con 1, 2 o 3 planos GT | `[0°, 90°]` | `evaluate.py::evaluate_plane` |
| **Error de traslación** (`translation_error`) | Distancia punto-a-plano: `\|dot(origen_pred − origen_GT_matcheado, n_GT_matcheado)\|` — contra el mismo plano GT que ganó el match angular | sin normalizar | `evaluate.py::point_to_plane_distance` |
| **Precisión@θ** / **AUC angular** | Idénticas a A.1 (mismos umbrales, mismo truncamiento en 45°) | — | igual que A.1 |
| **Manejo de múltiples planos GT** | Se toma el **mejor match únicamente** — no hay concepto de recall sobre el conjunto completo de planos anotados; no se penaliza por los GT no encontrados. Se guarda `matched_true_plane` (índice) y `n_true_planes` (conteo) para referencia, pero no se usan para penalizar | — | `evaluate.py::evaluate_plane` |
| **Imputación sin predicción válida** | Igual que A.1 (90° / precisión 0) | — | igual que A.1 |

### B.2 Sin ground truth (autoconsistencia geométrica) — **DOS FORMULAS DISTINTAS, ver advertencia arriba**

| Métrica | Fórmula exacta | Dónde se usa | Código |
|---|---|---|---|
| **SDE interno "de aceptación"** (`sde` en `predicted_symmetry_<exp>.json`) | 1. Muestrea hasta 1000 vértices al azar (`seed=0`). 2. Refleja cada vértice a través del plano predicho: `d=(v−origen)·n; reflejado=v−2d·n`. 3. Vecino más cercano dentro de la misma muestra (`KDTree`). 4. `SDE = mean(distancia) / diagonal_bbox` | Determina `accepted = SDE ≤ 0.05`, graficado como `% accepted` en `compare_results.py::plot_acceptance_rate` | `estimate_symmetry.py::sde_plane` |
| **SDE "de resumen"** (`sde`/`sde_mean`/`sde_median`/`sde_std` del CSV, distinto del anterior) | `mean(2·\|dot(v − origen_pred, n_pred)\|) / diagonal_bbox`, sobre **todos** los vértices de la malla (no muestreados) — **no refleja ni busca vecino**, es directamente distancia-al-plano promediada y duplicada | Llena `sde_mean`/`median`/`std` y `auc_sde`/`precision_sde_{010,020}` del CSV de resumen por experimento, y lo que grafica `compare_results.py` en los plots de SDE | `evaluate.py::symmetry_distance_error` |
| **AUC de SDE** (`auc_sde`) | Igual construcción que `auc_angular` pero integrando sobre `SDE ∈ [0, 0.10]` (`AUC_SDE_MAX = 0.10`), usando la fórmula "de resumen" de arriba, no la de aceptación | `[0, 1]` | `evaluate.py::auc_from_errors`, `AUC_SDE_MAX = 0.10` |
| **Precisión@SDE** (`precision_sde_010`, `precision_sde_020`) | Fracción de objetos con SDE "de resumen" `< 0.01` / `< 0.02` (1%/2% de la diagonal del bbox) | — | `evaluate.py::SDE_THRESHOLDS = [0.01, 0.02]` |

### B.3 Métricas de referencia externa (`Mapping/reference_metrics.py`)

| Métrica | Fórmula exacta | Origen | Código |
|---|---|---|---|
| **SDE_ref** (`sde_ref_mean/min/max`) | Igual construcción que A.3 (superficie ponderada por área, distancia al cuadrado contra malla real vía AABB, sin normalizar) aplicada a reflexión planar: `reflejado = puntos − 2·λ·plano`, con `plano=[nx,ny,nz,d]` | **Puerto verbatim** de `metric_SDE.py::calplaneloss` del repo de referencia (`EnhancedBackProjection`) | `reference_metrics.py::calplaneloss` |
| **F1_ref** | Para cada uno de 4 umbrales `{0.05, 0.10, 0.15, 0.20}` (distancia euclidiana entre representaciones `[nx,ny,nz,d]` de plano, considerando también el signo opuesto): matching greedy contra **todos** los planos GT del objeto, acumulando TP/FP/FN **globalmente sobre todo el dataset** (no promedio por objeto); F1 por umbral, luego promedio de los 4. Nuestro pipeline predice un solo plano por objeto/método → se trata como candidato único siempre-aceptado (sin filtro de confianza, a diferencia del repo de referencia que predice hasta 10 candidatos con score) | **Puerto verbatim** de `metric_F1.py::f1_score_calc` | `reference_metrics.py::f1_match_counts` |

---

## C. Métricas derivadas/de reporte (no recalculan nada, solo agregan)

| Métrica | Qué hace | Código |
|---|---|---|
| **`% accepted`** | Lee el campo `accepted` (de B.2/A.2, el SDE "de aceptación") de `predicted_symmetry_<exp>.json` y calcula el % de objetos aceptados, solo para métodos `svd_sde`/`ransac_svd_sde` | `compare_results.py::plot_acceptance_rate` |
| **`% objetos con predicción válida`** | `n_objects / denominador_esperado`, por experimento y n_views | `compare_results.py::plot_valid_by_prompt` |

---

## Preguntas concretas a resolver con literatura

1. ¿El **error angular sign-agnostic en [0°, 90°]** y la **precisión@{5,10,15}°** son la convención estándar actual para reportar error de orientación en detección de simetría 3D, o hay otras convenciones más citadas (p. ej. error en [0°,180°], u otros umbrales de precisión)?
2. ¿El **truncamiento del AUC angular en 45°** (en vez de 90°) es una práctica reconocida en algún benchmark, o es una elección ad hoc de este proyecto que convendría documentar mejor o cambiar?
3. Sobre el **SDE**: ¿cuál de las dos convenciones (A) vecino-más-cercano tras reflejar sobre una muestra de vértices — normalizado y lineal, o (B) muestreo de superficie ponderado por área + distancia real a la malla vía AABB — cuadrática y sin normalizar (la de `reference_metrics.py`, más fiel a PRS-Net/`EnhancedBackProjection`) — es la que reportan más papers recientes de forma comparable? ¿Existen otras variantes reconocidas?
4. La fórmula `evaluate.py::symmetry_distance_error` (distancia-al-plano promedio × 2, sin reflejar ni buscar vecino) — **¿corresponde a alguna métrica real de la literatura**, o es una simplificación/aproximación que no debería llamarse "SDE" porque no verifica autoconsistencia geométrica real?
5. ¿El **F1 con matching a 4 umbrales de distancia de vector de plano `{0.05,0.10,0.15,0.20}`** (acumulación global TP/FP/FN) sigue siendo una métrica vigente/citada para detección de simetría, o los benchmarks más recientes usan otra convención de matching (p. ej. basada en error angular + distancia por separado, en vez de la norma combinada del vector `[nx,ny,nz,d]`)?
6. ¿Existe alguna convención de **F1/AUC/precision para ejes de simetría** en papers 2023-2026 que no se haya encontrado en la revisión previa (recordar: se confirmó que `EnhancedBackProjection` no tiene F1 de eje)?
7. En general: de todas las métricas listadas arriba, ¿cuáles **mantendría** un revisor/comité por ser estándar reconocido en el área, cuáles son **poco usadas o específicas de un solo paper**, y cuáles **no tienen respaldo real en literatura** (uso interno/ad hoc que convendría no presentar como "la métrica de la literatura")?

---

## Prompt para Claude Web (usar tal cual, con websearch activado)

```
Estoy escribiendo la sección de metodología de mi tesis sobre detección de
simetría 3D (eje de rotación y plano de reflexión) a partir de puntos 2D
señalados por un VLM (Molmo2), retro-proyectados a 3D. Necesito que verifiques,
usando busqueda web actualizada (papers 2020-2026, benchmarks reconocidos como
PRS-Net, SymmetryNet, Reflect3D, EnhancedBackProjection/WACV 2026, y cualquier
survey reciente de symmetry detection), cuáles de las siguientes métricas que
uso actualmente son estándar/reconocidas en la literatura actual, cuáles son
poco comunes o específicas de un solo trabajo, y cuáles no tienen respaldo real
(serían "ad hoc" de mi propio pipeline). Para cada métrica te doy la fórmula
EXACTA que uso hoy -- no asumas que se llama igual en otro lado solo por el
nombre, compará la fórmula.

## EJE DE SIMETRÍA (axis_sym)

1. Error angular: arccos(|dir_pred · dir_GT|) en grados, ambos vectores
   normalizados, rango [0°,90°] (sign-agnostic).
2. Error de traslación: distancia punto-a-línea entre el origen predicho y el
   eje GT (sin normalizar).
3. Precisión@θ: % de objetos con error angular < θ, para θ ∈ {5°, 10°, 15°}.
4. AUC angular: área bajo la curva "% de objetos con error < t" integrando t
   en [0°, 45°] (100 pasos), normalizada a [0,1]. OJO: el tope de integración
   es 45°, no 90° -- todo error ≥45° cuenta como fallo total.
5. SDE interno ("de aceptación"): muestreo de hasta 1000 VÉRTICES al azar de
   la malla (no ponderado por área); se refleja cada vértice con una rotación
   de 180° sobre el eje predicho; se busca el vecino más cercano DENTRO DE LA
   MISMA MUESTRA de 1000 vértices (KDTree, no contra la malla completa);
   SDE = distancia_media / diagonal_del_bounding_box (lineal, normalizada).
   Se usa solo para un flag binario "accepted" (SDE ≤ 0.05), no se reporta
   como estadístico agregado para eje.
6. SDE_ref (extensión propia, no viene de un repo externo): muestreo de 1000
   puntos de la SUPERFICIE real (ponderado por área), reflexión de 180° igual
   que el punto 5, pero la distancia es al CUADRADO y contra la malla
   triangulada real completa (árbol AABB), SIN normalizar por bbox.
7. F1 para eje: no calculamos ninguno -- ya confirmamos que no existe una
   convención establecida de F1 multi-candidato para ejes en la literatura
   que revisamos. ¿Sigue siendo cierto, o apareció algo nuevo?

## PLANO DE SIMETRÍA (plane_sym)

1. Error angular: igual que el eje, pero contra el plano GT con menor error
   angular (matching por mejor-caso cuando el objeto tiene 1-3 planos GT; no
   se penaliza por planos GT no encontrados, no hay recall real).
2. Error de traslación: distancia punto-a-plano contra el plano GT que ganó
   el match angular.
3. Precisión@θ / AUC angular: misma definición y mismos umbrales que en eje.
4. SDE interno ("de aceptación"): igual mecánica que el punto 5 de eje pero
   reflejando a través del plano predicho (no rotación de 180°).
5. SDE "de resumen" (ESTA ES DISTINTA DE LA ANTERIOR, aunque en mi código
   ambas se llaman "sde"): mean(2 * |producto_punto(vertice - origen_pred,
   normal_pred)|) / diagonal_bbox, sobre TODOS los vértices de la malla, SIN
   reflejar ni buscar vecino -- es literalmente el doble de la distancia
   promedio de cada vértice al plano predicho. Esta es la que uso para
   reportar "SDE promedio" y "AUC de SDE" (tope de integración 0.10) en mis
   tablas comparativas entre experimentos. ¿Esta fórmula corresponde a algo
   reconocido en la literatura de symmetry detection, o es una simplificación
   que no debería presentarse como "SDE" porque no verifica autoconsistencia
   geométrica real (no comprueba que el reflejo caiga sobre superficie real)?
6. SDE_ref (puerto verbatim de metric_SDE.py de un repo de referencia externo,
   EnhancedBackProjection/WACV2026): muestreo de superficie ponderado por
   área, distancia al cuadrado contra la malla real via AABB tree, sin
   normalizar.
7. F1_ref (puerto verbatim de metric_F1.py del mismo repo de referencia):
   para 4 umbrales de distancia {0.05, 0.10, 0.15, 0.20} entre
   representaciones de plano [nx,ny,nz,d] (considerando también el signo
   opuesto del vector), matching contra TODOS los planos GT del objeto,
   acumulando TP/FP/FN globalmente sobre todo el dataset (no promedio por
   objeto), F1 promediado sobre los 4 umbrales. Mi pipeline predice un solo
   plano candidato por objeto/método (sin score de confianza), a diferencia
   del repo de referencia que predice hasta 10 candidatos con score.

## LO QUE NECESITO DE VOS

Para cada una de las 14 métricas de arriba (7 de eje + 7 de plano):
a) ¿Existe en papers/benchmarks recientes (2020-2026) de detección de
   simetría 3D, con esta fórmula o una equivalente? Citá el paper/benchmark
   específico si lo encontrás.
b) Si existe pero con una convención distinta (otro umbral, otra
   normalización, otro tipo de muestreo), señalá la diferencia exacta.
c) Clasificá cada una como: [ESTANDAR RECONOCIDO] / [USADO PERO POCO COMUN,
   especifico de 1-2 papers] / [NO ENCONTRE RESPALDO, parece ad hoc de este
   proyecto].
d) Al final, dame una recomendación honesta: de estas 14, ¿cuáles mantendrías
   para reportar en una tesis de forma que un comité evaluador las reconozca
   como válidas, y cuáles descartarías o reemplazarías por la convención más
   citada?

Sé exhaustivo pero concreto -- prefiero que digas "no encontré nada" a que
inventes una cita. Si una fórmula se parece a algo conocido pero no es
exactamente igual, decilo explícitamente en vez de asumir que son lo mismo.
```
