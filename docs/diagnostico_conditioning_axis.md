# Diagnóstico de conditioning geométrico — eje, pipeline sin malla

> Propósito: antes de invertir en un experimento caro (prompt de 4 puntos,
> triangulación robusta, etc.) para bajar el error angular de `axis_sym` en
> el pipeline sin malla — que ronda 58-65° en las 14 variantes de prompt ya
> evaluadas (`results/experiments_*/axis_sym_nomesh_comparison.csv`) — se
> corrió un diagnóstico de costo cero (sin llamadas nuevas a Molmo2) para
> aislar SI el cuello de botella es geométrico (conditioning del sistema de
> triangulación) o de otra naturaleza. Script:
> [`Mapping/diagnose_axis_conditioning.py`](../Mapping/diagnose_axis_conditioning.py).
> Corrida sobre `axis_v06_nomesh` (el mejor prompt del sweep, ver
> `docs/pipeline_sin_malla.md` y el análisis de ranking en la conversación
> de diseño de prompts — no versionado como doc separado).

---

## 1. Motivación

En una revisión anterior de resultados se propuso una hipótesis de **"lever
arm"** para explicar por qué, en el pipeline **con malla**
(`Mapping/estimate_symmetry.py`), la traslación del eje se estima bien pero
la dirección no: si los puntos usados para el ajuste están espacialmente
poco separados entre sí, el mismo ruido de localización por punto produce un
error angular mucho mayor (la dirección de una recta ajustada a puntos
"apretados" es mucho más sensible al ruido que la de una recta ajustada a
puntos muy separados).

Ese diagnóstico **no se traslada literalmente** al pipeline **sin malla**
(`Mapping/estimate_symmetry_no_mesh.py`): acá nunca se reconstruyen puntos 3D
individuales — cada vista aporta un **plano de interpretación** (normal
`n_i`, definido por el centro de cámara y dos rayos 2D→3D;
`pipeline_common.triangulation.interpretation_plane_normal`), y la dirección
del eje sale del vector nulo de la matriz de esas normales apiladas
(`triangulate_line`, SVD). Antes de correr un experimento de 4 puntos había
que:

1. Traducir correctamente la hipótesis de "lever arm" a esta geometría
   (no hay puntos 3D que medir directamente).
2. Contrastarla contra una hipótesis alternativa igual de plausible:
   que el problema sea de **conditioning multi-vista** (vistas casi
   coplanares/colineales, ya anticipado como "Riesgo conocido" en
   `docs/pipeline_sin_malla.md` §3.1), no de separación de puntos dentro de
   cada vista.
3. Descartar ambas antes de gastar cómputo/prompts nuevos, si ninguna
   resulta ser la causa real.

---

## 2. Métricas evaluadas, justificación y literatura

Las tres métricas se calculan **reprocesando datos ya generados**
(`molmo_multiview_<EXP>.json`, `predicted_symmetry_<EXP>.json`,
`eval_..._triangulation_results.json`) — cero llamadas nuevas al VLM.

### 2.1 `pixel_sep_mean` / `pixel_sep_min` — separación en píxeles dentro de una vista

**Qué mide:** distancia euclidiana en píxeles entre los dos puntos que
`widest_pair` selecciona en cada vista, promediada (o mínima) entre las
vistas usadas para ese objeto.

**Por qué debería importar:** cada vista construye su plano de interpretación
a partir de dos puntos 2D + el centro de cámara. Si esos dos puntos están
muy juntos en la imagen, el mismo error de localización de Molmo2 (siempre
presente) produce una normal `n_i` mucho más ruidosa — es el argumento
clásico de "baseline corto → error de profundidad/triangulación alto" en
visión estéreo.

**Literatura:** Hartley, R., & Zisserman, A. (2004). *Multiple View
Geometry in Computer Vision* (2nd ed.). Cambridge University Press —
capítulo de triangulación: el error de triangulación escala aproximadamente
con `1/sin(θ)`, donde `θ` es el ángulo/separación entre las observaciones
usadas.

**Signo esperado:** negativo (más separación en píxeles → menos error
angular).

### 2.2 `cond_number` (σ1/σ2) y `mean_pairwise_angle_deg` — conditioning multi-vista

**Qué mide:** `cond_number` es el cociente entre el mayor y el segundo mayor
valor singular de la matriz de normales `N` (k vistas × 3) que
`triangulate_line` usa para recuperar la dirección del eje como su vector
nulo. `mean_pairwise_angle_deg` es el ángulo promedio (en grados,
sign-agnostic) entre todos los pares de normales — versión interpretable de
lo mismo.

**Por qué debería importar:** en el caso ideal sin ruido, `N` tiene rango 2
exacto (el eje real está contenido en todos los planos) y la dirección queda
bien determinada siempre que `σ2` (el segundo valor singular) sea grande. Si
las cámaras usadas están casi coplanares o colineales — el "Riesgo conocido"
que ya predice `docs/pipeline_sin_malla.md` §3.1 sin haberlo cuantificado
nunca — las normales `n_i` tienden a ser casi paralelas entre sí, `σ2` se
acerca a `σ3`≈0, y pequeñas perturbaciones de ruido en los puntos 2D pueden
rotar mucho la dirección recuperada (el vector nulo queda mal determinado).

**Literatura:** Golub, G. H., & Van Loan, C. F. *Matrix Computations* (4th
ed.), Johns Hopkins University Press — capítulo de SVD y mínimos cuadrados
totales (TLS): el número de condición `σ1/σ2` acota la sensibilidad del
vector nulo recuperado ante perturbaciones de la matriz de entrada. Este es
también el argumento estándar detrás del riesgo de configuraciones de cámara
degeneradas en triangulación multivista (Bartoli, A., & Sturm, P. (2005),
ya citado en `pipeline_common/triangulation.py::interpretation_plane_normal`
para la formalización de "planos de interpretación").

**Signo esperado:** `cond_number` positivo (más mal condicionado → más
error); `mean_pairwise_angle_deg` negativo (más ángulo entre normales →
menos error).

### 2.3 `axis_span` — el "lever arm" real, traducido a esta geometría

**Qué mide:** dado el eje YA ajustado (`origin`, `direction`), para cada
punto 2D observado se calcula el punto de la recta del eje más cercano al
rayo cámara→punto (fórmula estándar de closest-point entre dos rectas en
3D, `closest_point_on_line_to_line`), y se proyecta ese punto sobre el eje.
`axis_span` es el rango (máximo − mínimo) de esas proyecciones entre todas
las observaciones usadas, en unidades del mundo.

**Por qué debería importar:** es la traducción correcta de la hipótesis de
"lever arm" original (que asumía puntos 3D explícitos vía ray-casting, cosa
que este pipeline no tiene). El argumento subyacente es el resultado
clásico de regresión/errores-en-variables: la varianza de una
pendiente/dirección ajustada por mínimos cuadrados es inversamente
proporcional a la dispersión de los regresores a lo largo del eje de
ajuste — a igual ruido por observación, observaciones más separadas dan una
dirección mejor determinada.

**Literatura:** Draper, N. R., & Smith, H. *Applied Regression Analysis*
(3rd ed.), Wiley — el argumento de leverage/dispersión del regresor sobre la
precisión de la pendiente estimada; York, D. (1966). "Least-squares fitting
of a straight line", *Canadian Journal of Physics*, 44(5) — mínimos
cuadrados totales con error en ambas variables, mismo argumento generalizado
a errores-en-variables.

**Signo esperado:** negativo (más `axis_span` → menos error angular).

### 2.4 Método de correlación

Pearson y Spearman (este último implementado a mano sobre rangos, sin
`scipy`, siguiendo la norma del repo de evitar esa dependencia cuando es
evitable — ver `CLAUDE.md`), calculados por separado por `n_views` y sobre
el pool completo. Umbral de lectura: `|r| ≥ 0.3` con el signo esperado se
interpreta como evidencia de que el mecanismo correspondiente contribuye al
error angular.

---

## 3. Resultados (corrida sobre `axis_v06_nomesh`, 2457 filas objeto×n_views)

| predictor | n=6 (Pearson/Spearman) | n=14 | n=26 | pooled | signo esperado |
|---|---|---|---|---|---|
| `pixel_sep_mean` | +0.05 / +0.06 | +0.05 / +0.05 | −0.00 / +0.01 | +0.03 / +0.04 | negativo |
| `pixel_sep_min` | +0.13 / +0.13 | +0.08 / +0.07 | −0.06 / −0.05 | +0.04 / +0.05 | negativo |
| `cond_number` | −0.02 / −0.04 | −0.03 / −0.04 | −0.10 / −0.11 | −0.02 / −0.04 | **positivo** |
| `mean_pairwise_angle_deg` | +0.10 / +0.09 | +0.11 / +0.12 | +0.07 / +0.06 | +0.09 / +0.08 | negativo |
| `axis_span` | +0.01 / +0.06 | +0.02 / +0.16 | +0.04 / +0.04 | +0.01 / +0.08 | negativo |

**Ninguna métrica alcanza el umbral `|r| ≥ 0.3`.** La correlación más alta
observada es `axis_span` a n=14 (Spearman +0.16), y va en el sentido
**contrario** al esperado (se esperaba negativo: más span → menos error;
salió positivo, débil). `cond_number` incluso tiene el signo invertido
respecto a lo esperado en las tres n_views, aunque con magnitud despreciable
(máximo −0.11). En criollo: **no hay señal geométrica** en ninguna de las
tres hipótesis.

---

## 4. Implicaciones

1. **Se descarta la hipótesis de "lever arm"** (`axis_span`) para este
   pipeline: la dispersión espacial de las observaciones a lo largo del eje
   no predice el error angular por objeto. Un prompt de 4 puntos diseñado
   para maximizar esta dispersión (el diseño de 3 brazos A/B/C propuesto
   originalmente para el pipeline con malla) **no tiene base empírica** en
   este pipeline sin malla — no vale la pena correrlo con ese objetivo.

2. **Se descarta también la hipótesis de conditioning multi-vista**
   (`cond_number`, `mean_pairwise_angle_deg`): el "Riesgo conocido" de
   `docs/pipeline_sin_malla.md` §3.1 (cámaras casi coplanares/colineales)
   está formalmente bien fundado, pero **no es lo que está limitando el
   error hoy** — de lo contrario, `n_views` más altos (más vistas, más
   chance de buena diversidad angular) deberían mostrar mejor conditioning
   correlacionado con menor error, y no se observa. Un ajuste robusto o
   ponderado en `triangulate_line` (Huber, RANSAC sobre las normales)
   **no atacaría la causa real** del error residual.

3. **Conclusión más importante**: el error angular alto y uniforme (~58-65°
   en las 14 variantes de prompt, incluyendo la mejor, `axis_v06`, en 57.96°)
   **no es explicado por ningún mecanismo de sensibilidad al ruido**
   (variance) que este diagnóstico pueda capturar. Esto es consistente con
   una explicación de **sesgo (bias), no de varianza**: si Molmo2 señala
   consistentemente estructuras 2D distintas entre vistas para lo que el
   prompt cree que es "el mismo punto físico" (inconsistencia de
   **identidad** entre vistas, no de precisión de localización), el sistema
   de triangulación puede estar perfectamente bien condicionado
   (`cond_number` bajo, `axis_span` alto) y aun así producir una dirección
   sistemáticamente equivocada — un sistema bien condicionado construido
   sobre correspondencias erróneas da una respuesta segura pero incorrecta,
   no una respuesta ruidosa. Esta distinción entre error por ruido (atacable
   con mejor conditioning/más observaciones) y error por correspondencias
   erróneas (que ningún ajuste robusto "normal" resuelve si la mayoría de las
   vistas, no una minoría, está afectada) es clásica en visión por
   computador — ver Fischler, M. A., & Bolles, R. C. (1981). "Random sample
   consensus: A paradigm for model fitting with applications to image
   analysis and automated cartography", *Communications of the ACM*, 24(6)
   — RANSAC asume una minoría de outliers; si la inconsistencia de identidad
   afecta a la mayoría de las vistas de forma leve y sistemática (no un
   pequeño subconjunto claramente erróneo), la robustificación estándar no
   ayuda.

---

## 5. Próximos pasos

1. **Priorizar el experimento híbrido de verificación de identidad
   cross-view** (`hybrid_v08`, ya propuesto para axis y plane en la ronda de
   diseño de prompts) sobre cualquier variante geométrica (4 puntos, ajuste
   robusto) — es la hipótesis que este diagnóstico deja como la más
   plausible y todavía no probada directamente.

2. **Diagnóstico de seguimiento, también de costo cero**, antes de correr el
   híbrido: en vez de medir conditioning, medir **consistencia de
   identidad** directamente — para cada vista y cada uno de los 2 puntos,
   calcular la distancia perpendicular del punto (retro-proyectado como
   rayo) al eje YA ajustado (reusando el mismo
   `closest_point_on_line_to_line` ya implementado, pero mirando el
   *residuo* perpendicular en vez de la proyección a lo largo del eje). Si
   el error viene de inconsistencia de identidad, se esperaría ver, para un
   mismo objeto, algunos puntos con residuo muy bajo (vistas donde Molmo
   señaló bien) mezclados con residuos muy altos (vistas donde señaló otra
   estructura) — una distribución bimodal/con outliers claros, distinta de
   un ruido gaussiano uniforme. Esto se puede extender fácilmente a partir
   de `Mapping/diagnose_axis_conditioning.py` (ya tiene los rayos y el eje
   ajustado disponibles en `compute_axis_span`).

3. Si el diagnóstico de consistencia de identidad (paso 2) muestra la
   distribución bimodal esperada, correr `hybrid_v08` (Flow C: descripción
   verbal del punto + verificación de que es la misma estructura física en
   otra vista) y comparar la distribución de residuos antes/después — la
   métrica de éxito no debería ser solo `angular_error_mean` agregado, sino
   la **reducción de la cola de residuos altos** (outliers de identidad),
   igual que se sugirió para el diseño original de 3 brazos, adaptado a esta
   nueva evidencia.

4. **No repetir este diagnóstico de conditioning para otros prompts de eje**
   salvo que cambie sustancialmente el mecanismo de selección de puntos
   (`widest_pair` es compartido por las 14 variantes actuales) — el
   resultado de "sin señal geométrica" es estructural al método de
   triangulación, no específico de `axis_v06`.

---

## 6. Diagnóstico complementario: geometría del set de cámaras vs. eje GT

Script: [`Mapping/diagnose_view_geometry.py`](../Mapping/diagnose_view_geometry.py).
A diferencia de la sección 2-5 (que mide el conditioning de las normales
construidas con los puntos que **devolvió Molmo2**), este mide algo previo e
independiente de la respuesta del VLM: qué tan alineado está, respecto al
eje GT real de cada objeto, el propio set de cámaras (Fibonacci sphere
sampling) que se le mandó a Molmo2. Se verificó primero, a partir de los
`.txt` de `data/objects/curated_axis_sym_obj/`, que la dirección del eje GT
**sí varía sustancialmente entre objetos** (no todos alineados al mismo eje
del mundo) — condición necesaria para que esto sea un predictor por-objeto
válido y no una constante del dataset.

**Hipótesis a testear:** una cámara que mira casi exactamente a lo largo del
eje real ("de punta", *end-on*) ve el eje proyectado casi como un punto, no
como una línea — cualquier par top/bottom que devuelva Molmo2 en esa vista
debería producir un plano de interpretación casi degenerado. Signo
esperado: más ángulo cámara-eje (más perpendicular, "vista de manual") →
menos error; más fracción de vistas "de punta" → más error.

### 6.1 Resultado (corrida sobre `axis_v06_nomesh`, 2457 filas)

| predictor | n=6 | n=14 | n=26 | pooled | signo esperado |
|---|---|---|---|---|---|
| `min_view_axis_angle_deg` | +0.17 / +0.15 | +0.15 / +0.09 | +0.03 / +0.02 | +0.11 / +0.08 | negativo |
| `mean_view_axis_angle_deg` | **+0.34 / +0.34** | +0.22 / +0.22 | **+0.31 / +0.31** | +0.25 / +0.26 | negativo |
| `median_view_axis_angle_deg` | +0.21 / +0.22 | +0.16 / +0.16 | +0.04 / +0.02 | +0.15 / +0.14 | negativo |
| `frac_degenerate` | −0.31 / −0.15 | −0.20 / −0.11 | −0.07 / −0.07 | −0.21 / −0.11 | **positivo** |

Comparación directa (n_views=6): objetos CON ≥1 vista "de punta" → media
51.05°/mediana 54.64° (n=231) vs. SIN ninguna → media 58.74°/mediana 59.70°
(n=593). Mismo patrón, más atenuado, en n=14 y n=26.

**Las cuatro métricas tienen el signo invertido respecto a la hipótesis.**
`mean_view_axis_angle_deg` es la señal más fuerte de todo el diagnóstico
(propio + el de la sección 2-5) — pero en la dirección contraria: **más
perpendicularidad promedio del set de cámaras correlaciona con MÁS error, no
menos**, y tener al menos una vista "de punta" correlaciona con **menos**
error.

### 6.2 Interpretación

La explicación más plausible no es que la triangulación se beneficie de mala
geometría, sino que **la variable de confusión es la ambigüedad del
objetivo semántico, no el conditioning matemático**:

- Una vista casi *end-on* (mirando a lo largo del eje) muestra al objeto
  desde "arriba"/"abajo": para un objeto de revolución, esa vista tiene un
  centro visualmente inequívoco (el centro de una silueta aproximadamente
  circular) — un blanco fácil de señalar con precisión en píxeles, aunque
  geométricamente esa vista aporte poca información angular sobre el eje.
- Una vista mayoritariamente perpendicular ("de manual", la que en teoría
  triangula mejor) muestra el cuerpo lateral completo del objeto — y ahí es
  exactamente donde vive la ambigüedad que ya motivó el fallback de
  curvatura de `axis_v06` (tapas redondeadas, sin un único "punto donde el
  eje sale de la superficie" claramente definido). Más vistas de este tipo
  no mejoran la triangulación si lo que degradan es la **identidad
  consistente** del punto señalado (Molmo puede elegir un punto ligeramente
  distinto de la región curva en cada vista lateral, mientras que en la
  vista *end-on* casi no hay margen de ambigüedad: el centro es el centro).

Esto **no contradice, sino que refuerza**, la conclusión de la sección 4: el
cuello de botella sigue sin ser conditioning geométrico (ni el de la sección
2-5, ni este). La señal que sí aparece acá es consistente con que la
dificultad está en la **identificabilidad semántica del punto pedido en
vistas laterales/perpendiculares**, no en la matemática de triangulación ni
en el muestreo de cámaras en sí.

### 6.3 Implicación práctica

**No rediseñar el muestreo de vistas** para excluir ángulos "de punta" — la
evidencia dice lo contrario de lo que se hubiera recomendado ingenuamente
(esas vistas, si acaso, ayudan). El camino sigue siendo el de la sección 5:
priorizar el diagnóstico de consistencia de identidad (residuo perpendicular
al eje por punto, buscando la distribución bimodal) y el experimento híbrido
`hybrid_v08` de verificación cross-view — con un matiz nuevo: si `hybrid_v08`
funciona, debería beneficiar desproporcionadamente a las vistas
**perpendiculares/laterales** (donde vive la ambigüedad), no a las *end-on*
(que ya funcionan bien). Vale la pena, al re-evaluar `hybrid_v08`,
desagregar el error por `mean_view_axis_angle_deg` (alto vs. bajo) en vez de
solo mirar el agregado — si la mejora se concentra en el grupo de ángulo
alto, confirma esta lectura.
