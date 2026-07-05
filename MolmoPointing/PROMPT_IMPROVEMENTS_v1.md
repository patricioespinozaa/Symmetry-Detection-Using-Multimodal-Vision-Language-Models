# Mejoras implementadas: Prompts v0 → v_1

Este documento describe los cambios aplicados a cada prompt en la transición de la versión original (v00–v05) a la versión mejorada (v00\_1–v05\_1), para simetría axial y planar.

---

## Ejes de mejora comunes a todos los v_1

Tres principios se cruzaron desde los prompts ganadores (`axis_v05` y `plane_v04`) hacia todos los demás:

1. **Split superior/inferior obligatorio** — `obj_id 1` DEBE estar en la mitad superior, `obj_id 2` en la inferior. Evita que ambos puntos colapsen a la misma altura, lo que impediría que SVD recupere la dirección del eje/plano.

2. **Definición explícita de la traza** — *"el eje/plano aparece como el centro horizontal del objeto, equidistante de los bordes izquierdo y derecho del silhouette."* Guía a Molmo hacia el elemento de simetría en vez de bordes o texturas salientes.

3. **Prohibición de silhouette edges** — *"do NOT place points on the silhouette boundary."* Los VLMs tienen un sesgo hacia bordes salientes; esta regla lo contrarresta explícitamente.

---

## Simetría Axial — Cambios específicos por prompt

| Prompt | Problema original | Mejora en v\_1 |
|---|---|---|
| **v00** | Puntos vagamente "sobre el eje", sin anclaje vertical | Upper/lower split forzado + definición centerline + prohibición de silhouette |
| **v01** | Todos los midpoints a la misma altura → SVD encontraba plano horizontal, no el eje | Diversidad de alturas entre vistas: *"elige alturas DISTINTAS — superior en unas vistas, inferior en otras"* |
| **v02** | Devolvía extremos del silhouette (izquierdo + derecho) → puntos en lados opuestos de la superficie → SVD perpendicular al eje | **Rediseño completo**: devuelve dos puntos sobre la centerline a alturas distintas. Cambio de `midpoint` → `independent` |
| **v03** | Pares estructurales sin verificación de que el punto medio esté sobre el eje | Verificación añadida: *"el punto medio debe ser equidistante de ambos bordes del silhouette"* + fallback al corte más ancho |
| **v04** | Polos topmost/bottommost sin verificación de posición horizontal | Verificación: *"traza una línea horizontal — el punto debe ser equidistante de bordes izquierdo y derecho"* + fallback para superficies planas |
| **v05** | Ya era el mejor; identificación del eje implícita | Step-0 explícito: *"identifica el eje global desde TODAS las vistas antes de responder"* + check de consistencia inter-vista |

---

## Simetría Planar — Cambios específicos por prompt

| Prompt | Problema original | Mejora en v\_1 |
|---|---|---|
| **v00** | Puntos sobre la traza sin anclaje superior/inferior ni definición del centro | Top/bottom split explícito + *"la traza pasa por el centro horizontal a cada altura"* |
| **v01** | Misma altura en todos los midpoints → SVD no recupera orientación del plano | Diversidad de alturas: *"varía la altura — superior en algunas vistas, inferior en otras"* + verificación de midpoint sobre la traza |
| **v02** | Seam identificado pero sin consistencia entre vistas | Step-0 global: *"identifica el plano desde TODAS las vistas"* + verificación de consistencia del seam entre vistas + top/bottom forzado |
| **v03** | Pares estructurales sin verificación del punto medio | Verificación: midpoint equidistante de ambos bordes + fallback a silhouette del corte más ancho |
| **v04** | Ya era el mejor; fórmula del midpoint implícita | Fórmula explícita: `X_mid = (X_izq + X_der) / 2` + step-0 identificación global + check de consistencia inter-vista |
| **v05** | *"Puntos más distantes en cualquier dirección"* → Molmo elegía puntos diagonales, no a lo largo de la traza | Reencuadre: *"topmost y bottommost sobre la traza, maximizando distancia VERTICAL específicamente (no diagonal)"* + guía de centro horizontal |

---

## Resumen del impacto por tipo de cambio

| Cambio | Prompts que lo aplicaron | Efecto observado |
|---|---|---|
| Split superior/inferior | v00, v01, v02, v03 (todos) | Mayor cobertura vertical de la nube de puntos → SVD recupera dirección correcta |
| Definición centerline | v00, v04, v05 | Reducción de puntos sobre silhouette; Molmo apunta al interior |
| Diversidad de alturas inter-vista | v01, v03 (modo midpoint) | Midpoints dejan de colapsar al mismo plano horizontal |
| Rediseño completo + cambio de modo | v02 axial | AUC 0.070 → 0.354, n\_obj 27 → 80 |
| Reencuadre "vertical not diagonal" | v05 planar | AUC 0.085 → 0.344 |
| Step-0 global + consistencia inter-vista | v04, v05, v02 planar | Mejora coherencia en modo multi-vista |

---

## Impacto en métricas (SVD, mejor n\_views)

### Simetría Axial

| Prompt | AUC v0 | AUC v\_1 | ΔAUC | n\_obj v0 | n\_obj v\_1 |
|---|---|---|---|---|---|
| v00 | 0.262 | 0.383 | **+0.121** | 74 | 82 |
| v01 | 0.176 | 0.193 | +0.017 | 58 | 61 |
| v02 | 0.070 | 0.354 | **+0.284** | 27 | 80 |
| v03 | 0.242 | 0.237 | −0.005 | 66 | 69 |
| v04 | 0.252 | 0.281 | +0.029 | 59 | 71 |
| v05 | 0.354 | 0.388 | **+0.034** | 92 | 87 |

### Simetría Planar

| Prompt | AUC v0 | AUC v\_1 | ΔAUC | n\_obj v0 | n\_obj v\_1 |
|---|---|---|---|---|---|
| v00 | 0.232 | 0.418 | **+0.186** | 76 | 94 |
| v01 | 0.315 | 0.334 | +0.019 | 77 | 76 |
| v02 | 0.380 | 0.436 | **+0.056** | 94 | 55 |
| v03 | 0.328 | 0.363 | +0.035 | 76 | 71 |
| v04 | 0.407 | 0.505 | **+0.098** | 93 | 66 |
| v05 | 0.179 | 0.344 | **+0.165** | 70 | 64 |

> Los mejores n\_views pueden diferir entre versiones. Datos de `experiments_20_06_2026`.

¿De dónde vino cada mejora?
Basadas en resultados + análisis de failure modes
Estas son las más sólidas y justificadas:

v02 axial — rediseño completo
Vino del análisis de resultados: solo 27/100 objetos válidos con AUC=0.070. La causa identificada fue geométrica: el prompt devolvía extremos del silhouette izquierdo/derecho. Esos puntos, en modo midpoint, generan midpoints laterales que no están sobre el eje, y en modo independent forman una nube perpendicular al eje → SVD encontraba la dirección equivocada. El rediseño a "dos puntos sobre la centerline a distintas alturas" fue una corrección directa al problema geométrico identificado.

v01 axial y v01 planar — diversidad de alturas
Vino del análisis de resultados: buena cobertura de objetos pero AUC bajo. La causa: pares bilaterales en mode midpoint desde vistas distintas, pero Molmo tendía a poner ambos puntos a la misma altura en todas las vistas → todos los midpoints 3D en un mismo plano horizontal → SVD encontraba la normal del plano (vertical) en vez del eje. La instrucción de variar alturas entre vistas fue la corrección directa.

v05 planar — "vertical not diagonal"
Vino del análisis de resultados: el peor prompt planar (AUC=0.085). El problema: "puntos más distantes" era ambiguo y Molmo elegía puntos diagonales en vez de a lo largo de la traza vertical. El reencuadre a "maximizar distancia vertical específicamente" fue la corrección.

Basadas en cross-pollination desde los prompts ganadores
Estas son más heurísticas — se tomaron elementos de axis_v05 y plane_v04 (los mejores) y se aplicaron a todos los demás:

Upper/lower split → venía de v05, se extendió a v00, v01, v02, v03
Definición "horizontal center = eje/plano" → venía de v04/v05, se extendió a todos
Prohibición de silhouette edges → observación de que Molmo tiende a bordes salientes
Estas son justificadas en la medida en que el análisis confirmó que esos elementos contribuían al éxito de v05/v04, pero no hubo ablation study formal que aísle cada elemento.

Las que suenan raras y por qué
v03_1 — fallback al corte más ancho:
Esto fue una adición heurística, no basada en análisis de resultados. El razonamiento fue: "si Molmo no encuentra pares estructurales, al menos que devuelva los extremos del silhouette más ancho". En la práctica resultó en una regresión leve (−0.005 AUC) — probablemente porque el fallback llevó a Molmo a puntos silhouette que son peores que los pares estructurales originales.

v04_1 y v02_1 planares — step-0 global + fórmula explícita:
Fueron adiciones basadas en la hipótesis de que más estructura = mejor. En la práctica, v04_1 fue mejor a n=1 (+0.118) pero peor a n=6 (−0.028). El step-0 global puede haber dificultado que Molmo adapte su estimación a cada vista individual. Es una mejora ambigua.

Conclusión para el markdown
Las mejoras se pueden clasificar en tres categorías de justificación:

Correcciones por failure mode análisis (más fuertes): v02 axial, v01 axial/planar, v05 planar
Cross-pollination desde prompts ganadores (moderadas): upper/lower split, centerline definition en v00, v03, v04
Adiciones heurísticas no validadas (más débiles): fallback de v03_1, step-0 global de v02_1/v04_1
