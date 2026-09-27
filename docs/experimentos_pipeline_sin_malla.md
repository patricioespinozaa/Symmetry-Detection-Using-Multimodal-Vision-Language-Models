# Experimentos del pipeline sin malla: prompts (Flujo A), metodología baseline y ablaciones

> Documento de síntesis para la escritura de tesis. No introduce información
> nueva — reorganiza y cita lo que ya está documentado en
> `docs/pipeline_sin_malla.md`, `docs/implementacion_pipeline_sin_malla.md`,
> `docs/implementacion_pipeline_experiments.md`,
> `docs/diagnostico_conditioning_axis.md`,
> `docs/verificacion_metricas_literatura.md`, `MolmoPointing/Experiments.md`,
> `MolmoPointing/PROMPT_IMPROVEMENTS_v1.md`, `MolmoPointing/prompts_registry.py`,
> `Pipeline_Experiments/README.md`, los docstrings de
> `Pipeline_Experiments/triangulation_variants/*.py`, y los CSV de
> `experiments_15_09_2026/` (corrida real más reciente, post-fix del gate NaN
> de `expF`, 2026-09-15). Cada afirmación cuantitativa de la sección 5 fue
> recalculada directamente de esos CSV para esta sesión — no está copiada de
> ningún resumen previo.
>
> **No existía, antes de este documento, un único `.md` que cubriera los tres
> temas juntos** (qué buscó cada prompt de Flujo A, cómo funciona el baseline
> sin malla, y qué resultado dieron las ablaciones) — la información vivía
> repartida en los documentos de arriba. Este documento las une; para el
> detalle completo de cualquier sección, seguir la referencia a la fuente
> original.

---

## 1. Alcance

1. **Prompts del Flujo A** (`MolmoPointing/`): qué se le pidió a Molmo2 en
   cada versión, qué buscaba corregir respecto a la anterior, y cómo salió
   rankeada en la corrida más reciente (§2).
2. **Metodología del pipeline sin malla** (baseline, `Mapping/estimate_symmetry_no_mesh.py`):
   cómo se estima eje/plano sin ray-casting contra la malla (§3).
3. **Experimentos asociados** (`Pipeline_Experiments/`, ablaciones sobre esa
   metodología) y **las métricas que dieron** en `experiments_15_09_2026/`
   (§4–§5).
4. Referencias bibliográficas de todo lo anterior, consolidadas en un solo
   lugar (§8).

**Qué NO cubre**: la definición formal de cada métrica (`angular_error_deg`,
`AUC`, `SDE_ref`, `F1_ref`) — eso vive en `docs/metricas_evaluacion.md` y se
referencia, no se repite. Tampoco cubre el pipeline **con** malla
(`Mapping/estimate_symmetry.py`, `map_to_3d.py`) más allá de mencionarlo como
punto de comparación — ver `docs/actualizacion_metricas.md` para ese.

---

## 2. Flujo A: prompts utilizados

### 2.1 Qué es "Flujo A"

`MolmoPointing/molmo_multiview_runner.py` soporta tres flujos ortogonales
(`--flow a/b/c`), que controlan cuánto contexto semántico se inyecta antes de
la llamada de *pointing*:

- **Flujo A** (default): *pointing* directo — el prompt versionado (`v00`…`v08`)
  es la única llamada a Molmo2 por vista. Es, por diseño, **byte-a-byte
  idéntico al comportamiento original** del runner (`docs/features/molmo-pointing.md`
  §Key Decisions) — Flujo B/C se agregaron sin tocar esta ruta.
- **Flujo B**: una llamada previa (*pre-pass*, una sola vista) le pide a
  Molmo2 una descripción libre del objeto; esa descripción se antepone al
  prompt de Flujo A (`augment_prompt_with_description`).
- **Flujo C**: la misma pre-pass, pero además extrae una pista cualitativa de
  ubicación (`Axis location:`/`Plane location:`) y **reconstruye** el prompt
  principal desde cero con esa información (`build_flow_c_prompts`) — no es
  una vista sin roles, ya no promete identidad persistente entre vistas
  (`obj_id` es un enumerador por imagen, no un landmark rastreado).

**Este documento se centra en Flujo A** — es el que corrió sobre las 19
variantes de prompt de eje y 19 de plano a escala completa. Flujo B/C solo se
corrieron sobre el mejor prompt de cada tipo (`axis_v05_1`, `plane_v04_1`) y
aparecen en los resultados de §5 marcados explícitamente como tal.

### 2.2 Prompts de eje (`axis_sym`)

Fuente: `MolmoPointing/prompts_registry.py::DESCRIPTIONS`,
`MolmoPointing/Experiments.md` §Registered prompts.

| ID | Qué le pide a Molmo2 | `--point-mode` |
|---|---|---|
| `axis_v00` | Dos puntos bien separados **directamente sobre** el eje proyectado | `independent` |
| `axis_v01` | Punto izquierdo + su espejo derecho, equidistantes del eje (par bilateral) | `midpoint` |
| `axis_v02` | Extremos de la silueta más ancha (izquierdo + derecho) | `midpoint` |
| `axis_v03` | Pares de elementos estructurales simétricos (asas, agujeros, nervaduras) | `midpoint` |
| `axis_v04` | Puntos polares — donde el eje "sale" de la superficie (arriba/abajo) | `independent` |
| `axis_v05` | Centerline del eje — un punto en la mitad superior + uno en la inferior | `independent` |
| `axis_v00_1` | v00 + split superior/inferior forzado + definición de centerline + prohibición de bordes de silueta | `independent` |
| `axis_v01_1` | v01 + verificación de que el midpoint cae en la centerline + diversidad de altura entre vistas | `midpoint` |
| `axis_v02_1` | Rediseño: midpoints de centerline a la altura más ancha y en polos (ya no extremos de silueta) — **cambia a** `independent` | `independent` |
| `axis_v03_1` | v03 + verificación de midpoint + fallback a corte más ancho | `midpoint` |
| `axis_v04_1` | v04 + verificación de centro horizontal + fallback para superficies planas | `independent` |
| `axis_v05_1` | v05 + paso 0 explícito de identificación global del eje + chequeo de consistencia entre vistas | `independent` |
| `axis_v06` | Polos de v04_1 + regla explícita de "centro de curvatura" para tapas redondeadas | `independent` |
| `axis_v07` | Corte **más angosto** (cuello/cintura) — inversión del ancla "más ancha" de v02 | `independent` |
| `axis_v08` | Polos + auto-verificación anti-colinealidad: STEP1 (geometría global) + STEP2 (por vista) + self-check "si ambos puntos caen en X≈500, la cámara probablemente no mira de frente al eje" | `independent` |
| `axis_v06_6pts` | v06 extendido a 6 puntos a lo largo del eje (polos + 4 intermedios), exige puntos dentro del objeto | `independent` (compatible con `widest_pair`) |
| `axis_v07_6pts` | v07 extendido a 6 puntos con self-check explícito de colinealidad | `independent` |
| `axis_lit2_grid` | Grilla 5×5 (A–E × 1–5) superpuesta al render — ancla por celda antes de coordenada fina, estilo MOKA | `independent` |
| `axis_lit3_cot` | Chain-of-Thought estructurado tipo scene-graph: declara categoría de objeto + orientación global + definición de los polos **antes** de señalar coordenadas | `independent` |

### 2.3 Prompts de plano (`plane_sym`)

| ID | Qué le pide a Molmo2 | `--point-mode` |
|---|---|---|
| `plane_v00` | Puntos arriba/abajo **sobre** la intersección visible del plano con la superficie | `independent` |
| `plane_v01` | Punto izquierdo + su espejo derecho, equidistantes del plano (par bilateral) | `midpoint` |
| `plane_v02` | Dos puntos directamente sobre la costura (*seam*) visible entre las dos mitades espejadas | `independent` |
| `plane_v03` | Elementos estructurales correspondientes a ambos lados del plano (patas, ruedas, brazos) | `midpoint` |
| `plane_v04` | Centro horizontal del ancho del objeto, cerca de arriba + cerca de abajo | `independent` |
| `plane_v05` | Los dos puntos **más distantes** a lo largo de la traza visible del plano | `independent` |
| `plane_v00_1` | v00 + split top/bottom explícito + guía de centro horizontal | `independent` |
| `plane_v01_1` | v01 + diversidad de altura entre vistas + verificación de midpoint sobre la traza | `midpoint` |
| `plane_v02_1` | v02 + paso 0 global + verificación de consistencia de la costura entre vistas + top/bottom forzado | `independent` |
| `plane_v03_1` | v03 + verificación de midpoint + fallback a corte más ancho | `midpoint` |
| `plane_v04_1` | v04 + paso 0 de identificación global del plano + fórmula explícita `X_mid=(X_izq+X_der)/2` + chequeo de consistencia entre vistas | `independent` |
| `plane_v05_1` | v05 reencuadrado: "maximizar distancia **vertical**, no diagonal" + guía de centro horizontal | `independent` |
| `plane_v06` | v02 (costura) + restricción explícita de no-bilateralidad + fallback estilo v04 | `independent` |
| `plane_v07` | Fórmula de midpoint de v04_1 anclada a la **orientación propia del plano**, no al vertical de la imagen | `independent` |
| `plane_v08` | Costura + auto-verificación anti-colinealidad (STEP1/STEP2 + self-check análogo a `axis_v08`) | `independent` |
| `plane_v04_1_6pts` | v04_1 extendido a 6 midpoints a lo largo de la traza, exige puntos dentro del objeto | `independent` (requiere extender la estimación a 3 pares, no solo `obj_id` 1/2) |
| `plane_v05_6pts` | v05 extendido a 6 puntos con self-check explícito de colinealidad | `independent` |

### 2.4 Ronda v0 → v1: qué cambió y por qué

Fuente: `MolmoPointing/PROMPT_IMPROVEMENTS_v1.md`. Tres principios cruzados
desde los prompts ganadores (`axis_v05`, `plane_v04`) a todos los demás:
**(1)** split superior/inferior obligatorio (evita que ambos puntos colapsen
a la misma altura, lo que le impide a SVD recuperar la dirección),
**(2)** definición explícita de la traza ("centro horizontal, equidistante de
ambos bordes"), **(3)** prohibición explícita de poner puntos sobre el borde
de la silueta (contrarresta el sesgo conocido de los VLM hacia bordes
salientes).

Impacto medido (SVD, mejor `n_views`, dataset `experiments_20_06_2026` —
**corrida con malla**, anterior al pipeline sin malla; se cita tal cual la
documentó `PROMPT_IMPROVEMENTS_v1.md`, no recalculada acá):

| Prompt | ΔAUC (v0→v1) | Causa identificada / tipo de justificación |
|---|---|---|
| `axis_v02` | **+0.284** (0.070→0.354, n_obj 27→80) | Failure-mode real: v0 devolvía extremos de silueta izq/der → nube perpendicular al eje, SVD encontraba la dirección equivocada. Rediseño completo a centerline. **Justificación fuerte.** |
| `plane_v00` | +0.186 | Cross-pollination desde v04/v05 (split + definición de traza) |
| `plane_v05` | **+0.165** | Failure-mode real: "puntos más distantes" era ambiguo, Molmo elegía diagonales. Reencuadre a "vertical no diagonal". **Justificación fuerte.** |
| `axis_v00` | +0.121 | Cross-pollination |
| `plane_v04` | +0.098 | Cross-pollination + fórmula explícita |
| `plane_v02` | +0.056 | Cross-pollination |
| `plane_v03` | +0.035 | Cross-pollination |
| `axis_v05` | +0.034 | Ya era el mejor; solo se agregó paso-0 |
| `axis_v04` | +0.029 | Cross-pollination |
| `plane_v01` | +0.019 | Diversidad de alturas (failure-mode real, pero mejora chica) |
| `axis_v01` | +0.017 | Diversidad de alturas |
| `axis_v03` | **−0.005** | Adición heurística no validada (fallback a corte más ancho) — regresión leve, documentada como tal |

El propio documento clasifica las mejoras en tres niveles de confianza:
**correcciones por análisis de failure-mode** (`axis_v02`, diversidad de
alturas en `v01`, reencuadre de `plane_v05` — las más sólidas, con causa
identificada y corrección directa), **cross-pollination heurística** desde
los prompts ganadores (split superior/inferior, definición de centerline —
razonables pero sin *ablation study* que aísle cada elemento) y **adiciones
no validadas** (`axis_v03_1`'s fallback, `axis_v02_1`/`plane_v04_1`'s paso-0
global — resultado ambiguo, en `plane_v04_1` mejor a n=1 pero peor a n=6).

### 2.5 Rondas 2–5: motivación de `v06`–`v08`, 6pts, lit2/lit3

- **Ronda 2** (`v06`/`v07` eje, `v06`/`v07` plano) — data-driven, sobre el
  ranking v0–v05_1 ya corrido: `axis_v06` prueba si el fallback de
  "centro de curvatura" ayuda en tapas redondeadas; `axis_v07` invierte
  deliberadamente el ancla "más ancha" de v02 por "más angosta" (cuello/cintura)
  como contraste controlado.
- **Ronda 3** (`_6pts`) — ver `docs/diagnostico_conditioning_axis.md` §7:
  motivada por la hipótesis (luego **descartada empíricamente**, ver
  `docs/diagnostico_conditioning_axis.md` §4) de que más puntos por vista
  mejorarían el *conditioning* de la triangulación. Exige explícitamente que
  los 6 puntos caigan dentro del objeto.
- **Ronda 4** (`v08`) — anti-degeneración explícita: además de STEP1/STEP2,
  agrega el self-check "si ambos puntos caen en X≈500, la cámara
  probablemente no mira de frente al eje" — respuesta directa al diagnóstico
  de colapso a centro de imagen que hace `Pipeline_Experiments/diagnostics/diagnose_center_bias.py`
  (ver §4.4; **pendiente de confirmar si `v08` efectivamente reduce ese
  colapso — el diagnóstico no se ha corrido todavía sobre él**, ver §6).
- **Ronda 5** (`axis_lit2_grid`, `axis_lit3_cot`) — primera implementación
  **real** (llamando a Molmo2 de verdad) de dos estrategias de literatura que
  antes solo se habían *simulado* con transformaciones geométricas sobre
  predicciones ya existentes, en `Experiments/sandbox_literatura.ipynb`:
  Grid Overlay (estilo MOKA, arXiv:2403.03174) y Structured CoT
  (arXiv:2507.13362).

---

## 3. Metodología cuando no se usa la malla (baseline)

Fuente completa: `docs/pipeline_sin_malla.md` (diseño) y
`docs/implementacion_pipeline_sin_malla.md` (implementación, estado
`✅ IMPLEMENTADO Y VALIDADO`).

### 3.1 Motivación del rediseño

El pipeline original (paper ECCV 2026 / tesis) convierte los puntos 2D de
Molmo2 a 3D vía *ray casting* contra la malla real (`Mapping/map_to_3d.py`) y
ajusta SVD/RANSAC sobre esos puntos 3D. Aunque Molmo2 nunca ve la malla al
señalar, **la predicción final sí depende de ella** — la retroproyección es
parte constitutiva de la estimación, no solo de la validación. Esta fue la
objeción central de una review de ECCV 2026 (citada textual en el documento
de diseño): *"A 2D point defines a ray, not a 3D location, so this step
cannot be applied to photographs alone without depth estimation or
cross-view triangulation."*

**Objetivo**: mover la malla completamente fuera de la ruta de estimación.
El eje/plano debe surgir únicamente de los puntos 2D de Molmo2 + los
parámetros de cámara ya conocidos (generados al renderizar) — la malla queda
reservada exclusivamente para el paso final de validación (`Mapping/evaluate.py`,
sin cambios de diseño).

### 3.2 Por qué comparar en 2D no alcanza

Descartado explícitamente en el diseño: comparar la traza 2D predicha contra
la proyección 2D del ground truth en una sola vista **no** valida la simetría
3D recuperada — bajo proyección, una vista 2D es consistente con una familia
entera de interpretaciones 3D distintas (ambigüedad clásica de
retroproyección). La solución es **triangulación multivista real**: combinar
≥2 vistas calibradas para resolver la ambigüedad geométricamente.

### 3.3 Eje — triangulación de líneas (implementado, `estimate_axis_no_mesh`)

Por cada vista, con la cámara ya calibrada (`R`, `T`, FoV — en
`metadata_all.json`), se construye el **plano de interpretación**: el plano
en 3D que contiene el centro de cámara y la línea 2D que forman los dos
puntos que Molmo2 devolvió en esa vista
(`pipeline_common/triangulation.py::interpretation_plane_normal`). Con ≥2
vistas, se intersectan esos planos por mínimos cuadrados (SVD,
`triangulate_line`) para recuperar la línea 3D del eje.

**Por qué funciona sin correspondencia punto-a-punto entre vistas**: el eje
de rotación es invariante al punto de vista — un objeto con simetría
rotacional se ve bilateralmente simétrico respecto a la proyección del eje
**desde cualquier ángulo de cámara**. No hace falta rastrear qué punto
específico corresponde a cuál entre vistas, solo que la línea 2D por vista
aproxime la proyección del eje.

**`widest_pair`** (nueva función, no existía en el pipeline con malla): de
todos los puntos que trae una vista (sin importar `obj_id`), toma el par con
mayor distancia euclidiana en píxeles. Con exactamente 2 puntos (Flujo A) es
idéntico a tomar los roles fijos "arriba"/"abajo" — no cambia el resultado
de los 12 prompts v00–v05_1 ya validados con roles fijos. Generaliza además
la selección para habilitar Flujo C (que devuelve puntos sin rol fijo).

**Riesgo conocido**: mal condicionamiento cuando los centros de cámara y la
línea son casi coplanares/colineales (Hartley 1997; Hartley & Zisserman
2004) — el muestreo Fibonacci debería hacerlo infrecuente, pero se
verificó empíricamente (ver §6, `docs/diagnostico_conditioning_axis.md`: **no
se encontró señal de que esto explique el error residual**).

### 3.4 Plano — consolidación secuencial multi-plano (implementado, `detect_planes_no_mesh`)

Más difícil que el eje: la simetría de reflexión **no** tiene la garantía
"silueta simétrica desde cualquier ángulo" que sí tiene la rotacional — la
silueta de un objeto espejo-simétrico solo es bilateralmente simétrica en la
imagen cuando la cámara mira **desde una dirección contenida en el propio
plano de simetría** ("de canto"). Desde otros ángulos, el heurístico del
prompt (p. ej. `plane_v04_1`: "punto medio horizontal") deja de aproximar la
traza real. Esto explica el hallazgo ya reportado: *más vistas ayudan al eje
hasta n≈26 pero perjudican la precisión planar más allá de n=1* — cada vista
adicional es señal válida para el eje, pero para el plano la mayoría de las
vistas Fibonacci no están "de canto" (desconocido a priori), así que agregan
ruido sistemático.

Algoritmo (esquema tipo RANSAC secuencial):

1. Con ≥4 vistas independientes del pool disponible, generar candidatos de
   normal: tomar pares de líneas 3D (dos pares de vistas → dos líneas, igual
   método que el eje) y su producto cruzado.
2. Puntuar cada vista respecto al candidato por el ángulo entre la dirección
   de cámara (ya conocida) y la normal candidata — ángulo cercano a 90°
   ("de canto") = vista confiable; esto **no requiere la malla**.
3. Reajustar (SVD) usando solo las vistas con buen puntaje (`good_views`).
4. **Consolidación multi-plano**: identificar qué vistas quedaron mal
   explicadas por el plano 1 (sin descartar las que sí lo apoyaron — en
   objetos con planos ortogonales, una vista puede estar "de canto" a más de
   uno), repetir sobre ese pool para el plano 2 y 3 (dataset: 692 objetos con
   1 plano GT, 152 con 2, 6 con 3).
5. Criterio de parada: <4 vistas independientes disponibles, o el candidato
   duplica (ángulo bajo umbral, `dup_angle_thresh_deg`) un plano ya aceptado.

**Hiperparámetros hardcodeados en el baseline, candidatos a barrer** (§6 de
`docs/implementacion_pipeline_sin_malla.md`): `edge_on_thresh=0.5` (umbral
"de canto"), `dup_angle_thresh_deg=15°`, `max_planes` (producción corrió con
`3`).

### 3.5 Qué se reutiliza, qué es nuevo

| Componente | Estado |
|---|---|
| `ImagesGenerator/`, `MolmoPointing/` completos | Se reutilizan sin cambios — el pipeline sin malla lee los mismos `molmo_multiview_<EXP>.json` |
| `pipeline_common/camera.py`, `datasets.py`, `naming.py` | Se reutilizan (rayos de cámara, carga de malla para GT/validación, convención de nombres) |
| `pipeline_common/triangulation.py` | **Nuevo** — geometría pura: `ray_dir_for_point`, `view_forward_direction`, `interpretation_plane_normal`, `triangulate_line`, `widest_pair` |
| `Mapping/estimate_symmetry_no_mesh.py` | **Nuevo** — CLI espejo de `estimate_symmetry.py`, sin `--objects-root` obligatorio, escribe bajo la clave de método `"triangulation"`/`"triangulation_multiplane"` (no `"svd"`, para no implicar que hubo ray-casting) |
| `Mapping/estimate_symmetry.py`, `map_to_3d.py` | **Se mantienen intactos** como baseline "con malla" de comparación — nada se borra |
| `Mapping/evaluate.py` | Modificado: agrega `"triangulation"`/`"triangulation_multiplane"` a `METHODS`; rama separada para el resumen multi-plano (`recall_planes_mean`, `precision_planes_mean`, `n_planes_matched_mean`) |

### 3.6 Métricas (referencia, no repetidas acá)

Definición exacta en `docs/metricas_evaluacion.md`. Resumen mínimo necesario
para leer §5:

- **`angular_error_deg`**: `arccos(|dir_pred·dir_GT|)`, sign-agnostic,
  `[0°,90°]`. Para plano, contra el **mejor match** entre los 1–3 planos GT
  del objeto.
- **`AUC_angular`**: trunca en 45°, no en 90° — todo error ≥45° cuenta como
  fallo total.
- **`recall_planes_mean` / `precision_planes_mean`**: exclusivas del modo
  multi-plano (`triangulation_multiplane`) — fracción de planos GT
  encontrados / fracción de planos predichos que matchean un GT real
  (`evaluate_plane_multiset`, ver `docs/actualizacion_metricas.md` §5.6).
  **No están definidas para el modo un-solo-plano** (columna vacía en el CSV
  cuando `method="triangulation"`) — ver caveat en §5.4.
- **`SDE_ref` / `F1_ref`**: métricas de referencia externa
  (`EnhancedBackProjection`/WACV2026), no data leakage salvo `F1_ref` (sí usa
  GT, es una métrica de detección). No intercambiables con
  `angular_error`/`translation_error` — normalización y muestreo distintos.

---

## 4. Experimentos asociados (`Pipeline_Experiments/`)

Todos leen `molmo_multiview_<EXP>.json` ya generado (ningún experimento de
esta sección llama a Molmo2 de nuevo, salvo EXP-LIT-1) y escriben
`predicted_symmetry_<EXP>_<variante>_nomesh.json` en el mismo esquema que
`Mapping/evaluate.py` ya entiende — ninguno modifica el baseline sin malla,
corren en paralelo. Fuente completa: `Pipeline_Experiments/README.md`.

### 4.1 Ablaciones de triangulación (EXP-A a EXP-F)

| Variante | Mecanismo | Motivación / literatura | Aplica a |
|---|---|---|---|
| **EXP-A** | SVD ponderado por separación en píxeles de los dos puntos de cada vista (`pixel_length`) — reemplaza el `triangulate_line` sin pesos del baseline | Bartoli & Sturm, CVIU 2005 — un baseline más largo hace la normal del plano de interpretación menos sensible al ruido de localización de Molmo2 | Eje y plano |
| **EXP-B** | "Point-then-direction": desacopla el ancla (punto más cercano a todos los rayos individuales, por mínimos cuadrados) de la dirección (misma SVD ponderada de EXP-A) — el baseline resuelve ambos del mismo sistema | Wu et al. 2021 (dominio: triangulación de pose de mano) — instanciación propia, no hay código de referencia publicado para 3D-point/VLM pointing | Solo eje |
| **EXP-C** | Reemplaza la selección de par de puntos por RANSAC 2D sobre la vista (ajusta una línea 2D, descarta outliers, toma el par de inliers más separado) — mantiene el peso uniforme del baseline | Recker et al., WACV 2013 | Eje y plano — **con los prompts actuales (2 puntos/vista) es un no-op determinista** (confirmado en código y en los resultados, §5.2) |
| **EXP-D** | Mismo peso por separación en píxeles de EXP-A, pero penalizado si alguno de los dos puntos está cerca del borde de la imagen (FOV recortado = localización menos confiable) | AssemblyHands-X, arXiv:2509.23888 | Eje y plano |
| **EXP-E** | Ajusta sin pesos (igual al baseline), luego itera 3 veces recalculando pesos por qué tanto disiente cada vista de la dirección actual (`residual = |normal·direction|`) y re-triangula | Hess-Flores et al.; Zhang et al., arXiv:2008.01258 | Solo eje |
| **EXP-F** | Mismo algoritmo de consolidación secuencial del baseline, pero cada plano candidato se verifica contra la malla real (`SDE_ref` real vía `gpytoolbox`, muestra de superficie ponderada por área) antes de aceptarlo — rechaza si `SDE_ref > sde_gate` (default 0.02) **o es NaN** (fix del bug de malla degenerada, ver §5.5) | PRS-Net, Gao et al., IEEE TVCG 2021 — motivado por la sobre-predicción de planos del baseline (`n_planes_predicted≈1.9` vs GT`≈1.18` a n=14) | Solo plano — **única variante que usa la malla en tiempo de predicción**, rompiendo a propósito la premisa "sin malla" como experimento controlado |

Notas de diseño (`Pipeline_Experiments/README.md`): `expB`/`expE` son
exclusivos de eje y `expF` exclusivo de plano por cómo se prototiparon en el
sandbox (no hay contraparte planar documentada para "point-then-direction"/
reweighting iterativo; el gate de `expF` es intrínsecamente un concepto de
consolidación de plano). `--dup-angle-thresh` y `--sde-gate` (antes
constantes hardcodeadas en `estimate_symmetry_no_mesh.py`) quedaron expuestos
como flags reales de CLI, barribles.

### 4.2 Políticas de descarte de vistas (`view_filtering.py`)

Generaliza la única política que ya tenía el baseline (descartar puntos
individuales fuera del objeto, invalidar la vista si sobreviven muy pocos) en
4 políticas nombradas, prototipadas en
`Experiments/sandbox_pipeline_server_final.ipynb` §14:
`none` (no-op, comportamiento actual) / `per_point` / `any` / `both` — un
punto se clasifica dentro/fuera del objeto comparando el pixel renderizado en
esa coordenada contra el umbral de fondo blanco plano
(`BACKGROUND_THRESH=250`, PyTorch3D `HardFlatShader`). Motivado por
`docs/diagnostico_conditioning_axis.md` §7: los puntos fuera de la silueta no
explican `angular_error` por sí solos, pero correlacionan con outliers de
`translation_error` (lift 1.2–1.4×).

**Estado: código implementado, pero no incluido en `full_sweep.yaml`** — la
corrida real de `experiments_15_09_2026/` **no ejerció ninguna política
distinta de `none`** (confirmado: cero experiment_id con sufijo `filt-` en
los CSV de resultados). Ver §6.

### 4.3 EXP-LIT-1 — Candidates-then-Select (`candidates_then_select/`)

Única ablación de esta sección que hace **llamadas nuevas** a Molmo2 (necesita
GPU): lee los puntos de una corrida previa como "pase 1" (anclas), marca
K+1 candidatos numerados alrededor de cada punto sobre el render real, y le
pide a Molmo2 que elija uno ("pase 2") — a diferencia de
`Experiments/sandbox_literatura.ipynb`, que solo **simuló** esto con
transformaciones geométricas sobre predicciones ya existentes, sin inferencia
real. Literatura: CVPC (arXiv:2512.04686), ZeroDex (arXiv:2606.19340).

**Estado: implementado y probado a nivel de geometría/parsing, nunca corrido
contra el modelo real** — pendiente de GPU en servidor.

### 4.4 Diagnóstico de colapso a centro de imagen (`diagnose_center_bias.py`)

Puerto batch, config-driven, de
`Experiments/sandbox_pipeline_server_updated_6_pts_v2.ipynb`: mide qué tan
seguido Molmo2 coloca un punto en `X≈500` (centro horizontal de la imagen)
**independientemente del ángulo de cámara** — separa vistas cenitales
legítimas (elevación cerca de ±90°, donde `X≈500` es la respuesta correcta)
de todas las demás (donde indicaría el sesgo/atajo, no razonamiento
geométrico real). No necesita GPU ni nuevas llamadas a Molmo2 — relee JSON ya
generados.

**Estado: configurado en `full_sweep.yaml` (`diagnostics: [center_bias]`),
pero sin evidencia de haber corrido** — no se encontró ningún
`center_bias_detail.csv`/`center_bias_summary.csv` en el repo local ni en
`~/results/diagnostics/` del servidor (verificado en esta misma sesión, ver
§6).

---

## 5. Métricas obtenidas (`experiments_15_09_2026/`)

> **Corrección (2026-09-26) — leer antes de usar esta sección en la tesis.**
> Un análisis posterior (`docs/reporte_resultados_sin_malla.md`, resumen en
> `docs/resumen_ejecutivo_sin_malla.md`) encontró que: (1) los resultados de
> eje de esta sección son **indistinguibles de una dirección al azar**
> (57.3° de error medio analítico; los ejes GT están rotados al azar y las
> predicciones colapsan al eje vertical del render); (2) los de plano **no
> se han comparado todavía contra un predictor trivial** ("siempre el plano
> X"), que en la muestra local los supera, porque los planos GT están en
> pose canónica; (3) las métricas de plano **excluyen los objetos sin
> predicción**, así que las comparaciones entre filas con distinta cobertura
> están sesgadas. Las conclusiones de §5.2bis (EXP-B) y §5.4 (EXP-F) quedan
> corregidas abajo. Los números de esta sección siguen siendo correctos como
> lectura de los CSV; lo que cambia es su interpretación.

Todo lo que sigue se recalculó directamente de
`experiments_15_09_2026/{axis,plane}_sym_nomesh_comparison.csv` (342 filas de
eje, 282 de plano) en esta sesión — no son cifras citadas de memoria.

### 5.1 Baseline de eje — ranking por prompt (mejor `n_views`, sin ninguna ablación)

`angular_error_mean` en grados, sobre las 850 objetos curados de
`axis_sym` (`n_objects` = con predicción válida):

| Prompt | mejor n_views | angular_error_mean | AUC | precision@10° | n_objects |
|---|---|---|---|---|---|
| `axis_v05_1` | 26 | **57.36°** | 0.104 | 0.013 | 831 |
| `axis_v06` | 26 | 57.37° | 0.103 | 0.018 | 821 |
| `axis_v05` | 26 | 57.67° | 0.102 | 0.014 | 830 |
| `axis_v04` | 6 | 57.68° | 0.100 | 0.017 | 831 |
| `axis_v00_1` | 26 | 57.78° | 0.100 | 0.013 | 828 |
| `axis_v04_1` | 26 | 57.84° | 0.099 | 0.018 | 829 |
| `axis_v00` | 6 | 59.09° | 0.093 | 0.015 | 793 |
| `axis_v03` | 6 | 59.35° | 0.105 | 0.019 | 805 |
| `axis_v03_1` | 6 | 59.74° | 0.102 | 0.022 | 814 |
| `axis_v01` | 6 | 59.89° | 0.102 | 0.019 | 818 |
| `axis_v01_1` | 6 | 60.20° | 0.093 | 0.015 | 827 |
| `axis_v02` | 6 | 60.53° | 0.098 | 0.018 | 816 |
| `axis_v02_1` | 14 | 63.62° | 0.087 | 0.013 | 675 |
| `axis_v07` | 26 | 70.10° | 0.065 | 0.008 | 500 |

**Lectura**: los 6 mejores prompts (57.3°–57.9°) son casi indistinguibles
entre sí — la mejora de la ronda "centerline"/"polar" (v04, v05, v06 y sus
`_1`) sobre "par bilateral"/"silueta ancha" (v01, v02) es real pero modesta
(~2–3° de error angular, no un salto cualitativo). `axis_v07` (corte más
angosto, inversión deliberada de v02) es notoriamente el peor — tanto en
error (70°, ~13° peor que el resto) como en cobertura (500/850 válidos, muy
por debajo de los ~800+ del resto), consistente con lo ya visto en el audit
de faltantes. `axis_v08`, `_6pts`, `lit2_grid`, `lit3_cot` y Flujo B/C no
tienen fila de baseline "pura" en este dataset — solo se evaluaron con
ablaciones EXP-A..E encima (§5.2), no como corrida independiente de
`estimate_symmetry_no_mesh.py`.

### 5.2 Efecto de las ablaciones EXP-A..E sobre el eje

Comparación pareada (misma combinación prompt × n_views, ablación vs.
baseline sin ablación), promediada sobre las 42 combinaciones baseline
disponibles:

| Variante | Δ `angular_error_mean` promedio | desviación | rango |
|---|---|---|---|
| **EXP-A** (peso por separación px) | **+0.615°** (peor) | ±0.45° | [−0.16°, +1.53°] |
| **EXP-B** (ancla/dirección desacoplados) | **+0.615°** (peor) | ±0.45° | [−0.16°, +1.53°] |
| **EXP-C** (RANSAC 2D) | **0.000°** (idéntico) | 0 | — |
| **EXP-D** (peso penalizado por borde) | +0.614° (peor) | ±0.45° | [−0.16°, +1.53°] |
| **EXP-E** (reweighting iterativo) | +0.264° (peor, menos) | ±0.23° | [−0.13°, +0.72°] |

**Ninguna ablación mejora el baseline en promedio.** Tres hallazgos no
triviales, verificados en el código antes de reportarlos:

1. **EXP-C da exactamente 0.000° de diferencia en las 42 combinaciones** —
   no es una coincidencia numérica, es el comportamiento **documentado y
   esperado**: con exactamente 2 puntos por vista (todos los prompts de
   Flujo A actuales), `ransac_line_2d` es un *pass-through* determinista
   idéntico a `widest_pair` (`_shared.py`, docstring de `ransac_line_2d`).
   Recién divergería con los prompts `_6pts` (≥3 puntos candidatos por
   vista).
2. **EXP-A y EXP-B dan exactamente el mismo delta en las 42 combinaciones** —
   tampoco es casualidad: `angular_error_deg` depende únicamente de
   `direction`, y ambas variantes calculan la dirección con la **misma**
   fórmula (`weighted_null_direction` sobre las mismas normales, mismo peso
   `pixel_length`) — solo difieren en cómo calculan `origin` (que afecta
   `translation_error`, no el error angular). No es un bug, es una
   consecuencia matemática directa de que ambas comparten el mismo paso de
   ajuste de dirección.
3. **Toda variante que pondera por separación en píxeles (A, B, D) empeora
   ~0.6°; la que repondera iterativamente por residuo (E) empeora menos
   (~0.26°) pero sigue sin mejorar.** Esto es **consistente** (no
   contradictorio) con la conclusión de `docs/diagnostico_conditioning_axis.md`
   §4: el error angular alto y uniforme (~58–70°) no está explicado por
   ningún mecanismo de *varianza*/ruido de conditioning (ese diagnóstico ya
   había descartado `pixel_sep_mean`, `cond_number` y `axis_span` como
   predictores, `|r|<0.35` en los tres). Si el cuello de botella fuera
   varianza por mala separación de puntos, ponderar por esa separación
   debería ayudar — en cambio, ponderar por ella la empeora levemente. Esto
   refuerza (no reemplaza) la hipótesis de **sesgo por inconsistencia de
   identidad entre vistas** como la explicación pendiente de confirmar
   directamente (§6).

**Desglose prompt-por-prompt (no solo el promedio)**: se repitió la
comparación anterior por cada una de las 42 combinaciones `prompt × n_views`
individualmente (14 prompts × 3 `n_views`), en vez de solo el promedio
agregado, para responder directamente "¿mejoró alguna ablación en algún
prompt concreto?" — usando un umbral de 0.5° (por debajo de eso, ruido):

| Variante | combinaciones con mejora real (Δ<−0.5°) | combinaciones claramente peor (Δ>+0.5°) |
|---|---|---|
| EXP-A | **0 / 42** | 21 / 42 |
| EXP-B | **0 / 42** | 21 / 42 |
| EXP-C | 0 / 42 (siempre idéntico, ver punto 1) | 0 / 42 |
| EXP-D | **0 / 42** | 21 / 42 |
| EXP-E | **0 / 42** | 8 / 42 |

**Cero de las 210 combinaciones (prompt × n_views × variante) muestra una
mejora real sobre el baseline** — ni siquiera para el prompt individual con
mejor rendimiento (`axis_v05_1`/`axis_v06`) ni para ningún otro. Las
"mejoras" que sí aparecen en la tabla cruda son ruido de menos de 0.16°
(ej. `axis_v05_1` a n=6: EXP-A da 58.52° vs 58.64° del baseline, una
diferencia de 0.12° — muy por debajo de cualquier umbral con sentido
estadístico dado el `std` de ~22° del error angular por objeto). En cambio,
la mitad de las combinaciones (21/42) con EXP-A/B/D son claramente **peores**
que el baseline. **Conclusión (en `angular_error_mean`): en esta corrida,
ninguna de las 5 ablaciones mejoró el baseline para ningún prompt de eje,
individualmente considerado.** Esta conclusión es específica del error
angular — el punto siguiente muestra que en **traslación** el panorama es
muy distinto.

### 5.2bis Efecto de las ablaciones sobre el eje — traslación (`translation_error_normalized_mean`)

`angular_error_deg` solo mide la **dirección** del eje; el error de
traslación (`Mapping/evaluate.py::point_to_line_distance`, distancia
punto-a-línea entre el origen predicho y el eje GT, normalizada por la
diagonal del bounding box) mide qué tan bien ubicado está el eje en el
espacio — son ejes de error independientes (un eje puede tener la dirección
correcta pero estar desplazado, o viceversa). Repetir la comparación
pareada de §5.2 con esta métrica cambia la conclusión para **EXP-B**:

| Variante | Δ media (normalizada) | mejoras (Δ<−0.01) | empeoramientos (Δ>+0.01) |
|---|---|---|---|
| EXP-A | +0.0044 (peor) | 0/42 | 5/42 |
| **EXP-B** | **−0.2200 (mucho mejor)** | **32/42** | 2/42 |
| EXP-C | 0.0000 (idéntico) | 0/42 | 0/42 |
| EXP-D | +0.0052 (peor) | 0/42 | 7/42 |
| EXP-E | +0.0014 | 0/42 | 0/42 |

**EXP-B reduce el error de traslación entre 50% y 96% en 26/42 combinaciones**
(mediana de reducción: 62%), sin cambiar el error angular (ver §5.2 punto 2:
`direction` es idéntica a EXP-A/D por construcción). Ejemplos:

| Prompt | n_views | trans. norm. base → EXP-B | reducción | error angular base → EXP-B |
|---|---|---|---|---|
| `axis_v05_1` | 6 | 1.305 → 0.047 | **−96.4%** | 58.64° → 58.52° (≈ igual) |
| `axis_v00_1` | 6 | 0.969 → 0.051 | **−94.8%** | 59.59° → 59.76° (≈ igual) |
| `axis_v05` | 26 | 0.419 → 0.053 | **−87.3%** | 57.67° → 58.05° (≈ igual) |

**Mecanismo**: EXP-B (§4.1) calcula el origen como el punto de mínimos
cuadrados más cercano a *todos* los rayos individuales
(`closest_point_to_rays`), en vez de resolverlo junto con la dirección en el
mismo sistema ponderado que usa el baseline (`weighted_triangulate_line`) —
un ancla mejor determinada, independiente de si la dirección ajustada es
correcta o no.

**El efecto depende del tipo de prompt**: es fuerte y consistente en los
prompts `independent` (puntos directamente sobre el eje: `v00`, `v04`,
`v05`, `v06`, `v07` y sus `_1`, 52%–96% de reducción) pero débil o incluso
**negativo** en los prompts `midpoint`/pares bilaterales (`v01`, `v03` y
`_1`, 18%–28% a `n_views=6`, pero −1% a −21% a `n_views=14`). Consistente
con el mecanismo: en un prompt bilateral los dos puntos de una vista no
están sobre el eje (están a los lados), así que "el punto más cercano a
todos los rayos individuales" es un ancla peor definida ahí que cuando cada
punto observado sí está sobre el eje.

**Conclusión combinada de §5.2/§5.2bis**: de las 5 ablaciones de eje, EXP-B
es la única con una mejora real y sustancial — pero en traslación, no en
ángulo, y concentrada en los prompts de tipo `independent`. Si la tesis
reporta solo `angular_error`/`AUC`, esta mejora queda invisible; vale la
pena reportar `translation_error_normalized_mean` como métrica secundaria
para los prompts `independent` de eje.

> **Corregido (2026-09-26)**: la mejora de EXP-B es casi toda sobre la
> **media** (−0.22); la **mediana** apenas cambia (Δ típico −0.003; 22
> combinaciones mejoran y 20 empeoran). `translation_error` mide la distancia
> del ancla a la recta GT, y el GT pasa por el origen, que es el punto al que
> miran todas las cámaras: el ancla de EXP-B (punto más cercano a todos los
> rayos) cae ahí sin importar si la dirección es correcta. EXP-B elimina
> outliers del ancla del baseline; **no es evidencia de que localice mejor el
> eje**. Ver `docs/reporte_resultados_sin_malla.md` §2.4 y §3.6.

### 5.3 Baseline de plano — ranking por prompt (multi-plano, `max_planes=3`)

`recall_planes_mean` (fracción de planos GT encontrados) en el mejor
`n_views` por prompt:

| Prompt | mejor n_views | n_planes_predicted | n_true_planes | recall | precision |
|---|---|---|---|---|---|
| `plane_v07` | 26 | 2.49 | 1.19 | **0.858** | 0.443 |
| `plane_v04_1` | 14 | 1.89 | 1.17 | 0.854 | 0.518 |
| `plane_v04` | 14 | 1.94 | 1.17 | 0.845 | 0.488 |
| `plane_v06` | 14 | 1.93 | 1.18 | 0.830 | 0.487 |
| `plane_v02` | 14 | 1.93 | 1.18 | 0.821 | 0.476 |
| `plane_v00_1` | 14 | 1.94 | 1.20 | 0.816 | 0.477 |
| `plane_v02_1` | 14 | 1.95 | 1.19 | 0.808 | 0.467 |
| `plane_v05_1` | 14 | 1.97 | 1.18 | 0.679 | 0.380 |
| `plane_v00` | 14 | 1.97 | 1.19 | 0.687 | 0.351 |
| `plane_v05`, `v03`, `v03_1`, `v01`, `v01_1` | 6 | ≈1.0 | ≈1.19 | 0.10–0.20 | 0.08–0.20 |

**Lectura**: sanity check — `n_true_planes_mean` da ≈1.17–1.20 en todas las
filas, consistente con la composición real del dataset curado (692 obj×1 +
152×2 + 6×3 planos = 1014/850 ≈ 1.193 planos/objeto promedio) — confirma que
la evaluación multi-plano está leyendo el ground truth correcto. Los prompts
de **costura/silueta** (`v00`, `v02`, `v04`, `v06`, `v07` y sus `_1`) llegan
a recall 0.68–0.86; los de **par bilateral** (`v01`, `v03`, `v05` y `_1`)
colapsan a recall 0.10–0.20 en el modo multi-plano — un patrón mucho más
marcado que en el modo de un-solo-plano (donde `v01`/`v03` no estaban tan
lejos en `angular_error`), porque `role_pair_selector` (roles fijos
izquierda/derecha) tiene mucha más dificultad para encontrar ≥4 vistas
independientes con ambos roles presentes que el heurístico de costura/centro.

### 5.4 Efecto de las ablaciones sobre plano — con una advertencia metodológica importante

**EXP-A/C/D no son comparables contra el baseline en esta corrida.**
`Pipeline_Experiments/configs/full_sweep.yaml` solo fija `max_planes: 3`
para `expF` — `expA`/`expC`/`expD` corrieron con el default
(`max_planes=1`, modo un-solo-plano, `method="triangulation"`), mientras que
**el baseline de plano en este dataset es siempre multi-plano**
(`method="triangulation_multiplane"`, verificado: 0 filas baseline en modo
single-plane). `Mapping/evaluate.py` no comparte ninguna columna de métrica
entre ambos modos — ni siquiera `angular_error_mean` (está vacía para las
filas multiplano). **No se puede construir una tabla de comparación honesta
con los datos de esta corrida** para A/C/D en plano; para tenerla, hay que
re-correr esas tres variantes con `--max-planes 3` — config ya armado en
`Pipeline_Experiments/configs/plane_maxplanes3_rerun.yaml` (§6, ítem 3).

**`f1_ref` — la métrica de mayor interés para plano — todavía no existe para
ninguna ablación.** Verificado directamente sobre el CSV: la columna
`f1_ref` está poblada en las **42/42 filas baseline** y **vacía en las
240/240 filas de ablaciones** (A/B/C/D/E/F), sin excepción — `full_sweep.yaml`
corrió con `with_reference_metrics: false` por costo (`gpytoolbox` sobre 190
combinaciones × 850 objetos). El mismo config
`plane_maxplanes3_rerun.yaml` de arriba también activa
`with_reference_metrics: true`, así que la misma corrida resuelve ambos
gaps a la vez: deja a expA/C/D comparables en modo multiplano **y** calcula
`SDE_ref`/`F1_ref` para expA/C/D/F (expF ya estaba en modo multiplano, solo
le faltaba el re-score con la flag encendida). Hasta que esa corrida no se
haga, cualquier número de `f1_ref` para una ablación es una laguna de datos,
no un resultado — no reportar "F1_ref de EXP-F" en la tesis todavía.

**EXP-F sí es comparable** (ambos en modo multiplano, `max_planes=3`) — 42
combinaciones prompt×n_views con datos válidos en ambos lados:

| Métrica | Δ promedio (expF − baseline) |
|---|---|
| `recall_planes_mean` | **+0.054** |
| `precision_planes_mean` | **+0.118** |
| `n_planes_predicted_mean` | **−0.283** (predice menos planos por objeto) |

**Lectura**: el gate de `expF` cumple exactamente lo que buscaba (§4.1,
motivado por la sobre-predicción del baseline) — predice ~0.28 planos menos
por objeto en promedio, y eso se traduce en una ganancia de precisión
sustancial (+0.118, hasta +0.2 en varios prompts a n_views=14, ej.
`plane_v04`: 0.487→0.701) sin sacrificar recall en general (+0.054
promedio). El patrón por `n_views` es consistente entre prompts: a
`n_views=6` el gate mejora tanto recall como precisión (menos vistas → menos
candidatos espurios para empezar); a `n_views=26` el recall a veces **baja**
levemente (ej. `plane_v04`: 0.819→0.754, `plane_v07`: 0.858→0.807) — el gate
se vuelve, a veces, conservador de más con muchas vistas disponibles y
rechaza algún plano real, no solo los espurios. Es la ablación con evidencia
empírica más sólida de las 6 — vale la pena priorizarla para la escritura de
resultados de plano.

> **Corregido (2026-09-26)**: esta lectura no consideraba la cobertura. EXP-F
> predice sobre **281 objetos menos en promedio** que el baseline (−75 a
> −531), y las métricas de plano solo promedian sobre objetos con
> predicción. Contando los objetos abstenidos como recall 0, el **recall
> ajustado de EXP-F baja 0.078** (0 combinaciones mejoran, 23 empeoran) y el
> F1 ajustado sube apenas +0.011. La mejora de `SDE_ref` es circular (el gate
> filtra por `SDE_ref`) y EXP-F usa la malla para predecir. La corrida
> `experiments_22_09_2026` (`--max-planes 3` + `f1_ref`) confirma además que
> EXP-A/C/D no tienen efecto en plano. Ver
> `docs/reporte_resultados_sin_malla.md` §4.5.

**Desglose prompt-por-prompt de EXP-F** (14 prompts × 3 `n_views` = 42 filas,
umbral 0.03 para contar como cambio real en vez de ruido):

| Métrica | mejoras (Δ>+0.03) | empeoramientos (Δ<−0.03) |
|---|---|---|
| `precision_planes_mean` | **34 / 42** | **0 / 42** |
| `recall_planes_mean` | 16 / 42 | 3 / 42 |

**Precisión mejora en el 81% de las combinaciones y nunca empeora de forma
apreciable** — es la señal más limpia de las 6 ablaciones. Los únicos 3
casos donde el recall empeora de forma apreciable son, exactamente, los tres
prompts con **mejor baseline** a `n_views=26`: `plane_v04` (0.819→0.754),
`plane_v04_1` (0.830→0.775), `plane_v07` (0.858→0.807) — en los prompts
donde el baseline ya predice bien 2 planos con muchas vistas, el gate a
veces rechaza el segundo plano real además de los espurios. En los prompts
con peor baseline (`v01`, `v03`, `v05` y variantes bilaterales, recall
0.01–0.15) el gate **nunca empeora** y mejora recall de forma sustancial en
varios (ej. `plane_v05` a n=14: 0.098→0.206, +0.108). **Conclusión: EXP-F es
la única de las 6 ablaciones con mejora real y consistente, prompt por
prompt — más fuerte en precisión que en recall, y con el único costo de
recall concentrado en los mejores prompts a `n_views` alto.**

### 5.5 Limitaciones/bugs ya identificados en esta corrida (contexto necesario para no re-descubrirlos)

- **Bug real, ya arreglado** (2026-09-14): el gate de `expF` comparaba
  `if real_sde > sde_gate: break`, y en Python `nan > x` es `False` — un
  plano con `SDE_ref=NaN` (malla con triángulo de área cero, artefacto
  conocido de algunos objetos ShapeNet) pasaba el gate en vez de ser
  rechazado, exactamente lo opuesto del diseño. Fix:
  `if not np.isfinite(real_sde) or real_sde > sde_gate: break`. Los números
  de §5.4 son de la corrida **posterior** al fix.
- **`axis_v06_nomesh_filtered` / `plane_v04_1_nomesh_filtered` son
  duplicados** de `axis_v06`/`plane_v04_1` (mismos puntos de Molmo2, mismos
  números hasta el último dígito en `audit_results.py`) — confirmado en esta
  sesión, pendiente de sumar a la regla de dedup de `discover_experiment_ids`.
- **Faltantes de predicción son 100% escasez de datos, no bugs de código** —
  verificado directamente para `plane_v04_1_flowC`+`expF` (604/850 objetos,
  2415 intentos re-corridos con captura de excepción real: 0 `UNEXPECTED`,
  2415 `insufficient_data`) con el nuevo
  `Pipeline_Experiments/diagnostics/diagnose_missing_reason.py`. El faltante
  es mayor y uniforme entre `n_views` para prompts Flujo C (sin tracking de
  identidad entre vistas) y para `expF` (requisitos extra: ≥4 vistas +
  geometría de malla válida).

### 5.6 Por qué el sandbox (30 objetos) parecía mostrar mejoras que la corrida completa (850) no confirma

Antes de escribir "ninguna ablación mejora el baseline" en una tesis conviene
poder explicar por qué el prototipo en `Experiments/*.ipynb` había dado la
impresión contraria — si no, el resultado de §5.2/§5.4 parece contradecir el
trabajo previo en vez de refinarlo. Se verificaron los outputs realmente
guardados en los notebooks (no la memoria de haberlos corrido) para
responder esto con evidencia, no con especulación.

**Aclaración previa**: `Experiments/sandbox_pipeline_server_extended.ipynb`
define el código de EXP-A..F pero **nunca se ejecutó** — todas sus celdas de
comparación (`11a`, `12a`) tienen `outputs` vacío en el `.ipynb`. Los
resultados reales de las ablaciones sobre el subconjunto curado viven en
`Experiments/sandbox_pipeline_server_final.ipynb`, sobre **30 objetos**
(dice literalmente `=== Comparación EJE (30 objetos) ===` en el output
guardado).

**1. La propia tabla agregada del notebook (30 objetos) ya mostraba al
baseline ganando la mayoría de las columnas** — no es un resultado nuevo de
la corrida de 850, es el mismo patrón a menor escala:

```
                        AUC@45_nv6  P@10_nv6  med_err_nv6   AUC@45_nv26  P@10_nv26  med_err_nv26
Baseline (original)        0.2195    0.1724         60.9       0.2882    0.2759        57.7
EXP-A: Weighted             0.2174    0.1724         62.3       0.1985    0.1724        61.0
EXP-C: RANSAC2D              0.2074    0.1538         67.1       0.1913    0.1818        76.5
```

A `n_views=26` el baseline gana con claridad (AUC 0.288 vs. 0.198–0.199 de
todas las ablaciones; error mediano 57.7° vs. 61–76°). El "cambio" que
puede haber quedado en la memoria no viene de esta tabla — viene de la
siguiente.

**2. La tabla per-objeto del notebook es un artefacto de selección
("mejor-de-6"), no evidencia de que una ablación específica mejore**: para
cada uno de los 30 objetos, el notebook toma el **mínimo** error entre
Baseline y las 5 ablaciones, y cuenta cuál ganó:

```
Frecuencia mejor experimento (eje):
EXP-C       13
Baseline    11
EXP-E        5
EXP-A        1
```

Esto da la impresión de "EXP-C mejora en la mayoría de los objetos" — pero
si tenés 6 estimadores ruidosos (Baseline + 5 ablaciones) y para cada
objeto te quedás con el que dio menor error, alguno de los 5 no-baseline va
a "ganar" la mayoría de las veces **aunque ninguno sea sistemáticamente
mejor** — es la aritmética de "el mínimo de N variables ruidosas casi
siempre es menor que 1 sola sin elegir" (sesgo de comparaciones múltiples,
"winner's curse"). La propia tabla agregada del punto 1 ya contradice la
lectura ingenua: el mismo EXP-C que "gana" más seguido por objeto es, en
promedio, **peor** que el baseline a `n_views=26` (76.5° vs. 57.7°). Sobre
850 objetos (§5.2) ese sesgo desaparece porque ya no se compara "el mínimo
de 6" sino cada variante contra el baseline directamente, por eso ahí las
"mejoras" bajan a 0/210.

**3. El `EXP-C` del sandbox no es el mismo `EXP-C` que corrió en
producción** — esto sí es una diferencia real de mecanismo, no solo de
escala. La celda del notebook se llama *"EXP-C — Filtro objeto + RANSAC
2D"*: combina dos cambios en una sola ablación (filtrar puntos fuera del
objeto **y** aplicar RANSAC 2D). En `Pipeline_Experiments/`, esos dos
mecanismos se separaron a propósito en dos ablaciones independientes: el
filtro pasó a `view_filtering.py` (§4.2) y el RANSAC puro quedó en
`exp_c_ransac2d.py` — que, con los prompts de 2 puntos actuales, es un
*no-op* determinista comprobado en código (§5.2, punto 1). El cambio real
que el notebook veía en su `EXP-C` casi con certeza venía del **filtro de
objeto**, no del RANSAC — y esa pieza (`view_filtering.py`) es exactamente
la que quedó **sin correr a escala completa** (§4.2, §6 ítem 2). No es que
el hallazgo del notebook fuera falso — es que se está evaluando un mecanismo
distinto del que corrió en `experiments_15_09_2026/`.

**4. La única ablación que sí coincide entre notebook y corrida completa es
EXP-F** — en el notebook, F1/recall/precision de plano mejoran claramente
sobre el baseline en casi todas las filas (ej. F1 a n=6: 0.221 vs. 0.139 del
baseline); en la corrida de 850 objetos, la mejora de precisión es aún más
consistente (34/42 combinaciones, §5.4). Es la ablación cuya evidencia se
sostiene al pasar de 30 a 850 objetos — refuerza la recomendación de §5.4 de
priorizarla.

**Conclusión de esta sección**: la aparente discrepancia no es una
contradicción entre el sandbox y la corrida completa — es la combinación de
(a) una muestra 28× más chica con más ruido, (b) una forma de reportar
resultados (mejor-de-6 por objeto) que infla artificialmente la sensación de
mejora, y (c) al menos una ablación (`EXP-C`) que en el sandbox mezclaba dos
mecanismos que en producción se separaron, uno de los cuales todavía no se
evaluó a escala completa.

---

## 6. Trabajo pendiente (relevante para no repetir análisis ya hechos)

1. **Center-bias** (§4.4): configurado, sin evidencia de haber corrido —
   correr `Pipeline_Experiments/diagnostics/diagnose_center_bias.py --all`
   (o vía `run_batch.py` con `--skip-variants --skip-evaluate --skip-compare`)
   y revisar si `axis_v08`/`plane_v08` (diseñados específicamente contra este
   sesgo) lo reducen frente a `axis_v06`/`plane_v04_1`.
2. **View-filtering** (§4.2): código listo, nunca corrido a escala completa —
   agregar entradas `filter_policy: per_point/any/both` a un config y
   correr `run_batch.py`.
3. **Re-correr EXP-A/C/D de plano con `--max-planes 3` + activar
   `--with-reference-metrics`** — resuelve a la vez el gap de comparación
   justa (§5.4, primer párrafo) y el de `f1_ref`/`SDE_ref` sin poblar para
   ninguna ablación (§5.4, segundo párrafo; hoy 42/42 filas baseline lo
   tienen, 0/240 filas de ablación). Config ya armado:
   ```bash
   conda activate tesis_env
   # 1. smoke test (20 objetos) para medir tiempo real antes de comprometer
   #    la corrida completa — la parte cara es gpytoolbox (SDE_ref/F1_ref),
   #    no la consolidación de planos en sí:
   time python Pipeline_Experiments/run_batch.py \
       --config Pipeline_Experiments/configs/plane_maxplanes3_rerun.smoke.yaml
   # 2. full run (850 objetos), una vez validado el tiempo del smoke test:
   python Pipeline_Experiments/run_batch.py \
       --config Pipeline_Experiments/configs/plane_maxplanes3_rerun.yaml
   ```
   No toca `axis_sym` ni pisa los `predicted_symmetry_*_expA/C/D_nomesh.json`
   de un solo plano ya existentes — escribe IDs nuevos (`..._expA_mp3_nomesh`,
   etc.) al lado. `expF` ya estaba en `max_planes=3`, así que sus
   predicciones se saltan (`overwrite: false`) y solo se re-corre su
   `evaluate.py` con la flag nueva. El nuevo
   `plane_sym_nomesh_comparison.csv` cae en
   `../results/experiments_<fecha>/`.

   **Estimado de tiempo**: no hay una medición empírica de
   `--with-reference-metrics` para este subconjunto exacto (4 variantes × 3
   `n_views` × 850 objetos ≈ 10 200 evaluaciones objeto×n_views, ~5% del
   universo de 190 combinaciones que `full_sweep.yaml` evitó por costo). El
   único dato real de esta sesión es la generación de predicciones de
   `expF` (no la evaluación con `gpytoolbox`) sobre 604 objetos en 2m19s
   (~0.23s/objeto) — el paso de `evaluate.py --with-reference-metrics` hace
   un trabajo de orden similar por objeto (muestreo de superficie + consulta
   a árbol AABB), así que una estimación razonable (no medida) es **entre
   15 y 90 minutos** para las 4 variantes completas, corriendo secuencial en
   CPU. **No comprometer la corrida completa sin antes correr el smoke test
   de arriba y multiplicar su tiempo real por ~42** (850/20) — es la única
   forma de tener un número confiable en vez de una conjetura.
4. **EXP-LIT-1 real** (§4.3): implementado, nunca corrido contra el modelo
   (necesita GPU).
5. **Diagnóstico de consistencia de identidad** (residuo perpendicular al eje
   por punto, buscando distribución bimodal) — próximo paso priorizado por
   `docs/diagnostico_conditioning_axis.md` §5.2, todavía no implementado; es
   el que decidiría si vale la pena el experimento híbrido `hybrid_v08`
   (verificación cross-view) antes que cualquier otra variante geométrica.

---

## 7. Diagnósticos complementarios (fuera del alcance de este documento, referenciados)

`docs/diagnostico_conditioning_axis.md` corrió 3 diagnósticos de costo cero
(sin llamar a Molmo2) sobre por qué el error angular de eje es alto y
uniforme (~58–70°, §5.1): conditioning de las normales de triangulación
(§2–5), ángulo cámara-eje vs. GT (§6), y fracción de puntos que caen fuera
del objeto (§7). **Los tres descartan una explicación geométrica** — ninguno
predice el error angular (`|r|<0.35` en los tres, y el de ángulo cámara-eje
sale con el signo invertido a lo esperado). Deja la inconsistencia de
identidad entre vistas como la única hipótesis no descartada — consistente
con el hallazgo de §5.2 de este documento (ponderar por conditioning
tampoco ayuda). Los puntos fuera del objeto sí correlacionan parcialmente con
outliers de `translation_error` (lift 1.2–1.4×), no con `angular_error`.

---

## 8. Referencias bibliográficas

Consolidado de las citas que ya aparecen, con su fuente exacta, en los
documentos y notebooks del repo — ninguna se agrega de memoria en este
documento.

### 8.1 Triangulación de líneas 3D (base del eje sin malla, §3.3)

Fuente: `docs/pipeline_sin_malla.md` §8.1.

- Bartoli, A., Sturm, P. (2005). *Structure-from-motion using lines:
  Representation, triangulation, and bundle adjustment*. CVIU, 100(3),
  416–441. — También citado en `Pipeline_Experiments/README.md` como base de
  **EXP-A** (§4.1).
- Wu, F., Zhang, M., Wang, G., Hu, Z. (2015). *Algebraic Error Based
  Triangulation and Metric of Lines*. PLOS ONE, 10(7).
- Hartley, R. I. (1997). *Lines and Points in Three Views and the Trifocal
  Tensor*. IJCV, 22(2), 125–140.
- Hartley, R., Zisserman, A. (2004). *Multiple View Geometry in Computer
  Vision* (2ª ed.). Cambridge University Press. — Cap. 15 (degeneración);
  también citado en `docs/diagnostico_conditioning_axis.md` §2.1/2.2 (base
  del argumento de conditioning/SVD).
- Josephson, K., Kahl, F. *Triangulation of Points, Lines and Conics*. SCIA
  2007 / JMIV. *(verificar venue exacto antes de citar formalmente)*.
- Hofer, M., Maurer, M., Bischof, H. (2016). *Line3D++: Efficient 3D Scene
  Abstraction Using Line Segments*. CVIU, 157, 167–178.
- Liu, S., Yu, Y., Pautrat, R., Pollefeys, M., Larsson, V. (2023). *3D Line
  Mapping Revisited* (LIMAP). CVPR 2023.
- Mateus, A., Tahri, O., Aguiar, A. P., Lima, P. U., Miraldo, P. (2021). *On
  Incremental Structure-from-Motion using Lines*. arXiv:2105.11196 / IEEE
  T-RO.

### 8.2 Reconstrucción de planos desde trazas multivista (base del plano sin malla, §3.4)

Fuente: `docs/pipeline_sin_malla.md` §8.2.

- François, A. R. J., Medioni, G. G., Waupotitsch, R. (2003). *Mirror
  symmetry ⇒ 2-view stereo geometry*. Image and Vision Computing, 21(2),
  137–143. — Base del camino alternativo no implementado (ver
  `docs/pipeline_sin_malla.md` §3.2).
- Criminisi, A., Reid, I., Zisserman, A. (2000). *Single View Metrology*.
  IJCV, 40(2), 123–148.
- Olsson, C., Eriksson, A. (2011). *Triangulating a Plane*. SCIA 2011, LNCS
  6688, 13–23.
- Gao, Y., Yuille, A. L. (2017). *Exploiting Symmetry and/or Manhattan
  Properties for 3D Object Structure Estimation from Single and Multiple
  Images*. CVPR 2017 (ext. IJCV 2019).
- Hong, W., Yang, A. Y., Huang, K., Ma, Y. (2004). *On Symmetry and
  Multiple-View Geometry: Structure, Pose, and Calibration from a Single
  Image*. IJCV, 60(3), 241–265.
- Sinha, S. N., Ramnath, K., Szeliski, R. (2012). *Detecting and
  Reconstructing 3D Mirror Symmetric Objects*. ECCV 2012.
- Wang, R., Geraghty, D., Matzen, K., Szeliski, R., Frahm, J.-M. (2020).
  *VPLNet: Deep Single View Normal Estimation with Vanishing Points and
  Lines*. CVPR 2020.
- Li, X. et al. (2025). *Symmetry Strikes Back* (Reflect3D). CVPR 2025
  (arXiv:2411.17763).

### 8.3 Robustez con detecciones semánticas ruidosas, pocas vistas (base de EXP-B/C/E, §4.1)

Fuente: `docs/pipeline_sin_malla.md` §8.3, `Pipeline_Experiments/README.md`.

- Iskakov, K., Burkov, E., Lempitsky, V., Malkov, Y. (2019). *Learnable
  Triangulation of Human Pose*. ICCV 2019. — Precedente directo del esquema
  de pesos de confianza por vista (EXP-A/D/E).
- Recker, S., Hess-Flores, M., Joy, K. I. (2013). *Statistical Angular
  Error-Based Triangulation for Efficient and Accurate Multi-View Scene
  Reconstruction*. WACV 2013. — Base de **EXP-C** (§4.1).
- Hess-Flores, M. et al.; Zhang et al. (2020). arXiv:2008.01258. — Base de
  **EXP-E** (reweighting iterativo, §4.1).
- Lee, S. H., Civera, J. (2019). *Closed-Form Optimal Two-View Triangulation
  Based on Angular Errors*. ICCV 2019 (arXiv:1903.09115).
- Ghasemi, A. et al. *Improving Triangulation by Enforcing Consistency*
  (arXiv:1804.10448).
- Wu, F. et al. (2021) — dominio de triangulación de pose de mano, base de
  **EXP-B** (`Pipeline_Experiments/README.md`; instanciación propia, no
  puerto verbatim).
- AssemblyHands-X (arXiv:2509.23888) — base de **EXP-D** (peso penalizado
  por cercanía al borde, `Pipeline_Experiments/README.md`).

### 8.4 VLM-pointing + triangulación multivista sin malla densa

Fuente: `docs/pipeline_sin_malla.md` §8.4, `Pipeline_Experiments/README.md`.

- ZeroDex (2026, arXiv:2606.19340). — Precedente más directo: keypoints 2D de
  un VLM elevados a 3D por fusión multivista, sin malla densa. También base
  de **EXP-LIT-1** (§4.3).
- Gong, B. et al. (2025). *ZeroKey: Point-Level Reasoning and Zero-Shot 3D
  Keypoint Detection from Large Language Models*. ICCV 2025
  (arXiv:2412.06292). — Precedente metodológico ya citado en el paper de la
  tesis; usa Molmo + ray-casting contra malla (lo que este pipeline
  reemplaza).
- Varma T., M. et al. (2024). *Lift3D: Zero-Shot Lifting of Any 2D Vision
  Model to 3D*. CVPR 2024.
- CVPC (arXiv:2512.04686) — base adicional de **EXP-LIT-1**
  (`Pipeline_Experiments/README.md`).

### 8.5 Por qué evitar correspondencia punto-a-punto explícita entre vistas (motivación de Flujo C sin identidad persistente, §2.1)

Fuente: `docs/pipeline_sin_malla.md` §8.5.

- Bhat, S. D., Yamasaki, T. (2026). *Consistent Yet Wrong: Evidence
  Insensitivity in Spatial Vision-Language Models*. arXiv:2606.02742.
- ZeroKey (arriba) — fracaso empírico de MLLMs en *point-level reasoning*
  con identidad persistente.
- Evidencia empírica propia: un diseño anterior de Flujo C que pedía
  rastrear landmarks con identidad persistente entre vistas degeneró en
  patrones mecánicamente equiespaciados (`molmo_multiview_runner.py`,
  docstring de `build_flow_c_prompts`; `docs/features/molmo-pointing.md`
  §Results & Observations) — mismo fallo central de ZeroKey.

### 8.6 Prompts de literatura implementados en Ronda 5 (§2.5, `axis_lit2_grid`/`axis_lit3_cot`)

Fuente: `MolmoPointing/prompts_registry.py`, comentarios inline.

- MOKA (arXiv:2403.03174) — estilo de Grid Overlay, base de `axis_lit2_grid`.
- Structured CoT (arXiv:2507.13362) — base de `axis_lit3_cot`.

### 8.7 Gate de consolidación de plano con malla real (EXP-F, §4.1)

Fuente: `Pipeline_Experiments/README.md`,
`Pipeline_Experiments/triangulation_variants/exp_f_sde_gate.py`.

- Gao, Y. et al. *PRS-Net*. IEEE TVCG 2021 (DOI:10.1109/TVCG.2020.3003823).
- `EnhancedBackProjection` (WACV 2026) — repositorio de referencia del que
  `Mapping/reference_metrics.py`/`Mapping/evaluate.py` portan `SDE_ref`/`F1_ref`
  verbatim (`docs/verificacion_metricas_literatura.md` §A.3/B.3) — la misma
  fórmula que `expF` usa para el gate, en vez de solo para puntuar.

### 8.8 Nota de verificación

Varias referencias de arriba son muy recientes (2025–2026) —
`docs/pipeline_sin_malla.md` §9 marca explícitamente que conviene verificar
su estado de publicación/peer-review antes de citarlas como establecidas en
la tesis, y tratar sus cifras como preliminares.
