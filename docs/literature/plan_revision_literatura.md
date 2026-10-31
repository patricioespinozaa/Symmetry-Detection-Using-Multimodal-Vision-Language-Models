# Capítulo 2 — Revisión de la literatura: plan de reorganización y textos borrador

**Tesis:** Detección de simetrías en objetos tridimensionales mediante modelos de lenguaje-visión multimodales a partir de múltiples vistas
**Flujo actual que este capítulo debe respaldar:**

- **Método principal (sin malla):** renderizado multivista con cámaras conocidas → consulta multi-imagen al VLM (Molmo2; contraste con Qwen3-VL y MolmoPoint-8B) → parseo de los puntos 2D → retroproyección de cada punto a un rayo de cámara → triangulación del eje (planos de interpretación) o del plano (rectas por pares de vistas y consolidación) → evaluación contra la simetría de referencia y contra predictores que no miran la imagen.
- **Análisis de referencia (con malla, solo para los mejores prompts):** los mismos puntos se retroproyectan sobre la malla (ray casting) y se estima la simetría a partir de los puntos de contacto. No es una alternativa al método: sirve para ver qué parte del error se debe a la triangulación (PI3, H2).
- **Otros usos de la malla:** la evaluación (simetría de referencia, SDE_ref, F1_ref) y el control EXP-F.

**Regla para el capítulo:** la literatura del método principal y la del análisis de referencia se mantienen separadas. La del análisis de referencia va en una subsección propia (2.3.6), que empieza aclarando que no describe el método propuesto. En el resto del capítulo, cuando se menciona la retroproyección sobre la malla, se remite a 2.3.6.

## Cómo usar este archivo

- **Parte A**: el plan (estructura, qué sacar, qué mantener, qué añadir, vacíos, inconsistencias).
- **Parte B**: textos borrador en LaTeX, uno por subsección, listos para copiar y pegar. Cada bloque comienza con su `\section` o `\subsection`.
- **Parte C**: tabla de claves de cita. Los borradores usan `\cite{clave}` con claves propuestas; la tabla dice cuáles ya están en tu bibliografía (con su número actual) y cuáles hay que agregar, con los datos tomados de tus `.md`.
- **Parte D** (opcional): textos de reemplazo para las partes de la introducción que aún describen el flujo con malla.
- **Parte E**: lista de control.

**Convenciones dentro de los borradores (comentarios LaTeX, no se imprimen):**

- `% VERIFICAR:` la afirmación sale de tus notas o del resumen de un trabajo que aún no lees completo. Confírmala en la fuente antes de entregar.
- `% PENDIENTE P1`, `P2`, `P4`, `P5`: depende de una decisión tuya (ver más abajo). Hay una versión por defecto y una alternativa.
- `[falta cita]`: la afirmación necesita una referencia que no está en tus archivos.

No se inventaron referencias: todas las citas vienen de la tesis o de los `.md`.

### Decisiones pendientes

| ID | Pregunta | Qué cambia en el capítulo |
|---|---|---|
| P1 | ¿Se mantiene la comparación con Aguirre y Sipiran (3.2.12)? | Si sí, usa el párrafo extendido de 2.2.3. Si no, basta el párrafo base |
| P2 | ¿La intensidad de la luz se cambia en el renderizador o sobre la imagen ya renderizada? | Solo en el primer caso se cita a Kar et al. (2022) en 2.5.3 |
| ~~P3~~ | **Resuelta en la nueva versión:** se mantiene el contraste con Qwen3-VL y se agrega MolmoPoint-8B como tercer modelo | Los párrafos de Qwen3-VL y MolmoPoint-8B en 2.4.1 ya no son opcionales. Falta la cita de MolmoPoint-8B |
| P4 | ¿Están implementadas la línea base (iii) de PI1 (puntos al azar dentro de la silueta) y el predictor del centro? | Frase final de 2.6.3; vacío 2 |
| P5 | En el análisis con malla, ¿los puntos de contacto se consolidan antes de la SVD/PCA (por ejemplo, con HDBSCAN o eliminando redundantes) o la SVD se aplica directamente? | Si se consolidan, se cita a Campello et al. [6] en 2.3.6; si no, se elimina [6] |

---

# PARTE A — Plan de reorganización

## A.1 Estructura propuesta

El capítulo sigue el orden del flujo: el problema (simetría), cómo se ha resuelto, las herramientas (geometría multivista y VLMs), sus riesgos, cómo evaluar y, al final, la brecha.

| Sección | Qué cubre | Por qué la necesita el flujo actual |
|---|---|---|
| **Intro del cap. 2** | Presenta los 7 bloques en el orden del flujo | Orienta al lector |
| **2.1 Simetría en objetos tridimensionales** | | |
| 2.1.1 Definición y representación | Simetría especular (normal + punto) y rotacional (dirección + punto); dirección sin signo; simetría como grado de coincidencia | Define qué predice el método y de dónde viene el SDE |
| 2.1.2 Detección sobre geometría explícita | Métodos clásicos que requieren la malla | Muestra la dependencia que el trabajo elimina |
| 2.1.3 La simetría como prior para reconstrucción | Patrimonio cultural, completación, segmentación | Motivación aplicada (un párrafo) |
| **2.2 Detección de simetría con aprendizaje y desde imágenes** | | |
| 2.2.1 Métodos aprendidos sobre geometría | PRS-Net; origen del SDE como criterio | Antecedente de la métrica y del gate de EXP-F |
| 2.2.2 Desde imagen única | SymmetryNet, Reflect3D: entrenamiento supervisado; ambigüedad de la vista única | Justifica usar varias vistas |
| 2.2.3 Modelos fundacionales de visión aplicados a simetría | Aguirre y Sipiran, Reflect3D (DINOv2), CLIPSym, Orient Anything V2 | Contraste directo: la información existe, pero siempre con malla o decoder entrenado |
| **2.3 Geometría multivista: de puntos 2D a elementos 3D** | | |
| 2.3.1 Cámara, retroproyección y rayos | Modelo de cámara; un punto 2D define un rayo | Respaldo de la etapa de retroproyección |
| 2.3.2 Triangulación de puntos y de líneas | Plano de interpretación; intersección de planos | Base de la estimación del eje |
| 2.3.3 Simetría especular y geometría multivista | Recuperar el plano de simetría desde vistas | Base de la estimación del plano y de las vistas "de canto" |
| 2.3.4 Condicionamiento y propagación del error | Error del rayo → error de la dirección; degeneraciones | Reemplaza la antigua 2.5; respalda el diagnóstico 4.3.1 |
| 2.3.5 Estimación robusta y ponderada | RANSAC (y secuencial), IRLS, pesos por vista, NMS, silueta | Respalda las ablaciones EXP-A a E, la consolidación y el filtro de silueta |
| **2.3.6 Retroproyección sobre la malla: condición de referencia** | Ray casting contra la malla, consolidación de puntos de contacto, SVD/PCA sobre la nube, propagación del error 2D a la superficie. **No es parte del método: respalda el análisis con malla** | Respalda PI3 y H2. Separada del resto para que no se confunda con el método sin malla |
| **2.4 VLMs con capacidad de pointing** | | |
| 2.4.1 Grounding mediante puntos | Molmo, Molmo2, Qwen3-VL; formato de salida; evaluación de terceros | Respaldo de consulta y parseo |
| 2.4.2 Prompting visual y razonamiento | CoT, Set-of-Mark, grilla, CoT estructurado | Respaldo de familias de prompt y Flujos B y C |
| 2.4.3 Llevar a 3D las predicciones 2D de un VLM | ZeroKey, Garosi, Lift3D, ZeroDex, CVPC | Posiciona el método: sin malla, sin correspondencias |
| **2.5 Capacidades y limitaciones espaciales de los VLMs** | | |
| 2.5.1 ¿Representan estructura 3D los modelos fundacionales? | Evidencia a favor y en contra | Respalda la hipótesis y acota su alcance |
| 2.5.2 Fallas de razonamiento espacial | Insensibilidad a la evidencia, confusión de lados | Anticipa resultados cerca del azar |
| 2.5.3 Robustez a resolución y exposición | Corrupciones fotométricas, severidad | Respalda los experimentos de resolución e iluminación |
| **2.6 Evaluación: datos, métricas, líneas base y sesgos** | | |
| 2.6.1 ShapeNet y su alineación canónica | Banco de pruebas y fuente de sesgo | Respalda los predictores triviales |
| 2.6.2 Métricas en detección de simetría | SDE, F1 con varios candidatos, AUC angular | Sitúa las métricas de 3.2.9 |
| 2.6.3 Atajos y líneas base sin imagen | Shortcut learning, priors, sesgo central | Respalda PI1 (ii)(iii) y PI4 |
| 2.6.4 Comparaciones estadísticas (opcional) | Pruebas pareadas, bootstrap | Un párrafo; el detalle va en metodología |
| **2.7 Síntesis y brecha** | Tabla comparativa y brecha | Cierra el capítulo |

## A.2 Qué sacar del texto actual

En la versión nueva de la tesis, el texto antiguo quedó como secciones 2.8 a 2.13, debajo de los títulos nuevos. Abajo se usa esa numeración nueva, con la antigua entre paréntesis. Cuando pegues los borradores, elimina las secciones 2.8–2.13 completas: lo que se conserva ya está incorporado en los borradores de la Parte B.

| Ubicación | Contenido | Motivo |
|---|---|---|
| 2.8.1 (antes 2.1.1), "Su adopción en lugar de… Kazhdan" y "fundamento… en los trabajos clásicos [24, 39]" | Justificación de la SVD | Se contradice y la atribución a [24] no está respaldada. **El resto de 2.8.1 no se elimina: pasa, corregido, a 2.3.6** (ver A.3) |
| 2.10 (antes 2.3), Harish y Prasad [21] (Photo2CAD) | CAD desde dibujos 2D | Fuera de alcance; aporta poco; preprint |
| 2.11 (antes 2.4), "lo que sugiere una mayor consistencia espacial… se espera que el modelo identifique puntos estructuralmente correspondientes a través de distintas vistas" | Expectativa de correspondencia | El eje no necesita correspondencias (3.2.6); CVPC muestra correspondencia casi al azar; el Flujo C con identidad persistente degeneró |
| 2.11, "en el conjunto de 11 benchmarks…", Elo, CountBench | Benchmarks generales | No miden pointing ni la tarea |
| 2.11, "La contribución diferencial de Molmo2… pointing de video…" | Cifras de video | El método usa multi-imagen. Eliminar o dejar en el anexo |
| 2.11, ZeroKey: "en lugar de un esquema de consenso tipo RANSAC" | Comparación con RANSAC | No describe ni el método ni el análisis con malla |

**Cambio respecto a la versión anterior de este plan:** la SVD sobre la nube de puntos (2.8.1), la primera frase de 2.12 (antes 2.5), Kim et al. [25] y Campello [6] ya no se eliminan. Pasan a 2.3.6, porque respaldan el análisis con malla.

## A.3 Qué mantener

| Contenido actual | Acción | Destino |
|---|---|---|
| Intro del capítulo | Reescribir (borrador en B) | Intro cap. 2 |
| 2.1: Kazhdan [24], Xu [39], Sipiran [37] | Mantener, acortando | 2.1.2 |
| 2.1: "En conjunto, estos trabajos comparten dos características…" | Mantener; añadir que la dirección no tiene signo | 2.1.1 |
| 2.2 completa (Gregor [19], Papaioannou [30], Mavridis [28]) | Reducir a un párrafo; añadir Simari [36], Mitra [29], Pöllabauer [33] | 2.1.3 |
| 2.3: Render4Completion [23] | Una frase (opcional) | Intro de 2.3 |
| 2.3: Aguirre y Sipiran [1] | Reescribir: ellos retroproyectan features a la malla; aquí se retroproyectan puntos a rayos y se triangula sin malla | 2.2.3 |
| 2.3, cierre ("ninguno de ellos incorpora…") | Matizar con CLIPSym | 2.7 |
| 2.4: Molmo [9], Molmo2 [8], caveat de imágenes naturales | Mantener (sin la expectativa de correspondencia) | 2.4.1 |
| Recuadro [REVISIÓN] de criterios de selección | Mover a metodología 3.2.3; en el cap. 2 quedan solo los hechos | 3.2.3 / 2.4.1 |
| 2.4: Garosi [15], ZeroKey [17] | Reencuadrar como contrastes que dependen de la geometría | 2.4.3 |
| Cierre de 2.4 (brecha relacional) | Ampliar con "sin malla ni correspondencias" | 2.7 |
| 2.5: Hao [20] | Reencuadrar hacia la resolución del pointing | 2.5.3 |
| 2.5: Zhang y Wang [42] | Mantener, moderando la generalización | 2.4.1 |
| 2.5: Fooladgar [12] | Reescribir: el error se propaga a la normal del plano de interpretación. Se cita también en 2.3.6 para el caso con malla | 2.3.4 y 2.3.6 |
| 2.8.1 (antes 2.1.1) "Ajuste mediante SVD": PCA sobre una nube 3D (eje = máxima varianza, normal = mínima, centroide como ancla) | Mantener y corregir: (a) quitar la atribución a [24, 39]; (b) aclarar que solo se aplica en el análisis con malla; (c) añadir que los puntos de contacto caen sobre la superficie y no sobre el eje, salvo en sus polos | 2.3.6 |
| 2.12 (antes 2.5), primera frase ("…punto 3D significativo sobre la superficie del objeto") | Reescribir para el caso con malla | 2.3.6 |
| 2.12: Kim et al. [25] | Mantener (opcional), solo para el análisis con malla: pequeños errores 2D se amplifican en 3D tras el ray casting | 2.3.6 |
| 2.11: descripción de ZeroKey (ray casting + HDBSCAN) | Además de 2.4.3, usarla en 2.3.6 como el antecedente de la condición con malla | 2.4.3 y 2.3.6 |
| Bibliografía [6] Campello (HDBSCAN) | Mantener solo si el análisis con malla consolida puntos por clustering (P5), o si se cita al describir ZeroKey | 2.3.6 |
| 2.6: ShapeNet [7] | Añadir alineación canónica y fuente de anotaciones | 2.6.1 |
| Tabla A.1 | Separar video, mover Qwen3-VL fuera de propietarios, marcar autorreportes | Anexo |

## A.4 Qué añadir desde los `.md`

| Subsección | Contenido | Archivo (sección) | Referencias |
|---|---|---|---|
| 2.1.1 / 2.1.2 | Transformada de simetría especular; panorama | `01_triangulacion_sin_malla.md` §4 | Podolak [31], Mitra [29] |
| 2.2.1 | SDE como pérdida y criterio de validación | `03_consolidacion_multiplano.md`, `09_metricas_evaluacion.md` | PRS-Net [14] |
| 2.2.2 | Ambigüedad de la vista única; varias simetrías | `01` §2, `03` | Reflect3D [26], SymmetryNet [35] |
| 2.2.3 | Candidatos + Chamfer reflectivo + duplicados; DINOv2 congelado; lenguaje; categoría | `03`, `literatura-por-leer.md` | [1], [26], CLIPSym, Orient Anything V2 |
| 2.3.1 | Modelo de cámara; epipolar | `01` §1, `07_experimentos_literatura_vlm.md` | [22] |
| 2.3.2 | Triangulación de líneas; Plücker | `01` §1 | [4], Hartley y Sturm 1997, Hartley 1997, Wu 2015, Josephson y Kahl, Line3D++, LIMAP, Mateus 2021 |
| 2.3.3 | Plano de simetría desde vistas | `01` §2 | [13], Criminisi 2000, Olsson y Eriksson 2011, Gao y Yuille 2017, Hong 2004, Sinha 2012 |
| 2.3.4 | 1/sin θ, número de condición, brazo de palanca, degeneraciones | `08_diagnostico_casos_conditioning.md`, `01`, `02_ablaciones_triangulacion.md` | [22], [16], [10], [40], [4], Ghasemi |
| 2.3.5 | M-estimadores, IRLS, RANSAC secuencial, pesos, NMS, silueta | `02`, `03`, `04_filtrado_vistas_puntos.md` | [11], Huber 1964, Holland y Welsch 1977, [34], Iskakov 2019, Lee y Civera 2019, Neubeck y Van Gool 2006, Laurentini 1994 |
| 2.3.6 | Método con malla de contraste: ray casting + SVD/PCA; ZeroKey como antecedente | `01_triangulacion_sin_malla.md` (encabezado "Qué se hace" y §3) | [17], [1], [15], [22], [12], [25]; [6] según P5 |
| 2.4.1 | Formato `<point>`, coordenadas | `01` §3, `05_sesgo_centro_x500.md`, `06_prompts_pointing.md` | [9] |
| 2.4.2 | CoT libre vs. estructurado; marcas; grilla | `06`, `07` | Wei 2022, Set-of-Mark 2023, [27], SketchVLM, [38] |
| 2.4.3 | ZeroDex; CVPC 3,02 % vs 60–66 %; identidad persistente | `01` §3, `07`, `06`, `08` | [17], ZeroDex, Lift3D, CVPC |
| 2.5.1 | Estructura 3D en representaciones | `literatura-por-leer.md` | El Banani 2024, GIQ, [26], CLIPSym |
| 2.5.2 | Fallas espaciales; lados simétricos; consistentes pero incorrectas | `literatura-por-leer.md`, `05`, `06`, `04` | Tong 2024, Kamath 2023, SpatialVLM, VRB, Telling Left from Right, [5] |
| 2.5.3 | Robustez fotométrica; severidad | `lectura_iluminacion.md` P1–P2 | VLM-RobustBench, Hendrycks 2019, Usama 2025, B-AVIBench, [20]; Kar 2022 (P2) |
| 2.6.1 | Alineación por categoría; sesgo de dataset | `lectura_baselines.md` §2.1–2.2 | [7], Torralba y Efros 2011 |
| 2.6.2 | AUC truncado; F1 húngaro; SDE | `09` | SuperGlue 2020, [35], Kuhn 1955, [14], [1], [26] |
| 2.6.3 | Sin imagen; shortcut; sesgo central; categoría | `lectura_baselines.md` §2.3 y §3, `05`, `baseline_aleatorio.md` §7, `literatura-por-leer.md` | Goyal 2017, Agrawal 2018, MMStar 2024, Geirhos 2020, Tatler 2007, Judd 2009 |
| 2.6.4 | Pruebas pareadas, bootstrap | `lectura_baselines.md` §4, `08` | Dror 2018, Efron y Tibshirani 1993, Demšar 2006, Benjamini y Hochberg 1995 |

**No incluir en el cap. 2 (va en otro capítulo o no se cita aún):**

- `baseline_aleatorio.md` §1–6 (fórmulas, código, simulación, opciones A/B/C, lista de control) → metodología 3.2.9 o anexo. Muller y Marsaglia se citan allí.
- `lectura_iluminacion.md` Prioridad 3 (Ramachandran, Horn) → solo con luz direccional. "Cómo usar" → metodología.
- `literatura-por-leer.md`: el párrafo de reescritura es para `intro.tex:19`; la conjetura sobre *Telling Left from Right* va en la discusión como hipótesis propia.
- `02`: asignación de referencias por EXP (Wu 2021, AssemblyHands-X [2], COMPOSE, Zhang 2020 [41]) → solo Tabla 3.8. Hess-Flores no tiene título: no citar. "Lectura de resultados" → discusión.
- `03`: E3Sym y Je et al. (no verificados) → no citar todavía. Sobrepredicción y techo de EXP-F → resultados y discusión.
- `01`: VPLNet → marginal, opcional.
- Cifras propias (82,8 % → 51,5 %, 63 % y 98,9 %, techos de LIT-x, 18–24 %) → resultados.
- Implementación de LIT-1 sin correr → no mencionar.
- Estratificación BUENO/MEDIO/MALO, regresión a la media (Barnett 2005), "mejor de N" (Cawley y Talbot 2010) → metodología o amenazas a la validez.
- Herencia del sesgo central humano en PixMo (`05`) → discusión, como hipótesis.
- "No existe un F1 estándar para eje" (`09`) → observación propia o buscar cita.

## A.5 Vacíos (literatura por buscar)

1. **Simetría rotacional desde imágenes.** Casi todo lo disponible es especular. Búsqueda: *surface of revolution axis from silhouettes*, *rotational symmetry axis detection images*.
2. **Línea base (iii) de PI1** (puntos al azar dentro de la silueta): sin respaldo (P4).
3. **Origen de las anotaciones de simetría** del subconjunto de 1.700 objetos [falta cita].
4. **Resolución y tokenización en VLMs** (224/448/1136 px, recortes de Molmo2): Hao [20] es de detección de objetos.
5. **Multi-imagen vs. una vista por llamada**: solo CVPC. Búsqueda: *multi-image spatial reasoning benchmark VLM*, *multi-view consistency VLM*.
6. **Pointing de Qwen3-VL evaluado por terceros**: la cifra de Point-Bench no tiene fuente.
7. **Contaminación** (ShapeNet/CAD en PixMo): sin literatura ni verificación.
8. **Priors de categoría sobre el tipo de simetría** (4.12): solo Orient Anything V2. Búsqueda: *semantic keypoints symmetry*, *category-level symmetry prior*.
9. **Error de traslación como métrica de simetría** [falta cita o declarar propio].
10. **Convención de cámara/NDC de PyTorch3D** (3.2.5) [falta cita].
11. **Modelos con pointing nombrados en 1.1** (Gemini-2.5-Pro, Gemini-Robotics-ER-1.5, Poivre-7B) [falta cita].
12. **Condicionamiento "de canto" del plano**: solo cualitativo en la literatura revisada → declararlo contribución.
13. **MolmoPoint-8B**: ningún archivo tiene su referencia ni la fuente de la cifra de PointBench (70,7) [falta cita].
14. **Descomposición del error entre detección y geometría** (PI3): no hay literatura en los `.md` sobre separar ambos factores con una condición de referencia o con puntos oráculo. Búsqueda: *oracle analysis error decomposition keypoint triangulation*, *upper bound analysis perfect detections*.
15. **Intersección rayo-malla** (algoritmo de ray casting usado en el análisis con malla) [falta cita, o citar la biblioteca usada].

## A.6 Inconsistencias

**A. Introducción y metodología (revisado sobre la versión nueva de la tesis; textos de reemplazo en la Parte D)**

PI3 y H2 son coherentes con el análisis con malla. Lo que falla es que la introducción todavía presenta la malla como parte del método principal, o dice que solo se usa para evaluar:

- 1.2 ("Utilizando la malla exclusivamente para obtener la simetría de referencia…") y PI1 ("La malla se utiliza exclusivamente…") contradicen a PI3, que también usa la malla. Hay que decir que la malla no interviene en la estimación del método, y que se usa además en un análisis de referencia.
- PI3 y H2 no dicen que el análisis se hace solo sobre los mejores prompts.
- H2 dice "eliminando el error de triangulación", pero la condición con malla también cambia el estimador (SVD/PCA sobre puntos de contacto en lugar de intersección de planos). La diferencia entre H1 y H2 mezcla el efecto de tener la profundidad con el del estimador; conviene declararlo.
- Objetivo específico 3 ("conjunto de puntos 3D consistente sobre la malla original") y objetivo 4 describen el flujo con malla como el principal. Además, falta un objetivo para el análisis con malla.
- 1.5 (iv) sigue invertido: dice que el esquema principal usa la malla y que el alternativo prescinde de ella.
- 1.6, etapas 3–4: describen el ray casting y la SVD sobre la nube como el método principal.
- 1.1, MolmoPoint-8B: la frase quedó cortada ("…en PointBench con 70.7") y no tiene cita. Además, el párrafo anterior dice que "las conclusiones… se restringen a estos dos modelos", PI4 y el objetivo 6 solo mencionan Qwen3-VL, y la sección 4.11 lo llama "MolmoPointing". Hay que unificar el nombre y pasar de dos a tres modelos.
- 1.5, vistas: dice "desde 1 hasta 114, en configuraciones de 1, 6, 14, 26, 42". El máximo ya no es 114, y nv = 1 no se puede triangular (3.2.6), aunque sí podría usarse en el análisis con malla. Aclararlo.
- 1.1: "Molmo-8B pertenece a la familia Qwen3" sigue sin corregir.
- 3.2.11: "método sin malla (H2)" debería ser H1.
- Discusión, "Preguntas": clustering espacial; [9] usado en lugar de [1]; [10, 11, 12] presentados como literatura de grounding (son Draper, Fischler y Fooladgar).
- 3.2.12: llama "métodos tradicionales" a [1], que usa DINOv2.
- La metodología todavía no tiene una sección que describa el análisis con malla. 2.3.6 la referencia como `sec:analisis-malla`.

**B. Dentro del cap. 2**

- 2.1.1 se contradice (SVD "en lugar de" Kazhdan vs. "fundamento" de [24, 39]).
- La correspondencia esperada de Molmo2 (2.4) choca con CVPC y con 3.2.6.
- "Más semántica, mejor geometría" (cierres de 2.3 y 2.4) choca con Reflect3D.

**C. Entre los `.md` y la tesis**

- z de `axis_v08`: −3,6 en la tesis vs. ≈ −2,4 con la fórmula simple. Documentar la varianza usada.
- Desviación estándar del azar: 21,57° (`baseline_aleatorio.md`) vs. 21,56° (ec. 3.20). Correcto: 21,56°.
- `lectura_baselines.md` §4 dice N = 100; son 850 por tipo (30 en el sandbox).
- 3.2.9 dice que en plano no se usa el azar, pero la Tabla 4.17 lo reporta.
- El predictor del centro (`lectura_baselines.md` §3, PI4) no aparece en 3.2.9.

**D. Referencias**

- Sin citas en el cuerpo: [5], [6], [13]. [5] y [13] pasan a citarse en 2.5.2 y 2.3.3.
- Marcadas [nuevo] en los `.md` pero ya en la bibliografía: [31], [29], [7], [9], [35], [11].
- Confusiones posibles: Wu 2015 ≠ Wu 2021; Zhang 2020 [41] ≠ Zhang y Wang 2024 [42] ≠ Zhang et al. 2024 (*Telling Left from Right*); Chen L. 2024 (MMStar) ≠ Chen B. 2024 (SpatialVLM).
- Incompletas: [2], [3], [32], [38], [41], Hess-Flores, B-AVIBench, venue de Josephson y Kahl.
- Preprints por verificar: GIQ, VRB, ZeroDex, [5], SketchVLM, CVPC, Usama, COMPOSE, VLM-RobustBench.
- Citadas de memoria según los `.md`: Muller, Marsaglia, Efron y Tibshirani, Hendrycks.

**E. Afirmaciones sin cita o demasiado generales**

- 2.6: anotaciones de simetría en ShapeNet [falta cita].
- Cierre de 2.3: "información suficiente" con dos trabajos.
- 2.5: un estudio de GPT-4V en observación terrestre [42] generalizado a todos los VLMs.
- 2.4: la atención bidireccional "sugiere mayor consistencia espacial" (interpretación propia).
- Recuadro: Point-Bench 68,5 vs. 65,1 sin fuente; hallazgo de PointArena sobre CoT [VERIFICAR].
- 1.1: "Molmo-8B" → Molmo2-8B.

---

# PARTE B — Textos borrador por sección (LaTeX)

> Copia cada bloque en tu `.tex`. Los comentarios `%` no se imprimen. Reemplaza las claves de `\cite{}` por las tuyas usando la Parte C.

## Introducción del capítulo

```latex
\chapter{Revisión de la literatura}

Este capítulo revisa la literatura que sustenta el método propuesto, siguiendo el orden de sus etapas. La Sección~\ref{sec:lit-simetria} define la simetría en objetos tridimensionales y revisa su detección sobre geometría explícita y su uso como prior de reconstrucción. La Sección~\ref{sec:lit-aprendizaje} revisa los métodos que detectan simetría mediante aprendizaje, a partir de imágenes o apoyados en modelos fundacionales de visión. La Sección~\ref{sec:lit-multivista} presenta la geometría multivista que permite llevar puntos marcados en imágenes a elementos 3D sin acceso a la malla del objeto, y cierra con la retroproyección sobre la malla, que no forma parte del método y se emplea solo como condición de referencia para analizar el error de la triangulación. Las Secciones~\ref{sec:lit-pointing} y~\ref{sec:lit-limitaciones} revisan los modelos de lenguaje-visión (VLM) con capacidad de \textit{pointing} y la evidencia sobre sus capacidades y limitaciones espaciales. La Sección~\ref{sec:lit-evaluacion} revisa los datos, las métricas y las líneas base con que se evalúa la detección de simetría, junto con los sesgos que pueden inflar los resultados. Finalmente, la Sección~\ref{sec:lit-brecha} sintetiza lo revisado y define la brecha que aborda este trabajo.
```

## 2.1 Simetría en objetos tridimensionales

```latex
\section{Simetría en objetos tridimensionales}
\label{sec:lit-simetria}
```

### 2.1.1 Definición y representación

```latex
\subsection{Definición y representación}

Un objeto presenta simetría especular (o reflectiva) respecto de un plano $\Pi$ si la reflexión sobre $\Pi$ lo deja invariante, y simetría rotacional respecto de un eje si alguna rotación en torno a él lo deja invariante. En el caso de la simetría de revolución, la invariancia se cumple para cualquier ángulo de rotación, como en una botella o un cuenco; en la simetría rotacional discreta, solo para múltiplos de un ángulo, como en una mesa de cuatro patas.

Ambos tipos se representan de forma compacta mediante una dirección y un punto. Un plano de simetría queda definido por su normal unitaria $\hat{n}$ y un punto $o$ contenido en él, y un eje por su dirección unitaria $\hat{d}$ y un punto $o$ por el que pasa. Esta dirección no tiene signo: $\hat{n}$ y $-\hat{n}$ describen el mismo plano, y $\hat{d}$ y $-\hat{d}$ el mismo eje. Por ello, la discrepancia entre dos direcciones se mide habitualmente con el valor absoluto de su producto interno, lo que acota el error angular entre $0^\circ$ y $90^\circ$.

Un concepto fundacional es el descriptor de simetría reflectiva de Kazhdan et al.~\cite{kazhdan2002}, que caracteriza el grado de simetría de un objeto respecto de un plano arbitrario mediante una medida continua. Esta formulación establece una idea que perdura en trabajos posteriores~\cite{podolak2006,mitra2013}: la simetría no se trata como una propiedad presente o ausente, sino como un grado de coincidencia entre el objeto y su transformación (reflejada o rotada), que puede optimizarse o muestrearse. De ella se desprenden dos rasgos que comparten los trabajos revisados en esta sección. Primero, la simetría se expresa mediante un vector (normal del plano o dirección del eje) y un punto de referencia. Segundo, una hipótesis de simetría se valida midiendo el error residual tras aplicar la transformación correspondiente sobre la geometría, idea que constituye el antecedente directo del Error de Distancia de Simetría (SDE) empleado en la evaluación (Sección~\ref{sec:lit-metricas}).
```

### 2.1.2 Detección sobre geometría explícita

```latex
\subsection{Detección sobre geometría explícita}

La detección de simetrías en modelos tridimensionales se ha abordado tradicionalmente mediante el análisis directo de la geometría disponible, es decir, mallas poligonales, nubes de puntos o vóxeles. Además del descriptor de Kazhdan et al.~\cite{kazhdan2002}, Podolak et al.~\cite{podolak2006} proponen una transformada de simetría especular planar que asigna a cada plano posible una medida de simetría, de modo que los planos de simetría de un objeto aparecen como máximos locales de esa función; esto permite extraer varios planos por objeto. Xu et al.~\cite{xu2022} abordan la detección de planos de simetría reflectiva extrínseca operando sobre la superficie del objeto sin requerir una parametrización previa, lo que resulta relevante para colecciones de topologías heterogéneas como ShapeNet. Sipiran et al.~\cite{sipiran2014} extienden el problema a mallas parciales o incompletas, proponiendo descriptores que toleran la ausencia de regiones del objeto sin perder la capacidad de identificar su simetría global. Mitra et al.~\cite{mitra2013} reúnen estos y otros enfoques en una revisión de la extracción de simetrías en geometría 3D y sus aplicaciones.

Todos estos métodos comparten un supuesto: la geometría explícita del objeto está disponible en el momento de la inferencia. Cuando no lo está, o está incompleta más allá de lo que el método tolera, no son aplicables. Este supuesto es el que el presente trabajo busca eliminar en la etapa de estimación. La malla se reserva para la evaluación y para un análisis de referencia que cuantifica el error atribuible a la triangulación (Sección~\ref{sec:lit-retro-malla}).
```

### 2.1.3 La simetría como prior para reconstrucción

```latex
\subsection{La simetría como prior para reconstrucción}

Más allá de su detección, la simetría se utiliza como prior estructural en tareas posteriores, como la segmentación de mallas en partes~\cite{simari2006,podolak2006} y la alineación de formas de una misma clase~\cite{kazhdan2002}. Su uso es especialmente frecuente en la reconstrucción de objetos de patrimonio cultural, donde los escaneos suelen presentar fracturas, oclusiones o pérdida de material. Gregor et al.~\cite{gregor2014} describen un flujo de reconstrucción automatizada en el que la identificación de simetrías permite inferir la forma de las regiones faltantes a partir de las conservadas; Papaioannou et al.~\cite{papaioannou2017} amplían la idea a un sistema completo que va del reensamblaje de fragmentos a la completación del objeto, y Mavridis et al.~\cite{mavridis2015} incorporan la simetría como una de las restricciones de un enfoque de optimización $k$-\textit{sparse} para la completación.

En estos trabajos la simetría no es el objetivo final, sino un medio para resolver un problema posterior, y su estimación sigue dependiendo de una captura 3D. Esto motiva la pregunta de si la simetría puede inferirse a partir de imágenes, lo que extendería su utilidad como prior a escenarios en que solo se dispone de registros fotográficos, como objetos destruidos o inaccesibles~\cite{pollabauer2024}. El presente trabajo no aborda ese escenario directamente, sino que estudia, en condiciones controladas, si un VLM puede aportar la información necesaria a partir de vistas renderizadas.
```

## 2.2 Detección de simetría con aprendizaje y desde imágenes

```latex
\section{Detección de simetría con aprendizaje y desde imágenes}
\label{sec:lit-aprendizaje}
```

### 2.2.1 Métodos aprendidos sobre geometría

```latex
\subsection{Métodos aprendidos sobre geometría}

PRS-Net~\cite{gao2021prsnet} reemplaza la búsqueda explícita por una red que predice planos de simetría a partir de una representación volumétrica del objeto, entrenada de forma no supervisada. Su función de pérdida mide la distancia entre el objeto transformado por la simetría predicha y la superficie original, y el mismo criterio se emplea para validar los planos predichos. Esta medida, conocida como error de distancia de simetría (SDE), se ha convertido en una métrica habitual para la simetría planar en ShapeNet~\cite{gao2021prsnet} y es la que se adopta en este trabajo para evaluar la coherencia de una predicción con la forma del objeto. PRS-Net, sin embargo, mantiene la dependencia de la geometría: su entrada es el propio objeto 3D.
% VERIFICAR: si PRS-Net también predice ejes de rotación, mencionarlo aquí.
```

### 2.2.2 Desde imagen única

```latex
\subsection{Detección desde una imagen única}

Una segunda familia de métodos predice la simetría a partir de una sola imagen. SymmetryNet~\cite{shi2020symmetrynet} predice simetrías reflectivas y rotacionales de formas 3D a partir de una imagen RGB-D de una vista, admite varias simetrías por objeto y se evalúa mediante error angular y curvas de precisión-exhaustividad. Reflect3D~\cite{li2025reflect3d} aborda la detección de simetría 3D a partir de una imagen RGB y predice varios planos por objeto. Ambos requieren entrenamiento supervisado con anotaciones de simetría, lo que liga su desempeño a la distribución de los datos de entrenamiento.

Reflect3D señala además una limitación intrínseca del caso de vista única: una sola imagen es ambigua para determinar el plano de simetría~\cite{li2025reflect3d}. Esta ambigüedad es uno de los motivos por los que el presente trabajo combina información de varias vistas con parámetros de cámara conocidos, en lugar de inferir la simetría desde una imagen aislada.
```

### 2.2.3 Modelos fundacionales de visión aplicados a simetría

```latex
\subsection{Modelos fundacionales de visión aplicados a simetría}

Trabajos recientes incorporan modelos fundacionales de visión, preentrenados a gran escala, como fuente de información sobre la forma. Aguirre y Sipiran~\cite{aguirre2026} extraen características visuales con DINOv2 desde renderizados multivista del objeto y las retroproyectan sobre la malla original para detectar su simetría 3D. Su procedimiento genera varios planos candidatos, los filtra mediante una distancia de Chamfer reflectiva y elimina los duplicados por similitud rotacional. Reflect3D~\cite{li2025reflect3d} usa características congeladas de DINOv2 como base de un detector entrenado aparte, y sus autores argumentan que DINOv2, menos orientado a la semántica que CLIP, se alinea mejor con la geometría. En la dirección opuesta, CLIPSym~\cite{yang2025clipsym} muestra que un modelo preentrenado con lenguaje (CLIP), combinado con prompts agrupados por objeto, aporta información útil para detectar simetría en imágenes 2D, también con un decodificador entrenado. Orient Anything V2~\cite{wang2025orientv2} unifica la estimación de orientación absoluta, la rotación relativa y la simetría rotacional a partir de imágenes 2D, y reporta el mejor desempeño \textit{zero-shot} en reconocimiento de simetría; su procedimiento, sin embargo, supone que la categoría del objeto determina el tipo de simetría.
% VERIFICAR (lectura pendiente): descripciones de CLIPSym y Orient Anything V2 tomadas de notas.

Estos trabajos indican que las representaciones de los modelos fundacionales contienen información aprovechable sobre la simetría. Sin embargo, en todos ellos esa información se explota con acceso a la malla durante la inferencia~\cite{aguirre2026} o mediante un componente entrenado específicamente para la tarea~\cite{li2025reflect3d,yang2025clipsym}. El presente trabajo difiere en ambos aspectos: no usa la malla para estimar y no entrena ningún componente. La información proviene de los puntos que un VLM marca en respuesta a una instrucción en lenguaje natural, y la geometría 3D se recupera únicamente por triangulación con cámaras conocidas.
```

**Si P1 = sí (se mantiene la comparación de 3.2.12), agrega este párrafo después del anterior:**

```latex
% PENDIENTE P1: incluir solo si se mantiene la comparación de la Sección 3.2.12.
Por su cercanía con el problema y por compartir el banco de pruebas, el método de Aguirre y Sipiran~\cite{aguirre2026} se emplea además como referencia de comparación (Sección~\ref{sec:comparacion-metodos}). Ambos enfoques extraen información en el dominio 2D a partir de renderizados multivista, pero difieren en dos aspectos que la comparación permite cuantificar: el origen de la información (un extractor de características visuales frente a un VLM guiado por lenguaje) y el acceso a la geometría (retroproyección de características sobre la malla frente a triangulación de rayos sin malla). La diferencia de desempeño entre ambos se interpreta, por tanto, como el costo de prescindir de la malla y de cualquier entrenamiento específico, y no como una comparación entre métodos en igualdad de condiciones.
% VERIFICAR: si el método de Aguirre y Sipiran entrena algún componente.
```

## 2.3 Geometría multivista

```latex
\section{Geometría multivista: de puntos 2D a elementos 3D}
\label{sec:lit-multivista}

Prescindir de la malla exige recuperar la información 3D exclusivamente a partir de observaciones 2D. Que un conjunto de vistas contiene información suficiente para recuperar aspectos de la estructura de un objeto se ha mostrado, por ejemplo, en la completación de formas a partir de mapas de profundidad multivista~\cite{hu2019render4completion}. Esta sección revisa las herramientas geométricas que permiten hacerlo cuando la observación disponible es un conjunto de puntos marcados en imágenes con cámaras conocidas (Secciones~\ref{sec:lit-multivista}.1 a~\ref{sec:lit-multivista}.5). La última subsección (\ref{sec:lit-retro-malla}) revisa, por separado, la retroproyección sobre la malla, que no forma parte del método y se usa solo como condición de referencia.
% Ajustar la forma de referenciar las subsecciones a tu estilo (por ejemplo, con un \label en cada una).
% Opcional: eliminar la frase de Render4Completion si se prefiere no citarlo.
```

### 2.3.1 Cámara, retroproyección y rayos

```latex
\subsection{Cámara, retroproyección y rayos}

Bajo el modelo de cámara de perspectiva~\cite{hartley2004}, cada vista queda definida por su rotación $R$, su traslación $T$ y sus parámetros intrínsecos, en este caso el campo de visión. Un punto de la imagen no determina un punto 3D, sino el rayo que parte del centro de la cámara $C$ y pasa por él: $r(\lambda) = C + \lambda\hat{u}$, con $\lambda > 0$. La profundidad $\lambda$ no puede conocerse desde una sola vista. Para recuperar información 3D es necesario combinar los rayos de varias vistas, ya sea intersectándolos con una superficie conocida (lo que requiere la malla) o triangulándolos entre sí, que es la opción que adopta este trabajo. La relación entre vistas se describe mediante la geometría epipolar~\cite{hartley2004}: un punto en una vista restringe su correspondiente en otra a una línea, la línea epipolar.
```

### 2.3.2 Triangulación de puntos y de líneas

```latex
\subsection{Triangulación de puntos y de líneas}

La triangulación de puntos estima la posición 3D que mejor explica un punto observado en dos o más vistas~\cite{hartleysturm1997,hartley2004}, y requiere conocer qué punto de cada imagen corresponde a qué punto de las demás. Esa correspondencia es precisamente lo que no puede suponerse cuando los puntos los marca un VLM en vistas distintas (Sección~\ref{sec:lit-elevacion}).

La triangulación de líneas ofrece una alternativa. Dos puntos de una imagen definen una recta 2D, y esa recta, junto con el centro de la cámara, define un plano en el espacio, el plano de interpretación, que contiene a la recta 3D observada. En la representación de Plücker de una recta 3D, el vector momento coincide con la normal de ese plano~\cite{bartoli2005}. Una recta 3D observada en varias vistas es entonces la intersección común de sus planos de interpretación, y puede estimarse por mínimos cuadrados a partir de las normales de esos planos~\cite{bartoli2005,wu2015lines,josephson2007}. Con dos vistas la recta queda determinada pero poco restringida, por lo que la estimación mejora al agregar vistas~\cite{hartley1997trifocal}. Estas técnicas sustentan sistemas de reconstrucción basados en segmentos de recta~\cite{hofer2016line3d,liu2023limap,mateus2021}.

Una propiedad de esta formulación es central para el presente trabajo: no exige correspondencias punto a punto entre vistas, sino solo que la recta 2D de cada vista sea la proyección de la misma recta 3D. En un objeto con simetría de revolución, la proyección del eje es el eje de simetría bilateral de la silueta desde cualquier punto de vista, de modo que basta con que los dos puntos marcados en cada vista caigan sobre esa proyección, sin importar a qué altura del eje correspondan. La estimación del eje en este trabajo (Sección~\ref{sec:metodo-eje}) es una aplicación directa de este principio.
```

### 2.3.3 Simetría especular y geometría multivista

```latex
\subsection{Simetría especular y geometría multivista}

La relación entre simetría y geometría multivista se ha estudiado principalmente para la simetría especular. François et al.~\cite{francois2003} muestran que una imagen de un objeto con simetría especular equivale a un par estéreo de dos vistas, lo que permite aplicar herramientas de geometría multivista a una sola imagen. Hong et al.~\cite{hong2004} explotan la simetría para recuperar estructura, pose y calibración desde una imagen, y Criminisi et al.~\cite{criminisi2000} establecen las bases de la metrología desde una sola vista. Gao y Yuille~\cite{gaoyuille2017} aprovechan las propiedades de simetría y de tipo Manhattan para estimar la estructura 3D de objetos desde una o varias imágenes, y Sinha et al.~\cite{sinha2012} detectan y reconstruyen objetos 3D con simetría especular. Olsson y Eriksson~\cite{olsson2011} abordan la triangulación de un plano a partir de observaciones en varias vistas.

Una dificultad propia de la simetría especular es que la silueta de un objeto solo se ve bilateralmente simétrica cuando la cámara lo observa desde una dirección contenida en el plano de simetría (vista ``de canto''); desde otras direcciones, la traza aparente del plano en la imagen deja de corresponder a la proyección del plano real. Además, una recta 3D determina solo una dirección dentro del plano, por lo que se necesitan dos rectas independientes para fijar su normal. En la literatura revisada, el efecto de observar el plano lejos de la condición ``de canto'' se describe de forma cualitativa, pero no se encontró un análisis cuantitativo de cómo condiciona la estimación del plano. La estimación del plano en este trabajo (Sección~\ref{sec:metodo-plano}) aborda esta dificultad seleccionando las vistas según su cercanía a la condición ``de canto''.
% VERIFICAR: que la afirmación sobre la ausencia de análisis cuantitativo se sostenga tras revisar [francois2003], [sinha2012] y [li2025reflect3d].
```

### 2.3.4 Condicionamiento y propagación del error

```latex
\subsection{Condicionamiento y propagación del error}

La precisión de una triangulación no depende solo del error de cada observación, sino de la configuración de las vistas. Fooladgar et al.~\cite{fooladgar2013} formalizan, para sistemas estéreo, cómo los errores de medición en la imagen se propagan al espacio 3D en función de la geometría de las cámaras. El principio se extiende al caso multivista: el error de un punto triangulado crece aproximadamente como $1/\sin\theta$, donde $\theta$ es el ángulo entre los rayos que lo observan~\cite{hartley2004}, y ciertas configuraciones de cámaras, por ejemplo vistas casi coplanares o concentradas en una región de la esfera, son degeneradas o están mal condicionadas~\cite{hartley2004,bartoli2005}.

En la triangulación de líneas, el error angular de cada rayo se traduce en un error en la normal del plano de interpretación, que es mayor cuando los dos puntos de la vista están próximos entre sí. La dirección de la recta se obtiene como el vector más ortogonal a todas las normales, y su sensibilidad al ruido está acotada por el número de condición de la matriz de normales, es decir, por el cociente entre sus valores singulares~\cite{golub2013}: si los planos de interpretación son casi paralelos, la intersección queda mal determinada. Análogamente, en un ajuste lineal la precisión de la pendiente depende de la dispersión de las observaciones a lo largo de la recta (su brazo de palanca)~\cite{draper1998}, también cuando ambas coordenadas tienen error~\cite{york1966}. Por último, se ha mostrado que el error de triangulación disminuye con el número de cámaras siempre que las observaciones adicionales sean consistentes~\cite{ghasemi2018}. Estos factores fundamentan los diagnósticos de condicionamiento de la Sección~\ref{sec:diag-condicionamiento}.
```

### 2.3.5 Estimación robusta y ponderada

```latex
\subsection{Estimación robusta y ponderada}

Cuando algunas observaciones son erróneas, el ajuste por mínimos cuadrados se degrada, y existen dos familias de remedios. La primera descarta observaciones: RANSAC~\cite{fischler1981} ajusta el modelo a subconjuntos aleatorios y conserva el que reúne más observaciones consistentes, y su variante secuencial extrae varios modelos de un mismo conjunto, encontrando uno, retirando las observaciones que lo apoyan y repitiendo el proceso. La segunda pondera las observaciones según su fiabilidad: los M-estimadores~\cite{huber1964} reducen la influencia de los residuos grandes, y los mínimos cuadrados reponderados iterativamente (IRLS)~\cite{holland1977} implementan esa idea alternando ajuste y reponderación.

En triangulación multivista, estas ideas se han aplicado mediante criterios basados en el error angular~\cite{recker2013,lee2019} y mediante pesos de confianza por vista, por ejemplo en la triangulación de la pose humana~\cite{iskakov2019}. Cuando se generan varios candidatos para un mismo elemento, la supresión de no-máximos~\cite{neubeck2006} conserva el mejor y elimina los que están demasiado cerca de él, lo que, aplicado a direcciones, equivale a fusionar normales separadas por menos de un umbral angular. Finalmente, la consistencia con la silueta ofrece un criterio geométrico de rechazo: todo punto 3D válido debe proyectarse dentro de la silueta del objeto en todas las vistas, idea en la que se basa el concepto de \textit{visual hull}~\cite{laurentini1994}.

Estos esquemas suponen que el error de las observaciones es aleatorio. Ponderar o descartar observaciones no corrige un sesgo sistemático, como el que se produce si todas las vistas señalan de forma consistente una estructura equivocada. Las variantes del estimador evaluadas en este trabajo (Tabla~\ref{tab:ablaciones}), la consolidación secuencial de planos y el filtro de puntos fuera del objeto se apoyan en estas familias de técnicas.
```

### 2.3.6 Retroproyección sobre la malla: condición de referencia

> Esta subsección es la única del capítulo que respalda el uso de la malla con los puntos del VLM. Su primer párrafo deja claro que no describe el método. Si prefieres separarla aún más, puedes convertirla en la sección 2.4, antes de los VLMs, con el mismo texto.

```latex
\subsection{Retroproyección sobre la malla: condición de referencia}
\label{sec:lit-retro-malla}

Lo que se revisa en esta subsección no forma parte del método propuesto, que estima la simetría sin acceder a la malla. Sirve de respaldo a un análisis complementario (Sección~\ref{sec:analisis-malla}): para separar el error atribuible a la triangulación del atribuible a la calidad de los puntos marcados, los mismos puntos obtenidos con los mejores prompts se retroproyectan directamente sobre la malla del objeto. Esta condición usa la geometría 3D, que el método no supone disponible, por lo que no constituye una alternativa de uso ni compite con el método sin malla: su función es diagnóstica.

\paragraph{Intersección de rayos con la malla.} Cuando la superficie del objeto es conocida, cada rayo de cámara (Sección~\ref{sec:lit-multivista}) puede intersectarse con ella, y el primer punto de contacto convierte una detección 2D en un punto 3D sin necesidad de combinar vistas. Es la estrategia de ZeroKey~\cite{gong2025zerokey}, que retroproyecta las predicciones de Molmo intersectando cada rayo con la malla y consolida las detecciones de distintas vistas mediante agrupamiento por densidad (HDBSCAN)~\cite{campello2013}. Esa misma idea, aplicada a características visuales en lugar de puntos, está en la base de los trabajos de Aguirre y Sipiran~\cite{aguirre2026}, que retroproyectan características de DINOv2 sobre la malla, y de Garosi et al.~\cite{garosi2025}, que las agregan sobre la representación 3D.
% PENDIENTE P5: mantener la cita a campello2013 y la frase sobre la consolidación solo si se usan en el análisis con malla o al describir ZeroKey.
% Si el análisis con malla consolida puntos redundantes antes del ajuste, añadir aquí una frase describiéndolo.

\paragraph{Estimación de la simetría a partir de puntos sobre la superficie.} Con un conjunto de puntos 3D $P = \{p_1, \dots, p_n\}$, el elemento de simetría puede estimarse a partir de sus direcciones principales. Se centra el conjunto en su centroide $\bar{p}$ y se calcula la descomposición en valores singulares de la matriz de puntos centrados, $\tilde{P} = U\Sigma V^\top$, cuyas columnas de $V$ son las direcciones de mayor a menor varianza, equivalentes a los vectores propios de la matriz de covarianza~\cite{golub2013}. Si los puntos se distribuyen a lo largo de un eje, la dirección de máxima varianza aproxima su dirección; si se distribuyen sobre un plano, la de mínima varianza aproxima su normal. En ambos casos, el centroide sirve como punto de referencia.

Esta propiedad depende de que los puntos se distribuyan efectivamente sobre el elemento de simetría, y la retroproyección sobre la superficie lo cumple solo en parte. En el caso planar, si la vista es ``de canto'', todo rayo que pasa por la traza del plano en la imagen está contenido en el plano, de modo que el punto de contacto también lo está; en las demás vistas no ocurre así. En el caso axial, los puntos de contacto caen sobre la superficie del objeto y no sobre el eje, salvo donde este sale del objeto (sus polos). Por ello, la dirección de máxima varianza aproxima el eje solo si los puntos se distribuyen alrededor de él a lo largo de su extensión.
% VERIFICAR: que esta descripción coincida con la implementación del análisis con malla (SVD/PCA directa o con consolidación previa).

\paragraph{Propagación del error de localización.} La retroproyección sobre la malla no está libre de error. Un error de localización en la imagen desplaza el punto de contacto según la profundidad y el ángulo de incidencia del rayo, por lo que pequeños errores 2D pueden traducirse en desviaciones 3D considerables~\cite{kim2024,fooladgar2013}. El efecto es mayor cerca del borde de la silueta, donde los rayos inciden de forma rasante o pueden no intersectar el objeto. Por lo tanto, esta condición elimina el error de la triangulación, pero no el de los puntos.

\paragraph{Alcance de la comparación.} La condición con malla no solo aporta la profundidad de cada punto: también cambia el estimador, que pasa de la intersección de planos de interpretación a las direcciones principales de una nube de puntos. La diferencia entre ambas condiciones mide, por tanto, el efecto conjunto de disponer de la geometría y de cambiar el estimador, y así se interpreta en este trabajo.
```

## 2.4 VLMs con capacidad de pointing

```latex
\section{Modelos de lenguaje-visión con capacidad de \textit{pointing}}
\label{sec:lit-pointing}
```

### 2.4.1 Grounding mediante puntos

```latex
\subsection{\textit{Grounding} mediante puntos}

Los modelos de lenguaje-visión con capacidad de \textit{pointing} devuelven coordenadas específicas de una imagen en respuesta a una instrucción en lenguaje natural. Deitke et al.~\cite{deitke2025molmo} presentan Molmo, un modelo multimodal de pesos y datos abiertos entrenado, entre otras tareas, para localizar puntos referidos mediante descripciones textuales, con datos propios de anotación de puntos (PixMo-Points) y sin requerir cajas delimitadoras ni máscaras de segmentación como supervisión. Los puntos se emiten como texto estructurado con coordenadas normalizadas a la imagen, lo que permite extraerlos mediante un parseo directo. En una evaluación dedicada de \textit{pointing} sobre 493 pares imagen-pregunta, Molmo-72B alcanza una precisión de 75,8\,\%, una exhaustividad de 75,4\,\% y un F1 de 75,2\,\%~\cite{deitke2025molmo}.
% VERIFICAR: escala de coordenadas (0–100 en Molmo; 0–1000 en Molmo2) y formato exacto de la etiqueta <point>.

Molmo2~\cite{clark2026molmo2} extiende esta capacidad a la comprensión de múltiples imágenes y de video. Hereda los datos de \textit{pointing} de su predecesor e incorpora un conjunto de 470 mil ejemplos de \textit{pointing} sobre múltiples imágenes, construido sin destilación de modelos propietarios. Esto permite enviar todas las vistas de un objeto en una única consulta, mientras que Molmo procesa cada imagen por separado. Cabe precisar que ``sin destilación'' se refiere a los datos: las variantes de 4B y 8B parámetros usan Qwen3 como modelo de lenguaje~\cite{clark2026molmo2}.

Las evaluaciones de terceros indican que los modelos entrenados con supervisión explícita de \textit{pointing} superan de forma consistente a los que no la tienen~\cite{pointarena2025}. En contraste, modelos de propósito general muestran limitaciones de localización marcadas en algunos dominios: Zhang y Wang~\cite{zhang2024gpt4v} reportan para GPT-4V, sobre datos de observación terrestre, una precisión (Pr@0.5) de 0,076 y una distancia media de 147 píxeles al centroide de referencia. Aunque se trata de un único dominio, el contraste es coherente con la ventaja que se atribuye a la supervisión a nivel de punto~\cite{deitke2025molmo,pointarena2025}.
% VERIFICAR: hallazgo y cifras de PointArena en su tabla principal.

Estas cifras sirven como referencia de la fiabilidad esperada del \textit{pointing}, pero deben leerse con dos cautelas. Primero, las evaluaciones se realizaron sobre imágenes naturales, por lo que el desempeño sobre renderizados sintéticos de objetos CAD no está garantizado en la misma magnitud. Segundo, los benchmarks de \textit{pointing} evalúan referentes con un correlato visual directo (objetos, partes, relaciones espaciales), mientras que un eje o plano de simetría es una propiedad relacional que no siempre es visible en la imagen.
```

**Párrafo de los modelos de contraste (Qwen3-VL y MolmoPoint-8B).** Va después del párrafo de Molmo2:

```latex
Qwen3-VL~\cite{bai2025qwen3vl} es un VLM de pesos abiertos con \textit{grounding} nativo por coordenadas, cuyo modelo de lenguaje pertenece a la misma familia que el de Molmo2-8B, pero que difiere en su codificador visual y en sus datos de entrenamiento. Por ello constituye un punto de contraste natural para distinguir las limitaciones propias de Molmo2 de las que podrían compartir los VLMs con \textit{pointing} en general, y para evaluar si la supervisión explícita de \textit{pointing} de Molmo2 aporta una ventaja en esta tarea. MolmoPoint-8B~\cite{molmopoint} [falta cita], por su parte, reporta un nuevo estado del arte en PointBench, con 70,7 [VERIFICAR: métrica y fuente].
% VERIFICAR: codificadores visuales de Qwen3-VL y Molmo2; comparación de pointing por un tercero (la cifra de Point-Bench del recuadro antiguo no tiene fuente).
% VERIFICAR: relación de MolmoPoint-8B con Molmo2 (¿variante de la misma familia?) y nombre exacto (la sección 4.11 lo llama "MolmoPointing").
```

### 2.4.2 Prompting visual y razonamiento

```latex
\subsection{\textit{Prompting} visual y razonamiento}

Dado que el modelo no se reentrena, la instrucción es la principal variable de diseño. El \textit{prompting} con cadena de razonamiento (\textit{chain-of-thought}, CoT) pide al modelo explicitar pasos intermedios antes de responder y mejora el desempeño de los modelos de lenguaje en tareas de razonamiento~\cite{wei2022cot}. Su efecto sobre el \textit{grounding} visual es menos claro: PointArena reporta que añadir razonamiento en lenguaje no mejora el \textit{pointing}~\cite{pointarena2025}, y un trabajo sobre razonamiento espacial en VLMs distingue el CoT libre, que no ayuda, de un CoT estructurado a la manera de un grafo de escena, que sí lo hace~\cite{structuredcot2025}.
% VERIFICAR: hallazgo de PointArena sobre CoT.

Otra línea modifica la imagen en lugar del texto. Set-of-Mark~\cite{yang2023som} superpone marcas numeradas sobre regiones de la imagen y pide al modelo referirse a ellas, lo que mejora su \textit{grounding}. MOKA~\cite{liu2024moka} superpone una grilla de coordenadas visible, con notación de tipo ajedrez, para que un VLM señale ubicaciones en tareas de manipulación robótica, y SketchVLM~\cite{sketchvlm2026} explora una estrategia relacionada de anclaje visual. Estas técnicas fundamentan las familias de prompts evaluadas en este trabajo: las instrucciones de pointing directo, las variantes con descripción previa del objeto (Flujos B y C), la grilla superpuesta y el CoT estructurado (Sección~\ref{sec:prompts}).
% VERIFICAR: descripción de SketchVLM (preprint).
```

### 2.4.3 Llevar a 3D las predicciones 2D de un VLM

```latex
\subsection{De predicciones 2D a estructura 3D}
\label{sec:lit-elevacion}

Varios trabajos llevan a 3D información extraída en 2D. Garosi et al.~\cite{garosi2025} segmentan partes de objetos 3D agregando sobre la representación 3D las características visuales extraídas en vistas 2D, y Lift3D~\cite{varma2024lift3d} propone elevar a 3D, sin entrenamiento específico, las predicciones de modelos de visión 2D arbitrarios. En el caso concreto de los VLMs con \textit{pointing}, ZeroKey~\cite{gong2025zerokey} detecta \textit{keypoints} 3D de forma \textit{zero-shot}: renderiza varias vistas de un objeto, consulta a Molmo con un prompt por \textit{keypoint} y por vista, retroproyecta cada predicción intersectando su rayo con la malla y consolida las detecciones de las distintas vistas mediante agrupamiento por densidad (HDBSCAN). El método supera ampliamente a líneas base basadas en CLIP-DINOiser y a la sustitución de Molmo por GPT-4o, lo que sus autores atribuyen al entrenamiento de Molmo con anotaciones a nivel de punto. Sus autores también documentan que los modelos multimodales fallan cuando se les pide mantener la identidad de un punto entre vistas. ZeroDex~\cite{zerodex2026} eleva a 3D \textit{keypoints} 2D predichos por un VLM mediante fusión multivista y una estrategia de selección y votación, sin requerir una malla densa, y constituye el precedente más cercano a este trabajo en cuanto a combinar \textit{pointing} y geometría multivista sin malla.
% VERIFICAR: descripciones de Lift3D y ZeroDex (preprint).

La dificultad de la correspondencia entre vistas está cuantificada: en el benchmark CVPC, Molmo-7B-D acierta solo el 3,02\,\% de las veces al predecir la coordenada continua del punto correspondiente en otra vista, frente al 60--66\,\% cuando debe elegir entre tres candidatos~\cite{cvpc2025}. Esto sugiere que un método que dependa de que el VLM marque el mismo punto físico en todas las vistas es frágil.

Frente a estos antecedentes, el presente trabajo se distingue en tres aspectos. A diferencia de ZeroKey y de Garosi et al., el método no intersecta los rayos con la malla: la información 3D proviene únicamente de la triangulación entre vistas con cámaras conocidas (Sección~\ref{sec:lit-multivista}). La intersección con la malla al estilo de ZeroKey se emplea solo como condición de referencia, para analizar el error de la triangulación (Sección~\ref{sec:lit-retro-malla}). A diferencia de los métodos de \textit{keypoints}, no exige correspondencias punto a punto, sino que cada vista aporte una recta 2D sobre la proyección del elemento de simetría. Y, a diferencia de todos ellos, el objetivo no es localizar puntos o partes, sino inferir una propiedad global del objeto, el eje o el plano de simetría, definida por la relación entre múltiples puntos.
```

## 2.5 Capacidades y limitaciones espaciales de los VLMs

```latex
\section{Capacidades y limitaciones espaciales de los VLMs}
\label{sec:lit-limitaciones}
```

### 2.5.1 ¿Representan estructura 3D los modelos fundacionales?

```latex
\subsection{¿Representan estructura 3D los modelos fundacionales?}

La hipótesis de este trabajo supone que un modelo entrenado para reconocer y localizar en imágenes 2D dispone de información sobre la estructura 3D del objeto. El Banani et al.~\cite{elbanani2024} formulan explícitamente esta pregunta para los modelos fundacionales de visión y obtienen una respuesta mixta: estas representaciones codifican parcialmente estructura tridimensional. Más específicamente, GIQ~\cite{giq2025} muestra mediante \textit{probing} que las representaciones contienen información sobre simetría 3D, pero que los modelos fallan en tareas que requieren una diferenciación geométrica fina, como la rotación mental. Por otra parte, las características de estos modelos han servido de base para detectores de simetría entrenados~\cite{li2025reflect3d}, y el lenguaje ha mostrado aportar información útil para detectar simetría~\cite{yang2025clipsym}.
% VERIFICAR (lectura pendiente): conclusiones de El Banani et al. y GIQ; estado de publicación de GIQ.

Dos matices acotan lo que esta evidencia permite afirmar. Primero, que la información esté presente en las representaciones no implica que sea accesible a través de la salida del modelo, en este caso las coordenadas de unos pocos puntos. Segundo, la evidencia no respalda que ``más semántica'' implique ``más geometría'': Reflect3D argumenta que DINOv2, menos orientado a la semántica, se alinea mejor con la geometría que CLIP~\cite{li2025reflect3d}. Por ello, este trabajo no supone que la comprensión semántica de un VLM se traduzca en precisión geométrica, sino que lo plantea como una pregunta empírica.
% VERIFICAR: si los modelos evaluados por El Banani et al. y GIQ incluyen el codificador visual de Molmo2; si no, declarar el supuesto de transferencia.
```

### 2.5.2 Fallas de razonamiento espacial

```latex
\subsection{Fallas de razonamiento espacial}

Existe evidencia consistente de que los VLMs tienen limitaciones espaciales. Tong et al.~\cite{tong2024eyes} documentan fallas sistemáticas de los modelos multimodales en patrones visuales simples, como la orientación y la posición, y Kamath et al.~\cite{kamath2023whatsup} muestran sus dificultades con relaciones espaciales básicas. SpatialVLM~\cite{chen2024spatialvlm} busca dotar a estos modelos de razonamiento espacial 3D y discute sus límites. Un benchmark reciente indica que los modelos multimodales identifican y cuentan elementos discretos, pero fallan en transformaciones espaciales como rotar, reflejar o plegar en 3D~\cite{vrb2025}, operaciones directamente ligadas a la simetría.
% VERIFICAR: estado de publicación del benchmark VRB.

Dos hallazgos son especialmente pertinentes. Zhang et al.~\cite{zhang2024leftright} observan que las características de los modelos preentrenados a menudo no captan la orientación y confunden los lados simétricos y las partes repetidas de un objeto. Bhat y Yamasaki~\cite{bhat2026} muestran que los VLMs espaciales pueden dar respuestas consistentes pero incorrectas, insensibles a la evidencia visual. En el contexto de este trabajo, ambos fenómenos implican que el VLM podría señalar de forma consistente, a través de las vistas, una estructura distinta del elemento de simetría, un error sistemático que ningún esquema de triangulación robusta corregiría (Sección~\ref{sec:lit-multivista}). Estos antecedentes justifican evaluar el método contra predictores que no miran la imagen (Sección~\ref{sec:lit-lineasbase}).
```

### 2.5.3 Robustez a resolución y exposición

```latex
\subsection{Robustez a la resolución y a la exposición}

La evaluación de la robustez de los modelos de visión ante perturbaciones controladas tiene como referencia el protocolo de corrupciones comunes de Hendrycks y Dietterich~\cite{hendrycks2019}, que aplica cada perturbación, incluido el cambio de brillo, en varios niveles de severidad. Para VLMs, VLM-RobustBench~\cite{saxena2026robustbench} evalúa 49 tipos de perturbación en modelos de cuatro familias, entre ellas Molmo, y concluye que estos modelos son relativamente robustos a las perturbaciones fotométricas y frágiles ante las espaciales. Usama et al.~\cite{usama2025} llegan a una conclusión compatible al analizar VLMs ante las corrupciones de ImageNet-C, y B-AVIBench~\cite{bavibench2024} reporta que el brillo está entre las corrupciones menos efectivas para degradar el desempeño de los LVLMs. A partir de esta evidencia, las condiciones de iluminación de este trabajo se interpretan como una prueba de robustez a la exposición: varían la intensidad de la luz, no su tipo ni su dirección, por lo que no evalúan si el sombreado ayuda a percibir la forma.
% VERIFICAR: venue final de VLM-RobustBench; estado de Usama et al. y autores de B-AVIBench.
```

**Frase de Kar et al. (P2). Agrégala al final del párrafo anterior solo si la luz se cambia en el renderizador:**

```latex
% PENDIENTE P2: incluir solo si la intensidad se modifica en la luz de la escena (renderizador), no sobre la imagen final.
Las condiciones se aplican en el renderizador y no sobre la imagen final, de modo que la iluminación resultante es coherente con la geometría del objeto, lo que Kar et al.~\cite{kar2022} argumentan que produce perturbaciones más realistas que los ajustes 2D.
```

```latex
En cuanto a la resolución, Hao et al.~\cite{hao2023} analizan el impacto de la calidad de imagen y de la distancia del objeto sobre el desempeño de detección de objetos, lo que sugiere que una menor resolución podría degradar la capacidad de identificar las estructuras relevantes. No se encontró literatura que estudie específicamente el efecto de la resolución de entrada sobre el \textit{pointing} de un VLM, cuyo preprocesamiento divide la imagen en recortes, por lo que el experimento de resolución de este trabajo (224, 448 y 1136 píxeles) tiene un carácter exploratorio.
```

## 2.6 Evaluación: datos, métricas, líneas base y sesgos

```latex
\section{Evaluación: datos, métricas, baselines y sesgos}
\label{sec:lit-evaluacion}
% Título igualado al que ya pusiste en la tesis. Si prefieres "líneas base", cámbialo también en el índice.
```

### 2.6.1 ShapeNet y su alineación canónica

```latex
\subsection{ShapeNet y su alineación canónica}

ShapeNet~\cite{chang2015shapenet} es uno de los repositorios de modelos 3D más utilizados en visión por computadora, con una colección extensa de mallas organizadas por categorías, que abarcan desde formas de revolución simples hasta objetos con estructura compuesta, como muebles y vehículos. Esta diversidad permite evaluar si un método generaliza a distintos grados de complejidad estructural. Las anotaciones de simetría (ejes de rotación y planos de reflexión) del subconjunto utilizado provienen de [falta cita] y permiten construir una referencia cuantificable para la evaluación; sus detalles se describen en la Metodología.

ShapeNet tiene, además, una propiedad que condiciona la evaluación: sus modelos están alineados de forma consistente dentro de cada categoría, con orientaciones ``arriba'' y ``frente'' comunes~\cite{chang2015shapenet}. En consecuencia, los elementos de simetría tienden a coincidir con los ejes del sistema de coordenadas, y un predictor que devuelva siempre la misma dirección puede obtener aciertos sin observar el objeto. Que los conjuntos de datos de visión contengan sesgos sistemáticos que los modelos explotan, inflando los resultados, es un fenómeno documentado~\cite{torralba2011}, y en este caso obliga a incluir predictores triviales en el diseño experimental (Sección~\ref{sec:lit-lineasbase}).
% VERIFICAR: sección de ShapeNet que describe la alineación canónica.
```

### 2.6.2 Métricas en detección de simetría

```latex
\subsection{Métricas en detección de simetría}
\label{sec:lit-metricas}

Las métricas de detección de simetría se dividen entre las que comparan la predicción con una simetría de referencia y las que miden su coherencia con la forma del objeto. Entre las primeras, el error angular entre direcciones es la medida básica, empleada por ejemplo por SymmetryNet junto con curvas de precisión-exhaustividad~\cite{shi2020symmetrynet}. Para resumir la distribución de errores en un solo número, se usa el área bajo la curva de la fracción de aciertos en función de un umbral angular, truncada en un ángulo máximo, práctica popularizada en la estimación de pose relativa con umbrales de 5\textdegree, 10\textdegree{} y 20\textdegree~\cite{sarlin2020superglue}; este trabajo emplea la misma idea con un umbral máximo de 45\textdegree.

Entre las segundas, el error de distancia de simetría (SDE) transforma puntos de la superficie con la simetría predicha y mide su distancia a la superficie original~\cite{gao2021prsnet}; la implementación utilizada sigue la de Aguirre y Sipiran~\cite{aguirre2026}. Cuando un objeto tiene varios planos de simetría y el método predice varios candidatos, se utiliza un F1 que empareja predicciones y referencias; Reflect3D adopta un emparejamiento por asignación óptima~\cite{li2025reflect3d}, que puede resolverse con el algoritmo húngaro~\cite{kuhn1955}. No se encontró en la literatura revisada una métrica F1 equivalente para ejes, ni una referencia para el error de traslación como medida de la posición del elemento de simetría [falta cita, o declarar como definición propia]. Las definiciones exactas se presentan en la Sección~\ref{sec:metricas}.
```

### 2.6.3 Atajos y líneas base sin información visual

```latex
\subsection{Atajos y líneas base sin información visual}
\label{sec:lit-lineasbase}

Un modelo puede resolver una tarea mediante una regla superficial en lugar de la estructura que se le pide, un fenómeno conocido como aprendizaje por atajos (\textit{shortcut learning})~\cite{geirhos2020}. En los modelos multimodales este riesgo está bien documentado. Goyal et al.~\cite{goyal2017} muestran que modelos de respuesta a preguntas visuales obtienen buen desempeño explotando los priors del lenguaje sin usar la imagen, y rebalancean el conjunto de datos para evitarlo; Agrawal et al.~\cite{agrawal2018} cambian la distribución de respuestas entre entrenamiento y prueba para exponer a los modelos que dependen de esos priors. Más recientemente, Chen et al.~\cite{chen2024mmstar} muestran que en varios benchmarks de VLMs muchas preguntas pueden responderse sin la imagen, y proponen evaluar explícitamente el desempeño alcanzable sin contenido visual. En la detección de simetría, el riesgo análogo es acertar por la categoría del objeto o por su pose canónica, y no por su geometría, un supuesto que algunos métodos incorporan de forma explícita~\cite{wang2025orientv2}.

Un segundo atajo es espacial. En la predicción de fijaciones visuales, la mirada humana presenta un sesgo hacia el centro de la escena, independiente de su contenido~\cite{tatler2007}, y un mapa que solo usa la distancia al centro predice las fijaciones mejor que varios modelos de saliencia~\cite{judd2009}. Un predictor ``del centro'' constituye, por tanto, una línea base exigente para cualquier método que devuelva posiciones en la imagen.

Siguiendo esta práctica, el método se compara con referencias que no usan la información visual: un predictor que elige una dirección uniformemente al azar sobre la esfera, cuyo desempeño esperado puede derivarse de forma analítica, y predictores triviales que devuelven siempre los mismos planos o ejes del sistema de coordenadas, explotando la alineación canónica de ShapeNet.
```

**Frase final de 2.6.3 (P4):**

```latex
% PENDIENTE P4: incluir solo si estas líneas base están implementadas.
Se incluyen además, como controles de los puntos 2D, un predictor que marca el centro de la imagen y puntos elegidos al azar dentro de la silueta del objeto y procesados con el mismo estimador.
```

### 2.6.4 Comparaciones estadísticas (opcional)

```latex
\subsection{Comparación estadística entre métodos}

Que un método obtenga un error medio menor que una línea base no basta para afirmar que es mejor: la diferencia debe distinguirse de la variabilidad entre objetos. Para comparar dos sistemas sobre los mismos ejemplos se recomiendan pruebas pareadas y no paramétricas~\cite{dror2018,demsar2006}, como la prueba de rangos con signo de Wilcoxon, junto con intervalos de confianza por remuestreo (\textit{bootstrap})~\cite{efron1993}. Cuando se comparan muchas variantes contra una misma referencia, conviene además controlar la tasa de descubrimientos falsos~\cite{benjamini1995}. Los procedimientos concretos se describen en la Metodología.
```

## 2.7 Síntesis y brecha

```latex
\section{Síntesis y brecha}
\label{sec:lit-brecha}

La Tabla~\ref{tab:lit-sintesis} resume los trabajos más cercanos según lo que requieren en el momento de la inferencia. Los métodos clásicos~\cite{kazhdan2002,podolak2006} y los aprendidos sobre geometría~\cite{gao2021prsnet} necesitan el objeto 3D. Los métodos desde imagen única~\cite{shi2020symmetrynet,li2025reflect3d} prescinden de la geometría, pero requieren entrenamiento supervisado y enfrentan la ambigüedad de una sola vista. Los métodos apoyados en modelos fundacionales~\cite{aguirre2026,li2025reflect3d,yang2025clipsym} muestran que estas representaciones contienen información sobre la simetría, pero la explotan con la malla o con un decodificador entrenado. Por su parte, los métodos que combinan \textit{pointing} de un VLM con geometría multivista~\cite{gong2025zerokey,zerodex2026} localizan puntos o partes, no propiedades globales, y en el caso de ZeroKey dependen de la malla. En este trabajo, la retroproyección sobre la malla al estilo de ZeroKey se usa únicamente como condición de referencia para analizar el error, y no como parte del método.

\begin{table}[htbp]
\centering
\small
\caption{Trabajos relacionados según lo que requieren en la inferencia.}
\label{tab:lit-sintesis}
\begin{tabular}{p{3.2cm}p{3.2cm}cp{2.6cm}p{2.4cm}}
\hline
Trabajo & Entrada en la inferencia & ¿Malla? & Entrenamiento para la tarea & Salida \\
\hline
Kazhdan et al.~\cite{kazhdan2002}; Podolak et al.~\cite{podolak2006} & Geometría 3D & Sí & No & Planos de simetría \\
PRS-Net~\cite{gao2021prsnet} & Representación volumétrica & Sí & Sí (no supervisado) & Planos % VERIFICAR: ejes
\\
SymmetryNet~\cite{shi2020symmetrynet} & RGB-D, una vista & No & Sí (supervisado) & Planos y ejes \\
Reflect3D~\cite{li2025reflect3d} & RGB, una vista & No & Sí (decodificador sobre DINOv2) & Planos \\
Aguirre y Sipiran~\cite{aguirre2026} & Renders multivista y malla & Sí & % VERIFICAR
--- & Planos y ejes % VERIFICAR
\\
ZeroKey~\cite{gong2025zerokey} & Renders multivista y malla & Sí & No (\textit{zero-shot}) & \textit{Keypoints} 3D \\
ZeroDex~\cite{zerodex2026} & Vistas múltiples & No (sin malla densa) & % VERIFICAR
--- & \textit{Keypoints} 3D \\
\textbf{Este trabajo} & Renders multivista y cámaras conocidas & No\textsuperscript{a} & No & Eje o planos de simetría \\
\hline
\multicolumn{5}{p{14cm}}{\footnotesize \textsuperscript{a} El método no usa la malla para estimar. La malla se usa en la evaluación y en un análisis de referencia sobre los mejores prompts (Sección~\ref{sec:lit-retro-malla}).} \\
\end{tabular}
\end{table}

De esta revisión se desprende la brecha que aborda el presente trabajo. Hasta donde llega la literatura revisada, no existen trabajos que evalúen si la capacidad de \textit{pointing} de un VLM de propósito general, sin entrenamiento adicional, permite inferir una propiedad estructural global como el eje o el plano de simetría de un objeto, definida por la relación entre múltiples puntos y no por un punto o región visible. Tampoco se ha estudiado la integración de esas detecciones 2D en un marco geométrico que recupere los parámetros 3D sin acceder a la malla ni requerir correspondencias punto a punto entre vistas. Finalmente, la evidencia sobre las limitaciones espaciales de los VLMs y sobre los sesgos de los conjuntos de datos exige que cualquier resultado se contraste con predictores que no observan la imagen. Este trabajo aborda empíricamente esas tres cuestiones.
```

> Nota: en la tabla, las celdas con `% VERIFICAR` comentan el resto de la línea en LaTeX; por eso el contenido sigue en la línea siguiente. Cuando verifiques, reemplaza `--- ` por el valor y elimina el comentario.

---

# PARTE C — Claves de cita

## C.1 Ya en tu bibliografía

Reemplaza la clave propuesta por la tuya.

| Clave propuesta | N.º actual | Referencia |
|---|---|---|
| `aguirre2026` | [1] | Aguirre y Sipiran, WACV 2026 |
| `bai2025qwen3vl` | [3] | Qwen3-VL technical report (VERIFICAR datos) |
| `bartoli2005` | [4] | Bartoli y Sturm, CVIU 2005 |
| `bhat2026` | [5] | Bhat y Yamasaki, arXiv:2606.02742 (hoy sin citar) |
| `chang2015shapenet` | [7] | ShapeNet |
| `clark2026molmo2` | [8] | Molmo2, CVPR 2026 |
| `deitke2025molmo` | [9] | Molmo y PixMo, CVPR 2025 |
| `draper1998` | [10] | Draper y Smith, *Applied Regression Analysis* |
| `fischler1981` | [11] | RANSAC |
| `fooladgar2013` | [12] | Fooladgar et al., IEEE Sensors J. |
| `francois2003` | [13] | François et al., IVC 2003 (hoy sin citar) |
| `gao2021prsnet` | [14] | PRS-Net, TVCG 2021 |
| `garosi2025` | [15] | Garosi et al., WACV 2025 |
| `golub2013` | [16] | Golub y Van Loan |
| `gong2025zerokey` | [17] | ZeroKey, ICCV 2025 |
| `gregor2014` | [19] | Gregor et al., GCH 2014 |
| `hao2023` | [20] | Hao et al., IROS 2023 |
| `hartley2004` | [22] | Hartley y Zisserman |
| `hu2019render4completion` | [23] | Render4Completion |
| `kazhdan2002` | [24] | Kazhdan et al., ECCV 2002 |
| `li2025reflect3d` | [26] | Reflect3D, CVPR 2025 |
| `liu2024moka` | [27] | MOKA, RSS 2024 |
| `mavridis2015` | [28] | Mavridis et al., CGF 2015 |
| `mitra2013` | [29] | Mitra et al., CGF 2013 |
| `papaioannou2017` | [30] | Papaioannou et al., JOCCH 2017 |
| `podolak2006` | [31] | Podolak et al., SIGGRAPH 2006 |
| `pointarena2025` | [32] | PointArena (completar autores) |
| `pollabauer2024` | [33] | Pöllabauer y Kühn, arXiv 2024 |
| `recker2013` | [34] | Recker et al., WACV 2013 |
| `shi2020symmetrynet` | [35] | SymmetryNet, TOG 2020 |
| `simari2006` | [36] | Simari et al., SGP 2006 |
| `sipiran2014` | [37] | Sipiran et al., CGF 2014 |
| `structuredcot2025` | [38] | arXiv:2507.13362 (completar título y autores) |
| `xu2022` | [39] | Xu et al., Visual Computer 2022 |
| `york1966` | [40] | York 1966 |
| `zhang2024gpt4v` | [42] | Zhang y Wang, CVPRW 2024 |
| `campello2013` | [6] | Campello et al., PAKDD 2013 (HDBSCAN). Solo en 2.3.6, según P5 |
| `kim2024` | [25] | Kim, Pollefeys y Barath, ECCV 2024. Solo en 2.3.6 (opcional) |

**Ya no se citan en el cap. 2:** [21] Harish y Prasad (eliminar). [2] AssemblyHands-X, [18] González y [41] Zhang 2020 se mantienen solo en metodología.

**Sin referencia en ningún archivo:** `molmopoint` (MolmoPoint-8B) [falta cita].

## C.2 Nuevas (datos tal como aparecen en tus `.md`; verificar antes de agregar)

| Clave propuesta | Referencia | Fuente |
|---|---|---|
| `hartleysturm1997` | Hartley, R. I., Sturm, P. (1997). Triangulation. CVIU, 68(2), 146–157 | `01` |
| `hartley1997trifocal` | Hartley, R. I. (1997). Lines and Points in Three Views and the Trifocal Tensor. IJCV, 22(2), 125–140 | `01` |
| `wu2015lines` | Wu, F., Zhang, M., Wang, G., Hu, Z. (2015). Algebraic Error Based Triangulation and Metric of Lines. PLOS ONE, 10(7) | `01` |
| `josephson2007` | Josephson, K., Kahl, F. Triangulation of Points, Lines and Conics. SCIA 2007 / JMIV (verificar venue) | `01` |
| `hofer2016line3d` | Hofer, M., Maurer, M., Bischof, H. (2016). Line3D++. CVIU, 157, 167–178 | `01` |
| `liu2023limap` | Liu, S., Yu, Y., Pautrat, R., Pollefeys, M., Larsson, V. (2023). 3D Line Mapping Revisited. CVPR 2023 | `01` |
| `mateus2021` | Mateus, A., Tahri, O., Aguiar, A. P., Lima, P. U., Miraldo, P. (2021). On Incremental Structure-from-Motion using Lines. arXiv:2105.11196 / IEEE T-RO | `01` |
| `criminisi2000` | Criminisi, A., Reid, I., Zisserman, A. (2000). Single View Metrology. IJCV, 40(2), 123–148 | `01` |
| `olsson2011` | Olsson, C., Eriksson, A. (2011). Triangulating a Plane. SCIA 2011, LNCS 6688, 13–23 | `01` |
| `gaoyuille2017` | Gao, Y., Yuille, A. L. (2017). Exploiting Symmetry and/or Manhattan Properties for 3D Object Structure Estimation from Single and Multiple Images. CVPR 2017 | `01` |
| `hong2004` | Hong, W., Yang, A. Y., Huang, K., Ma, Y. (2004). On Symmetry and Multiple-View Geometry. IJCV, 60(3), 241–265 | `01` |
| `sinha2012` | Sinha, S. N., Ramnath, K., Szeliski, R. (2012). Detecting and Reconstructing 3D Mirror Symmetric Objects. ECCV 2012 | `01` |
| `ghasemi2018` | Ghasemi, A. et al. Improving Triangulation by Enforcing Consistency. arXiv:1804.10448 | `02` |
| `huber1964` | Huber, P. J. (1964). Robust Estimation of a Location Parameter. Ann. Math. Stat., 35(1), 73–101 | `02` |
| `holland1977` | Holland, P. W., Welsch, R. E. (1977). Robust Regression Using Iteratively Reweighted Least-Squares. Comm. Stat. – Theory and Methods, 6(9) | `02` |
| `iskakov2019` | Iskakov, K., Burkov, E., Lempitsky, V., Malkov, Y. (2019). Learnable Triangulation of Human Pose. ICCV 2019 | `02` |
| `lee2019` | Lee, S. H., Civera, J. (2019). Closed-Form Optimal Two-View Triangulation Based on Angular Errors. ICCV 2019 | `02` |
| `neubeck2006` | Neubeck, A., Van Gool, L. (2006). Efficient Non-Maximum Suppression. ICPR 2006 | `03` |
| `laurentini1994` | Laurentini, A. (1994). The Visual Hull Concept for Silhouette-Based Image Understanding. IEEE TPAMI, 16(2), 150–162 | `04` |
| `varma2024lift3d` | Varma T., M. et al. (2024). Lift3D: Zero-Shot Lifting of Any 2D Vision Model to 3D. CVPR 2024 | `01` |
| `zerodex2026` | ZeroDex (2026). arXiv:2606.19340 (completar título y autores) | `01`, `07` |
| `cvpc2025` | Towards Cross-View Point Correspondence in VLMs. arXiv:2512.04686 (completar autores) | `07` |
| `sketchvlm2026` | SketchVLM. arXiv:2604.22875 (completar) | `05`, `07` |
| `wei2022cot` | Wei, J. et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. NeurIPS 2022 | `06`, `07` |
| `yang2023som` | Yang, J. et al. (2023). Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V. arXiv:2310.11441 | `06`, `07` |
| `elbanani2024` | El Banani et al. Probing the 3D Awareness of Visual Foundation Models. CVPR 2024 | `literatura-por-leer` |
| `giq2025` | GIQ. arXiv:2506.08194 (preprint) | `literatura-por-leer` |
| `yang2025clipsym` | Yang, Rahman, Yeh. CLIPSym. ICCV 2025 | `literatura-por-leer` |
| `wang2025orientv2` | Wang et al. Orient Anything V2. NeurIPS 2025 | `literatura-por-leer` |
| `tong2024eyes` | Tong, S. et al. (2024). Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs. CVPR 2024 | `05` |
| `kamath2023whatsup` | Kamath, A., Hessel, J., Chang, K.-W. (2023). What's "up" with vision-language models? EMNLP 2023 | `05` |
| `chen2024spatialvlm` | Chen, B. et al. (2024). SpatialVLM. CVPR 2024 | `06` |
| `vrb2025` | Benchmark VRB (preprint, completar datos) | `literatura-por-leer` |
| `zhang2024leftright` | Zhang et al. Telling Left from Right. CVPR 2024 | `literatura-por-leer` |
| `hendrycks2019` | Hendrycks, D., Dietterich, T. (2019). Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. ICLR 2019 | `lectura_iluminacion` |
| `saxena2026robustbench` | Saxena, R., Suglia, A., Minervini, P. (2026). VLM-RobustBench. arXiv:2603.06148 (verificar venue) | `lectura_iluminacion` |
| `usama2025` | Usama, M. et al. (2025). Analysing the Robustness of Vision-Language-Models to Common Corruptions. arXiv:2504.13690 | `lectura_iluminacion` |
| `bavibench2024` | B-AVIBench. arXiv:2403.09346 (completar autores) | `lectura_iluminacion` |
| `kar2022` | Kar, O. F., Yeo, T., Atanov, A., Zamir, A. (2022). 3D Common Corruptions and Data Augmentation. CVPR 2022 (solo si P2) | `lectura_iluminacion` |
| `torralba2011` | Torralba, A., Efros, A. A. (2011). Unbiased Look at Dataset Bias. CVPR 2011 | `lectura_baselines` |
| `sarlin2020superglue` | Sarlin, P.-E. et al. (2020). SuperGlue. CVPR 2020 | `09` |
| `kuhn1955` | Kuhn, H. W. (1955). The Hungarian Method for the Assignment Problem. NRLQ, 2, 83–97 | `09` |
| `geirhos2020` | Geirhos, R. et al. (2020). Shortcut Learning in Deep Neural Networks. Nature Machine Intelligence, 2, 665–673 | `05` |
| `goyal2017` | Goyal, Y. et al. (2017). Making the V in VQA Matter. CVPR 2017 | `lectura_baselines` |
| `agrawal2018` | Agrawal, A. et al. (2018). Don't Just Assume; Look and Answer. CVPR 2018 | `lectura_baselines` |
| `chen2024mmstar` | Chen, L. et al. (2024). Are We on the Right Way for Evaluating Large Vision-Language Models? NeurIPS 2024 | `lectura_baselines`, `baseline_aleatorio` |
| `tatler2007` | Tatler, B. W. (2007). The central fixation bias in scene viewing. Journal of Vision, 7(14):4 | `lectura_baselines`, `05` |
| `judd2009` | Judd, T. et al. (2009). Learning to Predict Where Humans Look. ICCV 2009 | `lectura_baselines` |
| `dror2018` | Dror, R. et al. (2018). The Hitchhiker's Guide to Testing Statistical Significance in NLP. ACL 2018 | `lectura_baselines`, `baseline_aleatorio` |
| `demsar2006` | Demšar, J. (2006). Statistical Comparisons of Classifiers over Multiple Data Sets. JMLR, 7, 1–30 | `08` |
| `efron1993` | Efron, B., Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall/CRC | `lectura_baselines`, `baseline_aleatorio` |
| `benjamini1995` | Benjamini, Y., Hochberg, Y. (1995). Controlling the False Discovery Rate. JRSS-B, 57(1), 289–300 | `08` |

**Solo para metodología (no en el cap. 2):** Muller 1959 y Marsaglia 1972 (muestreo uniforme en la esfera), Cawley y Talbot 2010, Barnett et al. 2005.

---

# PARTE D (opcional) — Reemplazos fuera del capítulo 2

Textos revisados sobre la versión nueva de la tesis. La idea en todos es la misma: el método principal no usa la malla, y la retroproyección sobre la malla es un análisis de referencia sobre los mejores prompts.

**1.2, frase final** ("Utilizando la malla exclusivamente para…"):

```latex
La malla no interviene en la estimación: se utiliza para obtener la simetría de referencia, para validar las estimaciones y, sobre las mejores configuraciones, como condición de referencia para analizar el error de la triangulación.
```

**PI1, última frase** ("La malla se utiliza exclusivamente…"):

```latex
La malla no se utiliza en la estimación.
```

**PI3** (mantener la idea, acotar el alcance):

```latex
\item[PI3.] ¿Qué parte del error se atribuye a la triangulación y qué parte a la calidad de los puntos marcados? Para analizarlo, los mismos puntos de las mejores configuraciones de prompt se retroproyectan directamente sobre la malla como condición de referencia. Esta condición no forma parte del método y se emplea solo con fines de análisis.
% Si se incluyen, mencionar también los puntos simulados perfectos de la Sección 4.10 como otra referencia para separar ambos factores.
```

**H2:**

```latex
H2. Al retroproyectar esos mismos puntos directamente sobre la malla, lo que elimina el error de triangulación pero no el de los puntos, la estimación resulta al menos tan precisa como en H1. La diferencia entre ambas condiciones, evaluada sobre las mejores configuraciones de prompt, cuantifica el costo de prescindir de la geometría tridimensional, considerando que la condición con malla también cambia el estimador.
```

**Objetivos específicos 3 y 4** (fusionar en uno) **y un objetivo nuevo para el análisis:**

```latex
\item Desarrollar un método que, a partir de los puntos 2D y de los parámetros de cámara conocidos, retroproyecte cada punto a un rayo y triangule el eje o el plano de simetría sin acceder a la malla del objeto.
\item Analizar, sobre las mejores configuraciones, qué parte del error se debe a la triangulación, comparando el método con una condición de referencia en que los mismos puntos se retroproyectan sobre la malla.
```

**Objetivo 6** (tres modelos):

```latex
\item Contrastar la mejor configuración obtenida con Molmo2 frente a Qwen3-VL y MolmoPoint-8B, bajo las mismas vistas, prompts, post-procesamiento y métricas, para determinar si los resultados y sesgos observados son específicos de Molmo2 o compartidos por otros VLMs con capacidad de \textit{pointing}.
```

**1.1, MolmoPoint-8B** (completar la frase cortada y ajustar el párrafo anterior):

```latex
Asimismo, se evalúa un tercer modelo, MolmoPoint-8B~\cite{molmopoint}, que reporta un nuevo estado del arte en PointBench, con 70,7 [VERIFICAR: métrica y fuente]. Las conclusiones sobre el comportamiento de los VLMs se restringen, por tanto, a estos tres modelos.
% Borrar "Las conclusiones ... se restringen, por tanto, a estos dos modelos." del párrafo de Qwen3-VL.
```

**PI4, segunda pregunta:**

```latex
¿Se repiten esos sesgos en otros VLMs con capacidad de \textit{pointing} (Qwen3-VL y MolmoPoint-8B)?
```

**1.5 (iv):**

```latex
(iv) la malla del objeto no interviene en la estimación: se utiliza para obtener la simetría de referencia y calcular las métricas de evaluación, como condición de referencia en un análisis del error de la triangulación sobre las mejores configuraciones y en una variante de control explícitamente declarada (EXP-F).
```

**1.5, vistas:**

```latex
el número de vistas utilizadas (1, 6, 14, 26 y 42; con $n_v = 1$ solo en la condición con malla, ya que la triangulación requiere al menos dos vistas)
% VERIFICAR: qué valores de nv se usaron realmente en cada condición.
```

**1.6, etapas 3 y 4, más una etapa de análisis:**

```latex
\item En la tercera etapa, cada punto se retroproyecta a un rayo de cámara mediante los parámetros conocidos de cada vista (rotación, traslación y campo de visión), sin intersectarlo con la malla.
\item En la cuarta etapa, los rayos de varias vistas se triangulan para estimar el eje, como intersección de los planos de interpretación de cada vista, o el plano, a partir de rectas trianguladas entre pares de vistas y su consolidación en hasta tres planos.
\item Como análisis complementario sobre las mejores configuraciones, los mismos puntos se retroproyectan sobre la malla mediante \textit{ray casting} y la simetría se estima con SVD sobre los puntos de contacto, lo que permite separar el error de la triangulación del de los puntos marcados.
```

**Otras correcciones puntuales:**

- 3.2.11: "método sin malla (H2)" → "método sin malla (H1)".
- 3.2.12: el título "métodos tradicionales basados en geometría" → "Comparación con un método basado en retroproyección sobre la malla" (si se mantiene, P1).
- Metodología: agregar una sección para el análisis con malla, con `\label{sec:analisis-malla}`.
- Discusión, "Preguntas": eliminar la pregunta sobre clustering espacial (o reubicarla en el análisis con malla, si P5 = sí); corregir [9] → [1] y [10, 11, 12] → las referencias de 2.4.2 y 2.5.
- 1.1: "Molmo-8B pertenece a la familia Qwen3" → "Molmo2-8B usa Qwen3 como modelo de lenguaje".
- 4.11: unificar "MolmoPointing" → "MolmoPoint-8B".

---

# PARTE E — Lista de control

- [ ] Responder P1, P2, P4 y P5 y activar o borrar los bloques marcados `% PENDIENTE`.
- [ ] Pegar los borradores en 2.1–2.7 y eliminar las secciones antiguas 2.8–2.13 completas.
- [ ] Ajustar los `\label` de metodología a los que usan los borradores: `sec:metodo-eje`, `sec:metodo-plano`, `sec:prompts`, `sec:metricas`, `sec:diag-condicionamiento`, `sec:comparacion-metodos`, `sec:analisis-malla`, `tab:ablaciones`.
- [ ] Escribir en metodología la sección del análisis con malla (ray casting, estimador, prompts y nv usados) y revisar que 2.3.6 coincida con ella.
- [ ] Conseguir la referencia de MolmoPoint-8B y la fuente de la cifra de PointBench.
- [ ] Reemplazar las claves de cita por las tuyas (Parte C.1) y agregar las nuevas (Parte C.2).
- [ ] Leer al menos resultados y conclusiones de El Banani et al., GIQ, CLIPSym, Orient Anything V2, Telling Left from Right y VRB, y ajustar 2.2.3, 2.5.1 y 2.5.2.
- [ ] Verificar los preprints: GIQ, VRB, ZeroDex, Bhat y Yamasaki, SketchVLM, CVPC, Usama et al., B-AVIBench, VLM-RobustBench.
- [ ] Completar las entradas incompletas: [2], [3], [32], [38], [41].
- [ ] Encontrar la fuente de las anotaciones de simetría (2.6.1).
- [ ] Decidir si el error de traslación y el F1 ausente para eje se declaran como definiciones u observaciones propias (2.6.2).
- [ ] Verificar las celdas `% VERIFICAR` de la Tabla de síntesis (2.7).
- [ ] Eliminar del texto actual lo listado en A.2 y de la bibliografía [6], [21] y [25].
- [ ] Mover el recuadro de criterios de selección del VLM a 3.2.3.
- [ ] Aplicar las correcciones de la Parte D en la introducción, 3.2.11, 3.2.12 y la discusión.
- [ ] Buscar literatura para los vacíos 1, 4, 5 y 8 (A.5).
