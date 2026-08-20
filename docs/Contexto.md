# Contexto del proyecto de tesis

> Documento de contexto general para usar como input en otros chats de Claude que
> no tienen el historial de esta conversación. Complementa a
> `docs/metricas_evaluacion.md` (que cubre en detalle las métricas de evaluación)
> — este documento cubre el proyecto completo: tema, objetivos, estado del
> pipeline, diseño experimental y estado de redacción de la tesis.

---

## 1. Tema y motivación

Tesis de magíster (Universidad de Chile, DCC/MDS — `umemoria` class) sobre
**detección de simetría 3D (axial y planar) usando un VLM (Vision-Language
Model) con capacidad de *pointing*, específicamente Molmo2**, a partir de
renders 2D multi-vista de un objeto, **sin acceso directo a su geometría 3D**
durante la inferencia semántica.

**Motivación**: la detección de simetría 3D es útil para reconstrucción de
patrimonio cultural, segmentación de partes y reconocimiento de objetos, pero
los métodos clásicos requieren geometría 3D explícita (mallas/nubes de
puntos/vóxeles), no disponible cuando solo se tienen fotografías/renders.
Trabajos previos en 2D-multivista capturan información geométrica pero sin
razonamiento semántico sobre qué puntos importan para la simetría; los VLMs con
*pointing* (Molmo, Molmo2) sí pueden localizar puntos semánticamente relevantes
a partir de instrucciones en lenguaje natural.

**Brecha identificada** (revisión de literatura, `cap2.tex`): *"Dentro de la
literatura actual revisada, no existen trabajos previos que exploren el uso de
VLMs para inferir propiedades globales de simetría tridimensional ni su
integración en un marco geométrico consistente que traduzca detecciones 2D en
parámetros 3D validados."*

**Hipótesis**: *"un VLM multimodal con capacidad de pointing, en específico
Molmo2, puede identificar puntos estructuralmente relevantes en renderizados 2D
de un objeto y, mediante un esquema de integración multivista, permitir la
inferencia de su eje o plano de simetría dominante, sin acceso directo a su
geometría tridimensional."*

---

## 2. Objetivos (`intro.tex`)

**Objetivo general**: evaluar dicha capacidad — si Molmo2 puede inferir el
eje/plano de simetría dominante de un objeto a partir de renders 2D, sin
entrenamiento ni acceso a geometría 3D.

**Objetivos específicos**:
1. Diseñar un procedimiento de renderizado multivista que represente
   visualmente las geometrías 3D en imágenes 2D.
2. Aplicar Molmo2 para identificar puntos de simetría relevantes en cada vista
   mediante prompts diseñados para tal fin.
3. Desarrollar un método de integración que consolide las detecciones en un
   sistema de referencia común, obteniendo un conjunto de puntos 3D consistente
   sobre la malla original.
4. Diseñar un algoritmo que, a partir de dicho conjunto consolidado, infiera el
   eje o plano de simetría correspondiente.
5. Validar el desempeño del sistema completo frente a objetos con simetrías
   conocidas mediante métricas cuantitativas de precisión geométrica.

---

## 3. Alcance

- Subconjunto curado de **1700 objetos de ShapeNet** — 850 con simetría axial,
  850 con simetría planar. Normalizados en cubo unitario centrado en el origen.
- De los objetos planares: 692 tienen 1 plano de simetría GT, 152 tienen 2, 6
  tienen 3 (esto es relevante para la sección de métricas — ver
  `docs/metricas_evaluacion.md` §2.1 y §5.2 sobre manejo de múltiples GT).
- Variables experimentales contempladas en el diseño: número de vistas (`1, 6,
  14, 26, 42, 62, 86, 114` — en la práctica el sweep ejecutado usa `{1,6,14,26}`
  fijo, ver §6 más abajo), resolución (`224/448/1136` px — en la práctica
  siempre `224`), iluminación (`flat/brighter/darker` — en la práctica siempre
  `flat`).
- Limitaciones declaradas: solo modalidad de imagen estática de Molmo2 (no
  video); se asume simetría exacta según el ground truth de ShapeNet (no se
  contemplan simetrías aproximadas).

---

## 4. Revisión de literatura (`cap2.tex`) — bloques temáticos

1. **Detección de simetría en representaciones 3D clásica** — Kazhdan et al.,
   Xu et al., Sipiran et al. Incluye la base matemática de **ajuste por SVD**
   que reutiliza el pipeline: axial → primera componente principal (máxima
   varianza) ≈ dirección del eje; planar → última componente principal (mínima
   varianza) ≈ normal del plano; centroide como punto de referencia.
2. **Simetría como prior geométrico para reconstrucción/completación** —
   Gregor et al., Papaioannou et al., Mavridis et al.
3. **Inferencia de estructura 3D desde 2D** — Hu et al. (Render4Completion),
   Harish & Prasad (Photo2CAD), y **Aguirre & Sipiran (2026)** — el trabajo más
   cercano: retro-proyectan features visuales (DINOv2) multivista sobre una
   malla para detectar simetría 3D, pero **mantienen acceso a la geometría 3D**
   durante el proceso; esta tesis lo remueve y usa razonamiento semántico del
   VLM en su lugar. Ver `docs/metricas_evaluacion.md` §4.1 para el detalle
   técnico completo de ese repo (`EnhancedBackProjection`), ya investigado a
   fondo.
4. **VLMs con capacidad de pointing** — Molmo (Deitke et al.) y Molmo2 (Clark et
   al.), con benchmarks citados (Molmo-72B: 75.8% precisión de pointing;
   Molmo2-8B: F1 de pointing en video 38.4% vs. Gemini 3 Pro 20.0%). También
   Garosi et al. (segmentación de partes 3D vía agregación de features 2D) y
   **Gong et al. (ZeroKey)** — antecedente metodológico más cercano (VLM
   pointing + retro-proyección de cámara + clustering HDBSCAN para keypoints 3D
   zero-shot), pero limitado a puntos/keypoints individuales, no a propiedades
   relacionales globales como la simetría, y usa Molmo (no Molmo2, sin
   capacidad multivista).
5. **Precisión geométrica y error de localización** — Kim et al. (precisión
   sub-píxel), Hao et al. (efecto de calidad/distancia de imagen — relevante
   para la variable experimental de resolución), Zhang & Wang (limitaciones de
   localización de GPT-4V), Fooladgar et al. (propagación de error en visión
   estéreo, extendido conceptualmente al ray-casting multivista).
6. **Dataset ShapeNet** — elegido por sus anotaciones de ground truth de
   simetría y diversidad de categorías.

**Brecha explícita declarada al final del bloque de pointing**: *"ninguno de
los trabajos revisados en este bloque evalúa la capacidad de un VLM para
inferir propiedades globales del objeto como un eje o plano de simetría...
Esto constituye la brecha específica que la presente tesis busca abordar."*

---

## 5. Pipeline (metodología, 5 etapas)

1. **Renderizado multivista** (`ImagesGenerator/`) — vistas muestreadas via
   esfera de Fibonacci, a distintas resoluciones/iluminaciones.
2. **Pointing con Molmo2** (`MolmoPointing/`) — prompts diseñados por versión
   (`v00`–`v05`, `v00_1`–`v05_1`, ver `MolmoPointing/Experiments.md` y
   `anexoA.tex` §Prompts) piden puntos 2D relevantes para la simetría en cada
   vista. `--point-mode` (`independent`/`midpoint`/`all`) determina cómo se
   combinan los puntos obj_id 1/2 por imagen.
3. **Retro-proyección a 3D** (`Mapping/map_to_3d.py`) — ray-casting usando
   parámetros de cámara conocidos + la malla real del objeto, produce puntos 3D
   consolidados. Soporta `--patch-size {1,3,5}` (promediar sub-rayos por punto).
4. **Estimación de simetría** (`Mapping/estimate_symmetry.py`) — SVD (+ opcional
   RANSAC, + opcional clustering `none`/`greedy`/`hdbscan`) sobre la nube de
   puntos consolidada, produce eje/plano candidato. Calcula también el SDE
   propio como score de auto-consistencia (ver `docs/metricas_evaluacion.md`).
5. **Evaluación** (`Mapping/evaluate.py`) — error angular, error de traslación,
   AUC, precisión@umbral, SDE, contra el ground truth de ShapeNet y (cuando
   corresponda) contra el baseline de acceso-a-geometría-3D de Aguirre &
   Sipiran.

---

## 6. Diseño experimental — qué se ejecuta exactamente

24 prompts registrados (12 axiales `axis_v00`–`v05`/`v00_1`–`v05_1`, 12
planares `plane_v00`–`v05`/`v00_1`–`v05_1`), cada uno con Flow A (siempre) y,
solo para el prompt "ganador" por tipo de simetría, también Flow B y Flow C
(descripción + pointing, ver `MolmoPointing/README.md`) → **28 experimentos
base** (24 + 4). Cada uno se post-procesa con `Mapping/run_all_postprocessing.py`
en **7 variantes**: baseline (C1, patch 1), clustering greedy (C2), HDBSCAN
`ms∈{2,3,5}` (C3), patch-size `{3,5}` — **196 configuraciones**, cada una con
4 métodos de ajuste × 4 valores de `n_views` (`1,6,14,26`).

**Limitación estructural conocida**: con `--point-mode midpoint` (prompts
`v01,v02,v03,v01_1,v03_1` en axial; `v01,v03,v01_1,v03_1` en planar),
`n_views=1` no puede producir resultado — se necesitan ≥2 imágenes para tener
≥2 puntos medios y así poder ajustar SVD (`estimate_symmetry.py` líneas
~294-309). No es un bug, es matemáticamente inevitable con un solo par
bilateral.

**Estado del sweep** (último chequeo confirmado): los 28 experimentos base
tienen Molmo 100% completo (850/850 objetos). Post-procesamiento: 133/196
variantes con las 4 evaluaciones completas; el resto limitado por la
limitación de `n_views=1` arriba descrita (no por fallas reales). Últimas
corridas usaron `experiments_17_08_2026/all_experiments_comparison.csv` como
consolidado más reciente (3268 filas, 220 `experiment_id` únicos — más que 196
porque 3 prompts tienen el cruce completo clustering×patch, 15 variantes en vez
de 7, corridas antes de estandarizar el sweep a 7).

**Scripts clave del repo** (todos en la raíz o en `Mapping/`):
- `Mapping/run_all_postprocessing.py` — corre el sweep completo (post-Molmo),
  resumible, con limpieza automática de JSON corruptos.
- `Mapping/reference_metrics.py` — re-puntúa predicciones ya ajustadas con
  métricas de referencia externa (SDE/F1), soporta `--all` para descubrir y
  correr todos los experimentos. Ver `docs/metricas_evaluacion.md`.
- `ranking_postprocesamiento.ipynb` (raíz) — notebook de análisis: inventario,
  rankings por métrica, ablation por eje de variación (clustering/patch/flow),
  validación de duplicados/completitud.
- `analisis_tiempos_postprocesamiento.ipynb` (raíz) — parsea los logs del sweep
  para tiempos por etapa y proyecciones de cuánto falta.

**Análisis ya hecho, con hallazgos concretos**:
- El cruce completo clustering×patch (interacción) **no mejora** sobre el
  mejor de cada eje por separado — en 2 de 3 casos probados empeora. Confirma
  la recomendación original de no barrerlo para todos los experimentos.

---

## 7. Estado de redacción de la tesis (`Tesis_DCC_MDS/`)

Estructura (`main.tex`): `intro.tex` → `cap2.tex` (revisión de literatura) →
`cap3.tex` (contiene 3 capítulos en un archivo: Metodología línea 8,
Resultados línea 755, Discusiones línea 1370) → `conclu.tex` → `anexoA.tex`.

| Capítulo | Estado |
|---|---|
| Introducción | Completo (objetivos, hipótesis, alcance ya redactados) |
| Revisión de literatura | Completo (6 bloques temáticos, brecha declarada) |
| Metodología | Redactada (dentro de `cap3.tex`) |
| **Resultados** | **PRELIMINAR** — marcado explícitamente así en el propio `.tex` (`\textcolor{red}{ADVERTENCIA: RESULTADOS PRELIMINARES}`). Usa un snapshot (`results_agosto/experiments_22_07_2026`) que **no** incluye todo lo que ya está corrido en el servidor — debe regenerarse cuando el sweep esté más completo. Incluye tablas de comparación v0-vs-v1, y 6 tablas de ranking (top-15 por AUC/error/SDE, individual y conjunto) con metodología de selección documentada |
| Discusiones | Placeholder — contiene preguntas que el autor se dejó a sí mismo para responder, no discusión real todavía |
| Conclusiones | **Placeholder/borrador** — el archivo `conclu.tex` no tiene conclusiones reales, solo una plantilla genérica + una tabla de ejemplo (dummy) + 5 preguntas abiertas que el autor se dejó para resolver antes de escribir la versión final (ver §8) |
| Anexo | Completo — tabla comparativa de benchmarks Molmo/Molmo2 y texto íntegro de los prompts usados |

---

## 8. Preguntas abiertas dejadas en `conclu.tex` (para discusión/conclusiones)

1. Qué decir si más vistas *no* mejora la precisión (¿degrada el SVD por fusión
   de clusters de zonas distintas?).
2. Cómo discutir las limitaciones del SDE como criterio de aceptación para
   objetos de ShapeNet con simetría aproximada (no exacta).
3. Qué patrones cualitativos se esperan según la variante de prompt (bilateral
   pair, structural features, etc.).
4. Cómo interpretar los resultados respecto al argumento de "razonamiento
   semántico" de los Antecedentes — ¿Molmo2 realmente "entiende" la estructura,
   o hay un sesgo geométrico más simple (ej. simetría especular trivial
   respecto al centro de la imagen)?
5. Cómo enmarcar la comparación contra Aguirre & Sipiran (2026) — que sí tiene
   acceso a geometría 3D — de forma justa: ¿la brecha de desempeño es una
   "limitación inherente" o un "costo aceptable" por no requerir geometría 3D?

---

## 9. Documentos relacionados en `docs/`

- `docs/metricas_evaluacion.md` — detalle completo de todas las métricas de
  evaluación (pipeline propio y de referencia), con origen/citas de las
  métricas de referencia y los hallazgos metodológicos de comparación contra
  `EnhancedBackProjection`. Consultar ese documento para cualquier pregunta
  sobre AUC/SDE/F1/precision@θ — no se duplica aquí.
