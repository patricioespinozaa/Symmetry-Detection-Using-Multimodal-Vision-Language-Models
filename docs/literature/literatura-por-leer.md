# Literatura por leer — VLM, conciencia 3D y simetría

Lista de trabajos citados en el feedback sobre `intro.tex:19`, organizados por el eslabón argumental que respaldan.

## Eslabón 1 → 2: "reconoce y localiza en 2D ⇒ representa estructura 3D"

- [ ] **El Banani et al.** — *"Probing the 3D Awareness of Visual Foundation Models"* (CVPR 2024)
  Cita ancla: formula la pregunta de si modelos que clasifican/localizan en 2D también representan estructura 3D. Conclusión mixta — legitima la pregunta, no confirma la respuesta.

## Eslabón 2 → 3: "estructura 3D ⇒ información sobre simetría"

- [ ] **GIQ** (arXiv 2506.08194)
  Probing lineal/no lineal muestra que hay info de simetría 3D en las representaciones, pero los modelos fallan en tareas que requieren diferenciación geométrica fina (ej. rotación mental). Info presente ≠ accesible. Es preprint — revisar si ya se publicó.

- [ ] **Li et al., Reflect3D** — *"Symmetry Strikes Back"* (CVPR 2025) — [ya citado como 25]
  Features congeladas de DINOv2 bastan como base para un detector de simetría 3D zero-shot (con decoder entrenado aparte). **Evidencia en contra**: argumentan que DINOv2 (menos semántico) se alinea mejor con geometría que CLIP — contradice "más semántica ⇒ más geometría". Abordar explícitamente.

- [ ] **Yang, Rahman y Yeh, CLIPSym** (ICCV 2025)
  CLIP preentrenado + prompts agrupados por objeto ayuda a detectar simetría usando pistas del lenguaje. Mejor evidencia de que el lenguaje aporta info útil sobre simetría. Limitaciones: 2D, decoder entrenado.

- [ ] **Wang et al., "Orient Anything V2"** (NeurIPS 2025)
  Unifica orientación absoluta, rotación relativa y simetría rotacional desde imágenes 2D; SOTA zero-shot en reconocimiento de simetría. Su pipeline asume que la categoría determina el tipo de simetría (usa VLM para anotar assets 3D en la versión anterior).
  ⚠️ Mismo argumento explica el hallazgo del capítulo 6 (ShapeNet en pose canónica): riesgo de que el modelo "acierte" por categoría, no por geometría → la línea base trivial debe estar en el diseño.

## Evidencia sobre partes simétricas

- [ ] **Zhang et al., "Telling Left from Right"** (CVPR 2024)
  Features de modelos preentrenados a menudo no captan geometría/orientación y confunden lados simétricos y partes repetidas.
  *Lectura propia (conjetura, no de los autores):* si el modelo confunde izquierda/derecha, podría ser porque las representa como equivalentes bajo reflexión — señal potencial para un prompt bilateral ("marca pares de partes simétricas"). Presentar como hipótesis propia si se usa.

## Evidencia en contra (citar para anticipar al jurado)

- [ ] **Benchmark reciente (VRB)** — arXiv, preprint
  Los modelos multimodales "ven" y cuentan elementos discretos, pero fallan en transformaciones espaciales (rotar, reflejar, plegar en 3D). Revisar si ya está publicado.

## Pendientes antes de incluir las citas

- [ ] Confirmar estado de publicación de **GIQ** y del **benchmark VRB** (¿siguen como preprint?).
- [ ] Leer al menos las secciones de resultados de **El Banani et al.** y **GIQ**: identificar qué modelos evaluaron y si Molmo2 (o su encoder de visión) pertenece a esa familia. Si no, declarar el supuesto de transferencia.
- [ ] Para el capítulo 2 (Revisión de Literatura), ampliar búsqueda con:
  - "spatial reasoning VLM benchmark" (2024–2026)
  - "semantic keypoints symmetry"

## Párrafo de reescritura propuesto para `intro.tex:19`

> Existe evidencia de que las representaciones de los modelos fundacionales de visión, entrenados para reconocer y localizar objetos en 2D, codifican parcialmente estructura tridimensional [El Banani 2024], incluyendo elementos de simetría 3D recuperables mediante *probing* [GIQ]. Asimismo, las features de estos modelos han servido como base para detectores de simetría entrenados [25], y la información semántica aportada por el lenguaje ha mostrado ser útil para la detección de simetría en imágenes [CLIPSym]. Sin embargo, que esta información esté presente en las representaciones no implica que sea accesible a través de la salida del modelo: los VLM muestran limitaciones en transformaciones espaciales como la reflexión [VRB], y sus features tienden a confundir los lados simétricos de un objeto [Telling Left from Right]. Queda abierto, por tanto, si la capacidad de *pointing* de un VLM, sin entrenamiento adicional, permite acceder a información sobre una propiedad relacional como la simetría. Este trabajo aborda empíricamente esa pregunta.
