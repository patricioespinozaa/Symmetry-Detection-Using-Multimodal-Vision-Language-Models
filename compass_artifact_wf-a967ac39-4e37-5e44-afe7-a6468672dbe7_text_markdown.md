# Revisión de literatura: Triangulación de líneas y planos, robustez, y VLM-pointing para detección de simetría 3D con Molmo2

La triangulación de una **línea 3D por intersección de "planos de interpretación" multivista** (el núcleo geométrico del método propuesto) es un problema clásico, plenamente formalizado en venues de primer nivel —el "moment vector" de la representación de Plücker ES la normal del plano de interpretación—, mientras que el problema análogo para **planos de simetría** (intersectar planos de interpretación de ejes de simetría 2D sin correspondencias ni malla densa) constituye una brecha genuina en la literatura formal, útil para enmarcar el aporte de la tesis.

## TL;DR
- **Tema 1 (triangulación de líneas 3D): EVIDENCIA SÓLIDA.** El método propuesto es una instancia directa de la triangulación clásica de líneas con coordenadas de Plücker, mínimos cuadrados óptimos y degeneraciones conocidas (Bartoli & Sturm 2005, CVIU; Wu et al. 2015, PLOS ONE; Hartley 1997, IJCV; Hartley & Zisserman 2004).
- **Tema 2 (orientación de un plano 3D desde trazas 2D multivista sin geometría densa): BRECHA GENUINA con bloques sólidos de apoyo.** Existen los ingredientes (mirror-symmetry ≡ two-view stereo; single-view metrology; plane triangulation por homografías), pero la síntesis específica —intersectar planos de interpretación de ejes de simetría sin correspondencias— no tiene tratamiento formal dedicado, y su mal condicionamiento sólo está documentado cualitativamente.
- **Temas 3, 4 y 5: EVIDENCIA SÓLIDA/PARCIAL.** La robustez de triangulación con detecciones semánticas ruidosas (pose humana multivista), el paradigma "VLM-pointing + triangulación multivista sin malla" (ZeroDex 2026 es el precedente más cercano) y la inconsistencia cross-view de los VLMs ("Consistent Yet Wrong" 2026; hallazgo de ZeroKey) están bien documentados y permiten fundamentar y diferenciar el método propuesto.

## Key Findings
- El método propuesto (ajuste de línea 2D → plano de interpretación por vista → intersección de ≥2 planos) es **exactamente** la maquinaria de la triangulación clásica de líneas expresada en coordenadas de Plücker: el "moment vector" se define como la normal del plano que pasa por el centro de cámara y la línea 2D. El método hereda garantías teóricas conocidas.
- Para líneas, dos vistas bastan geométricamente (dos planos de interpretación se intersectan en una línea), pero la configuración es mal condicionada cuando centros de cámara y línea son casi coplanares/colineales; tres vistas (tensor trifocal) imponen restricciones más fuertes.
- El análogo para planos es sustancialmente más difícil y aparentemente inexplorado como problema formal de "intersección de planos de interpretación", lo que es aprovechable como gap.
- La triangulación con detecciones semánticas ruidosas está muy estudiada en pose humana multivista (triangulación algebraica con confianzas, RANSAC), dando precedente directo para tolerar líneas ruidosas de Molmo2.
- La inconsistencia cross-view de los VLMs al pedir correspondencias explícitas está documentada: ZeroKey descubre que los MLLMs fallan en point-level reasoning, y "Consistent Yet Wrong" (2026) muestra que la consistencia entre vistas NO implica grounding geométrico. Esto justifica evitar correspondencias punto-a-punto.

---

## TEMA 1 — Triangulación / reconstrucción de una línea 3D a partir de proyecciones 2D en múltiples vistas calibradas

**Structure-from-motion using lines: Representation, triangulation, and bundle adjustment**
Adrien Bartoli, Peter Sturm. 2005. *Computer Vision and Image Understanding (CVIU)*, 100(3):416–441. DOI: 10.1016/j.cviu.2005.06.001 (PDF: perception.inrialpes.fr/Publications/2005/BS05).
Referencia canónica para el método exacto del proyecto: representa la línea 3D en coordenadas de Plücker, donde el "moment vector" es la normal del plano que pasa por el centro de cámara y la línea 2D — es decir, el "plano de interpretación" del método propuesto. Propone triangulación de máxima verosimilitud linealizando la restricción de Plücker + "Plücker correction", y una representación ortonormal minimal (4 parámetros) para bundle adjustment. Citable para justificar tanto la construcción como la optimización de los planos de interpretación.

**Algebraic Error Based Triangulation and Metric of Lines**
Fuchao Wu, Ming Zhang, Guanghui Wang, Zhanyi Hu. 2015. *PLOS ONE*, 10(7):e0132354. DOI: 10.1371/journal.pone.0132354.
Aborda la triangulación de líneas desde ≥2 vistas con matrices de proyección conocidas usando multiplicadores de Lagrange; aporta una fórmula de Plücker correction, un algoritmo lineal (LINa) y dos algoritmos óptimos (OPTa-I, OPTa-II) que minimizan el error algebraico. Provee el "método óptimo de mínimos cuadrados" solicitado (a) y métricas de error 3D sobre el espacio de líneas (ortogonal y quasi-Riemanniana), útiles para evaluar la línea de simetría recuperada.

**Lines and Points in Three Views and the Trifocal Tensor**
Richard I. Hartley. 1997. *International Journal of Computer Vision (IJCV)*, 22(2):125–140. DOI: 10.1023/A:1007936012022.
Establece el tensor trifocal como el objeto que relaciona líneas (y puntos) en tres vistas; muestra que se necesitan al menos tres vistas para reconstrucción proyectiva a partir de correspondencias de líneas y que 13 correspondencias de líneas bastan para el algoritmo lineal. Fundamental para el punto (c): a diferencia de los puntos, las líneas no imponen restricción en el caso de dos vistas del mismo modo, y tres vistas dan restricciones más fuertes.

**Triangulation of Points, Lines and Conics**
Klas Josephson, Fredrik Kahl. Scandinavian Conference on Image Analysis (SCIA 2007), LNCS 4522, pp. 162–172 (versión ampliada en *Journal of Mathematical Imaging and Vision*). *(Verificar año/venue exactos antes de citar: aparece tanto como SCIA 2007 como en JMIV.)*
Trata la triangulación de líneas desde tres proyecciones garantizando que la solución satisface la restricción del tensor trifocal, y **analiza explícitamente las configuraciones degeneradas**, incorporando vectores de flujo óptico como puntos espaciales para resolverlas. Directamente relevante al punto (c) sobre configuraciones mal condicionadas.

**Multiple View Geometry in Computer Vision (2ª ed.)**
Richard Hartley, Andrew Zisserman. 2004. Cambridge University Press.
El capítulo 15 (tensor trifocal) documenta la **degeneración del transfer de puntos/líneas cuando el punto 3D está sobre la baseline B12** y cuando las cámaras son colineales con el punto — la condición mal condicionada análoga a intersecciones de planos casi coplanares. Referencia de texto para las condiciones mínimas de vistas y degeneración.

**Bloque de apoyo (line-based SfM y detección de líneas para reconstrucción 3D):**
- **Line3D++: Efficient 3D Scene Abstraction Using Line Segments** — Manuel Hofer, Michael Maurer, Horst Bischof. 2016. *CVIU*, 157:167–178 (y GCPR 2015). Código abierto (github.com/manhofer/Line3Dpp). Pipeline de referencia para reconstrucción de líneas 3D multivista por clustering de segmentos 2D con restricciones epipolares débiles; baseline/estado del arte y triangulación robusta de líneas.
- **3D Line Mapping Revisited (LIMAP)** — Shaohui Liu, Yifan Yu, Rémi Pautrat, Marc Pollefeys, Viktor Larsson. 2023. *CVPR 2023*, pp. 21445–21455. Framework moderno de mapeo de líneas 3D con triangulación y optimización multivista; incluye tracks de líneas y de puntos de fuga (VP).
- **On Incremental Structure-from-Motion using Lines** — André Mateus, Omar Tahri, A. Pedro Aguiar, Pedro U. Lima, Pedro Miraldo. 2021 (arXiv:2105.11196; IEEE T-RO). Analiza parametrizaciones del momento de la línea y sus **singularidades**, relevante para el punto (c).

**EVIDENCIA SÓLIDA.** La triangulación de una línea 3D por intersección de planos de interpretación con representación de Plücker, mínimos cuadrados óptimos y condiciones de degeneración está exhaustivamente cubierta en CVIU, IJCV, PLOS ONE y CVPR. El método propuesto es una instancia directa de esta teoría; conviene citar Bartoli & Sturm (2005) como base metodológica y Hartley (1997)/Hartley & Zisserman (2004) para vistas mínimas y degeneración.

---

## TEMA 2 — Reconstrucción de la orientación de un PLANO 3D a partir de sus trazas/proyecciones 2D en múltiples vistas calibradas, sin geometría 3D densa

**Mirror symmetry ⇒ 2-view stereo geometry**
Alexandre R. J. François, Gérard G. Medioni, Roman Waupotitsch. 2003. *Image and Vision Computing*, 21(2):137–143. DOI: 10.1016/S0262-8856(02)00149-X (versión previa: ICPR 2002). *(Nota de corrección: el primer autor es Alexandre R. J. François.)*
Resultado formal fundacional: observar una escena con simetría de espejo desde una vista es geométricamente equivalente a observarla con dos cámaras simétricas respecto al plano de simetría 3D desconocido; se aplican todas las herramientas de estéreo de dos vistas (matriz fundamental/esencial, geometría epipolar, rectificación, disparidad). Sustento teórico de por qué un plano de simetría puede tratarse con maquinaria multivista, y por tanto de la validez conceptual del enfoque de planos de interpretación.

**Single View Metrology**
Antonio Criminisi, Ian Reid, Andrew Zisserman. 2000. *International Journal of Computer Vision (IJCV)*, 40(2):123–148. DOI: 10.1023/A:1026598000963.
Tratamiento fotogramétrico formal donde la orientación de un plano de referencia queda codificada por su **vanishing line** en la imagen, con propagación de error de primer orden; la tesis de Criminisi extiende esto a múltiples vistas. Base monocular que un método multivista de "normal del plano a partir de vanishing lines" extendería, y aporta el aparato de incertidumbre.

**Triangulating a Plane**
Carl Olsson, Anders Eriksson. 2011. *Image Analysis (SCIA 2011)*, LNCS 6688, pp. 13–23, Springer. DOI: 10.1007/978-3-642-21227-7_2.
Aborda directamente la recuperación de un plano de la escena minimizando errores de retroproyección como función de los parámetros del plano (en lugar de triangular puntos y luego ajustar un plano), mostrando que los residuos son quasiconvexos y globalmente optimizables. Contrapunto de "triangulación de plano" a la triangulación de puntos/líneas, aunque usa homografías inducidas por el plano (correspondencias de puntos), no planos de interpretación de líneas 2D.

**Exploiting Symmetry and/or Manhattan Properties for 3D Object Structure Estimation from Single and Multiple Images**
Yuan Gao, Alan L. Yuille. 2017. *CVPR 2017* (arXiv:1607.07129; versión de revista ampliada en *IJCV* 2019, DOI 10.1007/s11263-019-01195-z). *(Nota: dos autores; confirmado CVPR 2017.)*
Analiza formalmente la recuperación de estructura 3D y proyección de cámara usando simetría bilateral y restricciones de Manhattan desde una o múltiples imágenes, usando posiciones 2D de keypoints como entrada. Precedente de recuperación de plano de simetría multivista basada en simetría, aunque emplea SfM de puntos, no intersección pura de planos de interpretación.

**On Symmetry and Multiple-View Geometry: Structure, Pose, and Calibration from a Single Image**
Wei Hong, Allen Yang Yang, Kun Huang, Yi Ma. 2004. *IJCV*, 60(3):241–265. DOI: 10.1023/B:VISI.0000036837.76476.10.
Tratamiento formal (teoría de grupos) de la simetría como geometría multivista, que permite recuperar estructura, pose y calibración; refuerza la equivalencia simetría↔múltiples vistas.

**Bloque de apoyo (planos de simetría de reflexión y planos por vanishing lines):**
- **Detecting and Reconstructing 3D Mirror Symmetric Objects** — Sudipta N. Sinha, Krishnan Ramnath, Richard Szeliski. 2012. *ECCV 2012*, LNCS 7573. DOI: 10.1007/978-3-642-33709-3_42. Recupera planos de simetría, pero se apoya en correspondencias de features y SfM (usa geometría 3D).
- **VPLNet: Deep Single View Normal Estimation with Vanishing Points and Lines** — Rui Wang, David Geraghty, Kevin Matzen, Richard Szeliski, Jan-Michael Frahm. 2020. *CVPR 2020*. Estima normales de superficie combinando vanishing points/lines con deep learning; muestra la relación n = Kᵀℓ_horiz entre normal del plano y vanishing line.
- **Symmetry Strikes Back / Reflect3D** — Xiang Li et al. 2025. *CVPR 2025* (arXiv:2411.17763). Detector de simetría 3D zero-shot desde una imagen con un pipeline de "multi-view symmetry enhancement" que usa difusión multivista para resolver la ambigüedad de vista única; muestra que la vista única es ambigua para el plano de simetría (motiva el uso de múltiples vistas).

**Nota sobre mal condicionamiento:** No existe un paper que aísle y demuestre formalmente por qué estimar la orientación de un plano desde trazas 2D es más mal condicionado que estimar una línea/punto. La evidencia es cualitativa: François et al. (2003) y la literatura de shape-from-symmetry notan que los métodos basados en la vanishing point/line del eje de simetría son inestables salvo bajo perspectiva fuerte (foreshortening), con degeneración cuando la dirección de visión es casi paralela o casi ortogonal al plano de simetría (condición clásica de ~45° para recuperación veridical). Esto es aprovechable: el mal condicionamiento puede plantearse como contribución formal novedosa de la tesis.

**BRECHA GENUINA (con bloques SÓLIDOS de apoyo).** Los ingredientes están establecidos (mirror-symmetry ≡ two-view stereo; single-view metrology; plane triangulation por homografías; simetría↔multiview), pero la síntesis específica —ajustar un eje de simetría 2D por vista, formar planos de interpretación (centro de cámara + línea 2D) e intersectarlos/optimizarlos a través de vistas para recuperar el plano de simetría 3D **sin correspondencias y sin geometría densa**— no tiene tratamiento formal dedicado. Los métodos existentes de plano de simetría usan correspondencias de puntos, nubes/mallas densas, o priors aprendidos de vista única. El análogo formal más cercano (Bartoli–Sturm) resuelve el problema dual (planos→línea), no líneas→plano. Oportunidad clara de enmarcar el aporte como llenando el gap.

---

## TEMA 3 — Robustez de la triangulación con detecciones ruidosas / de detector semántico, en régimen de pocas vistas

**Learnable Triangulation of Human Pose**
Karim Iskakov, Egor Burkov, Victor Lempitsky, Yury Malkov. 2019. *ICCV 2019*, pp. 7718–7727 (arXiv:1905.05754).
Precedente directo de triangulación con keypoints semánticos aprendidos (ruidosos): propone triangulación algebraica diferenciable con **pesos de confianza** por vista estimados de la imagen, y una variante volumétrica; demuestra buen desempeño con pocas vistas (incluso 2 cámaras en CMU Panoptic). Citable para (b) triangulación ponderada que tolera detecciones ruidosas y (c) pocas vistas.

**Uncertainty, Baseline, and Noise Analysis for L1 Error-Based Multi-View Triangulation**
Mauricio Hess-Flores, Shawn Recker, Kenneth I. Joy. 2014. (eScholarship UC; basado en trabajo ISVC/WACV).
Análisis Monte Carlo de covarianza y elipsoides de confianza sobre un rango amplio de baselines y niveles de ruido; muestra que la triangulación multivista verdadera produce mucha menos incertidumbre de profundidad que la fusión de pares estéreo, y que la precisión depende del ángulo de paralaje. Relevante a (a) sensibilidad al ruido y (c) cuántas vistas se necesitan.

**Closed-Form Optimal Two-View Triangulation Based on Angular Errors**
Seong Hun Lee, Javier Civera. 2019. *ICCV 2019* (arXiv:1903.09115).
Deriva soluciones cerradas óptimas L1 y L∞ minimizando errores angulares de reproyección (invariantes a rotación); relevante para (a) análisis de sensibilidad angular, que es la métrica natural para la orientación de la línea/plano de simetría.

**Statistical Angular Error-Based Triangulation for Efficient and Accurate Multi-View Scene Reconstruction**
Shawn Recker, Mauricio Hess-Flores, Kenneth I. Joy. (WACV 2013 / IEEE).
Propone una función de costo basada en error angular robusta a outliers y muestreo estadístico por niveles de confianza; modela la **incertidumbre del punto triangulado como función de tres factores: número de cámaras, error medio de reproyección y ángulo máximo de paralaje**. Muy útil para justificar cuántas vistas y qué configuraciones se necesitan (c).

**Bloque de apoyo (RANSAC/robustez de líneas y triangulación robusta):**
- **DSAC / Differentiable RANSAC: Learning Robust Line Fitting** — Eric Brachmann et al. (base de DSAC, CVPR 2017; código vislearn/DSACLine). Muestra ajuste robusto de líneas 2D a puntos predichos por una CNN — exactamente el paso de "ajustar una línea 2D a los puntos que Molmo señaló" con robustez a outliers.
- **Robust multi-view L2 triangulation via optimal inlier selection and 3D structure refinement** — *Pattern Recognition*, 2014. Estimación del scale del ruido y selección de inliers vía SOCP.
- **Improving Triangulation by Enforcing Consistency** — Ghasemi et al. (EPFL, arXiv:1804.10448). Demuestra un límite inferior del error de reconstrucción inversamente cuadrático en el número de cámaras — argumento formal para "más vistas ayudan".
- **LOSTU: Fast, Scalable, and Uncertainty-Aware Triangulation** (arXiv:2311.11171) — triangulación multivista consciente de incertidumbre.
- **View Consistency Aware Holistic Triangulation for 3D Human Pose Estimation** — Wan, Chen, Zhao. 2023. *CVIU*. Refina keypoints 2D por consistencia de vistas antes de triangular; relevante a detecciones semánticas ruidosas.

**EVIDENCIA SÓLIDA.** La literatura de triangulación robusta con detecciones semánticas ruidosas (especialmente pose humana multivista) es extensa y en venues top (ICCV, CVPR, PR, CVIU). La transferencia es directa: Molmo2 es un detector semántico ruidoso análogo a un detector de keypoints de pose; triangulación ponderada con confianzas + RANSAC + análisis de incertidumbre por ángulo de paralaje son aplicables. Advertencia: casi toda esta literatura trata puntos, no líneas; la robustez para líneas es más escasa (Line3D++ usa RANSAC; Bartoli & Sturm usan IRLS), lo que deja espacio para contribución.

---

## TEMA 4 — VLMs / detectores de pointing-keypoints + triangulación multivista clásica sin malla

**ZeroDex: Zero-Shot Long-Horizon Dexterous Manipulation via Multi-View 3D-Grounded VLM Reasoning**
2026 (arXiv:2606.19340).
Es el precedente **más cercano y directo**: usa un VLM para producir keypoints 2D semánticos por vista y los eleva a 3D por fusión multivista que combina **triangulación de los groundings del VLM** con "reference-view ray voting" (triangulación RANSAC-style con score por número de vistas cuya reproyección cae bajo un umbral de píxeles). Demuestra exactamente el paradigma "detecciones de VLM → triangulación multivista clásica" sin malla densa. Diferencia clave con el método propuesto: ZeroDex triangula puntos, no líneas/planos de interpretación.

**ZeroKey: Point-Level Reasoning and Zero-Shot 3D Keypoint Detection from Large Language Models**
Bingchen Gong, Diego Gomez, Abdullah Hamdi, Abdelrahman Eldesokey, Ahmed Abdelreheem, Peter Wonka, Maks Ovsjanikov. 2025. *ICCV 2025* (arXiv:2412.06292).
El trabajo que el usuario ya conoce: usa Molmo para detectar keypoints 2D por vista sobre renders y los retroproyecta a la malla 3D + clustering. Confirma que Molmo es la elección adecuada para pointing y descubre empíricamente el fracaso de los MLLMs en point-level reasoning (TL;DR verbatim del sitio del proyecto: *"We discovered that ChatGPT does not work on detecting points and we introduce a new zero-shot method for detecting keypoints on 3D shapes"*; el paper añade que *"even advanced models like GPT-4o and Claude Sonnet 3.5 … still struggle with point-level understanding"*). Debe citarse como el precedente que el método propuesto supera al usar triangulación pura (intersección de planos) en lugar de ray-casting contra malla.

**Lift3D: Zero-Shot Lifting of Any 2D Vision Model to 3D**
Mukund Varma T. et al. 2024. *CVPR 2024* (arXiv:2403.18922).
Eleva salidas de modelos 2D (DINO, CLIP) a predicciones 3D consistentes vía correspondencias epipolares y volume rendering; precedente de "lifting 2D→3D" de predicciones semánticas con geometría multivista, aunque usa NeRF-style rendering, no triangulación algebraica.

**Bloque de apoyo:**
- **Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models** — Matt Deitke et al. (Allen AI). 2024 (arXiv:2409.17146). El modelo base; la capacidad de pointing proviene de PixMo-Points, que incluye (verbatim) *"2.3M question-point pairs from 428k images"*. Su modelo de 72B *"outperforms larger proprietary models including Claude 3.5 Sonnet, and Gemini 1.5 Pro and Flash, second only to GPT-4o."*
- **Molmo 2** (Ai2, 2025/2026) — el modelo específico del proyecto. Según el blog de Ai2: *"Molmo 2 (8B) outperforms the original Molmo (72B) on key image pointing and grounding benchmarks,"* entrenado con *"9.19M videos versus 72.5M"* (menos de un octavo del video de PerceptionLM de Meta). Su arquitectura interpola tokens visuales con timestamps e índices de imagen para razonar sobre espacio, tiempo y lenguaje.
- **MolmoPoint** (Ai2, arXiv:2603.28069) — arquitectura de pointing coarse-to-fine con tres tokens especiales `<PATCH>`, `<SUBPATCH>`, `<LOCATION>`; *"MolmoPoint-8B reaches a new state of the art on PointBench with 70.7% average accuracy, up from 68.7% for Molmo 2 (8B). On PixMo-Points, it reaches 89.2 F1, compared with 85.2 for Molmo 2 (8B)"*; libera MolmoPoint-GUISyn (*"36K high-resolution screenshots with over 2 million annotated points"*).
- **A Tale of Two Features: Stable Diffusion Complements DINO for Zero-Shot Semantic Correspondence** — Junyi Zhang et al. 2023. *NeurIPS 2023*. Correspondencia semántica zero-shot con features de fundación; contexto para lifting semántico.
- **PlückeRF: A Line-based 3D Representation for Few-view Reconstruction** — Sam Bahrami, Dylan Campbell. 2025. *CVPRW 2025* (arXiv:2506.03713). Representación 3D basada en líneas (Plücker) para few-view; conecta líneas de Plücker con reconstrucción few-view.

**EVIDENCIA PARCIAL (fuerte y creciente).** ZeroDex (2026) demuestra que el paradigma VLM-pointing + triangulación multivista clásica sin malla ya existe para puntos, y ZeroKey/Lift3D cubren el lifting semántico. Sin embargo, **ningún trabajo combina detecciones de líneas/ejes de un VLM con triangulación de planos de interpretación para recuperar líneas/planos 3D sin malla** — el método propuesto ocupa un nicho no cubierto (líneas y planos de simetría, no puntos; intersección de planos, no ray-casting ni voting). Argumento de novedad defendible.

---

## TEMA 5 — Por qué la correspondencia punto-a-punto explícita con VLMs tiende a fallar

**ZeroKey (Gong et al., ICCV 2025, arXiv:2412.06292)** — descubre empíricamente que ChatGPT/GPT-4V no funcionan para detectar puntos y que los MLLMs entrenados con alineación a modelos visión-lenguaje tradicionales heredan sus limitaciones en point-level reasoning; por eso ZeroKey agrega consistencia 3D vía clustering en lugar de pedir identidad persistente de landmarks. Justifica directamente evitar correspondencias explícitas.

**Consistent Yet Wrong: Evidence Insensitivity in Spatial Vision-Language Models**
S. Divakar Bhat, Toshihiko Yamasaki (University of Tokyo). 2026 (arXiv:2606.02742).
Introduce ViewDiag, un protocolo de evaluación multivista controlado que comprende *"176 object-pair tracks across 80 scenes with 2–10 views per track"* (Hypersim, ScanNet, KITTI360). Hallazgo central (verbatim): *"leading VLMs often produce view-invariant and consistent answers even when those answers are incorrect, indicating weak coupling between predictions and viewpoint-specific visual evidence."* Crítico: muestra que la consistencia cross-view NO implica grounding geométrico, lo que socava pedir correspondencias explícitas a un VLM.

**Forgotten Polygons: Multimodal Large Language Models are Shape-Blind** (arXiv:2502.15969) — muestra que el mecanismo de pointing de Molmo falla al mapear features geométricos (p. ej. asigna 10 lados en vez de 7 a un heptágono al puntear, aunque cuenta 7 correctamente sin puntear). Evidencia de que el pointing es ruidoso y el binding geométrico falla.

**Binding Visual Features Point by Point** (arXiv:2605.25427) — analiza el "binding problem" en VLMs: asociar un conjunto compartido de features con entidades distintas produce errores de binding; explica por qué pedir identidad persistente de landmarks entre vistas degenera.

**Bloque de apoyo:**
- **Seeing Across Views: Benchmarking Spatial Reasoning of VLMs in Robotic Scenes** (arXiv:2510.19400) — documenta "single-view bias and weak multi-view fusion": los VLMs alinean posiciones 2D en pantalla entre cámaras en lugar de razonar en un frame 3D compartido.
- **Bring My Cup! Personalizing VLA Models with Visual Attentive Prompting** (arXiv:2512.20014) — análisis de error con "multi-view inconsistency (Case 2)": el matching/tracking por vista se engancha a instancias distintas de la misma categoría entre cámaras, produciendo prompts contradictorios.
- **PointArena: Probing Multimodal Grounding Through Language-Guided Pointing** (arXiv:2505.09990) — benchmark de pointing con *"Point-Bench, a curated dataset containing approximately 1,000 pointing tasks across five reasoning categories"* y Point-Battle (*"over 4,500 anonymized votes"*). Molmo-72B lidera Point-Bench (63.8 promedio), con Gemini-2.5-Pro comparable (62.8); notablemente, *"adding language reasoning (e.g., Chain-of-Thought) does not improve visual grounding for pointing tasks."*

**EVIDENCIA SÓLIDA/PARCIAL.** Hay evidencia sólida y muy reciente (2025-2026) de que los VLMs son inconsistentes entre vistas y que la consistencia observada no refleja grounding geométrico ("Consistent Yet Wrong"), además del hallazgo directo de ZeroKey sobre point-level reasoning. La advertencia: pocos trabajos abordan explícitamente el fracaso de pedir **correspondencias de landmarks persistentes** (identidad punto-a-punto) entre vistas — la mayoría estudia grounding/pointing por vista o tracking. Esto respalda la decisión de diseño de evitar correspondencias y deja espacio para que la tesis lo documente empíricamente.

---

## Recommendations
1. **Fundamentar el Tema 1 en Bartoli & Sturm (2005) + Wu et al. (2015)** como la base metodológica exacta (Plücker, plano de interpretación = moment vector, LS óptimo), y usar Hartley (1997)/Hartley & Zisserman (2004) para las condiciones mínimas de vistas y degeneración. Umbral de cambio: si la evaluación muestra que 2 vistas bastan con bajo error angular, reportarlo como confirmación; si no, invocar el análisis de paralaje (Recker/Hess-Flores) para justificar ≥3 vistas.
2. **Enmarcar el Tema 2 como el gap central de la tesis**: citar François et al. (2003), Criminisi et al. (2000), Olsson & Eriksson (2011), Gao & Yuille (2017) y Hong et al. (2004) como bloques de apoyo, y declarar explícitamente que la intersección de planos de interpretación de ejes de simetría 2D sin correspondencias no tiene tratamiento formal. Contribución adicional de alto valor: un análisis de condicionamiento del plano vs. línea (degeneración a 0°/90° de la dirección de visión respecto al plano de simetría).
3. **Para el paper de workshop ECCV 2026, posicionar contra ZeroDex y ZeroKey**: destacar que el método usa líneas/planos (no puntos), intersección de planos (no ray-casting ni voting), y evita correspondencias explícitas. Reforzar la decisión anti-correspondencias con "Consistent Yet Wrong" (2026) y el hallazgo de ZeroKey.
4. **Adoptar triangulación ponderada + RANSAC para líneas** siguiendo Iskakov et al. (2019) (confianzas) y DSAC (ajuste robusto de línea 2D), y reportar sensibilidad al ruido en función del ángulo de paralaje (Recker et al.). Benchmark: si el error angular de la línea/plano de simetría supera ~5–10° con 2–3 vistas, aumentar vistas o aplicar M-estimators.
5. **Documentar empíricamente el Tema 5** en la tesis: incluir un experimento que muestre que pedir correspondencias explícitas de landmarks a Molmo2 entre vistas degrada frente al enfoque sin correspondencias — esto llenaría un hueco de la literatura y reforzaría la motivación.

## Caveats
- La mayoría de la literatura de robustez (Tema 3) trata **puntos**, no líneas; la transferencia a triangulación de líneas requiere adaptación argumentativa (los endpoints de líneas son inestables; conviene usar restricciones relajadas de endpoints à la Micusik & Wildenauer, *IJCV* 2017).
- ZeroDex (arXiv:2606.19340), Molmo 2, MolmoPoint (arXiv:2603.28069), "Consistent Yet Wrong" (arXiv:2606.02742) y "Binding Visual Features Point by Point" (arXiv:2605.25427) son **muy recientes**; verificar su estado de publicación/peer-review antes de citarlos como establecidos, y tratar sus números como preliminares.
- El mal condicionamiento del caso de planos (Tema 2) está documentado **cualitativamente**, no con una prueba formal; presentarlo como hipótesis a validar, no como hecho establecido — y potencialmente como contribución teórica propia.
- Verificar la cita exacta de Josephson & Kahl "Triangulation of Points, Lines and Conics" (aparece como SCIA 2007 y en JMIV) antes de citar, así como la numeración de páginas de Line3D++ en CVIU.
- Correcciones de citación confirmadas por verificación: (i) el primer autor de "Mirror symmetry ⇒ 2-view stereo geometry" es **Alexandre R. J. François** (no "Aaron Francois"); (ii) "Exploiting Symmetry and/or Manhattan Properties…" tiene **dos autores** (Gao & Yuille) y es **CVPR 2017** con extensión en IJCV 2019.