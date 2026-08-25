# Pipeline sin acceso a malla: eje/plano de simetría por triangulación multivista

> Documento de planificación para el siguiente capítulo de la tesis, motivado
> por la review de hjNG al paper de ECCV 2026 ("The central 'without mesh
> access' claim does not match the method... A 2D point defines a ray, not a
> 3D location, so this step cannot be applied to photographs alone without
> depth estimation or cross-view triangulation") y por la observación del
> profesor guía sobre la ambigüedad de proyectar una simetría 3D a 2D.
> Complementa a `docs/Contexto.md` y `docs/metricas_evaluacion.md` — este
> documento cubre específicamente el rediseño del pipeline para que la malla
> deje de estar en la ruta de estimación y pase a ser solo insumo de
> validación.

---

## 1. Motivación

El pipeline actual (paper ECCV 2026 / tesis) convierte los puntos 2D de
Molmo2 a 3D mediante *ray casting* contra la malla original, y ajusta el
eje/plano sobre esos puntos 3D. Esto significa que, aunque Molmo2 no ve la
malla al señalar puntos, **la predicción final sí depende de la malla** — la
retroproyección es un paso constitutivo del método de estimación, no solo de
evaluación. Esa es la objeción central y correcta de hjNG.

**Objetivo de este rediseño**: mover la malla fuera de la ruta de estimación
por completo. La predicción del eje/plano debe surgir únicamente de los
puntos 2D de Molmo2 y de los parámetros de cámara conocidos (ya generados al
renderizar) — la malla queda reservada exclusivamente para el paso final de
validación (SDE, error angular contra GT).

---

## 2. Por qué esto no es tan simple como "comparar en 2D"

Descartado explícitamente (observación del profesor guía): comparar la traza
2D predicha contra la proyección 2D del GT en una sola vista **no** valida
que la simetría 3D recuperada sea correcta. Bajo proyección, una única vista
2D es consistente con una familia entera de interpretaciones 3D simétricas
distintas (ambigüedad de retroproyección clásica; ver Referencias, Tema 6 de
`Prompted_Geometry.../doc_literatura_paper.md` y el hallazgo formal en
Referencias §8.2 de este documento).

La solución correcta es **triangulación multivista real**: combinar las
observaciones de ≥2 vistas calibradas para resolver la ambigüedad
geométricamente, no comparar proyecciones sueltas.

---

## 3. Pipeline propuesto

```
1. Renderizado multivista               [ya hecho — ImagesGenerator/]
2. Pointing con Molmo2 (varios prompts) [ya hecho — MolmoPointing/, reusar resultados existentes]
3.1 Inferencia de eje (axial), sin malla     [NUEVO]
3.2 Inferencia de plano (planar), sin malla  [NUEVO — más difícil]
4. Consolidación de simetrías detectadas (incl. 2-3 planos) [NUEVO]
5. Mapeo a la malla 3D SOLO para validar métricas (SDE, error angular) [reusa Mapping/evaluate.py]
```

### 3.1 — Simetría axial (eje), sin malla

**Qué hace**: por cada vista, ajusta una línea 2D a los puntos que Molmo2
señaló en esa vista. Con la cámara calibrada de cada vista (ya conocida:
$R$, $T$, FoV, guardados en `metadata_all.json`), construye el **plano de
interpretación** de esa vista (el plano en 3D que contiene el centro de
cámara y la línea 2D observada). Con ≥2 vistas, intersecta (o ajusta por
mínimos cuadrados) esos planos de interpretación para recuperar la línea 3D
del eje.

**Por qué funciona sin correspondencia punto-a-punto entre vistas**: el eje
de rotación es un objeto 3D fijo e invariante al punto de vista — un objeto
con simetría rotacional se ve bilateralmente simétrico respecto a la
proyección del eje **desde cualquier ángulo de cámara** (rotarlo alrededor
del eje no cambia su apariencia). Por eso el heurístico actual del prompt
("centro horizontal de la silueta a cada altura") es un buen proxy del eje
en *todas* las vistas, y no hace falta rastrear qué punto específico
corresponde a cuál entre vistas — solo que la línea 2D por vista aproxime
correctamente la proyección del eje.

**Riesgo conocido**: la triangulación de líneas es mal condicionada cuando
los centros de cámara y la línea son casi coplanares/colineales (Hartley
1997; Hartley & Zisserman 2004). Con muestreo Fibonacci esto debería ser
infrecuente, pero conviene verificarlo empíricamente por objeto.

**Base teórica**: es una instancia directa de triangulación clásica de
líneas en coordenadas de Plücker — evidencia sólida, sin necesidad de
desarrollar teoría nueva (ver §8.1).

### 3.2 — Simetría planar (plano), sin malla

**Por qué es fundamentalmente más difícil que el eje**: la simetría de
reflexión *no* tiene la garantía de silueta-simétrica-desde-cualquier-ángulo
que sí tiene la rotacional. La silueta de un objeto con simetría de espejo
solo es bilateralmente simétrica en la imagen cuando la cámara mira **desde
una dirección contenida en el propio plano de simetría** ("de canto"). Desde
otros ángulos (mirando más "de frente" al plano), el heurístico actual del
prompt (`plane_v04_1`: "punto medio horizontal entre bordes de silueta")
deja de aproximar la traza real — mide algo esencialmente arbitrario para
esa vista.

**Esto explica un hallazgo ya reportado en el paper**: *"more views help
axial symmetry up to $n_v=26$ but hurt planar accuracy past $n_v=1$"*
(`main.tex`, Results). Cada vista adicional es una observación válida más
para el eje (mejora la triangulación); para el plano, la mayoría de las
vistas Fibonacci *no* están de canto al plano (desconocido a priori), así
que agregan ruido sistemático, no señal — de ahí que más vistas empeoren el
resultado. Esta pieza (documentada solo cualitativamente en la literatura
existente, ver §8.2) es un candidato a contribución formal propia de la
tesis.

**Además**: incluso con puntos perfectos de vistas "de canto", una sola
línea 3D triangulada (vía el mismo truco de 3.1) determina solo una
dirección *dentro* del plano — el plano podría rotar libremente alrededor de
esa línea y seguir siendo consistente. Hace falta una segunda línea
independiente, no paralela, para fijar la normal (producto cruzado).

**Algoritmo propuesto** (esquema iterativo tipo RANSAC / mínimos cuadrados
reponderados, sin necesitar saber de antemano qué vistas son confiables):

1. **Generar hipótesis**: tomar 2 pares de vistas al azar (4 vistas), obtener
   una línea 3D por par (igual que en 3.1), tomar el producto cruzado de
   ambas direcciones → normal candidata $\hat{n}$.
2. **Puntuar cada vista respecto al candidato**: calcular el ángulo entre la
   dirección de vista de cada cámara (ya conocida) y $\hat{n}$. Ángulo
   cercano a 90° (de canto) → vista confiable; cercano a 0°/180° (de frente)
   → vista no confiable. Este puntaje **no requiere la malla**, solo la pose
   de cámara ya conocida y el candidato actual.
3. **Reajustar**: recalcular el plano usando únicamente las líneas de las
   vistas con buen puntaje.
4. **Iterar** puntuación → reajuste hasta convergencia.
5. **Criterio de aceptación**: si no se consiguen ≥2 pares de vistas
   geométricamente consistentes con buen puntaje, no hay evidencia
   suficiente de un plano — descartar.

**Por qué esto llena un hueco real de la literatura**: el problema formal
"intersectar planos de interpretación de trazas de simetría 2D sin
correspondencias ni geometría densa" no tiene tratamiento dedicado (ver
§8.2). Existen los ingredientes (mirror-symmetry ≡ two-view stereo;
single-view metrology; plane triangulation por homografías de puntos) pero
no la síntesis específica para este régimen de datos (observaciones
dispersas de un detector semántico tipo VLM, no matching denso de
features). Documento este esquema iterativo, y en particular el criterio de
puntaje por ángulo cámara-normal como proxy de confiabilidad, como el
candidato a aporte formal de este capítulo.

**Camino alternativo, más elegante pero de mayor riesgo** (para explorar si
el tiempo lo permite): adaptar François et al. (2003) — tratar la
simetría del propio objeto como una relación de estéreo de dos vistas
(la reflexión de la cámara real respecto al plano candidato actúa como
"cámara virtual"), buscando el plano que hace autoconsistente el conjunto de
puntos de Molmo2 bajo esa relación, sin necesitar rastrear silueta ni
correspondencias explícitas. No implementado aún; queda como extensión.

### 4 — Consolidación, incluyendo 2 y 3 planos

El dataset curado incluye objetos con 1 plano (692), 2 planos (152) y 3
planos (6) de simetría. Extensión vía **RANSAC secuencial / multi-modelo**:

1. Encontrar el plano 1 con el algoritmo de 3.2 (usando las 26 vistas).
2. Identificar qué vistas quedaron *mal* explicadas por el plano 1 (puntaje
   bajo, líneas inconsistentes) — **no** descartar las que sí lo apoyaron,
   porque en objetos con planos mutuamente ortogonales una misma vista puede
   estar "de canto" a más de un plano simultáneamente (si mira cerca de la
   línea de intersección compartida entre planos).
3. Repetir el algoritmo de 3.2 usando las vistas mal explicadas como pool
   candidato, para encontrar el plano 2, y análogamente para el plano 3.
4. **Criterio de parada**: si no hay ≥2 pares de vistas independientes y
   geométricamente consistentes disponibles, detener la búsqueda.
5. **Evitar duplicados**: descartar un plano candidato si su normal es
   esencialmente igual (ángulo bajo algún umbral) a la de un plano ya
   aceptado.
6. **Refinamiento opcional**: dado que los planos múltiples suelen ser
   ortogonales entre sí, restringir la búsqueda de planos 2/3 a normales
   aproximadamente ortogonales a los ya encontrados, para reducir el
   espacio de búsqueda (no necesario para una primera versión funcional).

**Validación**: el dataset ya trae el número real de planos por objeto como
ground truth — permite testear directamente si el criterio de parada acierta
(con la salvedad de que solo hay 6 objetos con 3 planos, así que esa parte
del test será más cualitativa que estadísticamente robusta).

### 5 — Mapeo a la malla, solo para validar

Este es el paso que hace honesta la afirmación "sin acceso a malla": el eje o
plano ya está estimado (pasos 3-4) sin haber tocado la malla en ningún
momento. La malla se usa *después*, exactamente como en `Mapping/evaluate.py`
ya existente: para calcular error angular contra el eje/plano GT y el SDE.
No requiere cambios de diseño, solo asegurarse de que el input a esa etapa
sea la predicción del nuevo pipeline en vez de la del pipeline con
ray-casting.

---

## 6. Reutilización de resultados existentes de Molmo2

Los puntos 2D crudos que Molmo2 ya generó (para todos los prompts/variantes
ya corridos: `axis_v00`–`axis_v05`, `plane_v00`–`plane_v05`, con y sin
flowB/flowC) están disponibles en `predicted_symmetry*.json` / en los JSON de
`molmo_multiview_runner.py` por objeto — **no hace falta volver a correr
inferencia de Molmo2** para probar 3.1/3.2/4 sobre ellos. Plan:

1. Correr el pipeline nuevo (3.1 para axial, 3.2+4 para planar) sobre los
   puntos ya generados por **todas** las variantes de prompt existentes.
2. Identificar qué variantes de prompt dan mejores resultados bajo este
   nuevo esquema de estimación (puede diferir del ranking bajo el pipeline
   con ray-casting, porque ahora el heurístico "de canto vs. de frente" pesa
   distinto).
3. Con esas variantes ganadoras como base, diseñar una nueva generación de
   prompts que corrija específicamente los problemas identificados en esta
   conversación:
   - Para plano: preferir pares bilaterales (`v01`/`v03`, relación
     $P - P' \parallel \hat{n}$) sobre el heurístico de silueta (`v04`), ya
     que los pares bilaterales no dependen de que la cámara esté "de canto".
   - Considerar pedir explícitamente **dos** direcciones/trazas por vista
     (no solo una), para dar al algoritmo de 3.2 más de una línea candidata
     por vista sin depender de combinar vistas distintas.
4. Solo si estas dos rondas no alcanzan, evaluar nueva inferencia de Molmo2
   con prompts rediseñados desde cero.

---

## 7. Próximos pasos (orden sugerido)

1. Implementar y validar 3.1 (axial) — bajo riesgo, base teórica cerrada.
2. Implementar el esquema iterativo de 3.2 (un solo plano) y validarlo contra
   GT vía el paso 5.
3. Extender a 4 (2-3 planos).
4. Correr 1-4 sobre los prompts/resultados de Molmo2 ya existentes (§6.1).
5. Iterar prompts (§6.2-6.3) usando lo aprendido sobre qué vistas resultan
   "de canto" vs. "de frente" en la práctica.
6. (Opcional, mayor riesgo/aporte) explorar la vía François et al. (2003)
   como alternativa/complemento a 3.2.

---

## 8. Referencias bibliográficas

Extraídas de la revisión de literatura hecha para este capítulo
(`compass_artifact_wf-a967ac39-4e37-5e44-afe7-a6468672dbe7_text_markdown.md`).
Ver ese documento para el detalle completo, evaluación de fuerza de
evidencia por tema, y caveats de verificación de citas.

### 8.1 — Triangulación de líneas 3D (base de 3.1) — evidencia sólida

- Bartoli, A., Sturm, P. (2005). *Structure-from-motion using lines:
  Representation, triangulation, and bundle adjustment*. CVIU, 100(3),
  416–441. — Referencia canónica: el "moment vector" en coordenadas de
  Plücker es la normal del plano de interpretación.
- Wu, F., Zhang, M., Wang, G., Hu, Z. (2015). *Algebraic Error Based
  Triangulation and Metric of Lines*. PLOS ONE, 10(7). — Algoritmo óptimo de
  mínimos cuadrados (LINa, OPTa-I/II) y métricas de error sobre el espacio de
  líneas.
- Hartley, R. I. (1997). *Lines and Points in Three Views and the Trifocal
  Tensor*. IJCV, 22(2), 125–140. — Vistas mínimas y por qué 2 vistas no
  restringen igual que para puntos; 3 vistas dan restricciones más fuertes.
- Hartley, R., Zisserman, A. (2004). *Multiple View Geometry in Computer
  Vision* (2ª ed.). Cambridge University Press. — Condiciones de
  degeneración (cap. 15, tensor trifocal).
- Josephson, K., Kahl, F. *Triangulation of Points, Lines and Conics*. SCIA
  2007 / JMIV. — Análisis explícito de configuraciones degeneradas con 3
  vistas. *(Verificar venue exacto antes de citar formalmente.)*
- Hofer, M., Maurer, M., Bischof, H. (2016). *Line3D++: Efficient 3D Scene
  Abstraction Using Line Segments*. CVIU, 157, 167–178. — Pipeline de
  referencia, código abierto.
- Liu, S., Yu, Y., Pautrat, R., Pollefeys, M., Larsson, V. (2023). *3D Line
  Mapping Revisited* (LIMAP). CVPR 2023. — Framework moderno de mapeo de
  líneas 3D multivista.
- Mateus, A., Tahri, O., Aguiar, A. P., Lima, P. U., Miraldo, P. (2021). *On
  Incremental Structure-from-Motion using Lines*. arXiv:2105.11196 / IEEE
  T-RO. — Singularidades en la parametrización del momento de la línea.

### 8.2 — Reconstrucción de planos desde trazas multivista (base de 3.2) — brecha genuina, ingredientes sólidos

- François, A. R. J., Medioni, G. G., Waupotitsch, R. (2003). *Mirror
  symmetry ⇒ 2-view stereo geometry*. Image and Vision Computing, 21(2),
  137–143. — Base teórica del "camino alternativo": ver una escena
  espejo-simétrica desde una vista equivale a verla con dos cámaras
  simétricas respecto al plano.
- Criminisi, A., Reid, I., Zisserman, A. (2000). *Single View Metrology*.
  IJCV, 40(2), 123–148. — Orientación de un plano vía su vanishing line;
  aparato de propagación de incertidumbre.
- Olsson, C., Eriksson, A. (2011). *Triangulating a Plane*. SCIA 2011, LNCS
  6688, 13–23. — Recuperación de plano por minimización directa de
  reproyección (vía homografías de puntos, no líneas de interpretación —
  insumo distinto al nuestro).
- Gao, Y., Yuille, A. L. (2017). *Exploiting Symmetry and/or Manhattan
  Properties for 3D Object Structure Estimation from Single and Multiple
  Images*. CVPR 2017 (ext. IJCV 2019). — Precedente de recuperación de plano
  de simetría multivista, vía SfM de puntos.
- Hong, W., Yang, A. Y., Huang, K., Ma, Y. (2004). *On Symmetry and
  Multiple-View Geometry: Structure, Pose, and Calibration from a Single
  Image*. IJCV, 60(3), 241–265. — Tratamiento formal (teoría de grupos) de
  simetría como geometría multivista.
- Sinha, S. N., Ramnath, K., Szeliski, R. (2012). *Detecting and
  Reconstructing 3D Mirror Symmetric Objects*. ECCV 2012. — Recupera planos
  de simetría pero vía correspondencias de features + SfM (usa geometría 3D
  densa).
- Wang, R., Geraghty, D., Matzen, K., Szeliski, R., Frahm, J.-M. (2020).
  *VPLNet: Deep Single View Normal Estimation with Vanishing Points and
  Lines*. CVPR 2020. — Relación $n = K^\top \ell_{horiz}$ entre normal del
  plano y vanishing line.
- Li, X. et al. (2025). *Symmetry Strikes Back* (Reflect3D). CVPR 2025
  (arXiv:2411.17763). — Confirma que la vista única es ambigua para el plano
  de simetría; usa difusión multivista para resolverlo (enfoque distinto,
  no geométrico puro).

**Nota**: el mal condicionamiento del caso de planos (§3.2, "de canto vs. de
frente") está documentado solo cualitativamente en la literatura revisada —
es terreno abierto para una contribución formal propia.

### 8.3 — Robustez con detecciones semánticas ruidosas, pocas vistas — evidencia sólida (transferible desde pose humana)

- Iskakov, K., Burkov, E., Lempitsky, V., Malkov, Y. (2019). *Learnable
  Triangulation of Human Pose*. ICCV 2019. — Triangulación algebraica
  diferenciable con pesos de confianza por vista; funciona bien con pocas
  cámaras (2+). Precedente directo del esquema de puntaje/reponderación
  propuesto en 3.2.
- Recker, S., Hess-Flores, M., Joy, K. I. (2013). *Statistical Angular
  Error-Based Triangulation for Efficient and Accurate Multi-View Scene
  Reconstruction*. WACV 2013. — Incertidumbre en función de número de
  cámaras, error de reproyección y ángulo de paralaje.
- Lee, S. H., Civera, J. (2019). *Closed-Form Optimal Two-View Triangulation
  Based on Angular Errors*. ICCV 2019 (arXiv:1903.09115). — Soluciones
  cerradas óptimas para error angular (métrica natural para orientación de
  eje/plano).
- Ghasemi, A. et al. *Improving Triangulation by Enforcing Consistency*
  (arXiv:1804.10448). — Cota inferior del error, inversamente cuadrática en
  el número de cámaras (argumento formal de "más vistas ayudan", cuando son
  vistas válidas).

### 8.4 — VLM-pointing + triangulación multivista sin malla — precedente más cercano

- ZeroDex (2026, arXiv:2606.19340). — Precedente más directo: keypoints 2D
  de un VLM elevados a 3D por fusión multivista (triangulación +
  "reference-view ray voting"), sin malla densa. Triangula puntos, no
  líneas/planos — diferencia clave con lo propuesto acá.
- Gong, B. et al. (2025). *ZeroKey: Point-Level Reasoning and Zero-Shot 3D
  Keypoint Detection from Large Language Models*. ICCV 2025
  (arXiv:2412.06292). — El precedente metodológico ya citado en el paper;
  usa Molmo + ray-casting contra malla (lo que este documento busca
  reemplazar).
- Varma T., M. et al. (2024). *Lift3D: Zero-Shot Lifting of Any 2D Vision
  Model to 3D*. CVPR 2024. — Lifting 2D→3D vía correspondencias epipolares +
  NeRF-style rendering (enfoque distinto).

### 8.5 — Por qué evitar correspondencia punto-a-punto explícita entre vistas

- Bhat, S. D., Yamasaki, T. (2026). *Consistent Yet Wrong: Evidence
  Insensitivity in Spatial Vision-Language Models*. arXiv:2606.02742. —
  Muestra que la consistencia entre vistas de un VLM NO implica grounding
  geométrico correcto.
- ZeroKey (arriba) — descubre empíricamente el fracaso de MLLMs en
  point-level reasoning, motivando evitar pedir identidad persistente de
  landmarks.
- Evidencia empírica propia (ya documentada en
  `MolmoPointing/molmo_multiview_runner.py`, docstring de
  `build_flow_c_prompts`): un diseño anterior que pedía a Molmo2 rastrear
  landmarks con identidad persistente entre vistas degeneró en patrones
  mecánicamente equiespaciados — el mismo fallo central del paper.

---

## 9. Notas de verificación

Varias referencias de §8 son muy recientes (2025-2026: ZeroDex,
MolmoPoint, "Consistent Yet Wrong", "Binding Visual Features Point by
Point") — verificar su estado de publicación/peer-review antes de citarlas
como establecidas en la tesis, y tratar sus cifras como preliminares. Ver
también las correcciones de citación ya identificadas en el documento de
revisión de literatura original (autor de François et al. 2003; número de
autores de Gao & Yuille 2017).
