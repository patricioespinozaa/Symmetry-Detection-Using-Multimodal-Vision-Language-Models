# Literatura — Triangulación multivista de eje y plano sin malla

**Experimentos:** A1–A3 (eje), P1–P2 (plano). Notebooks: `test_pipeline_sin_malla.ipynb`,
`sandbox_pipeline_server*.ipynb` §4–7.

**Qué se hace:** cada vista con 2 puntos de Molmo2 define un *plano de interpretación*
(centro de cámara + 2 rayos). Eje: intersección por mínimos cuadrados de ≥2 planos (vector
nulo por SVD). Plano: líneas trianguladas por pares de vistas "de canto" → normales candidatas
por producto cruzado → reajuste por SVD. Se compara con el método con malla (ray-casting +
SVD/PCA).

## 1. Triangulación de líneas 3D (base del eje)

- **[repo]** Bartoli, A., Sturm, P. (2005). *Structure-from-motion using lines: Representation,
  triangulation, and bundle adjustment*. CVIU, 100(3), 416–441. — Referencia canónica: el
  vector momento de Plücker es la normal del plano de interpretación. Fuente:
  `docs/pipeline_sin_malla.md` §8.1, `pipeline_common/triangulation.py`.
- **[repo]** Wu, F., Zhang, M., Wang, G., Hu, Z. (2015). *Algebraic Error Based Triangulation
  and Metric of Lines*. PLOS ONE, 10(7). — Triangulación de líneas por mínimos cuadrados y
  métricas de error.
- **[repo]** Hartley, R. I. (1997). *Lines and Points in Three Views and the Trifocal Tensor*.
  IJCV, 22(2), 125–140. — Por qué 2 vistas restringen poco a una línea.
- **[repo]** Hartley, R., Zisserman, A. (2004). *Multiple View Geometry in Computer Vision*
  (2ª ed.). Cambridge University Press. — Degeneraciones (cap. 15); justifica el riesgo de
  cámaras casi coplanares (A2: vistas concentradas en un polo).
- **[repo]** Josephson, K., Kahl, F. *Triangulation of Points, Lines and Conics*. SCIA 2007 /
  JMIV. *(El repo marca: verificar venue.)*
- **[repo]** Hofer, M., Maurer, M., Bischof, H. (2016). *Line3D++: Efficient 3D Scene
  Abstraction Using Line Segments*. CVIU, 157, 167–178.
- **[repo]** Liu, S., Yu, Y., Pautrat, R., Pollefeys, M., Larsson, V. (2023). *3D Line Mapping
  Revisited* (LIMAP). CVPR 2023.
- **[repo]** Mateus, A., Tahri, O., Aguiar, A. P., Lima, P. U., Miraldo, P. (2021). *On
  Incremental Structure-from-Motion using Lines*. arXiv:2105.11196 / IEEE T-RO.
- **[nuevo]** Hartley, R. I., Sturm, P. (1997). *Triangulation*. CVIU, 68(2), 146–157. —
  Triangulación óptima de puntos; contraste con el caso de líneas.

## 2. Plano de simetría desde varias vistas (base del plano)

Fuente de todas las **[repo]**: `docs/pipeline_sin_malla.md` §8.2.

- **[repo]** François, A. R. J., Medioni, G. G., Waupotitsch, R. (2003). *Mirror symmetry ⇒
  2-view stereo geometry*. Image and Vision Computing, 21(2), 137–143.
- **[repo]** Criminisi, A., Reid, I., Zisserman, A. (2000). *Single View Metrology*. IJCV,
  40(2), 123–148.
- **[repo]** Olsson, C., Eriksson, A. (2011). *Triangulating a Plane*. SCIA 2011, LNCS 6688,
  13–23.
- **[repo]** Gao, Y., Yuille, A. L. (2017). *Exploiting Symmetry and/or Manhattan Properties
  for 3D Object Structure Estimation from Single and Multiple Images*. CVPR 2017.
- **[repo]** Hong, W., Yang, A. Y., Huang, K., Ma, Y. (2004). *On Symmetry and Multiple-View
  Geometry: Structure, Pose, and Calibration from a Single Image*. IJCV, 60(3), 241–265.
- **[repo]** Sinha, S. N., Ramnath, K., Szeliski, R. (2012). *Detecting and Reconstructing 3D
  Mirror Symmetric Objects*. ECCV 2012.
- **[repo]** Wang, R., Geraghty, D., Matzen, K., Szeliski, R., Frahm, J.-M. (2020). *VPLNet:
  Deep Single View Normal Estimation with Vanishing Points and Lines*. CVPR 2020.
- **[repo]** Li, X. et al. (2025). *Symmetry Strikes Back* (Reflect3D). CVPR 2025,
  arXiv:2411.17763. — La vista única es ambigua para el plano de simetría.

El repo anota que el mal condicionamiento del plano ("de canto" vs. "de frente") está
documentado solo cualitativamente: es terreno para una contribución propia.

## 3. VLM pointing + elevación a 3D (contexto del método completo)

- **[repo]** Gong, B. et al. (2025). *ZeroKey: Point-Level Reasoning and Zero-Shot 3D Keypoint
  Detection from Large Language Models*. ICCV 2025, arXiv:2412.06292. — Molmo + ray-casting
  contra malla: el método con malla que A1/P2 usan de contraste.
- **[repo]** ZeroDex (2026). arXiv:2606.19340. — Keypoints 2D de un VLM elevados a 3D por
  fusión multivista, sin malla densa. Precedente más cercano.
- **[repo]** Varma T., M. et al. (2024). *Lift3D: Zero-Shot Lifting of Any 2D Vision Model to
  3D*. CVPR 2024.
- **[nuevo]** Deitke, M. et al. (2024). *Molmo and PixMo: Open Weights and Open Data for
  State-of-the-Art Multimodal Models*. arXiv:2409.17146. — Origen de la capacidad de pointing.

## 4. Detección de simetría 3D clásica (para situar el problema)

- **[nuevo]** Podolak, J., Shilane, P., Golovinskiy, A., Rusinkiewicz, S., Funkhouser, T.
  (2006). *A Planar-Reflective Symmetry Transform for 3D Shapes*. ACM TOG (SIGGRAPH 2006).
- **[nuevo]** Mitra, N. J., Pauly, M., Wand, M., Ceylan, D. (2013). *Symmetry in 3D Geometry:
  Extraction and Applications*. Computer Graphics Forum, 32(6).
- **[nuevo]** Chang, A. X. et al. (2015). *ShapeNet: An Information-Rich 3D Model Repository*.
  arXiv:1512.03012. — Fuente de las mallas.
