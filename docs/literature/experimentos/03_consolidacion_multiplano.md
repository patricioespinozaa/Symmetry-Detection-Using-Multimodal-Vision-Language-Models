# Literatura — Consolidación multi-plano, EXP-F y sweeps de parámetros

**Experimentos:** P3, P5–P11 (solo plano). Notebooks: `test_pipeline_sin_malla.ipynb` §4,
`sandbox_pipeline_server_final.ipynb` §10 (EXP-F), §12 (planos predichos vs. GT) y §13
(sweeps de `max_planes`, `dup_angle_thresh`, `sde_gate`, heatmap).

**Qué se hace:** detectar hasta 3 planos de forma secuencial (encontrar plano → quitar las
vistas que lo apoyaron → repetir). Se detiene por falta de vistas, por falta de pares
independientes o porque el candidato duplica a uno ya aceptado (< `dup_angle_thresh`). EXP-F
además exige `SDE_ref < sde_gate` medido sobre la malla real. Es la única variante que usa la
malla al predecir, y es un experimento controlado a propósito.

## Gate por error de simetría (EXP-F, P10)

- **[repo]** Gao, L. et al. *PRS-Net: Planar Reflective Symmetry Detection Net for 3D Models*.
  IEEE TVCG 2021, DOI:10.1109/TVCG.2020.3003823. — Popularizó la distancia de simetría (SDE)
  como pérdida y criterio de validación de planos. EXP-F reusa esa idea como criterio de parada.
  Fuente: celda EXP-F del notebook, `docs/experimentos_pipeline_sin_malla.md` §8.7. *(En
  `docs/metricas_evaluacion.md` el año aparece "a confirmar".)*
- **[repo]** Aguirre & Sipiran. *Enhanced Back-Projection of Vision Features for 3D Symmetry
  Detection*. WACV 2026 (repo `EnhancedBackProjection`). — Fuente verbatim de `SDE_ref`/`F1_ref`.
  Además genera varios candidatos de plano, los filtra por distancia Chamfer reflectiva y los
  deduplica por similitud rotacional, con tope de 10 candidatos. Es el análogo más directo del
  pipeline de candidatos + dedup + gate. Fuente: `docs/metricas_evaluacion.md` §4.1.

## Detección de múltiples planos y evaluación multi-candidato (P3, P8, P11)

- **[repo]** Li, X. et al. (2025). *Symmetry Strikes Back* (Reflect3D). CVPR 2025,
  arXiv:2411.17763. — Predice varios planos por objeto. Su convención de F1 con asignación
  óptima es la que adopta `f1_ref_hungarian` (CLAUDE.md).
- **[repo]** E3Sym (ICCV 2023) y Je et al., *Robust Symmetry Detection via Riemannian Langevin
  Dynamics* (SIGGRAPH Asia 2024). — Mencionados como trabajos que reportan F1 multi-candidato.
  `docs/metricas_evaluacion.md` §4.2 indica que **no se verificaron de primera mano**.
- **[nuevo]** Shi, Y., Huang, J., Zhang, H., Xu, X., Rusinkiewicz, S., Xu, K. (2020).
  *SymmetryNet: Learning to Predict Reflectional and Rotational Symmetries of 3D Shapes from
  Single-View RGB-D Images*. ACM TOG (SIGGRAPH Asia 2020). — Predicción de múltiples
  simetrías, con evaluación por curvas precisión-recall.
- **[nuevo]** Podolak, J. et al. (2006). *A Planar-Reflective Symmetry Transform for 3D Shapes*.
  ACM TOG (SIGGRAPH 2006). — Extracción de varios planos como máximos locales de una función de
  simetría.

## Deduplicación por umbral angular (P9)

- **[nuevo]** Neubeck, A., Van Gool, L. (2006). *Efficient Non-Maximum Suppression*. ICPR
  2006. — `dup_angle_thresh` funciona como una supresión de no-máximos sobre la esfera de
  normales. El sweep mostró que es casi insensible entre 5° y 30°.
- **[nuevo]** Fischler & Bolles (1981), RANSAC. — La extracción secuencial de modelos
  (encontrar uno → quitar sus inliers → repetir) es el esquema "sequential RANSAC" que replica
  la consolidación.

## Lectura

El sweep (P10) muestra que lo que mejora el F1 es **reducir la sobrepredicción** (de ~1.9 a
~0.8 planos por objeto), no recuperar planos nuevos. Esto es coherente con los
trabajos citados, que filtran candidatos por un score geométrico. Como EXP-F usa la malla al
predecir, su ganancia no es atribuible al VLM; sirve como techo para un gate sin malla.
