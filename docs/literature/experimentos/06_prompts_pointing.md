# Literatura — Diseño y ranking de prompts de pointing (y resolución)

**Experimentos:** A14, P15 (ranking de 14 prompts por simetría, ablación v0 vs. `_1`, métricas
vs. `n_views`), A1/P1 (barrido de 14 prompts en 1 objeto), A7/P13 (re-corrida a 448 px) e
inferencias `*_6pts` / `*_2pts` sin métricas. Notebooks: `analisis_prompts_no_mesh.ipynb`,
`test_pipeline_sin_malla.ipynb`, `sandbox_pipeline_server_updated_6_pts_v2.ipynb`.

**Familias de prompt involucradas:** independiente (puntos sobre el eje/plano), *silhouette
midpoint* (`plane_v04*`), bilateral (`plane_v01*`), 6 puntos por vista (`axis_v07_6pts`,
`plane_v05_6pts`), multi-vista de 2 puntos (`axis_v08_2pts`, `plane_v06_2pts`) y Flujos A/B/C.

## Referencias en el repo

- **[repo]** Gong, B. et al. (2025). *ZeroKey*. ICCV 2025, arXiv:2412.06292. — Fracaso de
  MLLMs en point-level reasoning con identidad persistente; motiva que Flujo C no pida
  correspondencias. Fuente: `docs/pipeline_sin_malla.md` §8.5.
- **[repo]** Bhat & Yamasaki (2026). *Consistent Yet Wrong*. arXiv:2606.02742.
- **[repo]** Evidencia propia: el docstring de `build_flow_c_prompts`
  (`MolmoPointing/molmo_multiview_runner.py`) documenta que un Flujo C con identidad
  persistente degeneró en patrones equiespaciados.
- **[repo]** MOKA (arXiv:2403.03174) → `axis_lit2_grid`; Structured CoT (arXiv:2507.13362) →
  `axis_lit3_cot`. Fuente: `MolmoPointing/prompts_registry.py`. Ver
  [07_experimentos_literatura_vlm.md](07_experimentos_literatura_vlm.md).
- **[repo]** `docs/literature/literatura-por-leer.md`: El Banani et al. (*Probing the 3D
  Awareness of Visual Foundation Models*, CVPR 2024), GIQ (arXiv:2506.08194), CLIPSym (ICCV
  2025), Orient Anything V2 (NeurIPS 2025), *Telling Left from Right* (CVPR 2024). Están
  pendientes de lectura y sirven para enmarcar la capacidad 3D del encoder.

## Fundamentos agregados

- **[nuevo]** Deitke, M. et al. (2024). *Molmo and PixMo*. arXiv:2409.17146. — Formato de
  salida `<point>` y datos de pointing.
- **[nuevo]** Wei, J. et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large
  Language Models*. NeurIPS 2022.
- **[nuevo]** Yang, J. et al. (2023). *Set-of-Mark Prompting Unleashes Extraordinary Visual
  Grounding in GPT-4V*. arXiv:2310.11441. — Marcas visuales sobre la imagen en lugar de
  coordenadas libres; relacionado con la grilla y los candidatos marcados.
- **[nuevo]** Chen, B. et al. (2024). *SpatialVLM: Endowing Vision-Language Models with Spatial
  Reasoning Capabilities*. CVPR 2024. — Razonamiento espacial 3D en VLMs y sus límites.

## Nota sobre resolución (224 vs. 448 px)

No encontré en el repo literatura específica que justifique el cambio de resolución. Los
resultados a escala completa están en `docs/resultados/tablas_448` y `tablas_1136`. Si se
quiere fundamentar, lo pertinente es la documentación del preprocesamiento de Molmo2 (recorte en
*crops*), que ya se analizó en `docs/resultados/tablas_tokens`.

## Lectura

El hallazgo de P1, donde el prompt elegido a priori fue de los peores, refuerza una idea
habitual en la literatura de prompting: el ranking de prompts es empírico y depende del objeto y
del modelo. Por eso conviene seleccionarlo sobre un conjunto de validación y no por argumento
teórico.
