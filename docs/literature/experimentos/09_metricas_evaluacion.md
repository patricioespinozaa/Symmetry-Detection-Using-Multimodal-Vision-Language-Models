# Literatura — Métricas de evaluación usadas en `Experiments/`

**Aplica a:** todos los experimentos. La definición exacta de cada métrica está en
`docs/metricas_evaluacion.md` y `docs/actualizacion_metricas.md`; aquí solo va la literatura.
Los notebooks copian o importan las fórmulas de `Mapping/evaluate.py` sin cambios.

| Métrica | Simetría | Dónde aparece |
|---|---|---|
| Error angular, mediana, Precision@{5,10,15}°, AUC angular truncado a 45° | eje (y plano en modo un-plano) | todos los sandbox, `analisis_prompts_no_mesh` |
| Error de traslación (distancia punto-recta) | eje | sandbox §8 |
| Recall / precisión sobre el conjunto de planos GT | plano (multi-plano) | sandbox §8–9, sweeps |
| SDE_ref | eje y plano | sandbox §8.1, ablaciones, LIT-x |
| F1_ref (greedy y Hungarian, umbrales 0.05–0.20, TP/FP/FN acumulados) | solo plano | `test_pipeline_sin_malla` §4.5, sandbox, sweeps |

## SDE_ref / F1_ref

- **[repo]** Aguirre & Sipiran. *Enhanced Back-Projection of Vision Features for 3D Symmetry
  Detection*. WACV 2026 (repo `EnhancedBackProjection`). — Fuente verbatim de `calplaneloss`,
  `calaxisloss` y `f1_match_counts`: 1000 puntos muestreados sobre la superficie con
  `gpytoolbox`, distancia cuadrática sin normalizar. *(El repo marca: confirmar la cita.)*
- **[repo]** Gao, L. et al. *PRS-Net*. IEEE TVCG 2021, DOI:10.1109/TVCG.2020.3003823. —
  Popularizó SDE como métrica estándar de simetría planar en ShapeNet.
- **[repo]** Li, X. et al. (2025). *Reflect3D*. CVPR 2025. — Convención de F1 con asignación
  óptima (`f1_ref_hungarian`).
- **[repo]** E3Sym (ICCV 2023); Je et al. (SIGGRAPH Asia 2024). — F1 multi-candidato, sin
  verificar de primera mano.
- **[repo]** No existe un F1 estándar para eje (`docs/metricas_evaluacion.md` §4.2; CLAUDE.md).

## Error angular, AUC y Precision@θ

- **[nuevo]** Sarlin, P.-E., DeTone, D., Malisiewicz, T., Rabinovich, A. (2020). *SuperGlue:
  Learning Feature Matching with Graph Neural Networks*. CVPR 2020. — Popularizó el AUC de
  error angular de pose truncado a umbrales (5°/10°/20°). El AUC@45° del repo es la misma idea
  con otro umbral.
- **[nuevo]** Shi, Y. et al. (2020). *SymmetryNet*. ACM TOG (SIGGRAPH Asia 2020). — Evaluación
  de simetrías reflectivas y rotacionales (ejes) por error angular y curvas precisión-recall.
- **[nuevo]** Hungarian algorithm: Kuhn, H. W. (1955). *The Hungarian Method for the
  Assignment Problem*. Naval Research Logistics Quarterly, 2, 83–97.

## Referencias de nivel de azar (para interpretar cualquier tabla)

- **[repo]** `docs/reporte_resultados_sin_malla.md` §2.1: error angular esperado de una
  dirección aleatoria = 57.3°. El baseline de eje del sandbox (51–55°) está cerca de ese
  nivel.
- **[repo]** `docs/literature/baseline_aleatorio.md` y el módulo `RandomBaseline/`
  (`docs/features/random-baseline.md`).
