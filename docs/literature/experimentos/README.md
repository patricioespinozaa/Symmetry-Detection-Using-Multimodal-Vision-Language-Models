# Literatura por tipo de experimento (`Experiments/`)

Un archivo por tipo de experimento o análisis del inventario
[`docs/inventario_Experiments.md`](../../inventario_Experiments.md).

| Archivo | Tipo | Experimentos del inventario |
|---|---|---|
| [01_triangulacion_sin_malla.md](01_triangulacion_sin_malla.md) | Triangulación multivista de eje/plano sin malla vs. ray-casting | A1–A3, P1–P2 |
| [02_ablaciones_triangulacion.md](02_ablaciones_triangulacion.md) | Ablaciones EXP-A a EXP-E (triangulación robusta/ponderada) | A4, A5, A7, P6 |
| [03_consolidacion_multiplano.md](03_consolidacion_multiplano.md) | Consolidación multi-plano, EXP-F y sweeps de parámetros | P3, P5–P11 |
| [04_filtrado_vistas_puntos.md](04_filtrado_vistas_puntos.md) | Descarte de vistas/puntos fuera del objeto | A6, A15, P12, P16 |
| [05_sesgo_centro_x500.md](05_sesgo_centro_x500.md) | Diagnóstico de colapso al centro de la imagen | A8, P14 |
| [06_prompts_pointing.md](06_prompts_pointing.md) | Diseño y ranking de prompts, resolución 224/448 | A7, A14, P13, P15 |
| [07_experimentos_literatura_vlm.md](07_experimentos_literatura_vlm.md) | EXP-LIT-1/2/3 (candidatos, grilla, CoT) | A9–A12 |
| [08_diagnostico_casos_conditioning.md](08_diagnostico_casos_conditioning.md) | Inspección cualitativa, conditioning, sesgo de selección | A5, A13 |
| [09_metricas_evaluacion.md](09_metricas_evaluacion.md) | Métricas: error angular, AUC, Precision@θ, SDE_ref, F1_ref | todos |

## Convención de origen

- **[repo]**: la cita ya aparece en un documento o notebook del repo; se indica dónde.
- **[nuevo]**: la agregué en esta revisión desde conocimiento general (no aparece en el repo).
  Es una referencia conocida, pero conviene verificar los datos bibliográficos antes de
  citarla formalmente.

Las referencias 2025–2026 (arXiv) son preprints recientes: `docs/pipeline_sin_malla.md` §9 ya
advierte que hay que verificar su estado de publicación y tomar sus cifras como preliminares.
