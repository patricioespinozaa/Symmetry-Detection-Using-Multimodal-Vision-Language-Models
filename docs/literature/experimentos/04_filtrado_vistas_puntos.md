# Literatura — Descarte de vistas y puntos fuera del objeto

**Experimentos:** A6, A15 (eje), P12, P16 (plano). Notebooks:
`sandbox_pipeline_server_final.ipynb` §14, `_updated_6_pts_v2.ipynb` §14,
`analisis_prompts_no_mesh.ipynb` §7. Código de producción: `Pipeline_Experiments/view_filtering.py`
(políticas `none` / `per_point` / `any` / `both`).

**Qué se hace:** un punto de Molmo2 se considera "fuera del objeto" si el píxel del render es
fondo blanco (RGB > 250 en los 3 canales). `any` descarta la vista si algún punto está fuera y
`both` solo si lo están los dos. En §7 de `analisis_prompts_no_mesh` se compara la corrida con
`--filter-off-object` contra la corrida sin filtro.

## Referencias en el repo

- **[repo]** `docs/diagnostico_conditioning_axis.md` §7 (análisis propio): el porcentaje de
  puntos sobre el objeto baja de 82.8% (n=6) a 51.5% (n=26). No explica el error angular
  (lift ≈ 1), pero sí parte de los outliers de traslación (lift 1.2–1.4×). Es la motivación del
  experimento.
- **[repo]** Recker, S., Hess-Flores, M., Joy, K. I. (2013). WACV 2013. — En el sandbox, el
  filtro estaba fusionado con RANSAC 2D dentro de EXP-C. Ver
  [02_ablaciones_triangulacion.md](02_ablaciones_triangulacion.md).
- **[repo]** Bhat, S. D., Yamasaki, T. (2026). *Consistent Yet Wrong: Evidence Insensitivity
  in Spatial Vision-Language Models*. arXiv:2606.02742. — Que un VLM señale "algo" no implica
  grounding correcto. Fuente: `docs/pipeline_sin_malla.md` §8.5.

## Fundamentos agregados

- **[nuevo]** Laurentini, A. (1994). *The Visual Hull Concept for Silhouette-Based Image
  Understanding*. IEEE TPAMI, 16(2), 150–162. — Consistencia con la silueta: un punto 3D
  válido debe proyectarse dentro de la silueta en todas las vistas. Es la justificación
  geométrica de usar la máscara del objeto como filtro.
- **[nuevo]** Fischler & Bolles (1981), RANSAC, Comm. ACM 24(6). — Rechazo de outliers antes
  del ajuste. El filtro de silueta es un rechazo determinista que usa información externa (la
  máscara) en lugar de consenso.
- **[nuevo]** Hartley & Zisserman (2004), *Multiple View Geometry*, cap. 4 (estimación
  robusta). — Un descarte agresivo reduce el número de vistas y empeora el condicionamiento.
  Es la lectura natural de que `any` empeore a n=26: descarta el 58–61% de las vistas.

## Lectura

El filtro de silueta solo elimina errores de tipo "fuera del objeto". Los errores de tipo
"sobre el objeto pero en la estructura equivocada" (identidad) quedan intactos, y según el
diagnóstico del repo son los que dominan. Por eso `both` ≈ baseline y `any` empeora.
