# Literatura — Sesgo al centro de la imagen (X≈500)

**Experimentos:** A8 (eje), P14 (plano). Notebook: `sandbox_pipeline_server_updated_6_pts_v2.ipynb`
celdas 79–80 (figura `diagnostico_distribucion_xy.png`). Puerto batch:
`Pipeline_Experiments/diagnostics/diagnose_center_bias.py`.

**Qué se hace:** para cada vista se mide la distancia de los puntos al centro horizontal
(X = 500 en coordenadas Molmo 0–1000) y su dispersión en X, separando por banda de elevación.
Así se distinguen las vistas cenitales, donde X≈500 es legítimo, del resto, donde indicaría un
atajo. Resultado: `axis_v06` tiene 63% de vistas centradas y `plane_v04_1` tiene 98.9%, con
X_std = 2.3 px.

## Referencias en el repo

- **[repo]** `test_pipeline_sin_malla.ipynb` §2 y `docs/pipeline_sin_malla.md` §6. — Ya
  anticipaban que `plane_v04*` (silhouette-midpoint) devuelve casi siempre x=500, y que eso
  hace que la triangulación colapse al eje vertical del mundo.
- **[repo]** Bhat & Yamasaki (2026). *Consistent Yet Wrong*. arXiv:2606.02742. — Respuestas
  consistentes entre vistas que no dependen de la evidencia visual.
- **[repo]** MOKA (arXiv:2403.03174) y SketchVLM (arXiv:2604.22875). — El notebook de
  literatura los cita como remedio al prior X≈500: anclar a una celda de grilla visible. Ver
  [07_experimentos_literatura_vlm.md](07_experimentos_literatura_vlm.md).

## Fundamentos agregados (sesgos y atajos en VLMs)

- **[nuevo]** Geirhos, R. et al. (2020). *Shortcut Learning in Deep Neural Networks*. Nature
  Machine Intelligence, 2, 665–673. — Marco general: el modelo resuelve la tarea con una regla
  superficial ("el medio de la imagen") en lugar de la estructura pedida.
- **[nuevo]** Tong, S. et al. (2024). *Eyes Wide Shut? Exploring the Visual Shortcomings of
  Multimodal LLMs*. CVPR 2024. — Fallas sistemáticas de MLLMs en patrones visuales simples
  (orientación, posición).
- **[nuevo]** Kamath, A., Hessel, J., Chang, K.-W. (2023). *What's "up" with vision-language
  models? Investigating their struggle with spatial reasoning*. EMNLP 2023.
- **[nuevo]** Tatler, B. W. (2007). *The central fixation bias in scene viewing*. Journal of
  Vision, 7(14):4. — Sesgo central en la mirada humana. Los datos de pointing humanos con que se
  entrenan estos modelos (p. ej. PixMo-Points) podrían heredarlo; esto es una hipótesis, no algo
  demostrado.
- **[nuevo]** Deitke, M. et al. (2024). *Molmo and PixMo*. arXiv:2409.17146. — Describe el
  formato de coordenadas 0–100 / 0–1000 y los datos de pointing; útil para discutir de dónde
  puede venir el prior.

## Lectura

Que el sesgo sea casi total en el prompt de plano (98.9%) e independiente de la elevación
indica que la respuesta no depende de la geometría de la vista. Es el mismo modo de falla que la
literatura de *shortcut learning* describe, y justifica medir siempre contra un predictor
trivial (`docs/reporte_resultados_sin_malla.md`).
