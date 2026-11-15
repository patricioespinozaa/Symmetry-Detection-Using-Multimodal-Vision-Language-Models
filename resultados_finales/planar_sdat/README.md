# Resultados planares con S_dat

Experimentos de simetría planar post-procesados con S_dat: un plano por objeto, pipeline sin malla (triangulación), `evaluate.py` con métricas de referencia y `compare_results_no_mesh.py`. Generado por `utils/armar_planar_sdat.py`; no editar a mano. Los CSV son copias: los originales siguen en su carpeta de `resultados_finales/`.

No incluye análisis ni diagnósticos (rotación SO(3), oráculo, pointing 2D) ni la corrida multiplano, que está incompleta.

## CSV

`csv/todos_planar_sdat.csv`: todas las filas, con su grupo y carpeta de origen.

| Grupo | Archivo | Fecha | Contenido | Experimentos | Pareado |
|---|---|---|---|---|---|
| 224_flat_prompts | `csv/224_flat_prompts/ndcfix_flujoA.csv` | 05_11_2026 | Barrido de prompts v00-v07 y variantes _1 (Flujo A, multiimagen) | 14 | sí |
| 224_flat_prompts | `csv/224_flat_prompts/ndcfix_v08.csv` | 05_11_2026 | Prompt v08 (Flujo A, multiimagen) | 1 | sí |
| resolucion | `csv/resolucion/plane_448px.csv` | 07_11_2026 | 448px: v00_1, v02, v04 | 3 | - |
| resolucion | `csv/resolucion/plane_1136px.csv` | 09_11_2026 | 1136px: v00_1, v02 | 2 | - |
| resolucion | `csv/resolucion/ndcfix_448.csv` | 05_11_2026 | 448px: v02, v05_1, v06 (corrida anterior; v02 coincide con plane_448px) | 3 | sí |
| resolucion | `csv/resolucion/ndcfix_1136.csv` | 05_11_2026 | 1136px: v05_1 (sin n=26) | 1 | sí |
| v05_1_224 | `csv/v05_1_224/ndcfix_224_darker.csv` | 05_11_2026 | Iluminación darker | 1 | sí |
| v05_1_224 | `csv/v05_1_224/ndcfix_224_brighter.csv` | 05_11_2026 | Iluminación brighter | 1 | sí |
| v05_1_224 | `csv/v05_1_224/ndcfix_flujoB.csv` | 05_11_2026 | Flujo B (v04_1 y v05_1) | 2 | sí |
| v05_1_224 | `csv/v05_1_224/ndcfix_flujoC.csv` | 05_11_2026 | Flujo C (v04_1) | 1 | sí |
| v05_1_224 | `csv/v05_1_224/ndcfix_single_oct26.csv` | 05_11_2026 | Envío individual | 1 | sí |
| v05_1_224 | `csv/v05_1_224/molmopoint.csv` | 05_11_2026 | MolmoPoint-8B, envío individual | 1 | sí |
| v05_1_224 | `csv/v05_1_224/qwen3vl.csv` | 05_11_2026 | Qwen3-VL-8B, envío individual | 1 | sí |
| v02_224 | `csv/v02_224/plane_v02_224_darker.csv` | 10_11_2026 | Iluminación darker | 1 | - |
| v02_224 | `csv/v02_224/plane_v02_224_brighter.csv` | 10_11_2026 | Iluminación brighter | 1 | - |
| v02_224 | `csv/v02_224/plane_v02_224_single.csv` | 11_11_2026 | Envío individual | 1 | sí |
| v02_224 | `csv/v02_224/plane_v02_224_flujoB.csv` | 10_11_2026 | Flujo B | 1 | sí |
| v02_224 | `csv/v02_224/plane_v02_224_flujoC.csv` | 10_11_2026 | Flujo C | 1 | sí |
| v02_224 | `csv/v02_224/plane_v02_224_qwen3vl.csv` | 11_11_2026 | Qwen3-VL-8B, envío individual | 1 | sí |
| v02_224 | `csv/v02_224/plane_v02_224_molmopoint.csv` | 13_11_2026 | MolmoPoint-8B, texto completo de v02 | 1 | sí |
| v02_224 | `csv/v02_224/plane_v02_224_molmopoint_frase.csv` | 12_11_2026 | MolmoPoint-8B, versión de una frase | 1 | sí |

## Tablas LaTeX (`tex/`)

Requieren en el preámbulo: booktabs, multirow, adjustbox, float y `\newsavebox{\tablebox}`.

| Archivo | Generador | Contenido |
|---|---|---|
| `tex/plane_noviembre.tex` | `utils/generar_tablas_noviembre.py --solo plane` | Flujo A (base, variantes, adicionales, ranking), resolución 448/1136, iluminación, flujos, modo de envío y VLMs con v05_1 |
| `tex/plane_top.tex` | `utils/generar_tablas_seleccion.py` | Ranking por configuración (224px, flat) |
| `tex/plane_prompts.tex` | `utils/generar_tablas_seleccion.py` | Orden por prompt (224px, flat) |
| `tex/plane_v02_224.tex` | `utils/generar_tablas_planar_sdat_v02.py` | Serie v02 a 224px: iluminación, flujos, modo de envío, VLMs y pareado contra la línea base |
