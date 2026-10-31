# Inventario de `Experiments/` — experimentos y análisis por tipo de simetría

Revisión de los 8 notebooks, las inferencias de Molmo2 guardadas en `Experiments/renders/`,
las figuras PNG de la raíz y la carpeta `Experiments/analysis/`. Las cifras salen de los
**outputs guardados en los propios `.ipynb`** (no se re-ejecutó nada). La literatura de cada
tipo de experimento está en [`docs/literature/experimentos/`](literature/experimentos/README.md).

> **Advertencia de interpretación.** Todo lo de acá es *sandbox* (1 objeto o 30 objetos). Varias
> "mejoras" no se sostuvieron en la corrida completa de 850 objetos — ver
> `docs/experimentos_pipeline_sin_malla.md` §5.6 (winner's curse, EXP-C mezclaba dos mecanismos)
> y `docs/reporte_resultados_sin_malla.md` (eje a nivel de azar; plano sin comparar todavía
> contra el predictor trivial "siempre el plano X").

---

## 0. Material común

| Elemento | Contenido |
|---|---|
| `objects/curated_axis_sym_obj/` | 30 mallas `.obj` + GT `.txt` de simetría axial |
| `objects/curated_plane_sym_obj/` | 30 mallas `.obj` + GT `.txt` de simetría planar |
| `renders/{axis,plane}_sym/<id>/{224,448}/flat/` | 114 vistas Fibonacci por objeto, `manifest.json`, `metadata_all.json` |
| Selección de objetos | 10 BUENO / 10 MEDIO / 10 MALO por tipo, con `Mapping/select_sandbox_objects.py` (eje: `angular_error` de `axis_v06_nomesh`; plano: F1 por objeto aproximado de `plane_v04_1_nomesh`) |
| `n_views` evaluados | 6, 14, 26 (1 solo en `test_pipeline_sin_malla`) |

**Inferencias Molmo2 guardadas** (30 objetos cada una):

| Simetría | 224 px | 448 px |
|---|---|---|
| Eje | `axis_v06`, `axis_v06_sandbox`, `axis_v07_sandbox_6pts`, `axis_v08_2pts` | `axis_v06`, `axis_v08_2pts` |
| Plano | `plane_v04_1`, `plane_v04_1_sandbox`, `plane_v05_sandbox_6pts`, `plane_v06_2pts` | `plane_v04_1`, `plane_v06_2pts` |

Las corridas `*_6pts` y `*_2pts` (prompts multi-vista) **existen como JSON pero ningún notebook
tiene sus métricas guardadas en los outputs**; los resultados a escala completa de esas familias
están en `docs/resultados/Prompts_6pts.md` y `docs/resultados/Prompts_v08.md`.

---

## 1. Simetría axial (`axis_sym`)

| # | Experimento / análisis | Notebook (sección) | Qué se hizo | Resultado guardado | Literatura |
|---|---|---|---|---|---|
| A1 | **Prueba de concepto: eje por triangulación sin malla vs. ray-casting + SVD** | `test_pipeline_sin_malla` §1, §3.1, §5 | 1 objeto (`10433e5b…`), 14 prompts (`axis_v00`–`v05`, `_1`, Flow B/C) × n_views {1,6,14,26}. Planos de interpretación por vista → intersección por mínimos cuadrados. Comparado con el método con malla | Sin malla: mejor `axis_v02` n=6 → 28.1°; medias 44–47°. Con malla: medias 59–73°. n=1 imposible sin malla (<2 vistas) | [01](literature/experimentos/01_triangulacion_sin_malla.md) |
| A2 | **Auditoría de distribución de vistas** (bug `get_n_views_entries`) | `test_pipeline_sin_malla` §1 "Diagnóstico" | Se verificó que los índices Fibonacci de cada corrida coinciden con `linspace`; el JSON viejo usaba vistas 0–5 (todas cerca del polo norte) | Bug confirmado en datos pre-2026-06-16; se cambió de objeto | [01](literature/experimentos/01_triangulacion_sin_malla.md) |
| A3 | **Sandbox base de eje** (`axis_v06_sandbox`, 224 px) | `sandbox_pipeline_server`, `_extended`, `_final` §1–9 | Pipeline sin malla inline sobre 30 objetos, métricas por categoría | Error medio 55.4 / 55.2 / 51.5° (n=6/14/26). BUENO ≈ 6–26°, MEDIO ≈ 60°, MALO ≈ 80–89° | [01](literature/experimentos/01_triangulacion_sin_malla.md), [09](literature/experimentos/09_metricas_evaluacion.md) |
| A4 | **Ablaciones de triangulación EXP-A/B/C/D/E** | `_extended` §10–11 (definidas, **sin outputs**); `_final` §10–11 (ejecutadas) | A: SVD ponderado por separación 2D; B: point-then-direction; C: filtro objeto + RANSAC 2D; D: peso × penalización de borde; E: reponderación iterativa por residuo | Baseline gana casi todas las columnas: AUC@45 0.220 / 0.184 / 0.288 vs. ablaciones 0.19–0.22 / 0.18–0.19 / 0.19–0.20 | [02](literature/experimentos/02_ablaciones_triangulacion.md) |
| A5 | **Diagnóstico por objeto de las ablaciones** (+ curvas Precision@θ) | `_final` §12 | Para cada objeto, mejor de 6 estimadores. Figura `ablation_axis_precision_curves.png` | EXP-C "gana" 13/30, Baseline 11 — es artefacto de selección (mínimo de 6 ruidosos) | [02](literature/experimentos/02_ablaciones_triangulacion.md), [08](literature/experimentos/08_diagnostico_casos_conditioning.md) |
| A6 | **Políticas de descarte de vistas** `none` / `any` / `both` | `_final` §14; `_updated_6_pts_v2` §14 | Descartar la vista si ≥1 o los 2 puntos caen en el fondo blanco (umbral RGB 250). Figura `view_filter_policy_comparison.png` | 224 px: `any` descarta hasta el 61% de vistas a n=26 y baja AUC (0.288 → 0.211); `both` ≈ baseline | [04](literature/experimentos/04_filtrado_vistas_puntos.md) |
| A7 | **Re-corrida a 448 px** (`axis_v06`) + ablaciones + políticas | `sandbox_pipeline_server_updated_6_pts_v2` | Mismo pipeline sobre renders de 448 px | Error medio 53.9 / 53.8 / 59.2°; AUC@45 a n=26 cae a 0.201; EXP-C a n=26 colapsa (AUC 0.043) | [06](literature/experimentos/06_prompts_pointing.md), [02](literature/experimentos/02_ablaciones_triangulacion.md) |
| A8 | **Diagnóstico de sesgo al centro (X≈500)** | `_updated_6_pts_v2` celda 79; figura `diagnostico_distribucion_xy.png` | Dispersión horizontal de los puntos por vista y por elevación | `axis_v06` 448: 63% de vistas con X centrado (<30 px), 56% con X degenerado; 71% a n=26 | [05](literature/experimentos/05_sesgo_centro_x500.md) |
| A9 | **EXP-LIT-1 Candidates-then-Select** (simulado) | `sandbox_literatura` | Upper bound (oráculo GT) y lower bound (azar) de elegir entre K=4 candidatos epipolares. Figura `lit1_candidates_precision_curves.png` | ΔAUC del oráculo ≤ +0.028 → **marginal** | [07](literature/experimentos/07_experimentos_literatura_vlm.md) |
| A10 | **EXP-LIT-2 Grid Overlay** (simulado) | `sandbox_literatura` | Grilla 5×5; oráculo de celda correcta vs. celda central. Figura `lit2_grid_cell_heatmap.png` | Oráculo +0.12–0.15 AUC (**vale la pena**); acierto actual de celda 18–24% | [07](literature/experimentos/07_experimentos_literatura_vlm.md) |
| A11 | **EXP-LIT-3 Structured CoT** (simulado) | `sandbox_literatura` | Prior de orientación perfecto / aleatorio / modal por tipo de objeto | Perfecto +0.015–0.037 AUC; prior modal empeora (−0.10 a −0.15) | [07](literature/experimentos/07_experimentos_literatura_vlm.md) |
| A12 | **Comparación global LIT-1/2/3** | `sandbox_literatura` "Comparación global"; `lit_global_comparison.png` | Tabla AUC@45 y error mediano de todas las variantes | Solo la grilla supera claramente al baseline como techo | [07](literature/experimentos/07_experimentos_literatura_vlm.md) |
| A13 | **Visualización cualitativa de casos BUENO/MEDIO/MALO** | `visualizar_casos_axis`; PNGs en `Experiments/analysis/` | Puntos 2D de Molmo2 sobre renders + malla, eje GT, eje predicho y rayos cámara-punto en 3D (3 objetos por franja) | Herramienta de inspección: distingue "consistente pero equivocado" (identidad) de "rayos dispersos" (ruido) | [08](literature/experimentos/08_diagnostico_casos_conditioning.md) |
| A14 | **Ranking de prompts sin malla** (14 prompts, 850 objetos) | `analisis_prompts_no_mesh` §1–3, §5–6 | Carga de CSV de `compare_results_no_mesh.py`; ranking, tabla por n_views, ablación v0 vs. `_1`, métricas vs. n_views | Ganador `axis_v05_1` n=26 (57.4°, AUC 0.104); peor `axis_v07` (70.1°) | [06](literature/experimentos/06_prompts_pointing.md) |
| A15 | **Antes/después de `--filter-off-object`** | `analisis_prompts_no_mesh` §7 | Merge de `experiments_27_08_2026` vs. `_v2` para `axis_v06_nomesh` | **Sin efecto detectable**: el notebook informa que ningún experimento cambió (revisar si la corrida se hizo con `--overwrite`) | [04](literature/experimentos/04_filtrado_vistas_puntos.md) |

## 2. Simetría planar (`plane_sym`)

| # | Experimento / análisis | Notebook (sección) | Qué se hizo | Resultado guardado | Literatura |
|---|---|---|---|---|---|
| P1 | **Prueba de concepto: plano sin malla** (líneas por pares de vistas "de canto" → normal) | `test_pipeline_sin_malla` §2, §3.2 | 1 objeto (`10654ea6…`, 1 plano GT), 14 prompts × {1,6,14,26} | Mejor `plane_v02` n=14: 2.65°, F1_ref = 1.0; `plane_v01_1` (elegido a mano) era de los peores (84°) | [01](literature/experimentos/01_triangulacion_sin_malla.md) |
| P2 | **Comparación con ray-casting + SVD (plano)** | `test_pipeline_sin_malla` §3.2 "Comparación" | Normal = componente de mínima varianza de la nube 3D, igual que `fit_plane` | Comparación sin vs. con malla en las mismas 56 combinaciones | [01](literature/experimentos/01_triangulacion_sin_malla.md) |
| P3 | **Consolidación iterativa multi-plano** (hasta 3 planos) | `test_pipeline_sin_malla` §4 | Quitar vistas que apoyaron el plano k y repetir; parada por <4 vistas, sin pares o duplicado <15° | Segundo plano (espurio) en 19/34 combinaciones; el plano 0 ya era el bueno | [03](literature/experimentos/03_consolidacion_multiplano.md) |
| P4 | **Métricas de referencia SDE_ref / F1_ref** | `test_pipeline_sin_malla` §4.5–5; todos los sandbox §8 | Funciones importadas o copiadas verbatim de `Mapping/evaluate.py` (gpytoolbox) | F1_ref coincide con el ranking por error angular en el objeto de prueba | [09](literature/experimentos/09_metricas_evaluacion.md) |
| P5 | **Sandbox base de plano** (`plane_v04_1_sandbox`, `max_planes=3`, 224 px) | `sandbox_pipeline_server*` §1–9 | 30 objetos, recall/precisión de planos por categoría | Recall 0.20 / 0.67 / 0.65; precisión 0.23 / 0.48 / 0.42 (n=6/14/26) | [03](literature/experimentos/03_consolidacion_multiplano.md), [09](literature/experimentos/09_metricas_evaluacion.md) |
| P6 | **Ablaciones EXP-A / C / D / F** | `_final` §10–11 | A y D: pesos por longitud/borde; C: filtro objeto + RANSAC 2D; F: aceptar un plano solo si SDE_ref < 0.02 (usa la malla) | **EXP-F** gana F1: 0.221 / 0.420 / 0.334 vs. baseline 0.139 / 0.374 / 0.205; A y D ≈ baseline | [02](literature/experimentos/02_ablaciones_triangulacion.md), [03](literature/experimentos/03_consolidacion_multiplano.md) |
| P7 | **Planos predichos vs. GT** | `_final` §12; `ablation_plane_npred_vs_gt.png` | Sobrepredicción: ~1.9 planos predichos vs. ~1.18 GT a n=14 | Motivación directa de EXP-F y de los sweeps | [03](literature/experimentos/03_consolidacion_multiplano.md) |
| P8 | **Sweep `max_planes` ∈ {1,2,3}** | `_final` §13.1; `sweep_max_planes.png` | Sin re-inferencia | F1 casi igual; a n=26 `max_planes=1` es mejor (0.220 vs. 0.205) | [03](literature/experimentos/03_consolidacion_multiplano.md) |
| P9 | **Sweep `dup_angle_thresh`** ∈ {5,10,15,20,30}° | `_final` §13.2; `sweep_dup_angle.png` | Umbral para considerar duplicado un plano nuevo | Prácticamente sin efecto (F1 0.367–0.374 a n=14) | [03](literature/experimentos/03_consolidacion_multiplano.md) |
| P10 | **Sweep `sde_gate`** ∈ {0.005 … 0.05, ∞} | `_final` §13.3; `sweep_sde_gate.png` | Gate geométrico de EXP-F | Monótono: gate 0.005 → F1 0.313 / 0.531 / 0.484 (vs. 0.139 / 0.374 / 0.205 sin gate) | [03](literature/experimentos/03_consolidacion_multiplano.md) |
| P11 | **Mejor configuración + heatmap** `max_planes × dup_angle` | `_final` §13.4–13.5; `heatmap_maxplanes_dup.png` | Búsqueda en grilla | Mejor en todos los n_views: `max_planes=3, dup=15°, gate=0.005` | [03](literature/experimentos/03_consolidacion_multiplano.md) |
| P12 | **Políticas de descarte de vistas** | `_final` §14; `_updated_6_pts_v2` §14 | `none` / `any` / `both` | F1 `any` 0.203 / 0.241 / 0.093; `both` 0.204 / 0.357 / 0.203 (vs. 0.139 / 0.374 / 0.205) | [04](literature/experimentos/04_filtrado_vistas_puntos.md) |
| P13 | **Re-corrida a 448 px** (`plane_v04_1`) | `_updated_6_pts_v2` | Mismo pipeline | Recall 0.27 / 0.43 / 0.88; precisión 0.40 / 0.50 / 0.36 | [06](literature/experimentos/06_prompts_pointing.md) |
| P14 | **Sesgo al centro (X≈500)** | `_updated_6_pts_v2` celda 80 | Igual que A8 | `plane_v04_1` 448: **98.9%** de vistas con X centrado, X_std = 2.3 px → el prompt silhouette-midpoint colapsa al centro | [05](literature/experimentos/05_sesgo_centro_x500.md) |
| P15 | **Ranking de prompts sin malla** (14 prompts, 850 objetos) | `analisis_prompts_no_mesh` §4 | Ranking por recall y por F1_ref, ablación v0 vs. `_1` (n=14), métricas vs. n_views | Por recall: `plane_v07` n=26 (0.858); por F1_ref: `plane_v04_1` n=14; peor `plane_v01_1` | [06](literature/experimentos/06_prompts_pointing.md) |
| P16 | **Antes/después de `--filter-off-object`** | `analisis_prompts_no_mesh` §7 | `plane_v04_1_nomesh` | **Sin cambios detectados** (mismo aviso que A15) | [04](literature/experimentos/04_filtrado_vistas_puntos.md) |

---

## 3. Mapa notebook → experimentos

| Notebook | Fecha (mtime) | Simetría | Experimentos |
|---|---|---|---|
| `test_pipeline_sin_malla.ipynb` | 25-ago | ambas | A1, A2, P1–P4 |
| `visualizar_casos_axis.ipynb` | 27-ago | eje | A13 |
| `analisis_prompts_no_mesh.ipynb` | 27-ago | ambas | A14, A15, P15, P16 |
| `sandbox_pipeline_server.ipynb` | 29-ago | ambas | A3, P5 |
| `sandbox_pipeline_server_extended.ipynb` | 30-ago | ambas | A3, P5 + definición de A4/P6 (sin ejecutar) |
| `sandbox_pipeline_server_final.ipynb` | 31-ago | ambas | A3–A6, P5–P12 |
| `sandbox_pipeline_server_updated_6_pts_v2.ipynb` | 3-sep | ambas | A6–A8, P12–P14 (448 px; pese al nombre, la corrida activa es `axis_v06`/`plane_v04_1` y los prompts de 6 pts quedan comentados) |
| `sandbox_literatura.ipynb` | 6-sep | eje | A9–A12 |

## 4. Huecos detectados

- EXP-LIT-1/2/3 son **simulaciones con GT** (techos/pisos), no inferencias reales de Molmo2. La
  versión real de LIT-1 (`Pipeline_Experiments/candidates_then_select/`) no se corrió.
- No hay ningún experimento de literatura (LIT-x) para **plano**.
- Los JSON `*_6pts` y `*_2pts` del sandbox no tienen métricas en ningún notebook.
- §7 de `analisis_prompts_no_mesh` no detectó diferencias antes/después del filtro: probablemente
  la corrida "después" no sobrescribió nada.
