# Resumen de resultados por prompt — `ndcfix_flujoC` (experiments_06_10_2026)

Flujo C (descripción y pista en una vista, luego pointing multivista con 2 puntos por vista).

Fuentes: `axis_sym_axis_v05_1_flowC_ndcfix_nomesh_comparison.csv`, `plane_sym_plane_v04_1_flowC_ndcfix_mp1_nomesh_comparison.csv`.
2 prompts × 3 configuraciones de vistas (6, 14, 26), método `triangulation` (pipeline sin malla, conversión NDC corregida, `n_total = 850` objetos por celda; plano en modo de un solo plano, `max_planes = 1`). En las tablas, **negrita** = mejor prompt según el promedio sobre las 3 configuraciones de vistas (solo cuando hay más de un prompt).

> **Cautelas de lectura**
> - `n_objects` es la cantidad de objetos con predicción válida (de 850). Las métricas angulares imputan 90° a los objetos sin predicción; la traslación, SDE_ref y F1_ref se calculan solo sobre los objetos con predicción. Compará siempre contra la tabla de cobertura.
> - `AUC_angular` trunca en 45°. `SDE_ref`/`F1_ref` no son intercambiables con `angular_error`/`translation_error` (ver `docs/metricas_evaluacion.md`).
> - Un error angular mediano ≈ 57–60° (axis) o ≈ 83–87° (plane) es compatible con nivel de azar / predicción trivial constante, no con aprendizaje de simetría.

## 1. Ranking global (promedio sobre 6/14/26 vistas, ordenado por AUC angular)

### Simetría axial
| Rank | Prompt | auc_angular | angular_error_median | precision_10deg | sde_ref_mean | n_objects |
|---|---|---|---|---|---|---|
| 1 | v05_1_flowC | 0.218 | 90.0 | 0.214 | 0.029 | 262.0 |

### Simetría planar
| Rank | Prompt | auc_angular | angular_error_median | precision_10deg | f1_ref | f1_ref_hungarian | sde_ref_mean | n_objects |
|---|---|---|---|---|---|---|---|---|
| 1 | v04_1_flowC | 0.059 | 90.0 | 0.042 | 0.069 | 0.075 | 0.040 | 215.3 |

## 2. Conclusiones principales

**Cobertura: el problema dominante**
- El Flujo C deja **sin predicción a la mayoría de los objetos**: en eje, 374 / 244 / 168 objetos con predicción (de 850) con 6 / 14 / 26 vistas; en plano, 316 / 163 / 167. Por eso el error angular mediano es 90° en todas las configuraciones: más de la mitad de los objetos recibe la imputación de 90°.
- La causa ya identificada es que, en este flujo, Molmo2 devuelve con frecuencia menos de dos puntos útiles por vista, de modo que no se puede formar el par que necesita la triangulación (o los roles `obj_id` 1 y 2 en plano).

**Axis (`axis_sym`)**
- **Cuando hay predicción, es buena.** Sobre los objetos con predicción, la fracción con error menor que 10° es 0.72 (6 vistas), 0.77 (14) y 0.52 (26), muy por encima de `v05_1` en el Flujo A. Medido sobre los 850 objetos, P@10° sube de 0.107 a 0.318 con 6 vistas y el AUC de 0.208 a 0.316.
- Es decir, el Flujo C cambia cobertura por precisión: responde para pocos objetos, pero con puntos bien ubicados. Si se resolviera la pérdida de puntos, podría ser competitivo; hoy no lo es en métricas que penalizan los objetos sin predicción.

**Plane (`plane_sym`)**
- **Empeora en todo.** AUC 0.113 / 0.040 / 0.024 frente a 0.228 / 0.283 / 0.191 de `v04_1`, y F1_ref (H) más bajo con 14 y 26 vistas. A diferencia del eje, incluso los objetos con predicción tienen baja precisión (P@10° sobre los predichos: 0.24 / 0.13 / 0.05).

**Conclusión**
- El Flujo C no es utilizable en su forma actual por la pérdida de cobertura. El resultado del eje sugiere que la idea de describir el objeto y dar una pista en una vista podría mejorar la calidad de los puntos, pero hace falta un diseño que garantice al menos dos puntos por vista antes de sacar conclusiones.

## 3. Comparación con el prompt base del Flujo A

Cada celda muestra *Flujo A → este experimento* para el mismo número de vistas. Fuente del Flujo A: `ndc_fix/ndcfix_flujoA`.

### Simetría axial
| Prompt (vs. base Flujo A) | n_views | AUC | Error mediano (°) | P@10° | Objetos |
|---|---|---|---|---|---|
| v05_1_flowC (vs. v05_1) | 6 | 0.208 → 0.316 | 54.0 → 90.0 | 0.107 → 0.318 | 804 → 374 |
| v05_1_flowC (vs. v05_1) | 14 | 0.185 → 0.214 | 56.8 → 90.0 | 0.100 → 0.221 | 793 → 244 |
| v05_1_flowC (vs. v05_1) | 26 | 0.133 → 0.125 | 58.3 → 90.0 | 0.043 → 0.102 | 831 → 168 |

### Simetría planar
| Prompt (vs. base Flujo A) | n_views | AUC | Error mediano (°) | P@10° | F1_ref (voraz) | F1_ref (H) | Objetos |
|---|---|---|---|---|---|---|---|
| v04_1_flowC (vs. v04_1) | 6 | 0.228 → 0.113 | 39.8 → 90.0 | 0.182 → 0.091 | 0.112 → 0.127 | 0.121 → 0.136 | 707 → 316 |
| v04_1_flowC (vs. v04_1) | 14 | 0.283 → 0.040 | 63.9 → 90.0 | 0.269 → 0.025 | 0.265 → 0.056 | 0.286 → 0.064 | 553 → 163 |
| v04_1_flowC (vs. v04_1) | 26 | 0.191 → 0.024 | 45.6 → 90.0 | 0.155 → 0.009 | 0.161 → 0.023 | 0.172 → 0.025 | 687 → 167 |

## 4. Mejor configuración (n_views) por prompt

| Prompt | Axis: mejor n_views | Axis: AUC | Plane: mejor n_views (AUC) | Plane: AUC |
|---|---|---|---|---|
| v04_1_flowC | – | – | 6 | 0.113 |
| v05_1_flowC | 6 | 0.316 | – | – |

## 5. Tablas por métrica y tipo de simetría

Columnas = número de vistas.

# Parte A — Simetría axial

#### AUC angular (trunc. 45°) (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v05_1_flowC | 0.316 | 0.214 | 0.125 | 0.218 |

#### Error angular medio (°) (↓ menor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v05_1_flowC | 58.2 | 68.8 | 76.5 | 67.8 |

#### Error angular mediano (°) (↓ menor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v05_1_flowC | 90.0 | 90.0 | 90.0 | 90.0 |

#### Precisión@5° (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v05_1_flowC | 0.253 | 0.167 | 0.047 | 0.156 |

#### Precisión@10° (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v05_1_flowC | 0.318 | 0.221 | 0.102 | 0.214 |

#### Precisión@15° (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v05_1_flowC | 0.334 | 0.228 | 0.132 | 0.231 |

#### Error de traslación normalizado (mediana) (↓ menor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v05_1_flowC | 0.017 | 0.014 | 0.035 | 0.022 |

#### SDE_ref medio (↓ menor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v05_1_flowC | 0.032 | 0.012 | 0.044 | 0.029 |

#### Objetos con predicción (de 850) (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v05_1_flowC | 374 | 244 | 168 | 262 |

# Parte B — Simetría planar

#### AUC angular (trunc. 45°) (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_flowC | 0.113 | 0.040 | 0.024 | 0.059 |

#### Error angular medio (°) (↓ menor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_flowC | 76.9 | 84.3 | 85.2 | 82.1 |

#### Error angular mediano (°) (↓ menor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_flowC | 90.0 | 90.0 | 90.0 | 90.0 |

#### Precisión@5° (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_flowC | 0.061 | 0.009 | 0.001 | 0.024 |

#### Precisión@10° (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_flowC | 0.091 | 0.025 | 0.009 | 0.042 |

#### Precisión@15° (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_flowC | 0.108 | 0.039 | 0.020 | 0.056 |

#### Error de traslación normalizado (mediana) (↓ menor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_flowC | 0.067 | 0.038 | 0.027 | 0.044 |

#### SDE_ref medio (↓ menor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_flowC | 0.032 | 0.056 | 0.033 | 0.040 |

#### Objetos con predicción (de 850) (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_flowC | 316 | 163 | 167 | 215 |

#### F1_ref (voraz, promedio de los 4 umbrales) (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_flowC | 0.127 | 0.056 | 0.023 | 0.069 |

#### F1_ref (voraz) @t=0.05 (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_flowC | 0.060 | 0.014 | 0.005 | 0.026 |

#### F1_ref (voraz) @t=0.10 (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_flowC | 0.110 | 0.042 | 0.005 | 0.052 |

#### F1_ref (voraz) @t=0.15 (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_flowC | 0.148 | 0.075 | 0.019 | 0.081 |

#### F1_ref (voraz) @t=0.20 (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_flowC | 0.191 | 0.094 | 0.063 | 0.116 |

#### F1_ref (Hungarian, promedio de los 4 umbrales) (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_flowC | 0.136 | 0.064 | 0.025 | 0.075 |

#### F1_ref (Hungarian) @t=0.05 (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_flowC | 0.064 | 0.016 | 0.005 | 0.029 |

#### F1_ref (Hungarian) @t=0.10 (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_flowC | 0.117 | 0.048 | 0.005 | 0.057 |

#### F1_ref (Hungarian) @t=0.15 (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_flowC | 0.159 | 0.085 | 0.021 | 0.089 |

#### F1_ref (Hungarian) @t=0.20 (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_flowC | 0.204 | 0.106 | 0.070 | 0.127 |
