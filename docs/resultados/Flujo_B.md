# Resumen de resultados por prompt — `ndcfix_flujoB` (experiments_06_10_2026)

Flujo B (descripción previa del objeto + prompt de pointing).

Fuentes: `axis_sym_axis_v05_1_flowB_ndcfix_nomesh_comparison.csv`, `plane_sym_plane_v04_1_flowB_ndcfix_mp1_nomesh_comparison.csv`.
2 prompts × 3 configuraciones de vistas (6, 14, 26), método `triangulation` (pipeline sin malla, conversión NDC corregida, `n_total = 850` objetos por celda; plano en modo de un solo plano, `max_planes = 1`). En las tablas, **negrita** = mejor prompt según el promedio sobre las 3 configuraciones de vistas (solo cuando hay más de un prompt).

> **Cautelas de lectura**
> - `n_objects` es la cantidad de objetos con predicción válida (de 850). Las métricas angulares imputan 90° a los objetos sin predicción; la traslación, SDE_ref y F1_ref se calculan solo sobre los objetos con predicción. Compará siempre contra la tabla de cobertura.
> - `AUC_angular` trunca en 45°. `SDE_ref`/`F1_ref` no son intercambiables con `angular_error`/`translation_error` (ver `docs/metricas_evaluacion.md`).
> - Un error angular mediano ≈ 57–60° (axis) o ≈ 83–87° (plane) es compatible con nivel de azar / predicción trivial constante, no con aprendizaje de simetría.

## 1. Ranking global (promedio sobre 6/14/26 vistas, ordenado por AUC angular)

### Simetría axial
| Rank | Prompt | auc_angular | angular_error_median | precision_10deg | sde_ref_mean | n_objects |
|---|---|---|---|---|---|---|
| 1 | v05_1_flowB | 0.175 | 56.0 | 0.080 | 0.028 | 821.3 |

### Simetría planar
| Rank | Prompt | auc_angular | angular_error_median | precision_10deg | f1_ref | f1_ref_hungarian | sde_ref_mean | n_objects |
|---|---|---|---|---|---|---|---|---|
| 1 | v04_1_flowB | 0.238 | 49.6 | 0.205 | 0.170 | 0.184 | 0.017 | 667.0 |

## 2. Conclusiones principales

**Axis (`axis_sym`)**
- **La descripción previa no cambia el resultado del eje.** `v05_1_flowB` queda prácticamente igual a `v05_1` del Flujo A en las tres configuraciones (AUC 0.222 / 0.178 / 0.125 frente a 0.208 / 0.185 / 0.133; diferencias ≤ 0.015) y sigue con error mediano ≈ 56°, es decir, cerca del nivel de azar.
- Hay que tener en cuenta que el Flujo B se corrió sobre `v05_1`, que con la conversión NDC corregida es uno de los peores prompts de eje (rank 14 de 14 en el Flujo A). El experimento no dice si la descripción ayudaría a un prompt bueno como `v04` o `v06`.
- La cobertura sube levemente (≈ 809 → ≈ 821 objetos en promedio), sin efecto en la calidad.

**Plane (`plane_sym`)**
- **Efecto pequeño y no consistente.** Respecto de `v04_1`, el AUC sube con 6 y 14 vistas (0.228 → 0.252; 0.283 → 0.303) y baja con 26 (0.191 → 0.158). F1_ref (H) sube con 6 vistas (0.121 → 0.152), queda igual con 14 (0.286 → 0.280) y baja con 26 (0.172 → 0.119).
- El promedio (AUC 0.238) es casi idéntico al de `v04_1` (0.234) y queda lejos de los mejores prompts de plano del Flujo A (`v02` 0.405).

**Conclusión**
- Agregar una descripción semántica del objeto al prompt no mejora de forma consistente ninguno de los dos tipos de simetría. Este resultado coincide con lo que se observaba antes de la corrección NDC.

## 3. Comparación con el prompt base del Flujo A

Cada celda muestra *Flujo A → este experimento* para el mismo número de vistas. Fuente del Flujo A: `ndc_fix/ndcfix_flujoA`.

### Simetría axial
| Prompt (vs. base Flujo A) | n_views | AUC | Error mediano (°) | P@10° | Objetos |
|---|---|---|---|---|---|
| v05_1_flowB (vs. v05_1) | 6 | 0.208 → 0.222 | 54.0 → 52.5 | 0.107 → 0.117 | 804 → 809 |
| v05_1_flowB (vs. v05_1) | 14 | 0.185 → 0.178 | 56.8 → 56.7 | 0.100 → 0.089 | 793 → 817 |
| v05_1_flowB (vs. v05_1) | 26 | 0.133 → 0.125 | 58.3 → 58.8 | 0.043 → 0.033 | 831 → 838 |

### Simetría planar
| Prompt (vs. base Flujo A) | n_views | AUC | Error mediano (°) | P@10° | F1_ref (voraz) | F1_ref (H) | Objetos |
|---|---|---|---|---|---|---|---|
| v04_1_flowB (vs. v04_1) | 6 | 0.228 → 0.252 | 39.8 → 39.8 | 0.182 → 0.218 | 0.112 → 0.141 | 0.121 → 0.152 | 707 → 668 |
| v04_1_flowB (vs. v04_1) | 14 | 0.283 → 0.303 | 63.9 → 63.9 | 0.269 → 0.277 | 0.265 → 0.258 | 0.286 → 0.280 | 553 → 578 |
| v04_1_flowB (vs. v04_1) | 26 | 0.191 → 0.158 | 45.6 → 45.2 | 0.155 → 0.121 | 0.161 → 0.110 | 0.172 → 0.119 | 687 → 755 |

## 4. Mejor configuración (n_views) por prompt

| Prompt | Axis: mejor n_views | Axis: AUC | Plane: mejor n_views (AUC) | Plane: AUC |
|---|---|---|---|---|
| v04_1_flowB | – | – | 14 | 0.303 |
| v05_1_flowB | 6 | 0.222 | – | – |

## 5. Tablas por métrica y tipo de simetría

Columnas = número de vistas.

# Parte A — Simetría axial

#### AUC angular (trunc. 45°) (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v05_1_flowB | 0.222 | 0.178 | 0.125 | 0.175 |

#### Error angular medio (°) (↓ menor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v05_1_flowB | 49.7 | 53.0 | 55.9 | 52.9 |

#### Error angular mediano (°) (↓ menor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v05_1_flowB | 52.5 | 56.7 | 58.8 | 56.0 |

#### Precisión@5° (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v05_1_flowB | 0.062 | 0.046 | 0.011 | 0.040 |

#### Precisión@10° (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v05_1_flowB | 0.117 | 0.089 | 0.033 | 0.080 |

#### Precisión@15° (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v05_1_flowB | 0.174 | 0.123 | 0.058 | 0.118 |

#### Error de traslación normalizado (mediana) (↓ menor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v05_1_flowB | 0.273 | 0.297 | 0.106 | 0.225 |

#### SDE_ref medio (↓ menor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v05_1_flowB | 0.025 | 0.028 | 0.031 | 0.028 |

#### Objetos con predicción (de 850) (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v05_1_flowB | 809 | 817 | 838 | 821 |

# Parte B — Simetría planar

#### AUC angular (trunc. 45°) (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_flowB | 0.252 | 0.303 | 0.158 | 0.238 |

#### Error angular medio (°) (↓ menor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_flowB | 45.1 | 51.3 | 44.6 | 47.0 |

#### Error angular mediano (°) (↓ menor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_flowB | 39.8 | 63.9 | 45.2 | 49.6 |

#### Precisión@5° (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_flowB | 0.058 | 0.236 | 0.119 | 0.138 |

#### Precisión@10° (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_flowB | 0.218 | 0.277 | 0.121 | 0.205 |

#### Precisión@15° (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_flowB | 0.225 | 0.299 | 0.126 | 0.216 |

#### Error de traslación normalizado (mediana) (↓ menor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_flowB | 0.001 | 0.002 | 0.001 | 0.001 |

#### SDE_ref medio (↓ menor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_flowB | 0.019 | 0.011 | 0.022 | 0.017 |

#### Objetos con predicción (de 850) (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_flowB | 668 | 578 | 755 | 667 |

#### F1_ref (voraz, promedio de los 4 umbrales) (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_flowB | 0.141 | 0.258 | 0.110 | 0.170 |

#### F1_ref (voraz) @t=0.05 (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_flowB | 0.011 | 0.031 | 0.091 | 0.044 |

#### F1_ref (voraz) @t=0.10 (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_flowB | 0.082 | 0.314 | 0.115 | 0.170 |

#### F1_ref (voraz) @t=0.15 (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_flowB | 0.234 | 0.335 | 0.116 | 0.228 |

#### F1_ref (voraz) @t=0.20 (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_flowB | 0.237 | 0.351 | 0.117 | 0.235 |

#### F1_ref (Hungarian, promedio de los 4 umbrales) (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_flowB | 0.152 | 0.280 | 0.119 | 0.184 |

#### F1_ref (Hungarian) @t=0.05 (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_flowB | 0.012 | 0.033 | 0.099 | 0.048 |

#### F1_ref (Hungarian) @t=0.10 (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_flowB | 0.088 | 0.342 | 0.124 | 0.185 |

#### F1_ref (Hungarian) @t=0.15 (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_flowB | 0.252 | 0.364 | 0.126 | 0.247 |

#### F1_ref (Hungarian) @t=0.20 (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_flowB | 0.256 | 0.381 | 0.127 | 0.255 |
