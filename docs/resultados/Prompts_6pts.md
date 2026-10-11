# Resumen de resultados por prompt — `ndcfix_6pts` (experiments_06_10_2026)

Prompts de 6 puntos por vista (ronda posterior al Flujo A).

Fuentes: `axis_sym_axis_v06_6pts_ndcfix_nomesh_axis_v07_6pts_ndcfix_nomesh_comparison.csv`, `plane_sym_plane_v04_1_6pts_ndcfix_mp1_nomesh_plane_v05_6pts_ndcfix_mp1_nomesh_comparison.csv`.
4 prompts × 3 configuraciones de vistas (6, 14, 26), método `triangulation` (pipeline sin malla, conversión NDC corregida, `n_total = 850` objetos por celda; plano en modo de un solo plano, `max_planes = 1`). En las tablas, **negrita** = mejor prompt según el promedio sobre las 3 configuraciones de vistas (solo cuando hay más de un prompt).

> **Cautelas de lectura**
> - `n_objects` es la cantidad de objetos con predicción válida (de 850). Las métricas angulares imputan 90° a los objetos sin predicción; la traslación, SDE_ref y F1_ref se calculan solo sobre los objetos con predicción. Compará siempre contra la tabla de cobertura.
> - `AUC_angular` trunca en 45°. `SDE_ref`/`F1_ref` no son intercambiables con `angular_error`/`translation_error` (ver `docs/metricas_evaluacion.md`).
> - Un error angular mediano ≈ 57–60° (axis) o ≈ 83–87° (plane) es compatible con nivel de azar / predicción trivial constante, no con aprendizaje de simetría.

## 1. Ranking global (promedio sobre 6/14/26 vistas, ordenado por AUC angular)

### Simetría axial
| Rank | Prompt | auc_angular | angular_error_median | precision_10deg | sde_ref_mean | n_objects |
|---|---|---|---|---|---|---|
| 1 | v06_6pts | 0.214 | 53.5 | 0.101 | 0.036 | 761.7 |
| 2 | v07_6pts | 0.195 | 55.0 | 0.080 | 0.049 | 754.3 |

### Simetría planar
| Rank | Prompt | auc_angular | angular_error_median | precision_10deg | f1_ref | f1_ref_hungarian | sde_ref_mean | n_objects |
|---|---|---|---|---|---|---|---|---|
| 1 | v05_6pts | 0.299 | 49.6 | 0.267 | 0.176 | 0.191 | 0.017 | 766.7 |
| 2 | v04_1_6pts | 0.169 | 73.3 | 0.135 | 0.143 | 0.154 | 0.018 | 456.0 |

## 2. Conclusiones principales

**Axis (`axis_sym`)**
- **Pedir 6 puntos por vista empeora mucho el eje.** `v06_6pts` cae a AUC 0.214 en promedio frente a 0.547 de `v06` (Flujo A), con error mediano ≈ 54° frente a ≈ 10°: queda cerca del nivel de azar (mediana 60°, AUC 0.10). `v07_6pts` también queda por debajo de `v07` en AUC y precisión en las tres configuraciones.
- Una explicación compatible con el estimador: con más de dos puntos por vista, la triangulación de eje usa el par más separado (`widest_pair`), que con 6 puntos tiende a quedar en los extremos del objeto y no necesariamente sobre el eje. La hipótesis de que más puntos mejorarían el condicionamiento no se sostiene con la conversión NDC corregida.
- `v07_6pts` mejora la cobertura de `v07` (≈ 500 → ≈ 750 objetos), pero no la calidad.
- Como en el Flujo A, **más vistas empeoran**: el AUC cae con 26 vistas en ambos prompts.

**Plane (`plane_sym`)**
- **`v05_6pts` mejora claramente a `v05`**: AUC 0.154 → 0.299 en promedio (0.266 → 0.441 con 6 vistas), error mediano 83° → 50° y F1_ref (H) 0.082 → 0.191. Aun así queda por debajo de los mejores prompts de plano del Flujo A (`v02` 0.405, `v05_1` 0.384).
- **`v04_1_6pts` solo mejora con 6 vistas** (AUC 0.228 → 0.252, F1_ref (H) 0.121 → 0.174). Con 14 y 26 vistas su cobertura colapsa (425 y 294 objetos de 850): más de la mitad de los objetos queda sin predicción y el error mediano llega a 90° por la imputación.
- La mediana de 39.8° con 6 vistas se repite en varios prompts de plano (también en el Flujo A); es consistente con que muchos objetos reciben el mismo plano predicho con esa configuración de cámaras, no con un error que dependa del objeto.

**Comparación entre tipos de simetría**
- El efecto de los 6 puntos tiene signo opuesto: perjudica al eje (todos los prompts) y ayuda a `v05` en plano. Ninguna variante de 6 puntos supera al mejor prompt de 2 puntos de su tipo.

## 3. Comparación con el prompt base del Flujo A

Cada celda muestra *Flujo A → este experimento* para el mismo número de vistas. Fuente del Flujo A: `ndc_fix/ndcfix_flujoA`.

### Simetría axial
| Prompt (vs. base Flujo A) | n_views | AUC | Error mediano (°) | P@10° | Objetos |
|---|---|---|---|---|---|
| v06_6pts (vs. v06) | 6 | 0.577 → 0.247 | 8.1 → 48.5 | 0.535 → 0.121 | 824 → 797 |
| v06_6pts (vs. v06) | 14 | 0.563 → 0.249 | 8.6 → 56.1 | 0.533 → 0.148 | 812 → 682 |
| v06_6pts (vs. v06) | 26 | 0.499 → 0.147 | 14.2 → 55.9 | 0.363 → 0.034 | 821 → 806 |
| v07_6pts (vs. v07) | 6 | 0.280 → 0.234 | 69.4 → 49.4 | 0.236 → 0.108 | 502 → 797 |
| v07_6pts (vs. v07) | 14 | 0.299 → 0.206 | 70.8 → 56.6 | 0.258 → 0.095 | 493 → 735 |
| v07_6pts (vs. v07) | 26 | 0.194 → 0.146 | 75.4 → 58.9 | 0.122 → 0.035 | 500 → 731 |

### Simetría planar
| Prompt (vs. base Flujo A) | n_views | AUC | Error mediano (°) | P@10° | F1_ref (voraz) | F1_ref (H) | Objetos |
|---|---|---|---|---|---|---|---|
| v04_1_6pts (vs. v04_1) | 6 | 0.228 → 0.252 | 39.8 → 39.8 | 0.182 → 0.219 | 0.112 → 0.161 | 0.121 → 0.174 | 707 → 649 |
| v04_1_6pts (vs. v04_1) | 14 | 0.283 → 0.185 | 63.9 → 90.0 | 0.269 → 0.152 | 0.265 → 0.188 | 0.286 → 0.202 | 553 → 425 |
| v04_1_6pts (vs. v04_1) | 26 | 0.191 → 0.068 | 45.6 → 90.0 | 0.155 → 0.035 | 0.161 → 0.082 | 0.172 → 0.087 | 687 → 294 |
| v05_6pts (vs. v05) | 6 | 0.266 → 0.441 | 86.3 → 39.8 | 0.243 → 0.431 | 0.143 → 0.248 | 0.150 → 0.270 | 839 → 804 |
| v05_6pts (vs. v05) | 14 | 0.160 → 0.308 | 80.7 → 63.6 | 0.119 → 0.259 | 0.074 → 0.183 | 0.080 → 0.200 | 812 → 725 |
| v05_6pts (vs. v05) | 26 | 0.036 → 0.149 | 81.8 → 45.6 | 0.021 → 0.111 | 0.016 → 0.096 | 0.016 → 0.104 | 848 → 771 |

## 4. Mejor configuración (n_views) por prompt

| Prompt | Axis: mejor n_views | Axis: AUC | Plane: mejor n_views (AUC) | Plane: AUC |
|---|---|---|---|---|
| v04_1_6pts | – | – | 6 | 0.252 |
| v05_6pts | – | – | 6 | 0.441 |
| v06_6pts | 14 | 0.249 | – | – |
| v07_6pts | 6 | 0.234 | – | – |

## 5. Tablas por métrica y tipo de simetría

Columnas = número de vistas.

# Parte A — Simetría axial

#### AUC angular (trunc. 45°) (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| **v06_6pts** | 0.247 | 0.249 | 0.147 | **0.214** |
| v07_6pts | 0.234 | 0.206 | 0.146 | 0.195 |

#### Error angular medio (°) (↓ menor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| **v06_6pts** | 48.1 | 51.8 | 54.2 | **51.4** |
| v07_6pts | 48.8 | 53.0 | 56.6 | 52.8 |

#### Error angular mediano (°) (↓ menor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| **v06_6pts** | 48.5 | 56.1 | 55.9 | **53.5** |
| v07_6pts | 49.4 | 56.6 | 58.9 | 55.0 |

#### Precisión@5° (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| **v06_6pts** | 0.064 | 0.074 | 0.006 | **0.048** |
| v07_6pts | 0.052 | 0.040 | 0.007 | 0.033 |

#### Precisión@10° (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| **v06_6pts** | 0.121 | 0.148 | 0.034 | **0.101** |
| v07_6pts | 0.108 | 0.095 | 0.035 | 0.080 |

#### Precisión@15° (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| **v06_6pts** | 0.176 | 0.207 | 0.071 | **0.151** |
| v07_6pts | 0.173 | 0.140 | 0.071 | 0.128 |

#### Error de traslación normalizado (mediana) (↓ menor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v06_6pts | 0.237 | 0.105 | 0.143 | 0.162 |
| **v07_6pts** | 0.188 | 0.161 | 0.135 | **0.161** |

#### SDE_ref medio (↓ menor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| **v06_6pts** | 0.033 | 0.032 | 0.043 | **0.036** |
| v07_6pts | 0.045 | 0.051 | 0.051 | 0.049 |

#### Objetos con predicción (de 850) (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| **v06_6pts** | 797 | 682 | 806 | **762** |
| v07_6pts | 797 | 735 | 731 | 754 |

# Parte B — Simetría planar

#### AUC angular (trunc. 45°) (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_6pts | 0.252 | 0.185 | 0.068 | 0.169 |
| **v05_6pts** | 0.441 | 0.308 | 0.149 | **0.299** |

#### Error angular medio (°) (↓ menor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_6pts | 45.9 | 63.7 | 72.5 | 60.7 |
| **v05_6pts** | 30.3 | 45.7 | 44.9 | **40.3** |

#### Error angular mediano (°) (↓ menor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_6pts | 39.8 | 90.0 | 90.0 | 73.3 |
| **v05_6pts** | 39.8 | 63.6 | 45.6 | **49.6** |

#### Precisión@5° (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_6pts | 0.105 | 0.103 | 0.033 | 0.080 |
| **v05_6pts** | 0.201 | 0.212 | 0.105 | **0.173** |

#### Precisión@10° (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_6pts | 0.219 | 0.152 | 0.035 | 0.135 |
| **v05_6pts** | 0.431 | 0.259 | 0.111 | **0.267** |

#### Precisión@15° (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_6pts | 0.219 | 0.173 | 0.042 | 0.145 |
| **v05_6pts** | 0.434 | 0.269 | 0.122 | **0.275** |

#### Error de traslación normalizado (mediana) (↓ menor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_6pts | 0.001 | 0.003 | 0.001 | 0.001 |
| **v05_6pts** | 0.001 | 0.003 | 0.001 | **0.001** |

#### SDE_ref medio (↓ menor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_6pts | 0.018 | 0.014 | 0.022 | 0.018 |
| **v05_6pts** | 0.015 | 0.012 | 0.023 | **0.017** |

#### Objetos con predicción (de 850) (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_6pts | 649 | 425 | 294 | 456 |
| **v05_6pts** | 804 | 725 | 771 | **767** |

#### F1_ref (voraz, promedio de los 4 umbrales) (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_6pts | 0.161 | 0.188 | 0.082 | 0.143 |
| **v05_6pts** | 0.248 | 0.183 | 0.096 | **0.176** |

#### F1_ref (voraz) @t=0.05 (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| **v04_1_6pts** | 0.026 | 0.040 | 0.059 | **0.042** |
| v05_6pts | 0.031 | 0.011 | 0.078 | 0.040 |

#### F1_ref (voraz) @t=0.10 (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_6pts | 0.134 | 0.199 | 0.083 | 0.139 |
| **v05_6pts** | 0.205 | 0.221 | 0.099 | **0.175** |

#### F1_ref (voraz) @t=0.15 (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_6pts | 0.239 | 0.243 | 0.086 | 0.190 |
| **v05_6pts** | 0.374 | 0.248 | 0.101 | **0.241** |

#### F1_ref (voraz) @t=0.20 (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_6pts | 0.244 | 0.268 | 0.098 | 0.203 |
| **v05_6pts** | 0.384 | 0.252 | 0.106 | **0.247** |

#### F1_ref (Hungarian, promedio de los 4 umbrales) (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_6pts | 0.174 | 0.202 | 0.087 | 0.154 |
| **v05_6pts** | 0.270 | 0.200 | 0.104 | **0.191** |

#### F1_ref (Hungarian) @t=0.05 (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| **v04_1_6pts** | 0.028 | 0.043 | 0.064 | **0.045** |
| v05_6pts | 0.034 | 0.013 | 0.085 | 0.044 |

#### F1_ref (Hungarian) @t=0.10 (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_6pts | 0.145 | 0.215 | 0.089 | 0.149 |
| **v05_6pts** | 0.223 | 0.242 | 0.107 | **0.191** |

#### F1_ref (Hungarian) @t=0.15 (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_6pts | 0.258 | 0.263 | 0.092 | 0.204 |
| **v05_6pts** | 0.407 | 0.271 | 0.110 | **0.262** |

#### F1_ref (Hungarian) @t=0.20 (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v04_1_6pts | 0.264 | 0.288 | 0.105 | 0.219 |
| **v05_6pts** | 0.417 | 0.276 | 0.114 | **0.269** |
