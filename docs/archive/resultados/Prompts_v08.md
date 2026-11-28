# Resumen de resultados por prompt — `ndcfix_v08` (experiments_06_10_2026)

Prompts v08 (autoverificación contra el colapso al centro de la imagen).

Fuentes: `axis_sym_axis_v08_ndcfix_nomesh_comparison.csv`, `plane_sym_plane_v08_ndcfix_mp1_nomesh_comparison.csv`.
2 prompts × 3 configuraciones de vistas (6, 14, 26), método `triangulation` (pipeline sin malla, conversión NDC corregida, `n_total = 850` objetos por celda; plano en modo de un solo plano, `max_planes = 1`). En las tablas, **negrita** = mejor prompt según el promedio sobre las 3 configuraciones de vistas (solo cuando hay más de un prompt).

> **Cautelas de lectura**
> - `n_objects` es la cantidad de objetos con predicción válida (de 850). Las métricas angulares imputan 90° a los objetos sin predicción; la traslación, SDE_ref y F1_ref se calculan solo sobre los objetos con predicción. Compará siempre contra la tabla de cobertura.
> - `AUC_angular` trunca en 45°. `SDE_ref`/`F1_ref` no son intercambiables con `angular_error`/`translation_error` (ver `docs/metricas_evaluacion.md`).
> - Un error angular mediano ≈ 57–60° (axis) o ≈ 83–87° (plane) es compatible con nivel de azar / predicción trivial constante, no con aprendizaje de simetría.

## 1. Ranking global (promedio sobre 6/14/26 vistas, ordenado por AUC angular)

### Simetría axial
| Rank | Prompt | auc_angular | angular_error_median | precision_10deg | sde_ref_mean | n_objects |
|---|---|---|---|---|---|---|
| 1 | v08 | 0.550 | 8.4 | 0.516 | 0.020 | 823.7 |

### Simetría planar
| Rank | Prompt | auc_angular | angular_error_median | precision_10deg | f1_ref | f1_ref_hungarian | sde_ref_mean | n_objects |
|---|---|---|---|---|---|---|---|---|
| 1 | v08 | 0.144 | 82.8 | 0.116 | 0.068 | 0.071 | 0.041 | 843.0 |

## 2. Conclusiones principales

**Axis (`axis_sym`)**
- **`v08` es el mejor prompt de eje en precisión fina.** Con 6 vistas: AUC 0.599, error mediano 4.9°, P@5° 0.502 y P@10° 0.578. Ningún prompt del Flujo A llega a P@5° de 0.50 (el mejor, `v04` con 6 vistas, obtiene 0.404).
- En AUC promedio queda **empatado** con los mejores del Flujo A (`v08` 0.550, `v04` 0.554, `v06` 0.547), pero gana en las métricas exigentes en las tres configuraciones de vistas:
  - error mediano: 4.9 / 6.9 / 13.3° frente a 8.1 / 8.6 / 14.2° de `v06`;
  - P@5°: 0.502 / 0.418 / 0.172 frente a 0.394 / 0.326 / 0.119 de `v06`.
- Es decir, `v08` no amplía mucho la cantidad de objetos bien estimados, pero hace que los que se estiman bien queden más cerca del eje real.
- Mantiene buena cobertura (≈ 824 objetos) y un SDE_ref bajo (0.020). Como en el resto de los prompts, **más vistas empeoran** (AUC 0.599 → 0.554 → 0.498).

**Plane (`plane_sym`)**
- **La misma instrucción hunde el plano.** `plane_v08` queda con AUC 0.144 en promedio, error mediano ≈ 83° y F1_ref (H) 0.071, frente a 0.348 / 48.6° / 0.285 de `plane_v06` y 0.405 / 42.6° / 0.296 de `plane_v02`.
- Su patrón es el de los prompts que "predicen para todo pero mal" (cobertura ≈ 843 objetos, error mediano ≈ 80–87°), igual que los bilaterales en el Flujo A.
- Es consistente con el diagnóstico de la tesis: en plano, los buenos resultados están ligados a marcar la columna central de la imagen (con objetos en pose canónica), y una instrucción que aleja los puntos del centro los destruye.

**Comparación entre tipos de simetría**
- La autoverificación contra el colapso al centro tiene efectos opuestos: mejora la precisión fina del eje y destruye el plano. Para eje, `v08` es el candidato a mejor prompt; para plano, se deben seguir usando `v02` / `v05_1` / `v06`.

## 3. Comparación con el prompt base del Flujo A

Cada celda muestra *Flujo A → este experimento* para el mismo número de vistas. Fuente del Flujo A: `ndc_fix/ndcfix_flujoA`.

### Simetría axial
| Prompt (vs. base Flujo A) | n_views | AUC | Error mediano (°) | P@10° | Objetos |
|---|---|---|---|---|---|
| v08 (vs. v06) | 6 | 0.577 → 0.599 | 8.1 → 4.9 | 0.535 → 0.578 | 824 → 829 |
| v08 (vs. v06) | 14 | 0.563 → 0.554 | 8.6 → 6.9 | 0.533 → 0.561 | 812 → 813 |
| v08 (vs. v06) | 26 | 0.499 → 0.498 | 14.2 → 13.3 | 0.363 → 0.411 | 821 → 829 |
| v08 (vs. v04) | 6 | 0.582 → 0.599 | 8.6 → 4.9 | 0.527 → 0.578 | 831 → 829 |
| v08 (vs. v04) | 14 | 0.580 → 0.554 | 8.4 → 6.9 | 0.549 → 0.561 | 816 → 813 |
| v08 (vs. v04) | 26 | 0.501 → 0.498 | 14.8 → 13.3 | 0.339 → 0.411 | 806 → 829 |

### Simetría planar
| Prompt (vs. base Flujo A) | n_views | AUC | Error mediano (°) | P@10° | F1_ref (voraz) | F1_ref (H) | Objetos |
|---|---|---|---|---|---|---|---|
| v08 (vs. v06) | 6 | 0.371 → 0.240 | 39.8 → 86.6 | 0.361 → 0.219 | 0.219 → 0.129 | 0.237 → 0.135 | 730 → 841 |
| v08 (vs. v06) | 14 | 0.343 → 0.151 | 63.5 → 80.2 | 0.328 → 0.109 | 0.335 → 0.060 | 0.362 → 0.065 | 531 → 844 |
| v08 (vs. v06) | 26 | 0.331 → 0.042 | 42.4 → 81.5 | 0.283 → 0.020 | 0.236 → 0.013 | 0.257 → 0.014 | 822 → 844 |
| v08 (vs. v02) | 6 | 0.455 → 0.240 | 39.7 → 86.6 | 0.465 → 0.219 | 0.257 → 0.129 | 0.279 → 0.135 | 792 → 841 |
| v08 (vs. v02) | 14 | 0.391 → 0.151 | 46.6 → 80.2 | 0.371 → 0.109 | 0.301 → 0.060 | 0.327 → 0.065 | 658 → 844 |
| v08 (vs. v02) | 26 | 0.368 → 0.042 | 41.5 → 81.5 | 0.317 → 0.020 | 0.260 → 0.013 | 0.283 → 0.014 | 828 → 844 |

## 4. Mejor configuración (n_views) por prompt

| Prompt | Axis: mejor n_views | Axis: AUC | Plane: mejor n_views (AUC) | Plane: AUC |
|---|---|---|---|---|
| v08 | 6 | 0.599 | 6 | 0.240 |

## 5. Tablas por métrica y tipo de simetría

Columnas = número de vistas.

# Parte A — Simetría axial

#### AUC angular (trunc. 45°) (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v08 | 0.599 | 0.554 | 0.498 | 0.550 |

#### Error angular medio (°) (↓ menor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v08 | 28.0 | 31.3 | 33.0 | 30.8 |

#### Error angular mediano (°) (↓ menor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v08 | 4.9 | 6.9 | 13.3 | 8.4 |

#### Precisión@5° (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v08 | 0.502 | 0.418 | 0.172 | 0.364 |

#### Precisión@10° (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v08 | 0.578 | 0.561 | 0.411 | 0.516 |

#### Precisión@15° (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v08 | 0.625 | 0.585 | 0.527 | 0.579 |

#### Error de traslación normalizado (mediana) (↓ menor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v08 | 0.020 | 0.018 | 0.043 | 0.027 |

#### SDE_ref medio (↓ menor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v08 | 0.015 | 0.016 | 0.029 | 0.020 |

#### Objetos con predicción (de 850) (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v08 | 829 | 813 | 829 | 824 |

# Parte B — Simetría planar

#### AUC angular (trunc. 45°) (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v08 | 0.240 | 0.151 | 0.042 | 0.144 |

#### Error angular medio (°) (↓ menor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v08 | 62.9 | 64.7 | 75.3 | 67.6 |

#### Error angular mediano (°) (↓ menor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v08 | 86.6 | 80.2 | 81.5 | 82.8 |

#### Precisión@5° (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v08 | 0.189 | 0.054 | 0.017 | 0.087 |

#### Precisión@10° (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v08 | 0.219 | 0.109 | 0.020 | 0.116 |

#### Precisión@15° (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v08 | 0.234 | 0.133 | 0.032 | 0.133 |

#### Error de traslación normalizado (mediana) (↓ menor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v08 | 0.053 | 0.032 | 0.029 | 0.038 |

#### SDE_ref medio (↓ menor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v08 | 0.028 | 0.039 | 0.056 | 0.041 |

#### Objetos con predicción (de 850) (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v08 | 841 | 844 | 844 | 843 |

#### F1_ref (voraz, promedio de los 4 umbrales) (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v08 | 0.129 | 0.060 | 0.013 | 0.068 |

#### F1_ref (voraz) @t=0.05 (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v08 | 0.058 | 0.017 | 0.008 | 0.028 |

#### F1_ref (voraz) @t=0.10 (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v08 | 0.115 | 0.050 | 0.014 | 0.060 |

#### F1_ref (voraz) @t=0.15 (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v08 | 0.155 | 0.078 | 0.014 | 0.082 |

#### F1_ref (voraz) @t=0.20 (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v08 | 0.189 | 0.097 | 0.017 | 0.101 |

#### F1_ref (Hungarian, promedio de los 4 umbrales) (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v08 | 0.135 | 0.065 | 0.014 | 0.071 |

#### F1_ref (Hungarian) @t=0.05 (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v08 | 0.061 | 0.018 | 0.009 | 0.029 |

#### F1_ref (Hungarian) @t=0.10 (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v08 | 0.121 | 0.054 | 0.015 | 0.063 |

#### F1_ref (Hungarian) @t=0.15 (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v08 | 0.162 | 0.083 | 0.015 | 0.087 |

#### F1_ref (Hungarian) @t=0.20 (↑ mayor es mejor)

| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |
|---|---|---|---|---|
| v08 | 0.196 | 0.104 | 0.018 | 0.106 |
