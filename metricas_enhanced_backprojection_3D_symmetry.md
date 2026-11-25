# Métricas — *Enhanced Back-Projection of Vision Features for 3D Symmetry Detection*

**Autores:** Isaac Aguirre, Ivan Sipiran (Departamento de Ciencias de la Computación, Universidad de Chile)
**Venue:** IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2026 — versión Open Access de la CVF
**Código:** https://github.com/Spulp/EnhancedBackProjection

> **Cómo leer este documento**
> - Los valores de las **Tablas 1, 2 y 3** son **exactos**, copiados del paper.
> - Los valores de la **Figura 3** (invariancia *E*) **no aparecen numéricamente en el paper**; aquí se dan **aproximados (≈)**, leídos visualmente de las gráficas (precisión estimada ±1 unidad en RM/PC y ±0.3 en FM-planos). El orden de las curvas en cada leyenda (de peor a mejor) sí es exacto, porque lo indica el paper.
> - Las columnas marcadas como **"derivado"** son cálculos propios (diferencias, porcentajes, ratios) a partir de los valores exactos; no aparecen en el paper.

---

## Índice

1. [Resumen de resultados clave](#1-resumen-de-resultados-clave)
2. [Glosario de configuraciones y notación](#2-glosario-de-configuraciones-y-notación)
3. [Definición de las métricas](#3-definición-de-las-métricas)
4. [Datasets y protocolo de evaluación](#4-datasets-y-protocolo-de-evaluación)
5. [Invariancia de features (Figura 3)](#5-invariancia-de-features-figura-3)
6. [Tabla 1 — Back-projection regular vs. mejorado (6 y 114 vistas)](#6-tabla-1--back-projection-regular-vs-mejorado)
7. [Tabla 2 — Robustez ante ruido y eliminación de partes](#7-tabla-2--robustez-ante-ruido-y-eliminación-de-partes)
8. [Tabla 3 — Comparación con el estado del arte (planos)](#8-tabla-3--comparación-con-el-estado-del-arte-planos)
9. [Efecto del tamaño de grupo L (ejes)](#9-efecto-del-tamaño-de-grupo-l-ejes)
10. [Tiempo de ejecución y memoria](#10-tiempo-de-ejecución-y-memoria)
11. [Hiperparámetros y umbrales del método](#11-hiperparámetros-y-umbrales-del-método)
12. [Observaciones y advertencias para interpretar las métricas](#12-observaciones-y-advertencias)

---

## 1. Resumen de resultados clave

| Resultado | Valor | Fuente |
|---|---|---|
| Mejor F-score en **planos** (FM10K-4R-FI, 114 vistas) | **0.912** | Tablas 1, 2, 3 |
| Mejor SDE en **planos** (FM10K-4R-FI, 114 vistas) | **0.282 ×10⁻⁴** | Tablas 1, 2, 3 |
| Mejor método previo (E3Sym) | F-score 0.753 / SDE 0.46 ×10⁻⁴ | Tabla 3 |
| Mejora sobre E3Sym *(derivado)* | **+0.159 F-score (+21.1 % relativo)**; SDE **−38.7 %** | — |
| Mejor F-score en **ejes** (RM-4R-FI-L50, 114 vistas) | **0.803** | Tablas 1, 2 |
| SDE en ejes (misma config.) | 231.9 ×10⁻⁴ | Tablas 1, 2 |
| Planos con ruido 1 % / 3 % / 5 % | F-score 0.906 / 0.888 / 0.838 | Tabla 2 |
| Planos con 20 % de eliminación de partes | F-score 0.720 | Tabla 2 |
| Ejes con ruido 1 % / 3 % / 5 % | F-score 0.301 / 0.223 / 0.182 | Tabla 2 |
| Ejes con 20 % de eliminación de partes | F-score 0.581 | Tabla 2 |
| Tiempo por objeto (planos, FM10K-4R-FI) | 6.318 s (6 vistas) / 20.03 s (114 vistas) | Tabla 1 |
| Tiempo por objeto (ejes, RM-4R-FI-L50) | 40.01 s (6 vistas) / 48.00 s (114 vistas) | Tabla 1 |
| Memoria GPU | Cabe en una sola GPU de **8 GB** | Sec. 9.3, 10 |
| Backbone | DINOv2 *small* con registros (**84 MB**), features de **384 dim** | Sec. 3 |

---

## 2. Glosario de configuraciones y notación

Los nombres de configuración combinan **representación + tamaño de muestreo + rotaciones + muestreo de vistas (+ tamaño de grupo L para ejes)**. Ejemplo: `FM10K-4R-FI` = Feature-Mesh Sampling con 10 000 puntos, 4 rotaciones de imagen, muestreo de Fibonacci.

| Sigla | Significado |
|---|---|
| **RM** | *Raw-Mesh*: la malla original; cada vértice recibe un vector de features ∈ ℝ³⁸⁴. |
| **PC** | *Point Cloud*: nube de puntos (de entrada o remuestreada de la malla), S_PC puntos. Independiente de la topología. |
| **FM** | *Feature-Mesh Sampling*: S_FMS puntos muestreados sobre la superficie; su feature se interpola baricéntricamente: fᵢ = α·f_va + β·f_vb + γ·f_vc. |
| **1K, 3K, 5K, 7K, 10K** | Tamaño de muestreo para PC/FM (1K = 1 000 puntos). |
| **RE** | Muestreo de vistas *regular* latitud–longitud (estándar; densidad desigual cerca de los polos). |
| **FI** | Muestreo de vistas de **Fibonacci** (distribución más uniforme sobre la esfera). |
| **1R** | Sin rotación: 0°. |
| **2R** | Rotaciones de imagen: 0°, 180°. |
| **3R** | Rotaciones: 0°, 90°, 270°. |
| **4R** | Rotaciones: 0°, 90°, 180°, 270°. |
| **4T** | *Control*: la misma imagen repetida 4 veces (sin rotar). |
| **RAND** | *Control*: emparejamiento aleatorio de puntos (en lugar del emparejamiento por simetría). |
| **V** | Número de vistas renderizadas. Valores probados: {6, 14, 26, 42, 62, 86, 114}. |
| **L** | Nº de vecinos más cercanos en espacio de features usados para formar grupos en la detección de ejes (grupos de tamaño L+1). |
| **S** | Evaluación estándar (sin perturbación). |
| **N1 %, N3 %, N5 %** | Ruido gaussiano con ε = 0.01, 0.03, 0.05 · R_max. |
| **PR20 %** | *Partial removal*: se eliminan los puntos dentro de una esfera de radio 0.2 · R_max. |

---

## 3. Definición de las métricas

### 3.1 Invariancia de features — *E(O)* (más bajo = mejor)

Mide cuán parecidos son los descriptores de un punto y de su contraparte simétrica. Idealmente, si T es la transformación de simetría (reflexión respecto a un plano o rotación respecto a un eje):

$$F(x) \approx F(T(x)) \quad \forall x \in O$$

Se cuantifica con la discrepancia L1 media por punto:

$$\mathcal{E}(O) = \frac{1}{|O|}\sum_{x \in O} \| F(x) - F(\bar{x}) \|_1, \qquad \bar{x} = \arg\min_{y \in O} \|y - T(x)\|_2$$

- x̄ es el punto de la superficie más cercano al transformado T(x).
- Con varias simetrías del mismo tipo, se promedia sobre todas: todos los planos (planos), o rotaciones de 45°, 90°, 135° y 180° (ejes).
- Se calcula sobre el **10 %** de los datasets (**85 objetos con planos + 85 con ejes**) y se promedia para obtener un único valor por configuración.
- **Unidades:** distancia L1 entre vectores de 384 dimensiones (sin normalizar). **No es comparable entre representaciones distintas** (RM vs. FM vs. PC); el paper indica que solo deben compararse configuraciones del mismo tipo de feature.

### 3.2 Symmetry Distance Error — SDE (más bajo = mejor)

Mide la fidelidad geométrica de cada simetría **reportada**:

$$\mathrm{SDE}(O, T) = \frac{1}{|O|}\sum_{o \in O} \min_{\hat{o} \in T(O)} \|o - \hat{o}\|_2^2$$

- Se usan **1 000 muestras aleatorias** de la superficie por objeto.
- Se reporta en unidades de **×10⁻⁴**.
- **Importante:** las detecciones **faltantes no afectan** al SDE; solo mide la calidad de lo que se detecta.

### 3.3 F-score (más alto = mejor)

1. Toda simetría (plano o eje) se convierte de un par punto–normal (p, n) a un 4-vector homogéneo **P = [nᵀ, −pᵀn]**.
2. Distancia invariante al signo (resuelve la ambigüedad de dirección de la normal):
   $$\mathrm{dist}(P, Q) = \min(\|P - Q\|_2,\ \|P + Q\|_2)$$
3. Solo se conservan las predicciones con **confianza ≥ 0.95**; los duplicados se eliminan con esta distancia.
4. Cada predicción se empareja con un ground truth no emparejado si dist(T, T̂) < τ, con **τ ∈ {0.05, 0.10, 0.15, 0.20}**.
5. Emparejadas → TP; predicciones sin pareja → FP; ground truths sin pareja → FN.

$$PR = \frac{TP}{TP + FP}, \qquad RE = \frac{TP}{TP + FN}, \qquad F = \frac{2 \cdot PR \cdot RE}{PR + RE}$$

- Los valores finales (F-score y SDE) se **promedian sobre todos los planos/ejes evaluados** del dataset.
- El paper no detalla si el F-score reportado es el promedio sobre los cuatro τ o un τ concreto; solo enumera los cuatro umbrales.

### 3.4 Distancia de Chamfer (usada en el algoritmo y para construir el dataset)

$$h(P, P') = \frac{1}{|P|}\sum_{p \in P}\min_{p' \in P'}\|p - p'\|_2^2, \qquad d_{\text{Chamfer}}(P, P') = \tfrac{1}{2}\big(h(P,P') + h(P',P)\big)$$

Reflexión de un punto respecto a un plano (n, c): p′ = p − 2(n · (p − c)) n.

### 3.5 Tiempo y memoria

- **Tiempo:** segundos por objeto, de extremo a extremo (extracción de features → cálculo de planos/ejes).
- **Memoria:** GPU de 8 GB (RTX 3060 Ti / RTX 2080).

---

## 4. Datasets y protocolo de evaluación

| Aspecto | Detalle |
|---|---|
| Origen | Mallas de varias clases de **ShapeNet**, normalizadas y centradas |
| Tamaño | **850 mallas con planos** + **850 mallas con ejes** (M = 850 cada uno) |
| Candidatos de simetría | Planos YZ, XZ, XY; ejes X, Y, Z |
| Criterio de selección | Distancia de Chamfer entre la malla y su versión reflejada/rotada; se eligen las 850 con menor distancia y se toma el plano/eje canónico como ground truth |
| Sesgo de orientación | Se aplican rotaciones arbitrarias a cada malla |
| Subconjunto para invariancia y tiempos | 10 % → 85 objetos (planos) + 85 (ejes) |
| Tamaño medio de ese subconjunto | **4 424** vértices/puntos (planos) y **16 177** (ejes) |
| Datasets perturbados | **8** adicionales (4 para planos, 4 para ejes): N1 %, N3 %, N5 %, PR20 % |
| Ruido | v′ = v + ε·n, n ~ N(0, I), ε ∈ {0.01, 0.03, 0.05}·R_max, con R_max = max‖p‖₂ |
| Eliminación de partes | Esfera de radio 0.2·R_max alrededor de un punto aleatorio; semilla fija |

---

## 5. Invariancia de features (Figura 3)

> Todos los valores de esta sección son **aproximados (≈)**, leídos de las gráficas. Más bajo = más invariante. Los experimentos se hicieron de forma secuencial: cada panel reduce las configuraciones en función del resultado del anterior.

**Conclusiones textuales del paper (Sec. 5.3):**
- **4R-FI** es consistentemente la mejor combinación.
- Para **PC**, **1K** puntos supera a 10K.
- Para **FM**, **10K** es el óptimo.
- Más vistas reducen *E* en todas las configuraciones, con rendimientos decrecientes.

### 5.1 Planos — Raw-Mesh (Fig. 3a)

Orden de la leyenda (peor → mejor): RM-RAND, RM-4T, RM-1R-RE, RM-1R-FI, RM-2R-RE, RM-2R-FI, RM-3R-RE, RM-3R-FI, RM-4R-RE, **RM-4R-FI**.

| Config. | V=6 | 14 | 26 | 42 | 62 | 86 | 114 |
|---|---|---|---|---|---|---|---|
| RM-RAND (control) | ≈121 | ≈106 | ≈97 | ≈91 | ≈86 | ≈82 | ≈79 |
| RM-4T (control) | ≈113 | ≈94 | ≈83 | ≈75 | ≈69 | ≈64 | ≈60 |
| RM-1R-RE | ≈113 | ≈94 | ≈83 | ≈75 | ≈69 | ≈64 | ≈60 |
| RM-1R-FI | ≈114 | ≈94 | ≈83 | ≈75 | ≈69 | ≈64 | ≈60 |
| RM-2R-RE | ≈97 | ≈79 | ≈69 | ≈62 | ≈56 | ≈51 | ≈47 |
| RM-2R-FI | ≈98 | ≈79 | ≈68 | ≈60.5 | ≈55 | ≈50 | ≈46 |
| RM-3R-RE | ≈86 | ≈71 | ≈62.5 | ≈56 | ≈51 | ≈47 | ≈43.5 |
| RM-3R-FI | ≈89 | ≈71.5 | ≈62 | ≈55.5 | ≈50 | ≈46 | ≈43 |
| RM-4R-RE | ≈81 | ≈66.5 | ≈58 | ≈52 | ≈47 | ≈43 | ≈40 |
| **RM-4R-FI** | ≈83 | ≈66.5 | ≈57.5 | ≈51 | ≈46.5 | ≈42 | **≈39** |

Lecturas clave:
- **RM-4T ≈ RM-1R**: repetir la misma imagen 4 veces no aporta nada; la mejora viene de **rotar** la imagen.
- Pasar de 1R a 4R reduce *E* en ≈35 % (≈60 → ≈39 a 114 vistas).
- Emparejamiento por simetría vs. aleatorio (RAND): a 114 vistas, ≈39 vs. ≈79, la mitad aproximadamente.

### 5.2 Planos — Feature-Mesh Sampling (Fig. 3b)

Orden de la leyenda (peor → mejor): FM1K-4R-RE, FM1K-4R-FI, FM3K-4R-RE, FM5K-4R-RE, FM7K-4R-RE, FM3K-4R-FI, FM10K-4R-RE, FM5K-4R-FI, FM7K-4R-FI, **FM10K-4R-FI**.

| Config. | V=6 | 14 | 26 | 42 | 62 | 86 | 114 |
|---|---|---|---|---|---|---|---|
| FM1K-4R-RE (peor) | ≈33.4 | ≈22.4 | ≈18.1 | ≈15.8 | ≈14.7 | ≈13.9 | ≈13.4 |
| FM1K-4R-FI | ≈34.0 | ≈22.4 | ≈17.7 | ≈15.3 | ≈14.0 | ≈13.2 | ≈12.7 |
| FM3K-4R-RE | — | — | — | ≈14.6 | ≈13.4 | ≈12.6 | ≈12.1 |
| **FM10K-4R-FI** (mejor) | ≈32.0 | ≈20.6 | ≈15.7 | ≈13.3 | ≈11.9 | ≈11.1 | **≈10.6** |

Las configuraciones intermedias quedan entre ≈11 y ≈12 a 114 vistas; todas parten de ≈31–34 con 6 vistas.

### 5.3 Planos — Point Cloud (Fig. 3c)

Orden de la leyenda (peor → mejor): PC10K-4R-FI, PC7K-4R-FI, PC5K-4R-FI, PC3K-4R-FI, **PC1K-4R-FI**.

| Config. | V=6 | 14 | 26 | 42 | 62 | 86 | 114 |
|---|---|---|---|---|---|---|---|
| PC10K-4R-FI | ≈91 | ≈92.5 | ≈72 | ≈60 | ≈52 | ≈47 | ≈43.5 |
| PC7K-4R-FI | ≈75 | ≈81 | ≈61.5 | ≈51 | ≈43.5 | ≈39 | ≈36 |
| PC5K-4R-FI | ≈63.5 | ≈69 | ≈52 | ≈42.5 | ≈36.5 | ≈32.5 | ≈30 |
| PC3K-4R-FI | ≈51.5 | ≈52 | ≈38.5 | ≈31 | ≈27 | ≈24 | ≈22.5 |
| **PC1K-4R-FI** | ≈45.5 | ≈30 | ≈22.5 | ≈18.5 | ≈16 | ≈14.5 | **≈13.5** |

En PC, *E* **empeora** al aumentar la densidad de muestreo, y de 6 a 14 vistas sube ligeramente para 3K–10K (comportamiento no monótono).

### 5.4 Ejes — Raw-Mesh (Fig. 3d)

| Config. | V=6 | 14 | 26 | 42 | 62 | 86 | 114 |
|---|---|---|---|---|---|---|---|
| RM-1R-RE | ≈152 | ≈124 | ≈112 | ≈104.5 | ≈99.5 | ≈95 | ≈92 |
| **RM-4R-FI** | ≈121 | ≈97 | ≈86 | ≈79 | ≈75 | ≈71 | **≈68** |

Mejora de ≈26 % con 4R-FI a 114 vistas.

### 5.5 Ejes — Feature-Mesh Sampling (Fig. 3e)

Orden (peor → mejor): FM1K-1R-RE, FM10K-1R-RE, FM1K-4R-FI, **FM10K-4R-FI**.

| Config. | V=6 | 14 | 26 | 42 | 62 | 86 | 114 |
|---|---|---|---|---|---|---|---|
| FM1K-1R-RE | ≈119.5 | ≈84.5 | ≈70 | ≈62.5 | ≈58.5 | ≈56 | ≈54.5 |
| FM10K-1R-RE | ≈119.5 | ≈83.5 | ≈69 | ≈61.5 | ≈57.5 | ≈55.3 | ≈53.5 |
| FM1K-4R-FI | ≈94 | ≈61.5 | ≈49 | ≈42.5 | ≈39 | ≈37 | ≈36 |
| **FM10K-4R-FI** | ≈94 | ≈61 | ≈48.5 | ≈42 | ≈38.5 | ≈36.5 | **≈35.3** |

En FM-ejes, el factor dominante es 4R-FI; el tamaño de muestreo (1K vs. 10K) apenas influye.

### 5.6 Ejes — Point Cloud (Fig. 3f)

Orden (peor → mejor): PC10K-1R-RE, PC10K-4R-FI, PC1K-1R-RE, **PC1K-4R-FI**.

| Config. | V=6 | 14 | 26 | 42 | 62 | 86 | 114 |
|---|---|---|---|---|---|---|---|
| PC10K-1R-RE | ≈149 | ≈133 | ≈113 | ≈99 | ≈89.5 | ≈82.5 | ≈77.5 |
| PC10K-4R-FI | ≈112 | ≈102 | ≈83 | ≈71 | ≈62.5 | ≈57 | ≈52 |
| PC1K-1R-RE | ≈86 | ≈62 | ≈48.5 | ≈41 | ≈36 | ≈33 | ≈31 |
| **PC1K-4R-FI** | ≈61 | ≈42.5 | ≈32.5 | ≈27 | ≈23 | ≈21 | **≈19.5** |

> **Ojo:** aunque PC y FM muestran *E* numéricamente más bajos que RM, el paper advierte que **PC no refleja de forma fiable la invariancia** (le falta la información de malla que RM y FM aprovechan en el back-projection), y que en ejes **RM es la única representación competitiva** en la detección final.

---

## 6. Tabla 1 — Back-projection regular vs. mejorado

Valores exactos. Formato de cada celda: **6 vistas / 114 vistas**. FM y PC usan 10K puntos; ejes usan L = 50.

| Dataset | Config. | F-score 1R-RE | SDE (×10⁻⁴) 1R-RE | Tiempo (s) 1R-RE | F-score 4R-FI | SDE (×10⁻⁴) 4R-FI | Tiempo (s) 4R-FI |
|---|---|---|---|---|---|---|---|
| Planos | RM | 0.865 / 0.866 | 1.819 / 1.781 | 6.643 / 13.22 | 0.877 / 0.887 | 1.789 / 1.690 | 7.164 / 20.91 |
| Planos | PC10K | 0.748 / 0.783 | 7.416 / 9.565 | 6.315 / 8.065 | 0.761 / 0.803 | 9.445 / 9.363 | 6.584 / 12.42 |
| Planos | **FM10K** | 0.909 / 0.910 | 0.303 / 0.289 | 6.106 / 12.49 | 0.906 / **0.912** | 0.291 / **0.282** | 6.318 / 20.03 |
| Ejes | RM-L50 | 0.616 / 0.756 | 350.8 / 239.1 | 77.23 / 64.37 | 0.598 / **0.803** | 357.4 / **231.9** | 40.01 / 48.00 |

*1R-RE = back-projection regular; 4R-FI = back-projection mejorado.*

### 6.1 Efecto del back-projection mejorado (4R-FI − 1R-RE) — *derivado*

| Dataset | Config. | ΔF-score (6 v.) | ΔF-score (114 v.) | ΔSDE (6 v.) | ΔSDE (114 v.) | Tiempo 4R-FI / 1R-RE (6 v.) | Tiempo 4R-FI / 1R-RE (114 v.) |
|---|---|---|---|---|---|---|---|
| Planos | RM | +0.012 | +0.021 | −1.6 % | −5.1 % | ×1.08 | ×1.58 |
| Planos | PC10K | +0.013 | +0.020 | +27.4 % | −2.1 % | ×1.04 | ×1.54 |
| Planos | FM10K | −0.003 | +0.002 | −4.0 % | −2.4 % | ×1.03 | ×1.60 |
| Ejes | RM-L50 | −0.018 | **+0.047** | +1.9 % | −3.0 % | ×0.52 | ×0.75 |

### 6.2 Efecto del número de vistas (114 − 6) — *derivado*

| Dataset | Config. | ΔF-score 1R-RE | ΔF-score 4R-FI | Tiempo 114/6 (1R-RE) | Tiempo 114/6 (4R-FI) |
|---|---|---|---|---|---|
| Planos | RM | +0.001 | +0.010 | ×1.99 | ×2.92 |
| Planos | PC10K | +0.035 | +0.042 | ×1.28 | ×1.89 |
| Planos | FM10K | +0.001 | +0.006 | ×2.05 | ×3.17 |
| Ejes | RM-L50 | +0.140 | +0.205 | ×0.83 | ×1.20 |

**Interpretación:**
- En **planos**, pasar de 6 a 114 vistas apenas cambia el F-score en RM/FM (≤ 0.010), pero duplica o triplica el tiempo. Por eso el paper recomienda **6 vistas** salvo que se necesite máxima precisión.
- En **ejes**, más vistas sí importan mucho (+0.205 F-score con 4R-FI).
- En ejes, el tiempo con 4R-FI es **menor** que con 1R-RE. El paper no explica la causa.

---

## 7. Tabla 2 — Robustez ante ruido y eliminación de partes

Valores exactos. Todas las configuraciones usan **114 vistas**.

| Dataset | Configuración | F-score S | N1 % | N3 % | N5 % | PR20 % | SDE (×10⁻⁴) S | N1 % | N3 % | N5 % | PR20 % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Planos | FM10K-4R-FI | **0.912** | 0.906 | 0.888 | 0.838 | 0.720 | **0.282** | 0.343 | 0.477 | 0.479 | 0.740 |
| Ejes | RM-4R-FI-L50 | **0.803** | 0.301 | 0.223 | 0.182 | 0.581 | 231.9 | 197.0 | 143.0 | 117.0 | 238.9 |

### 7.1 Degradación relativa respecto a la evaluación estándar — *derivado*

| Dataset | Métrica | N1 % | N3 % | N5 % | PR20 % |
|---|---|---|---|---|---|
| Planos | F-score (cambio relativo) | −0.7 % | −2.6 % | −8.1 % | −21.1 % |
| Planos | SDE (múltiplo de S) | ×1.22 | ×1.69 | ×1.70 | ×2.62 |
| Ejes | F-score (cambio relativo) | **−62.5 %** | **−72.2 %** | **−77.3 %** | −27.6 % |
| Ejes | SDE (múltiplo de S) | ×0.85 | ×0.62 | ×0.50 | ×1.03 |

**Interpretación:**
- **Planos:** muy robustos al ruido (pierden < 10 % hasta 5 % de ruido). La eliminación de partes es la perturbación más dañina (−21 %).
- **Ejes:** colapsan ante cualquier nivel de ruido (incluso 1 %). Son algo más tolerantes a la eliminación de partes.
- El **SDE de ejes baja con ruido**. No significa mejor calidad: el SDE solo evalúa las simetrías reportadas, y con ruido se reportan muchas menos (el F-score cae). Es una inferencia coherente con la definición de la métrica, no una explicación del paper.

---

## 8. Tabla 3 — Comparación con el estado del arte (planos)

Valores exactos. Todos los resultados, salvo "Ours" y "Diffusion [67]", provienen del paper de E3Sym [36].

| Método | Tipo | F-score ↑ | SDE (×10⁻⁴) ↓ |
|---|---|---|---|
| PCA | Clásico | 0.692 | 3.32 |
| OBB [6] | Clásico (bounding box orientado) | 0.74 | 1.25 |
| RSD [28] | Geométrico | 0.684 | 0.9 |
| ADS [43] | Transformación (momentos generalizados) | 0.694 | 3.95 |
| PASD [46] | Votación | 0.322 | 14.2 |
| PRST [57] | Geométrico | 0.619 | 1.78 |
| PRST+GEDT [57] | Geométrico | 0.646 | 1.6 |
| PAS [29] | Optimización | 0.678 | 1.75 |
| PRS-Net [15] | Aprendizaje (voxels, autosupervisado) | 0.712 | 0.86 |
| E3Sym [36] | Aprendizaje (equivariante) | 0.753 | 0.46 |
| Diffusion [67] | Basado en difusión | 0.46 | 7.1 |
| **Ours (FM10K-4R-FI)** | Back-projection de DINOv2 | **0.912** | **0.282** |

### 8.1 Ranking y márgenes — *derivado*

| Comparación | ΔF-score | ΔF-score relativo | ΔSDE relativo |
|---|---|---|---|
| Ours vs. E3Sym (2.º mejor) | +0.159 | +21.1 % | −38.7 % |
| Ours vs. OBB (3.º en F-score) | +0.172 | +23.2 % | −77.4 % |
| Ours vs. PRS-Net | +0.200 | +28.1 % | −67.2 % |
| Ours vs. PCA | +0.220 | +31.8 % | −91.5 % |

Ranking por F-score: Ours (0.912) > E3Sym (0.753) > OBB (0.74) > PRS-Net (0.712) > ADS (0.694) > PCA (0.692) > RSD (0.684) > PAS (0.678) > PRST+GEDT (0.646) > PRST (0.619) > Diffusion (0.46) > PASD (0.322).

Ranking por SDE: Ours (0.282) < E3Sym (0.46) < PRS-Net (0.86) < RSD (0.9) < OBB (1.25) < PRST+GEDT (1.6) < PAS (1.75) < PRST (1.78) < PCA (3.32) < ADS (3.95) < Diffusion (7.1) < PASD (14.2).

> No hay tabla comparativa con el estado del arte para **ejes**.

---

## 9. Efecto del tamaño de grupo L (ejes)

| L | F-score (mejor configuración) |
|---|---|
| 10 | 0.649 |
| 50 | 0.803 |

- *Derivado:* +0.154 F-score (+23.7 % relativo).
- El paper afirma que el rendimiento mejora de forma sostenida de L = 10 a L = 50, con el mejor resultado siempre en L = 50. El desglose completo está en el material suplementario, que no se incluye en este PDF.

---

## 10. Tiempo de ejecución y memoria

| Aspecto | Valor |
|---|---|
| Hardware | NVIDIA RTX 3060 Ti / RTX 2080 (**8 GB**) |
| Backbone | DINOv2 *small* con registros, **84 MB** |
| Resolución de render | 224 × 224 px, iluminación plana, material gris uniforme |
| Radio de la esfera de cámaras | 1.2 · O_rad |
| Dimensión de features | 384 (se conserva en todo el pipeline) |
| Objetos medidos | 10 % del dataset (85 + 85); media de 4 424 vértices/puntos en planos y 16 177 en ejes |
| Medición | Tiempo total por objeto, de la extracción de features al cálculo de planos/ejes |

### Tiempos por objeto (Tabla 1, en segundos)

| Tarea | Config. | 1R-RE · 6 v. | 1R-RE · 114 v. | 4R-FI · 6 v. | 4R-FI · 114 v. |
|---|---|---|---|---|---|
| Planos | RM | 6.643 | 13.22 | 7.164 | 20.91 |
| Planos | PC10K | 6.315 | 8.065 | 6.584 | 12.42 |
| Planos | FM10K | 6.106 | 12.49 | 6.318 | 20.03 |
| Ejes | RM-L50 | 77.23 | 64.37 | 40.01 | 48.00 |

- La configuración más rápida en planos es FM10K-1R-RE con 6 vistas (6.106 s).
- E3Sym reporta ≈ **0.027 s** por forma. *Derivado:* el método propuesto (FM10K-4R-FI, 6 vistas, 6.318 s) es ≈ 234× más lento. Los autores atribuyen la diferencia a una implementación "relativamente ingenua" y optimizable.
- El desglose por etapas del tiempo está en el material suplementario (no incluido).

---

## 11. Hiperparámetros y umbrales del método

### 11.1 Back-projection

| Parámetro | Valor |
|---|---|
| Vistas V | {6, 14, 26, 42, 62, 86, 114} |
| Rotaciones de imagen | 1R, 2R, 3R, 4R (múltiplos de 90°) |
| Muestreo de vistas | RE (lat–long) o FI (Fibonacci) |
| Tamaños de muestreo PC/FM | 1K, 3K, 5K, 7K, 10K |
| Agregación multivista | Promedio de features de todas las vistas en que el punto es visible |

### 11.2 Detección de planos (Sec. 7)

| Parámetro | Valor | Función |
|---|---|---|
| Distancia en espacio de features | L1 (Manhattan) | Búsqueda de los 2 vecinos más cercanos → tríos (pᵢ, pⱼ, pₖ) |
| Candidatos por trío | 4 planos | 3 de pares (normal = dirección del par, punto = punto medio) + 1 por los tres puntos (centroide + producto cruz) |
| Filtro de centro | \|n · c\| ≤ **0.05 · R_max** | Descarta planos alejados del origen (5 % de la diagonal) |
| Umbral de validez τ₁ | Chamfer < **0.01** | Plano válido |
| Supresión de duplicados τ₂ | **5°** entre normales | Descarta planos redundantes |
| Confianza | C = max(0, 1 − d_Chamfer / τ₁) | — |
| Máx. planos retenidos | k = **10** | — |

### 11.3 Detección de ejes (Sec. 8.6)

| Parámetro | Valor | Función |
|---|---|---|
| L | 10 a **50** (mejor: 50) | Vecinos en espacio de features por grupo |
| T (iteraciones RANSAC) | **500** | Ajuste de círculos |
| τ_c (umbral de inlier del círculo) | **[10⁻⁴, 5 × 10⁻³]** | — |
| (α, β, K) clustering por orientación | **(0.015, 0.03, 10)** | d_ang(i, j) = 1 − nᵢᵀnⱼ |
| (α_ax, β_ax, K_ax) clustering coaxial | **(0.25, 0.5, 5)** | d_ax(i, j) = ‖nᵢ × Δᵢⱼ‖₂ + ‖nⱼ × (−Δᵢⱼ)‖₂ |
| Confianza del eje | C_t = \|A_t\| / Σ_s \|A_s\| | Proporcional al tamaño del clúster |

### 11.4 Evaluación

| Parámetro | Valor |
|---|---|
| Umbral de confianza de predicción | ≥ 0.95 |
| Umbrales de emparejamiento τ (F-score) | 0.05, 0.10, 0.15, 0.20 |
| Muestras de superficie para SDE | 1 000 por objeto |
| Rotaciones para invariancia de ejes | 45°, 90°, 135°, 180° |

---

## 12. Observaciones y advertencias

1. **Brecha plano–eje:** los planos (una sola reflexión) logran alta invariancia y F-score > 0.9; los ejes (infinitas rotaciones posibles en sólidos de revolución) son más difíciles: 0.803 en condiciones ideales y < 0.31 con ruido.
2. **Comparaciones válidas:** el paper indica que solo deben compararse configuraciones del **mismo tipo de feature** (p. ej., RM-1R-RE vs. RM-1R-FI), porque RM, FM y PC no ofrecen invariancia equivalente. Por eso los valores de *E* no son comparables entre paneles de la Figura 3.
3. **PC no es fiable:** en PC, una mejor invariancia (*E* bajo con 1K) no se traduce en mejor detección. En la Tabla 1, PC10K obtiene el peor F-score en planos.
4. **Mejor representación por tarea:** FM10K para planos, RM para ejes (la única competitiva).
5. **El SDE ignora los falsos negativos:** un método que reporta pocas simetrías pero precisas puede tener un SDE bajo y un F-score bajo a la vez. Por eso el SDE de ejes "mejora" con ruido.
6. **Coste frente a precisión:** con 6 vistas se obtiene casi el mismo F-score en planos que con 114, a ⅓–½ del tiempo.
7. **Lo que falta en este PDF:** el material suplementario, que contiene el desglose completo por L, los tiempos por etapa, los detalles del muestreo de Fibonacci y los resultados cualitativos extendidos.
