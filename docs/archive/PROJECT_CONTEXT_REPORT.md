# Reporte de Contexto del Proyecto de Tesis

---

## 1. Nombre del Tema de Tesis

**Symmetry Detection Using Multimodal Vision-Language Models**

---

## 2. Profesor Guía / Co-guía

_[Por completar]_

---

## 3. Marco del Proyecto

_[Por completar: indicar si existe vínculo con FONDECYT, centro de investigación u otro financiamiento]_

---

## 4. Antecedentes

La detección de simetrías en objetos 3D es un problema fundamental en visión computacional con aplicaciones en reconstrucción 3D, manufactura asistida por computador, robótica y análisis de formas. Las soluciones tradicionales operan directamente sobre geometría 3D (nubes de puntos, mallas) usando algoritmos como RANSAC, votación de Hough o análisis de componentes principales (PCA/SVD), lo que requiere acceso explícito a la representación tridimensional del objeto.

En paralelo, los **Modelos de Lenguaje de Visión (VLMs)** han emergido como sistemas multimodales capaces de razonar sobre imágenes en lenguaje natural. Entre ellos, **Molmo2-8B** (Allen AI, 2024) destaca por su capacidad de _pointing_: dado una imagen y una instrucción verbal, el modelo devuelve coordenadas 2D normalizadas que señalan elementos visuales específicos. Esta capacidad abre una interrogante no explorada en la literatura: **¿puede un VLM identificar la simetría 3D de un objeto a partir únicamente de imágenes 2D renderizadas, sin acceso a la geometría tridimensional?**

La hipótesis central de este trabajo es que Molmo2-8B puede señalar puntos sobre el eje o plano de simetría de un objeto 3D a partir de vistas renderizadas, y que estos puntos 2D, elevados a 3D mediante ray-casting, son suficientes para recuperar la simetría mediante ajuste SVD.

---

## 5. Objetivos

### Objetivo General


### Objetivos Específicos



---

## 6. Metodología

El pipeline implementado consta de **cinco etapas secuenciales**:

### Etapa 1 — Renderizado Multi-Vista (`ImagesGenerator/`)

Se renderizan los objetos 3D (.obj, ShapeNet) desde múltiples puntos de vista usando muestreo en esfera de Fibonacci, que garantiza cobertura angular casi uniforme. Se generan grupos de vistas de tamaño {1, 6, 14, 26}, en resolución 224×224 píxeles, iluminación plana.

**Parámetros:**
- 114 vistas totales por objeto (posiciones fijas en esfera de Fibonacci)
- 3 resoluciones: 224, 448, 1136 px
- 3 modos de iluminación: flat, darker, brighter

### Etapa 2 — Inferencia Molmo2-8B (`MolmoPointing/`)

Se envía al modelo un conjunto de imágenes renderizadas junto con un prompt en lenguaje natural que le pide señalar puntos sobre el eje o plano de simetría del objeto. El modelo responde con coordenadas en escala [0, 1000] en formato `<points coords="...">`.

**Diseño de prompts:**
- **6 prompts por tipo de simetría** (v00–v05), cada uno con variante single-view y multi-view
- **2 estrategias de puntos:**
  - `independent`: los puntos indicados caen directamente sobre el eje/plano
  - `midpoint`: el modelo señala pares bilaterales; el punto medio 3D es el que se usa en SVD
- Se evaluaron versiones originales (v0) y mejoradas (v_1), totalizando **12 prompts por tipo de simetría**

**Flujos de prompting** (`--flow` en `molmo_multiview_runner.py`):
- **Flujo A** (evaluado arriba): pointing directo, sin contexto semántico.
- **Flujo B**: una llamada extra sobre una vista única (semilla determinística por objeto, elevación filtrada a (-60°, 60°)) pide describir el objeto; la descripción se antepone al prompt de pointing.
- **Flujo C**: la llamada extra pide describir el objeto Y señalar puntos semilla en esa misma vista; ambos se anteponen al prompt de pointing multivista.
- B y C están implementados y verificados a nivel de código; falta la corrida contra el modelo real.

### Etapa 3 — Proyección 3D por Ray-Casting (`Mapping/map_to_3d.py`)

Las coordenadas 2D de Molmo se proyectan a 3D mediante ray-casting sobre la malla del objeto. Se utiliza la cámara `FoVPerspectiveCamera` de PyTorch3D (FOV=60°) que reconstruye el rayo desde el pixel hasta la malla. El punto de intersección 3D (en la superficie del objeto) es el que entra al ajuste de simetría.

**Salida:** JSON con `point_3d`, `hit` (bool), `face_id` por cada predicción.

**Retroproyección por parche** (`--patch-size 3/5`): en lugar de un solo rayo exacto, promedia los puntos 3D de una grilla de `h x h` sub-rayos alrededor del punto, para estabilizar el resultado ante ángulos rasantes. `--patch-size 1` (default) es el comportamiento exacto original, sin cambios.

### Etapa 4 — Ajuste de Simetría (`Mapping/estimate_symmetry.py`)

Con la nube de puntos 3D resultante, se ajusta la simetría mediante cuatro métodos:

| Método | Descripción |
|---|---|
| `svd` | SVD directo sobre todos los puntos. Eje = primer PC (máx varianza). Plano normal = último PC (mín varianza). Centroide = origen. |
| `ransac_svd` | RANSAC (1000 iteraciones, threshold = 5% diagonal bbox) + SVD sobre inliers. Mínimo de puntos: 2 (eje), 3 (plano). |
| `svd_sde` | SVD + aceptación solo si SDE ≤ 0.05 (error de distancia simétrica normalizado). |
| `ransac_svd_sde` | RANSAC + SVD + filtro SDE. |

**Opciones de clustering** (`--clustering-method`):
- `greedy`: agrupamiento por centroide (threshold = 5% diagonal bbox); todo punto se asigna siempre a algún cluster.
- `hdbscan`: agrupamiento por densidad (`min_samples` ∈ {2,3,5}); descarta explícitamente puntos aislados como ruido antes del ajuste SVD.

### Etapa 5 — Evaluación (`Mapping/evaluate.py`)

Se compara la simetría predicha contra la ground-truth de ShapeNet usando:

| Métrica | Descripción | Población |
|---|---|---|
| Error angular (media, mediana, std) | Ángulo entre eje/normal predicho y GT; sign-agnostic [0°, 90°] | 100 objetos (90° imputado para no predichos) |
| AUC angular | Área bajo curva de precision vs threshold [0°, 90°] | 100 objetos |
| Precision@5°, @10°, @15° | Fracción de objetos dentro del umbral angular | 100 objetos |
| Error de traslación (media, mediana) | Distancia del origen predicho al GT, normalizada por bbox | n_obj válidos |
| SDE (media, mediana) | Mean distance tras reflexión, normalizada por bbox diagonal | n_obj válidos (solo métodos _sde) |

---

## 7. Estado del Arte

### Detección de Simetría 3D

- **Mitra et al. (2006)** — *Partial and Approximate Symmetry Detection for 3D Geometry.* Pionero en detección de simetría parcial en nubes de puntos 3D mediante votación en espacio de transformaciones.
- **Podolak et al. (2006)** — *A Planar-Reflective Symmetry Transform for 3D Shapes.* Extiende la transformada de Hough para detección de planos de simetría en formas 3D.
- **Li et al. (2017)** — *Symmetry Hierarchy of Man-Made Objects.* Detección jerárquica de simetrías en objetos manufacturados usando análisis estructural de partes.
- **Gao et al. (2020)** — *SymNet: A Simple Symmetric Positive Definite Manifold Deep Learning Method.* Red neuronal para aprender representaciones simétricas.
- **Shi et al. (2020)** — *SymmetryNet: Learning to Predict Reflective and Rotational Symmetries of 3D Shapes from Single Images.* Primera propuesta de detección de simetría 3D desde imagen única.

### Vision-Language Models (VLMs)

- **Deitke et al. (2024)** — *Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models.* Introduce Molmo2-8B con capacidad de pointing pixel-preciso a partir de descripciones en lenguaje natural.
- **Radford et al. (2021)** — *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* Fundamento del alineamiento visión-lenguaje en VLMs modernos.
- **OpenAI (2023)** — *GPT-4 Technical Report.* Referencia para capacidades multimodales en modelos de gran escala.

### Prompting y Razonamiento Visual

- **Wei et al. (2022)** — *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* Fundamenta el uso de instrucciones paso-a-paso para mejorar razonamiento en LLMs/VLMs.
- **Yang et al. (2023)** — *Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V.* Muestra cómo el diseño del prompt afecta significativamente la localización visual en VLMs.

### Datasets

- **Chang et al. (2015)** — *ShapeNet: An Information-Rich 3D Model Repository.* Dataset base utilizado; subconjunto curado con anotaciones de simetría axial y planar.

---

## 8. Datos

### Fuente

Dataset curado a partir de **ShapeNet**, filtrado por objetos con anotaciones de simetría válidas.

### Estructura

```
data/
├── objects/
│   ├── curated_axis_sym_obj/      # 850 mallas .obj + etiquetas .txt
│   └── curated_plane_sym_obj/     # 850 mallas .obj + etiquetas .txt
└── renders/
    ├── axis_sym/
    │   └── <object_id>/224/flat/
    │       ├── IND_00_AZ_090_EL_+89.png   # una imagen por viewpoint
    │       ├── ...
    │       ├── metadata_all.json           # índice viewpoint, azimuth, elevation
    │       └── molmo_multiview_<EXP>.json  # predicciones Molmo + matrices cámara
    └── plane_sym/
        └── <object_id>/...
```

### Formato de etiquetas (ground-truth)

```
# Simetría axial
1
axis DX DY DZ  OX OY OZ     ← dirección (vector unitario) + origen

# Simetría planar (1–3 planos por objeto)
2
plane NX NY NZ  OX OY OZ    ← normal del plano + origen
plane NX NY NZ  OX OY OZ
```

Los objetos están normalizados a cubo unitario centrado en el origen.

### Estadísticas generales

| Característica | Valor |
|---|---|
| Objetos por tipo de simetría | 850 |
| Tipo de malla | OBJ (Wavefront) normalizado a cubo unitario |
| Evaluación realizada sobre | 100 objetos por experimento |
| Resolución de renders | 224 × 224 px (principal), 448, 1136 disponibles |
| Vistas generadas por objeto | 114 (esfera Fibonacci) |
| Grupos de vistas evaluados | 1, 6, 14, 26 |

---

## 9. Análisis Exploratorio de Datos (EDA)

Análisis realizado en `ExploratoryDataAnayisis/EDA.ipynb` sobre los 850 objetos de cada subconjunto.

### Completitud del dataset

- Pares OBJ/TXT: **100% completos** (0 objetos con TXT sin OBJ, 0 con OBJ sin TXT) para ambos subconjuntos.

### Complejidad de mallas (conteo de vértices)

| Estadístico | Simetría Planar |
|---|---|
| Media | 5,449 vértices |
| Desviación estándar | 7,784 vértices |
| Mediana (p50) | 2,143 vértices |
| Mínimo | 191 vértices |
| Máximo | 75,563 vértices |
| P25 | 882 vértices |
| P75 | 6,688 vértices |

**Observación:** La distribución de vértices es fuertemente sesgada a la derecha (media >> mediana). Hay una minoría de objetos muy densos que actúan como outliers. La mayoría de los objetos tiene entre 882 y 6,688 vértices.

### Distribución de número de simetrías (simetría planar)

| N° de planos | N° de objetos | % |
|---|---|---|
| 1 plano | 692 | 81.4% |
| 2 planos | 152 | 17.9% |
| 3 planos | 6 | 0.7% |

**Observación:** La gran mayoría de objetos tiene un único plano de simetría. La evaluación usa _best-match_ cuando hay múltiples planos GT.

### Relación complejidad–simetrías

No se observa correlación fuerte entre número de vértices y número de planos de simetría. Los objetos más complejos (top-10 por vértices, hasta 75,563) tienen en su mayoría 1 simetría.

---

## 10. Estado actual de la implementación

### Lo que está implementado y evaluado

| Componente | Estado |
|---|---|
| Pipeline completo (render → Molmo → ray-cast → SVD → evaluate) | ✅ Funcional |
| 6 prompts axiales + 6 planares (v00–v05) versión original — Flujo A | ✅ Evaluados |
| 6 prompts axiales + 6 planares (v00_1–v05_1) versión mejorada — Flujo A | ✅ Evaluados |
| 4 métodos de ajuste (svd, ransac_svd, svd_sde, ransac_svd_sde) | ✅ Implementados |
| Evaluación con grupos de vistas {1, 6, 14, 26} | ✅ Completado |
| Clustering greedy por centroide pre-SVD | ✅ Implementado |
| Comparación y generación de tablas CSV | ✅ Funcional |
| Flujo B (pointing + descripción) y Flujo C (descripción + pointing integrados) | ⏳ Código implementado (`--flow b/c`), pendiente corrida en GPU |
| Clustering HDBSCAN (`--clustering-method hdbscan`) | ⏳ Código implementado, pendiente barrido `min_samples` {2,3,5} |
| Retroproyección por parche (`--patch-size 3/5`) | ⏳ Código implementado, pendiente barrido sobre mejor prompt/N |

### Mejores resultados obtenidos (experimentos `experiments_20_06_2026`, SVD, mejor n_views)

**Simetría Axial**

| Prompt | AUC | n_obj válidos |
|---|---|---|
| axis_v05_1 | **0.388** | 87 |
| axis_v00_1 | 0.383 | 82 |
| axis_v02_1 | 0.354 | 80 |
| axis_v05 (original) | 0.354 | 92 |

**Simetría Planar**

| Prompt | AUC | n_obj válidos |
|---|---|---|
| plane_v04_1 | **0.505** | 66 |
| plane_v02_1 | 0.436 | 55 |
| plane_v00_1 | 0.418 | 94 |
| plane_v04 (original) | 0.407 | 93 |

### Impacto de las mejoras v_1 (ΔAUC promedio)

Las mejoras de prompts v_1 se diseñaron a partir de dos fuentes:
1. **Análisis de failure modes** en resultados experimentales (v02 axial: n_obj=27 → problema geométrico silhouette; v01: colapso de alturas; v05 planar: selección diagonal)
2. **Cross-pollination** de elementos de prompts ganadores (axis_v05 → upper/lower split; plane_v04 → definición explícita de centerline)

ΔAUC promedio sobre todos los prompts: **+0.076** (axial), **+0.093** (planar).

---

## Anexo — Carta Gantt

_[Por adjuntar]_

---

## Anexo — Repositorio y Estructura de Código

```
Symmetry-Detection-Using-Multimodal-Vision-Language-Models/
├── ImagesGenerator/          # Renderizado Fibonacci multi-vista
│   └── data_render.py        # Batch runner (movido desde utils/)
├── MolmoPointing/
│   ├── prompts/axis/         # axis_v00–v05, axis_v00_1–v05_1 × {single,multi}.txt
│   ├── prompts/plane/        # plane_v00–v05, plane_v00_1–v05_1 × {single,multi}.txt
│   ├── prompts/description/  # Flujo B/C: describe.txt, describe_and_point_{axis,plane}.txt
│   ├── molmo_multiview_runner.py  # --flow {a,b,c}
│   ├── prompts_registry.py
│   ├── Experiments.md        # Tabla de prompts + comandos completos
│   └── PROMPT_IMPROVEMENTS_v1.md
├── Mapping/
│   ├── map_to_3d.py          # Ray-casting PyTorch3D; --patch-size {1,3,5}
│   ├── estimate_symmetry.py  # SVD/RANSAC fitting; --clustering-method {none,greedy,hdbscan}
│   ├── evaluate.py           # Métricas angulares + SDE
│   ├── compare_results.py    # Tablas y plots comparativos
│   ├── visualize_rays.py     # Debug: visor 3D Polyscope
│   ├── cleanup_experiments.py
│   └── export_viz_samples.py
├── pipeline_common/          # Helpers compartidos (naming, datasets, camera, clustering, view_selection)
├── InteractiveViewer/        # Visor Open3D de simetrías GT
├── ExploratoryDataAnayisis/
│   └── EDA.ipynb
├── results/
│   ├── experiments_18_06_2026/
│   │   └── axis_sym_comparison.csv   # (aún sin plane_sym_comparison.csv)
│   └── plots/
└── README.md
```

Nota: la tabla de "Mejores resultados obtenidos" arriba cita la carpeta
`experiments_20_06_2026`, que no está presente en este repositorio (solo existe
`experiments_18_06_2026/axis_sym_comparison.csv`) — reconciliar antes de citar
esos números en la tesis final.

**Stack tecnológico:** Python 3.10, PyTorch 2.6, PyTorch3D 0.7.9, Transformers 4.57, Trimesh, Pandas, Matplotlib, Polyscope (visualización).
