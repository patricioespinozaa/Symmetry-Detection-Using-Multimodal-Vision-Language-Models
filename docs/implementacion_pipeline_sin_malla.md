# Implementación del pipeline sin malla: qué se mantiene, borra, modifica y agrega

> Documento de implementación (no de diseño — eso es `docs/pipeline_sin_malla.md`)
> para llevar el prototipo ya validado en `test_pipeline_sin_malla.ipynb`
> (triangulación de eje/plano sin ray-casting contra la malla, ver también el
> refactor de métricas en `docs/actualizacion_metricas.md`) a un módulo de
> producción, en paralelo al pipeline con malla existente — **no lo
> reemplaza**, corre al lado como alternativa comparable.

---

## 0. Resumen ejecutivo

| Archivo | Acción | Una línea |
|---|---|---|
| `ImagesGenerator/` (todo) | **SE MANTIENE** | Renderizado multivista, sin cambios — ninguna etapa del rediseño lo toca |
| `MolmoPointing/` (todo, incl. `molmo_multiview_runner.py`) | **SE MANTIENE** | Pointing 2D de Molmo2, mismos JSON de prompts — el pipeline sin malla los lee tal cual |
| `pipeline_common/camera.py` | **SE MANTIENE** | `molmo_to_ndc`/`build_camera_rays` se reusan tal cual; `cast_ray`/`cast_ray_patch` siguen usándose solo en `map_to_3d.py` (pipeline con malla) |
| `pipeline_common/datasets.py` | **SE MANTIENE** | `load_mesh`/`load_mesh_vertices`/`OBJECTS_SUBDIR` — reusados para GT y validación final |
| `pipeline_common/naming.py` | **SE MANTIENE** | `exp_filename` reusado por el nuevo estimador para nombrar `predicted_symmetry_<EXP>.json` |
| `Mapping/map_to_3d.py` | **SE MANTIENE, no se llama en la ruta sin malla** | Sigue existiendo para generar el baseline "con malla" de comparación — el pipeline sin malla simplemente no lo invoca |
| `Mapping/estimate_symmetry.py` | **SE MANTIENE, sin cambios** | Sigue siendo el ajustador SVD/RANSAC del pipeline con malla (baseline de comparación) |
| `Mapping/compare_results.py` | **SE MANTIENE, revisar en fase 2** | Agrega/grafica CSV de `evaluate.py` — funciona igual mientras el nuevo método se reporte con un nombre de método válido; revisar si hay nombres de método hardcodeados al agregar multi-plano |
| `docs/pipeline_sin_malla.md` | **SE MANTIENE como diseño** | Sigue siendo la referencia de *por qué*; este documento es el *cómo* |
| `pipeline_common/triangulation.py` | **✅ IMPLEMENTADO** | Geometría reusable: rayos de cámara → planos de interpretación → triangulación de línea + `widest_pair` (nueva, habilita Flow C para eje) |
| `Mapping/estimate_symmetry_no_mesh.py` | **✅ IMPLEMENTADO Y VALIDADO** | Estimador de producción: lee `molmo_multiview_<EXP>.json` directo, sin pasar por `map_to_3d.py`; escribe `predicted_symmetry_<EXP>.json` (mismo esquema, ver §3). Validado end-to-end contra `evaluate.py` reproduciendo exactamente los resultados del notebook (ver §8) |
| `Mapping/evaluate.py` | **✅ MODIFICADO (fase 1 y fase 2 completas)** | Fase 1: `"triangulation"`/`"triangulation_multiplane"` agregados a `METHODS`. Fase 2: `evaluate_object` detecta `{"planes": [...]}` y delega a `evaluate_plane_multi_from_pred`/`evaluate_plane_multiset`; `compute_summary`/`write_csv`/tabla de consola tienen una rama separada para el resumen multi-plano (`recall_planes_mean`, `precision_planes_mean`, etc.) — validado end-to-end (ver §8) |
| `MolmoPointing/molmo_multiview_runner.py` | **✅ MODIFICADO** | `build_flow_c_prompts` ahora exige exactamente 2 puntos por vista (antes "hasta 3, menos está bien") para ambos tipos de simetría — ver §6, riesgo #5 |
| `CLAUDE.md` | **SE MODIFICA (pendiente)** | Diagrama de flujo necesita la rama nueva |
| `README.md` | **SE MODIFICA (pendiente)** | Agregar comandos de la nueva ruta a "Full pipeline" |
| — | **SE BORRA: nada** | Ver §5 — el pipeline con malla se mantiene íntegro como baseline, no hay código productivo a eliminar |

---

## 1. Arquitectura: antes y después

**Antes (única ruta, con malla):**
```
ImagesGenerator/ → MolmoPointing/ → Mapping/map_to_3d.py → Mapping/estimate_symmetry.py → Mapping/evaluate.py → Mapping/compare_results.py
                                     (ray-casting 2D→3D)     (SVD/RANSAC sobre 3D)
```

**Después (dos rutas paralelas, mismo punto de entrada y de salida):**
```
ImagesGenerator/ → MolmoPointing/ ──┬──→ Mapping/map_to_3d.py ──→ Mapping/estimate_symmetry.py ──┐
                                     │        (ray-casting, CON malla)                            ├─→ Mapping/evaluate.py → Mapping/compare_results.py
                                     └──→ Mapping/estimate_symmetry_no_mesh.py ───────────────────┘
                                              (triangulación, SIN malla, ray-casting solo en 5)
```

El punto clave del rediseño (`docs/pipeline_sin_malla.md` §1): la malla deja
de estar en la ruta de estimación de la rama nueva — `estimate_symmetry_no_mesh.py`
nunca abre un `.obj`. La malla reaparece únicamente dentro de `evaluate.py`
(paso 5, validación), exactamente igual para ambas ramas — por eso
`evaluate.py` necesita cambios mínimos, no una reescritura.

---

## 2. SE AGREGA — detalle archivo por archivo

### 2.1 `pipeline_common/triangulation.py` (nuevo)

Geometría pura, sin dependencia de `estimate_symmetry_no_mesh.py` ni de
ningún formato de JSON específico — utilidades reusables, migradas
literalmente de la sección 3 (`f67f5878`) del notebook:

| Función | Firma | Qué hace |
|---|---|---|
| `ray_dir_for_point` | `(x, y, R, T, fov_deg, image_size) -> (origin, direction)` | Envuelve `molmo_to_ndc` + `build_camera_rays` — rayo 3D de cámara para un punto Molmo2 |
| `view_forward_direction` | `(R, T, fov_deg, image_size) -> direction` | Dirección del eje óptico de la cámara — usada por el puntaje "de canto" del plano |
| `interpretation_plane_normal` | `(dir_a, dir_b) -> normal \| None` | Normal del plano que contiene el centro de cámara y dos rayos — `None` si son casi paralelos |
| `triangulate_line` | `(camera_centers, plane_normals) -> (point, direction)` | Intersección por mínimos cuadrados (SVD) de ≥2 planos de interpretación → línea 3D |
| `get_point_by_obj_id` | `(pts, obj_id) -> dict \| None` | Busca un punto por `obj_id` dentro de la lista de puntos de una vista — se mantiene para los llamadores que sí dependen de roles fijos (plano bilateral, ver más abajo) |
| `widest_pair` | `(pts) -> (dict, dict) \| None` | **Nueva** — generaliza la búsqueda de "el par de puntos de una vista" para no depender de `obj_id`. Ver diseño abajo |

**Diferencia respecto al notebook**: en el notebook, `FOV_DEG=60.0` e
`IMAGE_SIZE=224` estaban hardcodeados (un solo objeto de prueba). En
producción, `ray_dir_for_point`/`view_forward_direction` deben recibir
`fov_deg`/`image_size` **por objeto**, leídos del mismo `manifest.json` que
ya lee `map_to_3d.py` (no asumir un tamaño fijo).

#### `widest_pair` — generalización para habilitar Flow C en el eje

**Motivación** (ver discusión completa en la conversación que originó este
documento): la garantía geométrica de `interpretation_plane_normal` no
depende de que los dos puntos de una vista tengan roles fijos
("arriba"/"abajo") — solo depende de que **ambos estén sobre la proyección
2D del eje real**, sean distintos, y estén razonablemente separados (para
buen condicionamiento numérico del producto cruzado). La convención
`obj_id 1`="arriba"/`obj_id 2`="abajo" de los prompts Flow A es una
convención de *prompt*, no un requisito de la *triangulación*. Flow C
(`build_flow_c_prompts` en `molmo_multiview_runner.py`) devuelve, a
propósito, una lista libre de 1 a `MAX_POINTS_PER_IMAGE=3` puntos sin roles
fijos — por eso `get_point_by_obj_id(pts, 1)`/`(pts, 2)` fallan sistemáticamente
con Flow C (confirmado: 0/14 corridas válidas para `axis_v05_1_flowC` en el
barrido del notebook), no porque la geometría lo prohíba.

```python
def widest_pair(pts: list[dict]) -> tuple[dict, dict] | None:
    """
    De TODOS los puntos que trae una vista (sin importar su obj_id), devuelve
    el par con mayor distancia euclidiana en pixeles entre si -- o None si hay
    menos de 2 puntos. Generaliza get_point_by_obj_id(pts,1)/(pts,2) para no
    depender de una convencion de roles fijos: con exactamente 2 puntos
    (Flow A) el resultado es identico a tomar "arriba"/"abajo" (es el unico
    par posible); con 1-3 puntos sin roles (Flow C) elige el par mas separado,
    descartando el resto como redundante -- evita ademas inyectar mas de una
    linea de interpretacion por vista (ver riesgo #1 de la seccion 6, sobre
    vistas con puntos casi duplicados sesgando el SVD: tomar TODAS las
    combinaciones de una vista de 3 puntos multiplicaria ese riesgo en vez de
    mitigarlo).
    """
    if len(pts) < 2:
        return None
    best_pair, best_dist_sq = None, -1.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dx = pts[i]["x"] - pts[j]["x"]
            dy = pts[i]["y"] - pts[j]["y"]
            dist_sq = dx * dx + dy * dy
            if dist_sq > best_dist_sq:
                best_dist_sq, best_pair = dist_sq, (pts[i], pts[j])
    return best_pair
```

**Por qué es seguro para los prompts existentes (Flow A)**: con exactamente
2 puntos por vista, `widest_pair` siempre devuelve esos 2 mismos puntos (es
el único par posible) — el resultado numérico de `estimate_axis_no_mesh` para
los 12 prompts ya validados en el notebook (`axis_v00`...`axis_v05_1`,
Flow B) **no cambia**. El orden de los dos puntos dentro del par tampoco
importa: `interpretation_plane_normal` solo puede cambiar de signo si se
invierten, y ni `angular_error_deg` (sign-agnostic) ni la restricción de
plano que usa `triangulate_line` (`n·(x−C)=0`, invariante al signo de `n`)
se ven afectados por eso.

**Qué NO se generaliza (a propósito)**: para el **plano**, `line_from_view_pair`
sigue usando `get_point_by_obj_id(pts, 1)`/`(pts, 2)` sin cambios — los
prompts bilaterales de plano (`plane_v01`/`v02`/`v03`, etc.) sí le dan un
significado geométrico real a los roles ("izquierda"/"derecha", un par
reflejado) que `widest_pair` no captura (el punto más separado del otro no
es necesariamente su reflejo especular). Generalizar Flow C también para
plano queda fuera de este documento — requiere revisar primero qué pide
exactamente `describe_and_point_plane.txt` antes de decidir si aplica la
misma lógica u otra distinta.

### 2.2 `Mapping/estimate_symmetry_no_mesh.py` (nuevo)

Script de producción, CLI espejado de `estimate_symmetry.py` pero **sin**
`--objects-root` obligatorio (la malla no hace falta para estimar — solo
para el SDE opcional de aceptación, si se decide portar esa heurística
también a esta rama, ver §6):

```bash
python Mapping/estimate_symmetry_no_mesh.py \
    --renders-root ../data/renders \
    --symmetry-type axis_sym \
    --sizes 224 --lightings flat \
    --experiment-id axis_v02 \
    --view-groups 1 6 14 26 \
    --overwrite
```

**Input**: lee `molmo_multiview_<EXP>.json` **directamente** de
`<renders_root>/<symmetry_type>/<object_id>/<size>/<lighting>/` — el mismo
archivo que ya escribe `MolmoPointing/molmo_multiview_runner.py`, sin pasar
por `map_to_3d.py` ni necesitar `mapped_points_3d.json`.

**Funciones migradas del notebook** (sección 3.1/3.2/4, celdas
`d9e9ebc5`/`e88653c8`/`a1ba78dc`), sin cambios de lógica, solo adaptadas a
recibir `fov_deg`/`image_size` como parámetro en vez de constantes globales:

| Función notebook | Uso en producción |
|---|---|
| `estimate_axis_no_mesh(points_by_image, images_sent)` | Eje — un resultado por `n_views`. **Modificada respecto al notebook**: internamente usa `widest_pair(pts)` (nuevo, §2.1) en vez de `get_point_by_obj_id(pts, 1)`/`(pts, 2)` — esto es lo que habilita usar Flow C en esta rama, sin cambiar el resultado para los prompts Flow A ya validados |
| `line_from_view_pair`, `estimate_plane_no_mesh` | Plano, modo "un solo plano" (compatible con el esquema actual, ver §3) — **sin cambios**, sigue usando `get_point_by_obj_id` (roles bilaterales fijos, ver justificación en §2.1) |
| `detect_planes_no_mesh(..., max_planes=3)` | Plano, modo multi-plano (`--max-planes`, ver §3, fase 2) — sin cambios |

**Output**: `predicted_symmetry_<EXP>.json`, mismo directorio y convención de
nombre que ya usa `estimate_symmetry.py` (vía `pipeline_common.naming.exp_filename`)
— la razón de reusar exactamente esa convención de nombres es que **no colisiona**
con las corridas con malla existentes en tanto se use un `--experiment-id`
distinto (p. ej. `axis_v02` para con-malla, `axis_v02_nomesh` para sin-malla)
— no hace falta un nombre de archivo base distinto.

**Qué NO migra (por diseño, no por olvido)**:
- `estimate_axis_old_raycast`/`estimate_plane_old_raycast` (celdas de
  comparación del notebook) — **no se portan**, porque son literalmente lo
  que ya hacen `map_to_3d.py` + `estimate_symmetry.py` en producción. El
  baseline "con malla" para comparar se obtiene corriendo esos dos scripts
  ya existentes, no reimplementándolos.
- `match_planes_to_gt` (celda `a1ba78dc`) — es lógica de *validación* (matchea
  contra GT), no de estimación; su equivalente de producción es
  `evaluate_plane_multiset`, que ya vive en `Mapping/evaluate.py` (agregada
  en el refactor de métricas, ver `docs/actualizacion_metricas.md` §5.6) —
  no hace falta duplicarla en el nuevo módulo.
- Las celdas de diagnóstico de índices de vistas (`c7e81519`) — son
  verificación puntual de este notebook de prueba, no lógica de producción;
  el chequeo real y reusable en `n_views` ya vive en
  `MolmoPointing/molmo_multiview_runner.py::get_n_views_entries` (sin
  cambios, es la fuente de verdad).

---

## 3. Compatibilidad de esquema JSON (`predicted_symmetry_<EXP>.json`)

### Fase 1 — eje y "mejor plano único" (compatible sin tocar `evaluate.py::evaluate_axis`/`evaluate_plane`)

El esquema actual (`estimate_symmetry.py`, confirmado en código) es:

```json
{
  "object_id": "...", "symmetry_type": "axis_sym", "point_mode": "...",
  "clustering_method": "none", "hdbscan_min_samples": null,
  "n_views_predictions": {
    "6": {
      "n_points_raw": 12, "n_points_fit": 12,
      "svd":            {"direction": [...], "origin": [...], "n_points": 12, "n_inliers": null, "sde": null, "accepted": null},
      "ransac_svd":     {...}, "svd_sde": {...}, "ransac_svd_sde": {...}
    }
  }
}
```

`estimate_symmetry_no_mesh.py` escribe el **mismo** esquema para eje y para
plano-de-un-solo-resultado, bajo una clave de método nueva `"triangulation"`
en vez de sobrecargar `"svd"` (que implicaría, incorrectamente, que hubo un
SVD sobre puntos 3D ray-casteados):

```json
{
  "object_id": "...", "symmetry_type": "axis_sym",
  "point_mode": "triangulation", "clustering_method": "none",
  "n_views_predictions": {
    "6": {
      "n_points_raw": 12, "n_points_fit": 12,
      "triangulation": {"direction": [...], "origin": [...], "n_points": null,
                         "n_views_used": 6, "n_inliers": null, "sde": null, "accepted": null}
    }
  }
}
```

**Cambio necesario en `evaluate.py`**: agregar `"triangulation"` a la tupla
`METHODS` (una línea) para que `--method triangulation` sea una opción
válida del CLI — `evaluate_axis`/`evaluate_plane` no necesitan ningún cambio,
ya funcionan sobre cualquier dict con `direction`/`normal` + `origin`.

### Fase 2 — plano multi-candidato (`detect_planes_no_mesh`) — ✅ IMPLEMENTADO

Cuando `--max-planes > 1`, la salida por `n_views` deja de tener un único
`normal`/`origin` y pasa a una **lista**:

```json
"triangulation_multiplane": {
  "planes": [
    {"normal": [...], "origin": [...], "n_views_used": 16, "n_candidates": 44850, "good_views": [2,3,6,...]},
    {"normal": [...], "origin": [...], "n_views_used": 8,  "n_candidates": 630,   "good_views": [1,4,5,...]}
  ]
}
```

**Cambio hecho en `evaluate.py`**:
- `evaluate_object` detecta `isinstance(pred, dict) and "planes" in pred` y
  delega a `evaluate_plane_multi_from_pred(pred["planes"], true_label["elements"])`
  (nueva, envuelve `evaluate_plane_multiset` agregando `"status"`/`"n_points"`
  para que el resto del pipeline la trate igual que cualquier otro resultado
  por objeto).
- `compute_summary` tiene una rama separada para entradas con
  `"n_planes_predicted"` — agrega `n_planes_predicted_mean`,
  `n_true_planes_mean`, `n_planes_matched_mean`, `recall_planes_mean`,
  `precision_planes_mean` por `n_views`, **sin** mezclarlos con
  `angular_error_mean`/`auc_angular`/etc. (son un resumen completamente
  distinto, no una fila más de la misma tabla).
- `write_csv` y la tabla de consola de `main()` detectan el resumen
  multi-plano (`any("n_planes_predicted_mean" in s for s in summary.values())`)
  y usan columnas propias en vez de las de eje/plano-único.
- **`SDE_ref`/`F1_ref` para multi-plano — ✅ conectado** (corrección hecha
  sobre la implementación inicial, ver más abajo): `evaluate_plane_multi_from_pred`
  ahora recibe `ref_ctx` y calcula `sde_ref_per_plane` (una `calplaneloss`
  por cada plano predicho — no hay un único "SDE_ref del conjunto") y
  `f1_counts_ref`/`f1_counts_ref_hungarian` sobre la **lista completa** de
  planos predichos contra la lista completa de GT (exactamente como ya
  probaba `test_pipeline_sin_malla.ipynb` en su propia sección de
  evaluación — `f1_match_counts`/`f1_match_counts_hungarian` ya estaban
  diseñadas desde el refactor de métricas para aceptar varios candidatos).
  `compute_summary` pool-ea `sde_ref_per_plane` de todos los objetos en la
  misma lista plana que usa el camino de un solo plano (mismo mean/min/max),
  y acumula `f1_counts_ref` en el mismo `f1_totals` dataset-level que ya
  existía. `write_csv`/tabla de consola muestran estas columnas cuando
  `--with-reference-metrics` está activo, en ambos modos.
  **Nota de la primera pasada de este documento (ya corregida)**: se había
  dejado esto sin conectar por error al implementar la fase 2 -- quedó
  documentado y corregido en la misma sesión de trabajo cuando se detectó
  la inconsistencia con el notebook.

**Validado end-to-end** (ver §8) con `--max-planes 3` sobre
`plane_v01_1`/objeto `10654ea604644c8eca7ed590d69b9804`: corre sin errores,
reproduce el mismo conteo de planos que ya había encontrado el notebook
(1/2/2 en `n_views` 6/14/26), y el caso de un solo plano (`--max-planes 1`,
`--method triangulation`) se re-testeó para confirmar que sigue dando
exactamente 84.14°/66.52°/85.87° — sin regresión.

---

## 4. SE MODIFICA — detalle archivo por archivo

| Archivo | Cambio exacto | Fase |
|---|---|---|
| `Mapping/evaluate.py` | Agregar `"triangulation"` a `METHODS` (una línea) | 1 |
| `Mapping/evaluate.py` | `evaluate_object`: detectar `pred.get("planes")` y delegar a `evaluate_plane_multiset`; `compute_summary`: agregar `recall_planes`/`precision_planes`/`n_planes_matched` | 2 |
| `Mapping/compare_results.py` | Revisar si algún nombre de columna/método está hardcodeado (no confirmado — no se leyó el archivo completo en este documento, marcar como pendiente de verificar antes de fase 2) | 2 |
| `docs/pipeline_sin_malla.md` | Marcar §3.1/3.2/4 como "implementado, ver `Mapping/estimate_symmetry_no_mesh.py`" en vez de `[NUEVO]` | 1 |
| `CLAUDE.md` | Diagrama de "Flujo del pipeline" — agregar la rama `estimate_symmetry_no_mesh.py` en paralelo a `map_to_3d.py`+`estimate_symmetry.py` | 1 |
| `README.md` | Sección "Full pipeline" — agregar el bloque de comandos de la rama sin malla (ver ejemplo en §2.2) | 1 |

---

## 5. SE BORRA — nada

Ningún archivo de código productivo se elimina. El pipeline con malla
(`map_to_3d.py` + `estimate_symmetry.py`) se mantiene **íntegro y en uso**
como baseline de comparación — la evidencia empírica del propio notebook
(barrido de 14 prompts × 4 n_views) mostró que el método sin malla es mejor
en la mayoría de los casos probados, pero eso es justamente algo que hay que
poder seguir demostrando comparando contra el método con malla, objeto por
objeto, no algo que se resuelva reemplazando el código viejo.

**Housekeeping aparte, no relacionado a la arquitectura**: el archivo local
`data/renders/axis_sym/1a9c1cbf1ca9ca24274623f5a5d0bcdc/224/flat/molmo_multiview.json`
(el JSON con el bug de vistas secuenciales, confirmado que no existe en el
servidor) es un artefacto de datos obsoleto, no código — se puede borrar sin
impacto, pero es independiente de este refactor.

---

## 6. Riesgos y limitaciones conocidas a resolver antes de escalar

Heredadas de los hallazgos empíricos del notebook (`test_pipeline_sin_malla.ipynb`,
sección de conclusiones) — documentarlas acá para que no se pierdan al pasar
a producción:

1. **`triangulate_line` no pondera por confiabilidad** — vistas con puntos
   2D casi idénticos (Molmo2 repitiendo coordenadas) inyectan líneas casi
   duplicadas que sesgan el SVD sin aportar información real. Se observó que
   esto hace que más vistas (14, 26) no mejoren sobre 6 para el eje en el
   único objeto probado. Antes de escalar, considerar un filtro de
   "vistas casi-duplicadas" o un esquema de pesos (ver Iskakov et al. 2019,
   citado en `docs/pipeline_sin_malla.md` §8.3).
2. **El heurístico de "mejor score inicial" en `estimate_plane_no_mesh` no
   siempre elige el mejor candidato disponible** — se observó un caso
   (n_views=26) donde un candidato secundario, construido con las vistas
   "sobrantes", ajustaba mejor al GT que el candidato inicial. Antes de
   confiar en el resultado de producción, evaluar los 2-3 mejores candidatos
   iniciales (no solo el primero) y quedarse con el de mejor `F1_ref`/`SDE_ref`
   tras el reajuste.
3. **Validado sobre 1 objeto axial y 1 objeto planar únicamente** — todo lo
   anterior es evidencia preliminar. `docs/pipeline_sin_malla.md` §7 ya
   preveía este orden: correr sobre el dataset completo (850 objetos por
   tipo) recién después de validar el mecanismo en pocos objetos.
4. **`edge_on_thresh=0.5` (umbral "de canto" para el plano) está hardcodeado**
   — no se barrió ni se justificó empíricamente más allá del valor por
   defecto; candidato a hiperparámetro a barrer junto con `max_planes` y
   `dup_angle_thresh_deg` cuando se corra sobre más objetos.
5. **Flow C — causa raíz identificada, fix aplicado, validación pendiente de
   re-correr Molmo2 en el servidor.** Historia completa: (a) se implementó
   `widest_pair` (§2.1) asumiendo que Flow C devolvía 1-3 puntos sin rol
   fijo; (b) se re-corrió el barrido y `axis_v05_1_flowC` siguió dando
   **0/4** corridas válidas; (c) inspeccionando el JSON crudo se confirmó
   que la causa real era otra: para `n_views > 1`, Molmo2 devolvía
   **exactamente 1 punto por vista** (no 2-3 sin rol, como se había
   asumido) — no hay ningún par que formar, con ningún criterio de
   selección, porque directamente no hay un segundo punto. Se confirmó
   además que el mismo colapso ocurre para **plane_v04_1_flowC** (mezcla de
   1/2/3 puntos según `n_views`), no es específico del eje. Causa: el
   prompt permitía "menos puntos si es lo único que el modelo puede
   identificar con confianza" (`anti_degenerate_rules`), y Molmo2 se vuelve
   conservador cuando hay muchas imágenes en la misma llamada.
   **Fix aplicado** (`MolmoPointing/molmo_multiview_runner.py::build_flow_c_prompts`,
   ambas ramas axis/plane): se reemplazó "return AT MOST N points, fewer OK"
   por "return EXACTLY 2 points", renombrando `MAX_POINTS_PER_IMAGE` →
   `FLOW_C_POINTS_PER_IMAGE = 2` y reescribiendo `task_single`/`task_multi`/
   `anti_degenerate_rules` para pedir explícitamente 2 puntos, igual que
   Flow A. `widest_pair` (§2.1) sigue siendo necesaria y correcta para
   consumir el resultado (Flow C no promete roles fijos aunque ahora
   siempre sean 2 puntos).
   **No se pudo validar empíricamente todavía**: el cambio de prompt solo
   afecta corridas *futuras* de Molmo2 — el JSON local
   (`molmo_multiview_axis_v05_1_flowC.json`) se generó con el prompt viejo y
   no se puede regenerar localmente (sin GPU/`transformers`, ver
   `CLAUDE.md`). **Pendiente**: re-correr `MolmoPointing/molmo_multiview_runner.py`
   con `--flow c` en el servidor para `axis_v05_1_flowC`/`plane_v04_1_flowC`
   y confirmar que ahora sí producen 2 puntos/vista consistentemente antes
   de darlo por resuelto.

---

## 7. Mapeo notebook → producción (referencia rápida)

| Celda del notebook (`test_pipeline_sin_malla.ipynb`) | Destino en producción |
|---|---|
| Setup (imports, `FOV_DEG`, `IMAGE_SIZE`) | `Mapping/estimate_symmetry_no_mesh.py` (constantes → parámetros por objeto vía `manifest.json`) |
| Carga de `axis_runs`/`plane_runs` (prompt × n_views) | `Mapping/estimate_symmetry_no_mesh.py::main()` — reemplazado por iteración sobre objetos reales del dataset, no una carpeta de muestra |
| Diagnóstico de índices de vistas | No migra — ya cubierto por `get_n_views_entries` + `Mapping/audit_view_indices.py`/`audit_view_indices_v2.py` (herramientas de auditoría ya existentes) |
| `ray_dir_for_point`, `view_forward_direction`, `interpretation_plane_normal`, `triangulate_line`, `get_point_by_obj_id` | `pipeline_common/triangulation.py` (nuevo, §2.1) |
| — (no existía en el notebook) | `widest_pair` — **nuevo**, `pipeline_common/triangulation.py` (§2.1) — generaliza la selección de pares para habilitar Flow C en el eje |
| `estimate_axis_no_mesh` | `Mapping/estimate_symmetry_no_mesh.py` (nuevo, §2.2) — **modificada** para usar `widest_pair` en vez de `get_point_by_obj_id(pts, 1/2)` |
| `estimate_axis_old_raycast` | No migra — usar `map_to_3d.py` + `estimate_symmetry.py` existentes como baseline |
| `line_from_view_pair`, `estimate_plane_no_mesh` | `Mapping/estimate_symmetry_no_mesh.py` (nuevo, §2.2) |
| `detect_planes_no_mesh` | `Mapping/estimate_symmetry_no_mesh.py`, modo `--max-planes` (nuevo, §2.2, fase 2) |
| `match_planes_to_gt` | Reemplazado por `evaluate_plane_multiset` (ya en `Mapping/evaluate.py`, ver `docs/actualizacion_metricas.md` §5.6) |
| Import de `reference_metrics` (celda `7221c7d7`) | Ya resuelto — `Mapping/evaluate.py` (post-merge, ver `docs/actualizacion_metricas.md`) |
| Sección de evaluación (angular error, SDE_ref, F1_ref) | `Mapping/evaluate.py`, sin cambios de fase 1 — el mismo módulo ya sirve para ambas ramas |

---

## 8. Estado: IMPLEMENTADO y validado end-to-end (fase 1 y fase 2 completas)

Todo lo de este documento marcado `✅` en §0 ya es código real, no un plan.
Verificación hecha con fixtures armados a partir de los mismos JSON de
Molmo2 ya usados en `test_pipeline_sin_malla.ipynb` (no datos sintéticos):

**Eje** (`estimate_symmetry_no_mesh.py --symmetry-type axis_sym`, objeto
`10433e5bd8fa2a337b00c7b93209c459`, prompt `axis_v02`), corrido y luego
evaluado con `evaluate.py --method triangulation` sin ningún código
especial — reproduce **exactamente** los números ya validados en el
notebook:

| n_views | dirección (`estimate_symmetry_no_mesh.py`) | error angular (`evaluate.py`) | notebook (referencia) |
|---|---|---|---|
| 6  | `[0.9749, 0.2096, 0.0748]` | 28.12° | 28.12° ✓ |
| 14 | — | 47.85° | 47.85° ✓ |
| 26 | — | 41.19° | 41.19° ✓ |

**Plano, un solo resultado** (`plane_v01_1`, objeto `10654ea604644c8eca7ed590d69b9804`):

| n_views | error angular | notebook (referencia) |
|---|---|---|
| 6  | 84.14° | 84.14° ✓ |
| 14 | 66.52° | 66.52° ✓ |
| 26 | 85.87° | 85.87° ✓ |

**Plano, multi-candidato** (`--max-planes 3`, mismo objeto/prompt), evaluado
con `evaluate.py --method triangulation_multiplane` (fase 2, ahora conectada):

| n_views | planos detectados | notebook (referencia) | recall/precision (`evaluate.py`) |
|---|---|---|---|
| 6  | 1 | 1 ✓ | recall=0.0, precision=0.0 (GT tiene 1 plano, ninguno matcheó bajo 15°) |
| 14 | 2 | 2 ✓ | recall=0.0, precision=0.0 |
| 26 | 2 | 2 ✓ | recall=0.0, precision=0.0 |

El recall/precision en 0 es coherente con lo ya sabido: el error angular de
esta combinación prompt/objeto es de 66-86° (ver §1 del notebook), muy por
encima del umbral de match (15°) — el conteo de planos coincide, la
detección de match también se comporta como se esperaba. Se re-testeó
además `--max-planes 1` / `--method triangulation` sobre el mismo fixture
para confirmar que sigue dando 84.14°/66.52°/85.87° sin cambios (no hubo
regresión al conectar la fase 2).

**`SDE_ref`/`F1_ref` en modo multi-plano** (`--max-planes 3
--with-reference-metrics`), mismo objeto/prompt — coincide con `F1_ref =
0.0000` que ya había reportado el notebook en su propia sección de
evaluación para este objeto:

| n_views | planos | SDE_ref por plano | F1_ref (dataset-level) |
|---|---|---|---|
| 6  | 1 | `[0.0249]` | 0.0000 |
| 14 | 2 | `[0.0153, 0.0601]` | 0.0000 |
| 26 | 2 | `[0.0231, 0.0426]` | 0.0000 |

### Lo que NO se hizo en esta pasada

- **`CLAUDE.md`/`README.md`** no se actualizaron todavía con la rama nueva
  del diagrama de pipeline ni los comandos de ejemplo — quedó marcado
  "pendiente" en §0.
- **Flow C**: el prompt ya pide 2 puntos exactos (§6, riesgo #5), pero
  falta re-correr Molmo2 en el servidor para confirmar que el fix realmente
  cambia el comportamiento observado — no se puede validar sin GPU local.
- **No se corrió sobre el dataset completo** (850 objetos por tipo) — la
  validación de arriba es, a propósito, sobre los mismos 2 objetos que ya
  usaba el notebook, para aislar "¿el código de producción reproduce el
  prototipo?" de "¿el método funciona en general?" (esa segunda pregunta
  sigue abierta, ver §6 riesgos #1-3). Tampoco se validó con GT de 2-3
  planos reales — el objeto usado tiene 1 solo plano GT, así que el
  criterio de parada de `detect_planes_no_mesh` (y el recall/precision de
  `evaluate_plane_multiset`) solo se ejercitó para el caso "hay menos
  planos reales que candidatos detectados", no al revés.
