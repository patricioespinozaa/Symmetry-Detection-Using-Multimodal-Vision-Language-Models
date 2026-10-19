# Contexto: saturación de Molmo2 por número de vistas y resolución

Documento de contexto para `contar_tokens_molmo2.py`. Alcance: **pipeline sin malla**
(`Mapping/estimate_symmetry_no_mesh.py`, triangulación de rayos multi-vista; ver
`docs/pipeline_sin_malla.md`). Las cifras de la sección 1 salen de
`docs/resultados/tablas_448/`, `tablas_1136/` y `tablas_iluminacion/iluminacion_224.tex`.

## 1. El problema

En el pipeline sin malla, las predicciones **empeoran al aumentar el número de vistas**,
y la caída es **más fuerte cuanto mayor es la resolución**. Datos reales (850 objetos por
tipo, iluminación `flat`, Flujo A, `--prompt-mode auto`, 1 plano por objeto):

**Eje — `axis_v04`** (AUC₄₅°; entre paréntesis, objetos con predicción de 850)

| Resolución | 6 vistas | 14 vistas | 26 vistas |
|---|---|---|---|
| 224 px | 0,582 (831) | 0,580 (816) | 0,501 (806) |
| 448 px | **0,605** (832) | 0,544 (801) | 0,282 (751) |
| 1136 px | 0,572 (819) | 0,413 (779) | 0,184 (769) |

**Plano — `plane_v05_1`** (AUC₄₅°; entre paréntesis, objetos con predicción de 850)

| Resolución | 6 vistas | 14 vistas | 26 vistas |
|---|---|---|---|
| 224 px | 0,494 (796) | 0,381 (684) | 0,277 (827) |
| 448 px | 0,528 (742) | 0,289 (540) | 0,118 (778) |
| 1136 px | **0,545** (717) | 0,230 (463) | **0,019 (114)** |

Tres observaciones:

1. **Con 6 vistas, más resolución no perjudica** (incluso ayuda: el plano pasa de 0,494 a
   0,545). El problema aparece en la **interacción entre resolución y número de vistas**.
2. Con 26 vistas, la caída es monótona con la resolución en ambos tipos de simetría
   (eje 0,50 → 0,28 → 0,18; plano 0,28 → 0,12 → 0,02). Pero 448 y 1136 px producen
   exactamente los mismos tokens (sección 4), así que la caída de 448 a 1136 px no se explica
   por el largo de la entrada.
3. **Plano a 1136 px con 26 vistas colapsa también en cobertura**: solo 114 de 850 objetos
   tienen predicción. Su entrada tiene el mismo largo que la de 448 px con 26 vistas (778
   objetos con predicción), así que la causa tampoco es el largo (ver el pendiente de la
   sección 4).

Con más vistas, el resultado debería mejorar, porque la triangulación tiene más rayos.

**Cómo llega la entrada al modelo** (`molmo_multiview_runner.py::_call_model`):

- Con `--prompt-mode auto` (el usado en todas las corridas), `n_views = 1` usa el prompt
  *single* y `n_views > 1` usa el prompt *multi*: **todas las vistas van en una sola llamada**.
- El mensaje es un único turno de usuario con **el texto primero y después las N imágenes**.
- `max_new_tokens = 2048` (constante fija).
- Las vistas se eligen con `linspace` sobre las 114 vistas de la esfera de Fibonacci
  (`get_n_views_entries`).
- En los Flujos B y C hay **una llamada previa aparte** (una sola vista) para la descripción.
  Esa llamada no comparte contexto con la principal, pero su descripción se agrega al texto
  del prompt principal.

**Hipótesis:** el modelo se satura. La entrada supera el largo con que Molmo2 fue
entrenado, o incluso su contexto máximo, y eso degrada o trunca la respuesta.

---

## 2. Lo que se sabe de Molmo2-8B

El modelo de lenguaje es Qwen3-8B, pero el límite relevante es el de **Molmo2**, no el de Qwen3.

| Dato | Valor | Fuente |
|---|---|---|
| Modelo de lenguaje | Qwen3-8B | Model card |
| Encoder de visión | SigLIP 2 | Model card |
| Contexto máximo (`max_position_embeddings`) | **36.864 tokens** | `config.json` |
| Largo de secuencia del fine-tuning principal | **16.384 tokens** | Paper, sec. 3.2 |
| Largo del fine-tuning de contexto largo | 36.864 tokens (solo con video) | Paper |
| Tamaño de cada recorte | 378 × 378 px | `preprocessor_config.json` |
| Tamaño de parche | 14 px → 27 × 27 parches por recorte | `preprocessor_config.json` |
| Pooling | 2 × 2 parches → 1 token | `preprocessor_config.json` y paper |
| Recortes por imagen | 1 vista global + hasta 8 recortes (`max_crops = 8`, márgenes de solapamiento 4 y 4). El paper usa hasta 24 en inferencia | `preprocessor_config.json` y paper |
| Imágenes por ejemplo en los datos multi-imagen de entrenamiento | 2 a 5 | Paper, sec. 2 |

**Implicación:** con 14 o 26 vistas el modelo está fuera de lo que vio en entrenamiento
(2–5 imágenes), aunque la entrada quepa en el contexto.

---

## 3. Por qué la saturación pesa más en el pipeline sin malla

El pipeline con malla tiene una red de seguridad: el *ray casting* contra la malla descarta
los puntos que no tocan el objeto. El pipeline sin malla **no tiene esa red de seguridad**:

- **Cada vista útil necesita ≥ 2 puntos** para definir su plano de interpretación (con
  `--filter-off-object`, ≥ 2 puntos sobre el objeto: `MIN_POINTS_ON_OBJECT_DEFAULT = 2`). Si en una vista queda 1 solo punto, esa vista se
  descarta. Con < 2 vistas útiles no hay predicción para ese `n_views`.
- **Un punto asignado a la imagen equivocada se triangula con la cámara equivocada.** Un
  punto bien ubicado se convierte en un rayo erróneo. El filtro opcional de silueta
  (`--filter-off-object`) solo descarta puntos que caen sobre el fondo de esa imagen.
- `parse_multi_coords` **descarta en silencio** los grupos con índice de imagen fuera de
  rango y las salidas mal formateadas. En los resultados, esto aparece como baja cobertura,
  sin ningún mensaje de error.

Por eso, además del largo de la entrada, interesa medir **cuántas vistas quedan con
≥ 2 puntos** y **cuántas veces el modelo escribe índices de imagen imposibles**.

---

## 4. Tokens medidos y primeros hallazgos

Resultados en `MolmoPointing/ContextLimits/resultados/` (CSV de los dos modos) y tablas LaTeX
en `docs/resultados/tablas_tokens/tokens_molmo2.tex`, generadas con
`python utils/generar_tablas_tokens.py`. El conteo del procesador coincide con
`n_input_tokens` de las corridas en las 31 combinaciones comparables.

Tokens de entrada por llamada (`axis_v04`; `plane_v05_1` suma 26 tokens de texto):

| Resolución | Tokens por imagen | 6 vistas | 14 vistas | 26 vistas |
|---|---|---|---|---|
| 224 px | 410 | 2.815 | 6.124 | 11.092 |
| 448 px | 752 | 4.867 | 10.912 | **19.984** (> 16.384) |
| 1136 px | 752 | 4.867 | 10.912 | **19.984** (> 16.384) |

Ninguna combinación evaluada (6/14/26 vistas) supera el contexto máximo de 36.864. Eso
pasa recién con 62 vistas a 448/1136 px y con 114 a 224 px. No se pierde ninguna imagen
(`imagenes_perdidas = 0`). El texto aporta solo 237–825 tokens.

**Hallazgo 1: 448 y 1136 px dan exactamente los mismos tokens.** El procesador produce el
mismo número de recortes para ambas resoluciones. Aun así, a igual largo de entrada, 1136 px
rinde peor (eje con 14 vistas: AUC 0,544 vs. 0,413; con 26: 0,282 vs. 0,184). **El largo del
contexto no explica la diferencia entre 448 y 1136 px**: tiene que haber otra causa, por
ejemplo cómo se reduce y recorta una imagen de 1136 px.

**Hallazgo 2: con muchas imágenes, Molmo2 cambia de formato de salida.** En vez de
`<points coords=...>` responde con `<tracks coords=...>`, su formato de *tracking* de video.
El parser del runner extrae los puntos igual, porque solo busca `coords=`. Con `axis_v04` a
26 vistas pasa en ~100 % de las llamadas en las tres resoluciones. Con 14 vistas, la
frecuencia crece con la resolución: 2 % (224), 14 % (448) y 54 % (1136). El cambio de formato
no implica por sí solo un mal resultado: eje a 224 px con 26 vistas usa `<tracks>` en casi
todas las llamadas y mantiene AUC 0,50.

**Hallazgo 3: las vistas útiles no son el cuello de botella.** En `axis_v04` y `plane_v05_1`,
≥ 99,6 % de las vistas reciben al menos 2 puntos en todas las combinaciones. Sí crecen con las
vistas los puntos duplicados dentro de una vista (1–2 % con 6 vistas, hasta ~7 % con 14–26).

**Pendiente:** el CSV de las corridas no tiene llamadas de `plane_v05_1` a 1136 px con
26 vistas, pero la evaluación sí reporta 114 objetos. Hay que revisar en el servidor dónde
quedaron esos `molmo_multiview` y por qué faltan los otros 736 objetos.

---

## 5. Qué hace el script

`contar_tokens_molmo2.py` tiene dos modos.

### Modo `procesador` (por defecto)

No ejecuta el modelo. Usa solo el procesador (tokenizador + preprocesamiento de
imágenes), así que corre en CPU en segundos. Reproduce el mensaje real del runner:

- **Prompt** tomado de `prompts_registry` con `--prompt-id`. Usa el prompt *single* si
  `n_views = 1` y el *multi* si `n_views > 1`, igual que `--prompt-mode auto`.
- **Mismo orden que `_call_model`:** texto primero, después las imágenes.
- `AutoProcessor.from_pretrained(..., trust_remote_code=True, use_fast=True)`, como
  `get_model()`.
- Imágenes grises sintéticas, porque los tokens dependen del **tamaño** de la imagen y no
  de su contenido. Con `--renders <obj>/<size>/<lighting>` usa renders reales, elegidos
  con el mismo `linspace` del runner.
- `--max-crops N` asigna `processor.image_processor.max_crops`.

| Columna | Qué significa |
|---|---|
| `tokens_total` | Largo total de la entrada |
| `tokens_imagen` / `tokens_texto` | Cuántos tokens son de imagen y cuántos de texto |
| `tokens_por_imagen` | Promedio de tokens de imagen por vista |
| `imagenes_en_entrada` | Cuántas imágenes llegaron realmente al modelo |
| `imagenes_perdidas` | Vistas enviadas que no aparecen en la entrada |
| `parches_alta_res` / `parches_baja_res` | Tokens de los recortes y de la vista global |
| `supera_largo_sft` | Si el total supera 16.384 |
| `supera_contexto_max` | Si el total supera 36.864 |

Los IDs de tokens de imagen se tomaron del `config.json` de Molmo2-8B
(`image_start_token_id = 151936`, `image_patch_id = 151938`, `image_col_id = 151939`,
`low_res_image_start_token_id = 151940`, `image_low_res_id = 151942`, etc.).

### Modo `json` (corridas existentes, sin `transformers`)

Recorre `<renders_root>/<symmetry_type>/<obj>/<size>/<lighting>/molmo_multiview*.json` y
agrupa por `symmetry_type × experimento × size × lighting × n_views`. Salta los JSON que son
enlaces simbólicos (los `<X>_ndcfix_nomesh` y `<X>_nomesh` de `utils/rerun_ndcfix_completo.py`
apuntan al `molmo_multiview_<X>.json` original); `--incluir-enlaces` los cuenta igual.

| Columna | Qué significa |
|---|---|
| `tokens_media` / `tokens_min` / `tokens_max` | `n_input_tokens` real de las llamadas |
| `frac_supera_largo_sft` / `frac_supera_contexto_max` | Fracción de llamadas sobre 16.384 / 36.864 |
| `frac_vistas_con_puntos` | Vistas con ≥ 1 punto / vistas enviadas |
| `frac_vistas_utiles` | Vistas con **≥ 2 puntos** / vistas enviadas: las que puede usar la triangulación |
| `puntos_por_vista` | Puntos parseados por vista enviada |
| `frac_puntos_duplicados` | Puntos con la misma (x, y) que otro de la misma vista |
| `frac_formato_points` / `frac_formato_tracks` | Llamadas que respondieron con `<points coords=...>` / con `<tracks coords=...>` (formato de *tracking* de video) |
| `frac_salidas_sin_coords` | Llamadas sin ningún bloque `coords=` (vacías o en prosa) |
| `frac_salidas_cortadas` | Llamadas que abren `coords="` y nunca lo cierran (p. ej. por `max_new_tokens`) |
| `frac_llamadas_fuera_de_rango` | Llamadas con algún índice de imagen fuera de `[1, N]` en la salida cruda: señal directa de confusión de índices |

**Ojo con los Flujos B y C:** su `n_input_tokens` incluye los tokens de la llamada previa
de descripción (`extra_tokens`), que es una llamada aparte. Para esas filas, el valor
sobreestima el contexto de la llamada principal en el largo de una llamada de una sola
imagen. El script lo avisa.

---

## 6. Cómo usarlo

Desde `MolmoPointing/ContextLimits/`:

```bash
# Conteo con el prompt real
python contar_tokens_molmo2.py --prompt-id axis_v04
python contar_tokens_molmo2.py --prompt-id plane_v05_1 --vistas 6 14 26

# Control: un solo recorte por imagen
python contar_tokens_molmo2.py --prompt-id axis_v04 --max-crops 1 --salida tokens_maxcrops1.csv

# Con renders reales de un objeto, a una resolución
python contar_tokens_molmo2.py --prompt-id axis_v04 --renders ruta/a/obj/448/flat --resoluciones 448

# Tokens reales e indicadores de saturación de las corridas existentes (en el servidor)
python contar_tokens_molmo2.py --modo json --renders-root /ruta/renders \
    --experimentos axis_v04 plane_v05_1 --salida saturacion_corridas.csv
```

### Antes de confiar en los números

1. **Compara ambos modos.** `tokens_total` (procesador) debería coincidir con
   `tokens_media` (json) para el mismo prompt, resolución y número de vistas en el Flujo A.
   Si no coinciden, el armado del mensaje en `contar` se desvió del de `_call_model`.
   Verificado: coinciden en las 31 combinaciones comparables (`generar_tablas_tokens.py`
   lo comprueba cada vez).
2. **`imagenes_perdidas` es 0** en todas las combinaciones medidas: el modelo recibe todas
   las vistas.

---

## 7. Plan completo

| Paso | Qué hacer | Estado |
|---|---|---|
| 1 | Confirmar cómo se envían las vistas | ✅ Una sola llamada; texto primero y después las imágenes (`_call_model`, `--prompt-mode auto`) |
| 2 | Medir los tokens reales | ✅ Medidos con ambos modos (sección 4) |
| 3 | Indicadores de saturación en las corridas existentes | ✅ Corrido. Volver a correr `--modo json` con la versión actual para tener `frac_formato_tracks` |
| 4 | Graficar métricas contra los tokens totales | Pendiente |
| 5 | Experimentos de control en una muestra pequeña | Pendiente |

### Paso 3: indicadores de saturación (por llamada)

Ya disponibles en el JSON o calculados por el modo `json`:

- Número de puntos y vistas con al menos un punto (`n_points`, `n_images_with_points`).
- **Vistas con ≥ 2 puntos**: el requisito real de la triangulación sin malla.
- **Índices de imagen fuera de rango** en la salida cruda. Con muchas imágenes, el modelo
  puede confundir los índices, y eso arruina la triangulación aunque los puntos estén bien
  ubicados.
- Salidas sin bloque `<points ...>` (vacías, mal formateadas o cortadas por
  `max_new_tokens = 2048`).
- Puntos duplicados dentro de una vista.

Pendiente (requiere instrumentar el runner): patrones equiespaciados o centrados, y tiempo y
memoria por llamada.

### Paso 4: análisis contra los tokens

Une el CSV del modo `json` con los resultados de `evaluate.py` (por experimento, size y
`n_views`). Grafica cada métrica (AUC₄₅°, error angular, cobertura, `frac_vistas_utiles`) en
función de `tokens_media`, con un color por resolución y un marcador por número de vistas.
Marca las líneas de 16.384 y 36.864.

- **Si las curvas de todas las resoluciones caen sobre una sola curva:** la causa es el
  largo del contexto.
- **Si a igual número de tokens la resolución alta sigue rindiendo peor:** hay otra causa,
  por ejemplo que el objeto quede partido entre recortes.

### Paso 5: controles (muestra de ~30–50 objetos, `--max-objects`)

| Control | Comparación | Qué aísla | ¿Soportado hoy? |
|---|---|---|---|
| **Conjunta vs. separada** | Las mismas vistas en una llamada (`--prompt-mode auto`) vs. una llamada por vista (`--prompt-mode single`) | **El efecto del contexto multi-imagen (el control más directo)** | ✅ Ya existe en el runner. Usa otro `--experiment-id` |
| Mismos tokens, distinta combinación | 26 vistas a 224 px vs. 6 vistas a 1136 px (~10.000 tokens cada una) | Largo del contexto vs. resolución | ✅ Las corridas ya existen; solo hay que compararlas |
| Misma imagen, distintos tokens | 1136 px con `max_crops = 1` vs. por defecto | El efecto de los recortes | ❌ Requiere exponer `max_crops` en `get_model()` |
| Posición en la secuencia | La misma vista en primera, media y última posición | Confusión de índices de imagen | ❌ Requiere permutar el orden de las vistas en `process_object` |

Nota sobre el control "conjunta vs. separada": con `--prompt-mode single`, el prompt
*single* pide puntos sobre una sola imagen. El pipeline sin malla los usa igual, porque cada
vista conserva su cámara en `images_sent`.

---

## 8. Cómo podría redactarse en la tesis

Con lo medido hasta ahora (sección 4), una redacción que los datos sostienen:

> El deterioro del desempeño al aumentar el número de vistas coincide con entradas fuera de
> la distribución de entrenamiento de Molmo2: más imágenes por ejemplo que en sus datos
> multi-imagen (2 a 5) y, a 448 y 1136 px con 26 vistas, más tokens que el largo de secuencia
> de su *fine-tuning* principal (16.384). Con muchas imágenes, el modelo además cambia a su
> formato de salida de *tracking* de video (`<tracks>`). El largo de la entrada no explica por
> sí solo la diferencia entre 448 y 1136 px, que producen el mismo número de tokens.

Si el control "conjunta vs. separada" muestra que las llamadas por vista rinden mejor, eso
además justifica cambiar el pipeline a una llamada por vista (`--prompt-mode single`).

---

## 9. Fuentes

- Resultados del repositorio: `docs/resultados/tablas_448/`, `docs/resultados/tablas_1136/`,
  `docs/resultados/tablas_iluminacion/iluminacion_224.tex`.
- Código: `MolmoPointing/molmo_multiview_runner.py` (`_call_model`, `get_n_views_entries`,
  `parse_multi_coords`, `process_object`), `Mapping/estimate_symmetry_no_mesh.py`.
- Molmo2-8B, model card: https://huggingface.co/allenai/Molmo2-8B
- `config.json`: https://huggingface.co/allenai/Molmo2-8B/blob/main/config.json
- `preprocessor_config.json`: https://huggingface.co/allenai/Molmo2-8B/blob/main/preprocessor_config.json
- Clark et al. (2026). *Molmo2: Open Weights and Data for Vision-Language Models with Video
  Understanding and Grounding*. arXiv:2601.10611. https://arxiv.org/pdf/2601.10611
- Versión CVPR 2026: https://openaccess.thecvf.com/content/CVPR2026/papers/Clark_Molmo2_Open_Weights_and_Data_for_Vision-Language_Models_with_Video_CVPR_2026_paper.pdf
