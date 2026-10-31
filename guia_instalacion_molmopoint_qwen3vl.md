# Guía de instalación: MolmoPoint-8B y Qwen3-VL-8B-Instruct

**Objetivo:** repetir los experimentos `axis_v04` y `plane_v05_1` con dos modelos adicionales, reutilizando el mismo pipeline sin malla (triangulación + evaluación) que se usa con Molmo2.

**Fuera del alcance de esta guía:** los prompts. Se asume que existen versiones `{prompt}_{nombre-modelo}` en el repositorio (por ejemplo, `axis_v04_molmopoint`, `plane_v05_1_qwen3vl`).

---

## 0. Resumen

| | Molmo2-8B (actual) | MolmoPoint-8B | Qwen3-VL-8B-Instruct |
|---|---|---|---|
| Hugging Face | `allenai/Molmo2-8B` | `allenai/MolmoPoint-8B` | `Qwen/Qwen3-VL-8B-Instruct` |
| Paper | arXiv 2601.10611 | arXiv 2603.28069 | arXiv 2511.21631 |
| Modelo de lenguaje | Qwen3-8B | Qwen3-8B | Qwen3 (propio) |
| Encoder de visión | SigLIP 2 | SigLIP 2 (`siglip-so400m-patch14-384`) | Propio |
| `transformers` | 4.57.1 | **4.57.1** | ≥ 4.57.0 |
| `trust_remote_code` | Sí | Sí | No |
| Cómo entrega los puntos | Texto con coordenadas | **Tokens especiales** (requiere metadatos para decodificar) | Texto JSON (`point_2d`) |
| Escala de coordenadas | 0–1000 | **Píxeles** | 0–1000 (relativa) |
| Multi-imagen | Sí | Sí (devuelve `image_num`) | Sí, pero el índice de imagen en la salida **no está documentado** |

**Conclusión práctica:** los tres modelos funcionan con `transformers==4.57.1`, así que pueden compartir un solo entorno. Lo que cambia entre ellos es cómo se leen los puntos, y eso se resuelve con una función de conversión por modelo (sección 4).

---

## 1. Entorno

### 1.1 Requisitos de hardware

Los tres modelos tienen ~8–9B parámetros. En BF16 necesitan **~18–20 GB de VRAM solo para los pesos**, más la memoria de la entrada. Con 26 vistas en una sola llamada, la entrada puede sumar decenas de miles de tokens (ver `contexto_tokens_molmo2.md`), así que una GPU de 24 GB puede quedarse corta. Las fichas no indican un mínimo; estas cifras son una estimación.

### 1.2 Entorno compartido (recomendado)

```bash
conda create -n vlm-pointing python=3.11 -y
conda activate vlm-pointing

# Versión común a los tres modelos
pip install transformers==4.57.1
pip install torch torchvision pillow einops accelerate decord2

# Molmo2 (ya lo usas)
pip install molmo_utils

# Qwen3-VL
pip install qwen-vl-utils==0.0.14

# Opcional, si la GPU lo soporta (ahorra memoria y tiempo)
pip install -U flash-attn --no-build-isolation
```

Si aparece algún conflicto, crea un entorno separado por modelo con los mismos comandos. La versión de `transformers` es lo único que debe coincidir con la indicada en la tabla.

### 1.3 Descargar y fijar las versiones de los pesos

Los repositorios de Hugging Face pueden cambiar. Para que los experimentos sean reproducibles, fija la revisión (commit) y regístrala en los resultados:

```bash
huggingface-cli download allenai/MolmoPoint-8B
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct
```

Luego, en el código, usa `revision="<hash>"` en `from_pretrained`. El hash aparece en la pestaña *Files and versions* de cada modelo.

Ojo: en Hugging Face hay copias de terceros de Qwen3-VL. La oficial empieza por `Qwen/`. Tampoco confundas la variante `Instruct` con `Thinking`: usa `Instruct`, que es la que compara el paper de Molmo2.

---

## 2. MolmoPoint-8B

### 2.1 Qué es

MolmoPoint (Clark et al., 2026; Allen AI) es un modelo de la familia Molmo que **no escribe las coordenadas como texto**. Para cada punto emite tres tokens especiales:

1. `<PATCH>`: elige el parche de la imagen que contiene el objetivo.
2. `<SUBPATCH>`: refina a una subregión de ese parche.
3. `<LOCATION>`: fija la posición dentro de la subregión.

Por eso, para obtener las coordenadas **se necesitan los metadatos que genera el procesador**. Se publicaron tres variantes: `MolmoPoint-8B` (general, imagen y video), `MolmoPoint-GUI-8B` (interfaces) y `MolmoPoint-Vid-4B` (video). **Usa `MolmoPoint-8B`.**

### 2.2 Ejemplo mínimo (adaptado de la ficha oficial)

```python
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "allenai/MolmoPoint-8B"

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, trust_remote_code=True, dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(
    MODEL_ID, trust_remote_code=True, padding_side="left"
)

vistas = [Image.open(p).convert("RGB") for p in rutas_de_las_vistas]  # tus renders
prompt = "..."  # leer desde tu archivo de prompts (axis_v04_molmopoint, etc.)

content = [{"type": "text", "text": prompt}]
content += [{"type": "image", "image": im} for im in vistas]
messages = [{"role": "user", "content": content}]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    padding=True,
    return_pointing_metadata=True,
)
metadata = inputs.pop("metadata")
inputs = {k: v.to("cuda") for k, v in inputs.items()}

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    output = model.generate(
        **inputs,
        logits_processor=model.build_logit_processor_from_inputs(inputs),
        max_new_tokens=200,   # aumentar con muchas vistas (ver 2.3)
        do_sample=False,      # decodificación determinista
    )

generated = output[:, inputs["input_ids"].size(1):]
text = processor.post_process_image_text_to_text(
    generated, skip_special_tokens=False, clean_up_tokenization_spaces=False
)[0]

points = model.extract_image_points(
    text,
    metadata["token_pooling"],
    metadata["subpatch_mapping"],
    metadata["image_sizes"],
)
points = np.array(points)   # cada fila: [object_id, image_num, x, y]
```

### 2.3 Puntos a cuidar

- **`skip_special_tokens=False` es obligatorio.** Si se eliminan los tokens especiales, se pierden los puntos.
- **El `logits_processor` es obligatorio.** Sin él, la generación de los tokens de puntos no funciona como se espera.
- **`image_num` indica a qué vista pertenece cada punto.** Comprueba si empieza en 0 o en 1 con una prueba de dos imágenes (sección 5) y alinéalo con el índice de cámara del `manifest.json`.
- **`x`, `y` vienen en píxeles** de la imagen original, no en 0–1000.
- **`max_new_tokens`:** cada punto usa ~3 tokens, pero con 26 vistas y varios puntos por vista, 200 puede quedarse corto. Revisa si las salidas se cortan y súbelo si es necesario.
- **Decodificación determinista** (`do_sample=False`) en todos los modelos, para que la comparación sea justa.

---

## 3. Qwen3-VL-8B-Instruct

### 3.1 Qué es

Modelo multimodal de Alibaba (Qwen Team). Para *pointing* devuelve **texto en formato JSON**, con la clave `point_2d`, en una **escala relativa de 0 a 1000**. El repositorio oficial incluye un cookbook de *grounding* 2D (`cookbooks/2d_grounding.ipynb`) con el formato de prompt recomendado. El paper de Molmo2 indica que los modelos Qwen no siguen bien el formato de puntos de Molmo, así que los prompts deben adaptarse (eso queda fuera de esta guía).

### 3.2 Ejemplo mínimo (adaptado de la ficha oficial)

```python
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID, dtype="auto", device_map="auto"
    # , attn_implementation="flash_attention_2"   # si instalaste flash-attn
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

vistas = [Image.open(p).convert("RGB") for p in rutas_de_las_vistas]
prompt = "..."  # leer desde tu archivo de prompts (axis_v04_qwen3vl, etc.)

content = [{"type": "image", "image": im} for im in vistas]
content.append({"type": "text", "text": prompt})
messages = [{"role": "user", "content": content}]

inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt",
).to(model.device)

output = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
output = [o[len(i):] for i, o in zip(inputs.input_ids, output)]
text = processor.batch_decode(output, skip_special_tokens=True)[0]
```

### 3.3 Puntos a cuidar

- **La salida es texto**, no una lista de puntos. Hay que parsear el JSON (el cookbook trae una función tipo `decode_json_points`). Prepárate para salidas mal formadas: bloques ```` ```json ````, comas sobrantes o texto adicional.
- **Escala 0–1000, relativa a cada imagen.** Conversión: `x_px = x / 1000 * ancho`, `y_px = y / 1000 * alto`. Algunos cookbooks usan 999 como denominador; verifícalo en `2d_grounding.ipynb` antes de fijarlo.
- **Multi-imagen: el índice de vista no está garantizado.** La documentación no indica cómo Qwen3-VL asigna puntos a imágenes cuando hay varias. El prompt tiene que pedir explícitamente el índice de la imagen en el JSON (por ejemplo, `{"image": 3, "point_2d": [x, y]}`). Si el modelo no lo respeta, la alternativa es **una llamada por vista**. Si llegas a usar esa alternativa, documéntalo: es un protocolo distinto al de Molmo2 (todas las vistas juntas) y afecta la comparación.
- **Resolución dinámica.** Qwen3-VL ajusta el número de tokens a la resolución de cada imagen, y el procesador acepta límites de píxeles (`min_pixels` / `max_pixels`). Déjalos fijos y regístralos: cambian los tokens por imagen y, por lo tanto, la comparación entre resoluciones.
- **Servirlo con vLLM** (opcional, para correr los 850 objetos más rápido): `vllm serve Qwen/Qwen3-VL-8B-Instruct` con vLLM ≥ 0.11.0, o la imagen Docker `qwenllm/qwenvl:qwen3vl-cu128`. Para empezar, `transformers` es suficiente.

---

## 4. Llevar las salidas al formato del pipeline

El pipeline sin malla lee `molmo_multiview_<ID>.json` con coordenadas en la escala 0–1000 de Molmo2. La forma más simple de reutilizarlo sin cambios es convertir la salida de cada modelo **al mismo formato y la misma escala**, y guardarla en una carpeta propia por modelo.

```python
def a_escala_molmo2(x, y, ancho, alto, modelo):
    """Devuelve (x, y) en la escala 0-1000 que usa el pipeline."""
    if modelo == "molmopoint":          # píxeles -> 0-1000
        return x / ancho * 1000, y / alto * 1000
    if modelo == "qwen3vl":             # ya viene en 0-1000 (verificar denominador)
        return x, y
    if modelo == "molmo2":
        return x, y
    raise ValueError(modelo)
```

Revisa también:

- **Índice de vista:** cada punto debe quedar asociado a la cámara correcta del `manifest.json`. Es la fuente de error más probable al cambiar de modelo.
- **Mismo orden de imágenes** que en Molmo2.
- **Mismo nombre de archivo por objeto** y carpeta separada por modelo, por ejemplo `results/<modelo>/axis_v04/...`.
- **Registro de fallas:** guarda el texto crudo de la salida junto a los puntos parseados, para poder auditar las salidas vacías o mal formadas.

---

## 5. Prueba de humo (antes de correr los 850 objetos)

1. **Una imagen, un punto fácil** (por ejemplo, "point to the top of the object"). Dibuja el punto sobre la imagen y verifica que cae donde debe. Esto valida la escala de coordenadas.
2. **Dos imágenes distintas** en una llamada, con un objeto que solo aparece en la segunda. Verifica que el punto quede asignado a la segunda imagen. Esto valida el índice de vista (y si empieza en 0 o en 1).
3. **Un objeto completo** con 6 vistas y el prompt real. Pasa la salida por el pipeline y confirma que la triangulación corre sin errores.
4. **Una muestra de ~30 objetos** con 6, 14 y 26 vistas. Revisa la cobertura (objetos con predicción), las salidas mal formadas y el tiempo y la memoria por llamada.

Solo después de esto conviene lanzar las corridas completas.

---

## 6. Qué mantener igual que con Molmo2

Para que la comparación entre modelos sea justa:

- [ ] Mismos objetos, mismas vistas (nv ∈ {6, 14, 26}), misma resolución (224 px) e iluminación.
- [ ] Mismo protocolo de envío (todas las vistas en una llamada), o documentar la excepción si Qwen3-VL requiere una llamada por vista.
- [ ] Decodificación determinista en los tres modelos.
- [ ] Mismo pipeline de triangulación y evaluación, sin cambios.
- [ ] Misma línea base aleatoria (`baseline_aleatorio.md`) con la cobertura de cada modelo.
- [ ] Registrar: revisión (hash) del modelo, versión de `transformers`, `max_new_tokens`, límites de píxeles (Qwen3-VL) y tokens de entrada por llamada.

---

## 7. Fuentes

**MolmoPoint**
- Clark, C., Yang, Y., Park, J. S., Ma, Z., Zhang, J., Tripathi, R., Salehi, M., Lee, S., Anderson, T., Han, W., y Krishna, R. (2026). *MolmoPoint: Better Pointing for VLMs with Grounding Tokens*. arXiv:2603.28069. https://arxiv.org/abs/2603.28069
- Ficha del modelo: https://huggingface.co/allenai/MolmoPoint-8B
- Blog de Ai2: https://allenai.org/blog/molmopoint
- Código (repositorio de Molmo2, enlazado desde el blog): https://github.com/allenai/molmo2

**Qwen3-VL**
- Qwen Team (2025). *Qwen3-VL Technical Report*. arXiv:2511.21631. https://arxiv.org/pdf/2511.21631
- Ficha del modelo: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- Repositorio: https://github.com/QwenLM/Qwen3-VL (cookbook `cookbooks/2d_grounding.ipynb`)
- Resumen de *grounding* 2D (DeepWiki, no oficial): https://deepwiki.com/QwenLM/Qwen3-VL/5.2-spatial-understanding-and-2d-grounding

**Molmo2**
- Clark et al. (2026). *Molmo2: Open Weights and Data for Vision-Language Models with Video Understanding and Grounding*. arXiv:2601.10611.

**Nota de verificación:** la información de MolmoPoint proviene de su ficha en Hugging Face y del blog de Ai2. El formato de salida multi-imagen de Qwen3-VL y su denominador de escala (1000 o 999) no están confirmados en fuentes oficiales; valídalos con la prueba de humo de la sección 5.
