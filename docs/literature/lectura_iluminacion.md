# Lecturas: robustez a la exposición

Lista de lectura para justificar las condiciones de iluminación (*darker* 0,30 / media 0,70 / *brighter* 0,95) como una **prueba de robustez a la exposición** del método.

**Encuadre en la tesis:** las tres condiciones varían la intensidad de la luz, no su tipo ni su dirección. Por lo tanto, no evalúan si el sombreado ayuda a percibir la geometría; evalúan si el desempeño se mantiene cuando cambia la exposición del render. La literatura predice un efecto pequeño de las perturbaciones fotométricas en VLM, así que un resultado de "no cambia" es informativo: indica que el método no depende de la exposición exacta.

---

## Prioridad 1: lectura obligatoria

### 1. VLM-RobustBench
- **Referencia:** Rohit Saxena, Alessandro Suglia, Pasquale Minervini. *VLM-RobustBench: A Comprehensive Benchmark for Robustness of Vision-Language Models*. 2026. arXiv:2603.06148.
- **Enlace:** https://www.alphaxiv.org/abs/2603.06148
- **Por qué leerlo:** es la cita más directa. Evalúa robustez a 49 tipos de perturbación (incluidas las fotométricas) en VLM de cuatro familias, **incluida Molmo**. Concluye que los VLM son relativamente robustos a perturbaciones fotométricas y frágiles ante perturbaciones espaciales.
- **Qué extraer:**
  - Qué variante de Molmo evalúan y cómo le va en brillo/contraste.
  - Cómo definen los niveles de severidad (te sirve para justificar tener tres niveles).
  - La distinción entre tareas con *grounding* visual (MMBench) y de razonamiento: tu tarea de *pointing* está del lado del *grounding*.
- **Verificar:** aparece asociado tanto a ICML 2026 como a ICLR 2026. Confirma la venue final antes de citarlo.

### 2. Hendrycks y Dietterich (ImageNet-C)
- **Referencia:** Dan Hendrycks, Thomas Dietterich. *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations*. ICLR 2019. arXiv:1903.12261.
- **Enlace:** https://arxiv.org/abs/1903.12261
- **Por qué leerlo:** es el origen del protocolo de "corrupciones comunes" con niveles de severidad, incluida la corrupción de brillo. Los trabajos 3 y 4 se construyen sobre él.
- **Qué extraer:** cómo definen la corrupción de brillo y sus niveles; el argumento de por qué evaluar robustez con perturbaciones controladas.
- **Nota:** no lo verifiqué en esta sesión. Confirma la referencia en la fuente.

### 3. 3D Common Corruptions
- **Referencia:** Oğuzhan Fatih Kar, Teresa Yeo, Andrei Atanov, Amir Zamir. *3D Common Corruptions and Data Augmentation*. CVPR 2022, pp. 18963–18974. DOI: 10.1109/CVPR52688.2022.01839.
- **Enlace:** https://openaccess.thecvf.com/content/CVPR2022/html/Kar_3D_Common_Corruptions_and_Data_Augmentation_CVPR_2022_paper.html
- **Por qué leerlo:** argumenta que las perturbaciones de iluminación coherentes con la geometría de la escena son más realistas que los ajustes 2D sobre la imagen. Justifica que cambies la intensidad **dentro del renderizador** y no editando la imagen final.
- **Qué extraer:** la sección de corrupciones de iluminación y el argumento 2D vs. 3D.
- **Condición para citarlo:** solo si tus condiciones se implementan modificando la luz de la escena. Si son un ajuste de brillo sobre la imagen ya renderizada, esta cita no aplica.

---

## Prioridad 2: complementarias

### 4. Robustez de VLM a corrupciones de ImageNet-C
- **Referencia:** Muhammad Usama, Syeda Aishah Asim, Syed Bilal Ali, Syed Talal Wasim, Umair Bin Mansoor. *Analysing the Robustness of Vision-Language-Models to Common Corruptions*. 2025. arXiv:2504.13690.
- **Enlace:** https://arxiv.org/abs/2504.13690
- **Por qué leerlo:** analiza VLM ante las 19 corrupciones de ImageNet-C. Útil como segunda fuente que respalde a VLM-RobustBench.
- **Qué extraer:** resultados específicos para brillo y contraste; su explicación basada en el sesgo de los transformers hacia bajas frecuencias.
- **Verificar:** es un preprint; revisa si fue publicado.

### 5. B-AVIBench
- **Referencia:** *B-AVIBench: Towards Evaluating the Robustness of Large Vision-Language Model on Black-box Adversarial Visual-Instructions*. arXiv:2403.09346.
- **Enlace:** https://arxiv.org/abs/2403.09346
- **Por qué leerlo:** entre las corrupciones de imagen que evalúan en LVLM, el brillo está entre las menos efectivas para degradar el desempeño. Es evidencia adicional de que la exposición afecta poco a estos modelos.
- **Qué extraer:** la tabla de efectividad por tipo de corrupción (sección de corrupciones de imagen).
- **Verificar:** completa autores y venue.

---

## Prioridad 3: solo si agregas una condición de luz direccional

Estas lecturas **no** respaldan "robustez a la exposición". Sirven únicamente si incorporas una condición con luz direccional para estudiar el sombreado como pista de forma.

### 6. Ramachandran (shape from shading)
- **Referencia:** V. S. Ramachandran. *Perception of shape from shading*. Nature, 331(6152):163–166, 1988. DOI: 10.1038/331163a0.
- **Por qué leerlo:** referencia clásica de que el sombreado transmite forma tridimensional.

### 7. Shape from shading computacional
- **Referencia:** trabajo clásico de B. K. P. Horn sobre *shape from shading*.
- **Nota:** no verificado. Busca la referencia exacta si usas esta línea.

---

## Cómo usar esta literatura en la tesis

**En la metodología (capítulo 3):**

> Las condiciones *darker*, media y *brighter* evalúan la robustez del método ante cambios de exposición, una perturbación fotométrica frente a la cual los VLM han mostrado ser relativamente robustos [VLM-RobustBench; Usama 2025]. Siguiendo el protocolo de niveles de severidad de [Hendrycks 2019], se consideran tres intensidades de luz, aplicadas en el renderizador para que la iluminación sea coherente con la geometría del objeto [Kar 2022].

**En la pregunta de investigación PI3:** agregar "la exposición del render" a la lista de factores, solo si los experimentos se completan.

**En la discusión:** si el desempeño no cambia entre condiciones, interpretarlo como robustez a la exposición, en línea con la literatura. Si cambia, revisar primero si se debe a saturación o pérdida de contraste (ver lista de control).

---

## Lista de control antes de escribir

- [ ] Confirmar el tipo de luz del renderizador (ambiental, direccional o puntual).
- [ ] Confirmar que las condiciones se aplican en el renderizador y no sobre la imagen final.
- [ ] Renombrar "Flat (0,70)" como condición "media" o "de referencia".
- [ ] Medir y reportar, por condición: luminancia media del objeto y porcentaje de píxeles saturados (blancos y casi negros).
- [ ] Revisar el contraste objeto/fondo en *brighter*.
- [ ] Verificar venues y autores de los preprints (1, 4, 5) y la referencia de Hendrycks (2).
- [ ] Usar "robustez a la exposición" de forma consistente en Introducción, Metodología y Resultados.
