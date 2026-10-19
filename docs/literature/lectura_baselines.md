# Lecturas: líneas base aleatorias y triviales

Lista de lectura para justificar las líneas base contra las que se compara el método (PI1 y PI4). Cada sección responde a una pregunta distinta que el jurado puede hacer:

| Línea base | Pregunta que responde | Sección |
|---|---|---|
| Eje/plano aleatorio uniforme | ¿El método es mejor que el azar? | 1 |
| Predictor de pose canónica ("siempre X") | ¿El método usa la imagen, o solo explota cómo está alineado el dataset? | 2 |
| Punto en el centro de la imagen | ¿Molmo2 marca estructura, o solo el centro del encuadre? | 3 |
| Prueba estadística de la diferencia | ¿La mejora sobre la línea base es real o ruido? | 4 |

---

## 1. Línea base aleatoria: direcciones uniformes en la esfera

**Idea:** el predictor aleatorio elige una dirección (normal del plano o eje) uniformemente sobre la esfera. Para que sea uniforme de verdad no basta con muestrear ángulos esféricos uniformes (eso concentra puntos en los polos); hay que normalizar un vector gaussiano.

### 1.1 Marsaglia (1972)
- **Referencia:** George Marsaglia. *Choosing a Point from the Surface of a Sphere*. The Annals of Mathematical Statistics, 43(2):645–646, 1972.
- **Para qué:** método estándar para muestrear direcciones uniformes en la esfera.

### 1.2 Muller (1959)
- **Referencia:** Mervin E. Muller. *A Note on a Method for Generating Points Uniformly on N-Dimensional Spheres*. Communications of the ACM, 2(4):19–20, 1959.
- **Para qué:** justifica el muestreo normalizando un vector gaussiano, que es lo que probablemente hace tu código.

### Valor analítico del azar (para reportar junto al empírico)

Los ejes y las normales de plano no tienen signo (n y −n son la misma solución), así que el error angular es θ = arccos|⟨n̂, n*⟩| ∈ [0°, 90°]. Si n̂ es uniforme en la esfera, |⟨n̂, n*⟩| es uniforme en [0, 1], de modo que:

- **Error medio esperado:** E[θ] = ∫₀¹ arccos(u) du = 1 rad ≈ **57,3°**
- **Probabilidad de acertar con tolerancia τ:** P(θ ≤ τ) = 1 − cos τ. Por ejemplo, ≈ 1,5 % para τ = 10° y ≈ 13,4 % para τ = 30°.

Esta derivación es elemental; puedes incluirla en la metodología sin cita o en un anexo. Reporta además el valor empírico del predictor aleatorio sobre tus objetos, para confirmar que coincide.

**Cuidado con los objetos con varios planos:** si el error se calcula contra el plano más cercano de los k válidos, el azar mejora (el mínimo de k errores es menor). El valor de 57,3° vale solo para k = 1. Calcula la línea base aleatoria con la misma regla de evaluación que usas para el método.

---

## 2. Predictores triviales que explotan sesgos del dataset

**Idea:** un predictor que ignora la imagen y devuelve siempre la respuesta más frecuente (p. ej., "siempre el plano X" en ShapeNet). Si este predictor es competitivo, parte del desempeño del método puede venir de la alineación canónica del dataset y no de la imagen.

### 2.1 ShapeNet: la alineación canónica
- **Referencia:** Angel X. Chang et al. *ShapeNet: An Information-Rich 3D Model Repository*. arXiv:1512.03012, 2015.
- **Para qué:** documenta que los modelos de ShapeNetCore están alineados de forma consistente dentro de cada categoría. Es la fuente del sesgo que explota el predictor "siempre X".
- **Qué extraer:** la sección sobre la alineación (orientación "arriba" y "frente") de los modelos.

### 2.2 Torralba y Efros (2011)
- **Referencia:** Antonio Torralba, Alexei A. Efros. *Unbiased Look at Dataset Bias*. CVPR 2011.
- **Para qué:** referencia clásica de que los datasets de visión tienen sesgos sistemáticos que los modelos explotan y que inflan los resultados. Justifica en general la necesidad de controlar sesgos del dataset.

### 2.3 Líneas base "sin imagen" en VLM / VQA

Estas son el paralelo directo de tu predictor trivial en la literatura multimodal: responder sin mirar la imagen para medir cuánto del desempeño viene de priors.

- **Goyal et al. (2017).** Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, Devi Parikh. *Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering*. CVPR 2017.
  - Muestran que modelos de VQA obtienen buen desempeño explotando priors del lenguaje sin usar la imagen, y rebalancean el dataset para evitarlo.
- **Agrawal et al. (2018).** Aishwarya Agrawal, Dhruv Batra, Devi Parikh, Aniruddha Kembhavi. *Don't Just Assume; Look and Answer: Overcoming Priors for Visual Question Answering* (VQA-CP). CVPR 2018.
  - Cambian la distribución de respuestas entre entrenamiento y prueba para exponer modelos que dependen de priors.
- **Chen et al. (2024), MMStar.** Lin Chen et al. *Are We on the Right Way for Evaluating Large Vision-Language Models?* NeurIPS 2024. arXiv:2403.20330.
  - La más actual y la más cercana a tu caso: muestran que en varios benchmarks de VLM muchas preguntas se pueden responder sin la imagen, y proponen evaluar explícitamente el desempeño "sin contenido visual". Justifica comparar contra un predictor que no usa la imagen.

**Cómo usarlo:** "Siguiendo la práctica de evaluar el desempeño alcanzable sin información visual [Goyal 2017; Chen 2024], se incluye un predictor que devuelve siempre el plano (o eje) más frecuente del conjunto, explotando la alineación canónica de ShapeNet [Chang 2015]."

---

## 3. Línea base del centro de la imagen (para el colapso al centro)

**Idea:** en predicción de fijaciones visuales, un mapa que solo marca el centro de la imagen supera a muchos modelos de saliencia. Es exactamente el análogo de tu hallazgo de que Molmo2 colapsa al centro. Usar un "predictor del centro" como línea base de los puntos 2D permite medir si Molmo2 aporta algo más que eso.

### 3.1 Tatler (2007)
- **Referencia:** Benjamin W. Tatler. *The central fixation bias in scene viewing: Selecting an optimal viewing position independently of motor biases and image feature distributions*. Journal of Vision, 7(14):4, 2007.
- **Para qué:** documenta el sesgo central en la observación de escenas, independiente del contenido de la imagen.

### 3.2 Judd et al. (2009)
- **Referencia:** Tilke Judd, Krista Ehinger, Frédo Durand, Antonio Torralba. *Learning to Predict Where Humans Look*. ICCV 2009.
- **Para qué:** muestran que una línea base que solo usa la distancia al centro predice las fijaciones mejor que varios modelos de saliencia. Es la cita para justificar el "predictor del centro" como línea base exigente.

**Advertencia de interpretación:** en tus renders el objeto está centrado, así que "centro de la imagen" y "centro del objeto" coinciden. Para separar ambos efectos necesitas el control de desplazar el objeto dentro del encuadre (ver la discusión sobre los renders).

---

## 4. Comparación estadística contra las líneas base

**Idea:** no basta con que el método tenga menor error medio; hay que mostrar que la diferencia no es atribuible al azar, con pruebas apareadas sobre los mismos objetos.

### 4.1 Dror et al. (2018)
- **Referencia:** Rotem Dror, Gili Baumer, Segev Shlomov, Roi Reichart. *The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing*. ACL 2018.
- **Para qué:** guía práctica para elegir pruebas de significancia al comparar sistemas (pruebas apareadas, bootstrap, permutación). Aunque es de NLP, el razonamiento se aplica a comparar el método contra las líneas base objeto a objeto.

### 4.2 Bootstrap
- **Referencia:** Bradley Efron, Robert J. Tibshirani. *An Introduction to the Bootstrap*. Chapman & Hall/CRC, 1993.
- **Para qué:** intervalos de confianza para el error medio y para la diferencia método − línea base.

**Recomendación:** reporta, para cada línea base, la diferencia apareada (método vs. línea base sobre los mismos objetos) con intervalo de confianza bootstrap, y una prueba no paramétrica apareada (Wilcoxon o permutación). Con N = 100 objetos los intervalos serán anchos; dilo explícitamente.

---

## Lista de control

- [ ] Implementar el predictor aleatorio con muestreo uniforme real (vector gaussiano normalizado).
- [ ] Calcular el azar con la misma regla de evaluación que el método (en especial en objetos con varios planos).
- [ ] Reportar el valor analítico (57,3°, 1 − cos τ) junto al empírico.
- [ ] Implementar el predictor de pose canónica ("siempre el plano/eje más frecuente") y reportarlo por tipo de simetría.
- [ ] Agregar el predictor del centro de la imagen para los puntos 2D.
- [ ] Comparar con pruebas apareadas e intervalos de confianza.
- [ ] Verificar todas las referencias en su fuente antes de citarlas (solo MMStar fue verificada en esta sesión; el resto son referencias clásicas citadas de memoria).
