# Línea base aleatoria para simetría axial y planar

**Objetivo:** saber si las predicciones de Molmo2 son mejores que adivinar al azar.

**Idea:** el predictor aleatorio no mira la imagen. Elige una dirección cualquiera en el espacio 3D (el eje, o la normal del plano) con la misma probabilidad para todas las direcciones. Si el método no supera a este predictor, sus puntos no aportan información sobre la simetría.

---

## 1. Qué se compara

En ambos tipos de simetría, la predicción es una **dirección**:

- **Axial:** la dirección del eje.
- **Planar:** la normal del plano (el vector perpendicular al plano).

El error es el ángulo entre la dirección predicha `n̂` y la verdadera `n*`. Como un eje o una normal no tienen sentido (apuntar "hacia arriba" o "hacia abajo" es lo mismo), se usa el valor absoluto:

```
θ = arccos( |n̂ · n*| )        con θ entre 0° y 90°
```

---

## 2. Valores teóricos del azar (un solo eje o un solo plano)

Si `n̂` se elige al azar de manera uniforme sobre la esfera, el error θ tiene valores conocidos. No hace falta simular para obtenerlos.

| Métrica | Fórmula | Valor del azar |
|---|---|---|
| Error medio | 1 radián | **57,30°** |
| Error mediano | arccos(0,5) | **60,0°** |
| Desviación estándar | √(π − 3) rad | **21,57°** |
| P@τ (fracción con error ≤ τ) | 1 − cos τ | P@5° = **0,0038** · P@10° = **0,0152** · P@15° = **0,0341** |
| AUC (área bajo P@τ, de 0° a 45°, normalizada) | 1 − sin(45°) / (π/4) | **0,0997** |

Estos valores se comprobaron con una simulación de 2 millones de direcciones aleatorias; coinciden hasta el tercer decimal.

**Por qué funcionan las fórmulas (en una línea):** si una dirección es uniforme en la esfera, el coseno del ángulo con cualquier dirección fija es uniforme entre 0 y 1 (teorema de Arquímedes). Por eso P(θ ≤ τ) = 1 − cos τ, y de ahí salen las demás.

**Ojo con la AUC:** el valor 0,0997 solo vale si tu AUC integra hasta 45°. Si integra hasta otro límite T, el azar es `1 − sin(T)/T` (con T en radianes).

---

## 3. Dos ajustes que cambian los valores

### 3.1 Objetos sin predicción (cobertura)

Si a los objetos sin predicción les asignas 90°, el método se compara con un azar que también falla en la misma proporción. Con cobertura `c` (fracción de objetos con predicción):

```
Error medio del azar = c · 57,30° + (1 − c) · 90°
```

Ejemplo: con 800 de 850 objetos con predicción, c = 0,941 → azar = 59,22°.

### 3.2 Objetos con varios planos (solo planar)

Si el error se mide contra el **plano verdadero más cercano**, adivinar es más fácil cuando hay varios planos. Para planos perpendiculares entre sí:

| Planos del objeto | Error medio del azar | P@10° del azar |
|---|---|---|
| 1 | 57,3° | 0,015 |
| 2 | 42,0° | 0,030 |
| 3 | 31,9° | 0,045 |

(Valores obtenidos por simulación.) Como tu conjunto planar mezcla objetos con 1, 2 y 3 planos (692 / 152 / 6), **el azar planar no es 57,3°**: hay que calcularlo con la simulación de la sección 4, usando los planos reales de cada objeto.

---

## 4. Cómo calcularlo: simulación (recomendado)

La simulación usa tu misma función de evaluación, así que aplica automáticamente la cobertura, los varios planos y cualquier otra regla (F1, SDE, etc.).

```python
import numpy as np

def direccion_aleatoria(rng):
    """Dirección uniforme en la esfera: normalizar un vector gaussiano."""
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)

def error_angular(n_pred, n_gt):
    """Error sin signo, en grados, entre 0 y 90."""
    c = abs(np.dot(n_pred, n_gt))
    return np.degrees(np.arccos(np.clip(c, 0.0, 1.0)))

def azar_un_objeto(gts, rng):
    """gts: lista de direcciones verdaderas del objeto (1 para eje; 1-3 para planos).
    Se compara contra la más cercana, igual que en la evaluación del método."""
    n = direccion_aleatoria(rng)
    return min(error_angular(n, g) for g in gts)

def linea_base_aleatoria(objetos, con_prediccion, repeticiones=1000, seed=0):
    """
    objetos: lista con las direcciones verdaderas de cada objeto.
    con_prediccion: lista de True/False, si el MÉTODO predijo ese objeto.
    Devuelve el error medio del azar en cada repetición.
    """
    rng = np.random.default_rng(seed)
    medias = []
    for _ in range(repeticiones):
        errores = []
        for gts, tiene in zip(objetos, con_prediccion):
            if tiene:
                errores.append(azar_un_objeto(gts, rng))
            else:
                errores.append(90.0)          # misma imputación que el método
        medias.append(np.mean(errores))
    return np.array(medias)
```

**Uso:**

1. Ejecuta `linea_base_aleatoria` con los mismos objetos y la misma cobertura del método.
2. Calcula de la misma forma las otras métricas (P@5°, P@10°, AUC, F1) reemplazando la predicción del método por `direccion_aleatoria`.
3. Reporta la media del azar y su intervalo del 95 % (percentiles 2,5 y 97,5 de las repeticiones).

**Para planos:** si tu evaluación necesita un plano completo (normal + posición), haz que el plano aleatorio pase por el centro del objeto. Como los objetos están normalizados y centrados en el origen, eso equivale a usar `d = 0`. Declara esta decisión en la metodología.

---

## 5. Cómo decidir si el método es mejor que el azar

### Opción A: comparar con la simulación (la más simple)

Si el error medio del método queda **por debajo** del percentil 2,5 de las medias del azar, el método es mejor que el azar (p < 0,025, unilateral). Si queda dentro del intervalo, no se distingue del azar.

### Opción B: prueba apareada por objeto (la más sólida)

1. Para cada objeto, genera un error aleatorio con la sección 4.
2. Calcula la diferencia `error_método − error_azar` en cada objeto.
3. Aplica la prueba de Wilcoxon de rangos con signo (`scipy.stats.wilcoxon`, `alternative="less"`).
4. Repite con varias semillas, o usa el error **esperado** del azar por objeto en vez de una sola muestra, para que el resultado no dependa de una tirada.

### Opción C: valor z (rápida, solo para un eje o un plano por objeto)

```
z = (error medio del método − 57,30°) / (21,57° / √n)
```

Con n = 850, el denominador es 0,74°. Un z ≤ −2 indica un resultado mejor que el azar.

**Revisión pendiente:** con esta fórmula, axis_v08 (55,5°, n = 850) da z ≈ −2,4, no el −3,6 que aparece en la tesis. Revisa cómo se calculó ese valor (otra n, varianza empírica o varios nv agrupados) y documéntalo.

---

## 6. Lista de control

- [ ] Confirmar hasta qué ángulo integra tu AUC (el azar de 0,0997 supone 45°).
- [ ] Calcular el azar con la **misma cobertura** que el método.
- [ ] Para planar, calcular el azar por simulación con los planos reales de cada objeto (no usar 57,3°).
- [ ] Reportar el azar en la misma tabla que el método, para cada nv.
- [ ] Recalcular el z de axis_v08.

---

## 7. Referencias

**Muestreo de direcciones uniformes en la esfera**

- Muller, M. E. (1959). A note on a method for generating points uniformly on n-dimensional spheres. *Communications of the ACM*, 2(4), 19–20. https://doi.org/10.1145/377939.377946
  → Justifica normalizar un vector gaussiano (lo que hace `direccion_aleatoria`).
- Marsaglia, G. (1972). Choosing a point from the surface of a sphere. *The Annals of Mathematical Statistics*, 43(2), 645–646. https://doi.org/10.1214/aoms/1177692644
  → Método alternativo clásico para el mismo muestreo.

**Pruebas estadísticas**

- Dror, R., Baumer, G., Shlomov, S., y Reichart, R. (2018). The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing. *Proceedings of ACL 2018*, 1383–1392. https://doi.org/10.18653/v1/P18-1128
  → Cómo elegir pruebas apareadas y no paramétricas al comparar sistemas.
- Efron, B., y Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC.
  → Intervalos de confianza por remuestreo.

**Contexto: líneas base de evaluación**

- Chen, L., et al. (2024). Are We on the Right Way for Evaluating Large Vision-Language Models? *NeurIPS 2024*. arXiv:2403.20330.
  → Muestra que los benchmarks de VLM deben compararse contra el desempeño alcanzable sin información visual; justifica incluir líneas base que no miran la imagen.

**Estado de verificación:** Dror et al. (2018) y Chen et al. (2024) se verificaron en su fuente durante esta conversación. Muller (1959), Marsaglia (1972) y Efron y Tibshirani (1993) son referencias clásicas citadas de memoria: confirma sus datos (en especial los DOI) en Crossref antes de incluirlas. Las fórmulas de la sección 2 son derivaciones estándar y se verificaron por simulación; puedes incluirlas en la metodología o en un anexo sin cita adicional.
