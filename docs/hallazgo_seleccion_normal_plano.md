# Selección de la normal en el estimador de plano sin malla

Fecha del análisis: 2026-10-18. Afecta a `Mapping/estimate_symmetry_no_mesh.py`
(`estimate_plane_no_mesh`), es decir, a todas las corridas `plane_*_nomesh` del pipeline
sin malla, con un plano o con varios.

## Resumen

El estimador de plano genera muchas normales candidatas y se queda con la que mejor
puntúa `score_normal`. Ese puntaje **solo usa la geometría de las cámaras**: mide qué tan
"de canto" quedan las vistas respecto a cada candidata, pero no comprueba si los puntos de
Molmo son consistentes con ese plano.

**No es un error de implementación.** El código hace lo que describe
`docs/pipeline_sin_malla.md` §3.2, paso 2. Es una **limitación del criterio de selección**:
con pocas vistas funciona, pero con muchas vistas, y por lo tanto muchos candidatos, varios
candidatos falsos encajan igual de bien con la distribución de cámaras, y la elección se
vuelve casi arbitraria.

Con puntos perfectos (sin error de Molmo), el estimador de plano obtiene un error mediano
de 1.6° con 6 vistas, pero de **27°** con 26 vistas. El estimador de eje, con los mismos
puntos perfectos, obtiene 0° en todas las configuraciones.

## Cómo funciona hoy el estimador de plano

En `Mapping/estimate_symmetry_no_mesh.py`, función `estimate_plane_no_mesh`:

1. **Recta por par de vistas.** En cada vista, los dos puntos (`obj_id` 1 y 2) y el centro
   de la cámara definen un plano de interpretación. Para cada par de vistas, se intersecan
   sus dos planos de interpretación y se obtiene una recta 3D (`_line_from_view_pair`).
2. **Normales candidatas.** Se toma el producto cruz de cada par de rectas que vienen de
   pares de vistas disjuntos. Esto produce aproximadamente 45 candidatos con 6 vistas,
   3 000 con 14 y 45 000 con 26.
3. **Selección.** Se elige el candidato con menor `score_normal`, que es la media de
   `|dirección_de_vista · n|` sobre la mitad de las vistas más de canto. **Solo depende
   de las cámaras**.
4. **Reajuste.** Con las vistas de canto respecto al candidato elegido
   (`|dirección_de_vista · n| < 0.5`), se recalcula la normal por SVD sobre las rectas de
   esos pares. **Este paso sí usa los datos.**

## Por qué el plano es más difícil que el eje

- **Eje:** en todas las vistas, los dos puntos están sobre la misma recta 3D, así que
  todos los planos de interpretación contienen esa recta y la intersección es exacta.
- **Plano:** cada vista marca puntos distintos de la traza, de modo que cada vista aporta
  una recta distinta dentro del plano. El plano de interpretación de una vista solo
  coincide con el plano de simetría cuando esa vista está de canto. En las demás vistas, la
  recta triangulada no está dentro del plano y mete ruido.
- **Si los puntos fueran idénticos entre vistas, el método fallaría.** Con los mismos dos
  puntos 3D en todas las vistas, solo hay una recta, y una recta no determina un plano. En
  la simulación, este caso no produce predicción en 28 de 30 objetos con 6 vistas.

## Evidencia: simulación con puntos perfectos

Se reemplazó a Molmo por puntos calculados desde el ground truth, y se usaron las mismas
cámaras de los renders (`R` de `metadata_all.json`, `T = (0, 0, d)` con la misma
distancia de cámara que usa `ImagesGenerator`). Esos puntos se pasaron sin modificar por
`estimate_axis_no_mesh` y `detect_planes_no_mesh` (`max_planes=1`):

- **Eje:** los dos polos, es decir, los extremos de la superficie a lo largo del eje de
  ground truth.
- **Plano:** los puntos más alto y más bajo de la traza visible en cada vista. La traza es
  la superficie a menos de 1% de la diagonal del plano de ground truth. La visibilidad se
  calcula con un z-buffer aproximado de puntos de superficie. Equivale a un `v05_1`
  perfecto.

### Error angular con puntos perfectos (30 objetos por tipo)

| Tipo | Vistas | Mediana | Media | P@10° |
|---|---|---|---|---|
| Eje | 6 / 14 / 26 | 0.0° | 0.0° | 1.00 |
| Plano | 6 | 1.6° | 2.6° | 0.97 |
| Plano | 14 | 2.9° | 7.1° | 0.87 |
| Plano | 26 | **27.0°** | 30.1° | **0.17** |

### Diagnóstico de candidatos (15 objetos planares, medianas)

| Vistas | Candidatos | Mejor candidato disponible | Elegido por `score_normal` | Tras el reajuste | Puntaje basado en datos | Normal real gana `score_normal` |
|---|---|---|---|---|---|---|
| 6 | 45 | 0.4° | 9.7° | 1.4° | 0.8° | 13% |
| 14 | 3 003 | 0.2° | 6.5° | 4.0° | 3.3° | 0% |
| 26 | 44 850 | 0.0° | **73.4°** | **31.3°** | **5.1°** | 0% |

- **Siempre hay un candidato casi exacto**, de menos de 1°. El problema está en elegirlo,
  no en generarlo.
- **La normal verdadera casi nunca obtiene el mejor `score_normal`.** El criterio no
  premia la respuesta correcta.
- **Con 6 vistas el reajuste compensa una mala elección inicial.** Con 26 vistas ya no
  alcanza.

### Coincide con los resultados reales

Con `plane_v05_1` a 224px en `flat`, el AUC$_{45°}$ cae de 0.494 (6 vistas) a 0.381 (14) y
a 0.277 (26). En eje (`axis_v04`) la caída es menor (0.582 → 0.580 → 0.502), y con puntos
perfectos el estimador de eje no se degrada, así que la caída del eje viene de Molmo.

## Qué implica para interpretar los resultados

1. **La caída del plano con 14 y 26 vistas** se debe en buena parte al criterio de
   selección, no solo a Molmo ni al prompt.
2. **Con 6 vistas**, el estimador recupera bien el plano a partir de puntos perfectos, así
   que la brecha entre plano y eje en esa configuración viene principalmente de los puntos
   de Molmo (por ejemplo, la traza es ambigua en las vistas que no están de canto).
3. **El método de plano depende de las vistas de canto**, mientras que el de eje usa todas.
   Esta asimetría es estructural y existe incluso con un criterio de selección mejor.

## Mejora propuesta: puntaje basado en datos

En una vista de canto, el plano de interpretación coincide con el plano de simetría. Por
eso, una normal correcta debe ser paralela a las normales de interpretación de las vistas
que mejor la explican. El criterio propuesto es:

```
score_datos(n) = media de (1 - |n_i · n|) sobre el tercio de vistas con menor valor
```

donde `n_i` es la normal del plano de interpretación de la vista `i`
(`interpretation_plane_normal`). Se elige el candidato con menor `score_datos`. El
reajuste por SVD se mantiene igual.

En la simulación con puntos perfectos bajó la mediana a 0.8°, 3.3° y 5.1° con 6, 14 y 26
vistas.

**Cuidados antes de adoptarlo:**
- **No está validado con datos reales de Molmo.** Con puntos ruidosos la mejora puede ser
  menor.
- **La fracción "un tercio" es un parámetro.** Debe fijarse antes de mirar los resultados
  de los 850 objetos y declararse; no debe ajustarse sobre el conjunto de evaluación.
- **No usa el ground truth**, solo los puntos de Molmo y las cámaras, así que no hay fuga
  de datos.
- **Implementarlo como opción**, por ejemplo `--plane-score {camaras,datos}` o una variante
  nueva en `Pipeline_Experiments/estimate_symmetry_variants.py`, dejando `score_normal`
  por defecto para que los resultados anteriores sigan siendo reproducibles.
- **No hace falta volver a correr Molmo**, solo repetir la estimación y la evaluación de
  plano. Comparar contra el criterio actual y contra los predictores triviales, ya que los
  objetos planares no están rotados al azar.

## Posibles experimentos para la tesis

1. **Cota superior del estimador (puntos perfectos).** Mide cuánto error aporta el método
   por sí solo, separado del error del VLM. Muestra que el estimador de eje es exacto y que
   el de plano se degrada con el número de vistas.
2. **Ablación del criterio de selección.** Comparar `score_normal` con `score_datos` usando
   los puntos reales de Molmo, para 6, 14 y 26 vistas (y para 42 a 114 vistas cuando estén
   los experimentos de robustez).

## Limitaciones de la simulación

- Usa una muestra de 30 objetos por tipo (15 en el diagnóstico de candidatos), elegidos
  al azar con semilla 0.
- La visibilidad se aproxima con un z-buffer de 20 000 puntos de superficie a una
  resolución de 112×112. La "traza" es la superficie a menos de 1% de la diagonal del plano.
- Supone que Molmo marca exactamente el punto más alto y el más bajo de la traza visible,
  sin ruido de localización.
- Los scripts de la simulación (`sim_oraculo.py`, `diag_candidatos.py`) se escribieron
  fuera del repositorio. Para reproducir estos números hay que llevarlos al repo, por
  ejemplo a `utils/`.
