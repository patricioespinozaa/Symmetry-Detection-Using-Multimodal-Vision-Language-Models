# Resumen ejecutivo — pipeline sin malla (triangulación multivista)

> Síntesis de `docs/reporte_resultados_sin_malla.md` (detalle, tablas
> completas y método). Datos: `results/experiments_15_09_2026` (eje) y
> `results/experiments_22_09_2026` (plano), 850 objetos por tipo, más los
> diagnósticos de `results/diagnostics/`. Fecha: 2026-09-26.

## Qué se evaluó

El pipeline estima el eje o plano de simetría triangulando, entre varias
vistas calibradas, los puntos 2D que Molmo2 marca en cada render, sin tocar
la malla (la malla solo se usa para evaluar). Se analizaron 20 variantes de
prompt por tipo de simetría en 6, 14 y 26 vistas, junto con 6 ablaciones
de triangulación (EXP-A..F).

## Resultado central

| | Mejor resultado | Referencia sin información | Veredicto |
|---|---|---|---|
| **Eje** | `axis_v05_1`, n=26: error medio 57.36°, AUC@45 0.104, P@10° 0.013 | **Dirección al azar**: 57.30°, AUC 0.0997, P@10° 0.015 (analítico) | **Indistinguible del azar.** 13/14 prompts con baseline ≈ azar o peor |
| **Eje, excepción** | `axis_v08`, n=6: 55.5°, P@5° 0.014 | Azar: 57.3°, P@5° 0.0038 | Único prompt claramente sobre el azar (z = −3.6, cobertura 98%), efecto chico |
| **Plano** | `plane_v04_1`, n=14: recall 0.854, `f1_ref` 0.482 (sobre 65% de los objetos) | **"Siempre el plano X"**: recall 0.767, `f1_ref` 0.742 (en la muestra de 30 objetos) | **No demostrado que supere al trivial.** En la muestra queda por debajo; falta medirlo en los 850 |

## Por qué

1. **Molmo2 marca la columna central de la imagen (X≈500)**: en la mayoría
   de los prompts, entre 40% y 99% de los puntos (los bilaterales y `v08`
   son la excepción, con 4–25%). Como todas las cámaras miran al origen con
   "arriba" = +Y, la triangulación de esos puntos devuelve el **eje vertical
   del render**, no el del objeto. En la muestra local, 12/29 ejes predichos
   son exactamente +Y.
2. **Los ejes GT están rotados al azar** → predecir siempre la vertical da
   error de azar. Los casos "buenos" del sandbox son objetos cuyo eje GT
   casualmente es vertical.
3. **Los planos GT están en pose canónica** (100% alineados con un eje del
   mundo en la muestra; ~75% con X) → el mismo colapso a la vertical
   produce normales alineadas con X y **acierta sin entender el objeto**.
   Por eso el centrado correlaciona +0.81 con el desempeño en plano y 0 en
   eje.
4. **Los prompts bilaterales (v01, v02, v03) son incompatibles con los
   estimadores sin malla**: estos suponen dos puntos *sobre* el eje/traza;
   un par izquierda-derecha da una línea perpendicular. Resultado: eje peor
   que el azar (z hasta +3.7) y plano colapsado (recall 0.01–0.15).

## Ablaciones

| Variante | Eje (Δ error medio) | Plano (comparación justa, modo multiplano) |
|---|---|---|
| EXP-A (peso por separación) | +0.62° (peor en 40/42) | sin efecto (\|Δ\| ≤ 0.02) |
| EXP-B (ancla/dirección separadas) | +0.62° en ángulo; traslación media −0.22, solo por outliers | — |
| EXP-C (RANSAC 2D) | idéntico al baseline (no-op con 2 puntos) | idéntico al baseline |
| EXP-D (penalización de borde) | +0.61° | sin efecto |
| EXP-E (reponderación iterativa) | +0.26° | — |
| EXP-F (gate con `SDE_ref` sobre la malla real) | — | `f1_ref` +0.060 y precisión +0.118 **pero** −281 objetos de cobertura: recall ajustado −0.078; usa la malla |

**Ninguna ablación mejora la tarea de fondo**: no se puede re-ponderar hacia
una dirección que los puntos de entrada no contienen. Esto **corrige** dos
conclusiones previas de `docs/experimentos_pipeline_sin_malla.md`: EXP-B no
localiza mejor el eje, y EXP-F no es "la ablación más sólida" (mejora
absteniéndose en ~1/3 de los objetos).

## Otras observaciones

- **Cobertura**: las métricas de plano excluyen los objetos sin predicción
  (eje imputa 90°). La cobertura va de 19% a 100%, así que las comparaciones
  crudas de plano están sesgadas. Ajustado por cobertura, el mejor plano es
  `plane_v05_6pts` n=14 / `plane_v02` n=26 (F1 ≈ 0.58–0.59), no `v04_1` n=14.
- **`n_views` en plano**: 6 vistas permiten 1 solo plano; 26 vistas
  sobre-predicen (2.1–2.65 planos contra 1.19 reales) y `f1_ref` cae a
  0.13–0.22.
- **Huella de predicción independiente del objeto**: en modo un-solo-plano
  a n=6, la mediana del error es exactamente 40.0° en 8 prompts distintos.
- **Flujo B** no cambia nada; **Flujo C** pierde 56–81% de los objetos
  (escasez real de puntos, 0 errores de código).
- **Duplicados**: `*_nomesh_filtered` = `axis_v06`/`plane_v04_1`;
  `experiments_27_08_2026_v2` = `27_08`.
- `axis_lit2_grid` y `axis_lit3_cot` no están en ningún CSV de resultados.

## Recomendaciones

1. **Correr `Pipeline_Experiments/diagnostics/trivial_baselines.py`** sobre
   los 850+850 objetos (segundos, sin GPU). Decide si el resultado de plano
   supera a un predictor sin información.
2. **Reportar siempre contra el azar (eje) y el trivial (plano)**, y con la
   cobertura al lado; idealmente imputar los faltantes de plano en
   `evaluate.py`.
3. **Quitar el atajo de pose canónica**: evaluar plano sobre objetos rotados
   al azar (como eje) o con "arriba" de cámara aleatorio.
4. **Dar a los prompts bilaterales un estimador compatible** (modo midpoint
   en eje; P−P' como normal en plano).
5. **Seguir la única pista positiva: `axis_v08`** (anti-centrado). Pausar
   nuevas ablaciones de triangulación: el cuello de botella está en los
   puntos de Molmo2, no en la geometría.
