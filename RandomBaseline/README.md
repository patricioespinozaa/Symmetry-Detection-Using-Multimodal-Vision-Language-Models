# RandomBaseline

Línea base aleatoria para simetría axial y planar: un predictor que **no mira
las imágenes** y elige la dirección del eje (o la normal del plano) uniforme
en la esfera. Responde si los resultados de Molmo2 son mejores que adivinar.
La teoría está en [`docs/literature/baseline_aleatorio.md`](../docs/literature/baseline_aleatorio.md)
y el diseño del módulo, en [`docs/features/random-baseline.md`](../docs/features/random-baseline.md).

El azar se puntúa con las **mismas funciones** de `Mapping/evaluate.py`: el mejor
plano GT, 90° para los objetos sin predicción y F1_ref acumulado sobre el
dataset. Solo lee las etiquetas `.txt`, así que no necesita renders, GPU ni mallas.

## Archivos

| Archivo | Qué hace |
|---|---|
| `chance.py` | Valores cerrados del azar (1 eje/plano GT), con cobertura `c` |
| `simulation.py` | Carga el GT, muestrea direcciones y puntúa como `evaluate.py` (vectorizado) |
| `multiplane.py` | Azar multiplano: varios planos separados por objeto, puntuados como `evaluate.py` |
| `significance.py` | Compara el método con el azar: IC 95 %, p empírico y z |
| `run_baseline.py` | CLI: subcomandos `reference` y `compare` |
| `tests/test_random_baseline.py` | Comprueba que el código vectorizado da lo mismo que `evaluate.py` y que coincide con las fórmulas cerradas |

## Uso

```bash
# 1. Valores del azar sobre el dataset real (ambos tipos, cobertura 100 % y 90 %)
python RandomBaseline/run_baseline.py reference --objects-root ../data/objects \
    --coverage 1.0 0.9 --out RandomBaseline/outputs/random_baseline_reference.csv

# 2a. Comparar CSVs de evaluate.py / compare_results.py
python RandomBaseline/run_baseline.py compare --objects-root ../data/objects \
    --symmetry-type axis_sym \
    --summary-csv resultados_finales/<carpeta>/experiments_*/axis_sym_final_comparison.csv \
    --out RandomBaseline/outputs/random_baseline_comparison.csv

# 2b. Comparar eval_*_results.json (solo útil con --coverage-mode matched)
python RandomBaseline/run_baseline.py compare --objects-root <objects_root> \
    --symmetry-type plane_sym \
    --eval-results <renders_root>/plane_sym/eval_s448_default_<EXP>_triangulation_results.json

# 3. Azar multiplano con 1, 2 y 3 planos por objeto, sobre el dataset real
python RandomBaseline/run_baseline.py reference --objects-root ../data/objects \
    --symmetry-type plane_sym --n-planes 1 2 3 \
    --out RandomBaseline/outputs/random_baseline_multiplane.csv

# Tests (no necesitan datos)
python RandomBaseline/tests/test_random_baseline.py
```

Opciones comunes: `--repetitions` (por defecto 1000) y `--seed` (por defecto 0).

**Cobertura (`--coverage-mode`, en `compare`):**

- `all` (por defecto): el azar predice **todos** los objetos. El método sigue
  con 90° en los objetos que no predijo, así que es la comparación más
  exigente para el método.
- `matched`: el azar recibe la misma cobertura que el método. Con
  `--summary-csv`, los objetos sin predicción se sortean en cada repetición;
  con `--eval-results`, son exactamente los mismos que le faltaron al método.

Ojo con F1_ref: `evaluate.py` no cuenta los objetos sin predicción en el F1 del
método. Con `all`, el F1 del azar sí incluye todos los objetos.

**Filas multiplano (`triangulation_multiplane`):** el azar predice tantos planos
como el método, separados al menos 15° entre sí, y se compara en recall, precisión,
F1_ref, `recall_adj` y `f1_adj`. Con `--eval-results` usa el número exacto de planos
por objeto; con `--summary-csv` solo conoce la media y reparte los planos de la forma
más pareja posible, así que el resultado es aproximado.

`compare` escribe una fila por `(experimento, método, n_views, métrica)`.
Necesita `numpy` y lo que ya importa `evaluate.py` (`trimesh`, `tqdm`). Todo
corre en local: solo lee los `.txt` y los CSV/JSON de resultados.

Las tablas de la tesis no las genera este módulo: `utils/generar_tablas_resultados_cap4.py`
lee `outputs/random_baseline_reference.csv`. Los subcomandos `latex` y `cap4`, que escribían
tablas LaTeX propias, se quitaron en la reestructuración y siguen en el historial de git.

## Cómo leer la salida

- `coverage` / `random_coverage`: fracción de objetos con predicción del
  método y del azar (1,0 con `--coverage-mode all`).
- `random_mean` y `random_ci_low`/`random_ci_high`: media del azar y su
  intervalo del 95 % (percentiles 2,5 y 97,5 de las repeticiones).
- `p_empirical`: p unilateral, `(1 + #repeticiones del azar iguales o mejores) / (R + 1)`.
  Su mínimo es `1/(R+1)` (0,001 con R = 1000). Cuando el método está muy lejos
  del azar, `z` y `p_normal` muestran la magnitud.
- `better_than_random`: `p_empirical < 0,025`.
- `closed_form_single_gt` (en `reference`) supone **un** GT por objeto. Para
  `plane_sym` (692/152/6 objetos con 1/2/3 planos) vale la columna simulada:
  el azar planar es **54,6°**, no 57,3°.

## Valores de referencia (dataset completo, cobertura 100 %, R = 1000)

| | Error medio | Mediana | AUC@45° | P@5° | P@10° | P@15° | F1_ref |
|---|---|---|---|---|---|---|---|
| `axis_sym` (850) | 57,28° [55,76; 58,66] | 59,98° | 0,0999 | 0,0039 | 0,0152 | 0,0340 | — |
| `plane_sym` (850) | 54,59° [53,11; 56,11] | 55,99° | 0,1175 | 0,0045 | 0,0178 | 0,0401 | 0,0093 |

Con `--coverage-mode matched` el azar empeora según la cobertura del método.
Por ejemplo, con 88 % de cobertura el error medio del azar axial sube a ~61°.
