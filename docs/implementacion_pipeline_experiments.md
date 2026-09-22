# Implementación de `Pipeline_Experiments/` y primera corrida completa

> Registro de la sesión en la que se auditó qué experimentos de
> `Experiments/*.ipynb` ya estaban productivizados a escala completa (850
> objetos) y cuáles no, se implementó `Pipeline_Experiments/` para cerrar esa
> brecha, y se corrió por primera vez sobre el dataset completo — incluye los
> bugs reales encontrados durante esa corrida (y su fix) y el estado en el
> que quedó el análisis. Complementa a `Pipeline_Experiments/README.md`
> (referencia de uso) sin duplicarlo — acá va la narrativa de decisiones y
> hallazgos; ahí va la documentación de API/config.

## 1. Punto de partida: auditoría de qué faltaba

Antes de escribir código se revisaron los 8 notebooks de `Experiments/` para
inventariar qué experimentos se habían prototipado ahí (sobre ~30 objetos
curados) y se comparó contra `Mapping/` y `MolmoPointing/` para ver qué de
eso ya corría a escala completa (850 objetos) y qué era exclusivo del
sandbox. Hallazgos clave:

- **Ya productivizado**: los prompts `axis_v00`..`v08` / `plane_v00`..`v08`
  (incluidas variantes `_1`/`_6pts`), Flow A/B/C, y los prompts de
  literatura `axis_lit2_grid` (Grid Overlay, MOKA) y `axis_lit3_cot`
  (Structured CoT) — todos corribles a escala completa vía
  `MolmoPointing/molmo_multiview_runner.py`. La estimación baseline
  sin malla (`Mapping/estimate_symmetry_no_mesh.py`) y su evaluación
  (`Mapping/evaluate.py`) también.
- **Solo en el sandbox, nunca portado**: las 6 ablaciones de triangulación
  (EXP-A a EXP-F: triangulación ponderada, point-then-direction, RANSAC 2D,
  peso por confianza, reweighting iterativo, gate de SDE_ref real contra la
  malla), las 3 políticas de descarte de vistas (`none`/`any`/`both`), la
  implementación REAL (no simulada con oráculo) de EXP-LIT-1
  (Candidates-then-Select), y el diagnóstico de colapso a `X≈500`.

Ese hueco es lo que se implementó en `Pipeline_Experiments/`.

## 2. Qué se implementó

```
Pipeline_Experiments/
├── config.py                          # YAML config + auto-discovery + dedup de experiment_ids
├── view_filtering.py                  # políticas none/per_point/any/both
├── estimate_symmetry_variants.py      # CLI de un (experiment_id, variante) -- hermano de estimate_symmetry_no_mesh.py
├── run_batch.py                       # orquestador config-driven (el entrypoint principal)
├── list_prompts.py                    # audita qué encontraría experiment_ids: auto, con/sin dedup
├── audit_results.py                   # audita qué se generó realmente en disco por combinación
├── triangulation_variants/            # EXP-A..F, registrados en __init__.py::VARIANTS
├── candidates_then_select/            # EXP-LIT-1 real (necesita GPU, no corrido todavía)
├── diagnostics/diagnose_center_bias.py
└── configs/
    ├── experiments.example.yaml
    └── full_sweep.yaml                # el config usado en la corrida real (ver §4)
```

Cada ablación reutiliza al máximo la infraestructura existente
(`pipeline_common/triangulation.py`, `Mapping/estimate_symmetry_no_mesh.py`,
`Mapping/evaluate.py`) y escribe `predicted_symmetry_<ID>.json` en el mismo
esquema (`triangulation`/`triangulation_multiplane`) que `evaluate.py` ya
entiende — ningún archivo de `Mapping/` se modificó para esto.

**Validación antes de tocar el servidor**: recuperación exacta de un eje/
plano sintético con cámaras reales de PyTorch3D (error angular ~0° para las
5 variantes de eje), equivalencia numérica exacta entre el motor genérico y
el baseline sin tocar cuando se configuran igual, y un round-trip completo
`process_object` → `Mapping/evaluate.py` real (sin modificar, vía subprocess)
dando `angular_error_deg = 0.0` exacto.

## 3. Config-driven: correr sobre todos los prompts ya ejecutados

`Pipeline_Experiments/config.py::discover_experiment_ids` escanea
`molmo_multiview_<ID>.json` en disco en vez de requerir una lista hardcodeada
— con `experiment_ids: auto` en el YAML, `run_batch.py` procesa automáticamente
cualquier prompt ya corrido, sin editar el config cada vez que se prueba uno
nuevo.

`run_batch.py` también soporta particionar el trabajo en paralelo por CPU
(`--shard-id`/`--num-shards`) — aclarado explícitamente que **nada acá usa
GPU** (las ablaciones son numpy puro; `expF`/`--with-reference-metrics` usan
`gpytoolbox`, que es CPU) — con un guard que bloquea correr
diagnósticos/evaluate/compare junto con `--num-shards > 1` (verían un
conjunto de objetos parcial).

## 4. Primera corrida real — bugs encontrados y arreglados

### 4.1 Duplicados `<X>` / `<X>_nomesh` (34 en vez de 17 prompts de `plane_sym`)

La primera corrida de `experiment_ids: auto` sobre `plane_sym` encontró 34
`experiment_id` en vez de los 17 esperados (según `MolmoPointing/prompts_registry.py`).
Confirmado por el usuario: los sufijados `_nomesh` son **exactamente el mismo
prompt y los mismos puntos de Molmo2** que su contraparte sin sufijo — se
guardaron dos veces bajo `--experiment-id` distintos únicamente para no
pisar el resultado del pipeline con malla del mismo `prompt_id`. Sin dedup,
cada ablación se calculaba dos veces sobre datos idénticos.

**Fix**: `discover_experiment_ids` ahora descarta `<X>_nomesh` cuando `<X>`
(sin sufijo) también existe, y conserva un `_nomesh` sin gemelo (por si es
un experimento legítimamente distinto sin contraparte). Verificado con un
fixture sintético. Resultado real tras el fix: **22 experiment_id en
`axis_sym`, 20 en `plane_sym`** (los 19+17 del registro, más `flowB`/`flowC`
de `axis_v05_1`/`plane_v04_1` — legítimos, Flow B/C generan puntos
distintos — más `axis_v06_nomesh_filtered` / `plane_v04_1_nomesh_filtered`,
cuyo origen exacto el usuario no recuerda con certeza; decidió incluirlos
igual por ahora — ver §5, punto 2).

### 4.2 Bug real: gate de `expF` aceptaba planos con SDE = NaN

Durante la corrida completa apareció, para algunos objetos, un
`RuntimeWarning: invalid value encountered in divide` desde
`gpytoolbox/barycentric_coordinates.py` — un triángulo de área cero en la
malla (artefacto conocido de algunos objetos de ShapeNet) produce un
`SDE_ref` = `NaN` para ese plano candidato. El gate de `exp_f_sde_gate.py`
comparaba `if real_sde > sde_gate: break` — en Python, `nan > 0.02` es
`False`, así que un plano con SDE inválido **pasaba el gate en vez de ser
rechazado**, exactamente lo opuesto de la intención del diseño.

**Fix**: `if not np.isfinite(real_sde) or real_sde > sde_gate: break`.
Verificado forzando `calplaneloss` a devolver `NaN` y confirmando que ahora
rechaza (0 planos aceptados) en vez de aceptar. Como `Mapping/evaluate.py` y
`compare_results_no_mesh.py` **siempre** recalculan sus archivos de salida
(no tienen noción de "ya existe, salteo") pero `process_object` sí tiene
resumability, `full_sweep.yaml` se dejó con `overwrite: true` para que la
re-corrida completa recalcule todo desde cero y no arrastre resultados de
`expF` generados con el bug.

## 5. Estado del análisis tras la primera corrida completa

`audit_results.py` (lee el mismo YAML que `run_batch.py`, cuenta archivos en
disco — no métricas) reportó **190 combinaciones auditadas, 164 con algún
faltante**. Lectura de esos números, no a tomar al pie de la letra como "164
bugs":

1. **El faltante es idéntico entre las 5 variantes de un mismo prompt**
   (ej. `axis_v00`: 841/850 en expA, expB, expC, expD y expE por igual).
   Las 5 comparten el mismo requisito estructural de "≥2 vistas con un par
   de puntos válido" — lo que cambia entre ellas es cómo ponderan/seleccionan
   después, no si hay suficientes vistas para empezar. Es **una causa
   compartida por prompt** (Molmo2 no devolvió suficientes puntos válidos
   para esos objetos puntuales bajo ese prompt), no 5 fallas independientes.
2. **`expF` siempre tiene más faltantes que expA/C/D del mismo prompt** —
   esperado: además de las ≥2 vistas, necesita ≥4 vistas independientes y
   que la malla real tenga geometría válida para el gate de SDE. Más
   requisitos, más objetos que no llegan; no es un bug.
3. **Outliers que sí ameritan revisión** (faltante grande, no un puñado):
   - `plane_v04_1_flowC`: 488–604/850 faltantes (hasta 71%) — el peor de
     todos.
   - `axis_v05_1_flowC`: 429/850 (más de la mitad).
   - `axis_v07`: 171/850 (20%).
   - `plane_v04_1_6pts`: 107–245/850.

   El patrón "Flow C mucho peor que Flow B del mismo prompt base" (ej.
   `axis_v05_1_flowC` 429 faltantes vs. `axis_v05_1_flowB` solo 1) es
   consistente con algo ya documentado en el propio docstring de
   `molmo_multiview_runner.py`: Flow C **no tiene tracking de identidad
   entre vistas** y el modelo devuelve "cuantos puntos tenga esa imagen,
   hasta un tope chico" — estructuralmente más propenso a vistas con <2
   puntos. Hipótesis plausible, no confirmada todavía (ver próximos pasos).
4. **Lo que el audit no dice todavía**: la razón exacta de cada falla —
   `process_object` tiene un `except Exception: continue` que traga el
   `ValueError` real ("need >=2 valid views, got 1", geometría degenerada,
   etc.).

## 6. Próximos pasos

1. **Construir un script de "razón de falla"** para un `(prompt, variante)`
   puntual: listar los objetos sin predicción junto con el mensaje de
   excepción real (hoy tragado por `except Exception: continue`), priorizando
   `plane_v04_1_flowC`, `axis_v05_1_flowC`, `axis_v07` y `plane_v04_1_6pts`
   para confirmar o descartar la hipótesis de Flow C del §5.3.
2. **Resolver el origen de `axis_v06_nomesh_filtered` /
   `plane_v04_1_nomesh_filtered`**: el usuario no recuerda si son una
   corrida genuinamente distinta (algún filtro aplicado en el pointing) o
   otra copia redundante de `axis_v06`/`plane_v04_1` bajo un tercer nombre.
   Más confiable que la memoria: diffear directamente
   `molmo_multiview_axis_v06.json` vs. `molmo_multiview_axis_v06_nomesh_filtered.json`
   (mismo `points_by_image` ⇒ duplicado; distinto ⇒ experimento real) y
   decidir si sumarlo a la regla de dedup de `discover_experiment_ids`.
3. **Revisar el ranking una vez termine `run_evaluate`/`run_compare`**: mirar
   `results/experiments_<fecha>/{axis,plane}_sym_nomesh_comparison.csv` (o
   `Experiments/analisis_prompts_no_mesh.ipynb`, que ya sabe leer esa
   carpeta) para ver si alguna de las 6 ablaciones realmente mejora el
   baseline `triangulation` sin malla, y para qué prompts.
4. **Pase de re-scoring con `--with-reference-metrics`**: `full_sweep.yaml`
   lo dejó en `false` para la primera pasada (gpytoolbox es caro sobre
   190 combinaciones × 850 objetos); correrlo aparte una vez que los
   resultados de error angular se vean razonables, para tener `SDE_ref`/`F1_ref`
   comparables con la literatura.
5. **EXP-LIT-1 (Candidates-then-Select) real**: `candidates_then_select/`
   está implementado y probado a nivel de geometría/parsing, pero nunca se
   corrió contra el modelo real (necesita GPU) — pendiente si se decide
   validar esa propuesta de literatura con inferencia real en vez de la
   simulación oráculo/random de `Experiments/sandbox_literatura.ipynb`.

## Archivos relacionados

- `Pipeline_Experiments/README.md` — referencia de uso, schema de config,
  comandos standalone por variante.
- `Pipeline_Experiments/configs/full_sweep.yaml` — el config usado en la
  corrida descrita acá.
- `docs/pipeline_sin_malla.md`, `docs/implementacion_pipeline_sin_malla.md`
  — diseño del pipeline sin malla que estas ablaciones extienden.
- `docs/metricas_evaluacion.md`, `docs/verificacion_metricas_literatura.md`
  — definición de las métricas que `evaluate.py`/`compare_results_no_mesh.py`
  producen sobre la salida de este módulo.
