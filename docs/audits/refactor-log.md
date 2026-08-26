# Refactor log

Ejecutado a partir de `docs/audits/architecture-audit.md`. Formato:
`[archivo o directorio] | [acción] | [motivo]`

## Fase 1 — Reproducibilidad

`environment.yml` | refactorizado | Auditoría marcó `gpytoolbox`/`rtree`/`scikit-learn` como
faltantes; verificado contra el archivo real: `rtree==1.4.1` y `scikit-learn==1.7.2`
YA estaban listados (la auditoría se equivocó en 2 de 3). Solo `gpytoolbox` faltaba
de verdad — agregado sin pin de versión (no hay ninguna registrada en el repo) con
un comentario explicándolo.

No se tocaron seeds (ya documentados y consistentes por diseño, ver auditoría §4)
ni paths hardcodeados (no se encontró ninguno).

## Fase 2 — Limpieza estructural

`Mapping/audit_view_indices.py` | movido a `Mapping/archive/` | Superseded por
`audit_view_indices_v2.py` según el propio docstring de v2; sin imports externos
que lo referencien (verificado con grep).

`Mapping/check_origin_compactness.py` | movido a `Mapping/archive/` | Superseded
por `check_origin_full.py` según el propio docstring de este último; sin imports
externos que lo referencien.

`pipeline_common/viz_colors.py` | creado | Consolida `COLOR_GT_AXIS`/`COLOR_GT_PLANE`/
`COLOR_PRED_AXIS`/`COLOR_PRED_PLANE` (y en `visualize_rays.py` también
`COLOR_MESH`/`COLOR_CAMERA`/`COLOR_HIT_RAY`/`COLOR_MISS_RAY`/`COLOR_HIT_PT`/
`COLOR_CLUSTER_PT`) copiados verbatim en `Mapping/visualize_rays.py`,
`Mapping/export_symmetry_overlay.py` y `Mapping/render_symmetry_comparison.py`.

`Mapping/visualize_rays.py`, `Mapping/export_symmetry_overlay.py`,
`Mapping/render_symmetry_comparison.py` | refactorizado | Reemplazadas las
definiciones locales de color por `from pipeline_common.viz_colors import ...`.
Sintaxis verificada con `ast.parse`. `Mapping/visualize_symmetry.py` NO se tocó:
usa nombres/valores distintos (`PRED_COLOR`, `GT_PLANE_COLOR`, sin equivalente
exacto a todos los de arriba) — no es una duplicación literal, se deja para una
revisión aparte si se decide unificar también su convención.

`EXPERIMENT_ROADMAP.md`, `POST_MOLMO_PIPELINE_COMMANDS.md`,
`PROJECT_CONTEXT_REPORT.md`, `compass_artifact_wf-a967ac39-4e37-5e44-afe7-a6468672dbe7_text_markdown.md`,
`_nb_sources.txt` | movidos a `docs/archive/` | Documentos sueltos en la raíz sin
referencia desde `README.md`/`CLAUDE.md`, con contenido de notas de sesión/roadmap
puntual, no documentación de referencia viva. Se archivaron (no se borraron) por
si tienen valor histórico.

`prompt-audit.md`, `prompt-generate-claude-md.md`, `prompt-refactor.md` | sin
cambios | Son los prompts activamente en uso en esta misma sesión de refactor;
no se archivaron para no interrumpir el flujo de trabajo en curso. Quedan como
candidato a mover a `docs/archive/` en una pasada futura si ya no se van a reusar.

`cleanup_all_experiments.sh`, `find_extra_renders.sh` | movidos a `Mapping/` |
Viven junto a `Mapping/cleanup_experiments.py`, el script Python equivalente;
antes estaban sueltos en la raíz sin motivo aparente. `README.md` actualizado
para reflejar la nueva ruta (`bash Mapping/cleanup_all_experiments.sh`).

`ExploratoryDataAnayisis/` → `ExploratoryDataAnalysis/` | renombrado (`git mv`) |
Typo que además no coincidía con el nombre que `README.md` ya usaba. Sin
referencias internas a la ruta vieja encontradas (el notebook no hardcodea su
propio directorio).

`Mapping/__pycache__/`, `MolmoPointing/__pycache__/`, `pipeline_common/__pycache__/` |
eliminados | No trackeados en git (cubiertos por `.gitignore`), incluían un
`.pyc` obsoleto de `reference_metrics.py` (módulo ya eliminado). Artefacto local
sin efecto en el repo versionado.

## Fase 3 — Normas de código

No se ejecutó una pasada de tipado/docstrings general: la auditoría no encontró
funciones sin docstring en el pipeline principal, y dividir `Mapping/evaluate.py`
o `MolmoPointing/molmo_multiview_runner.py` (los dos archivos grandes/multi-responsabilidad
identificados) tocaría lógica de evaluación/inferencia ya validada — **se deja
pendiente y se marca como "parar y preguntar"** según la regla del prompt de
refactor, no se hizo sin confirmación explícita.

## Fase 4 — Notebooks

`ExploratoryDataAnayisis/EDA.ipynb` (ahora `ExploratoryDataAnalysis/EDA.ipynb`) |
sin cambios | Tiene execution_counts fuera de orden (auditoría §3), pero
re-ejecutarlo requiere los datos/dependencias del entorno de trabajo original —
**se deja pendiente**, no se re-ejecutó sin confirmación.

## Fase 5 — Documentación

`README.md` | actualizado | Árbol de estructura del repo corregido: agrega
`pipeline_common/triangulation.py`, `Mapping/estimate_symmetry_no_mesh.py`,
`Mapping/archive/`, los scripts de diagnóstico de `Mapping/`, y reemplaza el
listado plano de `docs/` por una sección de "Module documentation" con enlaces
a cada doc (incluye `docs/audits/` y `docs/archive/`, quita el link roto a
`EXPERIMENT_ROADMAP.md` ahora archivado). Referencias a `cleanup_all_experiments.sh`
actualizadas a su nueva ruta.

`docs/data-schemas.md` | creado | Junta en un solo lugar el esquema JSON de
`molmo_multiview*.json`, `mapped_points_3d*.json`, `predicted_symmetry*.json`
y `eval_*_results.json`, antes disperso en docstrings de scripts individuales.

`Mapping/estimate_symmetry_no_mesh.py` | docstring corregido | Decía que el
wiring de `evaluate.py` con `triangulation_multiplane`/`evaluate_plane_multiset`
"está pendiente" — ya se conectó en la sesión anterior (incluyendo SDE_ref/F1_ref).
Actualizado para reflejar el estado real.

`docs/audits/architecture-audit.md` | creado (sesión anterior a este log) |
Auditoría de solo lectura que generó el plan ejecutado en este log.

## Abierto / no tocado (requiere decisión del autor)

- `InteractiveViewer/view_symmetries.py` vs. `Mapping/visualize_symmetry.py`:
  posible solapamiento, no verificado línea por línea — auditoría lo señala
  como pregunta abierta, no una duplicación confirmada.
- `ranking_postprocesamiento.ipynb`: helpers reusables (`enrich`, `inventory`,
  `best_per_experiment`, `plot_ablation_bar`) siguen viviendo solo en el
  notebook. Extraerlos a un módulo tocaría el notebook principal de análisis
  de resultados — se deja para una pasada explícita si se decide.
- División de `Mapping/evaluate.py` / `MolmoPointing/molmo_multiview_runner.py`
  en módulos más chicos — ver Fase 3.
- Re-ejecución de `ExploratoryDataAnalysis/EDA.ipynb` — ver Fase 4.
