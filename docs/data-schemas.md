# Esquemas de los JSON acumulativos del pipeline

Referencia rápida de la forma de cada archivo que las etapas del pipeline
leen/escriben bajo `<renders_root>/<symmetry_type>/<object_id>/<size>/<lighting>/`.
Antes esta información estaba dispersa en docstrings de cada script; este
documento la junta en un solo lugar. El nombre de archivo real siempre puede
llevar el sufijo `_<experiment_id>` (ver `pipeline_common/naming.py`).

## `molmo_multiview[_<EXP>].json`

Escrito por `MolmoPointing/molmo_multiview_runner.py`. Acumulativo por
`n_views` — cada corrida sólo agrega las claves que faltan.

```jsonc
{
  "6": {                         // clave = n_views del grupo Fibonacci
    "points": [[x_molmo, y_molmo], ...],   // escala [0,1000], independiente de resolución
    "point_obj_ids": [0, 1, ...],           // a qué vista/objeto de imagen corresponde cada punto
    "camera": {"R": [...], "T": [...], "fov_deg": 60.0},
    "n_views_used": 6,
    ...
  },
  "14": {...}
}
```

## `mapped_points_3d[_<EXP>].json` (solo pipeline con malla)

Escrito por `Mapping/map_to_3d.py` vía ray casting contra el `.obj`. Un punto
2D de Molmo por punto 3D (o `null` si el rayo no impacta la malla).

```jsonc
{
  "6": {
    "hit_points_3d": [[x, y, z], null, ...],
    "patch_size": 1
  }
}
```

## `predicted_symmetry[_<EXP>].json`

Escrito por `Mapping/estimate_symmetry.py` (con malla, 4 métodos:
`svd`/`ransac_svd`/`svd_sde`/`ransac_svd_sde`) o
`Mapping/estimate_symmetry_no_mesh.py` (sin malla, métodos
`triangulation`/`triangulation_multiplane`).

Forma de un solo eje/plano (`axis_sym`, o `plane_sym` con `--max-planes 1`):

```jsonc
{
  "6": {
    "svd": {
      "direction_or_normal": [nx, ny, nz],
      "origin": [ox, oy, oz],
      "n_points": 12,
      "n_views_used": 6,
      "n_inliers": null,      // solo RANSAC
      "sde": 0.031,           // heurística interna (KDTree), ver CLAUDE.md
      "accepted": true        // sde < SDE_THRESHOLD (0.05)
    }
  }
}
```

Forma multi-plano (`plane_sym`, `--max-planes > 1`, método `triangulation_multiplane`):

```jsonc
{
  "14": {
    "triangulation_multiplane": {
      "planes": [
        {"normal": [...], "origin": [...], "n_views_used": 7, ...},
        {"normal": [...], "origin": [...], "n_views_used": 7, ...}
      ]
    }
  }
}
```

## `eval_<size>_<lighting>_<EXP>_<method>_results.json`

Escrito por `Mapping/evaluate.py`. Un objeto por `object_id`, con métricas
por `n_views`. Para métodos de un solo plano/eje incluye
`angular_error_deg`, `translation_error`, y (opt-in con
`--with-reference-metrics`) `sde_ref`/`f1_counts_ref`/`f1_counts_ref_hungarian`.
Para `triangulation_multiplane` la forma cambia a recall/precisión sobre el
conjunto completo de planos (`n_planes_predicted`, `n_true_planes`,
`recall_planes`, `precision_planes`, y opcionalmente `sde_ref_per_plane`).
Ver `docs/metricas_evaluacion.md` para la definición exacta de cada métrica
y `docs/actualizacion_metricas.md` para el historial de qué se agregó/eliminó.

## Etiqueta ground truth (`.txt`)

```
# axis_sym
1
axis DX DY DZ  OX OY OZ

# plane_sym (1-3 planos)
2
plane NX NY NZ  OX OY OZ
plane NX NY NZ  OX OY OZ
```
