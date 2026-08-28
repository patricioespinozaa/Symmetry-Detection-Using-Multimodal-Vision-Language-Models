"""
select_sandbox_objects.py
--------------------------
Selecciona 10 BUENO / 10 MEDIO / 10 MALO por objeto, para axis_v06_nomesh
(por angular_error) y para plane_v04_1_nomesh (por F1 por-objeto), a partir
de los eval_..._results.json que ya escribio Mapping/evaluate.py. No corre
nada nuevo -- reprocesa los resultados existentes.

====================================================================
OJO -- f1_ref NO es un valor por objeto (aclaracion importante)
====================================================================
`Mapping/evaluate.py` calcula `f1_ref` como una metrica a NIVEL DATASET:
acumula TP/FP/FN de TODOS los objetos juntos por umbral, y reci'en ahi
calcula F1 (ver `compute_summary`, docstring: "F1_ref: dataset-level TP/FP/FN
accumulation... NOT averaged from a per-object F1"). No existe un "f1_ref de
este objeto en particular" en el JSON de resultados.

Lo que SI existe por objeto (`evaluate_plane_multiset`, ya guardado en
`eval_..._results.json`) es `recall_planes` y `precision_planes`. Este script
calcula, para bucketing, un **F1 por-objeto aproximado**:

    f1_por_objeto = 2 * recall_planes * precision_planes / (recall_planes + precision_planes)

Esto es una metrica *derivada*, no la `f1_ref` oficial que reporta
`evaluate.py` -- se usa solo para ORDENAR objetos de mejor a peor de forma
razonable, no para reportar como resultado final. Si preferis ordenar solo
por `recall_planes` (el criterio que ya usa "GANADOR plano" en
`Experiments/analisis_prompts_no_mesh.ipynb`), pasa `--plane-metric recall`.

====================================================================
Criterio de bucketing
====================================================================
Por cada symmetry_type, se ordenan los objetos con prediccion valida
(status=="ok") por su metrica en el n_views elegido:
  - axis:  angular_error_deg ASCENDENTE (menor = mejor)
  - plane: f1_por_objeto (o recall_planes) DESCENDENTE (mayor = mejor)

BUENO  = primeros --n-per-bucket
MALO   = ultimos  --n-per-bucket
MEDIO  = --n-per-bucket centrados en la mediana de la lista ordenada

Usage
-----
    python Mapping/select_sandbox_objects.py \\
        --renders-root ../data/renders \\
        --axis-experiment-id axis_v06_nomesh --axis-n-views 26 \\
        --plane-experiment-id plane_v04_1_nomesh --plane-n-views 14 \\
        --n-per-bucket 10 \\
        --out ../results/diagnostics/sandbox_object_selection.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_axis_metric(renders_root: Path, experiment_id: str, n_views: int,
                     sizes: list[int], lightings: list[str]) -> dict[str, float]:
    size_tag  = "s" + "_".join(str(s) for s in sizes)
    light_tag = "_".join(lightings)
    path = renders_root / "axis_sym" / f"eval_{size_tag}_{light_tag}_{experiment_id}_triangulation_results.json"
    if not path.exists():
        raise SystemExit(f"[error] No se encontro {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    out = {}
    nv_key = str(n_views)
    for obj_id, per_nv in data.get("objects", {}).items():
        if per_nv is None or nv_key not in per_nv:
            continue
        m = per_nv[nv_key]
        if isinstance(m, dict) and m.get("status") == "ok":
            out[obj_id] = m["angular_error_deg"]
    return out


def load_plane_metric(renders_root: Path, experiment_id: str, n_views: int,
                      sizes: list[int], lightings: list[str], metric: str) -> dict[str, float]:
    size_tag  = "s" + "_".join(str(s) for s in sizes)
    light_tag = "_".join(lightings)
    path = renders_root / "plane_sym" / f"eval_{size_tag}_{light_tag}_{experiment_id}_triangulation_multiplane_results.json"
    if not path.exists():
        raise SystemExit(f"[error] No se encontro {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    out = {}
    nv_key = str(n_views)
    for obj_id, per_nv in data.get("objects", {}).items():
        if per_nv is None or nv_key not in per_nv:
            continue
        m = per_nv[nv_key]
        if not isinstance(m, dict) or m.get("status") != "ok":
            continue
        r, p = m.get("recall_planes"), m.get("precision_planes")
        if r is None or p is None:
            continue
        if metric == "recall":
            out[obj_id] = r
        else:  # "f1" (default) -- F1 por-objeto aproximado, ver docstring del modulo
            out[obj_id] = 2 * r * p / (r + p) if (r + p) > 0 else 0.0
    return out


def bucket_objects(metric_by_id: dict[str, float], n_per_bucket: int,
                   higher_is_better: bool) -> dict[str, list[str]]:
    """BUENO/MEDIO/MALO, n_per_bucket cada uno, sin overlap (asume
    len(metric_by_id) >= 3*n_per_bucket, si no recorta con un warning)."""
    ordered = sorted(metric_by_id, key=lambda k: metric_by_id[k], reverse=higher_is_better)
    n = len(ordered)
    if n < 3 * n_per_bucket:
        print(f"[warn] Solo {n} objetos validos disponibles, menos que 3*{n_per_bucket}={3*n_per_bucket} "
              f"-- los buckets pueden solaparse o quedar mas chicos.")

    bueno = ordered[:n_per_bucket]
    malo  = ordered[-n_per_bucket:]
    mid_start = max(n // 2 - n_per_bucket // 2, 0)
    medio = ordered[mid_start: mid_start + n_per_bucket]

    return {"BUENO": bueno, "MEDIO": medio, "MALO": malo}


def print_and_format(buckets: dict[str, list[str]], metric_by_id: dict[str, float], label: str) -> list[str]:
    formatted = []
    print(f"\n=== {label} ===")
    for cat in ("BUENO", "MEDIO", "MALO"):
        print(f"-- {cat} --")
        for oid in buckets[cat]:
            entry = f"{cat}_{oid}_2d"
            formatted.append(entry)
            print(f"  {entry}   (metrica={metric_by_id[oid]:.4f})")
    return formatted


def main() -> None:
    p = argparse.ArgumentParser(
        description="Selecciona 10 BUENO/10 MEDIO/10 MALO para axis_v06_nomesh y "
                     "plane_v04_1_nomesh, a partir de eval_..._results.json ya generados.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--renders-root", required=True)
    p.add_argument("--sizes", type=int, nargs="+", default=[224])
    p.add_argument("--lightings", type=str, nargs="+", default=["flat"])

    p.add_argument("--axis-experiment-id", default="axis_v06_nomesh")
    p.add_argument("--axis-n-views", type=int, default=26,
                   help="n_views a usar para rankear el eje (26 = mejor n_views empirico de axis_v06).")

    p.add_argument("--plane-experiment-id", default="plane_v04_1_nomesh")
    p.add_argument("--plane-n-views", type=int, default=14,
                   help="n_views a usar para rankear el plano (14 = optimo empirico observado).")
    p.add_argument("--plane-metric", choices=["f1", "recall"], default="f1",
                   help="'f1' = F1 por-objeto aproximado (ver docstring, NO es el f1_ref oficial); "
                        "'recall' = recall_planes por objeto, igual criterio que 'GANADOR plano' "
                        "en el notebook de ranking.")

    p.add_argument("--n-per-bucket", type=int, default=10)
    p.add_argument("--out", default=None, help="Si se pasa, guarda la seleccion completa en JSON.")
    args = p.parse_args()

    renders_root = Path(args.renders_root)

    axis_metric = load_axis_metric(
        renders_root, args.axis_experiment_id, args.axis_n_views, args.sizes, args.lightings,
    )
    print(f"[axis]  {len(axis_metric)} objetos con prediccion valida "
          f"({args.axis_experiment_id}, n_views={args.axis_n_views})")
    axis_buckets = bucket_objects(axis_metric, args.n_per_bucket, higher_is_better=False)
    axis_formatted = print_and_format(axis_buckets, axis_metric, f"AXIS -- {args.axis_experiment_id} (angular_error_deg, menor=mejor)")

    plane_metric = load_plane_metric(
        renders_root, args.plane_experiment_id, args.plane_n_views, args.sizes, args.lightings, args.plane_metric,
    )
    print(f"\n[plane] {len(plane_metric)} objetos con prediccion valida "
          f"({args.plane_experiment_id}, n_views={args.plane_n_views})")
    plane_buckets = bucket_objects(plane_metric, args.n_per_bucket, higher_is_better=True)
    metric_label = "F1 por-objeto aprox." if args.plane_metric == "f1" else "recall_planes"
    plane_formatted = print_and_format(plane_buckets, plane_metric, f"PLANE -- {args.plane_experiment_id} ({metric_label}, mayor=mejor)")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "axis": {
                "experiment_id": args.axis_experiment_id, "n_views": args.axis_n_views,
                "metric": "angular_error_deg", "buckets": axis_buckets,
                "raw_object_list": axis_formatted,
            },
            "plane": {
                "experiment_id": args.plane_experiment_id, "n_views": args.plane_n_views,
                "metric": args.plane_metric, "buckets": plane_buckets,
                "raw_object_list": plane_formatted,
            },
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[ok] Guardado: {out_path}")

    print("\n\n--- Listas listas para pegar en el notebook (RAW_OBJECT_LIST_AXIS / RAW_OBJECT_LIST_PLANE) ---")
    print("\nRAW_OBJECT_LIST_AXIS = [")
    for e in axis_formatted:
        print(f'    "{e}",')
    print("]")
    print("\nRAW_OBJECT_LIST_PLANE = [")
    for e in plane_formatted:
        print(f'    "{e}",')
    print("]")


if __name__ == "__main__":
    main()
