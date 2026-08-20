"""
Chequeo para el punto de hjNG sobre geometría del ajuste axial: ray-casting
devuelve puntos de SUPERFICIE, no puntos sobre el eje interno de rotación.
Con --point-mode midpoint (pares bilaterales, ver cap3.tex), el punto medio
del par sí tiene una justificación geométrica más directa para aproximar el
eje; con --point-mode independent, el punto de superficie individual depende
de un sesgo más débil (que la vista sea razonablemente perpendicular al eje).

Compara el error angular de las variantes axis_sym con --point-mode
independent (v00, v04, v05) vs --point-mode midpoint (v01, v02, v03),
usando method=svd (sin RANSAC) y n_views=26 -- las condiciones fijas de la
"Fase 1" descrita en cap3.tex, elegidas ahí mismo para no enmascarar la
calidad de los puntos del VLM con la limpieza de outliers de RANSAC.

Uso:
    python Mapping/check_independent_vs_midpoint.py \
        --renders-root ../data/renders \
        --independent-ids axis_v00 axis_v04 axis_v05 \
        --midpoint-ids axis_v01 axis_v02 axis_v03 \
        --method svd \
        --n-views 26 \
        --out-csv scratch/independent_vs_midpoint.csv

    # Con los prompts mejorados (_1):
    python Mapping/check_independent_vs_midpoint.py \
        --renders-root ../data/renders \
        --independent-ids axis_v00_1 axis_v04_1 axis_v05_1 \
        --midpoint-ids axis_v01_1 axis_v03_1 \
        --method svd \
        --n-views 26 \
        --out-csv scratch/independent_vs_midpoint_v1.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_angular_errors(eval_json_path: Path, n_views_key: str) -> list[float]:
    if not eval_json_path.exists():
        print(f"  [aviso] no existe: {eval_json_path}")
        return []
    with open(eval_json_path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("symmetry_type") != "axis_sym":
        raise ValueError(f"{eval_json_path} no es axis_sym")

    errors = []
    for obj_id, per_nviews in (data.get("objects") or {}).items():
        if not per_nviews:
            continue
        m = per_nviews.get(n_views_key) or per_nviews.get(f"n_views_{n_views_key}")
        if not m or m.get("status") != "ok":
            continue
        ang = m.get("angular_error_deg")
        if ang is not None:
            errors.append(ang)
    return errors


def mann_whitney_u(a: np.ndarray, b: np.ndarray) -> tuple[float, float] | None:
    """p-valor bilateral vía scipy, si está disponible (ya es dependencia de Mapping)."""
    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        return None
    stat, p = mannwhitneyu(a, b, alternative="two-sided")
    return float(stat), float(p)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--renders-root", required=True)
    p.add_argument("--independent-ids", nargs="+", required=True,
                   help="experiment-ids con --point-mode independent (ray casting "
                        "devuelve puntos de superficie individuales).")
    p.add_argument("--midpoint-ids", nargs="+", required=True,
                   help="experiment-ids con --point-mode midpoint (punto medio de "
                        "pares bilaterales).")
    p.add_argument("--method", default="svd",
                   help="Método a evaluar. Default 'svd' (sin RANSAC), tal como "
                        "recomienda cap3.tex para aislar la calidad de los puntos.")
    p.add_argument("--n-views", default="26")
    p.add_argument("--out-csv", default=None,
                   help="CSV con una fila por objeto (todas las variantes juntas).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    renders_root = Path(args.renders_root)
    axis_dir = renders_root / "axis_sym"

    groups = {
        "independent": args.independent_ids,
        "midpoint": args.midpoint_ids,
    }

    pooled: dict[str, list[float]] = {"independent": [], "midpoint": []}
    all_rows = []

    for group_name, exp_ids in groups.items():
        print(f"\n=== Grupo: {group_name} ===")
        for exp_id in exp_ids:
            eval_path = axis_dir / f"eval_s224_flat_{exp_id}_{args.method}_results.json"
            errors = load_angular_errors(eval_path, args.n_views)
            if not errors:
                continue
            arr = np.array(errors)
            print(f"  {exp_id:20s} n={len(arr):4d}  mean={arr.mean():6.2f}  "
                  f"median={np.median(arr):6.2f}  p@10deg={(arr < 10).mean():.3f}")
            pooled[group_name].extend(errors)
            for e in errors:
                all_rows.append({"group": group_name, "experiment_id": exp_id,
                                  "angular_error_deg": e})

    print("\n=== Comparación agregada (todas las variantes del grupo, pooled) ===")
    for group_name, errors in pooled.items():
        if not errors:
            continue
        arr = np.array(errors)
        print(f"  {group_name:12s} n={len(arr):4d}  mean={arr.mean():6.2f}  "
              f"median={np.median(arr):6.2f}  p@10deg={(arr < 10).mean():.3f}  "
              f"p@15deg={(arr < 15).mean():.3f}")

    if pooled["independent"] and pooled["midpoint"]:
        ind = np.array(pooled["independent"])
        mid = np.array(pooled["midpoint"])
        result = mann_whitney_u(ind, mid)
        print(f"\nDiferencia de medianas (independent - midpoint) = "
              f"{np.median(ind) - np.median(mid):+.2f} deg")
        if result:
            stat, p_value = result
            print(f"Mann-Whitney U = {stat:.1f}  p-value = {p_value:.4f}  "
                  f"({'diferencia significativa (p<0.05)' if p_value < 0.05 else 'no significativa'})")
        else:
            print("(scipy no disponible: instalar scipy para el test de Mann-Whitney)")

    print("\nInterpretación:")
    print("- Si 'midpoint' tiene error angular sistemáticamente MENOR que 'independent',")
    print("  confirma la objeción del reviewer: el punto medio de un par bilateral")
    print("  aproxima mejor el eje interno que un punto de superficie individual.")
    print("- Si no hay diferencia clara, el argumento del reviewer sigue siendo válido")
    print("  en principio (la justificación teórica es correcta) pero no se refleja en")
    print("  una degradación medible con este dataset/método.")

    if args.out_csv and all_rows:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["group", "experiment_id", "angular_error_deg"])
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nGuardado: {out_path}  ({len(all_rows)} filas)")


if __name__ == "__main__":
    main()
