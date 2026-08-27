"""
diagnose_localization_vs_error.py
-----------------------------------
Cruza el diagnostico de localizacion de puntos (Mapping/diagnose_point_localization.py,
Nivel B: %puntos sobre el objeto por objeto x n_views) contra el error final
del pipeline (angular_error / translation_error por objeto, de
eval_..._triangulation_results.json) para responder la pregunta concreta:
**¿los objetos que tienen puntos cayendo fuera del objeto (fondo) son los
mismos objetos que salen mal (outliers) en el error final?**

No genera datos nuevos -- reprocesa el CSV que ya escribio
diagnose_point_localization.py y el JSON que ya escribio evaluate.py. Cero
llamadas nuevas a Molmo2.

====================================================================
DOS ANALISIS DISTINTOS, NO UNO SOLO
====================================================================

1. CORRELACION (Pearson/Spearman) -- ¿hay una relacion continua?
   pct_en_objeto (por objeto x n_views) vs. angular_error / translation_error.
   Signo esperado: NEGATIVO (mas % de puntos sobre el objeto -> menos error).
   Esto responde "en promedio, a traves de todos los objetos, importa esto".

2. TABLA DE CONTINGENCIA (2x2) -- ¿los mismos objetos cumplen AMBAS condiciones?
   Esto es lo que responde literalmente tu pregunta: no alcanza con que la
   correlacion sea negativa (eso podria ser un efecto debil y difuso); lo que
   importa para la hipotesis de "puntos alucinados en el fondo explican los
   outliers" es si el CONJUNTO de objetos con pct_en_objeto bajo COINCIDE con
   el conjunto de objetos que son outliers de error. Se arma asi, por n_views:
     - "pct_en_objeto BAJO"  = por debajo de la MEDIANA de ese n_views group
       (split data-driven, no un umbral arbitrario fijo).
     - "ERROR OUTLIER"       = por encima del percentil --outlier-percentile
       (default 90, o sea el 10% de objetos con peor error) en angular_error
       Y/O translation_error.
   Reporta un 2x2: cuantos objetos caen en (bajo pct_obj, outlier),
   (bajo pct_obj, no outlier), (alto pct_obj, outlier), (alto pct_obj, no
   outlier) -- y el "lift" (cuantas veces mas probable es ser outlier si tu
   pct_en_objeto es bajo, respecto a la tasa base de outliers).

Usage
-----
    python Mapping/diagnose_localization_vs_error.py \\
        --point-localization-csv ../results/diagnostics/axis_v06_point_localization_detail.csv \\
        --renders-root ../data/renders \\
        --experiment-id axis_v06_nomesh \\
        --out ../results/diagnostics/axis_v06_localization_vs_error.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_axis_conditioning import pearson, spearman  # noqa: E402 -- reusa las mismas funciones sin scipy

MOLMO_METHOD = "triangulation"


def load_axis_metrics(renders_root: Path, sizes: list[int], lightings: list[str],
                      experiment_id: str) -> pd.DataFrame:
    """Lee eval_..._triangulation_results.json y devuelve un DataFrame largo:
    object_id, n_views, angular_error_deg, translation_error -- solo status=='ok'."""
    size_tag  = "s" + "_".join(str(s) for s in sizes)
    light_tag = "_".join(lightings)
    path = renders_root / "axis_sym" / f"eval_{size_tag}_{light_tag}_{experiment_id}_{MOLMO_METHOD}_results.json"
    if not path.exists():
        raise SystemExit(f"[error] No se encontro {path}.")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for obj_id, per_nv in data.get("objects", {}).items():
        if per_nv is None:
            continue
        for nv_key, m in per_nv.items():
            if isinstance(m, dict) and m.get("status") == "ok":
                rows.append({
                    "object_id":         obj_id,
                    "n_views":           int(nv_key),
                    "angular_error_deg": m["angular_error_deg"],
                    "translation_error": m["translation_error"],
                })
    return pd.DataFrame(rows)


def contingency_table(df: pd.DataFrame, pct_col: str, error_col: str,
                      outlier_percentile: float) -> dict:
    """2x2: pct_en_objeto BAJO (< mediana) x error_col OUTLIER (> percentil dado),
    dentro de un mismo n_views (df ya debe venir filtrado a un n_views)."""
    median_pct = df[pct_col].median()
    threshold_err = np.percentile(df[error_col], outlier_percentile)

    low_pct    = df[pct_col] < median_pct
    is_outlier = df[error_col] > threshold_err

    n = len(df)
    both       = int((low_pct & is_outlier).sum())
    only_low   = int((low_pct & ~is_outlier).sum())
    only_out   = int((~low_pct & is_outlier).sum())
    neither    = int((~low_pct & ~is_outlier).sum())

    base_rate   = is_outlier.mean()
    rate_if_low = (low_pct & is_outlier).sum() / max(low_pct.sum(), 1)
    lift        = rate_if_low / base_rate if base_rate > 0 else float("nan")

    return {
        "n_total": n,
        "median_pct_en_objeto": round(float(median_pct), 4),
        "outlier_threshold": round(float(threshold_err), 4),
        "n_bajo_pct_Y_outlier":     both,
        "n_bajo_pct_Y_no_outlier":  only_low,
        "n_alto_pct_Y_outlier":     only_out,
        "n_alto_pct_Y_no_outlier":  neither,
        "tasa_outlier_base":        round(float(base_rate), 4),
        "tasa_outlier_si_bajo_pct": round(float(rate_if_low), 4),
        "lift":                     round(float(lift), 2),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Cruza %puntos-en-objeto (diagnose_point_localization.py) contra "
                     "angular_error/translation_error por objeto -- ver docstring del modulo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--point-localization-csv", required=True,
                   help="CSV Nivel B de diagnose_point_localization.py (--out-detail).")
    p.add_argument("--renders-root", required=True)
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--sizes", type=int, nargs="+", default=[224])
    p.add_argument("--lightings", type=str, nargs="+", default=["flat"])
    p.add_argument("--outlier-percentile", type=float, default=90.0,
                   help="Percentil de error por encima del cual un objeto cuenta como 'outlier'.")
    p.add_argument("--out", default=None, help="CSV combinado (una fila por objeto x n_views).")
    args = p.parse_args()

    renders_root = Path(args.renders_root)

    loc_df = pd.read_csv(args.point_localization_csv)
    loc_df = loc_df[loc_df["experiment_id"] == args.experiment_id].copy()
    if loc_df.empty:
        raise SystemExit(
            f"[error] '{args.point_localization_csv}' no tiene filas con experiment_id="
            f"'{args.experiment_id}'. Revisa que sea el CSV correcto (ver el bug de nombres "
            f"de archivo que encontramos antes -- el nombre del archivo no garantiza el contenido)."
        )
    loc_df["pct_en_objeto"] = loc_df["n_puntos_en_objeto"] / loc_df["n_puntos_molmo2"].replace(0, np.nan)

    err_df = load_axis_metrics(renders_root, args.sizes, args.lightings, args.experiment_id)

    merged = pd.merge(loc_df, err_df, on=["object_id", "n_views"], how="inner")
    merged = merged.dropna(subset=["pct_en_objeto"])
    print(f"[ok] {len(merged)} filas objeto x n_views cruzadas "
          f"(de {len(loc_df)} en localizacion, {len(err_df)} en error)")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_path, index=False)
        print(f"[ok] Guardado: {out_path}\n")

    n_views_present = sorted(merged["n_views"].unique())

    # ── 1. Correlacion ────────────────────────────────────────────────────────
    print("=" * 78)
    print("1. CORRELACION (Pearson / Spearman) -- pct_en_objeto vs. error")
    print("   (signo esperado: NEGATIVO -- mas %, menos error)")
    print("=" * 78)
    header = f"{'error_col':<20}" + "".join(f"  n={nv:<14}" for nv in n_views_present) + "  pooled"
    print(header)
    for err_col in ["angular_error_deg", "translation_error"]:
        line = f"{err_col:<20}"
        for nv in n_views_present:
            sub = merged[merged["n_views"] == nv]
            pr, sr = pearson(sub["pct_en_objeto"].values, sub[err_col].values), \
                     spearman(sub["pct_en_objeto"].values, sub[err_col].values)
            line += f"  {pr:+.3f}/{sr:+.3f}  "
        pr, sr = pearson(merged["pct_en_objeto"].values, merged[err_col].values), \
                 spearman(merged["pct_en_objeto"].values, merged[err_col].values)
        line += f"  {pr:+.3f}/{sr:+.3f}"
        print(line)

    # ── 2. Tabla de contingencia (responde la pregunta: ¿son los MISMOS objetos?) ──
    print()
    print("=" * 78)
    print(f"2. CONTINGENCIA -- pct_en_objeto BAJO (<mediana) x error OUTLIER (>p{args.outlier_percentile:.0f})")
    print("   lift > 1 => ser outlier de error es MAS probable si tenes pct_en_objeto bajo")
    print("   (esto es lo que responde si son o no los MISMOS objetos, no solo una correlacion difusa)")
    print("=" * 78)
    for err_col in ["angular_error_deg", "translation_error"]:
        print(f"\n--- {err_col} ---")
        for nv in n_views_present:
            sub = merged[merged["n_views"] == nv]
            ct = contingency_table(sub, "pct_en_objeto", err_col, args.outlier_percentile)
            print(f"  n_views={nv:3d}  (n={ct['n_total']}, mediana pct_obj={ct['median_pct_en_objeto']:.2%}, "
                  f"umbral outlier={ct['outlier_threshold']:.3f})")
            print(f"    {'':16s}  outlier   no-outlier")
            print(f"    {'pct_obj BAJO':16s}  {ct['n_bajo_pct_Y_outlier']:>7d}   {ct['n_bajo_pct_Y_no_outlier']:>9d}")
            print(f"    {'pct_obj ALTO':16s}  {ct['n_alto_pct_Y_outlier']:>7d}   {ct['n_alto_pct_Y_no_outlier']:>9d}")
            print(f"    tasa base de outliers: {ct['tasa_outlier_base']:.1%}   "
                  f"tasa si pct_obj bajo: {ct['tasa_outlier_si_bajo_pct']:.1%}   "
                  f"LIFT = {ct['lift']:.2f}x")

    print(
        "\nLectura: lift ~1.0 => no hay overlap, son problemas independientes (la hipotesis de "
        "'puntos alucinados en el fondo explican los outliers' NO se sostiene). lift >> 1 (ej. >2) "
        "=> fuerte evidencia de que S1 son, en gran medida, los MISMOS objetos -- vale la pena "
        "filtrar/descartar vistas con puntos fuera del objeto ANTES de triangular, no solo "
        "iterar el prompt."
    )


if __name__ == "__main__":
    main()
