#!/usr/bin/env python3
"""
Genera un resumen en Markdown (mismo formato que docs/resultados/Flujo_A.md)
a partir de los CSV de comparación de una corrida ndcfix
(ndc_fix/<carpeta>/experiments_<fecha>/*_comparison.csv).

Uso:
  python utils/generar_resumen_resultados.py --carpeta ndc_fix/ndcfix_6pts \
      --titulo "Prompts de 6 puntos por vista" --salida docs/resultados/Prompts_6pts.md \
      --base axis_v06_6pts=axis_v06 --base plane_v05_6pts=plane_v05 \
      --conclusiones notas.md

--base X=Y agrega una sección que compara el prompt X con el prompt Y del
Flujo A (ndc_fix/ndcfix_flujoA). --conclusiones inserta un texto propio en la
sección 2.
"""
import argparse
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
FLUJO_A = REPO / "ndc_fix" / "ndcfix_flujoA"
NV = [6, 14, 26]

# (columna, título, mayor_es_mejor, formato)
METRICAS_COMUNES = [
    ("auc_angular", "AUC angular (trunc. 45°)", True, "{:.3f}"),
    ("angular_error_mean", "Error angular medio (°)", False, "{:.1f}"),
    ("angular_error_median", "Error angular mediano (°)", False, "{:.1f}"),
    ("precision_5deg", "Precisión@5°", True, "{:.3f}"),
    ("precision_10deg", "Precisión@10°", True, "{:.3f}"),
    ("precision_15deg", "Precisión@15°", True, "{:.3f}"),
    ("translation_error_normalized_median", "Error de traslación normalizado (mediana)", False, "{:.3f}"),
    ("sde_ref_mean", "SDE_ref medio", False, "{:.3f}"),
    ("n_objects", "Objetos con predicción (de 850)", True, "{:.0f}"),
]
_UMBRALES = [("t005", "0.05"), ("t010", "0.10"), ("t015", "0.15"), ("t020", "0.20")]
METRICAS_PLANO = (
    [("f1_ref", "F1_ref (voraz, promedio de los 4 umbrales)", True, "{:.3f}")]
    + [(f"f1_ref_{t}", f"F1_ref (voraz) @t={v}", True, "{:.3f}") for t, v in _UMBRALES]
    + [("f1_ref_hungarian", "F1_ref (Hungarian, promedio de los 4 umbrales)", True, "{:.3f}")]
    + [(f"f1_ref_hungarian_{t}", f"F1_ref (Hungarian) @t={v}", True, "{:.3f}") for t, v in _UMBRALES]
)


def etiqueta(exp: str) -> str:
    """axis_v06_6pts_ndcfix_nomesh -> v06_6pts ; plane_v08_ndcfix_mp1_nomesh -> v08"""
    e = re.sub(r"^(axis|plane)_", "", exp)
    return re.sub(r"_ndcfix(_mp1)?_nomesh$", "", e)


def cargar(carpeta: Path, tipo: str) -> pd.DataFrame | None:
    csvs = sorted(carpeta.glob(f"experiments_*/{tipo}_*comparison.csv"))
    if not csvs:
        return None
    d = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    d = d[d["method"] == "triangulation"].copy()
    d["prompt"] = d["experiment"].map(etiqueta)
    return d


def pivot(d, col):
    return d.pivot_table(index="prompt", columns="n_views", values=col, aggfunc="first").reindex(columns=NV)


def tabla_metrica(d, col, titulo, mayor, fmt):
    if col not in d.columns:
        return ""
    p = pivot(d, col)
    media = p.mean(axis=1)
    mejor = media.idxmax() if mayor else media.idxmin()
    flecha = "↑ mayor es mejor" if mayor else "↓ menor es mejor"
    out = [f"#### {titulo} ({flecha})", "", "| Prompt | 6 vistas | 14 vistas | 26 vistas | Media |",
           "|---|---|---|---|---|"]
    for pr, fila in p.iterrows():
        celdas = [fmt.format(v) if pd.notna(v) else "–" for v in fila]
        m = fmt.format(media[pr])
        nombre = f"**{pr}**" if pr == mejor and len(p) > 1 else pr
        if pr == mejor and len(p) > 1:
            m = f"**{m}**"
        out.append(f"| {nombre} | " + " | ".join(celdas) + f" | {m} |")
    return "\n".join(out) + "\n"


def ranking(d, plano):
    cols = ["auc_angular", "angular_error_median", "precision_10deg"]
    cols += (["f1_ref", "f1_ref_hungarian"] if plano else []) + ["sde_ref_mean", "n_objects"]
    cols = [c for c in cols if c in d.columns]
    m = d.groupby("prompt")[cols].mean().sort_values("auc_angular", ascending=False)
    head = "| Rank | Prompt | " + " | ".join(cols) + " |"
    out = [head, "|" + "---|" * (len(cols) + 2)]
    for i, (pr, r) in enumerate(m.iterrows(), 1):
        vals = []
        for c in cols:
            v = r[c]
            vals.append(f"{v:.1f}" if c in ("angular_error_median", "n_objects") else f"{v:.3f}")
        out.append(f"| {i} | {pr} | " + " | ".join(vals) + " |")
    return "\n".join(out) + "\n"


def mejor_nv(dax, dpl):
    prompts = sorted(set(dax["prompt"] if dax is not None else []) |
                     set(dpl["prompt"] if dpl is not None else []))
    out = ["| Prompt | Axis: mejor n_views | Axis: AUC | Plane: mejor n_views (AUC) | Plane: AUC |",
           "|---|---|---|---|---|"]
    for pr in prompts:
        celdas = []
        for d in (dax, dpl):
            sub = None if d is None else d[d["prompt"] == pr]
            if sub is None or sub.empty:
                celdas += ["–", "–"]
            else:
                r = sub.loc[sub["auc_angular"].idxmax()]
                celdas += [str(int(r["n_views"])), f"{r['auc_angular']:.3f}"]
        out.append(f"| {pr} | " + " | ".join(celdas) + " |")
    return "\n".join(out) + "\n"


def comparacion_base(d, bases, tipo):
    """bases: lista de (label_nuevo, label_base) restringida a este tipo."""
    base_df = cargar(FLUJO_A, tipo)
    if d is None or base_df is None or not bases:
        return ""
    cols = [("auc_angular", "AUC", "{:.3f}"), ("angular_error_median", "Error mediano (°)", "{:.1f}"),
            ("precision_10deg", "P@10°", "{:.3f}"), ("n_objects", "Objetos", "{:.0f}")]
    if tipo == "plane_sym":
        cols.insert(3, ("f1_ref_hungarian", "F1_ref (H)", "{:.3f}"))
        cols.insert(3, ("f1_ref", "F1_ref (voraz)", "{:.3f}"))
    out = ["| Prompt (vs. base Flujo A) | n_views | " + " | ".join(c[1] for c in cols) + " |",
           "|" + "---|" * (len(cols) + 2)]
    for nuevo, base in bases:
        for nv in NV:
            fn = d[(d["prompt"] == nuevo) & (d["n_views"] == nv)]
            fb = base_df[(base_df["prompt"] == base) & (base_df["n_views"] == nv)]
            if fn.empty or fb.empty:
                continue
            fn, fb = fn.iloc[0], fb.iloc[0]
            celdas = []
            for c, _, fmt in cols:
                celdas.append(f"{fmt.format(fb[c])} → {fmt.format(fn[c])}")
            out.append(f"| {nuevo} (vs. {base}) | {nv} | " + " | ".join(celdas) + " |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--carpeta", required=True)
    ap.add_argument("--titulo", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--base", action="append", default=[], help="nuevo=base (labels con prefijo axis_/plane_)")
    ap.add_argument("--conclusiones", default=None, help="archivo .md con el texto de la sección 2")
    args = ap.parse_args()

    carpeta = (REPO / args.carpeta) if not Path(args.carpeta).is_absolute() else Path(args.carpeta)
    dax, dpl = cargar(carpeta, "axis_sym"), cargar(carpeta, "plane_sym")
    fechas = sorted({p.name for p in carpeta.glob("experiments_*")})
    fuentes = sorted(p.name for p in carpeta.glob("experiments_*/*_comparison.csv"))
    n_prompts = len(set(dax["prompt"]) if dax is not None else set()) + \
        len(set(dpl["prompt"]) if dpl is not None else set())

    pares = [b.split("=") for b in args.base]
    bases_ax = [(etiqueta(k), etiqueta(v)) for k, v in pares if k.startswith("axis_")]
    bases_pl = [(etiqueta(k), etiqueta(v)) for k, v in pares if k.startswith("plane_")]

    md = [f"# Resumen de resultados por prompt — `{carpeta.name}` ({', '.join(fechas)})", "",
          f"{args.titulo}.", "",
          "Fuentes: " + ", ".join(f"`{f}`" for f in fuentes) + ".",
          f"{n_prompts} prompts × 3 configuraciones de vistas (6, 14, 26), método `triangulation` "
          "(pipeline sin malla, conversión NDC corregida, `n_total = 850` objetos por celda; plano en modo "
          "de un solo plano, `max_planes = 1`). En las tablas, **negrita** = mejor prompt según el promedio "
          "sobre las 3 configuraciones de vistas (solo cuando hay más de un prompt).", "",
          "> **Cautelas de lectura**",
          "> - `n_objects` es la cantidad de objetos con predicción válida (de 850). Las métricas angulares "
          "imputan 90° a los objetos sin predicción; la traslación, SDE_ref y F1_ref se calculan solo sobre los "
          "objetos con predicción. Compará siempre contra la tabla de cobertura.",
          "> - `AUC_angular` trunca en 45°. `SDE_ref`/`F1_ref` no son intercambiables con "
          "`angular_error`/`translation_error` (ver `docs/metricas_evaluacion.md`).",
          "> - Un error angular mediano ≈ 57–60° (axis) o ≈ 83–87° (plane) es compatible con nivel de azar / "
          "predicción trivial constante, no con aprendizaje de simetría.", ""]

    md += ["## 1. Ranking global (promedio sobre 6/14/26 vistas, ordenado por AUC angular)", ""]
    if dax is not None:
        md += ["### Simetría axial", ranking(dax, False)]
    if dpl is not None:
        md += ["### Simetría planar", ranking(dpl, True)]

    md += ["## 2. Conclusiones principales", ""]
    if args.conclusiones:
        md += [Path(args.conclusiones).read_text(encoding="utf-8").strip(), ""]

    if bases_ax or bases_pl:
        md += ["## 3. Comparación con el prompt base del Flujo A", "",
               "Cada celda muestra *Flujo A → este experimento* para el mismo número de vistas. "
               "Fuente del Flujo A: `ndc_fix/ndcfix_flujoA`.", ""]
        if bases_ax:
            md += ["### Simetría axial", comparacion_base(dax, bases_ax, "axis_sym")]
        if bases_pl:
            md += ["### Simetría planar", comparacion_base(dpl, bases_pl, "plane_sym")]
        sec = 4
    else:
        sec = 3

    md += [f"## {sec}. Mejor configuración (n_views) por prompt", "", mejor_nv(dax, dpl)]
    md += [f"## {sec + 1}. Tablas por métrica y tipo de simetría", "", "Columnas = número de vistas.", ""]
    if dax is not None:
        md += ["# Parte A — Simetría axial", ""]
        md += [tabla_metrica(dax, *m) for m in METRICAS_COMUNES]
    if dpl is not None:
        md += ["# Parte B — Simetría planar", ""]
        md += [tabla_metrica(dpl, *m) for m in METRICAS_COMUNES + METRICAS_PLANO]

    salida = REPO / args.salida
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text("\n".join(md), encoding="utf-8")
    print(f"[ok] {salida}")


if __name__ == "__main__":
    main()
