#!/usr/bin/env python3
"""
Imprime en terminal una tabla con el conteo detallado de objetos por
categoria de ShapeNet y tipo de simetria (axial / planar), con porcentajes,
a partir del CSV generado por asignar_categorias_v2.py.

Columnas:
  Categoria | N_axial | %_axial | N_planar | %_planar | N_total | %_total

  %_axial  = porcentaje de la categoria dentro del total de objetos axiales
  %_planar = porcentaje de la categoria dentro del total de objetos planares
  %_total  = porcentaje de la categoria dentro del total general (axial+planar)

Uso:
    ./tabla_categorias.py --csv categorias_objetos.csv
    ./tabla_categorias.py --csv categorias_objetos.csv --sort-by total --top 15
    ./tabla_categorias.py --csv categorias_objetos.csv --latex tabla.tex
"""
import argparse
import csv
from collections import Counter


def read_counts(csv_path):
    counts = {"axial": Counter(), "planar": Counter()}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sym = row["tipo_simetria"]
            cat = row["nombre_categoria"] or row["synsetId"]
            if sym in counts:
                counts[sym][cat] += 1
    return counts


def build_rows(counts, sort_by, top):
    all_cats = set(counts["axial"]) | set(counts["planar"])
    n_axial = sum(counts["axial"].values()) or 1
    n_planar = sum(counts["planar"].values()) or 1
    n_total = n_axial + n_planar

    rows = []
    for cat in all_cats:
        a = counts["axial"][cat]
        p = counts["planar"][cat]
        t = a + p
        rows.append({
            "categoria": cat,
            "n_axial": a, "pct_axial": 100 * a / n_axial,
            "n_planar": p, "pct_planar": 100 * p / n_planar,
            "n_total": t, "pct_total": 100 * t / n_total,
        })

    key = {"total": lambda r: r["n_total"], "axial": lambda r: r["n_axial"],
          "planar": lambda r: r["n_planar"], "alpha": lambda r: r["categoria"]}[sort_by]
    reverse = sort_by != "alpha"
    rows.sort(key=key, reverse=reverse)
    if top:
        rows = rows[:top]
    return rows, n_axial, n_planar, n_total


def print_table(rows, n_axial, n_planar, n_total):
    cat_w = max(9, max((len(r["categoria"]) for r in rows), default=9))
    header = (f"{'Categoria':<{cat_w}}  {'N_axial':>8}  {'%_axial':>8}  "
              f"{'N_planar':>9}  {'%_planar':>9}  {'N_total':>8}  {'%_total':>8}")
    rule = "-" * len(header)

    print(rule)
    print(header)
    print(rule)
    for r in rows:
        print(f"{r['categoria']:<{cat_w}}  {r['n_axial']:>8d}  {r['pct_axial']:>7.1f}%  "
              f"{r['n_planar']:>9d}  {r['pct_planar']:>8.1f}%  "
              f"{r['n_total']:>8d}  {r['pct_total']:>7.1f}%")
    print(rule)
    print(f"{'TOTAL':<{cat_w}}  {n_axial:>8d}  {100.0:>7.1f}%  "
          f"{n_planar:>9d}  {100.0:>8.1f}%  {n_total:>8d}  {100.0:>7.1f}%")
    print(rule)


def write_latex(path, rows, n_axial, n_planar, n_total, caption, label):
    def esc(s):
        return s.replace("_", "\\_").replace("&", "\\&")

    lines = [
        "% Requiere \\usepackage{booktabs} en el preambulo",
        "\\begin{table}[ht]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        "  \\begin{tabular}{lrrrrrr}",
        "    \\toprule",
        "    Categoría & $N_{axial}$ & \\%$_{axial}$ & $N_{planar}$ & "
        "\\%$_{planar}$ & $N_{total}$ & \\%$_{total}$ \\\\",
        "    \\midrule",
    ]
    for r in rows:
        lines.append(
            f"    {esc(r['categoria'])} & {r['n_axial']} & {r['pct_axial']:.1f} & "
            f"{r['n_planar']} & {r['pct_planar']:.1f} & {r['n_total']} & "
            f"{r['pct_total']:.1f} \\\\"
        )
    lines += [
        "    \\midrule",
        f"    Total & {n_axial} & 100.0 & {n_planar} & 100.0 & {n_total} & 100.0 \\\\",
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nTabla LaTeX guardada en: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="salida de asignar_categorias_v2.py")
    ap.add_argument("--sort-by", choices=["total", "axial", "planar", "alpha"],
                    default="total")
    ap.add_argument("--top", type=int, default=None,
                    help="mostrar solo las N categorias mas frecuentes")
    ap.add_argument("--latex", default=None,
                    help="ademas de la tabla en terminal, exportar tabla booktabs a este .tex")
    ap.add_argument("--caption", default="Distribución de objetos por categoría y tipo de simetría")
    ap.add_argument("--label", default="tab:dist_categorias")
    args = ap.parse_args()

    counts = read_counts(args.csv)
    rows, n_axial, n_planar, n_total = build_rows(counts, args.sort_by, args.top)
    if not rows:
        raise SystemExit("[error] No se encontraron categorias en el CSV.")

    print_table(rows, n_axial, n_planar, n_total)

    if args.latex:
        write_latex(args.latex, rows, n_axial, n_planar, n_total, args.caption, args.label)


if __name__ == "__main__":
    main()
