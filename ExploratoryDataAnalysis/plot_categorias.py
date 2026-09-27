#!/usr/bin/env python3
"""
Genera la figura de distribucion de objetos por categoria de ShapeNet y tipo
de simetria (axial / planar), siguiendo convenciones habituales de figuras en
papers y tesis:

  - Paleta Okabe-Ito (colorblind-safe, recomendada por Nature Methods).
  - Patrones de relleno (hatching) redundantes al color, para impresion en
    blanco y negro.
  - Salida vectorial en PDF (para \\includegraphics en LaTeX sin perdida de
    calidad) y PNG a 300 dpi (para previsualizar / Word).
  - Sin titulo dentro de la figura (el titulo/caption va en el documento,
    convencion habitual de figuras de papers); usar --title si lo quieres
    de todas formas para revisar rapido.
  - Ejes sin marco superior/derecho, grilla horizontal tenue, fuente serif
    por defecto (coherente con el cuerpo del texto de la tesis).

Requiere: matplotlib (pip install matplotlib)

Uso basico:
    ./plot_categorias.py --csv categorias_objetos.csv --out figuras/dist_categorias

    (genera figuras/dist_categorias.pdf y figuras/dist_categorias.png)

Opciones utiles:
    --top 20            mostrar solo las 20 categorias mas frecuentes
    --sort-by total|axial|planar|alpha
    --grayscale          fuerza escala de grises (ademas del hatching)
    --width 6.3 --height auto   tamano en pulgadas (6.3in ~ ancho de pagina A4
                                  con margenes de 1in tipico de tesis)
    --usetex              usa el motor LaTeX de matplotlib si esta instalado
                          en el sistema (fuentes identicas al documento)
"""
import argparse
import csv
import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paleta Okabe-Ito (Okabe & Ito, 2002; recomendada por Nature Methods)
COLOR_AXIAL = "#0072B2"   # azul
COLOR_PLANAR = "#D55E00"  # bermellon
GRAY_AXIAL = "#4d4d4d"
GRAY_PLANAR = "#b3b3b3"


def read_counts(csv_path):
    counts = {"axial": Counter(), "planar": Counter()}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sym = row["tipo_simetria"]
            cat = row["nombre_categoria"] or row["synsetId"]
            if sym in counts:
                counts[sym][cat] += 1
    return counts


def order_categories(counts, sort_by, top):
    all_cats = set(counts["axial"]) | set(counts["planar"])
    if sort_by == "alpha":
        ordered = sorted(all_cats)
    elif sort_by == "axial":
        ordered = sorted(all_cats, key=lambda c: counts["axial"][c], reverse=True)
    elif sort_by == "planar":
        ordered = sorted(all_cats, key=lambda c: counts["planar"][c], reverse=True)
    else:  # total
        ordered = sorted(all_cats, key=lambda c: counts["axial"][c] + counts["planar"][c],
                         reverse=True)
    if top:
        ordered = ordered[:top]
    return ordered


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="salida de asignar_categorias_v2.py")
    ap.add_argument("--out", default="dist_categorias",
                    help="ruta base de salida, sin extension (se generan .pdf y .png)")
    ap.add_argument("--top", type=int, default=None,
                    help="mostrar solo las N categorias mas frecuentes")
    ap.add_argument("--sort-by", choices=["total", "axial", "planar", "alpha"],
                    default="total")
    ap.add_argument("--grayscale", action="store_true",
                    help="usar escala de grises en vez de la paleta Okabe-Ito")
    ap.add_argument("--width", type=float, default=6.3, help="ancho en pulgadas")
    ap.add_argument("--height", type=float, default=None,
                    help="alto en pulgadas (por defecto se ajusta al numero de categorias)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--title", default=None, help="titulo opcional dentro de la figura")
    ap.add_argument("--font-size", type=int, default=9)
    ap.add_argument("--usetex", action="store_true")
    ap.add_argument("--labels", action="store_true", default=True,
                    help="mostrar el conteo al final de cada barra (activado por defecto)")
    ap.add_argument("--no-labels", dest="labels", action="store_false")
    args = ap.parse_args()

    counts = read_counts(args.csv)
    categorias = order_categories(counts, args.sort_by, args.top)
    if not categorias:
        raise SystemExit("[error] No se encontraron categorias en el CSV.")

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": args.font_size,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#cccccc",
        "grid.linewidth": 0.5,
        "text.usetex": args.usetex,
    })

    color_axial = GRAY_AXIAL if args.grayscale else COLOR_AXIAL
    color_planar = GRAY_PLANAR if args.grayscale else COLOR_PLANAR

    n = len(categorias)
    height = args.height or max(2.5, 0.28 * n + 0.8)
    fig, ax = plt.subplots(figsize=(args.width, height))

    y = list(range(n))
    bar_h = 0.38
    cats_display = list(reversed(categorias))  # para que la 1a quede arriba
    axial_vals = [counts["axial"][c] for c in cats_display]
    planar_vals = [counts["planar"][c] for c in cats_display]

    ax.barh([i + bar_h / 2 for i in y], axial_vals, height=bar_h,
            label="Axial", color=color_axial, edgecolor="black",
            linewidth=0.4, hatch="//")
    ax.barh([i - bar_h / 2 for i in y], planar_vals, height=bar_h,
            label="Planar", color=color_planar, edgecolor="black",
            linewidth=0.4, hatch="\\\\")

    if args.labels:
        for i, v in enumerate(axial_vals):
            if v > 0:
                ax.text(v, i + bar_h / 2, f" {v}", va="center", fontsize=args.font_size - 2)
        for i, v in enumerate(planar_vals):
            if v > 0:
                ax.text(v, i - bar_h / 2, f" {v}", va="center", fontsize=args.font_size - 2)

    ax.set_yticks(y)
    ax.set_yticklabels(cats_display)
    ax.set_xlabel("Número de objetos")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=False)
    if args.title:
        ax.set_title(args.title)

    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(f"{args.out}.pdf")
    fig.savefig(f"{args.out}.png", dpi=args.dpi)
    print(f"Guardado: {args.out}.pdf  y  {args.out}.png")

    n_axial_tot = sum(counts["axial"].values())
    n_planar_tot = sum(counts["planar"].values())
    print(f"Total axial: {n_axial_tot}  |  Total planar: {n_planar_tot}  |  "
          f"Categorias mostradas: {n} de {len(set(counts['axial']) | set(counts['planar']))}")


if __name__ == "__main__":
    main()
