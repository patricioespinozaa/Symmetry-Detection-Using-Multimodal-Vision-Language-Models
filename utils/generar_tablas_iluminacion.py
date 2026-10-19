#!/usr/bin/env python3
# Genera las tablas LaTeX del experimento de iluminación (224px): un prompt por tipo de
# simetría (axis_v04, plane_v05_1) comparado en flat, darker y brighter.
#
# Uso:
#   python utils/generar_tablas_iluminacion.py                      # escribe el .tex por defecto
#   python utils/generar_tablas_iluminacion.py --salida otra/ruta.tex
#
# Fuentes: flat = ndc_fix/ndcfix_flujoA (corrida base del Flujo A); darker/brighter =
# ndc_fix/ndcfix_224_<iluminación> (se toma el CSV más reciente de cada carpeta).
# Requiere en el preámbulo: booktabs, multirow, adjustbox y \newsavebox{\tablebox}.
import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
NDC_FIX = REPO / "ndc_fix"
SALIDA = REPO / "docs" / "resultados" / "tablas_iluminacion" / "iluminacion_224.tex"
VISTAS = [6, 14, 26]
ILUMINACIONES = ["flat", "darker", "brighter"]

# tipo -> (experimento, carpeta/patrón del CSV de flat, patrón del CSV de darker/brighter)
FUENTES = {
    "axis_sym": ("axis_v04_ndcfix_nomesh", "ndcfix_flujoA/*/axis_sym_14exps_comparison.csv",
                 "axis_sym_axis_v04_ndcfix_nomesh_comparison.csv"),
    "plane_sym": ("plane_v05_1_ndcfix_mp1_nomesh", "ndcfix_flujoA/*/plane_sym_14exps_comparison.csv",
                  "plane_sym_plane_v05_1_ndcfix_mp1_nomesh_comparison.csv"),
}

# tipo -> [(columna, encabezado, mayor_es_mejor, formato)]
COLUMNAS = {
    "axis_sym": [
        ("n_objects", "Objetos", True, "{:.0f}"),
        ("auc_angular", r"AUC$_{45^\circ}$", True, "{:.3f}"),
        ("angular_error_median", r"\begin{tabular}[b]{@{}c@{}}Mediana \\ ($^\circ$)\end{tabular}",
         False, "{:.1f}"),
        ("precision_5deg", r"P@5$^\circ$", True, "{:.3f}"),
        ("precision_10deg", r"P@10$^\circ$", True, "{:.3f}"),
        ("translation_error_normalized_median",
         r"\begin{tabular}[b]{@{}c@{}}Trasl. \\ norm.\end{tabular}", False, "{:.4f}"),
        ("sde_ref_mean", r"SDE$_{\mathrm{ref}}$", False, "{:.4f}"),
    ],
    "plane_sym": [
        ("n_objects", "Objetos", True, "{:.0f}"),
        ("f1_ref", r"\begin{tabular}[b]{@{}c@{}}$F1_{\mathrm{ref}}$ \\ voraz\end{tabular}",
         True, "{:.3f}"),
        ("f1_ref_hungarian",
         r"\begin{tabular}[b]{@{}c@{}}$F1_{\mathrm{ref}}$ \\ húngaro\end{tabular}", True, "{:.3f}"),
        ("auc_angular", r"AUC$_{45^\circ}$", True, "{:.3f}"),
        ("precision_10deg", r"P@10$^\circ$", True, "{:.3f}"),
        ("angular_error_mean", r"\begin{tabular}[b]{@{}c@{}}Err.\ medio \\ ($^\circ$)\end{tabular}",
         False, "{:.1f}"),
        ("sde_ref_mean", r"SDE$_{\mathrm{ref}}$", False, "{:.4f}"),
    ],
}

NOTA_ILUMINACION = (r"Iluminación: valor de gris uniforme de la malla (\textit{flat} 0.7, "
                    r"\textit{darker} 0.3, \textit{brighter} 0.95). ")
META = {
    "axis_sym": {
        "label": "tab:axis_iluminacion_v04",
        "caption": (r"Efecto de la iluminación en simetría axial con el \textit{prompt} \texttt{v04}, "
                    r"resolución 224px."),
        "nota": (r"Nota: " + NOTA_ILUMINACION + r"Negrita: mejor valor de la columna. Las métricas "
                 r"angulares imputan $90^\circ$ a los objetos sin predicción; la traslación y "
                 r"SDE$_{\mathrm{ref}}$ usan solo los objetos con predicción. Objetos: cobertura, "
                 r"de 850."),
    },
    "plane_sym": {
        "label": "tab:plane_iluminacion_v05_1",
        "caption": (r"Efecto de la iluminación en simetría planar con el \textit{prompt} "
                    r"\texttt{v05\_1}, resolución 224px."),
        "nota": (r"Nota: " + NOTA_ILUMINACION + r"Se predice un plano por objeto. Negrita: mejor "
                 r"valor de la columna. Las métricas angulares imputan $90^\circ$ a los objetos sin "
                 r"predicción; SDE$_{\mathrm{ref}}$ y $F1_{\mathrm{ref}}$ usan solo los objetos con "
                 r"predicción. Objetos: cobertura, de 850."),
    },
}


def csv_mas_reciente(patron: str) -> Path:
    """ Devuelve el CSV más reciente bajo ndc_fix/ que coincide con un patrón glob.

    Args:
        * patron: patrón relativo a ndc_fix/ (p. ej. 'ndcfix_224_darker/*/axis_sym_...csv')

    Returns:
        * Path: ruta del CSV (el de nombre de carpeta experiments_<fecha> más reciente)

    """
    candidatos = list(NDC_FIX.glob(patron))
    if not candidatos:
        raise FileNotFoundError(f"no hay CSV para ndc_fix/{patron}")
    # experiments_DD_MM_AAAA: ordenar por fecha, no alfabéticamente
    return max(candidatos, key=lambda p: tuple(reversed(p.parent.name.split("_")[1:])))


def cargar(tipo: str) -> pd.DataFrame:
    """ Lee las tres iluminaciones de un tipo de simetría y las concatena.

    Args:
        * tipo: 'axis_sym' o 'plane_sym'

    Returns:
        * pd.DataFrame: una fila por (n_views, iluminación), con columna 'iluminacion'

    """
    experimento, patron_flat, nombre_csv = FUENTES[tipo]
    rutas = {"flat": csv_mas_reciente(patron_flat)}
    for ilum in ("darker", "brighter"):
        rutas[ilum] = csv_mas_reciente(f"ndcfix_224_{ilum}/*/{nombre_csv}")

    partes = []
    for ilum, ruta in rutas.items():
        d = pd.read_csv(ruta)
        d = d[(d["experiment"] == experimento) & (d["method"] == "triangulation")
              & d["n_views"].isin(VISTAS)].copy()
        if len(d) != len(VISTAS):
            raise ValueError(f"{ruta}: se esperaban {len(VISTAS)} filas de {experimento}, "
                             f"hay {len(d)}")
        d["iluminacion"] = ilum
        partes.append(d)
        print(f"  {tipo} {ilum:8s} <- {ruta.relative_to(REPO)}")
    d = pd.concat(partes, ignore_index=True)
    d["orden"] = d["iluminacion"].map(ILUMINACIONES.index)
    return d.sort_values(["n_views", "orden"]).reset_index(drop=True)


def tabla(d: pd.DataFrame, tipo: str) -> str:
    """ Construye la tabla LaTeX (lrbox + adjustbox + nota) de un tipo de simetría.

    Args:
        * d: DataFrame devuelto por cargar()
        * tipo: 'axis_sym' o 'plane_sym'

    Returns:
        * str: bloque \\begin{table}...\\end{table}

    """
    columnas = COLUMNAS[tipo]
    mejor = {c: (d[c].max() if mayor else d[c].min()) for c, _, mayor, _ in columnas}
    meta = META[tipo]
    lineas = [
        r"\begin{table}[htbp]", r"\centering", rf"\caption{{{meta['caption']}}}",
        rf"\label{{{meta['label']}}}", r"\begin{lrbox}{\tablebox}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{ll" + "r" * len(columnas) + "}", r"\toprule",
        " & ".join(["Vistas", "Iluminación"] + [h for _, h, _, _ in columnas]) + r" \\",
        r"\midrule",
    ]
    for k, n_views in enumerate(VISTAS):
        if k:
            lineas.append(r"\midrule")
        grupo = d[d["n_views"] == n_views]
        for j, (_, fila) in enumerate(grupo.iterrows()):
            celdas = []
            for c, _, _, fmt in columnas:
                texto = fmt.format(fila[c])
                celdas.append(rf"\textbf{{{texto}}}" if texto == fmt.format(mejor[c]) else texto)
            vistas = rf"\multirow{{{len(grupo)}}}{{*}}{{{n_views}}}" if j == 0 else ""
            lineas.append(" & ".join([vistas, rf"\textit{{{fila['iluminacion']}}}"] + celdas)
                          + r" \\")
    lineas += [
        r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{lrbox}",
        r"\usebox{\tablebox}", "", r"\vspace{1.5ex}", r"\begin{minipage}{\wd\tablebox}",
        r"\footnotesize", meta["nota"], r"\end{minipage}", r"\end{table}",
    ]
    return "\n".join(lineas)


def main() -> None:
    """ Genera las dos tablas de iluminación y las escribe en un único .tex. """
    parser = argparse.ArgumentParser(description="Tablas LaTeX del experimento de iluminación.")
    parser.add_argument("--salida", type=Path, default=SALIDA, help="archivo .tex de salida")
    args = parser.parse_args()

    tablas = [tabla(cargar(tipo), tipo) for tipo in FUENTES]
    cabecera = ("% Generado por utils/generar_tablas_iluminacion.py. "
                "Requiere booktabs, multirow, adjustbox y \\newsavebox{\\tablebox}.\n\n")
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(cabecera + "\n\n".join(tablas) + "\n", encoding="utf-8")
    print(f"Escrito {args.salida.relative_to(REPO) if args.salida.is_relative_to(REPO) else args.salida}")


if __name__ == "__main__":
    main()
