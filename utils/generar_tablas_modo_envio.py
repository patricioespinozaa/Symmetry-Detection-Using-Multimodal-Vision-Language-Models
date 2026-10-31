#!/usr/bin/env python3
# Genera las tablas LaTeX que comparan el modo de envío de imágenes a Molmo2 (224px, flat):
# todas las vistas en una sola llamada (Flujo A, --prompt-mode auto) frente a una imagen por
# llamada (--prompt-mode single, corrida single_oct26), para axis_v04 y plane_v05_1.
#
# Genera dos juegos de tablas en docs/resultados/tablas_modo_envio/:
#   modo_envio_224.tex             resumen: una tabla por tipo con las métricas principales
#   axis_modo_envio.tex            completas, mismas tablas que generar_tablas_latex.py
#   plane_modo_envio.tex           (angular y posición; en plano también F1 y conteos)
#
# Uso:
#   python utils/generar_tablas_modo_envio.py
#   python utils/generar_tablas_modo_envio.py --salida otra/ruta.tex   # solo cambia el resumen
#
# Requiere en el preámbulo: booktabs, multirow, adjustbox y \newsavebox{\tablebox}.
import argparse
from pathlib import Path

import pandas as pd

from generar_tablas_iluminacion import COLUMNAS, REPO, VISTAS, csv_mas_reciente
from generar_tablas_latex import BLOQUES, NDC_FIX, PARTES, cargar as cargar_completo, tabla_bloque

SALIDA = REPO / "docs" / "resultados" / "tablas_modo_envio" / "modo_envio_224.tex"
CARPETAS_COMPLETAS = [NDC_FIX / "ndcfix_flujoA", NDC_FIX / "ndcfix_single_oct26"]
PARTE = "modo_envio"
PARTES[PARTE] = (r"\texttt{v04}/\texttt{v05\_1}: todas las vistas en una llamada; "
                 r"\texttt{\_single\_oct26}: una vista por llamada")
MODOS = ["Multiimagen", "Una por llamada"]

# tipo -> {modo: (patrón del CSV relativo a ndc_fix/, experimento)}
FUENTES = {
    "axis_sym": {
        "Multiimagen": ("ndcfix_flujoA/*/axis_sym_14exps_comparison.csv",
                        "axis_v04_ndcfix_nomesh"),
        "Una por llamada": ("ndcfix_single_oct26/*/axis_sym_axis_v04_single_oct26_ndcfix_nomesh_comparison.csv",
                            "axis_v04_single_oct26_ndcfix_nomesh"),
    },
    "plane_sym": {
        "Multiimagen": ("ndcfix_flujoA/*/plane_sym_14exps_comparison.csv",
                        "plane_v05_1_ndcfix_mp1_nomesh"),
        "Una por llamada": ("ndcfix_single_oct26/*/plane_sym_plane_v05_1_single_oct26_ndcfix_mp1_nomesh_comparison.csv",
                            "plane_v05_1_single_oct26_ndcfix_mp1_nomesh"),
    },
}

NOTA_MODO = (r"Multiimagen: las $n_v$ vistas se envían a Molmo2 en una sola llamada, con el "
             r"\textit{prompt} multivista. Una por llamada: cada vista se envía por separado, con el "
             r"\textit{prompt} de imagen única, y los puntos se reúnen antes de la triangulación. ")
META = {
    "axis_sym": {
        "label": "tab:axis_modo_envio_v04",
        "caption": (r"Envío de todas las vistas en una llamada frente a una vista por llamada, "
                    r"simetría axial, \textit{prompt} \texttt{v04}, resolución 224px."),
        "nota": (r"Nota: " + NOTA_MODO + r"Negrita: mejor valor de la columna. Las métricas "
                 r"angulares imputan $90^\circ$ a los objetos sin predicción; la traslación y "
                 r"SDE$_{\mathrm{ref}}$ usan solo los objetos con predicción. Objetos: cobertura, "
                 r"de 850."),
    },
    "plane_sym": {
        "label": "tab:plane_modo_envio_v05_1",
        "caption": (r"Envío de todas las vistas en una llamada frente a una vista por llamada, "
                    r"simetría planar, \textit{prompt} \texttt{v05\_1}, resolución 224px."),
        "nota": (r"Nota: " + NOTA_MODO + r"Se predice un plano por objeto. Negrita: mejor valor "
                 r"de la columna. Las métricas angulares imputan $90^\circ$ a los objetos sin "
                 r"predicción; SDE$_{\mathrm{ref}}$ y $F1_{\mathrm{ref}}$ usan solo los objetos con "
                 r"predicción. Objetos: cobertura, de 850."),
    },
}


def cargar(tipo: str) -> pd.DataFrame:
    """ Lee los dos modos de envío de un tipo de simetría y los concatena.

    Args:
        * tipo: 'axis_sym' o 'plane_sym'

    Returns:
        * pd.DataFrame: una fila por (n_views, modo), con columna 'modo'

    """
    partes = []
    for modo, (patron, experimento) in FUENTES[tipo].items():
        ruta = csv_mas_reciente(patron)
        d = pd.read_csv(ruta)
        d = d[(d["experiment"] == experimento) & (d["method"] == "triangulation")
              & d["n_views"].isin(VISTAS)].copy()
        if len(d) != len(VISTAS):
            raise ValueError(f"{ruta}: se esperaban {len(VISTAS)} filas de {experimento}, "
                             f"hay {len(d)}")
        d["modo"] = modo
        partes.append(d)
        print(f"  {tipo} {modo:16s} <- {ruta.relative_to(REPO)}")
    d = pd.concat(partes, ignore_index=True)
    d["orden"] = d["modo"].map(MODOS.index)
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
        " & ".join(["Vistas", "Envío"] + [h for _, h, _, _ in columnas]) + r" \\",
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
            lineas.append(" & ".join([vistas, fila["modo"]] + celdas) + r" \\")
    lineas += [
        r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{lrbox}",
        r"\usebox{\tablebox}", "", r"\vspace{1.5ex}", r"\begin{minipage}{\wd\tablebox}",
        r"\footnotesize", meta["nota"], r"\end{minipage}", r"\end{table}",
    ]
    return "\n".join(lineas)


def tablas_completas(tipo: str) -> str:
    """ Genera todas las tablas por bloque (como generar_tablas_latex.py) para los dos modos.

    Args:
        * tipo: 'axis_sym' o 'plane_sym'

    Returns:
        * str: tablas LaTeX concatenadas (angular y posición; en plano, también F1 y conteos)

    """
    experimentos = [exp for _, exp in FUENTES[tipo].values()]
    d = cargar_completo(CARPETAS_COMPLETAS, tipo, "triangulation")
    d = d[d["experiment"].isin(experimentos) & d["n_views"].isin(VISTAS)]
    if d["experiment"].nunique() != len(experimentos):
        raise ValueError(f"{tipo}: faltan experimentos; encontrados {sorted(d['experiment'].unique())}")
    tablas = [tabla_bloque(d, tipo, PARTE, b) for b, (_, tipos, _) in BLOQUES.items() if tipo in tipos]
    return "\n".join(t for t in tablas if t)


def main() -> None:
    """ Genera las tablas de resumen y las tablas completas de modo de envío. """
    parser = argparse.ArgumentParser(description="Tablas LaTeX: multiimagen vs. una por llamada.")
    parser.add_argument("--salida", type=Path, default=SALIDA, help="archivo .tex del resumen")
    args = parser.parse_args()

    cabecera = ("% Generado por utils/generar_tablas_modo_envio.py. "
                "Requiere booktabs, multirow, adjustbox y \\newsavebox{\\tablebox}.\n\n")
    tablas = [tabla(cargar(tipo), tipo) for tipo in FUENTES]
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(cabecera + "\n\n".join(tablas) + "\n", encoding="utf-8")
    print(f"Escrito {args.salida}")

    for tipo in FUENTES:
        ruta = SALIDA.parent / f"{tipo.split('_')[0]}_{PARTE}.tex"
        ruta.write_text(cabecera + tablas_completas(tipo), encoding="utf-8")
        print(f"Escrito {ruta}")


if __name__ == "__main__":
    main()
