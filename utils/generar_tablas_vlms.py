#!/usr/bin/env python3
# Genera las tablas LaTeX que comparan Molmo2-8B, MolmoPoint-8B y Qwen3-VL-8B con una vista
# por llamada (--prompt-mode single): axis_v04 en simetría axial y plane_v05_1 en planar.
#
# Uso:
#   python utils/generar_tablas_vlms.py                      # escribe el .tex por defecto
#   python utils/generar_tablas_vlms.py --salida otra/ruta.tex
#
# Los CSV no guardan el modo de envío, la resolución ni la iluminación, así que se comprueban en
# los logs de inferencia de cada modelo (logs_single_oct26/ y logs/) antes de generar las tablas.
# Solo usa la biblioteca estándar. Requiere en el preámbulo: booktabs, multirow, adjustbox y
# \newsavebox{\tablebox}.
import argparse
import csv
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SALIDA = REPO / "docs" / "resultados" / "tablas_vlms" / "vlms_single_224.tex"
VISTAS = [6, 14, 26]
OBJETOS_TOTALES = 850
METODO = "triangulation"

# tipo -> [(modelo, patrón del CSV relativo a la raíz del repo, experimento)]
FUENTES = {
    "axis_sym": [
        ("Molmo2-8B", "ndc_fix/ndcfix_single_oct26/*/"
                      "axis_sym_axis_v04_single_oct26_ndcfix_nomesh_comparison.csv",
         "axis_v04_single_oct26_ndcfix_nomesh"),
        ("MolmoPoint-8B", "MolmoPointPointing/experiments_*/axis_sym_molmopoint_comparison.csv",
         "molmopoint"),
        ("Qwen3-VL-8B", "QwenPointing/experiments_*/axis_sym_qwen3vl_comparison.csv", "qwen3vl"),
    ],
    "plane_sym": [
        ("Molmo2-8B", "ndc_fix/ndcfix_single_oct26/*/"
                      "plane_sym_plane_v05_1_single_oct26_ndcfix_mp1_nomesh_comparison.csv",
         "plane_v05_1_single_oct26_ndcfix_mp1_nomesh"),
        ("MolmoPoint-8B", "MolmoPointPointing/experiments_*/plane_sym_molmopoint_comparison.csv",
         "molmopoint"),
        ("Qwen3-VL-8B", "QwenPointing/experiments_*/plane_sym_qwen3vl_comparison.csv", "qwen3vl"),
    ],
}
# tipo -> {modelo: (log de inferencia relativo a la raíz del repo, Prompt ID esperado)}
LOGS = {
    "axis_sym": {
        "Molmo2-8B": ("logs_single_oct26/molmo_axis_v04_single_oct26.log", "axis_v04"),
        "MolmoPoint-8B": ("logs/molmopoint_axis.log", "axis_v04_molmopoint"),
        "Qwen3-VL-8B": ("logs/qwen3vl_axis.log", "axis_v04_qwen3vl"),
    },
    "plane_sym": {
        "Molmo2-8B": ("logs_single_oct26/molmo_plane_v05_1_single_oct26.log", "plane_v05_1"),
        "MolmoPoint-8B": ("logs/molmopoint_plane.log", "plane_v05_1_molmopoint"),
        "Qwen3-VL-8B": ("logs/qwen3vl_plane.log", "plane_v05_1_qwen3vl"),
    },
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

NOTA_ENVIO = (r"Cada vista se envía en una llamada aparte y los puntos se reúnen antes de la "
              r"triangulación. MolmoPoint hace dos llamadas por vista, una por punto, con una "
              r"versión abreviada del \textit{prompt}; Qwen3-VL usa el mismo \textit{prompt}, "
              r"adaptado a su formato de salida JSON. ")
META = {
    "axis_sym": {
        "label": "tab:axis_vlms_v04",
        "caption": (r"Comparación de VLMs en simetría axial con el \textit{prompt} \texttt{v04}, "
                    r"una vista por llamada, resolución 224px e iluminación \textit{flat}."),
        "nota": (r"Nota: " + NOTA_ENVIO + r"Negrita: mejor valor de la columna. Las métricas "
                 r"angulares imputan $90^\circ$ a los objetos sin predicción; la traslación y "
                 r"SDE$_{\mathrm{ref}}$ usan solo los objetos con predicción. Objetos: cobertura, "
                 r"de 850."),
    },
    "plane_sym": {
        "label": "tab:plane_vlms_v05_1",
        "caption": (r"Comparación de VLMs en simetría planar con el \textit{prompt} "
                    r"\texttt{v05\_1}, una vista por llamada, resolución 224px e iluminación "
                    r"\textit{flat}."),
        "nota": (r"Nota: " + NOTA_ENVIO + r"Se predice un plano por objeto. Negrita: mejor valor "
                 r"de la columna. Las métricas angulares imputan $90^\circ$ a los objetos sin "
                 r"predicción; SDE$_{\mathrm{ref}}$ y $F1_{\mathrm{ref}}$ usan solo los objetos "
                 r"con predicción. Objetos: cobertura, de 850."),
    },
}


def verificar_log(tipo: str, modelo: str) -> None:
    """ Comprueba que cada arranque de la inferencia fue single, 224px, flat e incluyó 6/14/26.

    Un log puede tener varios encabezados si la corrida se reanudó; se revisan todos.

    Args:
        * tipo: 'axis_sym' o 'plane_sym'
        * modelo: nombre del modelo en LOGS

    """
    relativa, prompt_id = LOGS[tipo][modelo]
    ruta = REPO / relativa
    texto = ruta.read_text(encoding="utf-8", errors="replace")

    def valores(clave: str) -> list[str]:
        return re.findall(rf"^{clave}\s*:\s*(.+?)\s*$", texto, re.MULTILINE)

    esperado = {"Symmetry type": tipo, "Prompt mode": "single", "Prompt ID": prompt_id,
                "Sizes": "[224]", "Lightings": "['flat']"}
    for clave, valor in esperado.items():
        hallados = valores(clave)
        if not hallados or set(hallados) != {valor}:
            raise ValueError(f"{ruta}: se esperaba {clave} = {valor}, hay {sorted(set(hallados))}")
    grupos = valores("View groups")
    if not grupos or not all(set(VISTAS) <= set(map(int, re.findall(r"\d+", g))) for g in grupos):
        raise ValueError(f"{ruta}: View groups {grupos} no incluye {VISTAS}")
    print(f"  {tipo} {modelo}: verificado en {relativa} ({len(grupos)} arranque(s))")


def cargar_csv(patron: str, experimento: str) -> dict[int, dict[str, str]]:
    """ Lee las filas de un experimento desde el único CSV que coincide con un patrón.

    Args:
        * patron: patrón glob relativo a la raíz del repo
        * experimento: valor de la columna 'experiment'

    Returns:
        * dict[int, dict[str, str]]: fila del CSV por número de vistas

    """
    rutas = sorted(REPO.glob(patron))
    if len(rutas) != 1:
        raise ValueError(f"se esperaba un CSV para {patron}, hay {len(rutas)}")
    with open(rutas[0], encoding="utf-8", newline="") as archivo:
        filas = [f for f in csv.DictReader(archivo)
                 if f["experiment"] == experimento and f["method"] == METODO]
    por_vistas = {int(f["n_views"]): f for f in filas}
    if sorted(por_vistas) != VISTAS or len(filas) != len(VISTAS):
        raise ValueError(f"{rutas[0]}: vistas {sorted(por_vistas)}, se esperaban {VISTAS}")
    for fila in filas:
        if int(fila["n_total"]) != OBJETOS_TOTALES:
            raise ValueError(f"{rutas[0]}: n_total = {fila['n_total']}")
    print(f"  {experimento:45s} <- {rutas[0].relative_to(REPO)}")
    return por_vistas


def tabla(tipo: str) -> str:
    """ Construye la tabla LaTeX (lrbox + adjustbox + nota) de un tipo de simetría.

    Args:
        * tipo: 'axis_sym' o 'plane_sym'

    Returns:
        * str: bloque \\begin{table}...\\end{table}

    """
    columnas = COLUMNAS[tipo]
    datos = [(modelo, cargar_csv(patron, exp)) for modelo, patron, exp in FUENTES[tipo]]
    mejor = {}
    for col, _, mayor, _ in columnas:
        valores = [float(filas[n][col]) for _, filas in datos for n in VISTAS]
        mejor[col] = max(valores) if mayor else min(valores)

    meta = META[tipo]
    lineas = [
        r"\begin{table}[htbp]", r"\centering", rf"\caption{{{meta['caption']}}}",
        rf"\label{{{meta['label']}}}", r"\begin{lrbox}{\tablebox}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{ll" + "r" * len(columnas) + "}", r"\toprule",
        " & ".join(["Vistas", "Modelo"] + [h for _, h, _, _ in columnas]) + r" \\",
        r"\midrule",
    ]
    for k, n_views in enumerate(VISTAS):
        if k:
            lineas.append(r"\midrule")
        for j, (modelo, filas) in enumerate(datos):
            celdas = []
            for col, _, _, fmt in columnas:
                texto = fmt.format(float(filas[n_views][col]))
                celdas.append(rf"\textbf{{{texto}}}" if texto == fmt.format(mejor[col]) else texto)
            vistas = rf"\multirow{{{len(datos)}}}{{*}}{{{n_views}}}" if j == 0 else ""
            lineas.append(" & ".join([vistas, modelo] + celdas) + r" \\")
    lineas += [
        r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{lrbox}",
        r"\usebox{\tablebox}", "", r"\vspace{1.5ex}", r"\begin{minipage}{\wd\tablebox}",
        r"\footnotesize", meta["nota"], r"\end{minipage}", r"\end{table}",
    ]
    return "\n".join(lineas)


def main() -> None:
    """ Verifica las corridas de los tres modelos y escribe las dos tablas en un único .tex. """
    parser = argparse.ArgumentParser(description="Tablas LaTeX: VLMs en modo single.")
    parser.add_argument("--salida", type=Path, default=SALIDA, help="archivo .tex de salida")
    args = parser.parse_args()

    for tipo, modelos in LOGS.items():
        for modelo in modelos:
            verificar_log(tipo, modelo)
    tablas = [tabla(tipo) for tipo in FUENTES]
    cabecera = ("% Generado por utils/generar_tablas_vlms.py. "
                "Requiere booktabs, multirow, adjustbox y \\newsavebox{\\tablebox}.\n\n")
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(cabecera + "\n\n".join(tablas) + "\n", encoding="utf-8")
    print(f"Escrito {args.salida}")


if __name__ == "__main__":
    main()
