#!/usr/bin/env python3
# Genera las tablas con las que se eligen los prompts del Flujo A: el ranking por configuración
# (prompt y vistas) y el orden por prompt, promediando 6, 14 y 26 vistas. Usa solo 224px, flat y
# envío multiimagen, con EXP-E en eje y S_dat en plano. Escribe un .tex por tabla para incluirlo
# con \input. No modifica cap4.tex.
#
#   python utils/generar_tablas_seleccion.py
#   python utils/generar_tablas_seleccion.py --salida otra/carpeta
#
# Solo usa la biblioteca estándar. Requiere en el preámbulo: booktabs y float.
import argparse
from pathlib import Path

from generar_tablas_noviembre import (CRITERIOS, REPO, TITULO_TIPO, VISTAS, apilar, cargar,
                                      celdas_columna, escapar, ordenar_con_tolerancia)

SALIDA = REPO / "Tesis_DCC_MDS" / "tablas_seleccion"
CARPETAS_224 = ("ndcfix_flujoA", "ndcfix_v08")
TOP_CONFIGURACIONES = 3
PROMPTS_MOSTRADOS = 4
CORTO = {"axis_sym": "axis", "plane_sym": "plane"}
ETIQUETA_TOP = {"axis_sym": "tab:axis_top_224", "plane_sym": "tab:plane_top"}
CONDICIONES = r"resolución 224px, iluminación \textit{flat}, envío multiimagen"

_PLANO = {c[0]: c for c in CRITERIOS["plane_sym"]}
# El AUC va primero en plano porque imputa 90 grados a los objetos sin predicción; F1_ref no los
# cuenta, y al comparar prompts completos la cobertura de cada número de vistas debe pesar.
CRITERIOS_PROMPT = {
    "axis_sym": CRITERIOS["axis_sym"],
    "plane_sym": [_PLANO["auc_angular"], _PLANO["f1_ref"], _PLANO["precision_10deg"],
                  _PLANO["n_objects"], _PLANO["sde_ref_mean"], _PLANO["angular_error_mean"]],
}
# tipo -> (variante excluida antes de ordenar, prompt base que debe igualarla o superarla)
EXCLUIDOS = {"axis_sym": ("v04_1", "v04")}
# Métricas angulares con las que se comprueba la exclusión: (columna, mayor_es_mejor)
ANGULARES = {"axis_sym": [("auc_angular", True), ("angular_error_median", False),
                          ("precision_5deg", True), ("precision_10deg", True)]}


def cargar_224(tipo: str) -> dict[tuple[str, int], dict]:
    """ Lee los resultados del Flujo A a 224px, flat y multiimagen.

    Args:
        * tipo: 'axis_sym' o 'plane_sym'

    Returns:
        * dict[tuple[str, int], dict]: fila por (prompt, vistas)

    """
    datos = {}
    for carpeta in CARPETAS_224:
        datos.update(cargar(carpeta, tipo))
    return datos


def promedios_por_prompt(datos: dict[tuple[str, int], dict], columnas: list[str]) -> list[dict]:
    """ Promedia cada métrica de un prompt sobre sus tres cantidades de vistas.

    Args:
        * datos: fila por (prompt, vistas)
        * columnas: métricas que se promedian

    Returns:
        * list[dict]: una fila por prompt

    """
    filas = []
    for prompt in sorted({p for p, _ in datos}):
        fila = {"prompt": prompt}
        for columna in columnas:
            valores = [datos[(prompt, v)][columna] for v in VISTAS]
            if None in valores:
                raise ValueError(f"{prompt}: falta {columna} en alguna cantidad de vistas")
            fila[columna] = sum(valores) / len(valores)
        filas.append(fila)
    return filas


def comprobar_exclusion(datos: dict[tuple[str, int], dict], tipo: str) -> None:
    """ Comprueba que la variante excluida no supere a su base en ninguna métrica angular.

    Args:
        * datos: fila por (prompt, vistas)
        * tipo: 'axis_sym' o 'plane_sym'

    Raises:
        * ValueError: si la variante supera a la base, porque la exclusión dejaría de valer

    """
    variante, base = EXCLUIDOS[tipo]
    for columna, mayor in ANGULARES[tipo]:
        for vistas in VISTAS:
            propio, referencia = datos[(variante, vistas)][columna], datos[(base, vistas)][columna]
            if (propio > referencia) if mayor else (propio < referencia):
                raise ValueError(f"{variante} supera a {base} en {columna} con {vistas} vistas")


def envolver(encabezado: list[str], cuerpo: list[str], alineacion: str, titulo: str,
             etiqueta: str, nota: str) -> str:
    """ Envuelve una tabla con el formato de cap4.tex: tabular* al ancho de texto y nota al pie.

    Args:
        * encabezado: encabezados de columna
        * cuerpo: filas de datos
        * alineacion: columnas del tabular
        * titulo: título de la tabla
        * etiqueta: etiqueta completa
        * nota: texto de la nota, sin 'Nota:'

    Returns:
        * str: bloque \\begin{table}...\\end{table}

    """
    return "\n".join([
        r"\begin{table}[H]", r"\centering", rf"\caption{{{titulo}}}", rf"\label{{{etiqueta}}}",
        r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{0.95}",
        "", rf"\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}{alineacion}@{{}}}}",
        r"\toprule", " & ".join(encabezado) + r" \\", r"\midrule", *cuerpo, r"\bottomrule",
        r"\end{tabular*}", "", r"\vspace{0.5ex}", r"\begin{minipage}{\textwidth}",
        f"Nota: {nota}", r"\end{minipage}", r"\end{table}", "",
    ])


def tabla_configuraciones(datos: dict[tuple[str, int], dict], tipo: str) -> str:
    """ Ranking de las mejores combinaciones de prompt y número de vistas.

    Args:
        * datos: fila por (prompt, vistas)
        * tipo: 'axis_sym' o 'plane_sym'

    Returns:
        * str: tabla LaTeX

    """
    criterios = CRITERIOS[tipo]
    lista = [dict(valores, prompt=p, vistas=v) for (p, v), valores in datos.items()]
    orden, decisivo = ordenar_con_tolerancia(lista, criterios)
    orden, decisivo = orden[:TOP_CONFIGURACIONES], decisivo[:TOP_CONFIGURACIONES]
    celdas = {c[0]: celdas_columna([f.get(c[0]) for f in orden], c[0]) for c in criterios}
    cuerpo = [" & ".join([str(i + 1), rf"\texttt{{{escapar(f['prompt'])}}}", str(f["vistas"]),
                          str(decisivo[i])] + [celdas[c[0]][i] for c in criterios]) + r" \\"
              for i, f in enumerate(orden)]
    encabezado = [r"\#", "Prompt", "Vistas", "Crit."] + [apilar(c[1]) for c in criterios]
    return envolver(
        encabezado, cuerpo, "rlrc" + "r" * len(criterios),
        f"Ranking de las {TOP_CONFIGURACIONES} mejores configuraciones del Flujo A en "
        f"{TITULO_TIPO[tipo]}, {CONDICIONES}.",
        ETIQUETA_TOP[tipo],
        r"Crit.: número del criterio que resolvió la posición. Imputación y cobertura según la "
        r"Sección~\ref{sec:sm_cobertura}.")


def tabla_prompts(datos: dict[tuple[str, int], dict], tipo: str) -> tuple[str, list[str]]:
    """ Orden de los prompts según sus promedios sobre 6, 14 y 26 vistas.

    Args:
        * datos: fila por (prompt, vistas)
        * tipo: 'axis_sym' o 'plane_sym'

    Returns:
        * tuple[str, list[str]]: tabla LaTeX y prompts en orden

    """
    criterios = CRITERIOS_PROMPT[tipo]
    filas = promedios_por_prompt(datos, [c[0] for c in criterios])
    excluida = None
    if tipo in EXCLUIDOS:
        comprobar_exclusion(datos, tipo)
        excluida = next(f for f in filas if f["prompt"] == EXCLUIDOS[tipo][0])
        filas = [f for f in filas if f is not excluida]
    orden, decisivo = ordenar_con_tolerancia(filas, criterios)
    mostradas = orden[:PROMPTS_MOSTRADOS]
    if excluida is not None:
        mostradas = mostradas + [excluida]
    celdas = {c[0]: celdas_columna([f[c[0]] for f in mostradas], c[0]) for c in criterios}

    cuerpo = []
    for i, fila in enumerate(mostradas):
        es_excluida = fila is excluida
        inicio = ["--" if es_excluida else str(i + 1),
                  rf"\texttt{{{escapar(fila['prompt'])}}}",
                  "--" if es_excluida else str(decisivo[i])]
        cuerpo.append(" & ".join(inicio + [celdas[c[0]][i] for c in criterios]) + r" \\")

    nota = r"Crit.: criterio que resolvió la posición, según el orden de las columnas."
    if excluida is not None:
        nota += rf" \texttt{{{escapar(excluida['prompt'])}}} no participa en el orden."
    corto = CORTO[tipo]
    tabla = envolver(
        [r"\#", "Prompt", "Crit."] + [apilar(c[1]) for c in criterios], cuerpo,
        "rlc" + "r" * len(criterios),
        rf"Promedio por \textit{{prompt}} sobre 6, 14 y 26 vistas, en las mismas condiciones.",
        f"tab:{corto}_top_prompts", nota)
    return tabla, [f["prompt"] for f in orden]


def main() -> None:
    """ Escribe las cuatro tablas e imprime el orden de los prompts. """
    parser = argparse.ArgumentParser(description="Tablas de selección de prompts del Flujo A.")
    parser.add_argument("--salida", type=Path, default=SALIDA, help="carpeta de salida")
    args = parser.parse_args()

    args.salida.mkdir(parents=True, exist_ok=True)
    cabecera = "% Generado por utils/generar_tablas_seleccion.py. No editar a mano.\n"
    for tipo, corto in CORTO.items():
        datos = cargar_224(tipo)
        tabla, orden = tabla_prompts(datos, tipo)
        (args.salida / f"{corto}_top.tex").write_text(
            cabecera + tabla_configuraciones(datos, tipo), encoding="utf-8")
        (args.salida / f"{corto}_prompts.tex").write_text(cabecera + tabla, encoding="utf-8")
        print(f"{tipo}: {', '.join(orden[:PROMPTS_MOSTRADOS])}")
    print(f"Escrito en {args.salida}")


if __name__ == "__main__":
    main()
