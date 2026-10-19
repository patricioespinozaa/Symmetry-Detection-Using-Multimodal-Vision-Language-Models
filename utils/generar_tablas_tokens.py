#!/usr/bin/env python3
# Genera las tablas LaTeX del análisis de largo de contexto de Molmo2 a partir de los CSV de
# MolmoPointing/ContextLimits/contar_tokens_molmo2.py (ver contexto_tokens_molmo2.md):
#   1. Tokens de entrada por llamada (modo procesador) para axis_v04 y plane_v05_1.
#   2. Tokens reales e indicadores de saturación (modo json) junto al desempeño del pipeline
#      sin malla (CSV de comparación de ndc_fix/), por resolución y número de vistas.
#   3. Fracción de salidas en formato <tracks> por prompt (224px, flat).
#
# Uso:
#   python utils/generar_tablas_tokens.py
#   python utils/generar_tablas_tokens.py --resultados otra/carpeta --salida otra/ruta.tex
#
# Requiere en el preámbulo: booktabs, multirow, adjustbox y \newsavebox{\tablebox}.
import argparse
from pathlib import Path

import pandas as pd

from generar_tablas_iluminacion import csv_mas_reciente

REPO = Path(__file__).resolve().parents[1]
RESULTADOS = REPO / "MolmoPointing" / "ContextLimits" / "resultados"
SALIDA = REPO / "docs" / "resultados" / "tablas_tokens" / "tokens_molmo2.tex"

LARGO_SFT = 16_384            # largo de secuencia del fine-tuning principal de Molmo2
CONTEXTO_MAX = 36_864         # max_position_embeddings de Molmo2-8B
RESOLUCIONES = [224, 448, 1136]
VISTAS_EVAL = [6, 14, 26]

# tipo -> (prompt, experimento de Molmo2 en saturacion_corridas.csv, experimento evaluado)
PROMPTS = {
    "axis_sym": ("axis_v04", "axis_v04", "axis_v04_ndcfix_nomesh"),
    "plane_sym": ("plane_v05_1", "plane_v05_1", "plane_v05_1_ndcfix_mp1_nomesh"),
}
# resolución -> patrón (relativo a ndc_fix/) de los CSV de comparación, con {tipo}
FUENTES_EVAL = {
    224: "ndcfix_flujoA/*/{tipo}_14exps_comparison.csv",
    448: "ndcfix_448/*/{tipo}_*comparison.csv",
    1136: "ndcfix_1136/*/{tipo}_*comparison.csv",
}
NOMBRE_TIPO = {"axis_sym": "axial", "plane_sym": "planar"}
# ¿El CSV distingue <tracks>? -> (encabezado, descripción) de la columna de formato de salida.
# Un saturacion_corridas.csv de la versión anterior de contar_tokens_molmo2.py solo sabe que
# la salida no es <points>.
FORMATO = {
    True: (r"\texttt{<tracks>}",
           r"respondió con su formato de \textit{tracking} de video (\texttt{<tracks coords=...>}) "
           r"en lugar de \texttt{<points coords=...>}"),
    False: (r"sin \texttt{<points>}",
            r"respondió sin la etiqueta \texttt{<points coords=...>} (en las salidas "
            r"inspeccionadas, por usar \texttt{<tracks coords=...>}, el formato de "
            r"\textit{tracking} de video de Molmo2)"),
}

MARCA_SFT = r"$^{\dagger}$"
MARCA_CTX = r"$^{\ddagger}$"
NOTA_MARCAS = (rf"{MARCA_SFT}: supera el largo de secuencia del \textit{{fine-tuning}} principal "
               rf"de Molmo2 (16\,384 tokens). {MARCA_CTX}: supera además su contexto máximo "
               r"(36\,864 tokens). ")


def miles(n: float) -> str:
    """ Formatea un entero con espacio fino como separador de miles (16\\,384). """
    return f"{int(round(n)):,}".replace(",", r"\,")


def tokens_marcados(n: float) -> str:
    """ Formatea un número de tokens con la marca de largo de entrenamiento o contexto superado.

    Args:
        * n: tokens de entrada de la llamada

    Returns:
        * str: número formateado, con dagger si supera LARGO_SFT y doble dagger si supera CONTEXTO_MAX

    """
    marca = MARCA_CTX if n > CONTEXTO_MAX else MARCA_SFT if n > LARGO_SFT else ""
    return miles(n) + marca


def pct(x: float) -> str:
    """ Formatea una fracción como porcentaje con un decimal, o '--' si falta. """
    return "--" if pd.isna(x) else f"{100 * x:.1f}"


def envolver(caption: str, label: str, columnas: str, cuerpo: list[str], nota: str) -> str:
    """ Envuelve filas de tabular en el mismo entorno que el resto de tablas de la tesis.

    Args:
        * caption: texto del caption
        * label: etiqueta LaTeX
        * columnas: especificación de columnas del tabular (p. ej. 'lrrr')
        * cuerpo: líneas entre \\begin{tabular} y \\end{tabular} (encabezado incluido)
        * nota: nota al pie de la tabla

    Returns:
        * str: bloque \\begin{table}...\\end{table}

    """
    return "\n".join([
        r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\begin{lrbox}{\tablebox}", r"\begin{adjustbox}{max width=\textwidth}",
        rf"\begin{{tabular}}{{{columnas}}}", *cuerpo, r"\end{tabular}", r"\end{adjustbox}",
        r"\end{lrbox}", r"\usebox{\tablebox}", "", r"\vspace{1.5ex}",
        r"\begin{minipage}{\wd\tablebox}", r"\footnotesize", nota, r"\end{minipage}",
        r"\end{table}",
    ])


def cargar_procesador(resultados: Path) -> pd.DataFrame:
    """ Lee los CSV del modo procesador de los dos prompts y los concatena.

    Args:
        * resultados: carpeta con tokens_<prompt>.csv

    Returns:
        * pd.DataFrame: una fila por (prompt_id, resolución, vistas)

    """
    partes = []
    for prompt, _, _ in PROMPTS.values():
        ruta = resultados / f"tokens_{prompt}.csv"
        d = pd.read_csv(ruta)
        d = d[d["max_crops"].astype(str) == "defecto"]
        partes.append(d)
        print(f"  procesador {prompt:12s} <- {ruta.relative_to(REPO)}")
    return pd.concat(partes, ignore_index=True)


def cargar_saturacion(resultados: Path) -> pd.DataFrame:
    """ Lee saturacion_corridas.csv, sin las entradas duplicadas por enlaces simbólicos.

    Args:
        * resultados: carpeta con saturacion_corridas.csv

    Returns:
        * pd.DataFrame: una fila por (tipo, experimento, size, lighting, n_views), con la columna
          'frac_tracks' (formato <tracks>) ya resuelta

    """
    ruta = resultados / "saturacion_corridas.csv"
    d = pd.read_csv(ruta)
    print(f"  json                    <- {ruta.relative_to(REPO)}")
    # Los <X>_nomesh / <X>_ndcfix_nomesh son enlaces al molmo_multiview_<X>.json original
    # (utils/rerun_ndc_fix.sh, rerun_ndcfix_completo.py): mismas llamadas a Molmo2.
    d = d[~d["experimento"].str.contains("_nomesh")].copy()
    d.attrs["tracks"] = "frac_formato_tracks" in d
    if d.attrs["tracks"]:
        d["frac_tracks"] = d["frac_formato_tracks"]
    else:
        # CSV de la versión anterior del script: solo distinguía "sin <points". Como el parser
        # del runner sí extrajo puntos de esas salidas, corresponden a otra etiqueta (<tracks>).
        print("  [aviso] CSV sin frac_formato_tracks: se usa frac_salidas_sin_bloque "
              "(salidas sin <points>). Re-correr contar_tokens_molmo2.py --modo json.")
        d["frac_tracks"] = d["frac_salidas_sin_bloque"]
    return d


def cargar_eval(tipo: str) -> pd.DataFrame:
    """ Lee el desempeño del pipeline sin malla del prompt de un tipo en las tres resoluciones.

    Args:
        * tipo: 'axis_sym' o 'plane_sym'

    Returns:
        * pd.DataFrame: columnas size, n_views, auc_angular, n_objects (iluminación flat)

    """
    _, _, experimento = PROMPTS[tipo]
    partes = []
    for size, patron in FUENTES_EVAL.items():
        ruta = csv_mas_reciente(patron.format(tipo=tipo))
        d = pd.read_csv(ruta)
        d = d[(d["experiment"] == experimento) & (d["method"] == "triangulation")
              & d["n_views"].isin(VISTAS_EVAL)]
        if len(d) != len(VISTAS_EVAL):
            raise ValueError(f"{ruta}: se esperaban {len(VISTAS_EVAL)} filas de {experimento}, "
                             f"hay {len(d)}")
        partes.append(d[["n_views", "auc_angular", "n_objects"]].assign(size=size))
        print(f"  eval {tipo} {size:>4}px    <- {ruta.relative_to(REPO)}")
    return pd.concat(partes, ignore_index=True)


def verificar(proc: pd.DataFrame, sat: pd.DataFrame) -> None:
    """ Avisa si los tokens del procesador no coinciden con los registrados en las corridas.

    Args:
        * proc: DataFrame de cargar_procesador()
        * sat: DataFrame de cargar_saturacion()

    """
    reales = sat[sat["flow"] == "a"][["experimento", "size", "n_views", "tokens_media"]]
    m = proc.merge(reales, left_on=["prompt_id", "resolucion", "vistas"],
                   right_on=["experimento", "size", "n_views"])
    distintas = m[m["tokens_total"] != m["tokens_media"]]
    print(f"  verificación procesador vs. corridas: {len(m) - len(distintas)}/{len(m)} coinciden")
    for _, f in distintas.iterrows():
        print(f"    [aviso] {f['prompt_id']} {f['resolucion']}px {f['vistas']} vistas: "
              f"procesador {f['tokens_total']} vs. corridas {f['tokens_media']}")


def tabla_procesador(proc: pd.DataFrame) -> str:
    """ Tabla 1: tokens de entrada por llamada según vistas y resolución, para ambos prompts.

    Args:
        * proc: DataFrame de cargar_procesador()

    Returns:
        * str: bloque \\begin{table}...\\end{table}

    """
    prompts = [p for p, _, _ in PROMPTS.values()]
    encabezado = [
        r"\toprule",
        r"\multirow{2}{*}{Vistas} & "
        + " & ".join(rf"\multicolumn{{3}}{{c}}{{\texttt{{{p.replace('_', chr(92) + '_')}}}}}"
                     for p in prompts) + r" \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        " & " + " & ".join(f"{r} px" for _ in prompts for r in RESOLUCIONES) + r" \\",
        r"\midrule",
    ]
    filas = []
    for nv in sorted(proc["vistas"].unique()):
        celdas = []
        for p in prompts:
            for r in RESOLUCIONES:
                fila = proc[(proc["prompt_id"] == p) & (proc["resolucion"] == r)
                            & (proc["vistas"] == nv)]
                celdas.append(tokens_marcados(fila["tokens_total"].iloc[0]) if len(fila) else "--")
        filas.append(" & ".join([str(nv)] + celdas) + r" \\")
    por_imagen = proc.groupby("resolucion")["tokens_por_imagen"].first()
    texto = proc.groupby("prompt_id")["tokens_texto"].agg(["min", "max"])
    nota = (r"Nota: tokens de entrada de una llamada a Molmo2 con todas las vistas juntas, "
            r"contados con su procesador (\texttt{contar\_tokens\_molmo2.py}, "
            r"\texttt{max\_crops} por defecto = 8). Tokens de imagen por vista: "
            + ", ".join(f"{miles(por_imagen[r])} a {r} px" for r in RESOLUCIONES)
            + r"; a 448 y 1136 px el procesador produce el mismo número de recortes, por lo que "
              r"ambas resoluciones dan exactamente los mismos tokens. El texto aporta "
            + " y ".join(rf"{miles(texto.loc[p, 'min'])}--{miles(texto.loc[p, 'max'])} "
                         rf"(\texttt{{{p.replace('_', chr(92) + '_')}}})" for p in prompts)
            + r" tokens. " + NOTA_MARCAS)
    return envolver(
        r"Largo de la entrada de Molmo2 según el número de vistas y la resolución.",
        "tab:tokens_procesador", "r" + "r" * 3 * len(prompts),
        encabezado + filas + [r"\bottomrule"], nota)


def tabla_saturacion(tipo: str, sat: pd.DataFrame, ev: pd.DataFrame) -> str:
    """ Tabla 2: tokens reales, indicadores de saturación y desempeño de un tipo de simetría.

    Args:
        * tipo: 'axis_sym' o 'plane_sym'
        * sat: DataFrame de cargar_saturacion()
        * ev: DataFrame de cargar_eval(tipo)

    Returns:
        * str: bloque \\begin{table}...\\end{table}

    """
    prompt, exp_molmo, _ = PROMPTS[tipo]
    s = sat[(sat["experimento"] == exp_molmo) & (sat["lighting"] == "flat")
            & sat["n_views"].isin(VISTAS_EVAL)]
    d = ev.merge(s, on=["size", "n_views"], how="left").sort_values(["size", "n_views"])

    etiqueta, descripcion = FORMATO[sat.attrs["tracks"]]
    columnas = [
        ("Tokens", lambda f: "--" if pd.isna(f["tokens_media"]) else tokens_marcados(f["tokens_media"])),
        (r"\begin{tabular}[b]{@{}c@{}}Formato \\ " + etiqueta + r" (\%)\end{tabular}",
         lambda f: pct(f["frac_tracks"])),
        (r"\begin{tabular}[b]{@{}c@{}}Vistas \\ útiles (\%)\end{tabular}",
         lambda f: pct(f["frac_vistas_utiles"])),
        (r"\begin{tabular}[b]{@{}c@{}}Puntos \\ duplic. (\%)\end{tabular}",
         lambda f: pct(f["frac_puntos_duplicados"])),
        ("Objetos", lambda f: f"{f['n_objects']:.0f}"),
        (r"AUC$_{45^\circ}$", lambda f: f"{f['auc_angular']:.3f}"),
    ]
    cuerpo = [r"\toprule",
              " & ".join(["Resolución", "Vistas"] + [h for h, _ in columnas]) + r" \\",
              r"\midrule"]
    for k, size in enumerate(RESOLUCIONES):
        if k:
            cuerpo.append(r"\midrule")
        grupo = d[d["size"] == size]
        for j, (_, f) in enumerate(grupo.iterrows()):
            res = rf"\multirow{{{len(grupo)}}}{{*}}{{{size} px}}" if j == 0 else ""
            cuerpo.append(" & ".join([res, str(f["n_views"])] + [fn(f) for _, fn in columnas])
                          + r" \\")
    cuerpo.append(r"\bottomrule")

    faltan = d[d["tokens_media"].isna()]
    nota_faltan = ""
    if len(faltan):
        combos = ", ".join(f"{f['size']} px con {f['n_views']} vistas" for _, f in faltan.iterrows())
        nota_faltan = (rf"--: sin registro de llamadas a Molmo2 en los \texttt{{molmo\_multiview}} "
                       rf"analizados ({combos}). ")
    nota = (rf"Nota: \textit{{prompt}} \texttt{{{prompt.replace('_', chr(92) + '_')}}}, Flujo A, "
            r"iluminación \textit{flat}, 850 objetos; una llamada a Molmo2 por objeto con todas "
            r"las vistas. Tokens: largo real de la entrada registrado por el \textit{runner} "
            r"(igual en todos los objetos). Formato " + etiqueta + r": fracción de llamadas en "
            r"que Molmo2 " + descripcion + r". Vistas útiles: vistas con al menos 2 puntos, mínimo que exige la "
            r"triangulación. Puntos duplic.: puntos con la misma $(x, y)$ que otro de la misma "
            r"vista. Objetos y AUC$_{45^\circ}$: \textit{pipeline} sin malla (triangulación"
            + (", un plano por objeto" if tipo == "plane_sym" else "")
            + r"); objetos con predicción, de 850; el AUC imputa $90^\circ$ a los objetos sin "
              r"predicción. " + NOTA_MARCAS + nota_faltan)
    return envolver(
        rf"Largo de la entrada, indicadores de saturación de Molmo2 y desempeño en simetría "
        rf"{NOMBRE_TIPO[tipo]}.",
        f"tab:tokens_saturacion_{tipo.split('_')[0]}", "ll" + "r" * len(columnas), cuerpo, nota)


def tabla_tracks(sat: pd.DataFrame) -> str:
    """ Tabla 3: porcentaje de salidas <tracks> por prompt y número de vistas (224 px, flat).

    Args:
        * sat: DataFrame de cargar_saturacion()

    Returns:
        * str: bloque \\begin{table}...\\end{table}

    """
    etiqueta, descripcion = FORMATO[sat.attrs["tracks"]]
    d = sat[(sat["size"] == 224) & (sat["lighting"] == "flat")
            & (sat["n_llamadas"] == sat["n_llamadas"].max())]
    vistas = sorted(d["n_views"].unique())
    piv = d.pivot_table(index=["symmetry_type", "experimento"], columns="n_views",
                        values="frac_tracks")
    tokens = d.groupby("n_views")["tokens_media"].median()
    cuerpo = [r"\toprule",
              " & ".join(["Tipo", r"\textit{Prompt}"] + [f"{v} vista" + ("s" if v > 1 else "") for v in vistas]) + r" \\",
              " & ".join(["", r"\textit{Tokens (mediana)}"]
                         + [rf"\textit{{{miles(tokens[v])}}}" for v in vistas]) + r" \\",
              r"\midrule"]
    for k, tipo in enumerate(PROMPTS):
        if k:
            cuerpo.append(r"\midrule")
        grupo = piv.loc[tipo]
        for j, (exp, fila) in enumerate(grupo.iterrows()):
            nombre = exp.removeprefix(tipo.split("_")[0] + "_").replace("_", r"\_")
            t = rf"\multirow{{{len(grupo)}}}{{*}}{{{NOMBRE_TIPO[tipo].capitalize()}}}" if j == 0 else ""
            cuerpo.append(" & ".join([t, rf"\texttt{{{nombre}}}"] + [pct(fila.get(v)) for v in vistas])
                          + r" \\")
    cuerpo.append(r"\bottomrule")
    nota = (r"Nota: porcentaje de las 850 llamadas en que Molmo2 " + descripcion
            + r"; resolución 224 px, iluminación \textit{flat}, "
            r"\texttt{--prompt-mode auto} (1 vista: \textit{prompt} de imagen única). El "
            r"\textit{parser} del \textit{runner} extrae los puntos de ambos formatos. --: "
            r"combinación no ejecutada para ese \textit{prompt}. Los sufijos \texttt{flowB}/"
            r"\texttt{flowC} usan los Flujos B y C.")
    return envolver(
        r"Llamadas a Molmo2 con salida " + etiqueta + r" según el número de vistas, por "
        r"\textit{prompt}.",
        "tab:tokens_formato_tracks", "ll" + "r" * len(vistas), cuerpo, nota)


def main() -> None:
    """ Genera las tablas del análisis de tokens y las escribe en un único .tex. """
    parser = argparse.ArgumentParser(description="Tablas LaTeX del análisis de tokens de Molmo2.")
    parser.add_argument("--resultados", type=Path, default=RESULTADOS,
                        help="carpeta con tokens_<prompt>.csv y saturacion_corridas.csv")
    parser.add_argument("--salida", type=Path, default=SALIDA, help="archivo .tex de salida")
    args = parser.parse_args()

    proc = cargar_procesador(args.resultados)
    sat = cargar_saturacion(args.resultados)
    verificar(proc, sat)
    tablas = [tabla_procesador(proc)]
    tablas += [tabla_saturacion(tipo, sat, cargar_eval(tipo)) for tipo in PROMPTS]
    tablas.append(tabla_tracks(sat))

    cabecera = ("% Generado por utils/generar_tablas_tokens.py. "
                "Requiere booktabs, multirow, adjustbox y \\newsavebox{\\tablebox}.\n\n")
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(cabecera + "\n\n".join(tablas) + "\n", encoding="utf-8")
    print(f"Escrito {args.salida.relative_to(REPO) if args.salida.is_relative_to(REPO) else args.salida}")


if __name__ == "__main__":
    main()
