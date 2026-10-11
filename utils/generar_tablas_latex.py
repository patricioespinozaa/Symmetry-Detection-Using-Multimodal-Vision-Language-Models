#!/usr/bin/env python3
# Genera tablas LaTeX (booktabs) a partir de los CSV de comparación de ndc_fix/.
#
# Uso:
#   python utils/generar_tablas_latex.py                                   # todo, por consola
#   python utils/generar_tablas_latex.py --salida docs/resultados/tablas.tex
#   python utils/generar_tablas_latex.py --dir-salida Tesis_DCC_MDS/tablas  # un .tex por tipo y parte
#   python utils/generar_tablas_latex.py --tipo plane_sym --partes base variantes_1 --bloques f1
#   python utils/generar_tablas_latex.py --pivot auc_angular                # una métrica × n_views
#   python utils/generar_tablas_latex.py --ranking --tipo axis_sym --dir-salida Tesis_DCC_MDS/tablas  # axis_top.tex
#
# Partes (una tabla por parte y bloque de métricas):
#   base         prompts vNN del Flujo A
#   variantes_1  prompts vNN_1 del Flujo A
#   adicionales  v08 (ndcfix_v08)
#   6pts         prompts de 6 puntos por vista (ndcfix_6pts)
#   flujoB       descripción previa del objeto (ndcfix_flujoB)
#   flujoC       descripción + pista (ndcfix_flujoC)
# Cualquier otra carpeta ndcfix_<x> se convierte en la parte <x>.
#
# Bloques de métricas (cada uno es una tabla):
#   angular   AUC, error angular (media, mediana, desv.), precisión @5/10/15°
#   posicion  error de traslación (absoluto y normalizado), SDE_ref y cobertura
#   f1        F1_ref voraz y húngaro: promedio y por umbral (solo plano)
#   conteos   TP/FP/FN por umbral del emparejamiento voraz (solo plano)
#
# Además se genera, por tipo de simetría, una tabla resumen con todos los prompts
# (promedio sobre las configuraciones de vistas), agrupada por parte.
# Negrita = mejor valor de la columna dentro de cada tabla.
# Requiere en el preámbulo: booktabs, multirow, adjustbox.
import argparse
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
NDC_FIX = REPO / "ndc_fix"

# columna -> (mayor_es_mejor (None = sin negrita), formato)
PROPIEDADES = {
    "auc_angular": (True, "{:.3f}"),
    "angular_error_mean": (False, "{:.1f}"),
    "angular_error_median": (False, "{:.1f}"),
    "angular_error_std": (None, "{:.1f}"),
    "precision_5deg": (True, "{:.3f}"),
    "precision_10deg": (True, "{:.3f}"),
    "precision_15deg": (True, "{:.3f}"),
    "translation_error_mean": (False, "{:.4f}"),
    "translation_error_median": (False, "{:.4f}"),
    "translation_error_normalized_mean": (False, "{:.4f}"),
    "translation_error_normalized_median": (False, "{:.4f}"),
    "sde_ref_mean": (False, "{:.4f}"),
    "sde_ref_min": (None, "{:.4f}"),
    "sde_ref_max": (None, "{:.4f}"),
    "n_objects": (True, "{:.0f}"),
    "n_points_mean": (None, "{:.1f}"),
    "f1_ref": (True, "{:.3f}"),
    "f1_ref_hungarian": (True, "{:.3f}"),
}
UMBRALES = [("t005", "0.05"), ("t010", "0.10"), ("t015", "0.15"), ("t020", "0.20")]
for _t, _ in UMBRALES:
    PROPIEDADES[f"f1_ref_{_t}"] = (True, "{:.3f}")
    PROPIEDADES[f"f1_ref_hungarian_{_t}"] = (True, "{:.3f}")
    for _k in ("tp", "fp", "fn"):
        PROPIEDADES[f"{_k}_{_t}"] = (None, "{:.0f}")
# TP promedio sobre los 4 umbrales de F1_ref (columna derivada en `cargar`)
PROPIEDADES["tp_mean"] = (True, "{:.0f}")

TAU = r"$\tau{=}"
# bloque -> (descripción para el caption, tipos donde aplica, [(grupo, [(columna, subencabezado)])])
BLOQUES = {
    "angular": ("métricas angulares", ("axis_sym", "plane_sym"), [
        ("", [("auc_angular", r"AUC$_{45^\circ}$")]),
        (r"Error angular ($^\circ$)", [("angular_error_mean", "Media"),
                                       ("angular_error_median", "Mediana"),
                                       ("angular_error_std", "Desv.")]),
        ("Precisión", [("precision_5deg", r"@5$^\circ$"), ("precision_10deg", r"@10$^\circ$"),
                       ("precision_15deg", r"@15$^\circ$")]),
    ]),
    "posicion": (r"error de traslación, SDE$_{\mathrm{ref}}$ y cobertura", ("axis_sym", "plane_sym"), [
        ("Err.\\ trasl.", [("translation_error_mean", "Media"),
                           ("translation_error_median", "Mediana")]),
        ("Err.\\ trasl.\\ norm.", [("translation_error_normalized_mean", "Media"),
                                   ("translation_error_normalized_median", "Mediana")]),
        (r"SDE$_{\mathrm{ref}}$", [("sde_ref_mean", "Media"), ("sde_ref_min", "Mín."),
                                   ("sde_ref_max", "Máx.")]),
        ("Cobertura", [("n_objects", "Objetos"), ("n_points_mean", "Puntos")]),
    ]),
    "f1": (r"$F1_{\mathrm{ref}}$ por umbral", ("plane_sym",), [
        ("$F1_{\\mathrm{ref}}$ voraz", [("f1_ref", "Prom.")]
         + [(f"f1_ref_{t}", f"{TAU}{v}$") for t, v in UMBRALES]),
        ("$F1_{\\mathrm{ref}}$ húngaro", [("f1_ref_hungarian", "Prom.")]
         + [(f"f1_ref_hungarian_{t}", f"{TAU}{v}$") for t, v in UMBRALES]),
    ]),
    "conteos": (r"conteos TP/FP/FN del emparejamiento voraz de $F1_{\mathrm{ref}}$", ("plane_sym",), [
        (f"{TAU}{v}$", [(f"{k}_{t}", k.upper()) for k in ("tp", "fp", "fn")]) for t, v in UMBRALES
    ]),
}

# Columnas de la tabla resumen global (promedio sobre n_views)
RESUMEN = {
    "axis_sym": [("auc_angular", r"AUC$_{45^\circ}$"), ("angular_error_median", r"Mediana ($^\circ$)"),
                 ("precision_5deg", r"P@5$^\circ$"), ("precision_10deg", r"P@10$^\circ$"),
                 ("translation_error_normalized_median", "Trasl.\\ norm."),
                 ("sde_ref_mean", r"SDE$_{\mathrm{ref}}$"), ("n_objects", "Objetos")],
    "plane_sym": [("auc_angular", r"AUC$_{45^\circ}$"), ("angular_error_median", r"Mediana ($^\circ$)"),
                  ("precision_10deg", r"P@10$^\circ$"), ("f1_ref", r"$F1_{\mathrm{ref}}$ voraz"),
                  ("f1_ref_hungarian", r"$F1_{\mathrm{ref}}$ húng."),
                  ("sde_ref_mean", r"SDE$_{\mathrm{ref}}$"), ("n_objects", "Objetos")],
}

PARTES = {
    "base": "prompts base del Flujo A",
    "variantes_1": r"variantes \texttt{\_1} del Flujo A",
    "adicionales": r"prompt adicional \texttt{v08}",
    "6pts": "prompts de 6 puntos por vista",
    "flujoB": "Flujo B (descripción previa del objeto)",
    "flujoC": "Flujo C (descripción y pista)",
}
# Partes que entran por defecto en los rankings (--ranking): sin flujos B/C ni 6 puntos
PARTES_RANKING = ["base", "variantes_1", "adicionales"]
TITULO_TIPO = {"axis_sym": "simetría axial", "plane_sym": "simetría planar"}
CORTO_TIPO = {"axis_sym": "axis", "plane_sym": "plane"}
NOTA = ("Negrita: mejor valor de la columna en esta tabla. Las métricas angulares imputan "
        r"$90^\circ$ a los objetos sin predicción; traslación, SDE$_{\mathrm{ref}}$ y "
        r"$F1_{\mathrm{ref}}$ usan solo los objetos con predicción (de 850).")


def etiqueta_prompt(experiment: str) -> str:
    """ Acorta el nombre del experimento: plane_v06_ndcfix_mp1_nomesh -> v06. """
    e = re.sub(r"^(axis|plane)_", "", experiment)
    return re.sub(r"_ndcfix(_mp1)?_nomesh$", "", e)


def parte_de(flujo: str, prompt: str) -> str:
    """ Asigna la parte del estudio a la que pertenece un prompt.

    Args:
        * flujo: carpeta de origen sin el prefijo 'ndcfix_' (p. ej. 'flujoA')
        * prompt: etiqueta corta del prompt (p. ej. 'v04_1')

    Returns:
        * str: clave de PARTES, o el nombre del flujo si no es una parte conocida

    """
    if flujo == "flujoA":
        return "variantes_1" if prompt.endswith("_1") else "base"
    return {"v08": "adicionales"}.get(flujo, flujo)


def escapar(texto: str) -> str:
    """ Escapa los caracteres de LaTeX que aparecen en nombres de prompt. """
    return texto.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")


def cargar(carpetas: list[Path], tipo: str, metodo: str) -> pd.DataFrame:
    """ Lee y concatena todos los *_comparison.csv de un tipo de simetría.

    Args:
        * carpetas: carpetas donde buscar recursivamente los CSV
        * tipo: 'axis_sym' o 'plane_sym'
        * metodo: valor de la columna 'method' a conservar

    Returns:
        * pd.DataFrame: filas de todos los CSV con columnas extra 'flujo', 'prompt' y 'parte'

    """
    frames = []
    for carpeta in carpetas:
        for csv in sorted(carpeta.rglob(f"{tipo}_*comparison.csv")):
            d = pd.read_csv(csv)
            flujo = next((p.name for p in csv.parents if p.parent == NDC_FIX), carpeta.name)
            d["flujo"] = flujo.removeprefix("ndcfix_")
            frames.append(d)
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d = d[d["method"] == metodo].copy()
    d["prompt"] = d["experiment"].map(etiqueta_prompt)
    d["parte"] = [parte_de(f, p) for f, p in zip(d["flujo"], d["prompt"])]
    tp_cols = [f"tp_{t}" for t, _ in UMBRALES if f"tp_{t}" in d.columns]
    if tp_cols:
        d["tp_mean"] = d[tp_cols].mean(axis=1)
    return d.drop_duplicates(subset=["experiment", "method", "n_views"])


def orden_prompt(prompt: str) -> tuple:
    """ Clave de orden natural: v02 < v02_1 < v10. """
    return tuple(int(t) if t.isdigit() else t for t in re.split(r"(\d+)", prompt))


def celdas_formateadas(serie: pd.Series, columna: str) -> list[str]:
    """ Formatea una columna numérica poniendo en negrita el mejor valor (y sus empates).

    Args:
        * serie: valores de la métrica
        * columna: nombre de la métrica (clave de PROPIEDADES)

    Returns:
        * list[str]: celdas LaTeX, '--' para valores faltantes

    """
    mayor, fmt = PROPIEDADES.get(columna, (None, "{:.3f}"))
    validos = serie.dropna()
    mejor = None
    if mayor is not None and len(validos) > 1:
        mejor = fmt.format(validos.max() if mayor else validos.min())
    celdas = []
    for v in serie:
        if pd.isna(v):
            celdas.append("--")
        elif fmt.format(v) == mejor:
            celdas.append(rf"\textbf{{{mejor}}}")
        else:
            celdas.append(fmt.format(v))
    return celdas


def encabezado_dos_niveles(fijas: list[str], grupos: list[tuple[str, list[str]]]) -> list[str]:
    """ Construye el encabezado con grupos (\\multicolumn + \\cmidrule) y subcolumnas.

    Args:
        * fijas: encabezados de las columnas iniciales (ocupan las dos filas)
        * grupos: [(título del grupo, [subencabezados])]; título vacío = sin grupo

    Returns:
        * list[str]: líneas LaTeX del encabezado (sin \\toprule ni \\midrule)

    """
    if not any(titulo for titulo, _ in grupos):
        return [" & ".join(fijas + [s for _, subs in grupos for s in subs]) + r" \\"]
    fila1, fila2, reglas = [rf"\multirow{{2}}{{*}}{{{f}}}" for f in fijas], [""] * len(fijas), []
    col = len(fijas) + 1
    for titulo, subs in grupos:
        n = len(subs)
        if titulo:
            fila1.append(rf"\multicolumn{{{n}}}{{c}}{{{titulo}}}")
            reglas.append(rf"\cmidrule(lr){{{col}-{col + n - 1}}}")
            fila2 += subs
        else:
            fila1 += [rf"\multirow{{2}}{{*}}{{{s}}}" for s in subs]
            fila2 += [""] * n
        col += n
    return [" & ".join(fila1) + r" \\", " ".join(reglas), " & ".join(fila2) + r" \\"]


def envolver_tabla(encabezado: list[str], cuerpo: list[str], alineacion: str,
                   caption: str, label: str) -> str:
    """ Envuelve encabezado y filas en table + adjustbox + tabular con booktabs.

    Args:
        * encabezado: líneas del encabezado
        * cuerpo: filas de datos ya formateadas
        * alineacion: especificación de columnas del tabular, p. ej. 'llrr'
        * caption: texto del \\caption
        * label: texto del \\label

    Returns:
        * str: tabla LaTeX completa

    """
    lineas = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{adjustbox}{max width=\textwidth}",
        rf"\begin{{tabular}}{{{alineacion}}}",
        r"\toprule",
        *encabezado,
        r"\midrule",
        *cuerpo,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\end{table}",
    ]
    return "\n".join(lineas) + "\n"


def tabla_bloque(d: pd.DataFrame, tipo: str, parte: str, bloque: str) -> str:
    """ Tabla de una parte con una fila por (prompt, n_views) y las columnas de un bloque.

    Args:
        * d: filas de la parte (salida de `cargar` filtrada)
        * tipo: 'axis_sym' o 'plane_sym'
        * parte: clave de la parte (para caption y label)
        * bloque: clave de BLOQUES

    Returns:
        * str: tabla LaTeX, o '' si la parte no tiene ninguna columna del bloque

    """
    descripcion, _, grupos = BLOQUES[bloque]
    grupos = [(t, [(c, s) for c, s in cols if c in d.columns]) for t, cols in grupos]
    grupos = [(t, cols) for t, cols in grupos if cols]
    if not grupos:
        return ""
    metricas = [c for _, cols in grupos for c, _ in cols]

    d = d.assign(_orden=d["prompt"].map(orden_prompt)).sort_values(["_orden", "n_views"])
    d = d.reset_index(drop=True)
    columnas = {m: celdas_formateadas(d[m], m) for m in metricas}
    n_vistas = d.groupby("prompt")["n_views"].size()

    cuerpo, previo = [], None
    for i, r in d.iterrows():
        nuevo = r["prompt"] != previo
        if nuevo and previo is not None:
            cuerpo.append(r"\midrule")
        nombre = rf"\multirow{{{n_vistas[r['prompt']]}}}{{*}}{{\texttt{{{escapar(r['prompt'])}}}}}"
        celdas = [nombre if nuevo else "", str(int(r["n_views"]))]
        celdas += [columnas[m][i] for m in metricas]
        cuerpo.append(" & ".join(celdas) + r" \\")
        previo = r["prompt"]

    encabezado = encabezado_dos_niveles(["Prompt", "Vistas"],
                                        [(t, [s for _, s in cols]) for t, cols in grupos])
    titulo_parte = PARTES.get(parte, escapar(parte))
    caption = (rf"\textit{{Pipeline}} sin malla, {TITULO_TIPO[tipo]}, {titulo_parte}: "
               rf"{descripcion}. {NOTA}")
    label = f"tab:{CORTO_TIPO[tipo]}_{parte}_{bloque}"
    return envolver_tabla(encabezado, cuerpo, "lr" + "r" * len(metricas), caption, label)


def tabla_resumen(d: pd.DataFrame, tipo: str, partes: list[str]) -> str:
    """ Tabla con una fila por prompt (promedio sobre n_views), agrupada por parte.

    Args:
        * d: datos cargados por `cargar`
        * tipo: 'axis_sym' o 'plane_sym'
        * partes: partes a incluir, en orden

    Returns:
        * str: tabla LaTeX

    """
    columnas = [(c, s) for c, s in RESUMEN[tipo] if c in d.columns]
    metricas = [c for c, _ in columnas]
    m = d.groupby(["parte", "prompt"])[metricas].mean().reset_index()
    m = m[m["parte"].isin(partes)]
    m["_p"] = m["parte"].map(partes.index)
    m["_o"] = m["prompt"].map(orden_prompt)
    m = m.sort_values(["_p", "_o"]).reset_index(drop=True)
    celdas = {c: celdas_formateadas(m[c], c) for c in metricas}

    cuerpo, previo = [], None
    n = len(metricas) + 1
    for i, r in m.iterrows():
        if r["parte"] != previo:
            if previo is not None:
                cuerpo.append(r"\midrule")
            titulo = PARTES.get(r["parte"], escapar(r["parte"]))
            cuerpo.append(rf"\multicolumn{{{n}}}{{l}}{{\textit{{{titulo[0].upper() + titulo[1:]}}}}} \\")
        fila = [rf"\quad\texttt{{{escapar(r['prompt'])}}}"] + [celdas[c][i] for c in metricas]
        cuerpo.append(" & ".join(fila) + r" \\")
        previo = r["parte"]

    encabezado = [" & ".join(["Prompt"] + [s for _, s in columnas]) + r" \\"]
    vistas = ", ".join(str(v) for v in sorted(d["n_views"].unique()))
    caption = (rf"Resumen de todos los prompts del \textit{{pipeline}} sin malla, {TITULO_TIPO[tipo]} "
               rf"(promedio sobre {vistas} vistas). {NOTA}")
    return envolver_tabla(encabezado, cuerpo, "l" + "r" * len(metricas), caption,
                          f"tab:{CORTO_TIPO[tipo]}_resumen")


# Criterios de desempate del ranking, en orden de prioridad:
# (columna, encabezado, mayor_es_mejor, brecha de empate, brecha en texto para el caption).
# Brecha 0 = último criterio: gana el mejor valor sin tolerancia.
# En los encabezados, '|' separa las líneas de un encabezado apilado.
CRITERIOS = {
    "axis_sym": [
        ("auc_angular", r"AUC$_{45^\circ}$", True, 0.05, "0.05"),
        ("angular_error_median", r"Mediana|($^\circ$)", False, 2.0, r"$2^\circ$"),
        ("precision_5deg", r"P@5$^\circ$", True, 0.05, "0.05"),
        ("precision_10deg", r"P@10$^\circ$", True, 0.05, "0.05"),
        ("n_objects", "Objetos", True, 17, "17 objetos"),
        ("translation_error_normalized_median", r"Trasl.|norm.", False, 0.01, "0.01"),
        ("sde_ref_mean", r"SDE$_{\mathrm{ref}}$", False, 0, ""),
    ],
    "plane_sym": [
        ("f1_ref", r"$F1_{\mathrm{ref}}$|voraz", True, 0.05, "0.05"),
        ("auc_angular", r"AUC$_{45^\circ}$", True, 0.05, "0.05"),
        ("precision_10deg", r"P@10$^\circ$", True, 0.05, "0.05"),
        ("n_objects", "Objetos", True, 17, "17 objetos"),
        ("sde_ref_mean", r"SDE$_{\mathrm{ref}}$", False, 0.005, "0.005"),
        ("angular_error_mean", r"Err.\ medio|($^\circ$)", False, 0, ""),
    ],
}
# Columnas informativas que se muestran pero no participan en el desempate
INFORMATIVAS = {
    "axis_sym": [],
    "plane_sym": [("f1_ref_hungarian", r"$F1_{\mathrm{ref}}$|húngaro"),
                  ("tp_mean", r"TP|(prom.)")],
}


def apilar(encabezado: str) -> str:
    """ Convierte 'línea 1|línea 2' en una celda de encabezado de dos líneas. """
    partes = encabezado.split("|")
    if len(partes) == 1:
        return encabezado
    return r"\begin{tabular}[b]{@{}c@{}}" + r" \\ ".join(partes) + r"\end{tabular}"


def ordenar_con_tolerancia(d: pd.DataFrame, criterios: list[tuple]) -> tuple[pd.DataFrame, list]:
    """ Ordena configuraciones con desempate lexicográfico con tolerancia.

    En cada paso se toma el grupo de empate de la métrica de mayor prioridad
    (las configuraciones que difieren de la líder en menos que la brecha) y se
    refina con la métrica siguiente hasta que queda una sola configuración, que
    ocupa la siguiente posición del ranking. Se repite con las restantes.

    Args:
        * d: una fila por configuración
        * criterios: lista de CRITERIOS[tipo]

    Returns:
        * pd.DataFrame: filas de `d` en el orden del ranking
        * list: por posición, número (1-based) del criterio que la resolvió,
          o None si era la única configuración restante

    """
    criterios = [c for c in criterios if c[0] in d.columns]
    # Orden lexicográfico estricto de base: decide empates exactos de forma determinista
    restantes = d.sort_values([c[0] for c in criterios],
                              ascending=[not c[2] for c in criterios], na_position="last")
    orden, decisivo = [], []
    while len(restantes):
        grupo, nivel = restantes, None
        if len(restantes) > 1:
            for k, (col, _, mayor, brecha, _) in enumerate(criterios, 1):
                # NaN = peor valor posible
                v = grupo[col].fillna(-1e18 if mayor else 1e18)
                dif = (v.max() - v) if mayor else (v - v.min())
                grupo = grupo[dif < brecha] if brecha > 0 else grupo[dif <= 0]
                nivel = k
                if len(grupo) == 1:
                    break
        elegido = grupo.index[0]
        orden.append(elegido)
        decisivo.append(nivel)
        restantes = restantes.drop(elegido)
    return d.loc[orden], decisivo


def tabla_ranking(d: pd.DataFrame, tipo: str, partes: list[str], top: int | None) -> str:
    """ Ranking desagregado: una fila por (prompt, n_views), ordenado con CRITERIOS[tipo].

    Args:
        * d: datos cargados por `cargar`
        * tipo: 'axis_sym' o 'plane_sym'
        * partes: partes a incluir
        * top: si no es None, solo las primeras `top` filas

    Returns:
        * str: tabla LaTeX (longtable)

    """
    criterios = [c for c in CRITERIOS[tipo] if c[0] in d.columns]
    informativas = [c for c in INFORMATIVAS[tipo] if c[0] in d.columns]
    d, decisivo = ordenar_con_tolerancia(d[d["parte"].isin(partes)], criterios)
    if top:
        d, decisivo = d.head(top), decisivo[:top]
    d = d.reset_index(drop=True)
    columnas = [(c[0], c[1]) for c in criterios] + informativas
    metricas = [c for c, _ in columnas]
    celdas = {c: celdas_formateadas(d[c], c) for c in metricas}

    cuerpo = []
    for i, r in d.iterrows():
        if i and i % 5 == 0:
            cuerpo.append(r"\addlinespace")
        crit = str(decisivo[i]) if decisivo[i] else "--"
        fila = [str(i + 1), rf"\texttt{{{escapar(r['prompt'])}}}", str(int(r["n_views"])), crit]
        cuerpo.append(" & ".join(fila + [celdas[c][i] for c in metricas]) + r" \\")

    encabezado = [" & ".join([r"\#", "Prompt", "Vistas", "Crit."] + [apilar(s) for _, s in columnas])
                  + r" \\"]
    prioridad = "; ".join(
        f"({k}) {h.replace('|', ' ')}" + (f", brecha {b}" if b else ", gana el mejor valor")
        for k, (_, h, _, _, b) in enumerate(criterios, 1))
    alcance = f"las {top} mejores configuraciones" if top else "todas las configuraciones"
    nombres_info = [h.replace("|", " ") for _, h in informativas]
    if len(nombres_info) > 1:
        nombres_info = [", ".join(nombres_info[:-1]) + " y " + nombres_info[-1]]
    nota_info = (f" {nombres_info[0]} {'se muestran' if len(informativas) > 1 else 'se muestra'} "
                 "solo como información y no participan en el desempate."
                 if informativas else "")
    if "tp_mean" in metricas:
        nota_info += (r" TP (prom.): planos correctamente detectados (verdaderos positivos de "
                      r"$F1_{\mathrm{ref}}$ voraz), promedio sobre los cuatro umbrales.")
    caption = (rf"Ranking de {alcance} (prompt $\times$ número de vistas) del \textit{{pipeline}} "
               rf"sin malla, {TITULO_TIPO[tipo]}, con desempate lexicográfico con tolerancia. "
               rf"{nota_partes(partes)} "
               rf"Orden de prioridad: {prioridad}. Crit.: criterio que resolvió la posición. "
               r"Negrita: mejor valor de la columna. Las métricas angulares imputan $90^\circ$ a "
               r"los objetos sin predicción; traslación, SDE$_{\mathrm{ref}}$ y $F1_{\mathrm{ref}}$ "
               r"usan solo los objetos con predicción. Objetos: cobertura, de 850." + nota_info)
    # longtable: con decenas de filas se parte entre páginas en vez de reducir la letra
    n_cols = 4 + len(metricas)
    continua = rf"\multicolumn{{{n_cols}}}{{r}}{{\small\textit{{Continúa en la página siguiente}}}} \\"
    lineas = [
        # con más de 7 métricas (plano) la tabla no cabe a \small en el ancho del texto
        ("\\begingroup"
         + (r"\footnotesize\setlength{\tabcolsep}{3pt}" if len(metricas) > 7
            else r"\small\setlength{\tabcolsep}{4pt}")
         + r"\setlength{\LTcapwidth}{\textwidth}"),
        rf"\begin{{longtable}}{{rlrc{'r' * len(metricas)}}}",
        rf"\caption{{{caption}}}",
        rf"\label{{tab:{CORTO_TIPO[tipo]}_top}} \\",
        r"\toprule", *encabezado, r"\midrule",
        r"\endfirsthead",
        rf"\multicolumn{{{n_cols}}}{{l}}{{\small\textit{{\tablename~\thetable{{}} (continuación)}}}} \\",
        r"\toprule", *encabezado, r"\midrule",
        r"\endhead",
        r"\midrule", continua,
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
        *cuerpo,
        r"\end{longtable}",
        r"\endgroup",
    ]
    return "\n".join(lineas) + "\n"


def nota_partes(partes: list[str]) -> str:
    """ Oración del caption que indica qué partes del estudio entran en el ranking. """
    nombres = [PARTES.get(p, escapar(p)) for p in partes]
    lista = nombres[0] if len(nombres) == 1 else ", ".join(nombres[:-1]) + " y " + nombres[-1]
    return f"Incluye: {lista}."


def tabla_ranking_vistas(d: pd.DataFrame, tipo: str, partes: list[str], top: int | None) -> str:
    """ Ranking de prompts separado por número de vistas: una columna por n_views.

    En cada número de vistas se ordenan los prompts con `ordenar_con_tolerancia`
    y los mismos CRITERIOS[tipo] del ranking global. Cada celda muestra el prompt,
    el valor de la métrica principal y el criterio que resolvió la posición.

    Args:
        * d: datos cargados por `cargar`
        * tipo: 'axis_sym' o 'plane_sym'
        * partes: partes a incluir
        * top: si no es None, solo los primeros `top` puestos

    Returns:
        * str: tabla LaTeX

    """
    criterios = [c for c in CRITERIOS[tipo] if c[0] in d.columns]
    principal, encab_principal = criterios[0][0], criterios[0][1].replace("|", " ")
    fmt = PROPIEDADES.get(principal, (None, "{:.3f}"))[1]
    d = d[d["parte"].isin(partes)]
    vistas = sorted(d["n_views"].unique())

    columnas = {}
    for nv in vistas:
        orden, decisivo = ordenar_con_tolerancia(d[d["n_views"] == nv], criterios)
        columnas[nv] = [(r["prompt"], r[principal], k) for (_, r), k in zip(orden.iterrows(), decisivo)]
    n_filas = max(len(c) for c in columnas.values())
    if top:
        n_filas = min(n_filas, top)

    cuerpo = []
    for i in range(n_filas):
        if i and i % 5 == 0:
            cuerpo.append(r"\addlinespace")
        celdas = [str(i + 1)]
        for nv in vistas:
            if i < len(columnas[nv]):
                prompt, valor, k = columnas[nv][i]
                celdas += [rf"\texttt{{{escapar(prompt)}}}", fmt.format(valor), str(k) if k else "--"]
            else:
                celdas += ["", "", ""]
        cuerpo.append(" & ".join(celdas) + r" \\")

    encabezado = encabezado_dos_niveles(
        ["Puesto"], [(f"{nv} vistas", ["Prompt", encab_principal, "Crit."]) for nv in vistas])
    prioridad = "; ".join(f"({k}) {c[1].replace('|', ' ')}" for k, c in enumerate(criterios, 1))
    caption = (rf"Ranking de \textit{{prompts}} por número de vistas, \textit{{pipeline}} sin malla, "
               rf"{TITULO_TIPO[tipo]}. {nota_partes(partes)} En cada número de vistas se aplica el "
               rf"mismo desempate lexicográfico con tolerancia que en la "
               rf"Tabla~\ref{{tab:{CORTO_TIPO[tipo]}_top}} (prioridad: {prioridad}). Para cada "
               rf"prompt se muestra el valor de la métrica principal ({encab_principal}) y el "
               rf"criterio que resolvió su posición (Crit.).")
    return envolver_tabla(encabezado, cuerpo, "r" + "lrc" * len(vistas), caption,
                          f"tab:{CORTO_TIPO[tipo]}_top_vistas")


def tabla_pivot(d: pd.DataFrame, tipo: str, parte: str, metrica: str) -> str:
    """ Tabla de una parte con una fila por prompt y una columna por número de vistas.

    Args:
        * d: filas de la parte
        * tipo: 'axis_sym' o 'plane_sym'
        * parte: clave de la parte
        * metrica: columna a mostrar

    Returns:
        * str: tabla LaTeX

    """
    p = d.pivot_table(index="prompt", columns="n_views", values=metrica, aggfunc="first")
    p = p.loc[sorted(p.index, key=orden_prompt)]
    p["media"] = p.mean(axis=1)
    columnas = {v: celdas_formateadas(p[v], metrica) for v in p.columns}

    encabezado = [" & ".join(["Prompt"] + [f"{v} vistas" for v in p.columns[:-1]] + ["Media"]) + r" \\"]
    cuerpo = []
    for i, prompt in enumerate(p.index):
        fila = [rf"\texttt{{{escapar(prompt)}}}"] + [columnas[v][i] for v in p.columns]
        cuerpo.append(" & ".join(fila) + r" \\")

    nombres = {c: f"{g} ({s})" if g else s
               for _, _, grupos in BLOQUES.values() for g, cols in grupos for c, s in cols}
    caption = (rf"{nombres.get(metrica, escapar(metrica))} por prompt y número de vistas, {TITULO_TIPO[tipo]}, "
               rf"{PARTES.get(parte, escapar(parte))}.")
    return envolver_tabla(encabezado, cuerpo, "l" + "r" * len(p.columns), caption,
                          f"tab:{CORTO_TIPO[tipo]}_{parte}_{metrica}")


def main() -> None:
    """ Punto de entrada CLI. """
    parser = argparse.ArgumentParser(description="Genera tablas LaTeX desde ndc_fix/.")
    parser.add_argument("--carpetas", nargs="+", type=Path, default=[NDC_FIX],
                        help="carpetas a recorrer (default: ndc_fix/ completo)")
    parser.add_argument("--tipo", choices=["axis_sym", "plane_sym", "ambos"], default="ambos")
    parser.add_argument("--partes", nargs="+",
                        help=f"partes a incluir (default: todas, en orden {list(PARTES)})")
    parser.add_argument("--bloques", nargs="+", choices=list(BLOQUES), default=list(BLOQUES),
                        help="bloques de métricas, una tabla por bloque (default: todos)")
    parser.add_argument("--sin-resumen", action="store_true",
                        help="no generar la tabla resumen global por tipo")
    parser.add_argument("--pivot", metavar="METRICA",
                        help="en vez de los bloques, pivotea esta métrica por n_views (una tabla por parte)")
    parser.add_argument("--ranking", action="store_true",
                        help="solo genera los rankings <axis|plane>_top (prompt × vistas) y "
                             "<axis|plane>_top_vistas (uno por número de vistas); por defecto "
                             f"solo con las partes {PARTES_RANKING}")
    parser.add_argument("--top", type=int, help="con --ranking, solo las primeras N filas")
    parser.add_argument("--metodo", default="triangulation",
                        help="filtra por la columna 'method' (default: triangulation)")
    parser.add_argument("--salida", type=Path, help="archivo .tex único de salida")
    parser.add_argument("--dir-salida", type=Path,
                        help="carpeta donde escribir un .tex por tipo y parte (<axis|plane>_<parte>.tex)")
    args = parser.parse_args()

    tipos = ["axis_sym", "plane_sym"] if args.tipo == "ambos" else [args.tipo]
    archivos: dict[str, list[str]] = {}
    for tipo in tipos:
        d = cargar(args.carpetas, tipo, args.metodo)
        if d.empty:
            continue
        presentes = list(dict.fromkeys(list(PARTES) + sorted(d["parte"].unique())))
        partes = [p for p in (args.partes or presentes) if p in set(d["parte"])]
        faltantes = sorted(set(args.partes or []) - set(partes))
        if faltantes:
            print(f"% [aviso] {tipo}: partes sin datos: {faltantes}")
        corto = CORTO_TIPO[tipo]

        if args.ranking:
            # Por defecto el ranking excluye flujos B/C y 6 puntos (variantes de un solo prompt)
            partes_rank = (partes if args.partes
                           else [p for p in PARTES_RANKING if p in set(d["parte"])])
            archivos[f"{corto}_top"] = [tabla_ranking(d, tipo, partes_rank, args.top)]
            archivos[f"{corto}_top_vistas"] = [tabla_ranking_vistas(d, tipo, partes_rank, args.top)]
            continue
        if not args.sin_resumen and not args.pivot:
            archivos.setdefault(f"{corto}_resumen", []).append(tabla_resumen(d, tipo, partes))
        for parte in partes:
            dp = d[d["parte"] == parte]
            clave = f"{corto}_{parte}"
            if args.pivot:
                if args.pivot in dp.columns:
                    archivos.setdefault(clave, []).append(tabla_pivot(dp, tipo, parte, args.pivot))
                continue
            for bloque in args.bloques:
                if tipo in BLOQUES[bloque][1]:
                    tex = tabla_bloque(dp, tipo, parte, bloque)
                    if tex:
                        archivos.setdefault(clave, []).append(tex)

    cabecera = ("% Generado por utils/generar_tablas_latex.py. "
                "Requiere booktabs, multirow, adjustbox y longtable (ranking).\n\n")
    n_tablas = sum(len(v) for v in archivos.values())
    if args.dir_salida:
        args.dir_salida.mkdir(parents=True, exist_ok=True)
        for nombre, tablas in archivos.items():
            (args.dir_salida / f"{nombre}.tex").write_text(cabecera + "\n".join(tablas), encoding="utf-8")
        print(f"Escritos {len(archivos)} archivo(s), {n_tablas} tabla(s) en {args.dir_salida}")
    else:
        tex = cabecera + "\n".join(t for tablas in archivos.values() for t in tablas)
        if args.salida:
            args.salida.parent.mkdir(parents=True, exist_ok=True)
            args.salida.write_text(tex, encoding="utf-8")
            print(f"Escrito {args.salida} ({n_tablas} tabla(s))")
        else:
            print(tex)


if __name__ == "__main__":
    main()
