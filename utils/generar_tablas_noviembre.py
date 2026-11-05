#!/usr/bin/env python3
# Genera un .tex con las tablas de cap4.tex recalculadas desde resultados_finales/: eje con EXP-E y
# plano con S_dat (un plano por objeto). Cada título lleva "Noviembre" y la resolución, la
# iluminación y el modo de envío. No modifica cap4.tex.
#
#   python utils/generar_tablas_noviembre.py
#   python utils/generar_tablas_noviembre.py --salida otra/ruta.tex
#
# Solo usa la biblioteca estándar. Requiere en el preámbulo: booktabs, multirow, adjustbox y
# \newsavebox{\tablebox}.
import argparse
import csv
import re
from pathlib import Path

from generar_tablas_vlms import COLUMNAS as COLUMNAS_VLM
from generar_tablas_vlms import NOTA_ENVIO

REPO = Path(__file__).resolve().parents[1]
RESULTADOS = REPO / "resultados_finales"
SALIDA = REPO / "Tesis_DCC_MDS" / "cap4_tablas_noviembre.tex"
VISTAS = [6, 14, 26]
OBJETOS_TOTALES = 850
TOP = 15
TABLAS_POR_PAGINA = 2

ID_FINAL = re.compile(r"^(axis|plane)_(.+)_s\d+_[a-z]+_ndcfix_(expE|sdat_mp1)_nomesh$")

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
    "tp_mean": (True, "{:.0f}"),
}
UMBRALES = [("t005", "0.05"), ("t010", "0.10"), ("t015", "0.15"), ("t020", "0.20")]
for _t, _ in UMBRALES:
    PROPIEDADES[f"f1_ref_{_t}"] = (True, "{:.3f}")
    PROPIEDADES[f"f1_ref_hungarian_{_t}"] = (True, "{:.3f}")
    for _k in ("tp", "fp", "fn"):
        PROPIEDADES[f"{_k}_{_t}"] = (None, "{:.0f}")

TAU = r"$\tau{=}"
# bloque -> (descripción del título, [(grupo, [(columna, subencabezado)])])
BLOQUES = {
    "angular": ("Métricas angulares", [
        ("", [("auc_angular", r"AUC$_{45^\circ}$")]),
        (r"Error angular ($^\circ$)", [("angular_error_mean", "Media"),
                                       ("angular_error_median", "Mediana"),
                                       ("angular_error_std", "Desv.")]),
        ("Precisión", [("precision_5deg", r"@5$^\circ$"), ("precision_10deg", r"@10$^\circ$"),
                       ("precision_15deg", r"@15$^\circ$")]),
    ]),
    "posicion": ("Error de traslación, SDE y cobertura", [
        (r"Err.\ trasl.", [("translation_error_mean", "Media"),
                           ("translation_error_median", "Mediana")]),
        (r"Err.\ trasl.\ norm.", [("translation_error_normalized_mean", "Media"),
                                  ("translation_error_normalized_median", "Mediana")]),
        (r"SDE$_{\mathrm{ref}}$", [("sde_ref_mean", "Media"), ("sde_ref_min", "Mín."),
                                   ("sde_ref_max", "Máx.")]),
        ("Cobertura", [("n_objects", "Objetos"), ("n_points_mean", "Puntos")]),
    ]),
    "f1": ("F1-scores", [
        ("$F1_{\\mathrm{ref}}$ voraz", [("f1_ref", "Prom.")]
         + [(f"f1_ref_{t}", f"{TAU}{v}$") for t, v in UMBRALES]),
        ("$F1_{\\mathrm{ref}}$ húngaro", [("f1_ref_hungarian", "Prom.")]
         + [(f"f1_ref_hungarian_{t}", f"{TAU}{v}$") for t, v in UMBRALES]),
    ]),
    "conteos": (r"Conteos TP/FP/FN del emparejamiento voraz de $F1_{\mathrm{ref}}$", [
        (f"{TAU}{v}$", [(f"{k}_{t}", k.upper()) for k in ("tp", "fp", "fn")]) for t, v in UMBRALES
    ]),
}

METODO = {
    "axis_sym": r"El eje se estima con reponderación iterativa (EXP-E).",
    "plane_sym": r"La normal se elige con $S_{dat}$ y se predice un plano por objeto.",
}
TITULO_TIPO = {"axis_sym": "simetría axial", "plane_sym": "simetría planar"}
CORTO_TIPO = {"axis_sym": "axis", "plane_sym": "plane"}
NOTA_ANGULAR = r"Las métricas angulares imputan $90^\circ$ a los objetos sin predicción."
NOTA_POSICION = (r"La traslación y SDE$_{\mathrm{ref}}$ usan solo los objetos con predicción "
                 r"(de 850).")
NOTA_F1 = (r"Un plano predicho es acierto si su distancia a un plano de referencia es menor que "
           r"$\tau$. Prom.: promedio de los cuatro umbrales. Voraz y húngaro difieren en cómo se "
           r"emparejan los planos predichos con los de referencia. Como se predice un solo plano "
           r"por objeto, en objetos con varias simetrías no se pueden detectar todas. Solo se "
           r"usan los objetos con predicción. Negrita: mejor valor de la columna.")
NOTA_CONTEOS = (r"Conteos del emparejamiento voraz, solo sobre los objetos con predicción. "
                r"TP: aciertos; FP y FN: planos predichos y de referencia sin emparejar.")
NOTA_VARIANTE = (r"Cada prompt \texttt{\_1} representa una variante del prompt base en la que se "
                 r"busca mejorar los resultados respecto a ese. ")
NOTA_SIN_DATOS = " A esta resolución no se ejecutó la inferencia con {} vistas."
NOTA_COMPARATIVA = {
    "axis_sym": (r"Negrita: mejor valor de la columna. " + NOTA_ANGULAR + r" La traslación y "
                 r"SDE$_{\mathrm{ref}}$ usan solo los objetos con predicción. Objetos: cobertura, "
                 r"de 850."),
    "plane_sym": (r"Negrita: mejor valor de la columna. " + NOTA_ANGULAR + r" SDE$_{\mathrm{ref}}$ "
                  r"y $F1_{\mathrm{ref}}$ usan solo los objetos con predicción. Objetos: "
                  r"cobertura, de 850."),
}
NOTA_ILUMINACION = (r"Iluminación: valor de gris uniforme de la malla (\textit{flat} 0.7, "
                    r"\textit{darker} 0.3, \textit{brighter} 0.95). ")

# tipo -> columnas de las tablas comparativas: (columna, encabezado con '|' entre líneas)
COMPARATIVAS = {
    "axis_sym": [("n_objects", "Objetos"), ("auc_angular", r"AUC$_{45^\circ}$"),
                 ("angular_error_median", r"Mediana|($^\circ$)"),
                 ("precision_5deg", r"P@5$^\circ$"), ("precision_10deg", r"P@10$^\circ$"),
                 ("translation_error_normalized_median", "Trasl.|norm."),
                 ("sde_ref_mean", r"SDE$_{\mathrm{ref}}$")],
    "plane_sym": [("n_objects", "Objetos"), ("f1_ref", r"$F1_{\mathrm{ref}}$|voraz"),
                  ("f1_ref_hungarian", r"$F1_{\mathrm{ref}}$|húngaro"),
                  ("auc_angular", r"AUC$_{45^\circ}$"), ("precision_10deg", r"P@10$^\circ$"),
                  ("angular_error_mean", r"Err.\ medio|($^\circ$)"),
                  ("sde_ref_mean", r"SDE$_{\mathrm{ref}}$")],
}

# (columna, encabezado, mayor_es_mejor, brecha de empate); brecha 0 = gana el mejor valor
CRITERIOS = {
    "axis_sym": [
        ("auc_angular", r"AUC$_{45^\circ}$", True, 0.05),
        ("angular_error_median", r"Mediana|($^\circ$)", False, 2.0),
        ("precision_5deg", r"P@5$^\circ$", True, 0.05),
        ("precision_10deg", r"P@10$^\circ$", True, 0.05),
        ("n_objects", "Objetos", True, 17),
        ("translation_error_normalized_median", "Trasl.|norm.", False, 0.01),
        ("sde_ref_mean", r"SDE$_{\mathrm{ref}}$", False, 0),
    ],
    "plane_sym": [
        ("f1_ref", r"$F1_{\mathrm{ref}}$|voraz", True, 0.05),
        ("auc_angular", r"AUC$_{45^\circ}$", True, 0.05),
        ("precision_10deg", r"P@10$^\circ$", True, 0.05),
        ("n_objects", "Objetos", True, 17),
        ("sde_ref_mean", r"SDE$_{\mathrm{ref}}$", False, 0.005),
        ("angular_error_mean", r"Err.\ medio|($^\circ$)", False, 0),
    ],
}
INFORMATIVAS = {
    "axis_sym": [],
    "plane_sym": [("f1_ref_hungarian", r"$F1_{\mathrm{ref}}$|húngaro"), ("tp_mean", "TP|(prom.)")],
}

BASE = ["v00", "v01", "v02", "v03", "v04", "v05"]
VARIANTES = [f"{p}_1" for p in BASE]
ADICIONALES = ["v06", "v07", "v08"]
SELECCION = {
    "axis_sym": {"prompt": "v04", "resolucion_448": ["v04", "v06", "v08"],
                 "flujo_b": "v04_flowB", "flujo_c": "v05_1_flowC", "respaldo_c": "v05\\_1",
                 "traza": "del eje"},
    "plane_sym": {"prompt": "v05_1", "resolucion_448": ["v02", "v05_1", "v06"],
                  "flujo_b": "v05_1_flowB", "flujo_c": "v04_1_flowC", "respaldo_c": "v04\\_1",
                  "traza": "del plano"},
}
MULTI = r"envío multiimagen (\textit{multi})"
SINGLE = r"envío individual (\textit{single})"


def configuracion(size: int, luz: str, modo: str) -> str:
    """ Escribe la resolución, la iluminación y el modo de envío para un título.

    Args:
        * size: resolución en píxeles
        * luz: iluminación
        * modo: MULTI o SINGLE

    Returns:
        * str: fragmento LaTeX

    """
    return rf"resolución {size}px, iluminación \textit{{{luz}}}, {modo}"


def escapar(texto: str) -> str:
    """ Escapa los guiones bajos de los nombres de prompt. """
    return texto.replace("_", r"\_")


def numero(texto: str | None) -> float | None:
    """ Convierte una celda de CSV en número; una celda vacía es None. """
    return float(texto) if texto not in (None, "") else None


def filas_de_csv(ruta: Path) -> dict[tuple[str, int], dict[str, float | None]]:
    """ Lee un CSV de comparación con los ids de resultados_finales/.

    Args:
        * ruta: CSV de compare_results_no_mesh.py

    Returns:
        * dict[tuple[str, int], dict[str, float | None]]: fila por (prompt, vistas)

    """
    filas = {}
    with open(ruta, encoding="utf-8", newline="") as archivo:
        for fila in csv.DictReader(archivo):
            if fila["method"] != "triangulation":
                continue
            hallado = ID_FINAL.match(fila["experiment"])
            if hallado is None:
                raise ValueError(f"{ruta}: id inesperado {fila['experiment']}")
            if int(fila["n_total"]) != OBJETOS_TOTALES:
                raise ValueError(f"{ruta}: n_total = {fila['n_total']}")
            valores = {k: numero(v) for k, v in fila.items()
                       if k not in ("experiment", "method")}
            tp = [valores[f"tp_{t}"] for t, _ in UMBRALES if valores.get(f"tp_{t}") is not None]
            if tp:
                valores["tp_mean"] = sum(tp) / len(tp)
            filas[(hallado.group(2), int(fila["n_views"]))] = valores
    return filas


def cargar(carpeta: str, tipo: str) -> dict[tuple[str, int], dict[str, float | None]]:
    """ Lee el CSV final de una carpeta de resultados_finales/.

    Args:
        * carpeta: subcarpeta de resultados_finales/
        * tipo: 'axis_sym' o 'plane_sym'

    Returns:
        * dict[tuple[str, int], dict[str, float | None]]: fila por (prompt, vistas)

    """
    rutas = sorted((RESULTADOS / carpeta).glob(f"experiments_*/{tipo}_final_comparison.csv"))
    if len(rutas) != 1:
        raise ValueError(f"{carpeta}/{tipo}: se esperaba un CSV final, hay {len(rutas)}")
    return filas_de_csv(rutas[0])


def celdas_columna(valores: list[float | None], columna: str) -> list[str]:
    """ Formatea una columna y pone en negrita el mejor valor y sus empates.

    Args:
        * valores: valores de la métrica; None es un dato faltante
        * columna: clave de PROPIEDADES

    Returns:
        * list[str]: celdas LaTeX, con '--' en los valores faltantes

    """
    mayor, formato = PROPIEDADES[columna]
    validos = [v for v in valores if v is not None]
    mejor = None
    if mayor is not None and len(validos) > 1:
        mejor = formato.format(max(validos) if mayor else min(validos))
    celdas = []
    for valor in valores:
        if valor is None:
            celdas.append("--")
        elif formato.format(valor) == mejor:
            celdas.append(rf"\textbf{{{mejor}}}")
        else:
            celdas.append(formato.format(valor))
    return celdas


def apilar(encabezado: str) -> str:
    """ Convierte 'línea 1|línea 2' en una celda de encabezado de dos líneas. """
    partes = encabezado.split("|")
    if len(partes) == 1:
        return encabezado
    return r"\begin{tabular}[b]{@{}c@{}}" + r" \\ ".join(partes) + r"\end{tabular}"


def encabezado_dos_niveles(fijas: list[str], grupos: list[tuple[str, list[str]]]) -> list[str]:
    """ Construye un encabezado con grupos (\\multicolumn + \\cmidrule) y subcolumnas.

    Args:
        * fijas: encabezados de las columnas iniciales, que ocupan las dos filas
        * grupos: [(título del grupo, [subencabezados])]; título vacío = sin grupo

    Returns:
        * list[str]: líneas LaTeX del encabezado

    """
    fila1 = [rf"\multirow{{2}}{{*}}{{{f}}}" for f in fijas]
    fila2, reglas = [""] * len(fijas), []
    columna = len(fijas) + 1
    for titulo, subs in grupos:
        n = len(subs)
        if titulo:
            fila1.append(rf"\multicolumn{{{n}}}{{c}}{{{titulo}}}")
            reglas.append(rf"\cmidrule(lr){{{columna}-{columna + n - 1}}}")
            fila2 += subs
        else:
            fila1 += [rf"\multirow{{2}}{{*}}{{{s}}}" for s in subs]
            fila2 += [""] * n
        columna += n
    return [" & ".join(fila1) + r" \\", " ".join(reglas), " & ".join(fila2) + r" \\"]


def envolver(encabezado: list[str], cuerpo: list[str], alineacion: str, titulo: str,
             etiqueta: str, nota: str) -> str:
    """ Envuelve una tabla con el formato de cap4.tex: lrbox, adjustbox y nota al pie.

    Args:
        * encabezado: líneas del encabezado
        * cuerpo: filas de datos
        * alineacion: columnas del tabular
        * titulo: texto del título, sin 'Noviembre'
        * etiqueta: etiqueta sin el prefijo 'tab:nov_'
        * nota: texto de la nota, sin 'Nota:'

    Returns:
        * str: bloque \\begin{table}...\\end{table}

    """
    return "\n".join([
        r"\begin{table}[htbp]", r"\centering", rf"\caption{{Noviembre: {titulo}.}}",
        rf"\label{{tab:nov_{etiqueta}}}", r"\begin{lrbox}{\tablebox}",
        r"\begin{adjustbox}{max width=\textwidth}", rf"\begin{{tabular}}{{{alineacion}}}",
        r"\toprule", *encabezado, r"\midrule", *cuerpo, r"\bottomrule", r"\end{tabular}",
        r"\end{adjustbox}", r"\end{lrbox}", r"\usebox{\tablebox}", "", r"\vspace{1.5ex}",
        r"\begin{minipage}{\wd\tablebox}", r"\footnotesize", f"Nota: {nota}", r"\end{minipage}",
        r"\end{table}",
    ])


def tabla_bloque(filas: list[tuple[str, int, dict | None]], bloque: str, titulo: str,
                 etiqueta: str, nota: str, primera_columna: str = "Prompt") -> str:
    """ Tabla con una fila por (prompt, vistas) y las columnas de un bloque de métricas.

    Args:
        * filas: (nombre LaTeX, vistas, valores), con valores None si no hubo predicción
        * bloque: clave de BLOQUES
        * titulo: título sin 'Noviembre'
        * etiqueta: etiqueta sin prefijo
        * nota: nota al pie
        * primera_columna: encabezado de la primera columna

    Returns:
        * str: tabla LaTeX

    """
    _, grupos = BLOQUES[bloque]
    metricas = [c for _, columnas in grupos for c, _ in columnas]
    celdas = {}
    for metrica in metricas:
        valores = [0.0 if v is None and metrica == "n_objects"
                   else (v or {}).get(metrica) for _, _, v in filas]
        celdas[metrica] = celdas_columna(valores, metrica)

    cuerpo, previo = [], None
    for i, (nombre, n_views, _) in enumerate(filas):
        nuevo = nombre != previo
        if nuevo and previo is not None:
            cuerpo.append(r"\midrule")
        n_filas = sum(1 for otro, _, _ in filas if otro == nombre)
        fila = [rf"\multirow{{{n_filas}}}{{*}}{{{nombre}}}" if nuevo else "", str(n_views)]
        cuerpo.append(" & ".join(fila + [celdas[m][i] for m in metricas]) + r" \\")
        previo = nombre

    encabezado = encabezado_dos_niveles([primera_columna, "Vistas"],
                                        [(t, [s for _, s in cols]) for t, cols in grupos])
    return envolver(encabezado, cuerpo, "lr" + "r" * len(metricas), titulo, etiqueta, nota)


def ordenar_con_tolerancia(filas: list[dict], criterios: list[tuple]) -> tuple[list[dict], list]:
    """ Ordena configuraciones con desempate lexicográfico con tolerancia.

    Replica ordenar_con_tolerancia de generar_tablas_latex.py sin depender de pandas.

    Args:
        * filas: una fila por configuración
        * criterios: lista de CRITERIOS[tipo]

    Returns:
        * tuple[list[dict], list]: filas ordenadas y número del criterio que resolvió cada puesto

    """
    def peor(valor: float | None, mayor: bool) -> float:
        """ Reemplaza un dato faltante por el peor valor posible. """
        if valor is not None:
            return valor
        return -1e18 if mayor else 1e18

    restantes = sorted(filas, key=lambda f: tuple(
        -peor(f.get(c[0]), c[2]) if c[2] else peor(f.get(c[0]), c[2]) for c in criterios))
    orden, decisivo = [], []
    while restantes:
        grupo, nivel = restantes, None
        if len(restantes) > 1:
            for k, (columna, _, mayor, brecha) in enumerate(criterios, 1):
                valores = [peor(f.get(columna), mayor) for f in grupo]
                mejor = max(valores) if mayor else min(valores)
                diferencias = [(mejor - v) if mayor else (v - mejor) for v in valores]
                grupo = [f for f, d in zip(grupo, diferencias)
                         if (d < brecha if brecha > 0 else d <= 0)]
                nivel = k
                if len(grupo) == 1:
                    break
        elegido = grupo[0]
        orden.append(elegido)
        decisivo.append(nivel)
        restantes = [f for f in restantes if f is not elegido]
    return orden, decisivo


def tabla_ranking(filas: dict[tuple[str, int], dict], tipo: str, titulo: str, etiqueta: str,
                  nota: str, top: int = TOP) -> str:
    """ Ranking de las mejores combinaciones de prompt y número de vistas.

    Args:
        * filas: fila por (prompt, vistas)
        * tipo: 'axis_sym' o 'plane_sym'
        * titulo: título sin 'Noviembre'
        * etiqueta: etiqueta sin prefijo
        * nota: nota al pie
        * top: número de filas mostradas

    Returns:
        * str: tabla LaTeX

    """
    criterios = CRITERIOS[tipo]
    lista = [dict(valores, prompt=p, vistas=v) for (p, v), valores in filas.items()]
    orden, decisivo = ordenar_con_tolerancia(lista, criterios)
    orden, decisivo = orden[:top], decisivo[:top]
    columnas = [(c[0], c[1]) for c in criterios] + INFORMATIVAS[tipo]
    celdas = {c: celdas_columna([f.get(c) for f in orden], c) for c, _ in columnas}

    cuerpo = []
    for i, fila in enumerate(orden):
        if i and i % 5 == 0:
            cuerpo.append(r"\addlinespace")
        inicio = [str(i + 1), rf"\texttt{{{escapar(fila['prompt'])}}}", str(fila["vistas"]),
                  str(decisivo[i]) if decisivo[i] else "--"]
        cuerpo.append(" & ".join(inicio + [celdas[c][i] for c, _ in columnas]) + r" \\")
    encabezado = [" & ".join([r"\#", "Prompt", "Vistas", "Crit."]
                             + [apilar(h) for _, h in columnas]) + r" \\"]
    return envolver(encabezado, cuerpo, "rlrc" + "r" * len(columnas), titulo, etiqueta, nota)


def tabla_comparativa(grupos: list[tuple[str, dict]], tipo: str, encabezado_grupo: str,
                      titulo: str, etiqueta: str, nota: str,
                      columnas: list[tuple[str, str]] | None = None) -> str:
    """ Tabla por número de vistas con una fila por condición comparada.

    Args:
        * grupos: (nombre LaTeX de la condición, fila por número de vistas)
        * tipo: 'axis_sym' o 'plane_sym'
        * encabezado_grupo: encabezado de la columna de condición
        * titulo: título sin 'Noviembre'
        * etiqueta: etiqueta sin prefijo
        * nota: nota al pie
        * columnas: columnas; por defecto, COMPARATIVAS[tipo]

    Returns:
        * str: tabla LaTeX

    """
    columnas = columnas or COMPARATIVAS[tipo]
    planas = [(k, nombre, por_vistas.get(v)) for k, v in enumerate(VISTAS)
              for nombre, por_vistas in grupos]
    celdas = {c: celdas_columna([(f or {}).get(c) for _, _, f in planas], c) for c, _ in columnas}
    cuerpo = []
    for i, (k, nombre, _) in enumerate(planas):
        j = i % len(grupos)
        if j == 0 and i:
            cuerpo.append(r"\midrule")
        vistas = rf"\multirow{{{len(grupos)}}}{{*}}{{{VISTAS[k]}}}" if j == 0 else ""
        cuerpo.append(" & ".join([vistas, nombre] + [celdas[c][i] for c, _ in columnas]) + r" \\")
    encabezado = [" & ".join(["Vistas", encabezado_grupo] + [apilar(h) for _, h in columnas])
                  + r" \\"]
    return envolver(encabezado, cuerpo, "ll" + "r" * len(columnas), titulo, etiqueta, nota)


def filas_bloque(datos: dict[tuple[str, int], dict], prompts: list[str]) -> list[tuple]:
    """ Convierte los datos de varios prompts en filas de tabla_bloque.

    Args:
        * datos: fila por (prompt, vistas)
        * prompts: prompts en el orden de la tabla

    Returns:
        * list[tuple]: (nombre LaTeX, vistas, valores o None)

    """
    return [(rf"\texttt{{{escapar(p)}}}", v, datos.get((p, v))) for p in prompts for v in VISTAS]


def nota_bloque(tipo: str, bloque: str, extra: str = "") -> str:
    """ Compone la nota de una tabla por bloque de métricas.

    Args:
        * tipo: 'axis_sym' o 'plane_sym'
        * bloque: clave de BLOQUES
        * extra: texto antes de la nota del bloque

    Returns:
        * str: nota al pie

    """
    negrita = "Negrita: mejor valor de la columna. "
    propia = {"angular": negrita + NOTA_ANGULAR, "posicion": negrita + NOTA_POSICION,
              "f1": NOTA_F1, "conteos": NOTA_CONTEOS}[bloque]
    return f"{METODO[tipo]} {extra}{propia}"


def por_vistas(datos: dict[tuple[str, int], dict], prompt: str) -> dict[int, dict]:
    """ Extrae las filas de un prompt, indexadas por número de vistas. """
    return {v: datos[(p, v)] for p, v in datos if p == prompt}


def tablas_tipo(tipo: str) -> list[tuple[str, str]]:
    """ Genera en el orden de cap4.tex todas las tablas de un tipo de simetría.

    Args:
        * tipo: 'axis_sym' o 'plane_sym'

    Returns:
        * list[tuple[str, str]]: (sección, tabla LaTeX)

    """
    corto, nombre = CORTO_TIPO[tipo], TITULO_TIPO[tipo]
    bloques_tipo = ["angular", "posicion"] + (["f1"] if tipo == "plane_sym" else [])
    flujo_a = cargar("ndcfix_flujoA", tipo)
    v08 = cargar("ndcfix_v08", tipo)
    todos = {**flujo_a, **v08}
    config_224 = configuracion(224, "flat", MULTI)
    seleccion = SELECCION[tipo]
    prompt = seleccion["prompt"]
    tablas = []

    secciones = [("Flujo A: Base", "base", BASE, "prompts base del Flujo A", ""),
                 ("Flujo A: Variantes", "variantes_1", VARIANTES,
                  r"variantes \texttt{\_1} de los prompts del Flujo A", NOTA_VARIANTE),
                 ("Flujo A: Adicionales", "adicionales", ADICIONALES,
                  "prompts adicionales del Flujo A", "")]
    for seccion, clave, prompts, descripcion, extra in secciones:
        for bloque in bloques_tipo:
            titulo = f"{BLOQUES[bloque][0]} de los {descripcion} en {nombre}, {config_224}"
            tablas.append((seccion, tabla_bloque(
                filas_bloque(todos, prompts), bloque, titulo, f"{corto}_{clave}_{bloque}",
                nota_bloque(tipo, bloque, extra))))

    if tipo == "axis_sym":
        nota_top = (r"Crit.: número del criterio que resolvió la posición. " + NOTA_ANGULAR
                    + r" La traslación y SDE usan solo los objetos con predicción. Objetos: "
                    r"cobertura, de 850.")
    else:
        nota_top = (r"$F1_{\mathrm{ref}}$ húngaro y TP se muestran solo como información. Crit.: "
                    r"número del criterio que resolvió la posición. TP (prom.): aciertos promedio "
                    r"sobre los cuatro umbrales $\tau$. " + NOTA_ANGULAR
                    + r" SDE$_{\mathrm{ref}}$ y $F1_{\mathrm{ref}}$ usan solo los objetos con "
                    r"predicción.")
    tablas.append(("Flujo A: Ranking", tabla_ranking(
        todos, tipo,
        f"Ranking de desempeño de prompt por número de vistas en el Flujo A en {nombre}, "
        f"{config_224}", f"{corto}_top_224", f"{METODO[tipo]} {nota_top}")))

    resolucion = {448: (cargar("ndcfix_448", tipo), seleccion["resolucion_448"]),
                  1136: (cargar("ndcfix_1136", tipo), [prompt])}
    bloques_resolucion = bloques_tipo + (["conteos"] if tipo == "plane_sym" else [])
    for bloque in bloques_resolucion:
        for size, (datos, prompts) in resolucion.items():
            filas = filas_bloque(datos, prompts)
            faltantes = sorted({f[1] for f in filas if f[2] is None})
            filas = [f for f in filas if f[2] is not None]
            sin_datos = (NOTA_SIN_DATOS.format(", ".join(map(str, faltantes)))
                         if faltantes else "")
            titulo = (f"{BLOQUES[bloque][0]} de los prompts seleccionados en {nombre}, "
                      f"{configuracion(size, 'flat', MULTI)}")
            tablas.append(("Flujo A: Experimento de resolución", tabla_bloque(
                filas, bloque, titulo, f"{corto}_{size}_{bloque}",
                nota_bloque(tipo, bloque) + sin_datos)))

    condiciones = [(r"\textit{flat}", por_vistas(flujo_a, prompt)),
                   (r"\textit{darker}", por_vistas(cargar("ndcfix_224_darker", tipo), prompt)),
                   (r"\textit{brighter}", por_vistas(cargar("ndcfix_224_brighter", tipo), prompt))]
    tablas.append(("Flujo A: Experimento de iluminación", tabla_comparativa(
        condiciones, tipo, "Iluminación",
        rf"Efecto de la iluminación en {nombre} con el \textit{{prompt}} "
        rf"\texttt{{{escapar(prompt)}}}, resolución 224px, iluminación \textit{{flat}}, "
        rf"\textit{{darker}} y \textit{{brighter}}, {MULTI}",
        f"{corto}_iluminacion", f"{METODO[tipo]} {NOTA_ILUMINACION}{NOTA_COMPARATIVA[tipo]}")))

    flujos = [("A", por_vistas(flujo_a, prompt)),
              ("B", por_vistas(cargar("ndcfix_flujoB", tipo), seleccion["flujo_b"])),
              ("C", por_vistas(cargar("ndcfix_flujoC", tipo), seleccion["flujo_c"]))]
    prompt_tex = rf"\texttt{{{escapar(prompt)}}}"
    nota_flujos = (rf"Flujo A: \textit{{prompt}} {prompt_tex} sin contexto. Flujo B: el mismo "
                   r"\textit{prompt} precedido por una descripción del objeto. Flujo C: "
                   r"\textit{prompt} genérico construido a partir de la descripción y de una pista "
                   rf"de ubicación {seleccion['traza']}; no depende del \textit{{prompt}} base, "
                   rf"que solo se usa como respaldo (corrida con "
                   rf"\texttt{{{seleccion['respaldo_c']}}} "
                   r"como respaldo). ")
    tablas.append(("Flujos B y C", tabla_comparativa(
        flujos, tipo, "Flujo",
        rf"Comparación de los Flujos A, B y C en {nombre} con el \textit{{prompt}} {prompt_tex}, "
        f"{config_224}", f"{corto}_flujos",
        f"{METODO[tipo]} {nota_flujos}{NOTA_COMPARATIVA[tipo]}")))

    individual = cargar("ndcfix_single_oct26", tipo)
    filas_modo = (
        [(rf"{prompt_tex} (\textit{{multi}})", v, flujo_a.get((prompt, v))) for v in VISTAS]
        + [(rf"{prompt_tex} (\textit{{single}})", v, individual.get((f"{prompt}_single_oct26", v)))
           for v in VISTAS])
    nota_modo = (r"\textit{multi}: las $n_v$ vistas se envían en una sola llamada, con el "
                 r"\textit{prompt} multivista. \textit{single}: cada vista se envía por separado, "
                 r"con el \textit{prompt} de imagen única, y los puntos se reúnen antes de la "
                 r"triangulación. ")
    for bloque in bloques_resolucion:
        titulo = (f"{BLOQUES[bloque][0]} en {nombre} con el \\textit{{prompt}} {prompt_tex}, "
                  r"resolución 224px, iluminación \textit{flat}, envío multiimagen "
                  r"(\textit{multi}) frente a individual (\textit{single})")
        tablas.append(("Flujo A: Envío individual vs Multiimagen", tabla_bloque(
            filas_modo, bloque, titulo, f"{corto}_modo_envio_{bloque}",
            nota_bloque(tipo, bloque, nota_modo), "Prompt y envío")))

    columnas_vlm = [(c, h.replace(r" \\ ", "|").replace(r"\begin{tabular}[b]{@{}c@{}}", "")
                     .replace(r"\end{tabular}", "")) for c, h, _, _ in COLUMNAS_VLM[tipo]]
    modelos = [("Molmo2-8B", por_vistas(individual, f"{prompt}_single_oct26")),
               ("MolmoPoint-8B", por_vistas(cargar("molmopoint", tipo), f"{prompt}_molmopoint")),
               ("Qwen3-VL-8B", por_vistas(cargar("qwen3vl", tipo), f"{prompt}_qwen3vl"))]
    tablas.append(("Comparación de modelos", tabla_comparativa(
        modelos, tipo, "Modelo",
        rf"Comparación de VLMs en {nombre} con el \textit{{prompt}} {prompt_tex}, "
        f"{configuracion(224, 'flat', SINGLE)}", f"{corto}_vlms",
        f"{METODO[tipo]} {NOTA_ENVIO}{NOTA_COMPARATIVA[tipo]}", columnas_vlm)))
    return tablas


def main() -> None:
    """ Escribe todas las tablas de Noviembre en un único .tex. """
    parser = argparse.ArgumentParser(description="Tablas LaTeX desde resultados_finales/.")
    parser.add_argument("--salida", type=Path, default=SALIDA, help="archivo .tex de salida")
    args = parser.parse_args()

    lineas = ["% Generado por utils/generar_tablas_noviembre.py desde resultados_finales/.",
              "% Requiere booktabs, multirow, adjustbox y \\newsavebox{\\tablebox}.", ""]
    total = 0
    for tipo in ("axis_sym", "plane_sym"):
        lineas += [f"% {'=' * 70}", f"% {TITULO_TIPO[tipo].capitalize()}", f"% {'=' * 70}", ""]
        previa = None
        for seccion, tabla in tablas_tipo(tipo):
            if seccion != previa:
                lineas += [f"% --- {seccion}", ""]
                previa = seccion
            lineas += [tabla, ""]
            total += 1
            if total % TABLAS_POR_PAGINA == 0:
                lineas += [r"\clearpage", ""]
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text("\n".join(lineas), encoding="utf-8")
    print(f"Escrito {args.salida} ({total} tablas)")


if __name__ == "__main__":
    main()
