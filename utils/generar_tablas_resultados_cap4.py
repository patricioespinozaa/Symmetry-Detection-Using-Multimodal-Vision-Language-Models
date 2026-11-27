#!/usr/bin/env python3
# Genera las tablas del capítulo de Resultados (cap4.tex, esqueleto "Resultados") con los
# resultados finales del pipeline sin malla: eje con EXP-E y plano con S_dat, un plano por objeto,
# todos posteriores a la corrección NDC. Escribe un .tex por tabla en Tesis_DCC_MDS/tablas_resultados/
# y un registro JSON con la procedencia de cada celda (CSV, fila y columna, o la fórmula de las
# celdas derivadas). utils/verificar_tablas_resultados_cap4.py vuelve a leer las fuentes y comprueba
# cada celda.
#
#   python utils/generar_tablas_resultados_cap4.py
#
# Solo usa la biblioteca estándar. Requiere en el preámbulo: booktabs, multirow, adjustbox y
# \newsavebox{\tablebox}. No modifica cap4.tex.
import ast
import csv
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "resultados_finales"
SALIDA = REPO / "Tesis_DCC_MDS" / "tablas_resultados"
REGISTRO = SALIDA / "registro_celdas.json"
AZAR_CSV = REPO / "RandomBaseline" / "outputs" / "random_baseline_reference.csv"
TRIVIALES_LOG = RES / "logs" / "triviales.log"
AGUIRRE_CSV = RES / "referencias" / "aguirre2026_publicado.csv"   # valores publicados (Tabla 1)
VISTAS = [6, 14, 26]
TOTAL = 850
FECHA_MINIMA = (2026, 11, 5)          # recálculo completo tras la corrección NDC (5/10/2026)
ID_FINAL = re.compile(r"^(axis|plane)_(.+)_s(\d+)_([a-z]+)_ndcfix_(expE|sdat_mp1)_nomesh$")
# Constantes del azar analítico para una dirección uniforme (Sección sm_azar del Cap. 3)
AZAR_MEDIA, AZAR_DESV = 57.2958, 21.5597
FORMATOS = {"auc": "{:.3f}", "grados": "{:.1f}", "prec": "{:.3f}", "f1": "{:.3f}",
            "obj": "{:.0f}", "sde": "{:.4f}", "entero": "{:.0f}", "z": "{:.1f}", "sde4": "{:.2f}"}
TIPOS = {"axis_sym": "eje", "plane_sym": "plano"}
METODO = {"axis_sym": "El eje se estima con EXP-E.",
          "plane_sym": r"La normal se elige con $S_{\mathrm{dat}}$ y se predice un plano por objeto."}
NOTA_ANG = r"Las métricas angulares usan los 850 objetos e imputan $90^\circ$ a los objetos sin predicción."
NOTA_F1 = r"$F1$ y SDE usan solo los objetos con predicción."
NEGRITA = "Negrita: mejor valor de la columna."
REF = r"224~px, iluminación \textit{flat}, envío multiimagen y Flujo~A"

registro: dict[str, list] = {}


def rel(ruta: Path) -> str:
    """ Ruta relativa al repositorio, con barras normales. """
    return ruta.relative_to(REPO).as_posix()


def csv_final(carpeta: str, tipo: str) -> Path:
    """ CSV final de una carpeta, comprobando que sea único y posterior al recálculo NDC. """
    rutas = sorted((RES / carpeta).glob(f"experiments_*/{tipo}_final_comparison.csv"))
    if len(rutas) != 1:
        raise ValueError(f"{carpeta}/{tipo}: se esperaba un CSV final, hay {len(rutas)}")
    d, m, a = map(int, re.search(r"experiments_(\d+)_(\d+)_(\d+)", str(rutas[0])).groups())
    if (a, m, d) < FECHA_MINIMA:
        raise ValueError(f"{rutas[0]}: anterior al recálculo con la corrección NDC")
    return rutas[0]


def leer_final(ruta: Path) -> dict[tuple[str, int], dict]:
    """ Filas del CSV final por (id, vistas), validando id, estimador y n_total. """
    filas = {}
    with open(ruta, encoding="utf-8", newline="") as f:
        for fila in csv.DictReader(f):
            if fila["method"] != "triangulation":
                continue
            if not ID_FINAL.match(fila["experiment"]):
                raise ValueError(f"{ruta}: id fuera de la versión final {fila['experiment']}")
            if int(fila["n_total"]) != TOTAL:
                raise ValueError(f"{ruta}: n_total {fila['n_total']}")
            filas[(fila["experiment"], int(fila["n_views"]))] = fila
    return filas


_cache: dict[Path, dict] = {}


def celda(carpeta: str, tipo: str, exp_id: str, vistas: int, columna: str) -> dict | None:
    """ Celda de un CSV final, con su procedencia; None si esa combinación no existe. """
    ruta = csv_final(carpeta, tipo)
    datos = _cache.setdefault(ruta, leer_final(ruta))
    fila = datos.get((exp_id, vistas))
    if fila is None or fila.get(columna) in (None, ""):
        return None
    return {"v": float(fila[columna]), "t": {"fuente": "final", "csv": rel(ruta), "experiment": exp_id,
                                             "n_views": vistas, "columna": columna}}


def celda_pareado(ruta: Path, vistas: int, columna: str) -> dict:
    """ Celda de un CSV pareado (compare_plane_scores.py). """
    with open(ruta, encoding="utf-8", newline="") as f:
        fila = next(r for r in csv.DictReader(f) if int(r["n_views"]) == vistas)
    return {"v": float(fila[columna]), "t": {"fuente": "pareado", "csv": rel(ruta),
                                             "n_views": vistas, "columna": columna}}


def celda_centro(tipo: str, exp_id: str, vistas: int) -> dict:
    """ Fracción de puntos en la columna central en vistas no cenitales (diagnose_center_bias.py). """
    ruta = RES / "diagnosticos" / f"centro_{TIPOS[tipo]}_resumen.csv"
    with open(ruta, encoding="utf-8", newline="") as f:
        fila = next(r for r in csv.DictReader(f)
                    if r["experiment_id"] == exp_id and int(r["n_views"]) == vistas)
    columna = "pct_noncenital_centered"
    return {"v": float(fila[columna]), "t": {"fuente": "centro", "csv": rel(ruta), "experiment_id": exp_id,
                                             "n_views": vistas, "columna": columna}}


ID_MALLA = re.compile(r"^(axis|plane)_(.+)_s(\d+)_([a-z]+)_ndcfix_mesh(_[a-z0-9_]+)?$")


def leer_malla(ruta: Path) -> dict[tuple[str, str, int], dict]:
    """ Filas del CSV final con malla por (id, método, vistas), validando id y n_total. """
    filas = {}
    with open(ruta, encoding="utf-8", newline="") as f:
        for fila in csv.DictReader(f):
            if not ID_MALLA.match(fila["experiment"]):
                raise ValueError(f"{ruta}: id con malla inesperado {fila['experiment']}")
            if int(fila["n_total"]) != TOTAL:
                raise ValueError(f"{ruta}: n_total {fila['n_total']}")
            filas[(fila["experiment"], fila["method"], int(fila["n_views"]))] = fila
    return filas


def celda_malla(carpeta: str, tipo: str, exp_id: str, metodo: str, vistas: int, columna: str) -> dict:
    """ Celda de un CSV final del pipeline con malla (run_axis_malla.py / run_plane_malla.py). """
    ruta = csv_final(carpeta, tipo)
    datos = _cache.setdefault(ruta, leer_malla(ruta))
    fila = datos[(exp_id, metodo, vistas)]
    return {"v": float(fila[columna]), "t": {"fuente": "malla", "csv": rel(ruta), "experiment": exp_id,
                                             "method": metodo, "n_views": vistas, "columna": columna}}


def celda_pareado_malla(ruta: Path, variante: str, metodo: str, vistas: int, columna: str) -> dict:
    """ Celda del pareado con malla frente a sin malla, por variante, método y vistas. """
    with open(ruta, encoding="utf-8", newline="") as f:
        fila = next(r for r in csv.DictReader(f) if r["variante"] == variante
                    and r["metodo"] == metodo and int(r["n_views"]) == vistas)
    return {"v": float(fila[columna]), "t": {"fuente": "pareado_malla", "csv": rel(ruta),
                                             "variante": variante, "metodo": metodo,
                                             "n_views": vistas, "columna": columna}}


def celda_azar(tipo: str, metrica: str, columna: str) -> dict:
    """ Celda de RandomBaseline/outputs/random_baseline_reference.csv con cobertura 1.0. """
    with open(AZAR_CSV, encoding="utf-8", newline="") as f:
        fila = next(r for r in csv.DictReader(f) if r["symmetry_type"] == tipo
                    and float(r["coverage"]) == 1.0 and r["metric"] == metrica)
    return {"v": float(fila[columna]), "t": {"fuente": "azar", "csv": rel(AZAR_CSV), "tipo": tipo,
                                             "metrica": metrica, "columna": columna}}


def triviales() -> dict[str, list[dict]]:
    """ Predictores triviales por tipo, leídos del registro de trivial_baselines.py. """
    salida, tipo = {}, None
    for linea in TRIVIALES_LOG.read_text(encoding="utf-8").splitlines():
        m = re.match(r"=== (axis_sym|plane_sym):", linea)
        if m:
            tipo = m.group(1)
            salida[tipo] = []
        elif tipo and linea.strip().startswith("{'predictor'"):
            salida[tipo].append(ast.literal_eval(linea.strip()))
    return salida


def celda_trivial(tipo: str, predictor: str, clave: str) -> dict:
    """ Celda de un predictor trivial, con su procedencia. """
    fila = next(p for p in triviales()[tipo] if p["predictor"] == predictor)
    return {"v": float(fila[clave]), "t": {"fuente": "log", "archivo": rel(TRIVIALES_LOG),
                                           "tipo": tipo, "predictor": predictor, "clave": clave}}


def objetos_trivial(tipo: str) -> dict:
    """ Objetos evaluados por los predictores triviales, del encabezado del registro. """
    m = re.search(rf"=== {tipo}: (\d+) objects", TRIVIALES_LOG.read_text(encoding="utf-8"))
    return {"v": float(m.group(1)), "t": {"fuente": "log_objetos", "archivo": rel(TRIVIALES_LOG),
                                          "tipo": tipo}}


def celda_publicada(tipo: str, configuracion: str, vistas: int, columna: str) -> dict:
    """ Valor publicado por Aguirre y Sipiran, transcrito en resultados_finales/referencias/. """
    with open(AGUIRRE_CSV, encoding="utf-8", newline="") as f:
        fila = next(r for r in csv.DictReader(f) if r["symmetry_type"] == tipo
                    and r["configuracion"] == configuracion and int(r["n_views"]) == vistas)
    return {"v": float(fila[columna]), "t": {"fuente": "publicado", "csv": rel(AGUIRRE_CSV), "tipo": tipo,
                                             "configuracion": configuracion, "n_views": vistas,
                                             "columna": columna}}


def derivada(valor: float, formula: str, entradas: list[dict]) -> dict:
    """ Celda calculada a partir de otras; el verificador repite el cálculo. """
    return {"v": valor, "t": {"fuente": "derivada", "formula": formula,
                              "entradas": [e["t"] for e in entradas]}}


def media(celdas: list[dict | None]) -> dict | None:
    """ Media de las celdas presentes (None si no hay ninguna). """
    validas = [c for c in celdas if c is not None]
    if not validas:
        return None
    return derivada(sum(c["v"] for c in validas) / len(validas), "media", validas)


def formatear(c: dict | None, formato: str) -> str:
    """ Formatea una celda; un valor que redondea a cero se escribe sin signo ('-0.0' -> '0.0'). """
    if c is None:
        return "--"
    texto = FORMATOS[formato].format(c["v"])
    return texto.lstrip("-") if float(texto) == 0 else texto


def columna_celdas(celdas: list[dict | None], formato: str, mayor: bool | None) -> list[str]:
    """ Formatea una columna y pone en negrita el mejor valor (y sus empates). """
    textos = [formatear(c, formato) for c in celdas]
    validos = [c["v"] for c in celdas if c is not None]
    if mayor is None or len(validos) < 2:
        return textos
    mejor = FORMATOS[formato].format(max(validos) if mayor else min(validos))
    return [rf"\textbf{{{t}}}" if t == mejor else t for t in textos]


def tabla(nombre: str, titulo: str, encabezado: list[str], filas: list[tuple[list[str], list]],
          columnas: list[tuple[str, bool | None]], nota: str, separar: list[int] | None = None,
          separar_fino: list[int] | None = None, alto: str | None = None) -> str:
    """ Escribe la tabla y registra la procedencia de cada celda numérica.

    Args:
        * nombre: archivo (sin .tex) y etiqueta tab:res_<nombre>
        * titulo: título de la tabla
        * encabezado: líneas del encabezado
        * filas: (celdas de texto iniciales, celdas numéricas)
        * columnas: (formato, mayor_es_mejor) de cada columna numérica
        * nota: nota al pie
        * separar: índices de fila antes de los que va un \\midrule
        * separar_fino: índices de fila antes de los que va un \\cmidrule de todo el ancho
        * alto: alto máximo de la tabla (p. ej. 0.8\\textheight), para tablas largas

    Una fila sin celdas numéricas (lista vacía) se escribe tal cual, como fila de título.

    """
    por_columna = [[f[1][j] if f[1] else None for f in filas] for j in range(len(columnas))]
    textos = [columna_celdas(cs, fmt, mayor) for cs, (fmt, mayor) in zip(por_columna, columnas)]
    cuerpo, entradas = [], []
    for i, (inicio, celdas) in enumerate(filas):
        if separar and i in separar:
            cuerpo.append(r"\midrule")
        elif separar_fino and i in separar_fino:
            cuerpo.append(rf"\cmidrule{{1-{max(len(f[0]) for f in filas if f[1]) + len(columnas)}}}")
        if not celdas:
            cuerpo.append(" & ".join(inicio) + r" \\")
            continue
        cuerpo.append(" & ".join(inicio + [textos[j][i] + (celdas[j] or {}).get("sufijo", "")
                                           for j in range(len(columnas))]) + r" \\")
        for j, c in enumerate(celdas):
            if c is not None:
                entradas.append({"fila": i, "columna": j, "texto": formatear(c, columnas[j][0]),
                                 "formato": columnas[j][0], "traza": c["t"]})
    registro[nombre] = entradas
    fuentes = sorted({e["traza"].get("csv") or e["traza"].get("archivo") or "derivada"
                      for e in entradas})
    alineacion = "l" * max(len(f[0]) for f in filas if f[1]) + "r" * len(columnas)
    lineas = [f"% Generado por utils/generar_tablas_resultados_cap4.py. No editar a mano.",
              "% Fuentes:"] + [f"%   {f}" for f in fuentes] + [
        r"\begin{table}[htbp]", r"\centering", rf"\caption{{{titulo}}}",
        rf"\label{{tab:res_{nombre}}}", r"\begin{lrbox}{\tablebox}",
        r"\begin{adjustbox}{max width=\textwidth" + (f", max totalheight={alto}" if alto else "") + "}",
        rf"\begin{{tabular}}{{{alineacion}}}",
        r"\toprule", *encabezado, r"\midrule", *cuerpo, r"\bottomrule", r"\end{tabular}",
        r"\end{adjustbox}", r"\end{lrbox}", r"\usebox{\tablebox}", "", r"\vspace{1.5ex}",
        r"\begin{minipage}{\wd\tablebox}", r"\footnotesize", f"Nota: {nota}", r"\end{minipage}",
        r"\end{table}", ""]
    SALIDA.mkdir(parents=True, exist_ok=True)
    (SALIDA / f"{nombre}.tex").write_text("\n".join(lineas), encoding="utf-8", newline="\n")
    return nombre


def apilar(texto: str) -> str:
    partes = texto.split("|")
    return texto if len(partes) == 1 else (r"\begin{tabular}[b]{@{}c@{}}" + r" \\ ".join(partes)
                                           + r"\end{tabular}")


def tt(prompt: str) -> str:
    return rf"\texttt{{{prompt.replace('_', chr(92) + '_')}}}"


def ident(tipo: str, prompt: str, size: int = 224, luz: str = "flat") -> str:
    corto = "axis" if tipo == "axis_sym" else "plane"
    sufijo = "expE_nomesh" if tipo == "axis_sym" else "sdat_mp1_nomesh"
    return f"{corto}_{prompt}_s{size}_{luz}_ndcfix_{sufijo}"


# Columnas de las tablas comparativas por condición
COMP = {
    "axis_sym": [("n_objects", "Objetos", "obj", True), ("auc_angular", r"AUC$_{45^\circ}$", "auc", True),
                 ("angular_error_median", r"Mediana|($^\circ$)", "grados", False),
                 ("precision_5deg", r"P@5$^\circ$", "prec", True),
                 ("precision_10deg", r"P@10$^\circ$", "prec", True),
                 ("sde_ref_mean", r"SDE", "sde", False)],
    "plane_sym": [("n_objects", "Objetos", "obj", True), ("auc_angular", r"AUC$_{45^\circ}$", "auc", True),
                  ("angular_error_median", r"Mediana|($^\circ$)", "grados", False),
                  ("precision_10deg", r"P@10$^\circ$", "prec", True),
                  ("f1_ref", r"$F1$|voraz", "f1", True),
                  ("f1_ref_hungarian", r"$F1$|húngaro", "f1", True),
                  ("sde_ref_mean", r"SDE", "sde", False)],
}


def comparativa(nombre: str, tipo: str, condiciones: list[tuple[str, str, str]], encabezado_cond: str,
                titulo: str, nota: str) -> str:
    """ Tabla por número de vistas con una fila por condición: (nombre LaTeX, carpeta, id). """
    cols = COMP[tipo]
    filas, separar = [], []
    for k, v in enumerate(VISTAS):
        if k:
            separar.append(len(filas))
        for j, (etiqueta, carpeta, exp_id) in enumerate(condiciones):
            inicio = [rf"\multirow{{{len(condiciones)}}}{{*}}{{{v}}}" if j == 0 else "", etiqueta]
            filas.append((inicio, [celda(carpeta, tipo, exp_id, v, c) for c, _, _, _ in cols]))
    encabezado = [" & ".join(["Vistas", encabezado_cond] + [apilar(h) for _, h, _, _ in cols]) + r" \\"]
    if nombre.split("_")[0] in NOTA_CORTA:
        # Nota editada a mano en la tesis (sin estimador ni aviso de negritas); se conserva aquí.
        texto = NOTA_ANG + (f" {NOTA_F1}" if tipo == "plane_sym" else "") + (f" {nota}" if nota else "")
    else:
        texto = (f"{METODO[tipo]} {NOTA_ANG} {NOTA_F1 if tipo == 'plane_sym' else ''} "
                 + (f"{nota} " if nota else "") + NEGRITA)
    return tabla(nombre, titulo, encabezado, filas, [(f, m) for _, _, f, m in cols], texto, separar)


# Tablas comparativas cuya nota solo explica la población de cada métrica.
NOTA_CORTA = {"flujos", "envio", "resolucion", "iluminacion"}


# ----------------------------------------------------------------------------------------- tablas
def tabla_azar() -> str:
    filas = []
    metricas = [("angular_error_mean", r"Error angular medio ($^\circ$)", "grados"),
                ("angular_error_median", r"Error angular mediano ($^\circ$)", "grados"),
                ("auc_angular", r"AUC$_{45^\circ}$", "auc"), ("precision_5deg", r"P@5$^\circ$", "prec"),
                ("precision_10deg", r"P@10$^\circ$", "prec"), ("precision_15deg", r"P@15$^\circ$", "prec"),
                ("f1_ref", r"$F1$ voraz", "f1")]
    for metrica, texto, fmt in metricas:
        eje = None if metrica == "f1_ref" else celda_azar("axis_sym", metrica, "random_mean")
        plano = celda_azar("plane_sym", metrica, "random_mean")
        filas.append(([texto], [eje, plano], fmt))
    # columnas con formato propio por fila: se registran fila a fila
    entradas, cuerpo = [], []
    for i, (inicio, celdas, fmt) in enumerate(filas):
        cuerpo.append(" & ".join(inicio + [formatear(c, fmt) for c in celdas]) + r" \\")
        entradas += [{"fila": i, "columna": j, "texto": formatear(c, fmt), "formato": fmt,
                      "traza": c["t"]} for j, c in enumerate(celdas) if c is not None]
    encabezado = [r"Métrica & Eje & Plano \\"]
    nombre = "azar"
    registro[nombre] = entradas
    # Nota editada a mano en la tesis; se conserva aquí para que regenerar no la pise.
    nota = r"Predictor de la dirección del eje o la normal del plano uniforme en la esfera, sin mirar las imágenes."
    lineas = ["% Generado por utils/generar_tablas_resultados_cap4.py. No editar a mano.",
              f"% Fuentes:\n%   {rel(AZAR_CSV)}", r"\begin{table}[htbp]", r"\centering",
              r"\caption{Línea base aleatoria en eje y plano.}", r"\label{tab:res_azar}",
              r"\begin{lrbox}{\tablebox}", r"\begin{tabular}{lrr}", r"\toprule", *encabezado,
              r"\midrule", *cuerpo, r"\bottomrule", r"\end{tabular}", r"\end{lrbox}",
              r"\usebox{\tablebox}", "", r"\vspace{1.5ex}", r"\begin{minipage}{\wd\tablebox}",
              r"\footnotesize", f"Nota: {nota}", r"\end{minipage}", r"\end{table}", ""]
    (SALIDA / f"{nombre}.tex").write_text("\n".join(lineas), encoding="utf-8", newline="\n")
    return nombre


def tablas_triviales() -> list[str]:
    nombres = []
    # Encabezados, etiquetas y notas de estas dos tablas editados a mano en la tesis; se conservan aquí.
    predictores = [("uniform random direction (analytic)", "Dirección al azar"),
                   ("always X", r"Siempre $X$"), ("always Y", r"Siempre $Y$"), ("always Z", r"Siempre $Z$")]
    cols = [("angular_error_mean", r"\begin{tabular}[b]{@{}c@{}}Media ($^\circ$)\end{tabular}", "grados", False),
            ("angular_error_median", r"\begin{tabular}[b]{@{}c@{}}Mediana ($^\circ$)\end{tabular}", "grados", False),
            ("auc_angular", r"AUC$_{45^\circ}$", "auc", True), ("precision_5deg", r"P@5$^\circ$", "prec", True),
            ("precision_10deg", r"P@10$^\circ$", "prec", True), ("precision_15deg", r"P@15$^\circ$", "prec", True)]
    filas = [([texto], [celda_trivial("axis_sym", p, c) for c, _, _, _ in cols]) for p, texto in predictores]
    enc = [" & ".join(["Predictor"] + [apilar(h) for _, h, _, _ in cols]) + r" \\"]
    nombres.append(tabla("triviales_eje", "Predictores triviales en simetría axial.", enc, filas,
                         [(f, None) for _, _, f, _ in cols],
                         r"Predictores que siempre devuelven la misma dirección cartesiana, evaluados "
                         r"sobre los 850 objetos axiales. "))
    predictores = [("always {X}", r"Siempre $\{X\}$"), ("always {Y}", r"Siempre $\{Y\}$"),
                   ("always {Z}", r"Siempre $\{Z\}$"), ("always {X,Z}", r"Siempre $\{X,Z\}$"),
                   ("always {X,Y}", r"Siempre $\{X,Y\}$"), ("always {X,Y,Z}", r"Siempre $\{X,Y,Z\}$")]
    cols = [("n_planes_predicted", "Planos", "entero", None),
            ("recall_planes_mean", "Recall", "prec", True),
            ("precision_planes_mean", "Precisión", "prec", True),
            ("f1_ref", r"$F1$|voraz", "f1", True),
            ("f1_ref_hungarian", r"$F1$|húngaro", "f1", True)]
    filas = [([texto], [celda_trivial("plane_sym", p, c) for c, _, _, _ in cols]) for p, texto in predictores]
    enc = [" & ".join(["Predictor"] + [apilar(h) for _, h, _, _ in cols]) + r" \\"]
    nombres.append(tabla("triviales_plano", "Predictores triviales en simetría planar.", enc, filas,
                         [(f, m) for _, _, f, m in cols],
                         r"Predictores que siempre devuelven el mismo conjunto de planos cartesianos que "
                         r"pasan por el origen, evaluados sobre los 850 objetos planares. Recall y "
                         r"precisión con emparejamiento a $15^\circ$."))
    return nombres


PROMPTS = ["v00", "v01", "v02", "v03", "v04", "v05",
           "v00_1", "v01_1", "v02_1", "v03_1", "v04_1", "v05_1", "v06", "v07", "v08"]


def tabla_prompts(tipo: str) -> str:
    segunda = ("angular_error_median", r"Error mediano ($^\circ$)", "grados", False) \
        if tipo == "axis_sym" else ("f1_ref", r"$F1$ voraz", "f1", True)
    filas = []
    for p in PROMPTS:
        carpeta = "ndcfix_v08" if p == "v08" else "ndcfix_flujoA"
        auc = [celda(carpeta, tipo, ident(tipo, p), v, "auc_angular") for v in VISTAS]
        otra = [celda(carpeta, tipo, ident(tipo, p), v, segunda[0]) for v in VISTAS]
        objetos = [celda(carpeta, tipo, ident(tipo, p), v, "n_objects") for v in VISTAS]
        filas.append(([tt(p)], auc + [media(auc)] + otra + objetos))
    cols = [("auc", True)] * 4 + [(segunda[2], segunda[3])] * 3 + [("obj", None)] * 3
    enc = [rf"\multirow{{2}}{{*}}{{\textit{{Prompt}}}} & \multicolumn{{4}}{{c}}{{AUC$_{{45^\circ}}$}} & "
           rf"\multicolumn{{3}}{{c}}{{{segunda[1]}}} & \multicolumn{{3}}{{c}}{{Objetos}} \\",
           r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
           r" & $n_v{=}6$ & 14 & 26 & Prom. & $n_v{=}6$ & 14 & 26 & $n_v{=}6$ & 14 & 26 \\"]
    titulo = rf"Comparación de \textit{{prompts}} en simetría {'axial' if tipo == 'axis_sym' else 'planar'} ({REF})."
    # Nota editada a mano en la tesis; se conserva aquí para que regenerar no la pise.
    nota = ((r"Se predice un plano por objeto. $F1$ y $SDE$ usan solo los objetos con predicción. "
             if tipo == "plane_sym" else "")
            + r"Prom.: media de los tres valores de $n_v$. Objetos: objetos con predicción. "
            r"Base: \texttt{v00}--\texttt{v05}. Variantes: \texttt{\_1}. "
            r"Rondas posteriores: \texttt{v06}--\texttt{v08}.")
    return tabla(f"prompts_{TIPOS[tipo]}", titulo, enc, filas, cols, nota, separar=[6, 12])


SELECCION = {
    "axis_sym": [("v04", {224: ("ndcfix_flujoA",), 448: ("ndcfix_448",), 1136: ("ndcfix_1136",)}),
                 ("v06", {224: ("ndcfix_flujoA",), 448: ("ndcfix_448",)}),
                 ("v08", {224: ("ndcfix_v08",), 448: ("ndcfix_448",)})],
    "plane_sym": [("v00_1", {224: ("ndcfix_flujoA",), 448: ("plane_448px",), 1136: ("plane_1136px",)}),
                  ("v02", {224: ("ndcfix_flujoA",), 448: ("plane_448px",), 1136: ("plane_1136px",)}),
                  ("v04", {224: ("ndcfix_flujoA",), 448: ("plane_448px",)}),
                  ("v05_1", {224: ("ndcfix_flujoA",), 448: ("ndcfix_448",), 1136: ("ndcfix_1136",)}),
                  ("v06", {224: ("ndcfix_flujoA",), 448: ("ndcfix_448",)})],
}


# Prompts que muestran las tablas de selección (resumen y completa del anexo).
SELECCION_RESUMEN = {"axis_sym": ["v04", "v06", "v08"], "plane_sym": ["v00_1", "v02", "v04"]}


def tabla_seleccion(tipo: str) -> str:
    filas, incompletas = [], []
    for p, por_res in SELECCION[tipo]:
        if p not in SELECCION_RESUMEN[tipo]:
            continue
        valores = []
        for size in (224, 448, 1136):
            if size not in por_res:
                valores.append(None)
                continue
            cs = [celda(por_res[size][0], tipo, ident(tipo, p, size), v, "auc_angular") for v in VISTAS]
            if any(c is None for c in cs):
                incompletas.append(f"{tt(p)} a {size}~px")
            valores.append(media(cs))
        cambio = None
        if valores[0] and valores[1]:
            cambio = derivada(valores[1]["v"] - valores[0]["v"], "resta", [valores[1], valores[0]])
        filas.append(([tt(p)], valores + [cambio]))
    enc = [r"\textit{Prompt} & 224~px & 448~px & 1136~px & \begin{tabular}[b]{@{}c@{}}Cambio \\ 224$\to$448\end{tabular} \\"]
    # Nota reducida a pedido; las corridas incompletas no deberían aparecer con SELECCION_RESUMEN.
    if incompletas:
        raise ValueError(f"selección {tipo}: corridas incompletas {incompletas}")
    nota = r"$n_v\in\{6,14,26\}$, iluminación \textit{flat}, envío multiimagen."
    cols = [("auc", True)] * 3 + [("auc", None)]
    return tabla(f"seleccion_{TIPOS[tipo]}",
                 rf"Selección del \textit{{prompt}} por resolución en simetría "
                 rf"{'axial' if tipo == 'axis_sym' else 'planar'} (Flujo~A).", enc, filas, cols, nota)


# Tablas completas del Anexo (anx:tablas-completas): todas las métricas de cap3 por prompt y n_v.
# Sin negritas: son tablas de consulta; la elección se argumenta con las tablas del cuerpo.
COMPLETO = [("n_objects", "Objetos", "obj"), ("auc_angular", r"AUC$_{45^\circ}$", "auc"),
            ("angular_error_mean", r"Media|($^\circ$)", "grados"),
            ("angular_error_median", r"Mediana|($^\circ$)", "grados"),
            ("precision_5deg", r"P@5$^\circ$", "prec"), ("precision_10deg", r"P@10$^\circ$", "prec"),
            ("precision_15deg", r"P@15$^\circ$", "prec"),
            ("translation_error_normalized_median", r"$t_n$|mediana", "sde")]
COMPLETO_F1 = [("f1_ref", r"$F1$|voraz", "f1"), ("f1_ref_hungarian", r"$F1$|húngaro", "f1")]
COMPLETO_SDE = [("sde_ref_mean", "SDE", "sde")]


def completa(nombre: str, tipo: str, condiciones: list[tuple[str, str, str]], encabezado_cond: str,
             titulo: str, nota: str) -> str:
    """ Tabla completa por número de vistas, una fila por condición: (nombre LaTeX, carpeta, id). """
    cols = COMPLETO + (COMPLETO_F1 if tipo == "plane_sym" else []) + COMPLETO_SDE
    filas, separar = [], []
    for k, v in enumerate(VISTAS):
        if k:
            separar.append(len(filas))
        for j, (etiqueta, carpeta, exp_id) in enumerate(condiciones):
            inicio = [rf"\multirow{{{len(condiciones)}}}{{*}}{{{v}}}" if j == 0 else "", etiqueta]
            filas.append((inicio, [celda(carpeta, tipo, exp_id, v, c) for c, _, _ in cols]))
    encabezado = [" & ".join(["Vistas", encabezado_cond] + [apilar(h) for _, h, _ in cols]) + r" \\"]
    nota_comun = (f"{METODO[tipo]} {NOTA_ANG} Objetos: objetos con predicción. $t_n$: error de "
                  r"traslación normalizado, mediana. $t_n$, " + ("$F1$ y " if tipo == "plane_sym" else "")
                  + r"SDE usan solo los objetos con predicción. ")
    if any(all(c is None for c in celdas) for _, celdas in filas):
        nota_comun += "--: no se corrió esa combinación. "
    return tabla(nombre, titulo, encabezado, filas, [(f, None) for _, _, f in cols], nota_comun + nota,
                 separar, alto=r"0.8\textheight")


def tabla_centro(tipo: str) -> str:
    """ Fracción de puntos en la columna central de la imagen por prompt, flujo y VLM, con su AUC. """
    corto, base = ("axis", "v04") if tipo == "axis_sym" else ("plane", "v02")
    alias = "nomesh" if tipo == "axis_sym" else "mp1_nomesh"
    otras = {   # etiqueta: (id del diagnóstico, carpeta del CSV final, prompt del id final)
        "axis_sym": [(rf"Flujo~B ({tt('v04')})", "axis_v04_flowB", "ndcfix_flujoB", "v04_flowB"),
                     (rf"Flujo~C ({tt('v05_1')})", "axis_v05_1_flowC", "ndcfix_flujoC", "v05_1_flowC"),
                     ("Qwen3-VL-8B", f"axis_v04_qwen3vl_s224_flat_ndcfix_{alias}", "qwen3vl", "v04_qwen3vl"),
                     ("MolmoPoint-8B", f"axis_v04_molmopoint_completo_s224_flat_ndcfix_{alias}",
                      "axis_v04_224_axis_molmopoint", "v04_molmopoint_completo")],
        "plane_sym": [(rf"Flujo~B ({tt('v02')})", "plane_v02_flowB", "plane_v02_224_flujoB", "v02_flowB"),
                      (rf"Flujo~C ({tt('v02')})", "plane_v02_flowC", "plane_v02_224_flujoC", "v02_flowC"),
                      ("Qwen3-VL-8B", f"plane_v02_qwen3vl_s224_flat_ndcfix_{alias}", "plane_v02_224_qwen3vl",
                       "v02_qwen3vl"),
                      ("MolmoPoint-8B", f"plane_v02_molmopoint_s224_flat_ndcfix_{alias}",
                       "plane_v02_224_molmopoint", "v02_molmopoint")]}[tipo]
    filas = []
    for p in PROMPTS:
        carpeta = "ndcfix_v08" if p == "v08" else "ndcfix_flujoA"
        auc = media([celda(carpeta, tipo, ident(tipo, p), v, "auc_angular") for v in VISTAS])
        filas.append(([tt(p) + (r"$^{\dagger}$" if p == base else "")],
                      [celda_centro(tipo, f"{corto}_{p}", v) for v in VISTAS] + [auc]))
    for etiqueta, id_centro, carpeta, prompt in otras:
        auc = media([celda(carpeta, tipo, ident(tipo, prompt), v, "auc_angular") for v in VISTAS])
        filas.append(([etiqueta], [celda_centro(tipo, id_centro, v) for v in VISTAS] + [auc]))
    enc = [r"\multirow{2}{*}{Configuración} & \multicolumn{3}{c}{Puntos en la columna central} & "
           r"\multirow{2}{*}{\begin{tabular}[b]{@{}c@{}}AUC$_{45^\circ}$ \\ medio\end{tabular}} \\",
           r"\cmidrule(lr){2-4}",
           r" & 6 vistas & 14 vistas & 26 vistas & \\"]
    nota = (r"Fracción de los puntos de vistas no cenitales ($|$elevación$| \le 70^\circ$) a menos de 30 "
            r"unidades de $x = 500$, en la escala 0--1000 de Molmo2. Filas de \textit{prompts}: Molmo2 "
            rf"({REF}). $^{{\dagger}}$: \textit{{prompt}} de referencia. Qwen3-VL-8B y MolmoPoint-8B: "
            rf"\textit{{prompt}} {tt(base)} y envío de una imagen por llamada. AUC$_{{45^\circ}}$ medio: media "
            r"sobre 6, 14 y 26 vistas.")
    simetria = "axial" if tipo == "axis_sym" else "planar"
    return tabla(f"centro_{TIPOS[tipo]}", f"Puntos en la columna central de la imagen en simetría {simetria}.",
                 enc, filas, [("prec", None)] * 3 + [("auc", None)], nota, separar=[len(PROMPTS)])


LOG_PROFUNDIDAD = RES / "diagnosticos" / "profundidad_plano.log"
RANGOS_THETA = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90"]


def celda_profundidad(vistas: int, rango: str | None, columna: str) -> dict:
    """ Celda del log de diagnostico_profundidad_malla.py: total de puntos de un número de vistas
    (rango None) o n / dist. mediana / frac. cerca de un rango de theta. """
    texto = LOG_PROFUNDIDAD.read_text(encoding="utf-8")
    bloque = re.search(rf"^{vistas} vistas: (\d+) puntos\n(.*?)(?:\n\n|\Z)", texto, re.M | re.S)
    if rango is None:
        valor = float(bloque.group(1))
    else:
        fila = re.search(rf"^\s+{rango}\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s*$", bloque.group(2), re.M)
        valor = float(fila.group({"n": 1, "mediana": 2, "cerca": 3}[columna]))
    return {"v": valor, "t": {"fuente": "profundidad", "archivo": rel(LOG_PROFUNDIDAD), "n_views": vistas,
                              "rango": rango, "columna": columna if rango else "total"}}


def tabla_profundidad() -> str:
    """ Distancia de los puntos retroproyectados al plano de referencia según el ángulo de la cámara. """
    filas = []
    for rango in RANGOS_THETA:
        cs = []
        for v in VISTAS:
            n, total = celda_profundidad(v, rango, "n"), celda_profundidad(v, None, "total")
            cs += [derivada(n["v"] / total["v"], "division", [n, total]),
                   celda_profundidad(v, rango, "mediana"), celda_profundidad(v, rango, "cerca")]
        lo, hi = rango.split("-")
        filas.append(([rf"{lo}--{hi}$^\circ$"], cs))
    enc = [r"\multirow{2}{*}{$\theta$} & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{v} vistas}}" for v in VISTAS)
           + r" \\",
           "".join(rf"\cmidrule(lr){{{2 + 3 * k}-{4 + 3 * k}}}" for k in range(3)),
           " & " + " & ".join([r"Puntos & \begin{tabular}[b]{@{}c@{}}Dist.\ \\ mediana\end{tabular} & Cerca"] * 3)
           + r" \\"]
    nota = (rf"Molmo2, \textit{{prompt}} {tt('v02')} ({REF}), un rayo por punto. Puntos: fracción de los puntos "
            r"que impactan la malla. Cerca: fracción a menos de $0.02$ del plano.")
    return tabla("profundidad_plano", r"Distancia de los puntos con malla al plano de referencia según $\theta$.",
                 enc, filas, [("prec", None), ("sde", None), ("prec", None)] * 3, nota)


def tabla_completa_prompts(tipo: str) -> str:
    condiciones = [(tt(p), "ndcfix_v08" if p == "v08" else "ndcfix_flujoA", ident(tipo, p)) for p in PROMPTS]
    simetria = "axial" if tipo == "axis_sym" else "planar"
    return completa(f"completa_prompts_{TIPOS[tipo]}", tipo, condiciones, r"\textit{Prompt}",
                    rf"Métricas completas de los \textit{{prompts}} en simetría {simetria} ({REF}).",
                    rf"Resumen en la Tabla~\ref{{tab:res_prompts_{TIPOS[tipo]}}}.")


def tabla_completa_seleccion(tipo: str) -> str:
    condiciones = [(rf"{tt(p)}, {size}~px", por_res[size][0], ident(tipo, p, size))
                   for p, por_res in SELECCION[tipo] if p in SELECCION_RESUMEN[tipo]
                   for size in (224, 448, 1136) if size in por_res]
    simetria = "axial" if tipo == "axis_sym" else "planar"
    nota = rf"Resumen en la Tabla~\ref{{tab:res_seleccion_{TIPOS[tipo]}}}."
    return completa(f"completa_seleccion_{TIPOS[tipo]}", tipo, condiciones, r"\textit{Prompt}, resolución",
                    rf"Métricas completas de la selección de \textit{{prompts}} por resolución en simetría "
                    rf"{simetria} (iluminación \textit{{flat}}, envío multiimagen y Flujo~A).", nota)


def tabla_pareado_estimador(tipo: str) -> str:
    prompt = "v04" if tipo == "axis_sym" else "v02"
    ruta = RES / "ndcfix_flujoA" / "pareado" / f"{ident(tipo, prompt)}.csv"
    metricas = [("auc_angular", "auc", True), ("angular_error_median", "grados", False)]
    if tipo == "plane_sym":
        metricas.append(("f1_ref", "f1", True))
    filas = []
    for v in VISTAS:
        cs = []
        for col, _, _ in metricas:
            cs += [celda_pareado(ruta, v, f"{col}_base"), celda_pareado(ruta, v, f"{col}_nuevo")]
        cs += [celda_pareado(ruta, v, c) for c in ("dif_media", "ic_inf", "ic_sup", "mejoran", "empeoran")]
        filas.append(([str(v)], cs))
    nombres = {"auc_angular": r"AUC$_{45^\circ}$", "angular_error_median": r"Mediana ($^\circ$)",
               "f1_ref": r"$F1$"}
    original, final = (("SVD", "EXP-E") if tipo == "axis_sym" else ("Cámaras", r"$S_{\mathrm{dat}}$"))
    grupos = " & ".join(rf"\multicolumn{{2}}{{c}}{{{nombres[c]}}}" for c, _, _ in metricas)
    reglas = "".join(rf"\cmidrule(lr){{{2 + 2 * k}-{3 + 2 * k}}}" for k in range(len(metricas)))
    n = 2 + 2 * len(metricas)
    enc = [rf"\multirow{{2}}{{*}}{{Vistas}} & {grupos} & \multicolumn{{3}}{{c}}{{Dif.\ de error ($^\circ$)}}"
           r" & \multirow{2}{*}{Mejoran} & \multirow{2}{*}{Empeoran} \\",
           reglas + rf"\cmidrule(lr){{{n}-{n + 2}}}",
           " & " + " & ".join([original, final] * len(metricas)) + r" & Media & IC inf. & IC sup. & & \\"]
    cols = []
    for _, fmt, _ in metricas:
        cols += [(fmt, None), (fmt, None)]
    cols += [("grados", None)] * 3 + [("entero", None)] * 2
    if tipo == "axis_sym":
        intro = (r"Estimador original: SVD de las normales de interpretación sin reponderar. "
                 r"Estimador final: EXP-E.")
    else:
        intro = (r"Estimador original: puntaje por cámaras. Estimador final: $S_{\mathrm{dat}}$. Ambos "
                 r"predicen un plano por objeto.")
    nota = (rf"{intro} Mismos puntos de Molmo2 con el \textit{{prompt}} {tt(prompt)} ({REF}). Dif.\ de "
            r"error: diferencia objeto a objeto del error angular (final $-$ original) sobre los 850 "
            r"objetos, con $90^\circ$ imputados a los objetos sin predicción; negativo favorece al "
            r"estimador final. IC 95\% \textit{bootstrap} pareado. Mejoran/Empeoran: objetos cuyo "
            r"error baja/sube más de $1^\circ$.")
    return tabla(f"estimador_{TIPOS[tipo]}",
                 rf"Estimador original frente al final en simetría {'axial' if tipo == 'axis_sym' else 'planar'}.",
                 enc, filas, cols, nota)


# Puntos al azar dentro de la silueta (PI1 iii): PlaneScoreData/run_silueta_aleatoria.py, dos píxeles
# por vista en las mismas vistas que recibe Molmo2, sin usar ninguna salida del VLM.
def carpeta_silueta(tipo: str) -> str:
    return f"silueta2p_{'axis' if tipo == 'axis_sym' else 'plane'}_224"


def celda_silueta(tipo: str, semilla: str, vistas: int, columna: str) -> dict | None:
    return celda(f"{carpeta_silueta(tipo)}/{semilla}", tipo, ident(tipo, f"silueta2p_{semilla}"), vistas, columna)


def media_silueta(tipo: str, vistas: int, columna: str) -> dict | None:
    """ Media sobre las semillas completas de una columna del experimento de la silueta. """
    return media([celda_silueta(tipo, s, vistas, columna) for s in semillas_silueta(tipo)])


def filas_silueta(tipo: str, columnas: list[str], extra: int = 0) -> list[tuple[list[str], list]]:
    """ Filas 'Silueta, n_v' de las tablas de mejor configuración (vacío si aún no hay semillas). """
    if len(semillas_silueta(tipo)) < 2:
        return []
    return [([rf"Silueta, $n_v{{=}}{v}$"], [media_silueta(tipo, v, c) for c in columnas] + [None] * extra)
            for v in VISTAS]


def nota_silueta(tipo: str) -> str:
    n = len(semillas_silueta(tipo))
    return rf" Silueta: media sobre {n} semillas (Sección~\ref{{subsec:res-silueta}})." if n >= 2 else ""


def tabla_mejor_eje() -> str:
    filas = []
    for v in VISTAS:
        n = celda("ndcfix_flujoA", "axis_sym", ident("axis_sym", "v04"), v, "n_objects")
        m = celda("ndcfix_flujoA", "axis_sym", ident("axis_sym", "v04"), v, "angular_error_mean")
        auc = celda("ndcfix_flujoA", "axis_sym", ident("axis_sym", "v04"), v, "auc_angular")
        esperado = (n["v"] * AZAR_MEDIA + (TOTAL - n["v"]) * 90) / TOTAL
        e = derivada(esperado, "esperado_azar", [n])
        z = derivada((m["v"] - esperado) / (AZAR_DESV * n["v"] ** 0.5 / TOTAL), "z_azar", [m, n])
        filas.append(([rf"\texttt{{axis\_v04}}, $n_v{{=}}{v}$"], [n, auc, m, e, z]))
    filas += filas_silueta("axis_sym", ["n_objects", "auc_angular", "angular_error_mean"], extra=2)
    separar = sorted({3, len(filas)})
    azar_auc = celda_azar("axis_sym", "auc_angular", "closed_form_single_gt")
    filas.append(([r"Azar (analítico)"], [None, azar_auc, None, None, None]))
    enc = [r"Predictor & Objetos & AUC$_{45^\circ}$ & \begin{tabular}[b]{@{}c@{}}Error medio \\ ($^\circ$)\end{tabular}"
           r" & \begin{tabular}[b]{@{}c@{}}Esperado bajo \\ azar ($^\circ$)\end{tabular} & $z$ \\"]
    nota = rf"\textit{{Prompt}} {tt('v04')} ({REF})." + nota_silueta("axis_sym")
    return tabla("mejor_eje", "Mejor configuración axial frente a las líneas base.",
                 enc, filas, [("obj", None), ("auc", None), ("grados", None), ("grados", None), ("z", None)],
                 nota, separar=separar)


def tabla_mejor_plano() -> str:
    filas = []
    for v in VISTAS:
        filas.append(([rf"\texttt{{plane\_v02}}, $n_v{{=}}{v}$"],
                      [celda("ndcfix_flujoA", "plane_sym", ident("plane_sym", "v02"), v, c)
                       for c in ("n_objects", "auc_angular", "f1_ref", "f1_ref_hungarian")]))
    filas += filas_silueta("plane_sym", ["n_objects", "auc_angular", "f1_ref", "f1_ref_hungarian"])
    separar = sorted({3, len(filas)})
    filas.append((["Azar (simulación)"], [celda_azar("plane_sym", "auc_angular", "n_objects_gt"),
                                         celda_azar("plane_sym", "auc_angular", "random_mean"),
                                         celda_azar("plane_sym", "f1_ref", "random_mean"),
                                         celda_azar("plane_sym", "f1_ref_hungarian", "random_mean")]))
    for p, texto in (("always {X}", r"Siempre $\{X\}$"), ("always {X,Z}", r"Siempre $\{X,Z\}$")):
        # AUC de los triviales: solo existe en el registro desde que trivial_baselines.py la calcula
        # para plano; con dos planos se toma el mejor (valor optimista, marcado con *).
        fila = next(f for f in triviales()["plane_sym"] if f["predictor"] == p)
        auc = celda_trivial("plane_sym", p, "auc_angular") if "auc_angular" in fila else None
        if auc is not None and "," in p:
            auc["sufijo"] = "$^{*}$"
        filas.append(([texto], [objetos_trivial("plane_sym"), auc, celda_trivial("plane_sym", p, "f1_ref"),
                                celda_trivial("plane_sym", p, "f1_ref_hungarian")]))
    enc = [r"Predictor & Objetos & AUC$_{45^\circ}$ & \begin{tabular}[b]{@{}c@{}}$F1$ \\ voraz\end{tabular}"
           r" & \begin{tabular}[b]{@{}c@{}}$F1$ \\ húngaro\end{tabular} \\"]
    nota = (rf"\textit{{Prompt}} {tt('plane_v02')} ({REF})." + nota_silueta("plane_sym") +
            r" $^{*}$: el predictor propone dos planos y el AUC$_{45^\circ}$ toma el mejor de ellos frente a "
            r"la referencia, por lo que no es comparable con el de un solo plano.")
    return tabla("mejor_plano", "Mejor configuración planar frente a las líneas base.",
                 enc, filas, [("obj", None), ("auc", None), ("f1", None), ("f1", None)], nota, separar=separar)


def desvio(celdas: list[dict]) -> dict:
    """ Desviación estándar muestral (ddof=1) de las celdas. """
    v = [c["v"] for c in celdas]
    m = sum(v) / len(v)
    return derivada((sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5, "desvio", celdas)


def semillas_silueta(tipo: str) -> list[str]:
    """ Semillas completas (CSV final y pareado) del experimento de puntos al azar en la silueta. """
    raiz = RES / carpeta_silueta(tipo)
    return [d.name for d in sorted(raiz.glob("r[0-9][0-9]"))
            if list(d.glob(f"experiments_*/{tipo}_final_comparison.csv")) and list(d.glob("pareado/*.csv"))]


def tabla_silueta(tipo: str) -> str | None:
    """ Puntos sorteados dentro de la silueta, con las mismas vistas y estimador que Molmo2. """
    semillas = semillas_silueta(tipo)
    if len(semillas) < 2:
        print(f"silueta_{TIPOS[tipo]}: {len(semillas)} semillas completas, no se genera la tabla")
        return None
    metricas = [("n_objects", "obj"), ("auc_angular", "auc"), ("angular_error_median", "grados"),
                ("precision_10deg", "prec")] + ([("f1_ref", "f1")] if tipo == "plane_sym" else [])
    filas = []
    for v in VISTAS:
        cs = []
        for col, _ in metricas:
            por_semilla = [celda_silueta(tipo, s, v, col) for s in semillas]
            cs.append(media(por_semilla))
            if col == "auc_angular":
                cs.append(desvio(por_semilla))
        filas.append(([str(v)], cs))
    cols = []
    for col, fmt in metricas:
        cols += [(fmt, None)] * (2 if col == "auc_angular" else 1)
    nombres = {"n_objects": ["Objetos"], "auc_angular": [r"AUC$_{45^\circ}$", r"Desv.|AUC"],
               "angular_error_median": [r"Mediana|($^\circ$)"], "precision_10deg": [r"P@10$^\circ$"],
               "f1_ref": ["$F1$"]}
    titulos = [apilar(t) for col, _ in metricas for t in nombres[col]]
    enc = [r"Vistas & " + " & ".join(titulos) + r" \\"]
    nota = (r"Renders de 224~px, iluminación \textit{flat} y las mismas $n_v$ vistas que recibe Molmo2. "
            rf"Media sobre {len(semillas)} semillas. Desv.\ AUC: desviación estándar del AUC$_{{45^\circ}}$ "
            r"entre semillas.")
    simetria = "axial" if tipo == "axis_sym" else "planar"
    return tabla(f"silueta_{TIPOS[tipo]}", f"Puntos al azar dentro de la silueta en simetría {simetria}.",
                 enc, filas, cols, nota)


def tabla_rotacion() -> str:
    ruta = RES / "plane_v02_224_rot_so3" / "pareado" / f"{ident('plane_sym', 'v02_rot_so3')}.csv"
    filas = []
    for v in VISTAS:
        cs = [celda_pareado(ruta, v, c) for c in (
            "n_objects_base", "n_objects_nuevo", "auc_angular_base", "auc_angular_nuevo",
            "f1_ref_base", "f1_ref_nuevo", "dif_media", "ic_inf", "ic_sup", "mejoran", "empeoran")]
        filas.append(([str(v)], cs))
    enc = [r"\multirow{2}{*}{Vistas} & \multicolumn{2}{c}{Objetos} & \multicolumn{2}{c}{AUC$_{45^\circ}$} & "
           r"\multicolumn{2}{c}{$F1$} & \multicolumn{3}{c}{Dif.\ de error ($^\circ$)} & "
           r"\multirow{2}{*}{Mejoran} & \multirow{2}{*}{Empeoran} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-10}",
           r" & Original & Rotado & Original & Rotado & Original & Rotado & Media & IC inf. & IC sup. & & \\"]
    cols = [("obj", None)] * 2 + [("auc", None)] * 2 + [("f1", None)] * 2 + [("grados", None)] * 3 \
        + [("entero", None)] * 2
    nota = (rf"{METODO['plane_sym']} \textit{{Prompt}} {tt('v02')} ({REF}). Rotado: cada objeto y sus planos "
            r"de referencia se rotan con una rotación uniforme en SO(3) antes de renderizar. Dif.\ de "
            r"error: rotado $-$ original, objeto a objeto, sobre los 850 objetos con $90^\circ$ "
            r"imputados. IC 95\% \textit{bootstrap} pareado. Mejoran/Empeoran: más de $1^\circ$. "
            r"AUC$_{45^\circ}$ del azar analítico: 0.100 (Tabla~\ref{tab:res_azar}).")
    return tabla("rotacion_plano", "Efecto de la orientación canónica en simetría planar.", enc, filas,
                 cols, nota)


def tabla_aguirre() -> str:
    """ F1 y SDE de este trabajo frente a los valores publicados por Aguirre y Sipiran. """
    def propia(carpeta: str, tipo: str, exp_id: str, v: int) -> list:
        sde = celda(carpeta, tipo, exp_id, v, "sde_ref_mean")
        return [celda(carpeta, tipo, exp_id, v, "n_objects"),
                celda(carpeta, tipo, exp_id, v, "f1_ref") if tipo == "plane_sym" else None,
                derivada(sde["v"] * 1e4, "escala_1e4", [sde])]

    filas, separar, fino = [], [], []
    for tipo, config, carpeta, exp_id, etiqueta in (
            ("plane_sym", "FM10K-4R-FI", "ndcfix_flujoA", ident("plane_sym", "v02"), r"Este trabajo, pose canónica"),
            ("axis_sym", "RM-4R-FI-L50", "ndcfix_flujoA", ident("axis_sym", "v04"), r"Este trabajo")):
        if filas:
            separar.append(len(filas))
        # Fila de título del bloque: sin celdas numéricas
        filas.append(([rf"\multicolumn{{5}}{{l}}{{\textit{{{'Simetría planar' if tipo == 'plane_sym' else 'Simetría axial'}}}}}"],
                      []))
        for v in (6, 114):
            filas.append(([str(v), rf"Aguirre y Sipiran (\texttt{{{config}}})"],
                          [None, celda_publicada(tipo, config, v, "f1"), celda_publicada(tipo, config, v, "sde_e4")]))
        fino.append(len(filas))
        for v in VISTAS:
            filas.append(([str(v), etiqueta], propia(carpeta, tipo, exp_id, v)))
        if tipo == "plane_sym":
            fino.append(len(filas))
            for v in VISTAS:
                filas.append(([str(v), r"Este trabajo, rotación en $\mathrm{SO}(3)$"],
                              propia("plane_v02_224_rot_so3", tipo, ident("plane_sym", "v02_rot_so3"), v)))
    enc = [r"Vistas & Método & Objetos & $F1$ & $SDE$ ($\times 10^{-4}$) \\"]
    nota = (r"Valores de Aguirre y Sipiran tomados de su Tabla~1 (back-projection mejorado, 4R-FI), no de "
            r"una ejecución propia; conservan hasta 10 simetrías por objeto con confianza $\geq 0.95$, y no "
            r"reportan el número de objetos con predicción. Este trabajo predice una simetría por objeto: "
            rf"\textit{{prompt}} {tt('v02')} con $S_{{\mathrm{{dat}}}}$ en plano y {tt('v04')} con EXP-E en eje "
            rf"({REF}). {NOTA_F1} --: métrica no reportada o no calculada.")
    return tabla("aguirre", "Comparación con el método de Aguirre y Sipiran, con acceso a la malla.",
                 enc, filas, [("obj", None), ("f1", None), ("sde4", None)], nota, separar, fino)


ID_MULTI = re.compile(r"^plane_.+_s\d+_[a-z]+_ndcfix_sdat_multi_mp([123])_nomesh$")


def celda_multi(exp_id: str, vistas: int, columna: str) -> dict:
    """ Celda del CSV final del experimento multiplano (run_plane_multiplano.py). """
    ruta = csv_final("plane_v02_224_multiplano", "plane_sym")
    with open(ruta, encoding="utf-8", newline="") as f:
        filas = [r for r in csv.DictReader(f) if r["experiment"] == exp_id and int(r["n_views"]) == vistas]
    if len(filas) != 1 or not ID_MULTI.match(exp_id) or filas[0]["method"] != "triangulation_multiplane" \
            or int(filas[0]["n_total"]) != TOTAL:
        raise ValueError(f"{ruta}: fila multiplano no válida {exp_id} {vistas}")
    return {"v": float(filas[0][columna]), "t": {"fuente": "multiplano", "csv": rel(ruta),
                                                 "experiment": exp_id, "n_views": vistas, "columna": columna}}


def tabla_multiplano() -> str:
    """ Hasta 1, 2 o 3 planos por objeto con S_dat, mismos puntos de plane_v02. """
    columnas = [("n_objects", "obj"), ("n_planes_predicted_mean", "sde4"), ("n_planes_matched_mean", "sde4"),
                ("recall_planes_mean", "prec"), ("precision_planes_mean", "prec"),
                ("f1_ref", "f1"), ("f1_ref_hungarian", "f1"), ("sde_ref_mean", "sde")]
    filas, separar = [], []
    for k, v in enumerate(VISTAS):
        if k:
            separar.append(len(filas))
        for maximo in (1, 2, 3):
            exp_id = f"plane_v02_s224_flat_ndcfix_sdat_multi_mp{maximo}_nomesh"
            inicio = [rf"\multirow{{3}}{{*}}{{{v}}}" if maximo == 1 else "", f"${maximo}$"]
            filas.append((inicio, [celda_multi(exp_id, v, c) for c, _ in columnas]))
    enc = [r"Vistas & \begin{tabular}[b]{@{}c@{}}Máx.\ de \\ planos\end{tabular} & Objetos & "
           r"\begin{tabular}[b]{@{}c@{}}Planos \\ predichos\end{tabular} & "
           r"\begin{tabular}[b]{@{}c@{}}Planos \\ emparejados\end{tabular} & Recall & Precisión & "
           r"\begin{tabular}[b]{@{}c@{}}$F1$ \\ voraz\end{tabular} & "
           r"\begin{tabular}[b]{@{}c@{}}$F1$ \\ húngaro\end{tabular} & SDE \\"]
    nota = rf"Mismos puntos de Molmo2 con el \textit{{prompt}} {tt('v02')} ({REF})."
    return tabla("multiplano", "Un plano frente a varios planos por objeto en simetría planar.", enc, filas,
                 [(fmt, None) for _, fmt in columnas], nota, separar)


VARIANTES_MALLA = {"base": "1 rayo", "p3": "parche 3$\\times$3", "p5": "parche 5$\\times$5",
                   "cluster": "agrupamiento voraz", "hdbscan_ms2": "HDBSCAN, ms 2",
                   "hdbscan_ms3": "HDBSCAN, ms 3", "hdbscan_ms5": "HDBSCAN, ms 5"}


def tabla_malla(tipo: str, completa: bool = False) -> str:
    """ Pipeline sin malla frente a la retroproyección sobre la malla con los mismos puntos.

    Filas por número de vistas: sin malla, malla con un rayo por punto (SVD) y la variante con
    malla de mayor AUC promedio en 6, 14 y 26 vistas, con el pareado objeto a objeto. Con
    completa=True, las 14 variantes con malla (anexo).

    """
    prompt = "v04" if tipo == "axis_sym" else "v02"
    corto = "axis" if tipo == "axis_sym" else "plane"
    carpeta = f"{corto}_malla_224"
    alias = f"{corto}_{prompt}_s224_flat_ndcfix_mesh"
    id_var = lambda v: alias if v == "base" else f"{alias}_{v}"  # noqa: E731
    ruta_par = next((RES / carpeta / "pareado").glob(f"{alias}_vs_*.csv"))
    datos = leer_malla(csv_final(carpeta, tipo))
    mejor = max(((v, m) for v in VARIANTES_MALLA for m in ("svd", "ransac_svd")),
                key=lambda vm: sum(float(datos[(id_var(vm[0]), vm[1], n)]["auc_angular"]) for n in VISTAS))
    condiciones = [("base", "svd")] + ([mejor] if mejor != ("base", "svd") else [])
    if completa:
        condiciones = [(v, m) for v in VARIANTES_MALLA for m in ("svd", "ransac_svd")]
    metricas = ["n_objects", "auc_angular", "angular_error_median", "precision_10deg"]
    metricas += ["f1_ref"] if tipo == "plane_sym" else []
    pareado = ["dif_media", "ic_inf", "ic_sup", "mejoran", "empeoran"]
    filas, separar = [], []
    for k, v in enumerate(VISTAS):
        if k:
            separar.append(len(filas))
        n_filas = 1 + len(condiciones)
        sin = [celda("ndcfix_flujoA", tipo, ident(tipo, prompt), v, c) for c in metricas]
        filas.append(([rf"\multirow{{{n_filas}}}{{*}}{{{v}}}", "Sin malla"], sin + [None] * len(pareado)))
        for variante, metodo in condiciones:
            etiqueta = rf"Malla, {VARIANTES_MALLA[variante]}, {'SVD' if metodo == 'svd' else 'RANSAC+SVD'}"
            con = [celda_malla(carpeta, tipo, id_var(variante), metodo, v, c) for c in metricas]
            par = [celda_pareado_malla(ruta_par, variante, metodo, v, c) for c in pareado]
            filas.append((["", etiqueta], con + par))
    cols = [("obj", True), ("auc", True), ("grados", False), ("prec", True)]
    cols += [("f1", True)] if tipo == "plane_sym" else []
    cols += [("grados", None)] * 3 + [("entero", None)] * 2
    n_met = len(metricas)
    enc = [rf"\multirow{{2}}{{*}}{{Vistas}} & \multirow{{2}}{{*}}{{Condición}} & \multirow{{2}}{{*}}{{Objetos}} & "
           rf"\multirow{{2}}{{*}}{{AUC$_{{45^\circ}}$}} & \multirow{{2}}{{*}}{{\begin{{tabular}}[b]{{@{{}}c@{{}}}}Mediana \\ ($^\circ$)\end{{tabular}}}} & "
           rf"\multirow{{2}}{{*}}{{P@10$^\circ$}}"
           + (r" & \multirow{2}{*}{$F1$}" if tipo == "plane_sym" else "")
           + r" & \multicolumn{3}{c}{Dif.\ de error ($^\circ$)} & \multirow{2}{*}{Mejoran} & \multirow{2}{*}{Empeoran} \\",
           rf"\cmidrule(lr){{{3 + n_met}-{5 + n_met}}}",
           " & " * (2 + n_met) + r"Media & IC inf. & IC sup. & & \\"]
    nota = rf"Molmo2, \textit{{prompt}} {tt(prompt)} ({REF})."
    simetria = "axial" if tipo == "axis_sym" else "planar"
    if completa:
        return tabla(f"completa_malla_{TIPOS[tipo]}", f"Todas las variantes de retroproyección sobre la malla "
                     f"en simetría {simetria}.", enc, filas, cols,
                     nota + rf" Resumen en la Tabla~\ref{{tab:res_malla_{TIPOS[tipo]}}}.", separar,
                     alto=r"0.85\textheight")
    return tabla(f"malla_{TIPOS[tipo]}", f"Triangulación sin malla frente a retroproyección sobre la malla "
                 f"en simetría {simetria}.", enc, filas, cols, nota, separar)


def main() -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    hechas = [tabla_azar(), *tablas_triviales()]
    for tipo in ("axis_sym", "plane_sym"):
        hechas += [tabla_prompts(tipo), tabla_seleccion(tipo),
                   tabla_completa_prompts(tipo), tabla_completa_seleccion(tipo)]
    # Flujos
    # Flujo C no usa el prompt base (build_flow_c_prompts); en eje solo existe la corrida
    # axis_v05_1_flowC, donde v05_1 solo actúa si falta la pista de ubicación.
    hechas.append(comparativa("flujos_eje", "axis_sym", [
        ("A", "ndcfix_flujoA", ident("axis_sym", "v04")),
        ("B", "ndcfix_flujoB", ident("axis_sym", "v04_flowB")),
        ("C", "ndcfix_flujoC", ident("axis_sym", "v05_1_flowC"))],
        "Flujo", r"Flujos A, B y C en simetría axial con \texttt{axis\_v04} (224~px, \textit{flat}, "
                 r"envío multiimagen).", ""))
    hechas.append(comparativa("flujos_plano", "plane_sym", [
        ("A", "ndcfix_flujoA", ident("plane_sym", "v02")),
        ("B", "plane_v02_224_flujoB", ident("plane_sym", "v02_flowB")),
        ("C", "plane_v02_224_flujoC", ident("plane_sym", "v02_flowC"))],
        "Flujo", r"Flujos A, B y C en simetría planar con \texttt{plane\_v02} (224~px, \textit{flat}, "
                 r"envío multiimagen).", ""))
    # Modo de envío
    hechas.append(comparativa("envio_eje", "axis_sym", [
        ("Multiimagen", "ndcfix_flujoA", ident("axis_sym", "v04")),
        ("Una vista", "ndcfix_single_oct26", ident("axis_sym", "v04_single_oct26"))],
        "Envío", r"Modo de envío en simetría axial con \texttt{axis\_v04} (224~px, \textit{flat}, Flujo~A).", ""))
    hechas.append(comparativa("envio_plano", "plane_sym", [
        ("Multiimagen", "ndcfix_flujoA", ident("plane_sym", "v02")),
        ("Una vista", "plane_v02_224_single", ident("plane_sym", "v02_single"))],
        "Envío", r"Modo de envío en simetría planar con \texttt{plane\_v02} (224~px, \textit{flat}, Flujo~A).", ""))
    # Resolución
    hechas.append(comparativa("resolucion_eje", "axis_sym", [
        ("224~px", "ndcfix_flujoA", ident("axis_sym", "v04")),
        ("448~px", "ndcfix_448", ident("axis_sym", "v04", 448)),
        ("1136~px", "ndcfix_1136", ident("axis_sym", "v04", 1136))],
        "Resolución", r"Efecto de la resolución en simetría axial con \texttt{axis\_v04} (\textit{flat}, "
                      r"envío multiimagen).", ""))
    hechas.append(comparativa("resolucion_plano", "plane_sym", [
        ("224~px", "ndcfix_flujoA", ident("plane_sym", "v02")),
        ("448~px", "plane_448px", ident("plane_sym", "v02", 448)),
        ("1136~px", "plane_1136px", ident("plane_sym", "v02", 1136))],
        "Resolución", r"Efecto de la resolución en simetría planar con \texttt{plane\_v02} (\textit{flat}, "
                      r"envío multiimagen).", ""))
    # Iluminación
    hechas.append(comparativa("iluminacion_eje", "axis_sym", [
        (r"\textit{flat}", "ndcfix_flujoA", ident("axis_sym", "v04")),
        (r"\textit{darker}", "ndcfix_224_darker", ident("axis_sym", "v04", 224, "darker")),
        (r"\textit{brighter}", "ndcfix_224_brighter", ident("axis_sym", "v04", 224, "brighter"))],
        "Iluminación", r"Efecto de la iluminación en simetría axial con \texttt{axis\_v04} (224~px, envío "
                       r"multiimagen).", ""))
    hechas.append(comparativa("iluminacion_plano", "plane_sym", [
        (r"\textit{flat}", "ndcfix_flujoA", ident("plane_sym", "v02")),
        (r"\textit{darker}", "plane_v02_224_darker", ident("plane_sym", "v02", 224, "darker")),
        (r"\textit{brighter}", "plane_v02_224_brighter", ident("plane_sym", "v02", 224, "brighter"))],
        "Iluminación", r"Efecto de la iluminación en simetría planar con \texttt{plane\_v02} (224~px, envío "
                       r"multiimagen).", ""))
    # S_dat es el estimador por defecto: ya no se publica la comparación con el puntaje original.
    hechas += [tabla_pareado_estimador("axis_sym"),
               tabla_mejor_eje(), tabla_mejor_plano(), tabla_rotacion(), tabla_multiplano()]
    # Puntos al azar dentro de la silueta (PI1 iii): solo si ya hay semillas completas
    hechas += [t for t in (tabla_silueta("axis_sym"), tabla_silueta("plane_sym")) if t]
    # VLMs (una vista por llamada)
    hechas.append(comparativa("vlms_eje", "axis_sym", [
        ("Molmo2-8B", "ndcfix_single_oct26", ident("axis_sym", "v04_single_oct26")),
        ("MolmoPoint-8B (frase)", "molmopoint", ident("axis_sym", "v04_molmopoint")),
        ("MolmoPoint-8B (completo)", "axis_v04_224_axis_molmopoint", ident("axis_sym", "v04_molmopoint_completo")),
        ("Qwen3-VL-8B", "qwen3vl", ident("axis_sym", "v04_qwen3vl"))],
        "Modelo", r"Comparación de VLMs en simetría axial con \texttt{axis\_v04} (224~px, \textit{flat}, "
                  r"una vista por llamada).", ""))
    hechas.append(comparativa("vlms_plano", "plane_sym", [
        ("Molmo2-8B", "plane_v02_224_single", ident("plane_sym", "v02_single")),
        ("MolmoPoint-8B (frase)", "plane_v02_224_molmopoint_frase", ident("plane_sym", "v02_molmopoint_frase")),
        ("MolmoPoint-8B (completo)", "plane_v02_224_molmopoint", ident("plane_sym", "v02_molmopoint")),
        ("Qwen3-VL-8B", "plane_v02_224_qwen3vl", ident("plane_sym", "v02_qwen3vl"))],
        "Modelo", r"Comparación de VLMs en simetría planar con \texttt{plane\_v02} (224~px, \textit{flat}, "
                  r"una vista por llamada).", ""))
    hechas.append(tabla_aguirre())
    hechas += [tabla_centro("axis_sym"), tabla_centro("plane_sym"), tabla_profundidad()]
    hechas += [tabla_malla("axis_sym"), tabla_malla("plane_sym"),
               tabla_malla("axis_sym", completa=True), tabla_malla("plane_sym", completa=True)]
    REGISTRO.write_text(json.dumps(registro, indent=1, ensure_ascii=False), encoding="utf-8",
                        newline="\n")
    print(f"{len(hechas)} tablas en {SALIDA}; {sum(len(v) for v in registro.values())} celdas registradas")


if __name__ == "__main__":
    main()
