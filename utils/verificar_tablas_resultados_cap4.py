#!/usr/bin/env python3
# Verifica las tablas de Tesis_DCC_MDS/tablas_resultados/ contra sus fuentes, sin reutilizar el
# código del generador:
#   1. cada celda registrada se vuelve a leer de su CSV (o del registro de triviales) o se recalcula
#      con su fórmula, y debe dar el mismo texto que la tabla;
#   2. cada número del cuerpo de cada tabla debe ser una celda registrada (o un número de vistas);
#   3. cada CSV final usado debe ser de la versión final del pipeline sin malla: ids con ndcfix y
#      EXP-E (eje) o S_dat con un plano (plano), n_total = 850 y carpeta del 05/11/2026 o posterior
#      (los valores publicados por otros trabajos se leen de resultados_finales/referencias/);
#   4. cada tabla \input en cap4.tex debe existir.
#
#   python utils/verificar_tablas_resultados_cap4.py
import ast
import csv
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CARPETA = REPO / "Tesis_DCC_MDS" / "tablas_resultados"
FMT = {"auc": "{:.3f}", "grados": "{:.1f}", "prec": "{:.3f}", "f1": "{:.3f}", "obj": "{:.0f}",
       "sde": "{:.4f}", "entero": "{:.0f}", "z": "{:.1f}", "sde4": "{:.2f}"}
ID_OK = re.compile(r"^(axis_.+_ndcfix_expE_nomesh|plane_.+_ndcfix_sdat_mp1_nomesh)$")
ID_MALLA = re.compile(r"^(axis|plane)_.+_s\d+_[a-z]+_ndcfix_mesh(_[a-z0-9_]+)?$")
ID_MULTI = re.compile(r"^plane_.+_s\d+_[a-z]+_ndcfix_sdat_multi_mp[123]_nomesh$")
errores: list[str] = []


def fila_csv(ruta: str, filtro) -> dict:
    with open(REPO / ruta, encoding="utf-8", newline="") as f:
        filas = [r for r in csv.DictReader(f) if filtro(r)]
    if len(filas) != 1:
        raise LookupError(f"{ruta}: {len(filas)} filas coinciden")
    return filas[0]


def valor(t: dict) -> float:
    fuente = t["fuente"]
    if fuente == "final":
        r = fila_csv(t["csv"], lambda r: r["experiment"] == t["experiment"]
                     and r["method"] == "triangulation" and int(r["n_views"]) == t["n_views"])
        if not ID_OK.match(r["experiment"]) or int(r["n_total"]) != 850:
            errores.append(f"fila no final: {t['csv']} {r['experiment']} n_total={r['n_total']}")
        d, m, a = map(int, re.search(r"experiments_(\d+)_(\d+)_(\d+)", t["csv"]).groups())
        if (a, m, d) < (2026, 11, 5):
            errores.append(f"CSV anterior al recálculo NDC: {t['csv']}")
        return float(r[t["columna"]])
    if fuente == "pareado":
        return float(fila_csv(t["csv"], lambda r: int(r["n_views"]) == t["n_views"])[t["columna"]])
    if fuente == "azar":
        return float(fila_csv(t["csv"], lambda r: r["symmetry_type"] == t["tipo"]
                              and float(r["coverage"]) == 1.0 and r["metric"] == t["metrica"])[t["columna"]])
    if fuente == "malla":
        r = fila_csv(t["csv"], lambda r: r["experiment"] == t["experiment"]
                     and r["method"] == t["method"] and int(r["n_views"]) == t["n_views"])
        if not ID_MALLA.match(r["experiment"]) or int(r["n_total"]) != 850:
            errores.append(f"fila con malla no válida: {t['csv']} {r['experiment']} n_total={r['n_total']}")
        d, m, a = map(int, re.search(r"experiments_(\d+)_(\d+)_(\d+)", t["csv"]).groups())
        if (a, m, d) < (2026, 11, 5):
            errores.append(f"CSV anterior al recálculo NDC: {t['csv']}")
        return float(r[t["columna"]])
    if fuente == "multiplano":
        r = fila_csv(t["csv"], lambda r: r["experiment"] == t["experiment"]
                     and r["method"] == "triangulation_multiplane" and int(r["n_views"]) == t["n_views"])
        if not ID_MULTI.match(r["experiment"]) or int(r["n_total"]) != 850:
            errores.append(f"fila multiplano no válida: {t['csv']} {r['experiment']} n_total={r['n_total']}")
        d, m, a = map(int, re.search(r"experiments_(\d+)_(\d+)_(\d+)", t["csv"]).groups())
        if (a, m, d) < (2026, 11, 5):
            errores.append(f"CSV anterior al recálculo NDC: {t['csv']}")
        return float(r[t["columna"]])
    if fuente == "centro":
        return float(fila_csv(t["csv"], lambda r: r["experiment_id"] == t["experiment_id"]
                              and int(r["n_views"]) == t["n_views"])[t["columna"]])
    if fuente == "profundidad":
        texto = (REPO / t["archivo"]).read_text(encoding="utf-8")
        partes = texto.split(f"\n{t['n_views']} vistas: ", 1)[1]
        if t["columna"] == "total":
            return float(partes.split(" ", 1)[0])
        for linea in partes.split("\n\n", 1)[0].splitlines():
            campos = linea.split()
            if campos and campos[0] == t["rango"]:
                return float(campos[{"n": 1, "mediana": 2, "cerca": 3}[t["columna"]]])
        raise LookupError(f"{t['archivo']}: rango {t['rango']} con {t['n_views']} vistas")
    if fuente == "pareado_malla":
        return float(fila_csv(t["csv"], lambda r: r["variante"] == t["variante"]
                              and r["metodo"] == t["metodo"]
                              and int(r["n_views"]) == t["n_views"])[t["columna"]])
    if fuente == "publicado":
        return float(fila_csv(t["csv"], lambda r: r["symmetry_type"] == t["tipo"]
                              and r["configuracion"] == t["configuracion"]
                              and int(r["n_views"]) == t["n_views"])[t["columna"]])
    if fuente == "log":
        tipo, encontrados = None, []
        for linea in (REPO / t["archivo"]).read_text(encoding="utf-8").splitlines():
            m = re.match(r"=== (\w+):", linea)
            if m:
                tipo = m.group(1)
            elif tipo == t["tipo"] and "'predictor'" in linea:
                d = ast.literal_eval(linea.strip())
                if d["predictor"] == t["predictor"]:
                    encontrados.append(float(d[t["clave"]]))
        if len(encontrados) != 1:
            raise LookupError(f"{t['archivo']}: {t['predictor']} aparece {len(encontrados)} veces")
        return encontrados[0]
    if fuente == "log_objetos":
        hallados = re.findall(rf"=== {t['tipo']}: (\d+) objects", (REPO / t["archivo"]).read_text(encoding="utf-8"))
        if len(hallados) != 1:
            raise LookupError(f"{t['archivo']}: encabezado de {t['tipo']} aparece {len(hallados)} veces")
        return float(hallados[0])
    if fuente == "derivada":
        e = [valor(x) for x in t["entradas"]]
        if t["formula"] == "media":
            return sum(e) / len(e)
        if t["formula"] == "desvio":
            m = sum(e) / len(e)
            return (sum((x - m) ** 2 for x in e) / (len(e) - 1)) ** 0.5
        if t["formula"] == "division":
            return e[0] / e[1]
        if t["formula"] == "resta":
            return e[0] - e[1]
        if t["formula"] == "escala_1e4":
            return e[0] * 1e4
        if t["formula"] == "esperado_azar":
            return (e[0] * 57.2958 + (850 - e[0]) * 90) / 850
        if t["formula"] == "z_azar":
            media, n = e
            return (media - (n * 57.2958 + (850 - n) * 90) / 850) / (21.5597 * n ** 0.5 / 850)
    raise ValueError(f"fuente desconocida {t}")


def numeros_cuerpo(tex: str) -> list[str]:
    """ Números de las celdas numéricas del cuerpo; las etiquetas de fila no se revisan. """
    cuerpo = tex.split(r"\midrule", 1)[1].split(r"\bottomrule")[0]
    numeros = []
    for linea in cuerpo.split(r"\\"):
        for celda in linea.split("&")[1:]:     # la primera columna es la etiqueta o las vistas
            texto = celda.strip()
            hallado = re.fullmatch(r"(?:\\textbf\{)?(-?\d+(?:\.\d+)?)\}?(?:\$\^\{\*\}\$)?", texto)
            if hallado:
                numeros.append(hallado.group(1))
    return numeros


def main() -> None:
    registro = json.loads((CARPETA / "registro_celdas.json").read_text(encoding="utf-8"))
    total = 0
    for nombre, celdas in registro.items():
        tex = (CARPETA / f"{nombre}.tex").read_text(encoding="utf-8")
        textos = {c["texto"] for c in celdas}
        for c in celdas:
            total += 1
            try:
                esperado = FMT[c["formato"]].format(valor(c["traza"]))
                if float(esperado) == 0:
                    esperado = esperado.lstrip("-")       # '-0.0' se escribe '0.0'
            except (LookupError, KeyError) as error:
                errores.append(f"{nombre}: {error}")
                continue
            if esperado != c["texto"]:
                errores.append(f"{nombre} fila {c['fila']} col {c['columna']}: tabla {c['texto']} "
                               f"!= fuente {esperado}")
        # Celda por celda y en orden: la tabla debe mostrar exactamente los textos registrados
        numeros = numeros_cuerpo(tex)
        ordenadas = [c["texto"] for c in sorted(celdas, key=lambda c: (c["fila"], c["columna"]))]
        if len(numeros) != len(ordenadas):
            errores.append(f"{nombre}: {len(numeros)} celdas numéricas y {len(ordenadas)} registradas")
        for k, (en_tabla, registrada) in enumerate(zip(numeros, ordenadas)):
            if en_tabla != registrada:
                errores.append(f"{nombre}: celda numérica {k + 1} muestra {en_tabla}, la fuente da {registrada}")
    cap4 = (REPO / "Tesis_DCC_MDS" / "cap4.tex").read_text(encoding="utf-8")
    for incluido in re.findall(r"\\input\{tablas_resultados/([^}]+)\}", cap4):
        if not (CARPETA / f"{incluido}.tex").exists():
            errores.append(f"cap4.tex incluye tablas_resultados/{incluido}, que no existe")
    print(f"{len(registro)} tablas, {total} celdas revisadas")
    if errores:
        print("\n".join(errores[:40]))
        sys.exit(f"{len(errores)} problemas")
    print("OK: todas las celdas coinciden con sus fuentes y todas las fuentes son de la versión final")


if __name__ == "__main__":
    main()
