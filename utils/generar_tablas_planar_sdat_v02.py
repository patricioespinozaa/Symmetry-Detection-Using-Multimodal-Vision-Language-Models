#!/usr/bin/env python3
# Genera las tablas LaTeX de la serie plane_v02 a 224px con S_dat (un plano por objeto), corridas
# por PlaneScoreData/run_plane_v02_224.py (10-13/11/2026): iluminación, flujos A/B/C, envío
# multiimagen frente a individual, comparación de VLMs y la comparación pareada de cada
# experimento contra la línea base plane_v02 (224px, flat, multiimagen).
#
# Usa las funciones de utils/generar_tablas_noviembre.py sin modificarlo, así que el formato, las
# notas y la negrita son los mismos que en sus tablas (etiquetas tab:nov_plane_v02_*).
#
#   python utils/generar_tablas_planar_sdat_v02.py
#   python utils/generar_tablas_planar_sdat_v02.py --salida otra/ruta.tex
#
# Requiere en el preámbulo: booktabs, multirow, adjustbox y \newsavebox{\tablebox}.
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generar_tablas_noviembre as nov  # noqa: E402

TIPO = "plane_sym"
PROMPT = "v02"
SALIDA = nov.RESULTADOS / "planar_sdat" / "tex" / "plane_v02_224.tex"
BASE = ("ndcfix_flujoA", PROMPT)
# Comparaciones pareadas: (carpeta de resultados_finales, nombre LaTeX). La línea base es la misma
# en todas: plane_v02, 224px, flat, multiimagen.
PAREADOS = [
    ("plane_v02_224_single", r"Envío individual"),
    ("plane_v02_224_flujoB", r"Flujo B"),
    ("plane_v02_224_flujoC", r"Flujo C"),
    ("plane_v02_224_qwen3vl", r"Qwen3-VL-8B"),
    ("plane_v02_224_molmopoint", r"MolmoPoint-8B"),
    ("plane_v02_224_molmopoint_frase", r"MolmoPoint-8B (frase)"),
]
NOTA_PAREADO = (r"Diferencia del error angular por objeto (nuevo $-$ base) sobre los 850 objetos, "
                r"con $90^\circ$ imputados a los objetos sin predicción; valores negativos "
                r"favorecen al experimento. IC 95\%: \textit{bootstrap} pareado. Mejoran/Empeoran: "
                r"objetos cuyo error baja/sube más de $1^\circ$. Base: \textit{prompt} "
                r"\texttt{plane\_v02}, resolución 224px, iluminación \textit{flat}, envío "
                r"multiimagen.")


def datos(carpeta: str, prompt: str) -> dict[int, dict]:
    """ Filas por número de vistas de un experimento de resultados_finales/. """
    filas = nov.por_vistas(nov.cargar(carpeta, TIPO), prompt)
    if sorted(filas) != nov.VISTAS:
        raise ValueError(f"{carpeta}/{prompt}: vistas {sorted(filas)}")
    return filas


def tabla_pareado() -> str:
    """ Tabla con la comparación pareada de cada experimento contra la línea base. """
    columnas = ["n", "dif_media", "ic", "mejoran", "empeoran", "precision_10deg_base",
                "precision_10deg_nuevo"]
    encabezado = [" & ".join(["Vistas", "Experimento", "Objetos", nov.apilar(r"Dif.\ media|($^\circ$)"),
                              "IC 95\\%", "Mejoran", "Empeoran", nov.apilar(r"P@10$^\circ$|base"),
                              nov.apilar(r"P@10$^\circ$|nuevo")]) + r" \\"]
    filas = {}
    for carpeta, nombre in PAREADOS:
        rutas = sorted((nov.RESULTADOS / carpeta / "pareado").glob("*.csv"))
        if len(rutas) != 1:
            raise ValueError(f"{carpeta}: se esperaba un CSV pareado, hay {len(rutas)}")
        with open(rutas[0], encoding="utf-8", newline="") as archivo:
            for f in csv.DictReader(archivo):
                filas[(int(f["n_views"]), nombre)] = f
    cuerpo = []
    for k, v in enumerate(nov.VISTAS):
        if k:
            cuerpo.append(r"\midrule")
        for j, (_, nombre) in enumerate(PAREADOS):
            f = filas[(v, nombre)]
            vistas = rf"\multirow{{{len(PAREADOS)}}}{{*}}{{{v}}}" if j == 0 else ""
            celdas = [vistas, nombre, f"{int(float(f['n']))}", f"{float(f['dif_media']):+.2f}",
                      f"[{float(f['ic_inf']):+.2f}, {float(f['ic_sup']):+.2f}]",
                      f"{int(float(f['mejoran']))}", f"{int(float(f['empeoran']))}",
                      f"{float(f['precision_10deg_base']):.3f}",
                      f"{float(f['precision_10deg_nuevo']):.3f}"]
            cuerpo.append(" & ".join(celdas) + r" \\")
    titulo = (r"Comparación pareada de los experimentos con \texttt{plane\_v02} frente a su línea "
              r"base en simetría planar, resolución 224px")
    return nov.envolver(encabezado, cuerpo, "llrrcrrrr", titulo, "plane_v02_pareado",
                        f"{nov.METODO[TIPO]} {NOTA_PAREADO}")


def tablas() -> list[tuple[str, str]]:
    """ Todas las tablas de la serie plane_v02 a 224px, en orden. """
    nombre = nov.TITULO_TIPO[TIPO]
    prompt_tex = rf"\texttt{{{nov.escapar('plane_' + PROMPT)}}}"
    base = datos(*BASE)
    salida = []

    condiciones = [(r"\textit{flat}", base),
                   (r"\textit{darker}", datos("plane_v02_224_darker", PROMPT)),
                   (r"\textit{brighter}", datos("plane_v02_224_brighter", PROMPT))]
    salida.append(("Iluminación", nov.tabla_comparativa(
        condiciones, TIPO, "Iluminación",
        rf"Efecto de la iluminación en {nombre} con el \textit{{prompt}} {prompt_tex}, resolución "
        rf"224px, iluminación \textit{{flat}}, \textit{{darker}} y \textit{{brighter}}, {nov.MULTI}",
        "plane_v02_iluminacion",
        f"{nov.METODO[TIPO]} {nov.NOTA_ILUMINACION}{nov.NOTA_COMPARATIVA[TIPO]}")))

    flujos = [("A", base), ("B", datos("plane_v02_224_flujoB", f"{PROMPT}_flowB")),
              ("C", datos("plane_v02_224_flujoC", f"{PROMPT}_flowC"))]
    nota_flujos = (rf"Flujo A: \textit{{prompt}} {prompt_tex} sin contexto. Flujo B: el mismo "
                   r"\textit{prompt} precedido por una descripción del objeto. Flujo C: "
                   r"\textit{prompt} genérico construido a partir de la descripción y de una pista "
                   rf"de ubicación, con {prompt_tex} como respaldo. ")
    salida.append(("Flujos A, B y C", nov.tabla_comparativa(
        flujos, TIPO, "Flujo",
        rf"Comparación de los Flujos A, B y C en {nombre} con el \textit{{prompt}} {prompt_tex}, "
        f"{nov.configuracion(224, 'flat', nov.MULTI)}", "plane_v02_flujos",
        f"{nov.METODO[TIPO]} {nota_flujos}{nov.NOTA_COMPARATIVA[TIPO]}")))

    individual = datos("plane_v02_224_single", f"{PROMPT}_single")
    filas_modo = ([(rf"{prompt_tex} (\textit{{multi}})", v, base.get(v)) for v in nov.VISTAS]
                  + [(rf"{prompt_tex} (\textit{{single}})", v, individual.get(v))
                     for v in nov.VISTAS])
    nota_modo = (r"\textit{multi}: las $n_v$ vistas se envían en una sola llamada, con el "
                 r"\textit{prompt} multivista. \textit{single}: cada vista se envía por separado, "
                 r"con el \textit{prompt} de imagen única, y los puntos se reúnen antes de la "
                 r"triangulación. ")
    for bloque in ("angular", "posicion", "f1", "conteos"):
        titulo = (f"{nov.BLOQUES[bloque][0]} en {nombre} con el \\textit{{prompt}} {prompt_tex}, "
                  r"resolución 224px, iluminación \textit{flat}, envío multiimagen "
                  r"(\textit{multi}) frente a individual (\textit{single})")
        salida.append(("Envío individual vs multiimagen", nov.tabla_bloque(
            filas_modo, bloque, titulo, f"plane_v02_modo_envio_{bloque}",
            nov.nota_bloque(TIPO, bloque, nota_modo), "Prompt y envío")))

    columnas_vlm = [(c, h.replace(r" \\ ", "|").replace(r"\begin{tabular}[b]{@{}c@{}}", "")
                     .replace(r"\end{tabular}", "")) for c, h, _, _ in nov.COLUMNAS_VLM[TIPO]]
    modelos = [("Molmo2-8B", individual),
               ("MolmoPoint-8B", datos("plane_v02_224_molmopoint", f"{PROMPT}_molmopoint")),
               ("MolmoPoint-8B (frase)",
                datos("plane_v02_224_molmopoint_frase", f"{PROMPT}_molmopoint_frase")),
               ("Qwen3-VL-8B", datos("plane_v02_224_qwen3vl", f"{PROMPT}_qwen3vl"))]
    nota_frase = (r"MolmoPoint-8B (frase): versión de una frase del \textit{prompt}, con el mismo "
                  r"formato que el MolmoPoint axial. ")
    salida.append(("Comparación de modelos", nov.tabla_comparativa(
        modelos, TIPO, "Modelo",
        rf"Comparación de VLMs en {nombre} con el \textit{{prompt}} {prompt_tex}, "
        f"{nov.configuracion(224, 'flat', nov.SINGLE)}", "plane_v02_vlms",
        f"{nov.METODO[TIPO]} {nov.NOTA_ENVIO}{nota_frase}{nov.NOTA_COMPARATIVA[TIPO]}",
        columnas_vlm)))

    salida.append(("Comparación pareada contra la línea base", tabla_pareado()))
    return salida


def main() -> None:
    """ Escribe las tablas de la serie plane_v02 a 224px en un único .tex. """
    parser = argparse.ArgumentParser(description="Tablas LaTeX de la serie plane_v02 (S_dat).")
    parser.add_argument("--salida", type=Path, default=SALIDA, help="archivo .tex de salida")
    args = parser.parse_args()
    lineas = ["% Generado por utils/generar_tablas_planar_sdat_v02.py desde resultados_finales/.",
              "% Requiere booktabs, multirow, adjustbox y \\newsavebox{\\tablebox}.", ""]
    previa = None
    for total, (seccion, tabla) in enumerate(tablas(), 1):
        if seccion != previa:
            lineas += [f"% --- {seccion}", ""]
            previa = seccion
        lineas += [tabla, ""]
        if total % nov.TABLAS_POR_PAGINA == 0:
            lineas += [r"\clearpage", ""]
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text("\n".join(lineas), encoding="utf-8")
    print(f"Escrito {args.salida} ({total} tablas)")


if __name__ == "__main__":
    main()
