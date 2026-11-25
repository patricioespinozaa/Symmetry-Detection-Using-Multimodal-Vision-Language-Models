#!/usr/bin/env python3
# Genera la documentación de prompts del Anexo A desde los archivos que se enviaron a los modelos:
#   - Tesis_DCC_MDS/anexo_prompts_indice.tex: tabla con todos los prompts del anexo (modelo, flujo,
#     versiones y subsección donde está su texto);
#   - Tesis_DCC_MDS/anexo_prompts_texto.tex: texto completo de los prompts que el anexo no tenía
#     (Flujo A v06-v08, Flujos B y C, Qwen3-VL y MolmoPoint).
# Los textos de v00-v05 y v00_1-v05_1 ya están escritos a mano en anexoA.tex y solo se indexan.
# Los prompts que el runner arma en código (encabezado del Flujo B y plantilla del Flujo C) se
# obtienen ejecutando las funciones reales de MolmoPointing/molmo_multiview_runner.py, sin importar
# el módulo (que requiere torch), con marcadores en lugar de la descripción.
#
#   python utils/generar_anexo_prompts.py
import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SALIDA = REPO / "Tesis_DCC_MDS"
MOLMO = REPO / "MolmoPointing" / "prompts"
QWEN = REPO / "QwenPointing" / "prompts"
MPOINT = REPO / "MolmoPointPointing" / "prompts"
RUNNER = REPO / "MolmoPointing" / "molmo_multiview_runner.py"
MARCA_DESC, MARCA_UBIC = "<descripción generada por el modelo>", "<ubicación generada por el modelo>"
ORIGINALES = {  # subsecciones escritas a mano en anexoA.tex
    ("axis", False): "app:prompts_axis_base", ("axis", True): "app:prompts_axis_mejorados",
    ("plane", False): "app:prompts_plane_base", ("plane", True): "app:prompts_plane_mejorados",
}
ESCAPES = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "#": r"\#",
           "%": r"\%", "_": r"\_", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
           "<": r"\textless{}", ">": r"\textgreater{}", "≈": r"\ensuremath{\approx}"}


def escapar(texto: str) -> str:
    """ Escapa para LaTeX; '--' se separa para que no se imprima como guion largo. """
    return "".join(ESCAPES.get(c, c) for c in texto).replace("--", "-{}-").replace("--", "-{}-")


def tt(nombre: str) -> str:
    return rf"\texttt{{{escapar(nombre)}}}"


def caja(titulo: str, texto: str, fuente: str) -> str:
    """ Caja promptbox con el texto tal como se envió: respeta líneas y párrafos. """
    parrafos, actual = [], []
    for linea in texto.replace("\r\n", "\n").strip("\n").split("\n"):
        if linea.strip():
            actual.append(escapar(linea.rstrip()))
        elif actual:
            parrafos.append(actual)
            actual = []
    if actual:
        parrafos.append(actual)
    cuerpo = "\n\n\\smallskip\n".join(" \\\\\n".join(p) for p in parrafos)
    return "\n".join([f"% Fuente: {fuente}",
                      rf"\begin{{tcolorbox}}[promptbox={{{escapar(titulo)}}}]",
                      r"{\small\raggedright\sloppy", cuerpo, "}", r"\end{tcolorbox}", ""])


def archivo(ruta: Path, titulo: str) -> str:
    return caja(titulo, ruta.read_text(encoding="utf-8"), ruta.relative_to(REPO).as_posix())


def funciones_runner() -> dict:
    """ Ejecuta solo las funciones de construcción de prompts del runner, sin importar torch. """
    arbol = ast.parse(RUNNER.read_text(encoding="utf-8"))
    nodos = [n for n in arbol.body if
             (isinstance(n, ast.FunctionDef) and n.name in ("augment_prompt_with_description",
                                                           "build_flow_c_prompts"))
             or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "FLOW_C_POINTS_PER_IMAGE"
                                                    for t in n.targets))]
    espacio: dict = {}
    exec(compile(ast.Module(body=nodos, type_ignores=[]), str(RUNNER), "exec"), espacio)
    return espacio


def main() -> None:
    indice, texto = [], []

    def subsec(titulo: str, etiqueta: str) -> str:
        texto.extend([r"\newpage", rf"\section{{{titulo}}}", rf"\label{{{etiqueta}}}", ""])
        return etiqueta

    # ---- Flujo A, rondas 0 y 1 (texto ya presente en anexoA.tex): solo índice
    for tipo in ("axis", "plane"):
        for mejorado in (False, True):
            for k in range(6):
                p = f"v0{k}_1" if mejorado else f"v0{k}"
                indice.append((f"{tipo}_{p}", "Molmo2-8B", "A" + (" (ronda 1)" if mejorado else " (ronda 0)"),
                               "S, M", ORIGINALES[(tipo, mejorado)]))
    # ---- Flujo A, rondas posteriores (v06-v08)
    for tipo, nombre in (("axis", "axial"), ("plane", "planar")):
        et = subsec(rf"Simetría {nombre} — rondas posteriores del Flujo A (v06–v08)",
                    f"app:prompts_{tipo}_posteriores")
        for p in ("v06", "v07", "v08"):
            texto.append(rf"\promptsec{{{tt(f'{tipo}_{p}')}}}")
            texto.append(archivo(MOLMO / tipo / f"{p}_single.txt", f"{tipo}_{p} — vista única (single)"))
            texto.append(archivo(MOLMO / tipo / f"{p}_multi.txt", f"{tipo}_{p} — múltiples vistas (multi)"))
            indice.append((f"{tipo}_{p}", "Molmo2-8B", "A (rondas 2 y 4)", "S, M", et))
    # ---- Flujos B y C
    f = funciones_runner()
    et = subsec("Flujos B y C", "app:prompts_flujos")
    texto.append(r"\promptsec{Flujo B: descripción del objeto}")
    texto.append("Se envía con una vista del objeto (Sección~\\ref{sec:seleccion_vista}). La respuesta "
                 "se antepone al \\textit{prompt} del Flujo~A con el encabezado de la segunda caja, "
                 "generado por \\texttt{augment\\_prompt\\_with\\_description}.\n")
    texto.append(archivo(MOLMO / "description" / "describe.txt", "Flujo B — prompt de descripción"))
    texto.append(caja("Flujo B — encabezado antepuesto al prompt del Flujo A",
                      f["augment_prompt_with_description"]("<prompt del Flujo A>", MARCA_DESC),
                      "molmo_multiview_runner.py::augment_prompt_with_description"))
    indice.append(("describe", "Molmo2-8B", "B (descripción)", "S", et))
    indice.append(("encabezado del Flujo B", "Molmo2-8B", "B (pointing)", "S, M", et))
    texto.append(r"\promptsec{Flujo C: descripción y ubicación, y prompt de pointing}")
    texto.append("La primera caja de cada tipo se envía con una vista del objeto y pide una descripción "
                 "y una indicación verbal, sin coordenadas, de dónde pasa el eje o el plano. Las "
                 "otras dos son el \\textit{prompt} de \\textit{pointing}, generado por "
                 "\\texttt{build\\_flow\\_c\\_prompts} con esas dos respuestas.\n")
    for tipo, sym in (("axis", "axis_sym"), ("plane", "plane_sym")):
        texto.append(archivo(MOLMO / "description" / f"describe_and_point_{tipo}.txt",
                             f"Flujo C ({tipo}) — descripción y ubicación"))
        single, multi = f["build_flow_c_prompts"](MARCA_DESC, MARCA_UBIC, sym)
        fuente = "molmo_multiview_runner.py::build_flow_c_prompts"
        texto.append(caja(f"Flujo C ({tipo}) — pointing, vista única (single)", single, fuente))
        texto.append(caja(f"Flujo C ({tipo}) — pointing, múltiples vistas (multi)", multi, fuente))
        indice.append((f"describe_and_point_{tipo}", "Molmo2-8B", "C (descripción)", "S", et))
        indice.append((f"pointing del Flujo C ({tipo})", "Molmo2-8B", "C (pointing)", "S, M", et))
    # ---- Qwen3-VL
    et = subsec("Adaptaciones a Qwen3-VL-8B", "app:prompts_qwen")
    for tipo, p in (("axis", "v04"), ("plane", "v05_1"), ("plane", "v02")):
        nombre = f"{tipo}_{p}_qwen3vl"
        texto.append(rf"\promptsec{{{tt(nombre)}}}")
        texto.append(archivo(QWEN / tipo / f"{p}_qwen3vl_single.txt", f"{nombre} — vista única (single)"))
        texto.append(archivo(QWEN / tipo / f"{p}_qwen3vl_multi.txt", f"{nombre} — múltiples vistas (multi)"))
        indice.append((nombre, "Qwen3-VL-8B", "A", "S, M", et))
    # ---- MolmoPoint
    et = subsec("Adaptaciones a MolmoPoint-8B", "app:prompts_molmopoint")
    texto.append("MolmoPoint recibe una consulta por punto: una para el punto superior (\\textit{top}) "
                 "y otra para el inferior (\\textit{bottom}).\n")
    for tipo, variante, desc in (("axis", "v04_molmopoint", "una frase"),
                                 ("axis", "v04_molmopoint_completo", "texto completo"),
                                 ("plane", "v05_1_molmopoint", "una frase"),
                                 ("plane", "v02_molmopoint", "texto completo"),
                                 ("plane", "v02_molmopoint_frase", "una frase")):
        nombre = f"{tipo}_{variante}"
        texto.append(rf"\promptsec{{{tt(nombre)} ({desc})}}")
        for modo, modo_txt in (("single", "vista única"), ("multi", "múltiples vistas")):
            for punto in ("top", "bottom"):
                texto.append(archivo(MPOINT / tipo / f"{variante}_{modo}_{punto}.txt",
                                     f"{nombre} — {modo_txt}, {punto}"))
        indice.append((nombre, "MolmoPoint-8B", f"A ({desc})", "S, M (top y bottom)", et))

    filas = [rf"{tt(p)} & {m} & {fl} & {v} & \ref{{{e}}} \\" for p, m, fl, v, e in indice]
    tabla = [
        "% Generado por utils/generar_anexo_prompts.py. No editar a mano.",
        r"\begingroup", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
        r"\setlength{\LTcapwidth}{\textwidth}", r"\captionsetup{font=normalsize}",
        r"\begin{longtable}{@{}p{5.2cm}p{2.4cm}p{3.0cm}p{2.6cm}c@{}}",
        r"\caption{Índice de los \textit{prompts} documentados en este anexo, con la sección que "
        r"contiene su texto completo.}\label{tab:prompts_indice}\\",
        r"\toprule", r"\textbf{Prompt} & \textbf{Modelo} & \textbf{Flujo} & \textbf{Versiones} & \textbf{Texto} \\",
        r"\midrule", r"\endfirsthead",
        r"\multicolumn{5}{@{}l}{\tablename~\thetable{} (continuación)}\\", r"\toprule",
        r"\textbf{Prompt} & \textbf{Modelo} & \textbf{Flujo} & \textbf{Versiones} & \textbf{Texto} \\",
        r"\midrule", r"\endhead", r"\bottomrule",
        r"\multicolumn{5}{r@{}}{\textit{Continúa en la página siguiente}}\\", r"\endfoot", r"\bottomrule",
        r"\addlinespace[1ex]",
        r"\multicolumn{5}{@{}p{\textwidth}@{}}{\footnotesize Nota: S: versión de vista única "
        r"(\textit{single}); M: versión de múltiples vistas (\textit{multi}). Qué pide cada "
        r"\textit{prompt} se describe en la Tabla~\ref{tab:prompt-variants} y la motivación de las "
        r"rondas posteriores en la Tabla~\ref{tab:sm_prompts_nuevos}. Solo se incluyen los "
        r"\textit{prompts} usados en corridas a escala completa.}\\",
        r"\endlastfoot", *filas, r"\end{longtable}", r"\endgroup", ""]
    (SALIDA / "anexo_prompts_indice.tex").write_text("\n".join(tabla), encoding="utf-8", newline="\n")
    cabecera = ["% Generado por utils/generar_anexo_prompts.py desde los archivos de prompts. No editar a mano.", ""]
    (SALIDA / "anexo_prompts_texto.tex").write_text("\n".join(cabecera + texto), encoding="utf-8", newline="\n")
    print(f"índice: {len(indice)} prompts; texto: {sum(1 for t in texto if 'begin{tcolorbox}' in t)} cajas")


if __name__ == "__main__":
    main()
