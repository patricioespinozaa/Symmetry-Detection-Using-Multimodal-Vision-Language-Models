#!/usr/bin/env python3
# Genera la documentación de prompts del anexo de prompts desde los archivos que se enviaron a los
# modelos:
#   - Tesis_DCC_MDS/anexos/C1_prompts_indice.tex: tabla con todos los prompts del anexo (modelo,
#     flujo, versiones y subsección donde está su texto);
#   - Tesis_DCC_MDS/anexos/C3_prompts_posteriores.tex: texto completo de los prompts que no están
#     escritos a mano (Flujo A v06-v08, Flujos B y C, Qwen3-VL y MolmoPoint).
# Los textos de v00-v05 y v00_1-v05_1 ya están escritos a mano en anexos/C2_prompts_originales.tex
# y solo se indexan.
# Solo se documentan las versiones usadas en los resultados: Molmo2 con la versión multi (y además
# la single en SINGLE_MOLMO2), y Qwen3-VL y MolmoPoint solo con la single (una vista por llamada).
# Los prompts que el runner arma en código (encabezado del Flujo B y plantilla del Flujo C) se
# obtienen ejecutando las funciones reales de MolmoPointing/molmo_multiview_runner.py, sin importar
# el módulo (que requiere torch), con marcadores en lugar de la descripción.
#
#   python utils/generar_anexo_prompts.py
import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SALIDA = REPO / "Tesis_DCC_MDS" / "anexos"
MOLMO = REPO / "MolmoPointing" / "prompts"
QWEN = REPO / "QwenPointing" / "prompts"
MPOINT = REPO / "MolmoPointPointing" / "prompts"
RUNNER = REPO / "MolmoPointing" / "molmo_multiview_runner.py"
MARCA_DESC, MARCA_UBIC = "<descripción generada por el modelo>", "<ubicación generada por el modelo>"
ORIGINALES = {  # subsecciones escritas a mano en anexos/C2_prompts_originales.tex
    ("axis", False): "app:prompts_axis_base", ("axis", True): "app:prompts_axis_mejorados",
    ("plane", False): "app:prompts_plane_base", ("plane", True): "app:prompts_plane_mejorados",
}
# Prompts de Molmo2 que también se usaron con la versión de vista única (envío de una vista por llamada)
SINGLE_MOLMO2 = {"axis_v04", "plane_v02"}
# Ronda en que se introdujo cada prompt posterior del Flujo A
RONDA = {"v06": 2, "v07": 2, "v08": 3}
# Título y texto previo a la caja de cada prompt generado, con el mismo formato que los prompts
# escritos a mano en C2_prompts_originales.tex (encabezado "nombre — título" y párrafo Estrategia)
PRESENTACION = {
    "axis_v06": ("Polos con centro de curvatura",
                 r"pide los polos del eje como \texttt{v04\_1} y agrega una regla para las tapas "
                 r"redondeadas, que no tienen un punto de salida del eje bien definido, ubicando el polo en "
                 r"el centro de curvatura de la tapa."),
    "axis_v07": ("Corte más angosto",
                 r"pide un punto de la línea central en el corte más angosto del objeto (cuello o cintura) y "
                 r"otro cerca del extremo superior o inferior. Es un contraste controlado que invierte el "
                 r"ancla del corte más ancho de \texttt{v02}."),
    "axis_v08": ("Polos con verificación contra el colapso",
                 r"pide los polos con un paso de geometría global y un paso por vista, y agrega una "
                 r"verificación que evita que los puntos colapsen en la columna central de la imagen "
                 r"($x\approx 500$)."),
    "plane_v06": ("Costura sin pares bilaterales",
                  r"pide la costura de \texttt{v02} con una restricción que impide devolver pares "
                  r"bilaterales, y usa el centro horizontal del objeto, como \texttt{v04}, cuando la costura "
                  r"no es visible."),
    "plane_v07": ("Punto medio anclado al plano",
                  r"pide los puntos medios de \texttt{v04\_1}, pero anclados a la orientación del plano y no "
                  r"a la vertical de la imagen, para no suponer que el plano se ve vertical."),
    "plane_v08": ("Costura con verificación contra el colapso",
                  r"pide la costura con una verificación contra el colapso en la columna central análoga a "
                  r"la de \texttt{axis\_v08}."),
    "axis_v04_qwen3vl": ("Adaptación de \\texttt{axis\\_v04}",
                         r"conserva el texto de \texttt{axis\_v04} y solo reemplaza el formato de salida por el "
                         r"JSON nativo de Qwen3-VL, con la clave \texttt{point\_2d} en escala $[0, 1000]$."),
    "plane_v02_qwen3vl": ("Adaptación de \\texttt{plane\\_v02}",
                          r"conserva el texto de \texttt{plane\_v02} y solo reemplaza el formato de salida por "
                          r"el JSON nativo de Qwen3-VL, con la clave \texttt{point\_2d} en escala $[0, 1000]$."),
    "axis_v04_molmopoint": ("Una frase",
                            r"usa una sola frase de contexto y una consulta por polo, superior (\textit{top}) e "
                            r"inferior (\textit{bottom}), porque MolmoPoint tiende a responder con texto ante "
                            r"instrucciones largas."),
    "axis_v04_molmopoint_completo": ("Texto completo",
                                     r"usa el texto completo de \texttt{axis\_v04} con una consulta por polo, en "
                                     r"la que solo cambia la instrucción final."),
    "plane_v02_molmopoint": ("Texto completo",
                             r"usa el texto completo de \texttt{plane\_v02} con una consulta por extremo de la "
                             r"costura, en la que solo cambia la instrucción final."),
    "plane_v02_molmopoint_frase": ("Una frase",
                                   r"usa una sola frase de contexto y una consulta por extremo de la costura, "
                                   r"igual que \texttt{axis\_v04\_molmopoint}."),
}


def encabezado_prompt(nombre: str) -> list[str]:
    """ Encabezado 'nombre — título' y párrafo Estrategia previos a la caja de un prompt. """
    titulo, estrategia = PRESENTACION[nombre]
    return [rf"\promptsec{{{tt(nombre)} — {titulo}}}", "", rf"\textit{{Estrategia}}: {estrategia}", ""]
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
        # Sin \newpage: las secciones y los prompts siguen uno tras otro en las páginas
        texto.extend([rf"\section{{{titulo}}}", rf"\label{{{etiqueta}}}", ""])
        return etiqueta

    # ---- Flujo A, rondas 0 y 1 (texto ya presente en C2_prompts_originales.tex): solo índice
    for tipo in ("axis", "plane"):
        for mejorado in (False, True):
            for k in range(6):
                p = f"v0{k}_1" if mejorado else f"v0{k}"
                nombre = f"{tipo}_{p}"
                indice.append((nombre, "Molmo2-8B", "A" + (" (ronda 1)" if mejorado else " (ronda 0)"),
                               "S, M" if nombre in SINGLE_MOLMO2 else "M", ORIGINALES[(tipo, mejorado)]))
    # ---- Flujo A, rondas posteriores (v06-v08); solo se usó la versión multi
    for tipo, nombre in (("axis", "axial"), ("plane", "planar")):
        et = subsec(rf"Simetría {nombre}: rondas posteriores del Flujo A (v06 a v08)",
                    f"app:prompts_{tipo}_posteriores")
        for p in ("v06", "v07", "v08"):
            texto.extend(encabezado_prompt(f"{tipo}_{p}"))
            texto.append(archivo(MOLMO / tipo / f"{p}_multi.txt", f"{tipo}_{p} — múltiples vistas (multi)"))
            indice.append((f"{tipo}_{p}", "Molmo2-8B", f"A (ronda {RONDA[p]})", "M", et))
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
    indice.append(("encabezado del Flujo B", "Molmo2-8B", "B (pointing)", "M", et))
    texto.append(r"\promptsec{Flujo C: descripción y ubicación, y prompt de pointing}")
    texto.append("La primera caja de cada tipo se envía con una vista del objeto y pide una descripción "
                 "y una indicación verbal, sin coordenadas, de dónde pasa el eje o el plano. La "
                 "otra es el \\textit{prompt} de \\textit{pointing}, generado por "
                 "\\texttt{build\\_flow\\_c\\_prompts} con esas dos respuestas.\n")
    for tipo, sym in (("axis", "axis_sym"), ("plane", "plane_sym")):
        texto.append(archivo(MOLMO / "description" / f"describe_and_point_{tipo}.txt",
                             f"Flujo C ({tipo}) — descripción y ubicación"))
        _, multi = f["build_flow_c_prompts"](MARCA_DESC, MARCA_UBIC, sym)
        fuente = "molmo_multiview_runner.py::build_flow_c_prompts"
        texto.append(caja(f"Flujo C ({tipo}) — pointing, múltiples vistas (multi)", multi, fuente))
        indice.append((f"describe_and_point_{tipo}", "Molmo2-8B", "C (descripción)", "S", et))
        indice.append((f"pointing del Flujo C ({tipo})", "Molmo2-8B", "C (pointing)", "M", et))
    # ---- Qwen3-VL: solo la versión de vista única
    et = subsec("Adaptaciones a Qwen3-VL-8B", "app:prompts_qwen")
    for tipo, p in (("axis", "v04"), ("plane", "v02")):
        nombre = f"{tipo}_{p}_qwen3vl"
        texto.extend(encabezado_prompt(nombre))
        texto.append(archivo(QWEN / tipo / f"{p}_qwen3vl_single.txt", f"{nombre} — vista única (single)"))
        indice.append((nombre, "Qwen3-VL-8B", "A", "S", et))
    # ---- MolmoPoint: solo la versión de vista única
    et = subsec("Adaptaciones a MolmoPoint-8B", "app:prompts_molmopoint")
    texto.append("MolmoPoint recibe una consulta por punto: una para el punto superior (\\textit{top}) "
                 "y otra para el inferior (\\textit{bottom}).\n")
    for tipo, variante, desc in (("axis", "v04_molmopoint", "una frase"),
                                 ("axis", "v04_molmopoint_completo", "texto completo"),
                                 ("plane", "v02_molmopoint", "texto completo"),
                                 ("plane", "v02_molmopoint_frase", "una frase")):
        nombre = f"{tipo}_{variante}"
        texto.extend(encabezado_prompt(nombre))
        for punto in ("top", "bottom"):
            texto.append(archivo(MPOINT / tipo / f"{variante}_single_{punto}.txt",
                                 f"{nombre} — vista única, {punto}"))
        indice.append((nombre, "MolmoPoint-8B", f"A ({desc})", "S (top y bottom)", et))

    # Línea gris fina entre filas y una más marcada al cambiar de modelo o de flujo (estilo de la Tabla A.4)
    filas = []
    for k, (p, m, fl, v, e) in enumerate(indice):
        fila = rf"{tt(p)} & {m} & {fl} & {v} & \ref{{{e}}} \\"
        if k + 1 < len(indice):
            grupo = lambda x: (x[1], x[2][0])  # noqa: E731  modelo y flujo (A, B o C)
            fila += r" \cambio" if grupo(indice[k + 1]) != grupo(indice[k]) else r" \filasep"
        filas.append(fila)
    tabla = [
        "% Generado por utils/generar_anexo_prompts.py. No editar a mano.",
        r"\begingroup", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
        r"\setlength{\LTcapwidth}{\textwidth}", r"\captionsetup{font=normalsize}",
        r"\newcommand{\filasep}{\arrayrulecolor{gray!40}\specialrule{0.3pt}{0pt}{0pt}\arrayrulecolor{black}}",
        r"\newcommand{\cambio}{\arrayrulecolor{gray!70}\specialrule{0.6pt}{0pt}{0pt}\arrayrulecolor{black}}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.33\textwidth}"
        r">{\raggedright\arraybackslash}p{0.16\textwidth}>{\raggedright\arraybackslash}p{0.19\textwidth}"
        r">{\raggedright\arraybackslash}p{0.17\textwidth}>{\centering\arraybackslash}p{0.07\textwidth}@{}}",
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
        r"\textit{prompt} y la motivación de las rondas posteriores se describen en la "
        r"Tabla~\ref{tab:prompt-variants}. Solo se incluyen los "
        r"\textit{prompts} y las versiones que se usan en los resultados.}\\",
        r"\endlastfoot", *filas, r"\end{longtable}", r"\endgroup", ""]
    SALIDA.mkdir(parents=True, exist_ok=True)
    (SALIDA / "C1_prompts_indice.tex").write_text("\n".join(tabla), encoding="utf-8", newline="\n")
    cabecera = ["% Generado por utils/generar_anexo_prompts.py desde los archivos de prompts. No editar a mano.", ""]
    (SALIDA / "C3_prompts_posteriores.tex").write_text("\n".join(cabecera + texto), encoding="utf-8",
                                                       newline="\n")
    print(f"índice: {len(indice)} prompts; texto: {sum(1 for t in texto if 'begin{tcolorbox}' in t)} cajas")


if __name__ == "__main__":
    main()
