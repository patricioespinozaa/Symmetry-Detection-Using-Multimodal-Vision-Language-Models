#!/usr/bin/env python3
# Reúne en resultados_finales/planar_sdat/ los resultados de todos los experimentos planares
# post-procesados con S_dat (un plano por objeto, pipeline sin malla) y sus tablas LaTeX.
#
#   python utils/armar_planar_sdat.py
#
# - Copia (no mueve) cada plane_sym_final_comparison.csv y los CSV pareados; los originales siguen
#   en su carpeta porque otros scripts los leen (p. ej. la línea base de run_plane_v02_224.py).
# - Escribe csv/todos_planar_sdat.csv con todas las filas y su carpeta de origen.
# - Genera las tablas con generar_tablas_noviembre.py (--solo plane), generar_tablas_seleccion.py
#   (solo las planares) y generar_tablas_planar_sdat_v02.py, sin modificar esos scripts.
# - No incluye análisis ni diagnósticos (rotación SO(3), oráculo, pointing 2D) ni corridas
#   incompletas (multiplano). Se regenera completa en cada ejecución.
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTADOS = REPO / "resultados_finales"
DESTINO = RESULTADOS / "planar_sdat"
PY = sys.executable

# grupo -> [(carpeta de resultados_finales, descripción)]
GRUPOS = {
    "224_flat_prompts": [
        ("ndcfix_flujoA", "Barrido de prompts v00-v07 y variantes _1 (Flujo A, multiimagen)"),
        ("ndcfix_v08", "Prompt v08 (Flujo A, multiimagen)"),
    ],
    "resolucion": [
        ("plane_448px", "448px: v00_1, v02, v04"),
        ("plane_1136px", "1136px: v00_1, v02"),
        ("ndcfix_448", "448px: v02, v05_1, v06 (corrida anterior; v02 coincide con plane_448px)"),
        ("ndcfix_1136", "1136px: v05_1 (sin n=26)"),
    ],
    "v05_1_224": [
        ("ndcfix_224_darker", "Iluminación darker"),
        ("ndcfix_224_brighter", "Iluminación brighter"),
        ("ndcfix_flujoB", "Flujo B (v04_1 y v05_1)"),
        ("ndcfix_flujoC", "Flujo C (v04_1)"),
        ("ndcfix_single_oct26", "Envío individual"),
        ("molmopoint", "MolmoPoint-8B, envío individual"),
        ("qwen3vl", "Qwen3-VL-8B, envío individual"),
    ],
    "v02_224": [
        ("plane_v02_224_darker", "Iluminación darker"),
        ("plane_v02_224_brighter", "Iluminación brighter"),
        ("plane_v02_224_single", "Envío individual"),
        ("plane_v02_224_flujoB", "Flujo B"),
        ("plane_v02_224_flujoC", "Flujo C"),
        ("plane_v02_224_qwen3vl", "Qwen3-VL-8B, envío individual"),
        ("plane_v02_224_molmopoint", "MolmoPoint-8B, texto completo de v02"),
        ("plane_v02_224_molmopoint_frase", "MolmoPoint-8B, versión de una frase"),
    ],
}
TEX = [
    ("plane_noviembre.tex", "generar_tablas_noviembre.py --solo plane",
     "Flujo A (base, variantes, adicionales, ranking), resolución 448/1136, iluminación, flujos, "
     "modo de envío y VLMs con v05_1"),
    ("plane_top.tex", "generar_tablas_seleccion.py", "Ranking por configuración (224px, flat)"),
    ("plane_prompts.tex", "generar_tablas_seleccion.py", "Orden por prompt (224px, flat)"),
    ("plane_v02_224.tex", "generar_tablas_planar_sdat_v02.py",
     "Serie v02 a 224px: iluminación, flujos, modo de envío, VLMs y pareado contra la línea base"),
]


def csv_final(carpeta: str) -> Path:
    """ Único plane_sym_final_comparison.csv de una carpeta de resultados_finales/. """
    rutas = sorted((RESULTADOS / carpeta).glob("experiments_*/plane_sym_final_comparison.csv"))
    if len(rutas) != 1:
        raise SystemExit(f"[error] {carpeta}: se esperaba un CSV final, hay {len(rutas)}")
    return rutas[0]


def correr(*args: str) -> None:
    """ Ejecuta un generador desde la raíz del repo y se detiene si falla. """
    subprocess.run([PY, *args], cwd=REPO, check=True)


def main() -> None:
    """ Reconstruye resultados_finales/planar_sdat/. """
    if DESTINO.exists():
        shutil.rmtree(DESTINO)
    (DESTINO / "tex").mkdir(parents=True)
    todas, indice = [], []
    for grupo, carpetas in GRUPOS.items():
        for carpeta, descripcion in carpetas:
            origen = csv_final(carpeta)
            with open(origen, encoding="utf-8", newline="") as archivo:
                filas = list(csv.DictReader(archivo))
            if any("sdat" not in f["experiment"] for f in filas):
                raise SystemExit(f"[error] {origen} tiene filas que no son S_dat")
            destino = DESTINO / "csv" / grupo / f"{carpeta}.csv"
            destino.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origen, destino)
            # Las carpetas antiguas guardan también los pareados axiales (EXP-E): solo planos S_dat.
            pareados = sorted((RESULTADOS / carpeta / "pareado").glob("plane_*sdat*.csv"))
            for p in pareados:
                d = DESTINO / "csv" / grupo / "pareado" / p.name
                d.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, d)
            fecha = origen.parent.name.replace("experiments_", "")
            todas += [{"grupo": grupo, "carpeta_origen": carpeta, "fecha": fecha, **f} for f in filas]
            experimentos = sorted({f["experiment"] for f in filas})
            indice.append((grupo, carpeta, fecha, descripcion, experimentos, bool(pareados)))

    columnas = list(dict.fromkeys(k for f in todas for k in f))
    with open(DESTINO / "csv" / "todos_planar_sdat.csv", "w", encoding="utf-8", newline="") as a:
        w = csv.DictWriter(a, fieldnames=columnas)
        w.writeheader()
        w.writerows(todas)

    tex = DESTINO / "tex"
    correr("utils/generar_tablas_noviembre.py", "--solo", "plane", "--salida", str(tex / "plane_noviembre.tex"))
    with tempfile.TemporaryDirectory() as tmp:
        correr("utils/generar_tablas_seleccion.py", "--salida", tmp)
        for nombre in ("plane_top.tex", "plane_prompts.tex"):
            shutil.copy2(Path(tmp) / nombre, tex / nombre)
    correr("utils/generar_tablas_planar_sdat_v02.py", "--salida", str(tex / "plane_v02_224.tex"))

    md = ["# Resultados planares con S_dat", "",
          "Experimentos de simetría planar post-procesados con S_dat: un plano por objeto, pipeline "
          "sin malla (triangulación), `evaluate.py` con métricas de referencia y "
          "`compare_results_no_mesh.py`. Generado por `utils/armar_planar_sdat.py`; no editar a "
          "mano. Los CSV son copias: los originales siguen en su carpeta de `resultados_finales/`.",
          "", "No incluye análisis ni diagnósticos (rotación SO(3), oráculo, pointing 2D) ni la "
          "corrida multiplano, que está incompleta.", "", "## CSV", "",
          "`csv/todos_planar_sdat.csv`: todas las filas, con su grupo y carpeta de origen.", "",
          "| Grupo | Archivo | Fecha | Contenido | Experimentos | Pareado |",
          "|---|---|---|---|---|---|"]
    for grupo, carpeta, fecha, descripcion, experimentos, pareado in indice:
        md.append(f"| {grupo} | `csv/{grupo}/{carpeta}.csv` | {fecha} | {descripcion} | "
                  f"{len(experimentos)} | {'sí' if pareado else '-'} |")
    md += ["", "## Tablas LaTeX (`tex/`)", "",
           "Requieren en el preámbulo: booktabs, multirow, adjustbox, float y "
           "`\\newsavebox{\\tablebox}`.", "", "| Archivo | Generador | Contenido |", "|---|---|---|"]
    md += [f"| `tex/{n}` | `utils/{g}` | {c} |" for n, g, c in TEX]
    (DESTINO / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"{DESTINO}: {len(indice)} CSV, {len(todas)} filas, {len(TEX)} .tex")


if __name__ == "__main__":
    main()
