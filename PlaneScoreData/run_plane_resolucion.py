#!/usr/bin/env python3
# Corre prompts planares a otra resolución: inferencia con Molmo2 (Flujo A, multiimagen, flat) y
# post-procesamiento S_dat con un plano por objeto. Cada prompt recibe ids propios con resolución e
# iluminación en el nombre, porque predicted_symmetry_<ID>.json no los distingue en su ruta.
#
#   python3 PlaneScoreData/run_plane_resolucion.py chequeo            # solo lee; no usa GPU
#   python3 PlaneScoreData/run_plane_resolucion.py molmo --gpus 0 1   # solo inferencias faltantes
#   python3 PlaneScoreData/run_plane_resolucion.py post               # S_dat, evaluación y auditoría
#   python3 PlaneScoreData/run_plane_resolucion.py todo --gpus 0 1    # las tres etapas en orden
#
# Por defecto usa plane_v00_1, plane_v04 y plane_v02 a 448px y escribe en
# resultados_finales/plane_448px/. Otra resolución: --size 1136 --prompts plane_v04 ...
import argparse
import csv
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "MolmoPointing"))
from pipeline_common.camera import molmo_to_ndc  # noqa: E402
from prompts_registry import get_prompt  # noqa: E402

TIPO = "plane_sym"
LUZ = "flat"
VISTAS = (6, 14, 26)
TOTAL_OBJETOS = 850
PROMPTS_DEFECTO = ("plane_v00_1", "plane_v04", "plane_v02")
PY = sys.executable
TOLERANCIA = 1e-6
# Columnas que dependen solo de los puntos y el estimador; sirven para comparar corridas repetidas.
COLUMNAS_CONTROL = ("n_objects", "angular_error_mean", "angular_error_median", "auc_angular",
                    "precision_10deg")


def archivo_puntos(prompt: str) -> str:
    """ Nombre del JSON que escribe Molmo2 para un prompt. """
    return f"molmo_multiview_{prompt}.json"


def id_alias(prompt: str, size: int) -> str:
    """ Id del enlace de entrada del post-procesamiento. """
    return f"{prompt}_s{size}_{LUZ}_ndcfix_mp1_nomesh"


def id_final(prompt: str, size: int) -> str:
    """ Id de las predicciones y evaluaciones de S_dat. """
    return f"{prompt}_s{size}_{LUZ}_ndcfix_sdat_mp1_nomesh"


def carpetas_render(renders: Path, size: int) -> list[Path]:
    """ Carpetas de render de la configuración que tienen metadatos de cámara.

    Args:
        * renders: raíz de los renders
        * size: resolución

    Returns:
        * list[Path]: una carpeta por objeto renderizado

    """
    return [d for d in sorted((renders / TIPO).glob(f"*/{size}/{LUZ}"))
            if (d / "metadata_all.json").exists()]


def revisar_grupo(grupo: dict, prompt: str, vistas: int, texto_multi: str) -> str | None:
    """ Comprueba que un grupo de vistas venga de la inferencia pedida.

    Args:
        * grupo: entrada del JSON de Molmo2
        * prompt: id esperado
        * vistas: número de vistas del grupo
        * texto_multi: texto vigente del prompt multiimagen

    Returns:
        * str | None: descripción del problema, o None si el grupo es correcto

    """
    if grupo.get("experiment_id") != prompt or grupo.get("prompt_id") != prompt:
        return f"id {grupo.get('experiment_id')}/{grupo.get('prompt_id')}"
    if grupo.get("prompt_mode") != "auto" or grupo.get("flow", "a") != "a":
        return f"modo {grupo.get('prompt_mode')}, flujo {grupo.get('flow', 'a')}"
    if grupo.get("prompt_used") != texto_multi:
        return "texto del prompt distinto del registro"
    if len(grupo.get("images_sent", [])) != vistas:
        return f"{len(grupo.get('images_sent', []))} imágenes enviadas"
    return None


def revisar_puntos(renders: Path, prompt: str, size: int) -> tuple[Counter, list[str]]:
    """ Clasifica los JSON de un prompt en la configuración pedida.

    Args:
        * renders: raíz de los renders
        * prompt: id del prompt
        * size: resolución

    Returns:
        * tuple[Counter, list[str]]: conteos por estado y detalle de lo incoherente

    """
    texto_multi = get_prompt(prompt)["multi"]
    conteo, incoherentes = Counter(), []
    for carpeta in carpetas_render(renders, size):
        conteo["renders"] += 1
        objeto = carpeta.parents[1].name
        manifiesto = carpeta / "manifest.json"
        if manifiesto.exists():
            lado = json.loads(manifiesto.read_text(encoding="utf-8")).get("image_size", size)
            if lado != size:
                incoherentes.append(f"{objeto}: render de {lado}px")
        ruta = carpeta / archivo_puntos(prompt)
        if not ruta.exists():
            conteo["sin_json"] += 1
            continue
        grupos = json.loads(ruta.read_text(encoding="utf-8"))
        faltan = [v for v in VISTAS if str(v) not in grupos]
        conteo["parciales" if faltan else "completos"] += 1
        for vistas in VISTAS:
            if str(vistas) in grupos:
                problema = revisar_grupo(grupos[str(vistas)], prompt, vistas, texto_multi)
                if problema:
                    incoherentes.append(f"{objeto}/{vistas}: {problema}")
    return conteo, incoherentes


def prompt_igual_a_224(renders: Path, prompt: str) -> bool:
    """ Indica si el texto vigente del prompt es el que se usó a 224px, la referencia de la tesis.

    Args:
        * renders: raíz de los renders
        * prompt: id del prompt

    Returns:
        * bool: False si existe una corrida a 224px con otro texto

    """
    ruta = next((renders / TIPO).glob(f"*/224/{LUZ}/{archivo_puntos(prompt)}"), None)
    if ruta is None:
        return True
    grupos = json.loads(ruta.read_text(encoding="utf-8"))
    return all(g.get("prompt_used") == get_prompt(prompt)["multi"]
               for clave, g in grupos.items() if int(clave) > 1)


def chequear(args: argparse.Namespace) -> list[str]:
    """ Valida código, prompts y archivos antes de ejecutar. No escribe nada.

    Args:
        * args: argumentos de la línea de comandos

    Returns:
        * list[str]: prompts cuya inferencia falta completar

    Raises:
        * SystemExit: si encuentra un problema

    """
    if molmo_to_ndc(0, 0)[0] != 1.0:
        sys.exit("[error] pipeline_common/camera.py no tiene la corrección NDC")
    for ruta in (args.renders_root / TIPO, args.objects_root):
        if not ruta.is_dir():
            sys.exit(f"[error] no existe {ruta}")

    pendientes, errores = [], []
    for prompt in args.prompts:
        conteo, incoherentes = revisar_puntos(args.renders_root, prompt, args.size)
        print(f"{prompt} a {args.size}px: renders {conteo['renders']}, completos "
              f"{conteo['completos']}, parciales {conteo['parciales']}, sin JSON "
              f"{conteo['sin_json']} -> id final {id_final(prompt, args.size)}")
        if conteo["renders"] != TOTAL_OBJETOS:
            errores.append(f"{prompt}: {conteo['renders']} renders, se esperaban {TOTAL_OBJETOS}")
        if not prompt_igual_a_224(args.renders_root, prompt):
            errores.append(f"{prompt}: el texto vigente difiere del usado a 224px")
        if incoherentes:
            errores.append(f"{prompt}: {len(incoherentes)} grupos no coinciden con la inferencia "
                           f"pedida; por ejemplo {incoherentes[:3]}")
        if conteo["completos"] != TOTAL_OBJETOS:
            pendientes.append(prompt)
    if errores:
        sys.exit("[error] no se ejecutó nada:\n  " + "\n  ".join(errores))
    print(f"[ok] Inferencia pendiente: {pendientes or 'ninguna'}")
    return pendientes


def correr(comando: list, log: Path, entorno: dict | None = None) -> None:
    """ Ejecuta un comando desde la raíz del repo y guarda su salida.

    Args:
        * comando: programa y argumentos
        * log: archivo de salida
        * entorno: variables que se agregan al entorno

    Raises:
        * RuntimeError: si el comando termina con error

    """
    texto = [str(c) for c in comando]
    print(f"[{dt.datetime.now():%H:%M:%S}] {log.stem}", flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8") as salida:
        salida.write(f"$ {' '.join(texto)}\n\n")
        salida.flush()
        resultado = subprocess.run(texto, stdout=salida, stderr=subprocess.STDOUT, cwd=REPO,
                                   env={**os.environ, **(entorno or {})}, check=False)
    if resultado.returncode != 0:
        raise RuntimeError(f"falló {log.stem} (código {resultado.returncode}); revisa {log}")


def respaldar_tokens(renders: Path, prompt: str, size: int) -> None:
    """ Respalda el conteo de tokens, porque el runner lo reescribe sin indicar la resolución.

    Args:
        * renders: raíz de los renders
        * prompt: id del prompt
        * size: resolución de la corrida que lo reescribirá

    """
    origen = renders / TIPO / f"token_counts_{prompt}.txt"
    destino = origen.with_name(f"token_counts_{prompt}_antes_s{size}.txt")
    if origen.exists() and not destino.exists():
        shutil.copy2(origen, destino)


def inferir(args: argparse.Namespace, pendientes: list[str]) -> None:
    """ Corre Molmo2 con los prompts pendientes, en secuencia dentro de cada GPU.

    Args:
        * args: argumentos de la línea de comandos
        * pendientes: prompts por inferir

    """
    logs = args.salida / "logs"

    def correr_en_gpu(gpu: str, prompts: list[str]) -> None:
        """ Corre en orden los prompts asignados a una GPU. """
        for prompt in prompts:
            respaldar_tokens(args.renders_root, prompt, args.size)
            correr([PY, "MolmoPointing/molmo_multiview_runner.py",
                    "--renders-root", args.renders_root, "--symmetry-type", TIPO,
                    "--sizes", args.size, "--lightings", LUZ,
                    "--view-groups", *VISTAS, "--prompt-mode", "auto", "--flow", "a",
                    "--prompt-id", prompt, "--experiment-id", prompt, "--yes"],
                   logs / f"molmo_{prompt}.log", {"CUDA_VISIBLE_DEVICES": gpu})

    asignacion = {gpu: pendientes[i::len(args.gpus)] for i, gpu in enumerate(args.gpus)}
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as grupo:
        tareas = [grupo.submit(correr_en_gpu, g, p) for g, p in asignacion.items() if p]
    for tarea in tareas:
        tarea.result()


def crear_alias(renders: Path, prompt: str, size: int) -> int:
    """ Enlaza molmo_multiview_<alias>.json con los puntos del prompt en cada render.

    Args:
        * renders: raíz de los renders
        * prompt: id del prompt
        * size: resolución

    Returns:
        * int: enlaces creados

    Raises:
        * FileExistsError: si el alias ya existe como archivo real

    """
    nombre = archivo_puntos(id_alias(prompt, size))
    creados = 0
    for carpeta in carpetas_render(renders, size):
        if not (carpeta / archivo_puntos(prompt)).exists():
            continue
        enlace = carpeta / nombre
        if enlace.is_symlink():
            enlace.unlink()
        elif enlace.exists():
            raise FileExistsError(f"{enlace} es un archivo real; no se reemplaza")
        enlace.symlink_to(archivo_puntos(prompt))
        creados += 1
    return creados


def renombrar_comparacion(salida: Path) -> Path:
    """ Deja el CSV más reciente con el nombre que usan las demás carpetas de resultados.

    Args:
        * salida: carpeta de resultados

    Returns:
        * Path: ruta final del CSV

    """
    origen = max(salida.glob(f"experiments_*/{TIPO}_*comparison.csv"),
                 key=lambda ruta: ruta.stat().st_mtime)
    destino = origen.with_name(f"{TIPO}_final_comparison.csv")
    if origen != destino:
        os.replace(origen, destino)
    return destino


def leer_csv(ruta: Path) -> dict[tuple[str, str], dict]:
    """ Lee un CSV de comparación indexado por (experimento, vistas). """
    with open(ruta, newline="", encoding="utf-8") as archivo:
        return {(f["experiment"], f["n_views"]): f for f in csv.DictReader(archivo)}


def comprobar_resultados(ruta: Path, ids: list[str], salida: Path) -> dict:
    """ Exige todas las filas y que un id ya calculado reproduzca sus métricas angulares.

    Args:
        * ruta: CSV nuevo
        * ids: experimentos esperados
        * salida: carpeta de esta corrida, excluida de la comparación

    Returns:
        * dict: filas del CSV nuevo

    Raises:
        * ValueError: si faltan filas o una métrica repetida cambió

    """
    filas = leer_csv(ruta)
    faltan = [(i, str(v)) for i in ids for v in VISTAS if (i, str(v)) not in filas]
    if faltan:
        raise ValueError(f"faltan filas en {ruta}: {faltan}")

    for previo in sorted(salida.parent.glob(f"*/experiments_*/{TIPO}_final_comparison.csv")):
        if salida in previo.parents:
            continue
        for clave, fila in leer_csv(previo).items():
            if clave not in filas:
                continue
            for columna in COLUMNAS_CONTROL:
                diferencia = abs(float(fila[columna]) - float(filas[clave][columna]))
                if diferencia > TOLERANCIA:
                    raise ValueError(f"{clave} no reproduce {columna} de {previo}: "
                                     f"{fila[columna]} frente a {filas[clave][columna]}")
            print(f"[ok] {clave[0]}, {clave[1]} vistas: igual a {previo.parts[-3]}")
    return filas


def posprocesar(args: argparse.Namespace) -> None:
    """ Corre S_dat, evaluación, comparación y auditoría de todos los prompts.

    Args:
        * args: argumentos de la línea de comandos

    Raises:
        * RuntimeError: si un paso falla o la auditoría encuentra problemas

    """
    inicio = f"{dt.datetime.now():%Y-%m-%d %H:%M}"
    logs = args.salida / "logs"
    size = args.size
    finales = [id_final(p, size) for p in args.prompts]
    for prompt, final in zip(args.prompts, finales):
        print(f"{prompt}: {crear_alias(args.renders_root, prompt, size)} enlaces -> {final}")
        correr([PY, "PlaneScoreData/plane_score_data.py", "--renders-root", args.renders_root,
                "--input-id", id_alias(prompt, size), "--output-id", final, "--criterio", "datos",
                "--size", size, "--lighting", LUZ, "--jobs", args.jobs, "--overwrite"],
               logs / f"est_{final}.log")
        correr([PY, "Mapping/evaluate.py", "--renders-root", args.renders_root,
                "--objects-root", args.objects_root, "--symmetry-type", TIPO,
                "--sizes", size, "--lightings", LUZ, "--experiment-id", final,
                "--method", "triangulation", "--with-reference-metrics"],
               logs / f"eval_{final}.log")

    correr([PY, "Mapping/compare_results_no_mesh.py", "--renders-root", args.renders_root,
            "--symmetry-type", TIPO, "--sizes", size, "--lightings", LUZ,
            "--experiment-id", *finales, "--csv-dir", args.salida, "--no-plots",
            "--total-objects", TOTAL_OBJETOS],
           logs / f"compare_{TIPO}.log")
    ruta = renombrar_comparacion(args.salida)

    for prompt, final in zip(args.prompts, finales):
        correr([PY, "PlaneScoreData/audit_run.py", "--renders-root", args.renders_root,
                "--symmetry-type", TIPO, "--input-id", id_alias(prompt, size),
                "--expected-input", archivo_puntos(prompt), "--output-id", final,
                "--estimador", "sdat", "--size", size, "--lighting", LUZ, "--desde", inicio],
               logs / f"audit_{final}.log")

    filas = comprobar_resultados(ruta, finales, args.salida)
    print(f"\nResultados: {ruta}")
    print(f"{'experimento':50} {'vistas':>6} {'obj':>4} {'AUC':>6} {'F1':>6} {'P@10':>6}")
    for final in finales:
        for vistas in VISTAS:
            fila = filas[(final, str(vistas))]
            print(f"{final:50} {vistas:>6} {fila['n_objects']:>4} "
                  f"{float(fila['auc_angular']):6.3f} {float(fila['f1_ref']):6.3f} "
                  f"{float(fila['precision_10deg']):6.3f}")


def main() -> None:
    """ Ejecuta la etapa pedida y se detiene en el primer error. """
    parser = argparse.ArgumentParser(description="Molmo2 y S_dat para prompts planares.")
    parser.add_argument("etapa", choices=["chequeo", "molmo", "post", "todo"])
    parser.add_argument("--renders-root", type=Path, default=Path("/data/pespinoz/renders"))
    parser.add_argument("--objects-root", type=Path, default=Path("/data/pespinoz/objects"))
    parser.add_argument("--size", type=int, default=448)
    parser.add_argument("--prompts", nargs="+", default=list(PROMPTS_DEFECTO))
    parser.add_argument("--carpeta", default=None,
                        help="subcarpeta de resultados_finales; por defecto plane_<size>px")
    parser.add_argument("--gpus", nargs="+", default=["0"], help="ids de GPU para Molmo2")
    parser.add_argument("--jobs", type=int, default=12, help="procesos de S_dat")
    args = parser.parse_args()
    args.salida = REPO / "resultados_finales" / (args.carpeta or f"plane_{args.size}px")
    if "_nomesh" in "".join(args.prompts) or not all(p.startswith("plane_") for p in args.prompts):
        parser.error("--prompts recibe ids de prompt planar, como plane_v04")

    pendientes = chequear(args)
    try:
        if args.etapa in ("molmo", "todo") and pendientes:
            inferir(args, pendientes)
            pendientes = chequear(args)
        if args.etapa in ("post", "todo"):
            if pendientes:
                sys.exit(f"[error] falta completar la inferencia de {pendientes}; "
                         "repite la etapa molmo, que retoma lo que falta")
            posprocesar(args)
    except (RuntimeError, ValueError, FileExistsError) as error:
        sys.exit(f"[error] {error}")
    print("Terminó sin errores.")


if __name__ == "__main__":
    main()
