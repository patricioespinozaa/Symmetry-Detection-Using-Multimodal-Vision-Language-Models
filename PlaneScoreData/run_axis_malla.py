#!/usr/bin/env python3
# Retroproyecta a la malla los puntos de axis_v04 (224px, flat, Flujo A multiimagen) y ajusta el eje
# con SVD y RANSAC+SVD, en las 7 variantes del barrido original con malla: base (un rayo por punto),
# parche de 3x3 y 5x5 sub-rayos, agrupamiento voraz y HDBSCAN con min_samples 2, 3 y 5.
# Todo corre en CPU (trimesh, numpy, scipy, scikit-learn); no usa GPU ni vuelve a correr Molmo2.
#
#   python3 PlaneScoreData/run_axis_malla.py chequeo     # solo lee y valida; no escribe nada
#   python3 PlaneScoreData/run_axis_malla.py todo        # retroproyección, ajuste, evaluación
#   python3 PlaneScoreData/run_axis_malla.py todo --sin-referencia   # sin SDE_ref (más rápido)
#
# Los ids llevan resolución, iluminación y "_mesh", porque predicted_symmetry_<ID>.json se guarda
# por objeto y no distingue configuración. Los puntos se leen mediante un enlace
# molmo_multiview_<alias>.json -> molmo_multiview_axis_v04.json que solo existe en 224/flat, así
# que estimate_symmetry.py no puede mezclar puntos de otras resoluciones o iluminaciones.
import argparse
import csv
import datetime as dt
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "PlaneScoreData"))
from compare_plane_scores import comparar_pareado, errores_por_objeto  # noqa: E402
from run_plane_resolucion import correr, get_prompt, leer_csv, molmo_to_ndc  # noqa: E402
from pipeline_common.datasets import OBJECTS_SUBDIR  # noqa: E402

TIPO = "axis_sym"
PROMPT = "axis_v04"
SIZE = 224
LUZ = "flat"
MODO_PUNTOS = "independent"
VISTAS = (6, 14, 26)
TOTAL_OBJETOS = 850
METODOS = ("svd", "ransac_svd")  # svd_sde y ransac_svd_sde repiten estos ajustes
PARCHES = (1, 3, 5)
AGRUPAMIENTOS = {"cluster": ["--clustering-method", "greedy"],
                 **{f"hdbscan_ms{m}": ["--clustering-method", "hdbscan",
                                       "--hdbscan-min-samples", str(m)] for m in (2, 3, 5)}}
ID_SIN_MALLA = f"{PROMPT}_s{SIZE}_{LUZ}_ndcfix_expE_nomesh"
CSV_SIN_MALLA = (REPO / "resultados_finales" / "ndcfix_flujoA" / "experiments_05_11_2026"
                 / "axis_sym_final_comparison.csv")
PY = sys.executable

ALIAS = f"{PROMPT}_s{SIZE}_{LUZ}_ndcfix_mesh"


def id_variante(variante: str) -> str:
    """ Id de predicciones y evaluaciones de una variante ('base', 'p3', 'cluster', ...). """
    return ALIAS if variante == "base" else f"{ALIAS}_{variante}"


VARIANTES = ["base", "p3", "p5", *AGRUPAMIENTOS]


def carpetas_render(renders: Path) -> list[Path]:
    """ Carpetas 224/flat de axis_sym con metadatos de cámara. """
    return [d for d in sorted((renders / TIPO).glob(f"*/{SIZE}/{LUZ}"))
            if (d / "metadata_all.json").exists()]


def ruta_eval(renders: Path, experimento: str, metodo: str) -> Path:
    """ eval_*_results.json que escribe evaluate.py. """
    return renders / TIPO / f"eval_s{SIZE}_{LUZ}_{experimento}_{metodo}_results.json"


def chequear(args: argparse.Namespace) -> None:
    """ Valida código, puntos, mallas y nombres de archivo antes de ejecutar. No escribe nada.

    Args:
        * args: argumentos de la línea de comandos

    Raises:
        * SystemExit: si encuentra un problema

    """
    if molmo_to_ndc(0, 0)[0] != 1.0:
        sys.exit("[error] pipeline_common/camera.py no tiene la corrección NDC")
    mallas = args.objects_root / OBJECTS_SUBDIR[TIPO]
    for ruta in (args.renders_root / TIPO, mallas):
        if not ruta.is_dir():
            sys.exit(f"[error] no existe {ruta}")

    texto = get_prompt(PROMPT)["multi"]
    errores, avisos_texto, carpetas = [], 0, carpetas_render(args.renders_root)
    if len(carpetas) != TOTAL_OBJETOS:
        errores.append(f"{len(carpetas)} renders a {SIZE}px/{LUZ}, se esperaban {TOTAL_OBJETOS}")
    for carpeta in carpetas:
        objeto = carpeta.parents[1].name
        if not (mallas / f"{objeto}.obj").exists():
            errores.append(f"{objeto}: falta la malla")
        enlace = carpeta / f"molmo_multiview_{ALIAS}.json"
        if enlace.exists() and not enlace.is_symlink():
            errores.append(f"{enlace} es un archivo real; no se reemplaza")
        ruta = carpeta / f"molmo_multiview_{PROMPT}.json"
        if not ruta.exists():
            errores.append(f"{objeto}: falta {ruta.name}")
            continue
        grupos = json.loads(ruta.read_text(encoding="utf-8"))
        for vistas in VISTAS:
            grupo = grupos.get(str(vistas))
            if grupo is None:
                errores.append(f"{objeto}: falta el grupo de {vistas} vistas")
                continue
            if {grupo.get("experiment_id"), grupo.get("prompt_id")} - {PROMPT, None}:
                errores.append(f"{objeto}/{vistas}: id {grupo.get('experiment_id')}/"
                               f"{grupo.get('prompt_id')}")
            elif grupo.get("flow", "a") != "a":
                errores.append(f"{objeto}/{vistas}: flujo {grupo.get('flow')}")
            elif len(grupo.get("images_sent", [])) != vistas:
                errores.append(f"{objeto}/{vistas}: {len(grupo['images_sent'])} imágenes")
            elif grupo.get("prompt_used") != texto:
                avisos_texto += 1

    # El alias no debe existir en otra configuración: estimate_symmetry.py la mezclaría.
    ajenos = [p for p in (args.renders_root / TIPO).glob(f"*/*/*/molmo_multiview_{ALIAS}.json")
              if p.parent.parent.name != str(SIZE) or p.parent.name != LUZ]
    if ajenos:
        errores.append(f"el alias existe fuera de {SIZE}/{LUZ}, por ejemplo {ajenos[:2]}")

    print(f"{PROMPT} a {SIZE}px/{LUZ}: {len(carpetas)} renders; variantes "
          f"{', '.join(id_variante(v) for v in VARIANTES)}")
    if avisos_texto:
        print(f"[aviso] {avisos_texto} grupos usan un texto de prompt distinto del registro "
              "actual; son los mismos puntos que usó el pipeline sin malla")
    if not ruta_eval(args.renders_root, ID_SIN_MALLA, "triangulation").exists():
        print(f"[aviso] no está la evaluación de {ID_SIN_MALLA}; se omite la comparación pareada")
    if errores:
        sys.exit(f"[error] no se ejecutó nada ({len(errores)} problemas):\n  "
                 + "\n  ".join(errores[:20]))
    print("[ok] Chequeo sin problemas")


def crear_alias(renders: Path) -> int:
    """ Enlaza molmo_multiview_<alias>.json con los puntos de axis_v04 en cada carpeta 224/flat. """
    creados = 0
    for carpeta in carpetas_render(renders):
        enlace = carpeta / f"molmo_multiview_{ALIAS}.json"
        if enlace.is_symlink():
            enlace.unlink()
        enlace.symlink_to(f"molmo_multiview_{PROMPT}.json")
        creados += 1
    return creados


def en_paralelo(tareas: list[tuple[list, Path]], jobs: int) -> None:
    """ Ejecuta comandos independientes, a lo más jobs a la vez.

    Args:
        * tareas: pares (comando, log)
        * jobs: procesos simultáneos

    Raises:
        * RuntimeError: si algún comando falla

    """
    with ThreadPoolExecutor(max_workers=jobs) as grupo:
        futuros = [grupo.submit(correr, comando, log) for comando, log in tareas]
    for futuro in futuros:
        futuro.result()


def por_tajadas(comando: list, log: Path, jobs: int) -> list[tuple[list, Path]]:
    """ Reparte los objetos de un script en jobs procesos con --gpu-id/--num-gpus (solo CPU). """
    return [([*comando, "--gpu-id", i, "--num-gpus", jobs],
             log.with_name(f"{log.stem}_{i}de{jobs}.log")) for i in range(jobs)]


def comunes(args: argparse.Namespace) -> list:
    """ Argumentos que comparten los scripts de Mapping. """
    return ["--renders-root", args.renders_root, "--symmetry-type", TIPO,
            "--sizes", SIZE, "--lightings", LUZ]


def retroproyectar(args: argparse.Namespace, logs: Path) -> None:
    """ map_to_3d.py con un rayo por punto y con parches de 3x3 y 5x5. """
    tareas = []
    for parche in PARCHES:
        tareas += por_tajadas([PY, "Mapping/map_to_3d.py", *comunes(args),
                               "--objects-root", args.objects_root, "--experiment-id", ALIAS,
                               "--patch-size", parche, "--overwrite", "--yes"],
                              logs / f"map_p{parche}.log", args.jobs)
    en_paralelo(tareas, args.jobs)


def ajustar(args: argparse.Namespace, logs: Path) -> None:
    """ estimate_symmetry.py en las 7 variantes. """
    tareas = []
    for variante in VARIANTES:
        entrada = id_variante(variante) if variante in ("p3", "p5") else ALIAS
        extra = AGRUPAMIENTOS.get(variante, [])
        tareas += por_tajadas([PY, "Mapping/estimate_symmetry.py", *comunes(args),
                               "--experiment-id", entrada, "--point-mode", MODO_PUNTOS,
                               *extra, "--overwrite"],
                              logs / f"est_{id_variante(variante)}.log", args.jobs)
    en_paralelo(tareas, args.jobs)


def evaluar(args: argparse.Namespace, logs: Path) -> None:
    """ evaluate.py por variante y método, con SDE_ref salvo --sin-referencia. """
    referencia = [] if args.sin_referencia else ["--with-reference-metrics"]
    tareas = [([PY, "Mapping/evaluate.py", *comunes(args), "--objects-root", args.objects_root,
                "--experiment-id", id_variante(v), "--method", m, *referencia],
               logs / f"eval_{id_variante(v)}_{m}.log")
              for v in VARIANTES for m in METODOS]
    en_paralelo(tareas, args.jobs)


def comparar(args: argparse.Namespace, logs: Path) -> Path:
    """ compare_results.py con las 7 variantes; deja axis_sym_final_comparison.csv. """
    ids = [id_variante(v) for v in VARIANTES]
    correr([PY, "Mapping/compare_results.py", "--renders-root", args.renders_root,
            "--symmetry-type", TIPO, "--sizes", SIZE, "--lightings", LUZ,
            "--experiment-id", *ids, "--csv-dir", args.salida, "--no-plots",
            "--total-objects", TOTAL_OBJETOS], logs / f"compare_{TIPO}.log")
    origen = max(args.salida.glob(f"experiments_*/{TIPO}_*comparison.csv"),
                 key=lambda ruta: ruta.stat().st_mtime)
    destino = origen.with_name(f"{TIPO}_final_comparison.csv")
    if origen != destino:
        os.replace(origen, destino)
    return destino


def leer_filas(ruta: Path) -> dict[tuple[str, str, str], dict]:
    """ CSV de comparación indexado por (experimento, método, vistas). """
    with open(ruta, newline="", encoding="utf-8") as archivo:
        return {(f["experiment"], f["method"], f["n_views"]): f for f in csv.DictReader(archivo)}


def parear(args: argparse.Namespace, filas: dict) -> None:
    """ Diferencia de error angular objeto a objeto frente al pipeline sin malla (EXP-E).

    Args:
        * args: argumentos de la línea de comandos
        * filas: filas del CSV final

    """
    base_eval = ruta_eval(args.renders_root, ID_SIN_MALLA, "triangulation")
    if not base_eval.exists():
        print(f"[aviso] sin {base_eval.name}; se omite la comparación pareada")
        return
    sin_malla = leer_csv(CSV_SIN_MALLA) if CSV_SIN_MALLA.exists() else {}
    salida = []
    for vistas in VISTAS:
        base = errores_por_objeto(base_eval, str(vistas))
        fila_base = sin_malla.get((ID_SIN_MALLA, str(vistas)), {})
        for variante in VARIANTES:
            for metodo in METODOS:
                nuevo_id = id_variante(variante)
                nuevo = errores_por_objeto(ruta_eval(args.renders_root, nuevo_id, metodo),
                                           str(vistas))
                fila = filas[(nuevo_id, metodo, str(vistas))]
                salida.append({"variante": variante, "metodo": metodo, "n_views": vistas,
                               "auc_sin_malla": fila_base.get("auc_angular"),
                               "auc_malla": fila["auc_angular"],
                               "n_objects_sin_malla": fila_base.get("n_objects"),
                               "n_objects_malla": fila["n_objects"],
                               **comparar_pareado(base, nuevo)})
    ruta = args.salida / "pareado" / f"{ALIAS}_vs_{ID_SIN_MALLA}.csv"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(salida[0]))
        escritor.writeheader()
        escritor.writerows(salida)
    print(f"Pareado (error con malla - sin malla; negativo = mejora): {ruta}")


def resumir(filas: dict) -> None:
    """ Imprime las variantes ordenadas por AUC promedio, junto a la referencia sin malla. """
    sin_malla = leer_csv(CSV_SIN_MALLA) if CSV_SIN_MALLA.exists() else {}
    print(f"\n{'variante':12} {'método':11} " + " ".join(
        f"{'obj' + str(v):>6} {'AUC' + str(v):>7} {'med' + str(v):>6}" for v in VISTAS))
    referencia = [sin_malla.get((ID_SIN_MALLA, str(v))) for v in VISTAS]
    if all(referencia):
        print(f"{'sin malla':12} {'EXP-E':11} " + " ".join(
            f"{c['n_objects']:>6} {float(c['auc_angular']):7.3f} "
            f"{float(c['angular_error_median']):6.1f}" for c in referencia))
    orden = sorted(((v, m) for v in VARIANTES for m in METODOS),
                   key=lambda vm: -sum(float(filas[(id_variante(vm[0]), vm[1], str(n))]
                                             ["auc_angular"]) for n in VISTAS))
    for variante, metodo in orden:
        celdas = [filas[(id_variante(variante), metodo, str(v))] for v in VISTAS]
        print(f"{variante:12} {metodo:11} " + " ".join(
            f"{c['n_objects']:>6} {float(c['auc_angular']):7.3f} "
            f"{float(c['angular_error_median']):6.1f}" for c in celdas))


def main() -> None:
    """ Ejecuta el chequeo o el experimento completo y se detiene en el primer error. """
    parser = argparse.ArgumentParser(description="axis_v04 a 224px/flat retroproyectado a malla.")
    parser.add_argument("etapa", choices=["chequeo", "todo"])
    parser.add_argument("--renders-root", type=Path, default=Path("/data/pespinoz/renders"))
    parser.add_argument("--objects-root", type=Path, default=Path("/data/pespinoz/objects"))
    parser.add_argument("--carpeta", default="axis_malla_224",
                        help="subcarpeta de resultados_finales")
    parser.add_argument("--jobs", type=int, default=12, help="procesos de CPU simultáneos")
    parser.add_argument("--sin-referencia", action="store_true",
                        help="omite SDE_ref en evaluate.py, la etapa más lenta")
    args = parser.parse_args()
    args.salida = REPO / "resultados_finales" / args.carpeta

    chequear(args)
    if args.etapa == "chequeo":
        return
    logs = args.salida / "logs"
    try:
        print(f"{crear_alias(args.renders_root)} enlaces -> molmo_multiview_{ALIAS}.json")
        for nombre, etapa in (("retroproyección", retroproyectar), ("ajuste", ajustar),
                              ("evaluación", evaluar)):
            print(f"[{dt.datetime.now():%H:%M:%S}] {nombre}", flush=True)
            etapa(args, logs)
        ruta = comparar(args, logs)
        filas = leer_filas(ruta)
        faltan = [(id_variante(v), m, n) for v in VARIANTES for m in METODOS for n in VISTAS
                  if (id_variante(v), m, str(n)) not in filas]
        if faltan:
            raise ValueError(f"faltan filas en {ruta}: {faltan}")
        parear(args, filas)
    except (RuntimeError, ValueError) as error:
        sys.exit(f"[error] {error}")
    resumir(filas)
    print(f"\nResultados: {ruta}\nTerminó sin errores.")


if __name__ == "__main__":
    main()
