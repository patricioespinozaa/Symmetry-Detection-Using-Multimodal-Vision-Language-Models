#!/usr/bin/env python3
# Repite con plane_v02 los experimentos a 224px que antes se corrieron con plane_v05_1: iluminación
# darker y brighter, Flujos B y C, una vista por llamada y Qwen3-VL. Para cada experimento corre la
# inferencia (si falta), el post-procesamiento S_dat con un plano por objeto, la evaluación, la
# auditoría, una comparación pareada contra plane_v02 a 224px/flat, y sube la carpeta a GitHub.
#
#   python3 PlaneScoreData/run_plane_v02_224.py chequeo                      # solo lee
#   python3 PlaneScoreData/run_plane_v02_224.py todo --gpu 0 --experimentos darker single qwen3vl
#   python3 PlaneScoreData/run_plane_v02_224.py todo --gpu 1 --experimentos brighter flujoB flujoC
#
# Cada experimento corre fijo en una GPU. Dos colas pueden correr a la vez en GPUs distintas: escriben
# archivos distintos y el paso de git usa un candado para no chocar.
# MolmoPoint y Qwen3-VL usan versiones de v02 adaptadas a cada modelo (MolmoPointPointing/prompts y
# QwenPointing/prompts); MolmoPoint hace una consulta por punto (superior e inferior). MolmoPoint
# tiene dos versiones: "molmopoint" (texto completo de v02) y "molmopoint_frase" (una frase, como el
# MolmoPoint axial v04 existente).
import argparse
import csv
import datetime as dt
import fcntl
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "PlaneScoreData"))
from run_plane_resolucion import (correr, get_prompt, molmo_to_ndc,  # noqa: E402
                                  renombrar_comparacion)

TIPO = "plane_sym"
SIZE = 224
VISTAS = (6, 14, 26)
TOTAL_OBJETOS = 850
PROMPT = "plane_v02"
PY = sys.executable
RAMA = "refactor"
ID_REFERENCIA = f"{PROMPT}_s{SIZE}_flat_ndcfix_sdat_mp1_nomesh"
CSV_REFERENCIA = (REPO / "resultados_finales" / "ndcfix_flujoA" / "experiments_05_11_2026"
                  / "plane_sym_final_comparison.csv")

# nombre: carpeta de resultados, iluminación, modelo, id del experimento, prompt, modo y flujo.
EXPERIMENTOS = {
    "darker":   dict(luz="darker", modelo="molmo", exp=PROMPT, prompt=PROMPT, modo="auto", flujo="a"),
    "brighter": dict(luz="brighter", modelo="molmo", exp=PROMPT, prompt=PROMPT, modo="auto",
                     flujo="a"),
    "flujoB":   dict(luz="flat", modelo="molmo", exp=f"{PROMPT}_flowB", prompt=PROMPT, modo="auto",
                     flujo="b"),
    "flujoC":   dict(luz="flat", modelo="molmo", exp=f"{PROMPT}_flowC", prompt=PROMPT, modo="auto",
                     flujo="c"),
    "single":   dict(luz="flat", modelo="molmo", exp=f"{PROMPT}_single", prompt=PROMPT,
                     modo="single", flujo="a"),
    "qwen3vl":  dict(luz="flat", modelo="qwen", exp=f"{PROMPT}_qwen3vl", prompt=f"{PROMPT}_qwen3vl",
                     modo="single", flujo=None),
    "molmopoint": dict(luz="flat", modelo="molmopoint", exp=f"{PROMPT}_molmopoint",
                       prompt=f"{PROMPT}_molmopoint", modo="single", flujo=None),
    # Versión de una frase, con el mismo formato que el MolmoPoint axial v04 existente. Su texto
    # coincide con el de plane_v05_1_molmopoint.
    "molmopoint_frase": dict(luz="flat", modelo="molmopoint", exp=f"{PROMPT}_molmopoint_frase",
                             prompt=f"{PROMPT}_molmopoint_frase", modo="single", flujo=None),
}

# Runner, prefijo del JSON y prefijo del conteo de tokens de cada modelo.
MODELOS = {
    "molmo": ("MolmoPointing/molmo_multiview_runner.py", "molmo_multiview", "token_counts"),
    "qwen": ("QwenPointing/qwen3vl_multiview_runner.py", "qwen3vl_multiview",
             "token_counts_qwen3vl"),
    "molmopoint": ("MolmoPointPointing/molmopoint_multiview_runner.py", "molmopoint_multiview",
                   "token_counts_molmopoint"),
}


def archivo_puntos(e: dict) -> str:
    """ JSON que escribe el runner del modelo. """
    return f"{MODELOS[e['modelo']][1]}_{e['exp']}.json"


def id_alias(e: dict) -> str:
    """ Id del enlace de entrada del post-procesamiento. """
    return f"{e['exp']}_s{SIZE}_{e['luz']}_ndcfix_mp1_nomesh"


def id_final(e: dict) -> str:
    """ Id de las predicciones y evaluaciones de S_dat. """
    return f"{e['exp']}_s{SIZE}_{e['luz']}_ndcfix_sdat_mp1_nomesh"


def carpeta_salida(nombre: str) -> Path:
    """ Carpeta de resultados_finales de un experimento. """
    return REPO / "resultados_finales" / f"plane_v02_224_{nombre}"


def carpetas_render(renders: Path, luz: str) -> list[Path]:
    """ Carpetas 224/<luz> de plane_sym con metadatos de cámara. """
    return [d for d in sorted((renders / TIPO).glob(f"*/{SIZE}/{luz}"))
            if (d / "metadata_all.json").exists()]


def revisar(renders: Path, nombre: str) -> int:
    """ Cuenta los objetos con los 3 grupos completos y detiene si algo no corresponde.

    Args:
        * renders: raíz de los renders
        * nombre: experimento

    Returns:
        * int: objetos completos

    Raises:
        * SystemExit: si faltan renders, hay un JSON ilegible o un grupo de otra corrida

    """
    e = EXPERIMENTOS[nombre]
    carpetas = carpetas_render(renders, e["luz"])
    if len(carpetas) != TOTAL_OBJETOS:
        sys.exit(f"[error] {nombre}: {len(carpetas)} renders a {SIZE}px/{e['luz']}")
    completos, problemas = 0, []
    # En los flujos B y C el texto enviado incluye la descripción, así que no se compara.
    texto = (get_prompt(e["prompt"])["multi" if e["modo"] == "auto" else "single"]
             if e["modelo"] == "molmo" and e["flujo"] == "a" else None)
    for carpeta in carpetas:
        ruta = carpeta / archivo_puntos(e)
        if not ruta.exists():
            continue
        try:
            grupos = json.loads(ruta.read_text(encoding="utf-8"))
        except ValueError:
            problemas.append(f"{ruta} no se puede leer; bórralo para que se rehaga")
            continue
        for vistas in VISTAS:
            g = grupos.get(str(vistas))
            if g is None:
                continue
            if g.get("experiment_id") not in (e["exp"], None):
                problemas.append(f"{ruta.parent}: experiment_id {g.get('experiment_id')}")
            if e["flujo"] and g.get("flow", "a") != e["flujo"]:
                problemas.append(f"{ruta.parent}: flujo {g.get('flow', 'a')}")
            if len(g.get("images_sent", [])) != vistas:
                problemas.append(f"{ruta.parent}/{vistas}: {len(g.get('images_sent', []))} imágenes")
            if texto and g.get("prompt_used") != texto:
                problemas.append(f"{ruta.parent}/{vistas}: texto del prompt distinto del registro")
        completos += all(str(v) in grupos for v in VISTAS)
    if problemas:
        sys.exit(f"[error] {nombre}: {len(problemas)} problemas, por ejemplo {problemas[:3]}")
    return completos


def chequear(args: argparse.Namespace) -> dict[str, int]:
    """ Valida código, renders y JSON de cada experimento pedido. No escribe nada. """
    if molmo_to_ndc(0, 0)[0] != 1.0:
        sys.exit("[error] pipeline_common/camera.py no tiene la corrección NDC")
    if not CSV_REFERENCIA.exists():
        sys.exit(f"[error] falta {CSV_REFERENCIA}")
    estado = {}
    for nombre in args.experimentos:
        e = EXPERIMENTOS[nombre]
        estado[nombre] = revisar(args.renders_root, nombre)
        print(f"{nombre:9} {e['luz']:8} {archivo_puntos(e):45} completos {estado[nombre]:>3}/"
              f"{TOTAL_OBJETOS} -> {id_final(e)}")
    return estado


def inferir(args: argparse.Namespace, nombre: str, logs: Path) -> None:
    """ Corre el modelo en la GPU pedida. Retoma lo que falte. """
    e = EXPERIMENTOS[nombre]
    comunes = ["--renders-root", args.renders_root, "--symmetry-type", TIPO, "--sizes", SIZE,
               "--lightings", e["luz"], "--view-groups", *VISTAS, "--prompt-mode", e["modo"],
               "--prompt-id", e["prompt"], "--experiment-id", e["exp"], "--yes"]
    runner, _, prefijo_tokens = MODELOS[e["modelo"]]
    comando = [PY, runner, *comunes, *(["--flow", e["flujo"]] if e["modelo"] == "molmo" else [])]
    tokens = f"{prefijo_tokens}_{e['exp']}.txt"
    correr(comando, logs / f"inferencia_{nombre}.log", {"CUDA_VISIBLE_DEVICES": str(args.gpu)})
    origen = args.renders_root / TIPO / tokens
    if origen.exists():  # darker y brighter comparten el nombre; se guarda la copia de cada uno
        shutil.copy2(origen, logs / f"token_counts_{nombre}.txt")


def crear_alias(renders: Path, e: dict) -> int:
    """ Enlaza molmo_multiview_<alias>.json con los puntos del experimento. """
    nombre, creados = f"molmo_multiview_{id_alias(e)}.json", 0
    for carpeta in carpetas_render(renders, e["luz"]):
        if not (carpeta / archivo_puntos(e)).exists():
            continue
        enlace = carpeta / nombre
        if enlace.is_symlink():
            enlace.unlink()
        elif enlace.exists():
            raise FileExistsError(f"{enlace} es un archivo real; no se reemplaza")
        enlace.symlink_to(archivo_puntos(e))
        creados += 1
    return creados


def posprocesar(args: argparse.Namespace, nombre: str, salida: Path) -> None:
    """ S_dat, evaluación, comparación, auditoría y pareado contra plane_v02 a 224px/flat. """
    e, logs = EXPERIMENTOS[nombre], salida / "logs"
    inicio = f"{dt.datetime.now():%Y-%m-%d %H:%M}"
    final, luz = id_final(e), e["luz"]
    print(f"{nombre}: {crear_alias(args.renders_root, e)} enlaces -> {final}", flush=True)
    correr([PY, "PlaneScoreData/plane_score_data.py", "--renders-root", args.renders_root,
            "--input-id", id_alias(e), "--output-id", final, "--criterio", "datos",
            "--size", SIZE, "--lighting", luz, "--jobs", args.jobs, "--overwrite"],
           logs / f"est_{final}.log")
    correr([PY, "Mapping/evaluate.py", "--renders-root", args.renders_root,
            "--objects-root", args.objects_root, "--symmetry-type", TIPO, "--sizes", SIZE,
            "--lightings", luz, "--experiment-id", final, "--method", "triangulation",
            "--with-reference-metrics"], logs / f"eval_{final}.log")
    correr([PY, "Mapping/compare_results_no_mesh.py", "--renders-root", args.renders_root,
            "--symmetry-type", TIPO, "--sizes", SIZE, "--lightings", luz,
            "--experiment-id", final, "--csv-dir", salida, "--no-plots",
            "--total-objects", TOTAL_OBJETOS], logs / f"compare_{TIPO}.log")
    ruta = renombrar_comparacion(salida)
    with open(ruta, newline="", encoding="utf-8") as archivo:
        filas = {(f["experiment"], f["n_views"]): f for f in csv.DictReader(archivo)}
    faltan = [v for v in VISTAS if (final, str(v)) not in filas]
    if faltan:
        raise ValueError(f"faltan filas de {final} en {ruta}: {faltan} vistas")
    correr([PY, "PlaneScoreData/audit_run.py", "--renders-root", args.renders_root,
            "--symmetry-type", TIPO, "--input-id", id_alias(e),
            "--expected-input", archivo_puntos(e), "--output-id", final, "--estimador", "sdat",
            "--size", SIZE, "--lighting", luz, "--desde", inicio], logs / f"audit_{final}.log")
    if luz == "flat":  # el pareado lee ambas evaluaciones con la misma iluminación
        (salida / "pareado").mkdir(exist_ok=True)
        correr([PY, "PlaneScoreData/compare_plane_scores.py", "--renders-root",
                args.renders_root, "--symmetry-type", TIPO, "--comparison-csv", CSV_REFERENCIA,
                ruta, "--base-id", ID_REFERENCIA, "--new-id", final, "--size", SIZE,
                "--lighting", luz, "--salida", salida / "pareado" / f"{final}.csv"],
               logs / f"pareado_{final}.log")
    print(f"\n{nombre}: {ruta}")
    print(f"{'vistas':>6} {'obj':>4} {'AUC':>6} {'F1':>6} {'P@10':>6}")
    for v in VISTAS:
        f = filas[(final, str(v))]
        print(f"{v:>6} {f['n_objects']:>4} {float(f['auc_angular']):6.3f} "
              f"{float(f['f1_ref']):6.3f} {float(f['precision_10deg']):6.3f}")


def subir(nombre: str, salida: Path) -> None:
    """ git pull, add, commit y push de la carpeta del experimento, con candado entre colas. """
    with open(REPO / ".git" / "run_plane_v02.lock", "w") as candado:
        fcntl.flock(candado, fcntl.LOCK_EX)
        pasos = [["git", "pull", "--rebase", "--autostash", "origin", RAMA],
                 ["git", "add", str(salida.relative_to(REPO))]]
        for paso in pasos:
            subprocess.run(paso, cwd=REPO, check=True)
        if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO).returncode == 0:
            print(f"{nombre}: nada nuevo que subir")
            return
        subprocess.run(["git", "commit", "-m", f"Add plane_v02 224px results: {nombre}"],
                       cwd=REPO, check=True)
        subprocess.run(["git", "push", "origin", RAMA], cwd=REPO, check=True)
        print(f"{nombre}: SUBIDO A GITHUB", flush=True)


def main() -> None:
    """ Corre la cola de experimentos pedida y se detiene en el primer error. """
    parser = argparse.ArgumentParser(description="Experimentos a 224px con plane_v02.")
    parser.add_argument("etapa", choices=["chequeo", "todo"])
    parser.add_argument("--experimentos", nargs="+", default=list(EXPERIMENTOS),
                        choices=list(EXPERIMENTOS))
    parser.add_argument("--gpu", type=int, default=0, help="GPU fija para la inferencia")
    parser.add_argument("--renders-root", type=Path, default=Path("/data/pespinoz/renders"))
    parser.add_argument("--objects-root", type=Path, default=Path("/data/pespinoz/objects"))
    parser.add_argument("--jobs", type=int, default=12, help="procesos de S_dat")
    parser.add_argument("--sin-git", action="store_true", help="no sube los resultados")
    args = parser.parse_args()

    estado = chequear(args)
    if args.etapa == "chequeo":
        return
    for nombre in args.experimentos:
        salida = carpeta_salida(nombre)
        logs = salida / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        print(f"\n=== [{dt.datetime.now():%d/%m %H:%M}] {nombre} en GPU {args.gpu}", flush=True)
        try:
            if estado[nombre] < TOTAL_OBJETOS:
                inferir(args, nombre, logs)
                if revisar(args.renders_root, nombre) < TOTAL_OBJETOS:
                    raise RuntimeError(f"{nombre}: la inferencia no completó los {TOTAL_OBJETOS}")
            posprocesar(args, nombre, salida)
            if not args.sin_git:
                subir(nombre, salida)
        except (RuntimeError, ValueError, FileExistsError, subprocess.CalledProcessError) as error:
            sys.exit(f"[error] {nombre}: {error}")
    print(f"\nCola terminada sin errores: {', '.join(args.experimentos)}")


if __name__ == "__main__":
    main()
