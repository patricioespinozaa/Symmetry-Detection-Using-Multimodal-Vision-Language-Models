#!/usr/bin/env python3
# Post-procesamiento de plane_v02 (Molmo2, flujo A, 224px/flat) sobre las mallas planares con
# rotacion aleatoria SO(3) por objeto (ExploratoryDataAnalysis/rotar_planares.py + rotar_mallas.py).
#
# Reusa sin modificar las funciones de run_plane_v02_224.py (S_dat con un plano, evaluate.py con
# metricas de referencia, compare_results_no_mesh.py, auditoria, pareado contra plane_v02 sin rotar
# y subida a GitHub). El experimento "rot_so3" se registra solo en memoria. Diferencias:
#   - renders en /data/pespinoz/renders_rot_so3 (36 vistas: las de 6/14/26 de las 114) y objetos
#     (mallas + etiquetas rotadas) en /data/pespinoz/objects_rot_so3;
#   - la inferencia la hace MolmoPointing/molmo_multiview_runner_subset.py, no el runner original,
#     que elegiria mal las vistas en un render parcial. Este script espera a que terminen los
#     procesos de inferencia (--esperar-pid) y, si faltan objetos, la retoma una vez en --gpus;
#   - el pareado lee ambas evaluaciones desde --renders-root, asi que se enlaza ahi (solo lectura)
#     el eval_*_results.json de la linea base sin rotar. Los objetos son los mismos 850 ids.
#
#   python3 PlaneScoreData/run_plane_v02_224_rot_so3.py --esperar-pid 2100957 2100958
import argparse
import os
import shutil
import subprocess
import sys
import time
import datetime as dt
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "PlaneScoreData"))
import run_plane_v02_224 as base  # noqa: E402

NOMBRE = "rot_so3"
EXP = dict(tipo="plane_sym", luz="flat", modelo="molmo", exp=f"{base.PROMPT}_rot_so3",
           prompt=base.PROMPT, modo="auto", flujo="a")
RUNNER = "MolmoPointing/molmo_multiview_runner_subset.py"
# Codigo nuevo del experimento, que se sube junto con los resultados para poder reproducirlo.
CODIGO = ["ExploratoryDataAnalysis/rotar_planares.py", "ExploratoryDataAnalysis/rotar_mallas.py",
          "ImagesGenerator/export_fibonacci_views_subset.py", "ImagesGenerator/data_render_subset.py",
          "MolmoPointing/molmo_multiview_runner_subset.py",
          "MolmoPointPointing/molmopoint_multiview_runner_subset.py",
          "QwenPointing/qwen3vl_multiview_runner_subset.py",
          "PlaneScoreData/run_plane_v02_224_rot_so3.py"]
COAUTOR = "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"


def log(msg: str) -> None:
    """ Mensaje con hora. """
    print(f"[{dt.datetime.now():%d/%m %H:%M}] {msg}", flush=True)


def esperar(pids: list[int]) -> None:
    """ Espera a que terminen los procesos de inferencia. """
    vivos = [p for p in pids if Path(f"/proc/{p}").exists()]
    if vivos:
        log(f"esperando inferencia (PIDs {vivos})")
    while any(Path(f"/proc/{p}").exists() for p in pids):
        time.sleep(120)
    log("inferencia terminada")


def inferir(args: argparse.Namespace, logs: Path) -> None:
    """ Retoma la inferencia en todas las GPUs con el runner _subset (salta lo ya hecho). """
    procesos = []
    for i, gpu in enumerate(args.gpus):
        comando = [base.PY, RUNNER, "--renders-root", str(args.renders_root),
                   "--symmetry-type", EXP["tipo"], "--sizes", str(base.SIZE),
                   "--lightings", EXP["luz"], "--view-groups", *map(str, base.VISTAS),
                   "--prompt-mode", EXP["modo"], "--flow", EXP["flujo"],
                   "--prompt-id", EXP["prompt"], "--experiment-id", EXP["exp"],
                   "--gpu-id", str(i), "--num-gpus", str(len(args.gpus)), "--yes"]
        salida = open(logs / f"inferencia_retoma_gpu{gpu}.log", "a", encoding="utf-8")
        procesos.append(subprocess.Popen(comando, cwd=REPO, stdout=salida, stderr=subprocess.STDOUT,
                                         env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}))
    codigos = [p.wait() for p in procesos]
    if any(codigos):
        raise RuntimeError(f"la inferencia retomada termino con codigos {codigos}")


def enlazar_base(args: argparse.Namespace) -> None:
    """ Enlaza en renders_rot_so3 la evaluacion de la linea base, que lee el pareado. """
    id_base = base.REFERENCIA["plane_sym"][0]
    nombre = f"eval_s{base.SIZE}_flat_{id_base}_triangulation_results.json"
    origen = args.renders_base / "plane_sym" / nombre
    destino = args.renders_root / "plane_sym" / nombre
    if not origen.exists():
        raise FileNotFoundError(f"falta la evaluacion base {origen}")
    if destino.is_symlink():
        destino.unlink()
    elif destino.exists():
        raise FileExistsError(f"{destino} es un archivo real; no se reemplaza")
    destino.symlink_to(origen)


def subir(salida: Path) -> None:
    """ Como base.subir, con el mismo candado, pero agrega el codigo del experimento. """
    with open(REPO / ".git" / "run_plane_v02.lock", "w") as candado:
        import fcntl
        fcntl.flock(candado, fcntl.LOCK_EX)
        subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", base.RAMA],
                       cwd=REPO, check=True)
        subprocess.run(["git", "add", str(salida.relative_to(REPO)), *CODIGO], cwd=REPO, check=True)
        if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO).returncode == 0:
            log("nada nuevo que subir")
            return
        mensaje = (f"Add plane_v02 224px results: {NOMBRE} (planar meshes with random SO(3) rotation)"
                   f"\n\nRenders with the 36 views used by 6/14/26 of the 114 Fibonacci views;"
                   f" inference with the _subset runners.\n\n{COAUTOR}")
        subprocess.run(["git", "commit", "-m", mensaje], cwd=REPO, check=True)
        subprocess.run(["git", "push", "origin", base.RAMA], cwd=REPO, check=True)
        log(f"{NOMBRE}: SUBIDO A GITHUB")


def main() -> None:
    """ Espera la inferencia, la completa si falta algo, post-procesa y sube. """
    p = argparse.ArgumentParser(description="plane_v02 224px sobre planares rotadas SO(3).")
    p.add_argument("--esperar-pid", type=int, nargs="*", default=[])
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    p.add_argument("--renders-root", type=Path, default=Path("/data/pespinoz/renders_rot_so3"))
    p.add_argument("--objects-root", type=Path, default=Path("/data/pespinoz/objects_rot_so3"))
    p.add_argument("--renders-base", type=Path, default=Path("/data/pespinoz/renders"),
                   help="raiz con la evaluacion de la linea base sin rotar")
    p.add_argument("--jobs", type=int, default=12, help="procesos de S_dat")
    p.add_argument("--sin-git", action="store_true")
    args = p.parse_args()

    base.EXPERIMENTOS[NOMBRE] = EXP          # solo en memoria; el archivo original no cambia
    salida = base.carpeta_salida(NOMBRE)     # resultados_finales/plane_v02_224_rot_so3
    logs = salida / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    if base.molmo_to_ndc(0, 0)[0] != 1.0:
        sys.exit("[error] pipeline_common/camera.py no tiene la correccion NDC")

    try:
        esperar(args.esperar_pid)
        completos = base.revisar(args.renders_root, NOMBRE)
        if completos < base.TOTAL_OBJETOS:
            log(f"{completos}/{base.TOTAL_OBJETOS} completos; retomando inferencia en GPUs {args.gpus}")
            inferir(args, logs)
            completos = base.revisar(args.renders_root, NOMBRE)
            if completos < base.TOTAL_OBJETOS:
                raise RuntimeError(f"la inferencia no completo los {base.TOTAL_OBJETOS} ({completos})")
        log(f"{completos}/{base.TOTAL_OBJETOS} objetos con 6/14/26 vistas; post-procesando")
        tokens = args.renders_root / EXP["tipo"] / f"token_counts_{EXP['exp']}.txt"
        if tokens.exists():
            shutil.copy2(tokens, logs / f"token_counts_{NOMBRE}.txt")
        enlazar_base(args)
        base.posprocesar(args, NOMBRE, salida)
        if not args.sin_git:
            subir(salida)
    except (RuntimeError, ValueError, FileExistsError, FileNotFoundError,
            subprocess.CalledProcessError) as error:
        sys.exit(f"[error] {NOMBRE}: {error}")
    log(f"{NOMBRE}: terminado sin errores")


if __name__ == "__main__":
    main()
