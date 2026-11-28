#!/usr/bin/env python3
# Experimento de orientación con los tres VLMs: plane_v02 a 224px/flat, una vista por llamada, sobre
# las mallas planares rotadas al azar en SO(3) (mismos renders que run_plane_v02_224_rot_so3.py).
#
#   rot_so3_single            Molmo2, prompt plane_v02 (envío individual)
#   rot_so3_qwen3vl           Qwen3-VL-8B, prompt plane_v02_qwen3vl
#   rot_so3_molmopoint_frase  MolmoPoint-8B, versión de una frase de plane_v02, la que rinde
#                             mejor en plano sin rotar
#
# Para cada experimento: inferencia con el runner del modelo (admite renders parciales) repartida entre --gpus (los
# renders rotados solo tienen las 36 vistas de 6/14/26), S_dat con un plano, evaluate.py con
# métricas de referencia, comparación, auditoría, pareado contra la corrida SIN rotar del mismo
# modelo y envío individual, y subida a GitHub. Reusa run_plane_v02_224.py sin modificarlo.
#
#   python3 PlaneScoreData/run_plane_v02_224_rot_so3_vlms.py chequeo
#   python3 PlaneScoreData/run_plane_v02_224_rot_so3_vlms.py todo --gpus 0 1
#   python3 PlaneScoreData/run_plane_v02_224_rot_so3_vlms.py todo --gpus 0 1 --experimentos rot_so3_qwen3vl
import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "PlaneScoreData"))
import run_plane_v02_224 as base  # noqa: E402

RUNNERS = {
    "molmo": "MolmoPointing/molmo_multiview_runner.py",
    "qwen": "QwenPointing/qwen3vl_multiview_runner.py",
    "molmopoint": "MolmoPointPointing/molmopoint_multiview_runner.py",
}
SUFIJO_BASE = f"_s{base.SIZE}_flat_ndcfix_sdat_mp1_nomesh"
# nombre: experimento (formato de base.EXPERIMENTOS) y su corrida sin rotar (id y carpeta)
EXPERIMENTOS = {
    "rot_so3_single": (
        dict(tipo="plane_sym", luz="flat", modelo="molmo", exp="plane_v02_single_rot_so3",
             prompt="plane_v02", modo="single", flujo="a"),
        "plane_v02_single", "plane_v02_224_single"),
    "rot_so3_qwen3vl": (
        dict(tipo="plane_sym", luz="flat", modelo="qwen", exp="plane_v02_qwen3vl_rot_so3",
             prompt="plane_v02_qwen3vl", modo="single", flujo=None),
        "plane_v02_qwen3vl", "plane_v02_224_qwen3vl"),
    "rot_so3_molmopoint_frase": (
        dict(tipo="plane_sym", luz="flat", modelo="molmopoint",
             exp="plane_v02_molmopoint_frase_rot_so3", prompt="plane_v02_molmopoint_frase",
             modo="single", flujo=None),
        "plane_v02_molmopoint_frase", "plane_v02_224_molmopoint_frase"),
}
CODIGO = ["PlaneScoreData/run_plane_v02_224_rot_so3_vlms.py"]
COAUTOR = "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"


def log(msg: str) -> None:
    """ Mensaje con hora. """
    print(f"[{dt.datetime.now():%d/%m %H:%M}] {msg}", flush=True)


def registrar(nombre: str) -> dict:
    """ Registra el experimento en base (solo en memoria) y apunta el pareado a su corrida sin rotar.

    Returns:
        * dict: definición del experimento

    """
    e, id_base, carpeta_base = EXPERIMENTOS[nombre]
    base.EXPERIMENTOS[nombre] = e
    rutas = sorted((REPO / "resultados_finales" / carpeta_base).glob(
        "experiments_*/plane_sym_final_comparison.csv"))
    if len(rutas) != 1:
        raise FileNotFoundError(f"{carpeta_base}: se esperaba un CSV final, hay {len(rutas)}")
    base.REFERENCIA["plane_sym"] = (id_base + SUFIJO_BASE, rutas[0])
    return e


def inferir(args: argparse.Namespace, nombre: str, logs: Path) -> None:
    """ Inferencia con el runner (admite renders parciales), repartiendo los objetos entre las GPUs; retoma lo que falte. """
    e = base.EXPERIMENTOS[nombre]
    procesos = []
    for i, gpu in enumerate(args.gpus):
        comando = [base.PY, RUNNERS[e["modelo"]], "--renders-root", str(args.renders_root),
                   "--symmetry-type", e["tipo"], "--sizes", str(base.SIZE), "--lightings", e["luz"],
                   "--view-groups", *map(str, base.VISTAS), "--prompt-mode", e["modo"],
                   "--prompt-id", e["prompt"], "--experiment-id", e["exp"],
                   "--gpu-id", str(i), "--num-gpus", str(len(args.gpus)), "--yes"]
        if e["modelo"] == "molmo":
            comando += ["--flow", e["flujo"]]
        salida = open(logs / f"inferencia_{nombre}_gpu{gpu}.log", "a", encoding="utf-8")
        procesos.append(subprocess.Popen(comando, cwd=REPO, stdout=salida, stderr=subprocess.STDOUT,
                                         env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}))
    codigos = [p.wait() for p in procesos]
    if any(codigos):
        raise RuntimeError(f"la inferencia terminó con códigos {codigos}")
    tokens = args.renders_root / e["tipo"] / f"{base.MODELOS[e['modelo']][2]}_{e['exp']}.txt"
    if tokens.exists():
        shutil.copy2(tokens, logs / f"token_counts_{nombre}.txt")


def enlazar_base(args: argparse.Namespace) -> None:
    """ Enlaza en los renders rotados la evaluación sin rotar del mismo modelo, que lee el pareado. """
    id_base = base.REFERENCIA["plane_sym"][0]
    archivo = f"eval_s{base.SIZE}_flat_{id_base}_triangulation_results.json"
    origen = args.renders_base / "plane_sym" / archivo
    destino = args.renders_root / "plane_sym" / archivo
    if not origen.exists():
        raise FileNotFoundError(f"falta la evaluación sin rotar {origen}")
    if destino.is_symlink():
        destino.unlink()
    elif destino.exists():
        raise FileExistsError(f"{destino} es un archivo real; no se reemplaza")
    destino.symlink_to(origen)


def subir(nombre: str, salida: Path) -> None:
    """ Como base.subir (mismo candado), agregando este runner y un mensaje descriptivo. """
    import fcntl
    with open(REPO / ".git" / "run_plane_v02.lock", "w") as candado:
        fcntl.flock(candado, fcntl.LOCK_EX)
        subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", base.RAMA],
                       cwd=REPO, check=True)
        subprocess.run(["git", "add", str(salida.relative_to(REPO)), *CODIGO], cwd=REPO, check=True)
        if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO).returncode == 0:
            log(f"{nombre}: nada nuevo que subir")
            return
        mensaje = (f"Add plane_v02 224px results: {nombre} (SO(3)-rotated planar meshes, "
                   f"single-view sending)\n\n{COAUTOR}")
        subprocess.run(["git", "commit", "-m", mensaje], cwd=REPO, check=True)
        subprocess.run(["git", "push", "origin", base.RAMA], cwd=REPO, check=True)
        log(f"{nombre}: SUBIDO A GITHUB")


def main() -> None:
    """ Corre en orden los experimentos pedidos y se detiene en el primer error. """
    p = argparse.ArgumentParser(description="plane_v02 224px rotado SO(3) con los tres VLMs.")
    p.add_argument("etapa", choices=["chequeo", "todo"])
    p.add_argument("--experimentos", nargs="+", default=list(EXPERIMENTOS), choices=list(EXPERIMENTOS))
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    p.add_argument("--renders-root", type=Path, default=Path("/data/pespinoz/renders_rot_so3"))
    p.add_argument("--objects-root", type=Path, default=Path("/data/pespinoz/objects_rot_so3"))
    p.add_argument("--renders-base", type=Path, default=Path("/data/pespinoz/renders"),
                   help="raíz con las evaluaciones sin rotar")
    p.add_argument("--jobs", type=int, default=12, help="procesos de S_dat")
    p.add_argument("--sin-git", action="store_true")
    args = p.parse_args()

    if base.molmo_to_ndc(0, 0)[0] != 1.0:
        sys.exit("[error] pipeline_common/camera.py no tiene la corrección NDC")
    for nombre in args.experimentos:
        try:
            e = registrar(nombre)
            completos = base.revisar(args.renders_root, nombre)
            id_base = base.REFERENCIA["plane_sym"][0]
            log(f"{nombre:26} {base.archivo_puntos(e):55} completos {completos:>3}/"
                f"{base.TOTAL_OBJETOS}  pareado contra {id_base}")
            if not (args.renders_base / "plane_sym" /
                    f"eval_s{base.SIZE}_flat_{id_base}_triangulation_results.json").exists():
                raise FileNotFoundError(f"falta la evaluación sin rotar de {id_base}")
        except (FileNotFoundError, SystemExit) as error:
            sys.exit(f"[error] {nombre}: {error}")
    if args.etapa == "chequeo":
        return

    for nombre in args.experimentos:
        salida = base.carpeta_salida(nombre)      # resultados_finales/plane_v02_224_<nombre>
        logs = salida / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        log(f"=== {nombre} en GPUs {args.gpus}")
        try:
            registrar(nombre)
            if base.revisar(args.renders_root, nombre) < base.TOTAL_OBJETOS:
                inferir(args, nombre, logs)
                completos = base.revisar(args.renders_root, nombre)
                if completos < base.TOTAL_OBJETOS:
                    raise RuntimeError(f"la inferencia no completó los {base.TOTAL_OBJETOS} ({completos})")
            enlazar_base(args)
            base.posprocesar(args, nombre, salida)
            if not args.sin_git:
                subir(nombre, salida)
        except (RuntimeError, ValueError, FileExistsError, FileNotFoundError,
                subprocess.CalledProcessError) as error:
            sys.exit(f"[error] {nombre}: {error}")
    log(f"Cola terminada sin errores: {', '.join(args.experimentos)}")


if __name__ == "__main__":
    main()
