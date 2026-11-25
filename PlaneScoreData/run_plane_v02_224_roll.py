#!/usr/bin/env python3
# Control del experimento de orientación: objetos planares en su pose canónica, pero cada cámara
# girada al azar en torno a su eje óptico (ImagesGenerator/export_fibonacci_views_subset.py
# --roll-seed). Las posiciones y direcciones de las cámaras son las de siempre, así que las vistas
# de canto y la geometría 3D no cambian (el techo del estimador con puntos perfectos es el mismo que
# sin giro); solo cambia que la traza del plano deja de ser vertical en la imagen. Compara contra
# plane_v02 sin giro para separar "el modelo depende de que la traza sea vertical en la imagen" de
# "faltan vistas de canto" (lo que confunde el experimento rot_so3).
#
# Solo n_views = 6: se renderizan únicamente las 6 vistas de ese grupo (índices 0, 23, 45, 68, 90 y
# 113 de las 114 Fibonacci). Molmo2, plane_v02, 224px/flat, envío multiimagen y Flujo A, igual que
# la línea base. Pasos: render (si falta), inferencia repartida en --gpus, S_dat con un plano,
# evaluate.py con métricas de referencia, comparación, auditoría, pareado contra plane_v02 sin giro,
# diagnóstico de pointing 2D y subida a GitHub. Reusa run_plane_v02_224.py sin modificarlo.
#
#   python3 PlaneScoreData/run_plane_v02_224_roll.py chequeo
#   python3 PlaneScoreData/run_plane_v02_224_roll.py todo --gpus 0 1
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "PlaneScoreData"))
import run_plane_v02_224 as base  # noqa: E402
import run_plane_v02_224_rot_so3 as rot  # noqa: E402

NOMBRE = "roll"
VISTAS = (6,)
SEMILLA_GIRO = 0
REPO_VIEWS = 114
EXP = dict(tipo="plane_sym", luz="flat", modelo="molmo", exp=f"{base.PROMPT}_roll",
           prompt=base.PROMPT, modo="auto", flujo="a")
CODIGO = ["ImagesGenerator/export_fibonacci_views_subset.py", "ImagesGenerator/data_render_subset.py",
          "PlaneScoreData/run_plane_v02_224_roll.py"]
COAUTOR = "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"


def log(msg: str) -> None:
    """ Mensaje con hora. """
    print(f"[{dt.datetime.now():%d/%m %H:%M}] {msg}", flush=True)


def renders_listos(args: argparse.Namespace) -> int:
    """ Cuenta los objetos con las 6 vistas renderizadas con giro y detiene si hay renders sin giro.

    Args:
        * args: argumentos del script

    Returns:
        * int: objetos con render completo

    Raises:
        * SystemExit: si algún render existente no tiene el giro de --roll-seed 0

    """
    listos, problemas = 0, []
    for carpeta in sorted((args.renders_root / EXP["tipo"]).glob(f"*/{base.SIZE}/{EXP['luz']}")):
        manifest, metadata = carpeta / "manifest.json", carpeta / "metadata_all.json"
        if not (manifest.exists() and metadata.exists()):
            continue
        if json.loads(manifest.read_text(encoding="utf-8")).get("roll_seed") != SEMILLA_GIRO:
            problemas.append(f"{carpeta}: render sin roll_seed={SEMILLA_GIRO}")
            continue
        vistas = json.loads(metadata.read_text(encoding="utf-8"))
        if len(vistas) != VISTAS[0] or any("roll_deg" not in v for v in vistas):
            problemas.append(f"{carpeta}: {len(vistas)} vistas o falta roll_deg")
            continue
        listos += 1
    if problemas:
        sys.exit(f"[error] {len(problemas)} renders no corresponden, por ejemplo {problemas[:3]}")
    return listos


def renderizar(args: argparse.Namespace, logs: Path) -> None:
    """ Renderiza las 6 vistas con giro de los objetos que falten (salta los ya hechos). """
    comando = [base.PY, "ImagesGenerator/data_render_subset.py",
               "--input-folder", str(args.objects_root / "curated_plane_sym_obj"),
               "--output-folder", str(args.renders_root), "--symmetry-type", EXP["tipo"],
               "--repo-views", str(REPO_VIEWS), "--sizes", str(base.SIZE), "--lightings", EXP["luz"],
               "--subset-n-views", *map(str, VISTAS), "--roll-seed", str(SEMILLA_GIRO),
               "--device", "cuda:0", "--skip-existing", "--yes"]
    with open(logs / "render.log", "a", encoding="utf-8") as salida:
        codigo = subprocess.run(comando, cwd=REPO, stdout=salida, stderr=subprocess.STDOUT,
                                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpus[0])}).returncode
    if codigo:
        raise RuntimeError(f"el render terminó con código {codigo}")


def inferir(args: argparse.Namespace, logs: Path) -> None:
    """ Inferencia con el runner _subset de Molmo2, repartida en las GPUs; retoma lo que falte. """
    procesos = []
    for i, gpu in enumerate(args.gpus):
        comando = [base.PY, rot.RUNNER, "--renders-root", str(args.renders_root),
                   "--symmetry-type", EXP["tipo"], "--sizes", str(base.SIZE),
                   "--lightings", EXP["luz"], "--view-groups", *map(str, VISTAS),
                   "--prompt-mode", EXP["modo"], "--flow", EXP["flujo"],
                   "--prompt-id", EXP["prompt"], "--experiment-id", EXP["exp"],
                   "--gpu-id", str(i), "--num-gpus", str(len(args.gpus)), "--yes"]
        salida = open(logs / f"inferencia_gpu{gpu}.log", "a", encoding="utf-8")
        procesos.append(subprocess.Popen(comando, cwd=REPO, stdout=salida, stderr=subprocess.STDOUT,
                                         env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}))
    codigos = [p.wait() for p in procesos]
    if any(codigos):
        raise RuntimeError(f"la inferencia terminó con códigos {codigos}")


def diagnostico_2d(args: argparse.Namespace, salida: Path) -> None:
    """ Distancia 2D de los puntos de Molmo2 a la traza GT (pointing_2d_canto.py), grupo de 6. """
    (salida / "diag").mkdir(exist_ok=True)
    base.correr([base.PY, "ExploratoryDataAnalysis/pointing_2d_canto.py",
                 "--renders-root", args.renders_root, "--objects-root", args.objects_root,
                 "--molmo-json", base.archivo_puntos(EXP), "--out", salida / "diag" / "pointing2d_roll",
                 "--size", base.SIZE, "--grupos", *map(str, VISTAS)],
                salida / "logs" / "pointing2d_roll.log",
                {"CUDA_VISIBLE_DEVICES": str(args.gpus[0])})


def subir(salida: Path) -> None:
    """ Como base.subir (mismo candado), agregando el código del experimento. """
    import fcntl
    with open(REPO / ".git" / "run_plane_v02.lock", "w") as candado:
        fcntl.flock(candado, fcntl.LOCK_EX)
        subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", base.RAMA],
                       cwd=REPO, check=True)
        subprocess.run(["git", "add", str(salida.relative_to(REPO)), *CODIGO], cwd=REPO, check=True)
        if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO).returncode == 0:
            log("nada nuevo que subir")
            return
        mensaje = (f"Add plane_v02 224px results: {NOMBRE} (canonical planar meshes, cameras rolled "
                   f"about the optical axis, n_views=6)\n\n{COAUTOR}")
        subprocess.run(["git", "commit", "-m", mensaje], cwd=REPO, check=True)
        subprocess.run(["git", "push", "origin", base.RAMA], cwd=REPO, check=True)
        log(f"{NOMBRE}: SUBIDO A GITHUB")


def main() -> None:
    """ Render, inferencia, post-procesamiento, diagnóstico 2D y subida. """
    p = argparse.ArgumentParser(description="plane_v02 224px con cámaras giradas, n_views=6.")
    p.add_argument("etapa", choices=["chequeo", "todo"])
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    p.add_argument("--renders-root", type=Path, default=Path("/data/pespinoz/renders_roll"))
    p.add_argument("--objects-root", type=Path, default=Path("/data/pespinoz/objects"),
                   help="mallas y etiquetas SIN rotar")
    p.add_argument("--renders-base", type=Path, default=Path("/data/pespinoz/renders"),
                   help="raíz con la evaluación de la línea base sin giro")
    p.add_argument("--jobs", type=int, default=12, help="procesos de S_dat")
    p.add_argument("--sin-git", action="store_true")
    args = p.parse_args()

    base.EXPERIMENTOS[NOMBRE] = EXP   # solo en memoria; los archivos base no cambian
    base.VISTAS = VISTAS              # revisar() y posprocesar() leen esta constante del módulo
    if base.molmo_to_ndc(0, 0)[0] != 1.0:
        sys.exit("[error] pipeline_common/camera.py no tiene la corrección NDC")
    if not base.REFERENCIA["plane_sym"][1].exists():
        sys.exit(f"[error] falta {base.REFERENCIA['plane_sym'][1]}")

    renders = renders_listos(args) if args.renders_root.exists() else 0
    puntos = base.revisar(args.renders_root, NOMBRE) if renders == base.TOTAL_OBJETOS else 0
    log(f"renders con giro {renders}/{base.TOTAL_OBJETOS}, inferencia completa {puntos}/"
        f"{base.TOTAL_OBJETOS} -> {base.id_final(EXP)}")
    if args.etapa == "chequeo":
        return

    salida = base.carpeta_salida(NOMBRE)     # resultados_finales/plane_v02_224_roll
    logs = salida / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    try:
        if renders < base.TOTAL_OBJETOS:
            log(f"renderizando en GPU {args.gpus[0]}")
            renderizar(args, logs)
            if renders_listos(args) < base.TOTAL_OBJETOS:
                raise RuntimeError("el render no completó los 850 objetos")
        if base.revisar(args.renders_root, NOMBRE) < base.TOTAL_OBJETOS:
            log(f"inferencia en GPUs {args.gpus}")
            inferir(args, logs)
            if base.revisar(args.renders_root, NOMBRE) < base.TOTAL_OBJETOS:
                raise RuntimeError(f"la inferencia no completó los {base.TOTAL_OBJETOS}")
        tokens = args.renders_root / EXP["tipo"] / f"token_counts_{EXP['exp']}.txt"
        if tokens.exists():
            (logs / f"token_counts_{NOMBRE}.txt").write_bytes(tokens.read_bytes())
        rot.enlazar_base(args)
        base.posprocesar(args, NOMBRE, salida)
        diagnostico_2d(args, salida)
        if not args.sin_git:
            subir(salida)
    except (RuntimeError, ValueError, FileExistsError, FileNotFoundError,
            subprocess.CalledProcessError) as error:
        sys.exit(f"[error] {NOMBRE}: {error}")
    log(f"{NOMBRE}: terminado sin errores")


if __name__ == "__main__":
    main()
