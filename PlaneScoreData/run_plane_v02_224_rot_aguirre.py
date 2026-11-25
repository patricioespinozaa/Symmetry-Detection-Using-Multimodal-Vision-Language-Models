#!/usr/bin/env python3
# plane_v02 (Molmo2, 224px/flat, multiimagen, Flujo A) sobre las mallas planares LIMPIAS con la
# misma transformacion por objeto que Aguirre y Sipiran (WACV 2026) usaron en sus conjuntos de
# robustez, para comparar contra su Tabla 1 en la misma pose (docs/plan_rotacion_aguirre.md).
#
# Pasos (cada uno se salta si ya esta hecho):
#   1. objetos: ExploratoryDataAnalysis/rotaciones_aguirre.py recupera Q por Procrustes desde su
#      malla de 1 % de ruido (det Q = -1 en ~60 % de los objetos) y rotar_mallas.py la aplica a las
#      mallas limpias, invirtiendo las caras cuando hay reflexion;
#   2. render de las 36 vistas de 6/14/26 de las 114 (como rot_so3; asi 14 y 26 no necesitan
#      volver a renderizar), sin giro de camara;
#   3. inferencia de los grupos de --vistas (6 por defecto) con el runner _subset, repartida en
#      --gpus; los grupos ya inferidos se saltan, asi que --vistas 6 14 26 solo agrega 14 y 26;
#   4. sin malla: S_dat con un plano, evaluate.py con metricas de referencia, comparacion,
#      auditoria y pareado contra plane_v02 canonico (run_plane_v02_224.posprocesar);
#   5. con malla: map_to_3d.py + estimate_symmetry.py (svd y ransac_svd, un rayo por punto) y
#      evaluate.py, pareado contra la corrida canonica con malla (run_plane_malla.py, id
#      plane_v02_s224_flat_ndcfix_mesh). Con malla el techo con puntos perfectos no cae al
#      transformar (oraculo: ~0.93-0.98 en ambas poses), a diferencia de S_dat;
#   6. diagnostico de pointing 2D y subida a GitHub.
#
#   python3 PlaneScoreData/run_plane_v02_224_rot_aguirre.py chequeo
#   python3 PlaneScoreData/run_plane_v02_224_rot_aguirre.py todo --gpus 0 1 \
#       --aguirre ~/data/curated_plane_sym_obj_rot_robustness.zip
import argparse
import csv
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "PlaneScoreData"))
import run_plane_v02_224 as base  # noqa: E402
import run_plane_v02_224_rot_so3 as rot  # noqa: E402
import run_axis_malla as malla  # noqa: E402
from compare_plane_scores import comparar_pareado, errores_por_objeto  # noqa: E402
from run_plane_resolucion import correr  # noqa: E402

NOMBRE = "rot_aguirre"
VISTAS = (6,)
VISTAS_RENDER = (6, 14, 26)
VISTAS_RENDERIZADAS = 36          # union de los indices de 6, 14 y 26 sobre 114
REPO_VIEWS = 114
SUBDIR = "curated_plane_sym_obj"
EXP = dict(tipo="plane_sym", luz="flat", modelo="molmo", exp=f"{base.PROMPT}_rot_aguirre",
           prompt=base.PROMPT, modo="auto", flujo="a")
ID_MALLA = f"{EXP['exp']}_s{base.SIZE}_{EXP['luz']}_ndcfix_mesh"
ID_MALLA_BASE = f"{base.PROMPT}_s{base.SIZE}_{EXP['luz']}_ndcfix_mesh"
CSV_MALLA_BASE = (REPO / "resultados_finales" / "plane_malla_224" / "experiments_25_11_2026"
                  / "plane_sym_final_comparison.csv")
CODIGO = ["ExploratoryDataAnalysis/rotaciones_aguirre.py",
          "ExploratoryDataAnalysis/rotar_mallas.py",
          "ImagesGenerator/export_fibonacci_views_subset.py",
          "ImagesGenerator/data_render_subset.py",
          "PlaneScoreData/run_plane_v02_224_rot_aguirre.py"]
COAUTOR = "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"


def log(msg: str) -> None:
    """ Mensaje con hora. """
    print(f"[{dt.datetime.now():%d/%m %H:%M}] {msg}", flush=True)


def objetos_listos(args: argparse.Namespace) -> int:
    """ Cuenta las mallas transformadas y validadas; detiene si hay mallas de otro experimento.

    Args:
        * args: argumentos del script

    Returns:
        * int: objetos con .obj y .txt transformados (0 si la carpeta no existe)

    Raises:
        * SystemExit: si la carpeta existe pero no es de este experimento o tiene fallas

    """
    carpeta = args.objects_root / SUBDIR
    if not carpeta.is_dir():
        return 0
    rotaciones = carpeta / "rotaciones.csv"
    validacion = args.objects_root / "validacion_rotacion.csv"
    if not (rotaciones.exists() and validacion.exists()):
        sys.exit(f"[error] {carpeta} existe pero sin rotaciones.csv/validacion_rotacion.csv")
    with open(rotaciones, newline="", encoding="utf-8") as f:
        modos = {r["modo"] for r in csv.DictReader(f)}
    if modos != {"aguirre"}:
        sys.exit(f"[error] {rotaciones} no es de rotaciones_aguirre.py (modos {modos})")
    with open(validacion, newline="", encoding="utf-8") as f:
        fallas = [r["object_id"] for r in csv.DictReader(f) if r["ok"] != "1"]
    if fallas:
        sys.exit(f"[error] {len(fallas)} objetos fallan en {validacion}, por ejemplo {fallas[:3]}")
    return min(len(list(carpeta.glob("*.obj"))), len(list(carpeta.glob("*.txt"))))


def preparar_objetos(args: argparse.Namespace, salida: Path) -> None:
    """ Recupera las Q (si faltan), transforma las mallas limpias y guarda la evidencia.

    Args:
        * args: argumentos del script
        * salida: carpeta de resultados

    Raises:
        * RuntimeError: si falta --aguirre o algun paso falla

    """
    logs = salida / "logs"
    if not (args.rot_dir / "rotaciones.csv").exists():
        if args.aguirre is None:
            raise RuntimeError("faltan las transformaciones: pasa --aguirre <zip o carpeta>")
        correr([base.PY, "ExploratoryDataAnalysis/rotaciones_aguirre.py",
                "--obj-dir", args.objects_base / SUBDIR, "--aguirre", args.aguirre,
                "--out-dir", args.rot_dir], logs / "rotaciones_aguirre.log")
    with open(args.rot_dir / "rotaciones.csv", newline="", encoding="utf-8") as f:
        n_rot = sum(1 for _ in csv.DictReader(f))
    if n_rot != base.TOTAL_OBJETOS:
        raise RuntimeError(f"{args.rot_dir / 'rotaciones.csv'} tiene {n_rot} objetos")
    correr([base.PY, "ExploratoryDataAnalysis/rotar_mallas.py",
            "--obj-dir", args.objects_base / SUBDIR, "--rot-dir", args.rot_dir,
            "--out-root", args.objects_root], logs / "rotar_mallas.log")
    guardar_validacion(args, salida)


def guardar_validacion(args: argparse.Namespace, salida: Path) -> None:
    """ Copia los CSV de validacion de la transformacion a la carpeta de resultados. """
    destino = salida / "validacion"
    destino.mkdir(parents=True, exist_ok=True)
    for origen in (args.rot_dir / "validacion_rotacion_aguirre.csv",
                   args.objects_root / "validacion_rotacion.csv"):
        if origen.exists():
            shutil.copy2(origen, destino / origen.name)


def renders_listos(args: argparse.Namespace) -> int:
    """ Cuenta los objetos con las 36 vistas renderizadas desde las mallas transformadas.

    Args:
        * args: argumentos del script

    Returns:
        * int: objetos con render completo

    Raises:
        * SystemExit: si algun render es de otra malla, tiene giro o un numero de vistas distinto

    """
    if not args.renders_root.exists():
        return 0
    mallas = str((args.objects_root / SUBDIR).resolve())
    listos, problemas = 0, []
    for carpeta in sorted((args.renders_root / EXP["tipo"]).glob(f"*/{base.SIZE}/{EXP['luz']}")):
        manifest, metadata = carpeta / "manifest.json", carpeta / "metadata_all.json"
        if not (manifest.exists() and metadata.exists()):
            continue
        datos = json.loads(manifest.read_text(encoding="utf-8"))
        vistas = json.loads(metadata.read_text(encoding="utf-8"))
        if not str(Path(datos.get("mesh", "")).resolve()).startswith(mallas):
            problemas.append(f"{carpeta}: malla {datos.get('mesh')}")
        elif datos.get("roll_seed") is not None or any("roll_deg" in v for v in vistas):
            problemas.append(f"{carpeta}: render con giro de camara")
        elif len(vistas) != VISTAS_RENDERIZADAS:
            problemas.append(f"{carpeta}: {len(vistas)} vistas")
        else:
            listos += 1
    if problemas:
        sys.exit(f"[error] {len(problemas)} renders no corresponden, por ejemplo {problemas[:3]}")
    return listos


def renderizar(args: argparse.Namespace, logs: Path) -> None:
    """ Renderiza las 36 vistas de los objetos que falten (salta los ya hechos). """
    correr([base.PY, "ImagesGenerator/data_render_subset.py",
            "--input-folder", args.objects_root / SUBDIR, "--output-folder", args.renders_root,
            "--symmetry-type", EXP["tipo"], "--repo-views", REPO_VIEWS, "--sizes", base.SIZE,
            "--lightings", EXP["luz"], "--subset-n-views", *VISTAS_RENDER,
            "--device", "cuda:0", "--skip-existing", "--yes"],
           logs / "render.log", {"CUDA_VISIBLE_DEVICES": str(args.gpus[0])})


def inferir(args: argparse.Namespace, logs: Path) -> None:
    """ Inferencia de los grupos de VISTAS con el runner _subset (salta los ya hechos). """
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
        raise RuntimeError(f"la inferencia termino con codigos {codigos}")


def diagnostico_2d(args: argparse.Namespace, salida: Path) -> None:
    """ Distancia 2D de los puntos de Molmo2 a la traza GT transformada (pointing_2d_canto.py). """
    (salida / "diag").mkdir(exist_ok=True)
    correr([base.PY, "ExploratoryDataAnalysis/pointing_2d_canto.py",
            "--renders-root", args.renders_root, "--objects-root", args.objects_root,
            "--molmo-json", base.archivo_puntos(EXP),
            "--out", salida / "diag" / f"pointing2d_{NOMBRE}",
            "--size", base.SIZE, "--grupos", *VISTAS],
           salida / "logs" / f"pointing2d_{NOMBRE}.log",
           {"CUDA_VISIBLE_DEVICES": str(args.gpus[0])})


def configurar_malla() -> None:
    """ Ajusta las constantes de run_axis_malla.py a este experimento (solo en memoria). """
    malla.TIPO = EXP["tipo"]
    malla.PROMPT = EXP["exp"]            # crear_alias enlaza a molmo_multiview_<PROMPT>.json
    malla.ALIAS = ID_MALLA
    malla.VISTAS = VISTAS
    malla.VARIANTES = ["base"]           # un rayo por punto, sin agrupamiento
    malla.PARCHES = (1,)


def con_malla(args: argparse.Namespace, salida: Path) -> list[dict]:
    """ Retroproyeccion, ajuste, evaluacion y pareado contra la corrida canonica con malla.

    Args:
        * args: argumentos del script
        * salida: carpeta de resultados del experimento

    Returns:
        * list[dict]: una fila por metodo con AUC canonica, AUC transformada y el pareado

    Raises:
        * RuntimeError: si falta la evaluacion canonica con malla o algun paso falla

    """
    args_malla = argparse.Namespace(**vars(args), salida=salida / "con_malla",
                                    sin_referencia=False)
    logs = args_malla.salida / "logs"
    print(f"con malla: {malla.crear_alias(args.renders_root)} enlaces -> "
          f"molmo_multiview_{ID_MALLA}.json", flush=True)
    for etapa in (malla.retroproyectar, malla.ajustar, malla.evaluar):
        etapa(args_malla, logs)
    filas = malla.leer_filas(malla.comparar(args_malla, logs))
    canonica = malla.leer_filas(CSV_MALLA_BASE)   # (experimento, metodo, vistas) -> fila
    salida_pareado = []
    for metodo in malla.METODOS:
        ruta_base = malla.ruta_eval(args.renders_base, ID_MALLA_BASE, metodo)
        if not ruta_base.exists():
            raise RuntimeError(f"falta la evaluacion canonica con malla {ruta_base}")
        for vistas in map(str, VISTAS):
            nueva = filas[(ID_MALLA, metodo, vistas)]
            ref = canonica.get((ID_MALLA_BASE, metodo, vistas), {})
            salida_pareado.append({
                "metodo": metodo, "n_views": vistas,
                "auc_canonica": ref.get("auc_angular"), "auc_aguirre": nueva["auc_angular"],
                "f1_ref_canonica": ref.get("f1_ref"), "f1_ref_aguirre": nueva.get("f1_ref"),
                "n_objects_canonica": ref.get("n_objects"), "n_objects_aguirre": nueva["n_objects"],
                **comparar_pareado(errores_por_objeto(ruta_base, vistas),
                                   errores_por_objeto(malla.ruta_eval(args.renders_root, ID_MALLA,
                                                                      metodo), vistas))})
    ruta = args_malla.salida / "pareado" / f"{ID_MALLA}_vs_{ID_MALLA_BASE}.csv"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(salida_pareado[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(salida_pareado)
    print(f"con malla, pareado (error transformado - canonico; positivo = empeora): {ruta}")
    return salida_pareado


def chequear_malla_base(args: argparse.Namespace) -> None:
    """ Verifica que exista la corrida canonica con malla contra la que se parea. """
    faltan = [m for m in malla.METODOS
              if not malla.ruta_eval(args.renders_base, ID_MALLA_BASE, m).exists()]
    if faltan or not CSV_MALLA_BASE.exists():
        sys.exit(f"[error] falta la corrida canonica con malla ({ID_MALLA_BASE}, {faltan}, "
                 f"{CSV_MALLA_BASE}); corre antes PlaneScoreData/run_plane_malla.py todo")


def subir(salida: Path) -> None:
    """ Como base.subir (mismo candado), agregando el codigo del experimento. """
    import fcntl
    with open(REPO / ".git" / "run_plane_v02.lock", "w") as candado:
        fcntl.flock(candado, fcntl.LOCK_EX)
        subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", base.RAMA],
                       cwd=REPO, check=True)
        subprocess.run(["git", "add", str(salida.relative_to(REPO)), *CODIGO], cwd=REPO, check=True)
        if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO).returncode == 0:
            log("nada nuevo que subir")
            return
        mensaje = (f"Add plane_v02 224px results: {NOMBRE} (clean planar meshes with the "
                   f"per-object transforms of Aguirre and Sipiran, n_views="
                   f"{'/'.join(map(str, VISTAS))}, with and without mesh)\n\n{COAUTOR}")
        subprocess.run(["git", "commit", "-m", mensaje], cwd=REPO, check=True)
        subprocess.run(["git", "push", "origin", base.RAMA], cwd=REPO, check=True)
        log(f"{NOMBRE}: SUBIDO A GITHUB")


def main() -> None:
    """ Objetos, render, inferencia, post-procesamiento con y sin malla, diagnostico y subida. """
    global VISTAS   # lo leen inferir, diagnostico_2d y con_malla
    p = argparse.ArgumentParser(description="plane_v02 224px con las transformaciones de Aguirre "
                                            "y Sipiran.")
    p.add_argument("etapa", choices=["chequeo", "todo"])
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    p.add_argument("--aguirre", type=Path, default=None,
                   help="zip o carpeta de sus conjuntos de robustez (solo si faltan las Q)")
    p.add_argument("--objects-base", type=Path, default=Path("/data/pespinoz/objects"),
                   help="mallas y etiquetas canonicas limpias")
    p.add_argument("--rot-dir", type=Path,
                   default=Path("/data/pespinoz/objects_rot_aguirre_etiquetas"),
                   help="salida de rotaciones_aguirre.py (.txt transformados + rotaciones.csv)")
    p.add_argument("--objects-root", type=Path, default=Path("/data/pespinoz/objects_rot_aguirre"),
                   help="mallas y etiquetas transformadas (salida de rotar_mallas.py)")
    p.add_argument("--renders-root", type=Path, default=Path("/data/pespinoz/renders_rot_aguirre"))
    p.add_argument("--renders-base", type=Path, default=Path("/data/pespinoz/renders"),
                   help="raiz con las evaluaciones canonicas (sin malla y con malla)")
    p.add_argument("--jobs", type=int, default=12, help="procesos de CPU (S_dat y malla)")
    p.add_argument("--vistas", type=int, nargs="+", default=list(VISTAS), choices=VISTAS_RENDER,
                   help="grupos de vistas; los ya inferidos se saltan (p. ej. --vistas 6 14 26)")
    p.add_argument("--sin-git", action="store_true")
    args = p.parse_args()

    VISTAS = tuple(sorted(set(args.vistas)))
    base.EXPERIMENTOS[NOMBRE] = EXP   # solo en memoria; los archivos base no cambian
    base.VISTAS = VISTAS              # revisar() y posprocesar() leen esta constante del modulo
    if base.molmo_to_ndc(0, 0)[0] != 1.0:
        sys.exit("[error] pipeline_common/camera.py no tiene la correccion NDC")
    if not base.REFERENCIA["plane_sym"][1].exists():
        sys.exit(f"[error] falta {base.REFERENCIA['plane_sym'][1]}")
    configurar_malla()
    chequear_malla_base(args)

    objetos = objetos_listos(args)
    renders = renders_listos(args) if objetos == base.TOTAL_OBJETOS else 0
    puntos = base.revisar(args.renders_root, NOMBRE) if renders == base.TOTAL_OBJETOS else 0
    log(f"objetos transformados {objetos}/{base.TOTAL_OBJETOS}, renders {renders}/"
        f"{base.TOTAL_OBJETOS}, inferencia {puntos}/{base.TOTAL_OBJETOS} -> {base.id_final(EXP)} "
        f"y {ID_MALLA}")
    if args.etapa == "chequeo":
        if objetos == 0 and args.aguirre is None and not (args.rot_dir / "rotaciones.csv").exists():
            log("para 'todo' hace falta --aguirre (no hay transformaciones en --rot-dir)")
        return

    salida = base.carpeta_salida(NOMBRE)     # resultados_finales/plane_v02_224_rot_aguirre
    logs = salida / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    try:
        if objetos < base.TOTAL_OBJETOS:
            log("recuperando y aplicando las transformaciones")
            preparar_objetos(args, salida)
            if objetos_listos(args) < base.TOTAL_OBJETOS:
                raise RuntimeError("no quedaron los 850 objetos transformados")
        guardar_validacion(args, salida)
        if renders_listos(args) < base.TOTAL_OBJETOS:
            log(f"renderizando en GPU {args.gpus[0]}")
            renderizar(args, logs)
            if renders_listos(args) < base.TOTAL_OBJETOS:
                raise RuntimeError("el render no completo los 850 objetos")
        if base.revisar(args.renders_root, NOMBRE) < base.TOTAL_OBJETOS:
            log(f"inferencia en GPUs {args.gpus}")
            inferir(args, logs)
            if base.revisar(args.renders_root, NOMBRE) < base.TOTAL_OBJETOS:
                raise RuntimeError(f"la inferencia no completo los {base.TOTAL_OBJETOS}")
        tokens = args.renders_root / EXP["tipo"] / f"token_counts_{EXP['exp']}.txt"
        if tokens.exists():
            shutil.copy2(tokens, logs / f"token_counts_{NOMBRE}.txt")
        rot.enlazar_base(args)
        base.posprocesar(args, NOMBRE, salida)
        filas_malla = con_malla(args, salida)
        diagnostico_2d(args, salida)
        if not args.sin_git:
            subir(salida)
    except (RuntimeError, ValueError, FileExistsError, FileNotFoundError,
            subprocess.CalledProcessError) as error:
        sys.exit(f"[error] {NOMBRE}: {error}")
    print(f"\n{'con malla':10} {'metodo':11} {'vistas':>6} {'AUC canonica':>12} "
          f"{'AUC Aguirre':>12} {'dif media':>10}")
    for f in filas_malla:
        print(f"{'':10} {f['metodo']:11} {f['n_views']:>6} "
              f"{float(f['auc_canonica'] or 'nan'):12.3f} "
              f"{float(f['auc_aguirre']):12.3f} {f['dif_media']:10.2f}")
    log(f"{NOMBRE}: terminado sin errores")


if __name__ == "__main__":
    main()
