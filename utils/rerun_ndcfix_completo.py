#!/usr/bin/env python3
"""
Re-corre TODO el post-procesamiento del pipeline sin malla con la conversión
NDC corregida (pipeline_common/camera.py, 2026-10-05), sin tocar los
resultados anteriores ni volver a correr Molmo2.

Qué se re-corre (lo que depende de los rayos de cámara):
  1. Línea base: Mapping/estimate_symmetry_no_mesh.py
       axis_sym  -> triangulación del eje
       plane_sym -> --max-planes 3 (multiplano, como en la tesis)
  2. Ablaciones: Pipeline_Experiments/estimate_symmetry_variants.py
       axis_sym  -> expA, expB, expC, expD, expE           (como full_sweep.yaml)
       plane_sym -> expA/expC/expD/expF con max_planes=3    (como plane_maxplanes3_rerun.yaml)
                    + expC con max_planes=1 (tabla del modo un-solo-plano)
  3. Evaluación: Mapping/evaluate.py, con --with-reference-metrics donde la
     tesis lo usa (línea base de eje y todas las corridas multiplano de plano).
  4. Comparación: Mapping/compare_results_no_mesh.py, filtrada a las corridas
     nuevas, en <results-dir>/experiments_<fecha>/.

Qué NO se re-corre (no depende de la conversión NDC): el diagnóstico de
centrado (usa las coordenadas crudas de Molmo2) y los predictores triviales.

Nombres: para cada prompt <X> se crea el enlace
    molmo_multiview_<X>_ndcfix_nomesh.json -> molmo_multiview_<X>.json
en cada carpeta de render. Las salidas quedan como
    <X>_ndcfix_nomesh                 (línea base)
    <X>_ndcfix_expA_nomesh, ...       (ablaciones; expA_mp3, expF_mp3, ... en plano)
Todas terminan en "_nomesh" (requisito de compare_results_no_mesh.py) y
contienen "_ndcfix_", así que no chocan con ningún resultado anterior.

Uso (en el servidor, desde la raíz del repo):
  python3 utils/rerun_ndcfix_completo.py --data ~/data --jobs 8
  # solo algunos prompts (p. ej. los mejores, para tener resultados antes):
  python3 utils/rerun_ndcfix_completo.py --data ~/data --prompts axis_v08 plane_v04_1
  # ver qué haría sin ejecutar nada:
  python3 utils/rerun_ndcfix_completo.py --data ~/data --dry-run
"""
import argparse
import datetime as dt
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "Pipeline_Experiments"))
from config import VariantConfig, build_output_experiment_id, discover_experiment_ids  # noqa: E402
from pipeline_common.camera import molmo_to_ndc  # noqa: E402

TAG = "_ndcfix"
NOMESH = "_nomesh"
PY = sys.executable

VARIANTES = {
    "axis_sym": [VariantConfig("expA"), VariantConfig("expB"), VariantConfig("expC"),
                 VariantConfig("expD"), VariantConfig("expE")],
    "plane_sym": [VariantConfig("expA", max_planes=3), VariantConfig("expC", max_planes=3),
                  VariantConfig("expD", max_planes=3),
                  VariantConfig("expF", max_planes=3, sde_gate=0.02),
                  VariantConfig("expC", max_planes=1)],
}


def id_nuevo(src):
    return src.removesuffix(NOMESH) + TAG + NOMESH


def crear_enlaces(renders, tipo, src, nuevo, size, light, dry):
    n = 0
    for d in sorted((renders / tipo).glob(f"*/{size}/{light}")):
        origen = d / f"molmo_multiview_{src}.json"
        if not origen.exists():
            continue
        destino = d / f"molmo_multiview_{nuevo}.json"
        if not dry:
            if destino.is_symlink() or destino.exists():
                destino.unlink()
            try:
                destino.symlink_to(origen.name)
            except OSError:                      # Windows sin permiso de symlink
                shutil.copy2(origen, destino)
        n += 1
    return n


def correr(cmd, log, dry):
    linea = " ".join(str(c) for c in cmd)
    if dry:
        return f"[dry] {linea}", 0.0
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        f.write(f"$ {linea}\n\n")
        f.flush()
        r = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, cwd=REPO)
    if r.returncode != 0:
        raise RuntimeError(f"falló (código {r.returncode}), ver {log}")
    return linea, time.time() - t0


def en_paralelo(tareas, jobs, dry, titulo):
    """tareas: lista de (nombre, cmd, log)."""
    print(f"\n=== {titulo}: {len(tareas)} tareas, {jobs} en paralelo ===", flush=True)
    errores = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        fut = {ex.submit(correr, cmd, log, dry): nombre for nombre, cmd, log in tareas}
        for i, f in enumerate(as_completed(fut), 1):
            nombre = fut[f]
            try:
                linea, seg = f.result()
                print(f"  [{i}/{len(tareas)}] ok {nombre}" + ("" if dry else f" ({seg / 60:.1f} min)")
                      + (f"\n      {linea}" if dry else ""), flush=True)
            except Exception as e:  # noqa: BLE001
                errores.append((nombre, str(e)))
                print(f"  [{i}/{len(tareas)}] ERROR {nombre}: {e}", flush=True)
    return errores


def main():
    # Al cortar la salida con "| head", terminar en silencio en vez de lanzar
    # BrokenPipeError (solo POSIX; en Windows no existe SIGPIPE).
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="carpeta con renders/ y objects/ (p. ej. ~/data)")
    ap.add_argument("--tipos", nargs="+", default=["axis_sym", "plane_sym"],
                    choices=["axis_sym", "plane_sym"])
    ap.add_argument("--prompts", nargs="+", default=None,
                    help="solo estos prompts (ids de molmo_multiview_<ID>.json); por defecto, todos")
    ap.add_argument("--size", default="224")
    ap.add_argument("--lighting", default="flat")
    ap.add_argument("--jobs", type=int, default=4, help="procesos en paralelo (CPU)")
    ap.add_argument("--results-dir", default=None,
                    help="dónde guardar CSV y logs (por defecto <repo>/../results_ndcfix, "
                         "junto a la carpeta results/ que usan los configs)")
    ap.add_argument("--sin-variantes", action="store_true", help="solo línea base")
    ap.add_argument("--max-objects", type=int, default=None, help="prueba rápida")
    ap.add_argument("--dry-run", action="store_true", help="muestra qué haría, sin ejecutar")
    args = ap.parse_args()

    if molmo_to_ndc(0, 0)[0] != 1.0:
        sys.exit("[error] pipeline_common/camera.py NO tiene la corrección NDC; actualiza el código")
    print("[ok] camera.py con la convención corregida")

    data = Path(args.data).expanduser().resolve()
    renders, objects = data / "renders", data / "objects"
    results = Path(args.results_dir).expanduser() if args.results_dir else REPO.parent / "results_ndcfix"
    logs = results / "logs" / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.dry_run:
        logs.mkdir(parents=True, exist_ok=True)
    comunes = ["--sizes", args.size, "--lightings", args.lighting]
    lim = ["--max-objects", str(args.max_objects)] if args.max_objects else []

    est, ev, por_tipo = [], [], {t: [] for t in args.tipos}
    for tipo in args.tipos:
        fuentes = [s for s in discover_experiment_ids(renders, tipo) if TAG not in s]
        if args.prompts:
            fuentes = [s for s in fuentes if s in args.prompts]
        print(f"\n[{tipo}] {len(fuentes)} prompts: {', '.join(fuentes)}")
        for src in fuentes:
            nuevo = id_nuevo(src)
            n = crear_enlaces(renders, tipo, src, nuevo, args.size, args.lighting, args.dry_run)
            print(f"  {src} -> {nuevo}  ({n} objetos)")
            if n == 0:
                continue
            # línea base
            cmd = [PY, "Mapping/estimate_symmetry_no_mesh.py", "--renders-root", renders,
                   "--symmetry-type", tipo, *comunes, "--experiment-id", nuevo, "--overwrite", *lim]
            if tipo == "plane_sym":
                cmd += ["--max-planes", "3"]
            est.append((f"base {nuevo}", cmd, logs / f"est_{nuevo}.log"))
            metodo = "triangulation_multiplane" if tipo == "plane_sym" else "triangulation"
            ev.append((nuevo, tipo, metodo, True))
            por_tipo[tipo].append(nuevo)
            # ablaciones
            if args.sin_variantes:
                continue
            for v in VARIANTES[tipo]:
                out = build_output_experiment_id(nuevo, v)
                cmd = [PY, "Pipeline_Experiments/estimate_symmetry_variants.py",
                       "--renders-root", renders, "--objects-root", objects,
                       "--symmetry-type", tipo, *comunes, "--experiment-id", nuevo,
                       "--variant", v.variant, "--max-planes", str(v.max_planes),
                       "--sde-gate", str(v.sde_gate), "--overwrite", *lim]
                est.append((f"{v.output_suffix()} {nuevo}", cmd, logs / f"est_{out}.log"))
                multi = tipo == "plane_sym" and v.max_planes > 1
                ev.append((out, tipo, "triangulation_multiplane" if multi else "triangulation", multi))
                por_tipo[tipo].append(out)

    errores = en_paralelo(est, args.jobs, args.dry_run, "1/3 Estimación (línea base + ablaciones)")

    tareas_ev = []
    for exp, tipo, metodo, ref in ev:
        cmd = [PY, "Mapping/evaluate.py", "--renders-root", renders, "--objects-root", objects,
               "--symmetry-type", tipo, *comunes, "--experiment-id", exp, "--method", metodo, *lim]
        if ref:
            cmd.append("--with-reference-metrics")
        tareas_ev.append((f"eval {exp}", cmd, logs / f"eval_{exp}.log"))
    errores += en_paralelo(tareas_ev, args.jobs, args.dry_run, "2/3 Evaluación")

    tareas_cmp = []
    for tipo, ids in por_tipo.items():
        if not ids:
            continue
        cmd = [PY, "Mapping/compare_results_no_mesh.py", "--renders-root", renders,
               "--symmetry-type", tipo, *comunes, "--experiment-id", *ids,
               "--csv-dir", results, "--no-plots", "--total-objects", "850"]
        tareas_cmp.append((f"compare {tipo}", cmd, logs / f"compare_{tipo}.log"))
    errores += en_paralelo(tareas_cmp, 1, args.dry_run, "3/3 Comparación")

    print("\n" + "=" * 70)
    if errores:
        print(f"Terminado con {len(errores)} errores:")
        for n, e in errores:
            print(f"  - {n}: {e}")
    else:
        print("Terminado sin errores.")
    if not args.dry_run:
        print(f"CSV de comparación: {results}/experiments_<fecha>/  |  logs: {logs}")


if __name__ == "__main__":
    main()
