#!/usr/bin/env python3
# Línea base de PI1 (iii): puntos elegidos al azar dentro de la silueta del objeto y procesados igual
# que los del VLM. Responde si la elección de puntos de Molmo2 aporta algo más allá de lo que ya dan
# las cámaras, la forma del objeto y el estimador.
#
# La línea base no depende de ninguna salida de Molmo2. Para cada objeto, número de vistas
# n_v en {6, 14, 26} y semilla toma el mismo conjunto de vistas que recibe el VLM (los n_v renders
# espaciados uniformemente en la secuencia de Fibonacci de metadata_all.json, igual que
# get_n_views_entries del runner de Molmo2) y sortea en cada vista PUNTOS_POR_VISTA = 2 píxeles
# distintos dentro de la silueta (píxeles que no son fondo blanco), con obj_id 1 y 2 como el par
# que pide el prompt, en la escala [0, 1000] de Molmo. Así todas las vistas y todos los objetos
# tienen puntos. El resultado se escribe como molmo_multiview_<tipo>_silueta2p_rNN.json y pasa por
# el mismo post-procesamiento que el VLM (run_plane_v02_224.posprocesar): EXP-E en eje o S_dat con
# un plano en plano, evaluate.py con métricas de referencia, auditoría y pareado objeto a objeto
# contra Molmo2 (axis_v04 / plane_v02 a 224px/flat, Flujo A), que solo se usa para comparar.
# No usa GPU ni vuelve a correr ningún VLM.
#
#   python3 PlaneScoreData/run_silueta_aleatoria.py chequeo                 # solo lee y valida
#   python3 PlaneScoreData/run_silueta_aleatoria.py todo --tipo axis_sym     # 10 semillas en eje
#   python3 PlaneScoreData/run_silueta_aleatoria.py todo --tipo plane_sym --semillas 10
#
# Cada semilla deja su CSV en resultados_finales/silueta2p_<tipo>_224/rNN/ y, al final, resumen.csv
# con la media, el desvío y el rango sobre las semillas junto al valor de Molmo2, y el p empírico
# (1 + semillas con AUC >= Molmo2) / (1 + semillas). Las semillas ya completas se saltan.
import argparse
import csv
import datetime as dt
import fcntl
import json
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "PlaneScoreData"))
import run_plane_v02_224 as base  # noqa: E402

SIZE, LUZ = 224, "flat"
VISTAS = (6, 14, 26)
TOTAL_OBJETOS = 850
UMBRAL_FONDO = 250          # un píxel es del objeto si algún canal RGB es menor que este valor
PUNTOS_POR_VISTA = 2        # el par (obj_id 1 y 2) que piden los prompts de eje y plano
CORTO = {"axis_sym": "axis", "plane_sym": "plane"}
PROMPTS = {"axis_sym": "axis_v04", "plane_sym": "plane_v02"}     # Molmo2 con que se compara
METRICAS = ("n_objects", "auc_angular", "angular_error_median", "precision_10deg", "f1_ref")


def nombre_exp(tipo: str, semilla: int) -> str:
    """ Id del experimento de una semilla (también nombre del JSON de puntos). """
    return f"{CORTO[tipo]}_silueta2p_r{semilla:02d}"


def carpeta_resultados(tipo: str) -> Path:
    """ Carpeta de resultados_finales del experimento. """
    return REPO / "resultados_finales" / f"silueta2p_{CORTO[tipo]}_{SIZE}"


def carpetas_render(renders: Path, tipo: str) -> list[Path]:
    """ Carpetas 224/flat del tipo con metadatos de cámara. """
    return [d for d in sorted((renders / tipo).glob(f"*/{SIZE}/{LUZ}"))
            if (d / "metadata_all.json").exists()]


def silueta(ruta: Path) -> np.ndarray:
    """ Coordenadas (fila, columna) de los píxeles del objeto en un render. """
    rgb = np.asarray(Image.open(ruta).convert("RGB"))
    filas, columnas = np.nonzero((rgb < UMBRAL_FONDO).any(axis=2))
    return np.stack([filas, columnas], axis=1)


def entradas_vistas(metadata: list[dict], vistas: int) -> list[dict]:
    """ Las n_v vistas que recibe el VLM: copia de get_n_views_entries de
    MolmoPointing/molmo_multiview_runner.py (que importa torch); chequear() compara ambas. """
    total = len(metadata)
    if vistas >= total:
        return sorted(metadata, key=lambda e: e["index"])
    indices = {int(round(i)) for i in np.linspace(0, total - 1, vistas)}
    return sorted((m for m in metadata if m["index"] in indices), key=lambda e: e["index"])


def imagenes_enviadas(metadata: list[dict], vistas: int) -> list[dict]:
    """ images_sent con el mismo formato que escribe el runner de Molmo2. """
    return [{"img_idx": i, "filename": e["filename"], "index": e["index"], "azimuth": e["azimuth"],
             "elevation": e["elevation"], "eye": e["eye"], "R": e["R"], "T": e["T"]}
            for i, e in enumerate(entradas_vistas(metadata, vistas))]


def puntos_aleatorios(imagenes: list[dict], carpeta: Path, rng: np.random.Generator) -> dict:
    """ Sortea PUNTOS_POR_VISTA píxeles distintos dentro de la silueta de cada vista.

    Args:
        * imagenes: images_sent del grupo
        * carpeta: carpeta de renders del objeto
        * rng: generador de la semilla, el objeto y el número de vistas

    Returns:
        * dict: points_by_image con obj_id 1..PUNTOS_POR_VISTA en cada vista

    Raises:
        * ValueError: si una vista tiene menos píxeles del objeto que puntos

    """
    nuevos = {}
    for imagen in imagenes:
        ruta = carpeta / imagen["filename"]
        pixeles = silueta(ruta)
        if len(pixeles) < PUNTOS_POR_VISTA:
            raise ValueError(f"{ruta}: la silueta tiene {len(pixeles)} píxeles")
        alto, ancho = np.asarray(Image.open(ruta)).shape[:2]
        elegidos = pixeles[rng.choice(len(pixeles), size=PUNTOS_POR_VISTA, replace=False)]
        desplazamiento = rng.random((PUNTOS_POR_VISTA, 2))        # posición dentro del píxel
        nuevos[str(imagen["img_idx"])] = [
            {"obj_id": k + 1, "x": round(float((c + dx) / ancho * 1000), 3),
             "y": round(float((f + dy) / alto * 1000), 3)}
            for k, ((f, c), (dy, dx)) in enumerate(zip(elegidos, desplazamiento))]
    return nuevos


def generar(args: argparse.Namespace, tipo: str, semilla: int) -> int:
    """ Escribe molmo_multiview_<exp>.json en cada objeto; devuelve cuántos escribió. """
    exp = nombre_exp(tipo, semilla)
    escritos = 0
    for carpeta in carpetas_render(args.renders_root, tipo):
        destino = carpeta / f"molmo_multiview_{exp}.json"
        if destino.exists() and not args.overwrite:
            continue
        metadata = json.loads((carpeta / "metadata_all.json").read_text(encoding="utf-8"))
        objeto = carpeta.parents[1].name
        salida = {}
        for vistas in VISTAS:
            imagenes = imagenes_enviadas(metadata, vistas)
            rng = np.random.default_rng([semilla, zlib.crc32(objeto.encode()), vistas])
            por_imagen = puntos_aleatorios(imagenes, carpeta, rng)
            salida[str(vistas)] = {
                "experiment_id": exp, "origen": "silueta_aleatoria", "semilla": semilla,
                "puntos_por_vista": PUNTOS_POR_VISTA, "prompt_used": None, "raw_output": None,
                "points_by_image": por_imagen, "images_sent": imagenes,
                "n_points": sum(len(p) for p in por_imagen.values()),
                "n_images_with_points": len(por_imagen)}
        tmp = destino.with_suffix(".tmp")
        tmp.write_text(json.dumps(salida), encoding="utf-8")
        tmp.replace(destino)
        escritos += 1
    return escritos


def chequear(args: argparse.Namespace, tipos: list[str]) -> None:
    """ Valida renders y referencia antes de escribir nada. Donde existe la corrida de Molmo2,
    confirma además que entradas_vistas elige las mismas vistas que su runner (solo se compara,
    los puntos no la usan). """
    if base.molmo_to_ndc(0, 0)[0] != 1.0:
        sys.exit("[error] pipeline_common/camera.py no tiene la corrección NDC")
    for tipo in tipos:
        errores, comparadas = [], 0
        carpetas = carpetas_render(args.renders_root, tipo)
        if len(carpetas) != TOTAL_OBJETOS:
            errores.append(f"{len(carpetas)} renders a {SIZE}px/{LUZ}, se esperaban {TOTAL_OBJETOS}")
        for carpeta in carpetas:
            metadata = json.loads((carpeta / "metadata_all.json").read_text(encoding="utf-8"))
            vlm = carpeta / f"molmo_multiview_{PROMPTS[tipo]}.json"
            grupos = json.loads(vlm.read_text(encoding="utf-8")) if vlm.exists() else {}
            for vistas in VISTAS:
                imagenes = imagenes_enviadas(metadata, vistas)
                if len(imagenes) != vistas:
                    errores.append(f"{carpeta}: {len(imagenes)} vistas en vez de {vistas}")
                faltan = [i["filename"] for i in imagenes if not (carpeta / i["filename"]).exists()]
                if faltan:
                    errores.append(f"{carpeta}/{vistas}: faltan renders {faltan[:2]}")
                g = grupos.get(str(vistas))
                if g is not None:
                    comparadas += 1
                    if [i["filename"] for i in g["images_sent"]] != [i["filename"] for i in imagenes]:
                        errores.append(f"{carpeta}/{vistas}: vistas distintas de las de {PROMPTS[tipo]}")
        if not base.REFERENCIA[tipo][1].exists():
            errores.append(f"falta {base.REFERENCIA[tipo][1]}")
        if errores:
            sys.exit(f"[error] {tipo}: {len(errores)} problemas, por ejemplo {errores[:3]}")
        print(f"{tipo}: {len(carpetas)} objetos, {PUNTOS_POR_VISTA} puntos por vista, vistas iguales a "
              f"{PROMPTS[tipo]} en {comparadas} grupos; semillas {nombre_exp(tipo, args.semilla_inicial)}"
              f" ... {nombre_exp(tipo, args.semilla_inicial + args.semillas - 1)}")
    print("[ok] Chequeo sin problemas")


def semilla_lista(salida: Path, tipo: str, exp: str) -> bool:
    """ True si la semilla ya tiene su CSV final con las 3 cantidades de vistas. """
    final = base.id_final(base.EXPERIMENTOS[exp])
    for ruta in salida.glob(f"experiments_*/{tipo}_final_comparison.csv"):
        with open(ruta, newline="", encoding="utf-8") as f:
            vistas = {int(r["n_views"]) for r in csv.DictReader(f) if r["experiment"] == final}
        if set(VISTAS) <= vistas:
            return True
    return False


def leer_fila(ruta: Path, exp_id: str, vistas: int) -> dict:
    """ Fila de un CSV final por id y número de vistas. """
    with open(ruta, newline="", encoding="utf-8") as f:
        return next(r for r in csv.DictReader(f)
                    if r["experiment"] == exp_id and int(r["n_views"]) == vistas)


def resumir(tipo: str, semillas: list[int]) -> Path:
    """ Media, desvío y rango sobre las semillas, junto al valor de Molmo2 y el p empírico. """
    id_vlm, csv_vlm = base.REFERENCIA[tipo]
    raiz = carpeta_resultados(tipo)
    filas = []
    for vistas in VISTAS:
        vlm = leer_fila(csv_vlm, id_vlm, vistas)
        por_semilla = []
        for s in semillas:
            e = base.EXPERIMENTOS[nombre_exp(tipo, s)]
            ruta = next((raiz / f"r{s:02d}").glob(f"experiments_*/{tipo}_final_comparison.csv"))
            por_semilla.append(leer_fila(ruta, base.id_final(e), vistas))
        for metrica in METRICAS:
            if not vlm.get(metrica):
                continue
            valores = np.array([float(f[metrica]) for f in por_semilla])
            fila = {"symmetry_type": tipo, "n_views": vistas, "metric": metrica,
                    "n_semillas": len(valores), "molmo2": float(vlm[metrica]),
                    "silueta_media": round(float(valores.mean()), 6),
                    "silueta_std": round(float(valores.std(ddof=1)) if len(valores) > 1 else 0.0, 6),
                    "silueta_min": float(valores.min()), "silueta_max": float(valores.max())}
            if metrica == "auc_angular":
                fila["p_empirico"] = round((1 + int((valores >= fila["molmo2"]).sum()))
                                           / (1 + len(valores)), 6)
            filas.append(fila)
    ruta = raiz / "resumen.csv"
    campos = list(dict.fromkeys(k for f in filas for k in f))
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(filas)
    print(f"\n{tipo}: Molmo2 frente a puntos al azar en la silueta ({len(semillas)} semillas)")
    for f in filas:
        if f["metric"] in ("auc_angular", "f1_ref"):
            extra = f"  p={f['p_empirico']}" if "p_empirico" in f else ""
            print(f"  {f['n_views']:>2} vistas {f['metric']:12} Molmo2 {f['molmo2']:.3f}  silueta "
                  f"{f['silueta_media']:.3f} ± {f['silueta_std']:.3f} [{f['silueta_min']:.3f}, "
                  f"{f['silueta_max']:.3f}]{extra}")
    return ruta


def subir(carpeta: Path, mensaje: str) -> None:
    """ git pull, add, commit y push de la carpeta, con el mismo candado que las demás colas. """
    with open(REPO / ".git" / "run_plane_v02.lock", "w") as candado:
        fcntl.flock(candado, fcntl.LOCK_EX)
        subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", base.RAMA], cwd=REPO, check=True)
        subprocess.run(["git", "add", str(carpeta.relative_to(REPO))], cwd=REPO, check=True)
        if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO).returncode == 0:
            print("nada nuevo que subir")
            return
        subprocess.run(["git", "commit", "-m", mensaje], cwd=REPO, check=True)
        subprocess.run(["git", "push", "origin", base.RAMA], cwd=REPO, check=True)
        print("SUBIDO A GITHUB", flush=True)


def main() -> None:
    """ Genera los puntos de cada semilla, los post-procesa y resume; se detiene en el primer error. """
    parser = argparse.ArgumentParser(description="Puntos al azar dentro de la silueta (PI1 iii).")
    parser.add_argument("etapa", choices=["chequeo", "todo"])
    parser.add_argument("--tipo", choices=["axis_sym", "plane_sym", "ambos"], default="ambos")
    parser.add_argument("--semillas", type=int, default=10, help="cantidad de repeticiones")
    parser.add_argument("--semilla-inicial", type=int, default=0)
    parser.add_argument("--renders-root", type=Path, default=Path("/data/pespinoz/renders"))
    parser.add_argument("--objects-root", type=Path, default=Path("/data/pespinoz/objects"))
    parser.add_argument("--jobs", type=int, default=12, help="procesos de S_dat")
    parser.add_argument("--overwrite", action="store_true", help="regenera los puntos existentes")
    parser.add_argument("--sin-git", action="store_true", help="no sube los resultados")
    args = parser.parse_args()
    tipos = ["axis_sym", "plane_sym"] if args.tipo == "ambos" else [args.tipo]
    semillas = list(range(args.semilla_inicial, args.semilla_inicial + args.semillas))

    # Cada semilla se registra como un experimento más de run_plane_v02_224 (sin flujo, para que no
    # compare el texto del prompt, que aquí no existe).
    for tipo in tipos:
        for s in semillas:
            exp = nombre_exp(tipo, s)
            base.EXPERIMENTOS[exp] = dict(tipo=tipo, luz=LUZ, modelo="molmo", exp=exp,
                                          prompt=None, modo="auto", flujo=None)
    chequear(args, tipos)
    if args.etapa == "chequeo":
        return
    try:
        for tipo in tipos:
            raiz = carpeta_resultados(tipo)
            for s in semillas:
                exp = nombre_exp(tipo, s)
                salida = raiz / f"r{s:02d}"
                if semilla_lista(salida, tipo, exp) and not args.overwrite:
                    print(f"{exp}: ya está, se salta")
                    continue
                (salida / "logs").mkdir(parents=True, exist_ok=True)
                print(f"\n=== [{dt.datetime.now():%d/%m %H:%M}] {exp}", flush=True)
                print(f"{exp}: {generar(args, tipo, s)} archivos de puntos escritos", flush=True)
                if base.revisar(args.renders_root, exp) < TOTAL_OBJETOS:
                    raise RuntimeError(f"{exp}: no quedaron los 3 grupos en los {TOTAL_OBJETOS} objetos")
                base.posprocesar(args, exp, salida)
            resumir(tipo, semillas)
            if not args.sin_git:
                subir(raiz, f"Add random-silhouette baseline (PI1 iii): {CORTO[tipo]}, "
                            f"{PUNTOS_POR_VISTA} points per view, {len(semillas)} seeds")
    except (RuntimeError, ValueError, FileExistsError, StopIteration,
            subprocess.CalledProcessError) as error:
        sys.exit(f"[error] {error}")
    print("\nTerminó sin errores.")


if __name__ == "__main__":
    main()
