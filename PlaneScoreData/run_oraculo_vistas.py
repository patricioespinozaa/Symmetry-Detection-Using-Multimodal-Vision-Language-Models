#!/usr/bin/env python3
# Selección de vistas por oráculo: cota superior de lo que se ganaría eligiendo, en cada objeto,
# las vistas más útiles para inferir la simetría. Usa la referencia para elegir las vistas, así que
# no es un método aplicable, solo un techo.
#
# Parte de los puntos reales de Molmo2 de la configuración base con 26 vistas (axis_v04 o
# plane_v02, 224px/flat, Flujo A multiimagen) y arma, para cada objeto, subconjuntos de k vistas
# de esas 26 (k = 6 y 14). No vuelve a correr ningún VLM ni usa GPU: cambia solo qué vistas, con
# sus puntos ya obtenidos, entran al estimador.
#
#   Oráculo: vistas elegidas con la referencia (criterio abajo).
#   Azar:    k vistas al azar entre las mismas 26 (control del efecto de partir de la corrida de 26).
#   Base:    la corrida real con k vistas (otra llamada a Molmo2), leída de REFERENCIA.
#
# Criterio del oráculo. Sea v_i la dirección con que la cámara i mira al objeto (-eye/|eye|; todas
# las cámaras apuntan al origen):
#   * Eje (dirección a). Una vista sirve si ve el eje de costado: el eje se proyecta con longitud
#     proporcional a sen(v_i, a) y, mirado de punta, se reduce a un punto. Candidatas: ángulo entre
#     v_i y a de al menos 60° (|v_i·a| <= 0.5). Además, EXP-E corta los planos de interpretación
#     de las vistas, cuya normal es m_i ∝ v_i × a; dos vistas con el mismo acimut alrededor del eje
#     dan el mismo plano. Por eso entre las candidatas se elige por máxima diversidad de m_i.
#   * Plano (normal n). Una vista sirve si ve el plano de canto: su traza es una recta y el plano de
#     interpretación coincide con el de simetría. Candidatas: |v_i·n| <= sen 15°. Entre ellas se
#     elige por máxima diversidad de la línea de vista v_i (vistas de canto desde distintos lados).
#     Con varios planos de referencia se usa el que tiene más vistas candidatas.
# Selección: solo vistas con al menos dos puntos de Molmo2 (lo que el modelo devolvió, no la
# referencia). Se parte de la candidata más favorable y se agrega cada vez la candidata más
# alejada de las ya elegidas (ángulo entre rectas); si faltan, se completa con las vistas útiles
# restantes ordenadas por calidad y, por último, con vistas sin puntos.
#
# El grupo de 26 vistas se copia sin cambios: su resultado debe coincidir con la corrida base y
# sirve de control del armado. Salida en resultados_finales/oraculo_vistas_<prompt>_224/.
#
#   python3 PlaneScoreData/run_oraculo_vistas.py chequeo
#   python3 PlaneScoreData/run_oraculo_vistas.py todo --tipo axis_sym
#   python3 PlaneScoreData/run_oraculo_vistas.py todo --tipo ambos --semillas 5
import argparse
import csv
import json
import math
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "PlaneScoreData"))
sys.path.insert(0, str(REPO))
import run_plane_v02_224 as base  # noqa: E402
from pipeline_common.datasets import OBJECTS_SUBDIR  # noqa: E402

SIZE, LUZ = 224, "flat"
ORIGEN_VISTAS = 26                 # grupo de Molmo2 del que se toman las vistas
SUBCONJUNTOS = (6, 14)             # tamaños de los subconjuntos
VISTAS = (6, 14, 26)
TOTAL_OBJETOS = 850
PROMPTS = {"axis_sym": "axis_v04", "plane_sym": "plane_v02"}
COS_EJE = 0.5                      # |v·a| <= 0.5: al menos 60° entre la vista y el eje
SEN_CANTO = math.sin(math.radians(15))   # |v·n| <= sen 15°: vista de canto
METRICAS = ("n_objects", "auc_angular", "angular_error_median", "precision_10deg", "f1_ref")


# ----------------------------------------------------------------------------------------- nombres
def nombre_exp(prompt: str, semilla: int | None) -> str:
    """ Id del experimento (también nombre del JSON de puntos); semilla None es el oráculo. """
    return f"{prompt}_oraculo" if semilla is None else f"{prompt}_subazar_r{semilla:02d}"


def carpeta_resultados(prompt: str) -> Path:
    """ Carpeta de resultados_finales del experimento. """
    return REPO / "resultados_finales" / f"oraculo_vistas_{prompt}_{SIZE}"


def subcarpeta(semilla: int | None) -> str:
    return "oraculo" if semilla is None else f"r{semilla:02d}"


def carpetas_render(renders: Path, tipo: str) -> list[Path]:
    """ Carpetas 224/flat del tipo con metadatos de cámara. """
    return [d for d in sorted((renders / tipo).glob(f"*/{SIZE}/{LUZ}"))
            if (d / "metadata_all.json").exists()]


# -------------------------------------------------------------------------------------- referencia
def leer_referencia(ruta: Path) -> list[np.ndarray]:
    """ Direcciones de eje o normales de plano del .txt de referencia, normalizadas. """
    elementos = []
    for linea in ruta.read_text().splitlines():
        partes = linea.split()
        if partes and partes[0] in ("axis", "plane"):
            v = np.array([float(x) for x in partes[1:4]])
            elementos.append(v / np.linalg.norm(v))
    return elementos


# --------------------------------------------------------------------------------------- selección
def direcciones(images_sent: list[dict]) -> np.ndarray:
    """ Dirección con que cada cámara mira al objeto (desde la cámara hacia el origen). """
    eye = np.array([i["eye"] for i in images_sent], dtype=float)
    return -eye / np.linalg.norm(eye, axis=1, keepdims=True)


def geometria(v: np.ndarray, elemento: np.ndarray, tipo: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Calidad, condición de candidata y recta de diversidad de cada vista.

    Args:
        * v: direcciones de vista (N, 3)
        * elemento: dirección del eje o normal del plano de referencia
        * tipo: 'axis_sym' o 'plane_sym'

    Returns:
        * calidad: mayor es mejor (sen del ángulo vista-eje, o 1 - |v·n|)
        * candidata: cumple el umbral del tipo
        * recta: vector cuya dirección mide la diversidad (normal del plano de interpretación en
          eje, línea de vista en plano)

    """
    c = np.abs(v @ elemento)
    if tipo == "axis_sym":
        recta = np.cross(v, elemento)
        norma = np.linalg.norm(recta, axis=1, keepdims=True)
        recta = np.divide(recta, norma, out=np.zeros_like(recta), where=norma > 1e-9)
        return np.sqrt(np.clip(1 - c ** 2, 0, 1)), c <= COS_EJE, recta
    return 1 - c, c <= SEN_CANTO, v


def mas_diversas(indices: list[int], k: int, calidad: np.ndarray, recta: np.ndarray) -> list[int]:
    """ Muestreo del más lejano entre rectas: parte de la mejor y agrega la más alejada. """
    if not indices or k <= 0:
        return []
    restantes = sorted(indices, key=lambda i: -calidad[i])
    elegidas = [restantes.pop(0)]
    while restantes and len(elegidas) < k:
        def distancia(i: int) -> float:
            return min(math.acos(min(1.0, abs(float(recta[i] @ recta[j])))) for j in elegidas)
        mejor = max(restantes, key=lambda i: (round(distancia(i), 9), calidad[i]))
        restantes.remove(mejor)
        elegidas.append(mejor)
    return elegidas


def elegir_oraculo(v: np.ndarray, utiles: np.ndarray, elementos: list[np.ndarray], tipo: str,
                   k: int) -> tuple[list[int], dict]:
    """ Vistas del oráculo y un resumen de la selección.

    Args:
        * v: direcciones de vista (N, 3)
        * utiles: vistas con al menos dos puntos de Molmo2
        * elementos: ejes o normales de referencia del objeto
        * tipo: 'axis_sym' o 'plane_sym'
        * k: vistas a elegir

    Returns:
        * list[int]: índices elegidos, en el orden original
        * dict: candidatas útiles, cuántas de las elegidas cumplen el umbral y su calidad media

    """
    def preparar(e: np.ndarray):
        calidad, candidata, recta = geometria(v, e, tipo)
        return int((candidata & utiles).sum()), float(np.sort(calidad[utiles])[::-1][:k].mean()
                                                       if utiles.any() else 0), calidad, candidata, recta
    # Con varios elementos de referencia, el que tiene más candidatas útiles (luego mejor calidad)
    n_cand, _, calidad, candidata, recta = max((preparar(e) for e in elementos), key=lambda t: t[:2])
    todas = range(len(v))
    elegidas = mas_diversas([i for i in todas if candidata[i] and utiles[i]], k, calidad, recta)
    for grupo in ([i for i in todas if utiles[i] and i not in elegidas],
                  [i for i in todas if not utiles[i]]):
        faltan = k - len(elegidas)
        elegidas += sorted(grupo, key=lambda i: -calidad[i])[:max(faltan, 0)]
    elegidas = sorted(elegidas)
    return elegidas, {"candidatas_utiles": n_cand,
                      "elegidas_candidatas": int(sum(candidata[i] and utiles[i] for i in elegidas)),
                      "calidad_media": round(float(calidad[elegidas].mean()), 6)}


def elegir_azar(utiles: np.ndarray, k: int, rng: np.random.Generator) -> list[int]:
    """ k vistas al azar entre las útiles; si no alcanzan, se completa con las demás. """
    u, resto = np.flatnonzero(utiles), np.flatnonzero(~utiles)
    elegidas = list(rng.permutation(u)[:k])
    if len(elegidas) < k:
        elegidas += list(rng.permutation(resto)[:k - len(elegidas)])
    return sorted(int(i) for i in elegidas)


def subgrupo(grupo: dict, indices: list[int], exp: str, extra: dict) -> dict:
    """ Grupo con las vistas indicadas, reindexadas 0..k-1, con sus puntos sin cambios. """
    imagenes, puntos = [], {}
    for j, i in enumerate(indices):
        imagenes.append({**grupo["images_sent"][i], "img_idx": j})
        puntos[str(j)] = grupo["points_by_image"].get(str(grupo["images_sent"][i]["img_idx"]), [])
    return {"experiment_id": exp, **extra, "prompt_used": None, "raw_output": None,
            "points_by_image": puntos, "images_sent": imagenes,
            "n_points": sum(len(p) for p in puntos.values()),
            "n_images_with_points": sum(bool(p) for p in puntos.values())}


# ------------------------------------------------------------------------------------- generación
def generar(args: argparse.Namespace, tipo: str, semilla: int | None) -> int:
    """ Escribe molmo_multiview_<exp>.json en cada objeto; devuelve cuántos escribió.

    Para el oráculo también escribe seleccion.csv con el resumen de la selección por objeto.

    """
    prompt, exp = PROMPTS[tipo], nombre_exp(PROMPTS[tipo], semilla)
    objetos = args.objects_root / OBJECTS_SUBDIR[tipo]
    escritos, filas = 0, []
    for carpeta in carpetas_render(args.renders_root, tipo):
        destino = carpeta / f"molmo_multiview_{exp}.json"
        if destino.exists() and not args.overwrite and semilla is not None:
            continue
        objeto = carpeta.parents[1].name
        origen = json.loads((carpeta / f"molmo_multiview_{prompt}.json").read_text(encoding="utf-8"))
        grupo = origen.get(str(ORIGEN_VISTAS))
        if grupo is None:              # misma cobertura que la corrida base de 26 vistas
            continue
        v = direcciones(grupo["images_sent"])
        utiles = np.array([len(grupo["points_by_image"].get(str(i["img_idx"]), [])) >= 2
                           for i in grupo["images_sent"]])
        elementos = leer_referencia(objetos / f"{objeto}.txt")
        salida = {str(ORIGEN_VISTAS): subgrupo(grupo, list(range(len(v))), exp,
                                               {"origen": "copia del grupo de 26 vistas"})}
        for k in SUBCONJUNTOS:
            if semilla is None:
                indices, info = elegir_oraculo(v, utiles, elementos, tipo, k)
                filas.append({"objeto": objeto, "k": k, "utiles": int(utiles.sum()), **info,
                              "vistas": " ".join(grupo["images_sent"][i]["filename"] for i in indices)})
                extra = {"origen": "oraculo", **info}
            else:
                rng = np.random.default_rng([semilla, zlib.crc32(objeto.encode()), k])
                indices, extra = elegir_azar(utiles, k, rng), {"origen": "subconjunto_azar",
                                                               "semilla": semilla}
            salida[str(k)] = subgrupo(grupo, indices, exp, {**extra, "puntos_vlm":
                                                             f"molmo_multiview_{prompt}.json"})
        if destino.exists() and not args.overwrite:
            continue
        tmp = destino.with_suffix(".tmp")
        tmp.write_text(json.dumps(salida), encoding="utf-8")
        tmp.replace(destino)
        escritos += 1
    if semilla is None and filas:
        ruta = carpeta_resultados(prompt) / "seleccion.csv"
        ruta.parent.mkdir(parents=True, exist_ok=True)
        with open(ruta, "w", newline="", encoding="utf-8") as f:
            escritor = csv.DictWriter(f, fieldnames=list(filas[0]))
            escritor.writeheader()
            escritor.writerows(filas)
    return escritos


def chequear(args: argparse.Namespace, tipos: list[str]) -> None:
    """ Valida renders, puntos de Molmo2, cámaras y referencia antes de escribir nada. """
    if base.molmo_to_ndc(0, 0)[0] != 1.0:
        sys.exit("[error] pipeline_common/camera.py no tiene la corrección NDC")
    for tipo in tipos:
        prompt, errores, con_grupo = PROMPTS[tipo], [], 0
        carpetas = carpetas_render(args.renders_root, tipo)
        objetos = args.objects_root / OBJECTS_SUBDIR[tipo]
        if len(carpetas) != TOTAL_OBJETOS:
            errores.append(f"{len(carpetas)} renders a {SIZE}px/{LUZ}, se esperaban {TOTAL_OBJETOS}")
        for carpeta in carpetas:
            ruta = carpeta / f"molmo_multiview_{prompt}.json"
            txt = objetos / f"{carpeta.parents[1].name}.txt"
            if not ruta.exists() or not txt.exists():
                errores.append(f"falta {ruta if not ruta.exists() else txt}")
                continue
            if not leer_referencia(txt):
                errores.append(f"{txt} sin ejes ni planos")
            g = json.loads(ruta.read_text(encoding="utf-8")).get(str(ORIGEN_VISTAS))
            if g is None:
                continue
            con_grupo += 1
            if len(g["images_sent"]) != ORIGEN_VISTAS or any("eye" not in i for i in g["images_sent"]):
                errores.append(f"{carpeta}: grupo de {ORIGEN_VISTAS} vistas incompleto o sin 'eye'")
            if [i["img_idx"] for i in g["images_sent"]] != list(range(ORIGEN_VISTAS)):
                errores.append(f"{carpeta}: img_idx fuera de orden")
        if not base.REFERENCIA[tipo][1].exists():
            errores.append(f"falta {base.REFERENCIA[tipo][1]}")
        if errores:
            sys.exit(f"[error] {tipo}: {len(errores)} problemas, por ejemplo {errores[:3]}")
        print(f"{tipo}: {len(carpetas)} objetos, {con_grupo} con el grupo de {ORIGEN_VISTAS} vistas "
              f"de {prompt}")
    print("[ok] Chequeo sin problemas")


# ---------------------------------------------------------------------------------------- resumen
def lista(salida: Path, tipo: str, exp: str) -> bool:
    """ True si el experimento ya tiene su CSV final con las 3 cantidades de vistas. """
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


def fila_exp(tipo: str, semilla: int | None, vistas: int) -> dict:
    exp = nombre_exp(PROMPTS[tipo], semilla)
    carpeta = carpeta_resultados(PROMPTS[tipo]) / subcarpeta(semilla)
    ruta = next(carpeta.glob(f"experiments_*/{tipo}_final_comparison.csv"))
    return leer_fila(ruta, base.id_final(base.EXPERIMENTOS[exp]), vistas)


def resumir(tipo: str, semillas: list[int]) -> Path:
    """ Base, oráculo y azar por número de vistas; comprueba que el grupo de 26 coincida con la base. """
    id_base, csv_base = base.REFERENCIA[tipo]
    filas = []
    for vistas in VISTAS:
        real = leer_fila(csv_base, id_base, vistas)
        oraculo = fila_exp(tipo, None, vistas)
        azar = [fila_exp(tipo, s, vistas) for s in semillas]
        for metrica in METRICAS:
            if not real.get(metrica):
                continue
            valores = np.array([float(f[metrica]) for f in azar])
            fila = {"symmetry_type": tipo, "n_views": vistas, "metric": metrica,
                    "base": float(real[metrica]), "oraculo": float(oraculo[metrica]),
                    "azar_media": round(float(valores.mean()), 6) if len(valores) else "",
                    "azar_std": round(float(valores.std(ddof=1)), 6) if len(valores) > 1 else "",
                    "n_semillas": len(valores)}
            filas.append(fila)
            if vistas == ORIGEN_VISTAS and abs(fila["oraculo"] - fila["base"]) > 1e-9:
                print(f"[aviso] {tipo} {metrica} con 26 vistas: oráculo {fila['oraculo']} != base "
                      f"{fila['base']}; el armado no reproduce la corrida base", flush=True)
    ruta = carpeta_resultados(PROMPTS[tipo]) / "resumen.csv"
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)
    print(f"\n{tipo}: base, oráculo y subconjuntos al azar de las {ORIGEN_VISTAS} vistas "
          f"({len(semillas)} semillas)")
    for f in filas:
        if f["metric"] in ("auc_angular", "f1_ref"):
            azar = f"{f['azar_media']:.3f} ± {f['azar_std'] or 0:.3f}" if f["n_semillas"] else "-"
            print(f"  {f['n_views']:>2} vistas {f['metric']:12} base {f['base']:.3f}  "
                  f"oráculo {f['oraculo']:.3f}  azar {azar}")
    return ruta


def subir(carpeta: Path, mensaje: str) -> None:
    """ git pull, add, commit y push de la carpeta, con el mismo candado que las demás colas. """
    import fcntl
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
    """ Genera los subconjuntos, los post-procesa y resume; se detiene en el primer error. """
    parser = argparse.ArgumentParser(description="Selección de vistas por oráculo (cota superior).")
    parser.add_argument("etapa", choices=["chequeo", "todo"])
    parser.add_argument("--tipo", choices=["axis_sym", "plane_sym", "ambos"], default="ambos")
    parser.add_argument("--semillas", type=int, default=5, help="subconjuntos al azar de control")
    parser.add_argument("--renders-root", type=Path, default=Path("/data/pespinoz/renders"))
    parser.add_argument("--objects-root", type=Path, default=Path("/data/pespinoz/objects"))
    parser.add_argument("--jobs", type=int, default=12, help="procesos de S_dat")
    parser.add_argument("--overwrite", action="store_true", help="regenera los subconjuntos existentes")
    parser.add_argument("--sin-git", action="store_true", help="no sube los resultados")
    args = parser.parse_args()
    tipos = ["axis_sym", "plane_sym"] if args.tipo == "ambos" else [args.tipo]
    semillas = list(range(args.semillas))

    # Cada condición se registra como un experimento más de run_plane_v02_224 (sin flujo, para que
    # no compare el texto del prompt, que aquí no se envía).
    for tipo in tipos:
        for s in [None, *semillas]:
            exp = nombre_exp(PROMPTS[tipo], s)
            base.EXPERIMENTOS[exp] = dict(tipo=tipo, luz=LUZ, modelo="molmo", exp=exp,
                                          prompt=PROMPTS[tipo], modo="auto", flujo=None)
    chequear(args, tipos)
    if args.etapa == "chequeo":
        return
    try:
        for tipo in tipos:
            raiz = carpeta_resultados(PROMPTS[tipo])
            for s in [None, *semillas]:
                exp = nombre_exp(PROMPTS[tipo], s)
                salida = raiz / subcarpeta(s)
                if lista(salida, tipo, exp) and not args.overwrite:
                    print(f"{exp}: ya está, se salta")
                    continue
                (salida / "logs").mkdir(parents=True, exist_ok=True)
                print(f"\n=== {exp}", flush=True)
                print(f"{exp}: {generar(args, tipo, s)} archivos de puntos escritos", flush=True)
                base.revisar(args.renders_root, exp)
                base.posprocesar(args, exp, salida)
            resumir(tipo, semillas)
            if not args.sin_git:
                subir(raiz, f"Add oracle view-selection upper bound: {PROMPTS[tipo]}, "
                            f"{len(semillas)} random seeds")
    except (RuntimeError, ValueError, FileExistsError, StopIteration,
            subprocess.CalledProcessError) as error:
        sys.exit(f"[error] {error}")
    print("\nTerminó sin errores.")


if __name__ == "__main__":
    main()
