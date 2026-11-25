#!/usr/bin/env python3
# Recupera, objeto por objeto, la transformacion que Aguirre y Sipiran (WACV 2026) aplicaron a las
# mallas planares de sus conjuntos de robustez, y deja la misma salida que rotar_planares.py para
# transformar con rotar_mallas.py las mallas LIMPIAS propias (ver docs/plan_rotacion_aguirre.md).
#
# Sus conjuntos (curated_plane_sym_obj_rot_{1,3,5}_noise, _rot_20_extracted) traen la malla
# transformada con ruido y la etiqueta transformada, pero no hay version limpia. Como conservan el
# orden de los vertices, la transformacion sale de Procrustes ortogonal entre la malla canonica
# propia y la de 1 % de ruido:  Q = argmin sum |Q (a_i - a_c) - (b_i - b_c)|^2,  Q^T Q = I.
# Q NO se restringe a det = +1: en la muestra local ~60 % de los objetos tiene det Q = -1 (una
# rotacion compuesta con una reflexion; con Q el residuo es el del ruido y con la mejor rotacion
# pura es unas 10 veces mayor). rotar_mallas.py invierte el orden de las caras cuando det Q = -1.
#
# Verificaciones por objeto (umbrales fijados antes de ver los 850):
#   1. mismo numero de vertices en las tres mallas (propia, 1 % y 3 %);
#   2. escala de Umeyama |s - 1| <= UMBRAL_ESCALA (no reescalaron la malla);
#   3. residuo RMS / R_max <= UMBRAL_RMS (del orden del ruido; si el ruido es 1 % de R_max por
#      coordenada, el RMS 3D esperado es sqrt(3) * 0.01 = 0.017);
#   4. traslacion / R_max <= UMBRAL_TRASLACION (transformaron en torno al origen, como
#      rotar_mallas.py);
#   5. la Q estimada desde la malla de 3 % tiene el mismo determinante y su diferencia con la de
#      1 % es compatible con el ruido: angulo / sd <= UMBRAL_Z_R3, con sd la desviacion esperada de
#      Procrustes, sqrt(s1^2 + s3^2) / sqrt(lambda_min) (s = ruido por coordenada, lambda_min =
#      menor valor propio del tensor de inercia de la malla). Un umbral fijo en grados no sirve: la
#      incertidumbre va de ~0.04 a ~0.7 grados segun el tamano y la forma de la malla;
#   6. las etiquetas propias transformadas (n' = Q n) coinciden con las de ellos como conjunto:
#      mismo numero de planos y, con el emparejamiento que minimiza el peor error,
#      arccos|n'.n| <= UMBRAL_ANGULO para todos.
#
# La etiqueta que se usa es la propia transformada con Q (calza exactamente con la malla que se
# renderiza); la de ellos solo se usa para verificar. Salida en --out-dir:
#   <id>.txt                          etiqueta propia transformada (solo objetos aceptados)
#   rotaciones.csv                    object_id, modo, semilla, n_elementos, det, r00..r22
#   validacion_rotacion_aguirre.csv   una fila por objeto con todas las verificaciones
#
# --aguirre puede ser la carpeta con las subcarpetas _rot_1_noise/_rot_3_noise o el .zip descargado
# (se lee sin extraer). Termina con codigo 1 si algun objeto se rechaza, para revisarlo a mano.
#
#   python3 ExploratoryDataAnalysis/rotaciones_aguirre.py \
#       --obj-dir ~/data/objects/curated_plane_sym_obj \
#       --aguirre ~/data/curated_plane_sym_obj_rot_robustness.zip \
#       --out-dir ~/data/objects/curated_plane_sym_obj_rot_aguirre
import argparse
import csv
import itertools
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rotar_planares import rotar_txt  # noqa: E402

CARPETA_1 = "curated_plane_sym_obj_rot_1_noise"
CARPETA_3 = "curated_plane_sym_obj_rot_3_noise"
UMBRAL_ESCALA = 0.01
UMBRAL_RMS = 0.03
UMBRAL_TRASLACION = 0.01
UMBRAL_Z_R3 = 6.0
UMBRAL_ANGULO = 2.0


class Fuente:
    """ Lee archivos de los conjuntos de Aguirre y Sipiran desde una carpeta o desde el .zip. """

    def __init__(self, ruta: Path):
        """ Abre el .zip o recuerda la carpeta, e indexa sus archivos por (carpeta, id, ext). """
        self.zip = zipfile.ZipFile(ruta) if ruta.suffix.lower() == ".zip" else None
        self.ruta = ruta
        nombres = (self.zip.namelist() if self.zip else
                   [p.relative_to(ruta).as_posix() for p in ruta.glob("*/*") if p.is_file()])
        self.indice = {}
        for nombre in nombres:
            partes = nombre.rstrip("/").split("/")
            if len(partes) >= 2 and "." in partes[-1]:
                stem, ext = partes[-1].rsplit(".", 1)
                self.indice[(partes[-2], stem.strip().lower(), ext.lower())] = nombre

    def ids(self, carpeta: str) -> set[str]:
        """ Ids con malla en una subcarpeta. """
        return {oid for (c, oid, ext) in self.indice if c == carpeta and ext == "obj"}

    def texto(self, carpeta: str, oid: str, ext: str) -> str | None:
        """ Contenido de <carpeta>/<id>.<ext>, o None si no esta. """
        nombre = self.indice.get((carpeta, oid, ext))
        if nombre is None:
            return None
        datos = self.zip.read(nombre) if self.zip else (self.ruta / nombre).read_bytes()
        return datos.decode("utf-8", errors="replace")


def vertices(texto: str) -> np.ndarray:
    """ Vertices 'v' de un .obj como array (N, 3). """
    filas = [linea.split()[1:4] for linea in texto.splitlines() if linea.startswith("v ")]
    return np.array(filas, dtype=np.float64).reshape(-1, 3)


def planos(texto: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """ Lista de (normal unitaria, origen) de las lineas 'plane' de una etiqueta. """
    salida = []
    for linea in texto.splitlines():
        tok = linea.split()
        if tok and tok[0] == "plane" and len(tok) >= 7:
            v = np.array([float(t) for t in tok[1:7]])
            salida.append((v[:3] / np.linalg.norm(v[:3]), v[3:]))
    return salida


def procrustes(A: np.ndarray, B: np.ndarray) -> dict:
    """ Transformacion ortogonal, escala, traslacion y residuo que llevan A a B.

    Args:
        * A: vertices de la malla canonica (N, 3)
        * B: vertices de la malla de Aguirre y Sipiran, en el mismo orden (N, 3)

    Returns:
        * dict: Q (3x3 ortogonal, det +1 o -1), det, escala de Umeyama, traslacion t
          (B ~ Q A + t) y RMS del residuo |Q (a_i - a_c) - (b_i - b_c)|

    """
    ca, cb = A.mean(0), B.mean(0)
    Ac, Bc = A - ca, B - cb
    U, S, Vt = np.linalg.svd(Ac.T @ Bc)
    Q = Vt.T @ U.T
    residuo = Ac @ Q.T - Bc
    return {"Q": Q, "det": int(round(np.linalg.det(Q))), "escala": float(S.sum() / (Ac ** 2).sum()),
            "t": cb - Q @ ca, "rms": float(np.sqrt((residuo ** 2).sum(1).mean()))}


def sd_rotacion(A: np.ndarray, sigma: float) -> float:
    """ Desviacion (radianes, eje peor determinado) de la Q de Procrustes con ruido sigma. """
    Ac = A - A.mean(0)
    inercia = (Ac ** 2).sum() * np.eye(3) - Ac.T @ Ac
    return float(sigma / np.sqrt(max(np.linalg.eigvalsh(inercia).min(), 1e-300)))


def angulo_rotacion(R: np.ndarray) -> float:
    """ Angulo en grados de una rotacion (det +1). """
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def parte_rotacion(Q: np.ndarray) -> np.ndarray:
    """ Q si det Q = +1; si no, -Q (Q = -R: la rotacion R compuesta con la inversion x -> -x). """
    return Q if np.linalg.det(Q) > 0 else -Q


def emparejar(propios: list[np.ndarray], ajenos: list[np.ndarray]) -> tuple[float, list[int]]:
    """ Emparejamiento de normales (sin reutilizar) que minimiza el peor error, invariante al signo.

    Args:
        * propios: normales propias ya transformadas
        * ajenos: normales de la etiqueta de Aguirre y Sipiran (misma cantidad)

    Returns:
        * tuple[float, list[int]]: peor error en grados y, para cada plano propio, el indice del
          plano ajeno emparejado

    """
    error = np.degrees(np.arccos(np.clip(np.abs(np.array(propios) @ np.array(ajenos).T), 0, 1)))
    mejor = min(itertools.permutations(range(len(ajenos))),
                key=lambda perm: max(error[i, j] for i, j in enumerate(perm)))
    return float(max(error[i, j] for i, j in enumerate(mejor))), list(mejor)


def validar(oid: str, obj: Path, fuente: Fuente) -> tuple[dict, np.ndarray | None, str | None]:
    """ Estima Q de un objeto y corre las verificaciones.

    Args:
        * oid: id en minusculas
        * obj: malla canonica propia (.obj, con su .txt al lado)
        * fuente: conjuntos de Aguirre y Sipiran

    Returns:
        * tuple: fila del CSV de validacion, Q (o None si no se pudo estimar) y etiqueta propia
          transformada (o None)

    """
    fila = {"object_id": oid}
    motivos = []
    A = vertices(obj.read_text(encoding="utf-8", errors="replace"))
    textos = {c: fuente.texto(c, oid, "obj") for c in (CARPETA_1, CARPETA_3)}
    etiqueta_ajena = fuente.texto(CARPETA_1, oid, "txt")
    if textos[CARPETA_1] is None or etiqueta_ajena is None:
        fila.update(aceptado=0, motivo="sin_malla_o_etiqueta_aguirre")
        return fila, None, None
    B1 = vertices(textos[CARPETA_1])
    B3 = vertices(textos[CARPETA_3]) if textos[CARPETA_3] is not None else None
    fila.update(n_vertices=len(A), n_vertices_aguirre=len(B1))
    if len(A) != len(B1) or len(A) < 3:
        fila.update(aceptado=0, motivo="n_vertices_distinto")
        return fila, None, None

    r_max = float(np.linalg.norm(A, axis=1).max())
    p1 = procrustes(A, B1)
    Q = p1["Q"]
    fila.update(r_max=f"{r_max:.6f}", det=p1["det"],
                angulo_R_grados=f"{angulo_rotacion(parte_rotacion(Q)):.3f}",
                escala=f"{p1['escala']:.6f}",
                traslacion_norma=f"{np.linalg.norm(p1['t']):.6f}",
                traslacion_rel=f"{np.linalg.norm(p1['t']) / r_max:.6f}",
                rms_procrustes=f"{p1['rms']:.6f}", rms_relativo=f"{p1['rms'] / r_max:.6f}",
                incertidumbre_Q_grados=f"{np.degrees(sd_rotacion(A, p1['rms'] / np.sqrt(3))):.4f}")
    if abs(p1["escala"] - 1) > UMBRAL_ESCALA:
        motivos.append("escala")
    if p1["rms"] / r_max > UMBRAL_RMS:
        motivos.append("rms")
    if np.linalg.norm(p1["t"]) / r_max > UMBRAL_TRASLACION:
        motivos.append("traslacion")

    if B3 is None or len(B3) != len(A):
        fila["dif_R_3pct_grados"] = ""
        motivos.append("malla_3pct")
    else:
        p3 = procrustes(A, B3)
        dif = angulo_rotacion(Q.T @ p3["Q"]) if p3["det"] == p1["det"] else 180.0
        sd = np.degrees(sd_rotacion(A, np.hypot(p1["rms"], p3["rms"]) / np.sqrt(3)))
        fila.update(dif_R_3pct_grados=f"{dif:.4f}", dif_R_3pct_z=f"{dif / sd:.3f}")
        if dif / sd > UMBRAL_Z_R3:
            motivos.append("R_3pct_discrepa")

    etiqueta_propia = obj.with_suffix(".txt").read_text(encoding="utf-8")
    etiqueta_rotada, n_elementos = rotar_txt(etiqueta_propia, Q)
    propios = planos(etiqueta_rotada)
    ajenos = planos(etiqueta_ajena)
    fila.update(n_planos_propios=len(propios), n_planos_aguirre=len(ajenos),
                n_elementos=n_elementos)
    if not propios or len(propios) != len(ajenos):
        fila["error_max_grados"] = ""
        motivos.append("n_planos_distinto")
    else:
        peor, perm = emparejar([n for n, _ in propios], [n for n, _ in ajenos])
        fila["error_max_grados"] = f"{peor:.4f}"
        fila["emparejamiento"] = " ".join(map(str, perm))
        # Distancia de cada plano al origen (signo de la normal alineado), / R_max.
        offsets = [abs(n @ p - np.sign(n @ ajenos[j][0]) * (ajenos[j][0] @ ajenos[j][1])) / r_max
                   for (n, p), j in zip(propios, perm)]
        fila["dif_offset_rel"] = f"{max(offsets):.6f}"
        if peor > UMBRAL_ANGULO:
            motivos.append("angulo")
    fila.update(aceptado=int(not motivos), motivo=";".join(motivos))
    return fila, Q, etiqueta_rotada


def resumen(filas: list[dict]) -> None:
    """ Imprime la distribucion de cada verificacion. """
    dets = [f["det"] for f in filas if "det" in f]
    print(f"  det Q = -1 en {dets.count(-1)}/{len(dets)} objetos (rotacion + reflexion)")
    for nombre in ("angulo_R_grados", "rms_relativo", "escala", "traslacion_rel",
                   "incertidumbre_Q_grados", "dif_R_3pct_grados", "dif_R_3pct_z",
                   "error_max_grados", "dif_offset_rel"):
        v = np.array([float(f[nombre]) for f in filas if f.get(nombre) not in (None, "")])
        if len(v):
            print(f"  {nombre:18} min {v.min():10.5f}  mediana {np.median(v):10.5f}  "
                  f"p95 {np.percentile(v, 95):10.5f}  max {v.max():10.5f}")


def main() -> None:
    """ Estima y valida la transformacion de cada objeto y escribe la salida de rotar_planares. """
    ap = argparse.ArgumentParser(description="Transformaciones de Aguirre y Sipiran (Procrustes).")
    ap.add_argument("--obj-dir", required=True, type=Path,
                    help="mallas y etiquetas canonicas propias (.obj + .txt)")
    ap.add_argument("--aguirre", required=True, type=Path,
                    help="carpeta con curated_plane_sym_obj_rot_{1,3}_noise/ o el .zip descargado")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="carpeta nueva; queda lista para rotar_mallas.py --rot-dir")
    ap.add_argument("--forzar", action="store_true",
                    help="sobrescribe una carpeta de salida no vacia")
    args = ap.parse_args()

    if args.out_dir.resolve() == args.obj_dir.resolve():
        sys.exit("[error] --out-dir no puede ser la carpeta original")
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.forzar:
        sys.exit(f"[error] {args.out_dir} ya tiene archivos; usa --forzar para sobrescribir")
    objs = {p.stem.strip().lower(): p for p in sorted(args.obj_dir.glob("*.obj"))
            if p.with_suffix(".txt").exists()}
    if not objs:
        sys.exit(f"[error] no hay .obj con .txt en {args.obj_dir}")
    fuente = Fuente(args.aguirre)
    ids_aguirre = fuente.ids(CARPETA_1)
    if not ids_aguirre:
        sys.exit(f"[error] no hay mallas de {CARPETA_1} en {args.aguirre}")
    print(f"{len(objs)} objetos propios; {len(ids_aguirre)} en {CARPETA_1}; "
          f"{len(set(objs) & ids_aguirre)} en comun; {len(set(objs) - ids_aguirre)} sin par")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    filas, rotaciones = [], []
    for k, (oid, obj) in enumerate(objs.items(), 1):
        fila, Q, etiqueta = validar(oid, obj, fuente)
        filas.append(fila)
        if fila["aceptado"]:
            destino = args.out_dir / obj.with_suffix(".txt").name
            destino.write_text(etiqueta, encoding="utf-8", newline="\n")
            r = {"object_id": oid, "modo": "aguirre", "semilla": "",
                 "n_elementos": fila["n_elementos"], "det": fila["det"]}
            r.update({f"r{i}{j}": f"{Q[i, j]:.10f}" for i in range(3) for j in range(3)})
            rotaciones.append(r)
        if k % 50 == 0:
            print(f"  {k}/{len(objs)}", flush=True)

    columnas = list(dict.fromkeys(c for f in filas for c in f))
    with open(args.out_dir / "validacion_rotacion_aguirre.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columnas, lineterminator="\n")
        w.writeheader()
        w.writerows(filas)
    if rotaciones:
        with open(args.out_dir / "rotaciones.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rotaciones[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rotaciones)

    rechazados = [f for f in filas if not f["aceptado"]]
    print(f"\nAceptados {len(filas) - len(rechazados)}/{len(filas)} -> {args.out_dir}")
    resumen(filas)
    if rechazados:
        print(f"[aviso] {len(rechazados)} rechazados (revisar a mano):")
        for f in rechazados[:30]:
            print(f"  {f['object_id']}: {f['motivo']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
