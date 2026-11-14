#!/usr/bin/env python3
"""
Aplica a las mallas .obj las mismas rotaciones que rotar_planares.py aplico a
las etiquetas, para que malla y plano de simetria sigan calzando.

Lee la matriz de cada objeto desde rotaciones.csv (salida de rotar_planares.py)
y rota los vertices ('v') y las normales ('vn') alrededor del origen del mundo:
v' = R v, igual que n' = R n y p' = R p en las etiquetas. El resto del .obj
(caras, coordenadas de textura, grupos, materiales) queda igual: una rotacion
tiene det = +1, asi que no invierte la orientacion de las caras.

Escribe una raiz de objetos nueva con la misma estructura que ~/data/objects,
para usarla tal cual con --objects-root en el resto del pipeline:

  <out-root>/curated_plane_sym_obj/<id>.obj   malla rotada
  <out-root>/curated_plane_sym_obj/<id>.txt   etiqueta rotada (copiada de --rot-dir)
  <out-root>/curated_plane_sym_obj/rotaciones.csv
  <out-root>/validacion_rotacion.csv          chequeo por objeto (ver abajo)

Validacion por objeto (aborta si alguna falla):
  1. La etiqueta de --rot-dir es exactamente R aplicada a la original.
  2. Releyendo el .obj escrito, R^T v' recupera los vertices originales.
  3. Error de simetria: se refleja cada vertice respecto al plano y se mide la
     distancia al vertice mas cercano (mediana, en unidades del bbox). Debe ser
     igual para (malla original, plano original) y (malla rotada, plano rotado).
     Como control se informa tambien (malla rotada, plano SIN rotar), que debe
     ser mucho mayor: muestra que el calce depende de aplicar la rotacion.

Uso:
  python3 rotar_mallas.py --obj-dir ~/data/objects/curated_plane_sym_obj \\
      --rot-dir ~/data/objects/curated_plane_sym_obj_rot_so3 \\
      --out-root ~/data/objects_rot_so3
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

SUBDIR = "curated_plane_sym_obj"
TOL_VERTICES = 1e-6     # el .obj se escribe con 8 decimales
TOL_ETIQUETA = 1e-6
TOL_SIMETRIA = 1e-4     # diferencia admitida en el error de simetria (bbox ~ 1)


def leer_rotaciones(path: Path) -> dict[str, np.ndarray]:
    """ object_id -> matriz 3x3, desde rotaciones.csv de rotar_planares.py. """
    with open(path, newline="", encoding="utf-8") as f:
        return {r["object_id"]: np.array([[float(r[f"r{i}{j}"]) for j in range(3)] for i in range(3)])
                for r in csv.DictReader(f)}


def leer_planos(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """ Lista de (normal, origen) de las lineas 'plane' de una etiqueta. """
    planos = []
    for linea in path.read_text(encoding="utf-8").splitlines():
        tok = linea.split()
        if tok and tok[0] == "plane" and len(tok) >= 7:
            v = np.array([float(t) for t in tok[1:7]])
            planos.append((v[:3], v[3:]))
    return planos


def rotar_obj(texto: str, R: np.ndarray) -> tuple[str, np.ndarray, np.ndarray]:
    """ Rota las lineas 'v' y 'vn' de un .obj.

    Args:
        * texto: contenido del .obj original
        * R: matriz de rotacion 3x3

    Returns:
        * tuple: texto rotado, vertices originales (N, 3), vertices rotados (N, 3)

    """
    salida, orig, rot = [], [], []
    for linea in texto.splitlines():
        tok = linea.split()
        if tok and tok[0] in ("v", "vn") and len(tok) >= 4:
            x = np.array([float(t) for t in tok[1:4]])
            y = R @ x
            extra = tok[4:]           # p. ej. color por vertice o w; no se rota
            salida.append(" ".join([tok[0], *(f"{c:.8f}" for c in y), *extra]))
            if tok[0] == "v":
                orig.append(x)
                rot.append(y)
        else:
            salida.append(linea)
    return "\n".join(salida) + "\n", np.array(orig), np.array(rot)


def vertices_obj(path: Path) -> np.ndarray:
    """ Vertices 'v' de un .obj como array (N, 3). """
    return np.array([[float(t) for t in linea.split()[1:4]]
                     for linea in path.read_text(encoding="utf-8").splitlines()
                     if linea.startswith("v ")])


def error_simetria(V: np.ndarray, n: np.ndarray, p: np.ndarray, arbol: cKDTree) -> float:
    """ Mediana de la distancia entre cada vertice reflejado y el vertice mas cercano. """
    n = n / np.linalg.norm(n)
    reflejados = V - 2.0 * ((V - p) @ n)[:, None] * n[None, :]
    d, _ = arbol.query(reflejados)
    return float(np.median(d))


def main() -> None:
    """ Rota y valida todas las mallas listadas en rotaciones.csv. """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obj-dir", required=True, type=Path, help="carpeta con los .obj y .txt originales")
    ap.add_argument("--rot-dir", required=True, type=Path,
                    help="salida de rotar_planares.py (.txt rotados + rotaciones.csv)")
    ap.add_argument("--out-root", required=True, type=Path,
                    help="raiz nueva; se crea <out-root>/curated_plane_sym_obj/")
    ap.add_argument("--forzar", action="store_true", help="sobrescribe una carpeta de salida no vacia")
    args = ap.parse_args()

    out_dir = args.out_root / SUBDIR
    if out_dir.resolve() in (args.obj_dir.resolve(), args.rot_dir.resolve()):
        sys.exit("[error] la salida no puede ser una carpeta de entrada")
    if out_dir.exists() and any(out_dir.iterdir()) and not args.forzar:
        sys.exit(f"[error] {out_dir} ya tiene archivos; usa --forzar para sobrescribir")
    out_dir.mkdir(parents=True, exist_ok=True)

    rotaciones = leer_rotaciones(args.rot_dir / "rotaciones.csv")
    objs = {p.stem.strip().lower(): p for p in args.obj_dir.glob("*.obj")}
    faltan = sorted(set(rotaciones) - set(objs))
    if faltan:
        sys.exit(f"[error] {len(faltan)} objetos de rotaciones.csv sin .obj, ej.: {faltan[:3]}")

    filas, fallas = [], []
    for k, (oid, R) in enumerate(sorted(rotaciones.items()), 1):
        obj_in = objs[oid]
        txt_in, txt_rot = args.obj_dir / f"{obj_in.stem}.txt", args.rot_dir / f"{obj_in.stem}.txt"

        # 1. etiqueta rotada == R * etiqueta original
        planos_o, planos_r = leer_planos(txt_in), leer_planos(txt_rot)
        err_etq = max((max(np.abs(R @ no - nr).max(), np.abs(R @ po - pr).max())
                       for (no, po), (nr, pr) in zip(planos_o, planos_r)), default=np.inf)
        if len(planos_o) != len(planos_r) or not planos_o:
            err_etq = np.inf

        # 2. escribir y releer la malla
        texto, V, Vr = rotar_obj(obj_in.read_text(encoding="utf-8"), R)
        obj_out = out_dir / obj_in.name
        obj_out.write_text(texto, encoding="utf-8")
        shutil.copy2(txt_rot, out_dir / txt_rot.name)
        Vr_leido = vertices_obj(obj_out)
        err_vert = (np.abs(Vr_leido @ R - V).max()
                    if Vr_leido.shape == V.shape else np.inf)   # (R^T v')^T = v'^T R

        # 3. error de simetria antes / despues / control
        arbol_o, arbol_r = cKDTree(V), cKDTree(Vr_leido)
        for j, ((no, po), (nr, pr)) in enumerate(zip(planos_o, planos_r)):
            e_orig = error_simetria(V, no, po, arbol_o)
            e_rot = error_simetria(Vr_leido, nr, pr, arbol_r)
            e_ctrl = error_simetria(Vr_leido, no, po, arbol_r)
            ok = (err_etq < TOL_ETIQUETA and err_vert < TOL_VERTICES
                  and abs(e_rot - e_orig) < TOL_SIMETRIA)
            filas.append({"object_id": oid, "plano": j, "n_vertices": len(V),
                          "err_etiqueta": f"{err_etq:.2e}", "err_vertices": f"{err_vert:.2e}",
                          "sim_original": f"{e_orig:.6f}", "sim_rotado": f"{e_rot:.6f}",
                          "sim_control_sin_rotar_plano": f"{e_ctrl:.6f}", "ok": int(ok)})
            if not ok:
                fallas.append(f"{oid} plano {j}: etq={err_etq:.1e} vert={err_vert:.1e} "
                              f"sim {e_orig:.4f} -> {e_rot:.4f}")
        if k % 100 == 0:
            print(f"  {k}/{len(rotaciones)}")

    shutil.copy2(args.rot_dir / "rotaciones.csv", out_dir / "rotaciones.csv")
    with open(args.out_root / "validacion_rotacion.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)

    so = np.array([float(r["sim_original"]) for r in filas])
    sr = np.array([float(r["sim_rotado"]) for r in filas])
    sc = np.array([float(r["sim_control_sin_rotar_plano"]) for r in filas])
    print(f"{len(rotaciones)} mallas rotadas -> {out_dir}")
    print(f"Planos validados: {len(filas)}  (fallas: {len(fallas)})")
    print(f"Error de simetria, mediana entre planos:  original {np.median(so):.5f}  "
          f"rotado {np.median(sr):.5f}  control sin rotar plano {np.median(sc):.5f}")
    print(f"Max |rotado - original|: {np.abs(sr - so).max():.2e}")
    print(f"Detalle: {args.out_root / 'validacion_rotacion.csv'}")
    if fallas:
        print("[error] fallas:\n  " + "\n  ".join(fallas[:20]))
        sys.exit(1)


if __name__ == "__main__":
    main()
