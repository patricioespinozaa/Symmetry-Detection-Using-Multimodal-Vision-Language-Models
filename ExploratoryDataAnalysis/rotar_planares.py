#!/usr/bin/env python3
"""
Rota las anotaciones de simetria planar con una rotacion aleatoria por objeto,
para estudiar la orientacion y visibilidad de los planos sin la pose canonica
de ShapeNet (casi todas las normales estan alineadas con X o Z).

Por ahora solo rota las etiquetas .txt (normal y origen de cada plano); las
mallas no se tocan. La matriz de cada objeto queda en rotaciones.csv para
aplicar despues exactamente la misma rotacion a la malla antes de renderizar.

Rotacion alrededor del origen del mundo, que es el punto al que miran las
camaras:  n' = R n,  p' = R p.

Modos:
  so3  rotacion uniforme en SO(3) (cuaternion unitario uniforme, Shoemake 1992).
       Las normales quedan uniformes en la esfera, como los ejes del caso axial.
  y    rotacion uniforme alrededor de +Y (el "arriba" de las camaras). Cambia
       el azimut de la normal pero conserva su elevacion: un plano vertical
       sigue vertical.

Cada objeto usa su propio generador, derivado de --semilla y de su object_id,
asi que su rotacion no depende de que otros objetos haya en la carpeta.

Uso:
  python3 rotar_planares.py --plane-dir ~/data/objects/curated_plane_sym_obj \\
      --out-dir ~/data/objects/curated_plane_sym_obj_rot_so3 --modo so3
"""
import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

MODOS = ("so3", "y")


def generador(semilla: int, object_id: str) -> np.random.Generator:
    """ Generador propio del objeto: igual entre corridas e independiente del resto. """
    h = int(hashlib.md5(object_id.encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng([semilla, h])


def rotacion_so3(rng: np.random.Generator) -> np.ndarray:
    """ Matriz de rotacion uniforme en SO(3) a partir de un cuaternion unitario uniforme. """
    u1, u2, u3 = rng.random(3)
    x = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    y = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    z = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    w = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rotacion_y(rng: np.random.Generator) -> np.ndarray:
    """ Rotacion uniforme alrededor del eje +Y. """
    a = rng.uniform(0, 2 * np.pi)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotar_txt(texto: str, R: np.ndarray) -> tuple[str, int]:
    """ Rota las lineas 'plane'/'axis' de una etiqueta y deja el resto igual.

    Args:
        * texto: contenido del .txt
        * R: matriz de rotacion 3x3

    Returns:
        * tuple[str, int]: texto rotado y numero de elementos rotados

    """
    salida, n = [], 0
    for linea in texto.splitlines():
        tok = linea.split()
        if tok and tok[0] in ("plane", "axis") and len(tok) >= 7:
            u = R @ np.array([float(t) for t in tok[1:4]])
            p = R @ np.array([float(t) for t in tok[4:7]])
            resto = " ".join(tok[7:])
            nueva = f"{tok[0]} " + " ".join(f"{v:.8f}" for v in (*u, *p))
            salida.append(f"{nueva} {resto}".rstrip())
            n += 1
        else:
            salida.append(linea)
    return "\n".join(salida) + "\n", n


def main() -> None:
    """ Rota todas las etiquetas de --plane-dir y guarda las matrices usadas. """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plane-dir", required=True, type=Path, help="carpeta con los .txt originales")
    ap.add_argument("--out-dir", required=True, type=Path, help="carpeta nueva para los .txt rotados")
    ap.add_argument("--modo", choices=MODOS, default="so3")
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--forzar", action="store_true", help="sobrescribe una carpeta de salida no vacia")
    args = ap.parse_args()

    if args.out_dir.resolve() == args.plane_dir.resolve():
        sys.exit("[error] --out-dir no puede ser la carpeta original")
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.forzar:
        sys.exit(f"[error] {args.out_dir} ya tiene archivos; usa --forzar para sobrescribir")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    txts = sorted(args.plane_dir.glob("*.txt"))
    if not txts:
        sys.exit(f"[error] no hay .txt en {args.plane_dir}")
    crear = rotacion_so3 if args.modo == "so3" else rotacion_y
    filas, sin_planos = [], []
    for txt in txts:
        oid = txt.stem.strip().lower()
        R = crear(generador(args.semilla, oid))
        if not (np.allclose(R @ R.T, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(R), 1.0)):
            sys.exit(f"[error] {oid}: la matriz no es una rotacion")
        texto, n = rotar_txt(txt.read_text(encoding="utf-8"), R)
        if n == 0:
            sin_planos.append(txt.name)
        (args.out_dir / txt.name).write_text(texto, encoding="utf-8")
        fila = {"object_id": oid, "modo": args.modo, "semilla": args.semilla, "n_elementos": n}
        fila.update({f"r{i}{j}": f"{R[i, j]:.10f}" for i in range(3) for j in range(3)})
        filas.append(fila)

    with open(args.out_dir / "rotaciones.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    print(f"{len(filas)} etiquetas rotadas ({args.modo}, semilla {args.semilla}) -> {args.out_dir}")
    print(f"Matrices: {args.out_dir / 'rotaciones.csv'}")
    if sin_planos:
        print(f"[aviso] {len(sin_planos)} .txt sin lineas plane/axis, ej.: {sin_planos[:3]}")


if __name__ == "__main__":
    main()
