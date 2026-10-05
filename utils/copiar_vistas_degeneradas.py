#!/usr/bin/env python3
"""
Copia al repositorio un set de vistas "degeneradas" de un objeto axial y uno
planar, para ilustrar en la tesis por qué la vista de descripción se restringe
a elevaciones en (-60°, +60°).

Vistas degeneradas (convención de ImagesGenerator/export_fibonacci_views.py):
  - Elevación extrema |el| > 70°: con 114 vistas son los índices 0-3 y 110-113
    (el = ±89, ±79, ±74, ±71). La vista 0 es cenital (+Y).
  - Planar: en pose canónica las normales son horizontales, así que las vistas
    cenitales ven el plano de canto (como una línea).
  - Axial: el eje solo se proyecta como un punto si la cámara mira A LO LARGO
    del eje. Como los ejes están rotados al azar, el script elige por defecto el
    objeto axial con el eje más vertical (máximo |d·Y|), para que la vista
    cenital sea efectivamente degenerada, y copia además la vista más alineada
    con el eje.
Para comparar, se copia también una vista de la franja central (|el| < 60°)
que ve el eje o el plano de lado.

Estructura esperada (la del README):
  <data-root>/objects/curated_axis_sym_obj/<id>.txt
  <data-root>/objects/curated_plane_sym_obj/<id>.txt
  <data-root>/renders/{axis_sym,plane_sym}/<id>/<size>/<lighting>/IND_*.png
                                                              metadata_all.json

Salida (por defecto imgs/vistas_degeneradas/ en la raíz del repo):
  axial_<id>/   y   planar_<id>/   con las imágenes copiadas
  vistas_degeneradas.csv           qué se copió y por qué
  montaje_vistas_degeneradas.png   (si matplotlib está disponible)

Uso en el servidor (desde la raíz del repo):
  python3 utils/copiar_vistas_degeneradas.py --data-root ~/data7
  python3 utils/copiar_vistas_degeneradas.py --data-root ~/data7 \\
      --axis-id <id> --plane-id <id> --size 1134 --lighting flat
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT_DEFAULT = REPO / "imgs" / "vistas_degeneradas"
EXTREMA = 70.0          # |el| > 70°: vista degenerada (texto de la tesis)
RANGO_DESC = 60.0       # la vista de descripción se elige en (-60°, +60°)


# ----------------------------------------------------------------- lectura
def leer_anotacion(path):
    """Primer elemento de simetría del .txt: (tipo, dirección unitaria)."""
    for line in path.read_text(encoding="utf-8").splitlines():
        tok = line.split()
        if tok and tok[0] in ("axis", "plane"):
            u = np.array([float(t) for t in tok[1:4]])
            return tok[0], u / np.linalg.norm(u)
    return None, None


def leer_metadata(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = next(v for v in data.values() if isinstance(v, list))
    return sorted(data, key=lambda e: int(e["index"]))


def carpeta_render(renders, tipo, oid, size, lighting, avisar=True):
    base = renders / tipo / oid
    if not base.is_dir():
        return None
    pedida = base / str(size) / lighting
    if pedida.is_dir():
        return pedida
    # si no existe el tamaño pedido, usar el mayor disponible
    sizes = sorted((d for d in base.iterdir() if d.is_dir() and (d / lighting).is_dir()),
                   key=lambda d: int(d.name) if d.name.isdigit() else -1)
    if not sizes:
        return None
    if avisar:
        print(f"[aviso] {pedida} no existe; uso {sizes[-1] / lighting}")
    return sizes[-1] / lighting


def elegir_objeto(objs_dir, renders, tipo, size, lighting, criterio):
    """Recorre las anotaciones y elige el objeto que maximiza `criterio(u)`
    entre los que tienen renders."""
    mejor = None
    for txt in sorted(objs_dir.glob("*.txt")):
        _, u = leer_anotacion(txt)
        if u is None:
            continue
        if carpeta_render(renders, tipo, txt.stem, size, lighting, avisar=False) is None:
            continue
        score = criterio(u)
        if mejor is None or score > mejor[0]:
            mejor = (score, txt.stem)
    return mejor[1] if mejor else None


# ----------------------------------------------------------------- selección
def vistas_a_copiar(meta, u, nombre):
    """Devuelve [(entrada, motivo, cos)] para un objeto con dirección de
    simetría u (eje axial o normal del plano).

    cos = |cámara · u|. Eje: cos≈1 mira a lo largo del eje (el eje se ve como
    un punto, degenerado) y cos≈0 lo ve de lado (bien). Plano: cos≈0 ve el plano
    de canto (como una línea, degenerado para describir el objeto) y un cos
    intermedio muestra ambas mitades en perspectiva oblicua."""
    eyes = np.array([np.asarray(e["eye"], dtype=float).reshape(-1)[:3] for e in meta])
    w = eyes / np.linalg.norm(eyes, axis=1, keepdims=True)
    cos_u = np.abs(w @ u)
    sel = {}
    for e in meta:                              # elevación extrema
        if abs(e["elevation"]) > EXTREMA:
            sel[int(e["index"])] = f"elevación extrema ({e['elevation']:+d}°)"
    centro = [i for i, e in enumerate(meta) if abs(e["elevation"]) < RANGO_DESC]
    if nombre == "axial":
        i_largo = int(np.argmax(cos_u))
        ang = np.degrees(np.arccos(min(1.0, cos_u[i_largo])))
        sel.setdefault(int(meta[i_largo]["index"]),
                       f"degenerada: mira a lo largo del eje (a {ang:.1f}°)")
        i_comp = min(centro, key=lambda i: cos_u[i])
        motivo = "comparación: franja central, ve el eje de lado"
    else:
        i_comp = min(centro, key=lambda i: abs(cos_u[i] - 0.5))
        motivo = "comparación: franja central, vista oblicua del plano"
    sel[int(meta[i_comp]["index"])] = f"{motivo} (el {meta[i_comp]['elevation']:+d}°)"
    por_idx = {int(e["index"]): e for e in meta}
    return [(por_idx[i], sel[i], float(cos_u[i])) for i in sorted(sel)]


# ----------------------------------------------------------------- estadística
def estadistica(objects, meta, umbral, out):
    """% de vistas degeneradas sobre los conjuntos completos.

    (1) Por elevación: |el| > EXTREMA. Igual para todos los objetos (las 114
        cámaras son las mismas), por lo que se reporta una sola vez.
    (2) Por geometría, por objeto. Con w la dirección de la cámara y u el eje o la
        normal del plano (de la anotación):
          axial : degenerada si la cámara está a menos de `umbral` grados del
                  eje (el eje se proyecta casi como un punto): |w·u| > cos(umbral)
          planar: degenerada si la cámara está a menos de `umbral` grados del
                  plano (el plano se ve casi de canto): |w·n| < sin(umbral);
                  con varios planos, basta con uno.
        Se reporta sobre las 114 vistas y sobre las candidatas en (-60°, +60°).
    """
    eyes = np.array([np.asarray(e["eye"], dtype=float).reshape(-1)[:3] for e in meta])
    w = eyes / np.linalg.norm(eyes, axis=1, keepdims=True)
    el = np.array([e["elevation"] for e in meta])
    extrema = np.abs(el) > EXTREMA
    cand = (el > -RANGO_DESC) & (el < RANGO_DESC)
    n = len(meta)
    print("\n" + "=" * 72)
    print("ESTADÍSTICA DE VISTAS DEGENERADAS")
    print("=" * 72)
    print(f"(1) Por elevación (igual para todos los objetos): |el| > {EXTREMA:g}° en "
          f"{extrema.sum()}/{n} vistas ({100 * extrema.mean():.1f}%); "
          f"candidatas en (-{RANGO_DESC:g}°, +{RANGO_DESC:g}°): {cand.sum()}/{n} "
          f"({100 * cand.mean():.1f}%); excluidas por el filtro: {n - cand.sum()}")

    filas = []
    print(f"\n(2) Por geometría (umbral {umbral:g}°):")
    print(f"  {'tipo':<8}{'objetos':>8}{'% vistas degeneradas':>24}{'% en candidatas':>18}"
          f"{'% obj. con >=1 deg. en candidatas':>36}")
    for nombre, carpeta in (("axial", "curated_axis_sym_obj"), ("planar", "curated_plane_sym_obj")):
        degs = []
        for txt in sorted((objects / carpeta).glob("*.txt")):
            us = []
            for line in txt.read_text(encoding="utf-8").splitlines():
                tok = line.split()
                if tok and tok[0] in ("axis", "plane"):
                    u = np.array([float(t) for t in tok[1:4]])
                    us.append(u / np.linalg.norm(u))
            if not us:
                continue
            c = np.abs(w @ np.array(us).T)                  # (n_vistas, n_elementos)
            if nombre == "axial":
                deg = (c > np.cos(np.radians(umbral))).any(axis=1)
            else:
                deg = (c < np.sin(np.radians(umbral))).any(axis=1)
            degs.append(deg)
        if not degs:
            continue
        D = np.array(degs)                                   # (n_objetos, n_vistas)
        fila = {"tipo": nombre, "objetos": len(D), "umbral_deg": umbral,
                "pct_vistas_degeneradas": round(100 * D.mean(), 2),
                "pct_degeneradas_en_candidatas": round(100 * D[:, cand].mean(), 2),
                "pct_degeneradas_en_extremas": round(100 * D[:, extrema].mean(), 2),
                "pct_obj_con_degenerada_en_candidatas": round(100 * D[:, cand].any(axis=1).mean(), 2)}
        filas.append(fila)
        print(f"  {nombre:<8}{len(D):>8}{fila['pct_vistas_degeneradas']:>23.1f}%"
              f"{fila['pct_degeneradas_en_candidatas']:>17.1f}%"
              f"{fila['pct_obj_con_degenerada_en_candidatas']:>35.1f}%")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "estadistica_vistas_degeneradas.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        wr.writeheader()
        wr.writerows(filas)
    print(f"\n[ok] {out / 'estadistica_vistas_degeneradas.csv'}")


# ----------------------------------------------------------------- montaje
def montaje(grupos, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
    except ImportError:
        print("[aviso] matplotlib no disponible: se omite el montaje")
        return
    ncol = max(len(v) for _, v in grupos)
    fig, axs = plt.subplots(len(grupos), ncol, figsize=(1.7 * ncol, 2.0 * len(grupos)),
                            squeeze=False)
    for r, (titulo, vistas) in enumerate(grupos):
        for c in range(ncol):
            ax = axs[r, c]
            ax.axis("off")
            if c < len(vistas):
                img_path, e = vistas[c]
                ax.imshow(mpimg.imread(img_path))
                ax.set_title(f"#{e['index']}  el {e['elevation']:+d}°", fontsize=7)
        axs[r, 0].text(-0.08, 0.5, titulo, transform=axs[r, 0].transAxes, rotation=90,
                       ha="right", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"[ok] {out_png}")


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True,
                    help="carpeta con objects/ y renders/ (p. ej. ~/data7)")
    ap.add_argument("--renders-root", default=None, help="si no está en <data-root>/renders")
    ap.add_argument("--objects-root", default=None, help="si no está en <data-root>/objects")
    ap.add_argument("--axis-id", default=None, help="objeto axial (por defecto: eje más vertical)")
    ap.add_argument("--plane-id", default=None, help="objeto planar (por defecto: normal más horizontal)")
    ap.add_argument("--size", default="1134", help="resolución del render (por defecto 1134)")
    ap.add_argument("--lighting", default="flat")
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--umbral", type=float, default=20.0,
                    help="grados para la estadística geométrica (por defecto 20)")
    ap.add_argument("--solo-estadistica", action="store_true",
                    help="solo calcula e imprime la estadística, sin copiar imágenes")
    args = ap.parse_args()

    data = Path(args.data_root).expanduser()
    renders = Path(args.renders_root).expanduser() if args.renders_root else data / "renders"
    objects = Path(args.objects_root).expanduser() if args.objects_root else data / "objects"
    out = Path(args.out_dir).expanduser()
    for d in (renders, objects):
        if not d.is_dir():
            sys.exit(f"[error] no existe {d}")

    casos = [
        ("axial", "axis_sym", objects / "curated_axis_sym_obj", args.axis_id,
         lambda u: abs(u[1])),                   # eje lo más vertical posible
        ("planar", "plane_sym", objects / "curated_plane_sym_obj", args.plane_id,
         lambda u: -abs(u[1])),                  # normal horizontal (plano vertical)
    ]
    filas, grupos, meta_ref = [], [], None
    if args.solo_estadistica:
        casos = []
        meta_ref = leer_metadata(next(renders.glob("*/*/*/*/metadata_all.json")))
    for nombre, tipo, objs_dir, oid, criterio in casos:
        if oid is None:
            oid = elegir_objeto(objs_dir, renders, tipo, args.size, args.lighting, criterio)
            if oid is None:
                sys.exit(f"[error] no encontré objetos {nombre} con renders en {renders / tipo}")
        _, u = leer_anotacion(objs_dir / f"{oid}.txt")
        src = carpeta_render(renders, tipo, oid, args.size, args.lighting)
        if src is None or u is None:
            sys.exit(f"[error] faltan renders o anotación para {nombre} {oid}")
        meta = leer_metadata(src / "metadata_all.json")
        meta_ref = meta
        dst = out / f"{nombre}_{oid}"
        dst.mkdir(parents=True, exist_ok=True)
        print(f"\n{nombre}: {oid}  dirección de simetría {np.round(u, 3)}  "
              f"|u·Y| = {abs(u[1]):.3f}  ({src})")
        vistas = []
        for e, motivo, cos_u in vistas_a_copiar(meta, u, nombre):
            img = src / e["filename"]
            if not img.exists():
                print(f"  [aviso] no existe {img}")
                continue
            shutil.copy2(img, dst / img.name)
            vistas.append((dst / img.name, e))
            filas.append({"tipo": nombre, "object_id": oid, "index": e["index"],
                          "archivo": img.name, "elevacion": e["elevation"],
                          "azimut": e["azimuth"], "cos_camara_simetria": round(cos_u, 3),
                          "motivo": motivo})
            print(f"  #{e['index']:>3}  el {e['elevation']:+3d}°  {img.name}  <- {motivo}")
        grupos.append((f"{nombre}\n{oid[:8]}", vistas))

    out.mkdir(parents=True, exist_ok=True)
    if filas:
        with open(out / "vistas_degeneradas.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
            w.writeheader()
            w.writerows(filas)
        montaje(grupos, out / "montaje_vistas_degeneradas.png")
    estadistica(objects, meta_ref, args.umbral, out)
    print(f"\nSalida en {out}")


if __name__ == "__main__":
    main()
