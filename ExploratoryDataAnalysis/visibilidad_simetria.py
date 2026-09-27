#!/usr/bin/env python3
"""
P1 y P2: direcciones de los elementos de simetria y su visibilidad desde
las vistas fijas del pipeline.

Indice de visibilidad por (elemento, vista), unificado para ejes y planos:

    delta = arcsin(|u . w|)   en grados, 0 = vista ideal, 90 = peor vista

    u = direccion del eje (axial) o normal del plano (planar), unitaria
    w = direccion unitaria desde el punto del elemento (px,py,pz) a la camara

    Axial : delta = 0 cuando la camara mira perpendicular al eje (el eje se
            ve como una linea completa); 90 cuando mira a lo largo del eje.
    Planar: delta = 0 cuando la camara esta dentro del plano (el plano se ve
            como una linea y la imagen queda simetrica); 90 cuando mira
            perpendicular al plano.

Si un objeto tiene varios elementos de simetria, la visibilidad de una vista
es la del elemento MEJOR visto (delta minimo entre elementos).

Seleccion de vistas: replica MolmoPointing/molmo_multiview_runner.py
    {int(round(i)) for i in np.linspace(0, total - 1, n_views)}
Se calcula para cada n_v de --nv y ademas para todas las vistas ("all").

Salidas:
  <prefijo>_elementos.csv   una fila por elemento de simetria (P1)
  <prefijo>_objetos.csv     una fila por objeto, con metricas por n_v (P2)

Uso:
  python3 visibilidad_simetria.py \\
      --axis-dir ~/data/objects/curated_axis_sym_obj \\
      --plane-dir ~/data/objects/curated_plane_sym_obj \\
      --metadata ~/data/renders/axis_sym/<ID>/448/flat/metadata_all.json \\
      --categorias categorias_objetos_final.csv \\
      --out-prefix visibilidad
"""
import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

AXES = ["X", "Y", "Z"]


# ----------------------------------------------------------------- lectura
def load_camera_dirs(path):
    """Devuelve array (N,3) con las posiciones 'eye', ordenadas por indice."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    entries = None
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        lists = [v for v in data.values() if isinstance(v, list) and v and isinstance(v[0], dict)]
        if lists:
            entries = lists[0]
        elif all(isinstance(v, dict) for v in data.values()):
            entries = [data[k] for k in sorted(data, key=lambda k: int(k))]
    if not entries or "eye" not in entries[0]:
        sys.exit(f"[error] No reconozco la estructura de {path}. Primer nivel: "
                 f"{type(data).__name__}; revisa y ajusta load_camera_dirs().")
    idx_key = next((k for k in ("index", "idx", "ind", "view_index") if k in entries[0]), None)
    if idx_key:
        entries = sorted(entries, key=lambda e: int(e[idx_key]))
    eyes = np.array([np.asarray(e["eye"], dtype=float).reshape(-1)[:3] for e in entries])
    print(f"[info] {path}: {len(eyes)} vistas (orden por "
          f"{'campo ' + idx_key if idx_key else 'posicion en la lista'}); "
          f"eye[0] = {np.round(eyes[0], 4).tolist()}")
    return eyes


def parse_symmetry_txt(path):
    """Lee el .txt de anotacion. Devuelve lista de elementos:
    {'tipo': 'axis'|'plane', 'u': (3,), 'p': (3,), 'angles': [..]}"""
    elems = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            tok = line.split()
            if not tok:
                continue
            if tok[0] in ("axis", "plane"):
                vals = [float(t) for t in tok[1:7]]
                u = np.array(vals[:3])
                norm = np.linalg.norm(u)
                if norm == 0:
                    continue
                elems.append({"tipo": tok[0], "u": u / norm,
                              "p": np.array(vals[3:6]) if len(vals) >= 6 else np.zeros(3),
                              "angles": []})
            elif tok[0] == "angles" and elems:
                elems[-1]["angles"] = [float(t) for t in tok[1:]]
    return elems


def load_categories(path):
    if not path:
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["object_id"].strip().lower(): r.get("nombre_categoria", "")
                for r in csv.DictReader(f)}


def selected_indices(total, n_views):
    return sorted({int(round(i)) for i in np.linspace(0, total - 1, n_views)})


# ----------------------------------------------------------------- calculo
def deviation_deg(u, p, eyes):
    """delta (grados) de un elemento respecto a cada vista. eyes: (N,3)."""
    w = eyes - p
    w = w / np.linalg.norm(w, axis=1, keepdims=True)
    return np.degrees(np.arcsin(np.clip(np.abs(w @ u), 0.0, 1.0)))


def direction_stats(u):
    elev = math.degrees(math.asin(min(1.0, abs(u[1]))))       # respecto al plano horizontal XZ
    k = int(np.argmax(np.abs(u)))
    ang_axis = math.degrees(math.acos(min(1.0, abs(u[k]))))   # a eje del mundo mas cercano
    return elev, AXES[k], ang_axis


def histogram(values, edges):
    counts = Counter()
    for v in values:
        for lo, hi in zip(edges[:-1], edges[1:]):
            if lo <= v < hi or (hi == edges[-1] and v == hi):
                counts[(lo, hi)] += 1
                break
    return [(lo, hi, counts[(lo, hi)]) for lo, hi in zip(edges[:-1], edges[1:])]


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axis-dir", required=True)
    ap.add_argument("--plane-dir", required=True)
    ap.add_argument("--metadata", required=True, help="un metadata_all.json (las poses son iguales para todos)")
    ap.add_argument("--categorias", default=None, help="opcional: agrega nombre_categoria")
    ap.add_argument("--nv", default="1,6,14,26")
    ap.add_argument("--umbral", type=float, default=15.0,
                    help="delta maximo (grados) para contar una vista como 'buena'")
    ap.add_argument("--out-prefix", default="visibilidad")
    args = ap.parse_args()

    eyes = load_camera_dirs(args.metadata)
    total = len(eyes)
    nv_list = [int(x) for x in args.nv.split(",")]
    selections = {str(n): selected_indices(total, n) for n in nv_list}
    selections["all"] = list(range(total))
    for k, idx in selections.items():
        if k != "all":
            print(f"[info] n_v={k}: indices {idx}")

    cats = load_categories(args.categorias)
    elem_rows, obj_rows = [], []
    sin_elementos = []

    for tipo, folder in (("axial", args.axis_dir), ("planar", args.plane_dir)):
        for txt in sorted(Path(folder).glob("*.txt")):
            oid = txt.stem.strip().lower()
            elems = parse_symmetry_txt(txt)
            if not elems:
                sin_elementos.append(str(txt))
                continue
            dev = np.vstack([deviation_deg(e["u"], e["p"], eyes) for e in elems])  # (E,N)
            best_per_view = dev.min(axis=0)                                          # (N,)

            for j, e in enumerate(elems):
                elev, ax, ang = direction_stats(e["u"])
                elem_rows.append({
                    "object_id": oid, "tipo_simetria": tipo,
                    "nombre_categoria": cats.get(oid, ""), "idx_elemento": j,
                    "ux": round(e["u"][0], 5), "uy": round(e["u"][1], 5), "uz": round(e["u"][2], 5),
                    "elevacion_dir_deg": round(elev, 2),
                    "eje_mundo_cercano": ax, "ang_a_eje_mundo_deg": round(ang, 2),
                    "angles": " ".join(f"{a:g}" for a in e["angles"]),
                })

            row = {"object_id": oid, "tipo_simetria": tipo,
                   "nombre_categoria": cats.get(oid, ""), "n_elementos": len(elems)}
            for k, idx in selections.items():
                d = best_per_view[idx]
                row[f"delta_min_{k}"] = round(float(d.min()), 2)
                row[f"delta_media_{k}"] = round(float(d.mean()), 2)
                row[f"n_buenas_{k}"] = int((d <= args.umbral).sum())
                row[f"ind_mejor_{k}"] = idx[int(np.argmin(d))]
            obj_rows.append(row)

    if sin_elementos:
        print(f"[aviso] {len(sin_elementos)} .txt sin elementos axis/plane legibles, ej.: {sin_elementos[:3]}")

    def write(path, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write(f"{args.out_prefix}_elementos.csv", elem_rows)
    write(f"{args.out_prefix}_objetos.csv", obj_rows)

    # ------------------------------------------------------------- resumen P1
    print("\n" + "=" * 78)
    print("P1. DIRECCIONES DE SIMETRIA")
    print("=" * 78)
    for tipo in ("axial", "planar"):
        rows = [r for r in elem_rows if r["tipo_simetria"] == tipo]
        n_obj = len({r["object_id"] for r in rows})
        dist = dict(sorted(Counter(Counter(r["object_id"] for r in rows).values()).items()))
        print(f"\n{tipo.upper()}: {len(rows)} elementos en {n_obj} objetos "
              f"(elementos por objeto -> n objetos: {dist})")
        print("  Elevacion de la direccion (0 = horizontal, 90 = vertical/Y):")
        for lo, hi, c in histogram([r["elevacion_dir_deg"] for r in rows], [0, 15, 30, 45, 60, 75, 90]):
            print(f"    {lo:>2}-{hi:<2} deg: {c:5d}  {'#' * round(50 * c / max(1, len(rows)))}")
        alineados = sum(1 for r in rows if r["ang_a_eje_mundo_deg"] <= 5)
        print(f"  Alineados (<=5 deg) con un eje del mundo: {alineados}/{len(rows)} "
              f"({100 * alineados / max(1, len(rows)):.1f}%)  "
              f"por eje: {dict(Counter(r['eje_mundo_cercano'] for r in rows if r['ang_a_eje_mundo_deg'] <= 5))}")

    # ------------------------------------------------------------- resumen P2
    print("\n" + "=" * 78)
    print(f"P2. VISIBILIDAD DESDE LAS VISTAS SELECCIONADAS (vista buena: delta <= {args.umbral:g} deg)")
    print("=" * 78)
    header = f"{'tipo':<8}{'n_v':>5}{'delta_min med.':>16}{'delta_min p90':>15}{'% obj con >=1 buena':>22}"
    print(header)
    print("-" * len(header))
    for tipo in ("axial", "planar"):
        rows = [r for r in obj_rows if r["tipo_simetria"] == tipo]
        for k in selections:
            dm = np.array([r[f"delta_min_{k}"] for r in rows])
            pct = 100 * np.mean([r[f"n_buenas_{k}"] > 0 for r in rows])
            print(f"{tipo:<8}{k:>5}{np.median(dm):>16.1f}{np.percentile(dm, 90):>15.1f}{pct:>21.1f}%")
        print()

    # ------------------------------------------------------------- verificacion visual
    print("=" * 78)
    print("VERIFICACION VISUAL SUGERIDA (abre estas imagenes: deberian mostrar el eje/plano como una linea)")
    print("=" * 78)
    for tipo in ("planar", "axial"):
        rows = sorted((r for r in obj_rows if r["tipo_simetria"] == tipo),
                      key=lambda r: r["delta_min_all"])[:3]
        for r in rows:
            print(f"  [{tipo}] {r['object_id']}  mejor vista: IND_{r['ind_mejor_all']:02d}  "
                  f"(delta = {r['delta_min_all']} deg)")

    print(f"\nSalidas: {args.out_prefix}_elementos.csv, {args.out_prefix}_objetos.csv")


if __name__ == "__main__":
    main()
