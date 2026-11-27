#!/usr/bin/env python3
# Disposición de las cámaras en plano: ¿cuántas de las n_v vistas quedan "de canto" respecto del plano
# de simetría, es decir, con la cámara dentro del plano o cerca de él? En esas vistas el objeto se ve
# simétrico y la línea central de la imagen coincide con la proyección del plano. La pregunta viene de
# la caída del plano con 14 vistas, que aparece con envío multiimagen (AUC 0.628) y no con una vista
# por llamada (0.831).
#
# Una vista está de canto si el ángulo theta entre la dirección de la cámara (mira al origen) y el plano
# es menor que --umbral grados: theta = asin(|n . e|), con n la normal del plano y e la dirección de la
# cámara. Reporta, para cada n_v:
#   1. las vistas elegidas (las mismas que recibe el VLM: n_v renders espaciados en la secuencia de
#      Fibonacci de metadata_all.json) y cuántas quedan de canto respecto de los planos canónicos X y Z;
#   2. por objeto, cuántas vistas quedan de canto respecto de su plano de referencia (el más cercano
#      si tiene varios), y el error de plane_v02 con envío multiimagen y con una vista por llamada
#      agrupado por ese número.
# Solo lee archivos existentes; no usa GPU.
#
#   python3 PlaneScoreData/diagnostico_camaras_canto.py --salida resultados_finales/diagnosticos/camaras_canto.csv
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "PlaneScoreData"))
sys.path.insert(0, str(REPO / "Mapping"))
sys.path.insert(0, str(REPO))
from compare_plane_scores import errores_por_objeto  # noqa: E402
from evaluate import AUC_ANGULAR_MAX, auc_from_errors, parse_true_label  # noqa: E402
from pipeline_common.datasets import OBJECTS_SUBDIR  # noqa: E402

VISTAS = (6, 14, 26)
MODOS = {"multiimagen": "plane_v02_s224_flat_ndcfix_sdat_mp1_nomesh",
         "una vista": "plane_v02_single_s224_flat_ndcfix_sdat_mp1_nomesh"}
CANONICOS = {"X": np.array([1.0, 0.0, 0.0]), "Z": np.array([0.0, 0.0, 1.0])}


def entradas_vistas(metadata: list[dict], vistas: int) -> list[dict]:
    """ Las n_v vistas que recibe el VLM: copia de get_n_views_entries de
    MolmoPointing/molmo_multiview_runner.py (que importa torch). """
    total = len(metadata)
    if vistas >= total:
        return sorted(metadata, key=lambda e: e["index"])
    indices = {int(round(i)) for i in np.linspace(0, total - 1, vistas)}
    return sorted((m for m in metadata if m["index"] in indices), key=lambda e: e["index"])


def theta(normal: np.ndarray, eye: list[float]) -> float:
    """ Ángulo en grados entre la dirección de la cámara y el plano de normal dada. """
    e = np.asarray(eye, dtype=float)
    e = e / np.linalg.norm(e)
    return float(np.degrees(np.arcsin(min(1.0, abs(float(np.dot(normal, e)))))))


def main() -> None:
    """ Cuenta vistas de canto por número de vistas y cruza ese conteo con el error del plano. """
    ap = argparse.ArgumentParser(description="Vistas de canto respecto del plano de simetría.")
    ap.add_argument("--renders-root", type=Path, default=Path("/data/pespinoz/renders"))
    ap.add_argument("--objects-root", type=Path, default=Path("/data/pespinoz/objects"))
    ap.add_argument("--umbral", type=float, default=15.0, help="grados bajo los que una vista está de canto")
    ap.add_argument("--salida", type=Path, required=True, help="CSV por objeto y número de vistas")
    args = ap.parse_args()
    raiz = args.renders_root / "plane_sym"
    objetos = args.objects_root / OBJECTS_SUBDIR["plane_sym"]
    carpetas = sorted(d for d in raiz.glob("*/224/flat") if (d / "metadata_all.json").exists())
    if not carpetas:
        sys.exit(f"[error] no hay renders 224/flat en {raiz}")

    # 1. Planos canónicos, con las cámaras del primer objeto; se comprueba que todos usan las mismas
    ref = json.loads((carpetas[0] / "metadata_all.json").read_text(encoding="utf-8"))
    print(f"plane_sym: {len(carpetas)} objetos, {len(ref)} vistas en la secuencia de Fibonacci, "
          f"umbral {args.umbral:.0f} grados")
    direcciones = lambda md: {m["index"]: np.asarray(m["eye"], float) / np.linalg.norm(m["eye"]) for m in md}  # noqa: E731
    dir_ref, distintas = direcciones(ref), 0
    for c in carpetas[1:]:
        d = direcciones(json.loads((c / "metadata_all.json").read_text(encoding="utf-8")))
        if d.keys() != dir_ref.keys() or any(np.abs(d[k] - dir_ref[k]).max() > 1e-4 for k in d):
            distintas += 1
    print(f"objetos con cámaras distintas a las del primero: {distintas}")
    for v in VISTAS:
        sel = entradas_vistas(ref, v)
        thetas = {p: [theta(n, m["eye"]) for m in sel] for p, n in CANONICOS.items()}
        print(f"\n{v} vistas (índices {[m['index'] for m in sel]})")
        for p in CANONICOS:
            de_canto = sum(t < args.umbral for t in thetas[p])
            print(f"  plano {p}: {de_canto} de {v} vistas de canto "
                  f"(theta: {', '.join(f'{t:.0f}' for t in thetas[p])})")

    # 2. Por objeto: vistas de canto respecto de su plano de referencia, frente al error
    errores = {}
    for modo, exp in MODOS.items():
        ruta = raiz / f"eval_s224_flat_{exp}_triangulation_results.json"
        if ruta.exists():
            errores[modo] = {v: errores_por_objeto(ruta, str(v)) for v in VISTAS}
        else:
            print(f"[aviso] falta {ruta.name}")
    filas = []
    for c in carpetas:
        objeto = c.parents[1].name
        txt = objetos / f"{objeto}.txt"
        if not txt.exists():
            continue
        normales = [np.asarray(el["normal"]) for el in parse_true_label(txt)["elements"]]
        md = json.loads((c / "metadata_all.json").read_text(encoding="utf-8"))
        for v in VISTAS:
            sel = entradas_vistas(md, v)
            de_canto = sum(min(theta(n, m["eye"]) for n in normales) < args.umbral for m in sel)
            fila = {"object_id": objeto, "n_views": v, "vistas_de_canto": de_canto}
            for modo in errores:
                fila[f"error_{modo.replace(' ', '_')}"] = errores[modo][v].get(objeto, 90.0)
            filas.append(fila)

    for v in VISTAS:
        sub = [f for f in filas if f["n_views"] == v]
        print(f"\n{v} vistas: vistas de canto respecto del plano de referencia, por objeto")
        print(f"  {'de canto':>8} {'objetos':>8}" + "".join(f" {'AUC ' + m:>16} {'med. ' + m:>16}" for m in errores))
        grupos = defaultdict(list)
        for f in sub:
            grupos[min(f["vistas_de_canto"], 4)].append(f)      # 4 o más en un solo grupo
        for k in sorted(grupos):
            fs = grupos[k]
            texto = f"  {('>=4' if k == 4 else str(k)):>8} {len(fs):>8}"
            for m in errores:
                e = [f[f"error_{m.replace(' ', '_')}"] for f in fs]
                texto += f" {auc_from_errors(e, AUC_ANGULAR_MAX):16.3f} {np.median(e):16.1f}"
            print(texto)

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    with open(args.salida, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)
    print(f"\nCSV por objeto: {args.salida}")


if __name__ == "__main__":
    main()
