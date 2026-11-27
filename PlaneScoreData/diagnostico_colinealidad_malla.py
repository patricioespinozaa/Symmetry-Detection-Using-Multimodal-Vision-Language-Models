#!/usr/bin/env python3
# Diagnóstico de la retroproyección sobre la malla en plano: ¿los puntos 3D de plane_v02 quedan casi
# sobre una recta? Si es así, el ajuste del plano por SVD no puede fijar el giro alrededor de esa recta
# y la normal queda indeterminada, lo que explicaría por qué el plano con malla rinde mucho peor que
# la triangulación sin malla con S_dat (Sección res-malla del capítulo 4).
#
# Para cada objeto y número de vistas lee los puntos 3D que impactaron la malla (mapped_points_3d),
# los centra y calcula sus valores singulares s1 >= s2 >= s3. La razón s2/s1 mide qué tan alineados
# están: cerca de 0, casi sobre una recta; cerca de 1, repartidos en el plano. Luego cruza esa razón
# con el error angular del plano ajustado con malla (SVD, 1 rayo) y sin malla (S_dat), por terciles.
# Solo lee archivos existentes; no escribe nada salvo el CSV de salida.
#
#   python3 PlaneScoreData/diagnostico_colinealidad_malla.py
#   python3 PlaneScoreData/diagnostico_colinealidad_malla.py --tipo axis_sym   # control: el eje
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "PlaneScoreData"))
from compare_plane_scores import errores_por_objeto  # noqa: E402

VISTAS = (6, 14, 26)
CONFIG = {
    "plane_sym": ("plane_v02_s224_flat_ndcfix_mesh", "plane_v02_s224_flat_ndcfix_sdat_mp1_nomesh"),
    "axis_sym": ("axis_v04_s224_flat_ndcfix_mesh", "axis_v04_s224_flat_ndcfix_expE_nomesh"),
}


def razon_singular(puntos: list) -> float | None:
    """ s2/s1 de los puntos 3D centrados; None si hay menos de 3 puntos. """
    validos = np.array([p for p in puntos if p is not None], dtype=float)
    if len(validos) < 3:
        return None
    s = np.linalg.svd(validos - validos.mean(axis=0), compute_uv=False)
    return float(s[1] / s[0]) if s[0] > 0 else None


def main() -> None:
    """ Calcula la razón s2/s1 por objeto y la cruza con el error angular por terciles. """
    ap = argparse.ArgumentParser(description="Colinealidad de los puntos 3D retroproyectados.")
    ap.add_argument("--renders-root", type=Path, default=Path("/data/pespinoz/renders"))
    ap.add_argument("--tipo", choices=list(CONFIG), default="plane_sym")
    ap.add_argument("--salida", type=Path, default=None, help="CSV por objeto (opcional)")
    args = ap.parse_args()
    alias, sin_malla = CONFIG[args.tipo]
    raiz = args.renders_root / args.tipo
    eval_malla = raiz / f"eval_s224_flat_{alias}_svd_results.json"
    eval_sin = raiz / f"eval_s224_flat_{sin_malla}_triangulation_results.json"
    for ruta in (eval_malla, eval_sin):
        if not ruta.exists():
            sys.exit(f"[error] falta {ruta}")

    filas = []
    for v in VISTAS:
        err_malla = errores_por_objeto(eval_malla, str(v))
        err_sin = errores_por_objeto(eval_sin, str(v))
        for archivo in sorted(raiz.glob(f"*/224/flat/mapped_points_3d_{alias}.json")):
            objeto = archivo.parents[2].name
            datos = json.loads(archivo.read_text(encoding="utf-8"))
            grupo = datos.get("n_views_results", {}).get(str(v)) or {}
            razon = razon_singular([p["point_3d"] for p in grupo.get("points_3d", []) if p.get("hit")])
            filas.append({"object_id": objeto, "n_views": v, "razon_s2_s1": razon,
                          "error_malla": err_malla.get(objeto), "error_sin_malla": err_sin.get(objeto)})

    print(f"{args.tipo}: {len(filas)} pares objeto-vistas")
    for v in VISTAS:
        datos = [f for f in filas if f["n_views"] == v and f["razon_s2_s1"] is not None
                 and f["error_malla"] is not None and f["error_sin_malla"] is not None]
        if not datos:
            continue
        razones = np.array([f["razon_s2_s1"] for f in datos])
        print(f"\n{v} vistas: {len(datos)} objetos con >= 3 puntos; s2/s1 mediana {np.median(razones):.3f}, "
              f"p10 {np.percentile(razones, 10):.3f}, p90 {np.percentile(razones, 90):.3f}")
        cortes = np.percentile(razones, [100 / 3, 200 / 3])
        print(f"  {'tercil s2/s1':22} {'n':>4} {'err. malla med.':>16} {'err. sin malla med.':>20}")
        for k, (lo, hi) in enumerate(((-np.inf, cortes[0]), (cortes[0], cortes[1]), (cortes[1], np.inf))):
            sub = [f for f in datos if lo < f["razon_s2_s1"] <= hi]
            print(f"  {k + 1} ({max(lo, 0):.3f}-{min(hi, 1):.3f}){'':6} {len(sub):>4} "
                  f"{np.median([f['error_malla'] for f in sub]):16.1f} "
                  f"{np.median([f['error_sin_malla'] for f in sub]):20.1f}")

    if args.salida:
        with open(args.salida, "w", newline="", encoding="utf-8") as f:
            escritor = csv.DictWriter(f, fieldnames=list(filas[0]))
            escritor.writeheader()
            escritor.writerows(filas)
        print(f"\nCSV por objeto: {args.salida}")


if __name__ == "__main__":
    main()
