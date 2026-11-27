#!/usr/bin/env python3
# Diagnóstico de la retroproyección sobre la malla en plano: ¿los puntos 3D quedan lejos del plano
# de simetría cuando la cámara mira el objeto de lado? Hipótesis: el rayo de un punto de la línea
# central impacta la primera superficie visible, que solo está sobre el plano de simetría cuando la
# cámara está cerca de ese plano. En una vista lateral (plano visto de frente) la línea central de
# la imagen corresponde a superficies que pueden estar lejos del plano, y el ajuste con malla falla.
#
# Para cada punto 3D que impactó la malla (mapped_points_3d, un rayo por punto) calcula:
#   - la distancia al plano de referencia más cercano, dividida por la diagonal de la caja de la malla;
#   - el ángulo theta entre la dirección de la cámara y ese plano: 0 = cámara dentro del plano
#     (ve la línea central de frente), 90 = plano visto de frente (vista lateral).
# Resume la distancia por rangos de theta y, por objeto, cruza la distancia mediana de sus puntos
# con el error angular del plano ajustado con malla (SVD, 1 rayo) por terciles.
# Solo lee archivos existentes; no escribe nada salvo el CSV de salida.
#
#   python3 PlaneScoreData/diagnostico_profundidad_malla.py --salida resultados_finales/diagnosticos/profundidad_plano.csv
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "PlaneScoreData"))
sys.path.insert(0, str(REPO / "Mapping"))
sys.path.insert(0, str(REPO))
from compare_plane_scores import errores_por_objeto  # noqa: E402
from evaluate import bbox_diagonal, parse_true_label  # noqa: E402
from pipeline_common.datasets import OBJECTS_SUBDIR, load_mesh_vertices  # noqa: E402

VISTAS = (6, 14, 26)
ALIAS = "plane_v02_s224_flat_ndcfix_mesh"
RANGOS = (0, 15, 30, 45, 60, 75, 90.01)
CERCA = 0.02        # fracción de la diagonal bajo la cual un punto cuenta como "sobre el plano"


def medir_punto(p: np.ndarray, eye: np.ndarray, planos: list[dict], diag: float) -> tuple[float, float]:
    """ Distancia normalizada al plano de referencia más cercano y ángulo de la cámara a ese plano.

    Args:
        * p: punto 3D sobre la malla
        * eye: posición de la cámara (mira al origen)
        * planos: planos de referencia (normal unitaria y origen)
        * diag: diagonal de la caja de la malla

    Returns:
        * tuple[float, float]: (distancia / diag, theta en grados)

    """
    dist = [abs(float(np.dot(pl["normal"], p - np.asarray(pl["origin"])))) for pl in planos]
    k = int(np.argmin(dist))
    e = eye / np.linalg.norm(eye)
    theta = float(np.degrees(np.arcsin(min(1.0, abs(float(np.dot(planos[k]["normal"], e)))))))
    return dist[k] / diag, theta


def main() -> None:
    """ Mide la distancia de los puntos con malla al plano de referencia, por ángulo de cámara. """
    ap = argparse.ArgumentParser(description="Distancia de los puntos retroproyectados al plano de referencia.")
    ap.add_argument("--renders-root", type=Path, default=Path("/data/pespinoz/renders"))
    ap.add_argument("--objects-root", type=Path, default=Path("/data/pespinoz/objects"))
    ap.add_argument("--salida", type=Path, default=None, help="CSV por punto (opcional)")
    args = ap.parse_args()
    raiz = args.renders_root / "plane_sym"
    objetos = args.objects_root / OBJECTS_SUBDIR["plane_sym"]
    eval_malla = raiz / f"eval_s224_flat_{ALIAS}_svd_results.json"
    if not eval_malla.exists():
        sys.exit(f"[error] falta {eval_malla}")

    filas, sin_referencia = [], 0
    archivos = sorted(raiz.glob(f"*/224/flat/mapped_points_3d_{ALIAS}.json"))
    for archivo in archivos:
        objeto = archivo.parents[2].name
        txt, obj = objetos / f"{objeto}.txt", objetos / f"{objeto}.obj"
        if not (txt.exists() and obj.exists()):
            sin_referencia += 1
            continue
        planos = [{"normal": np.asarray(el["normal"]), "origin": el["origin"]}
                  for el in parse_true_label(txt)["elements"]]
        vertices = load_mesh_vertices(obj)
        if not planos or vertices is None or bbox_diagonal(vertices) <= 0:
            sin_referencia += 1
            continue
        diag = bbox_diagonal(vertices)
        datos = json.loads(archivo.read_text(encoding="utf-8")).get("n_views_results", {})
        for v in VISTAS:
            grupo = datos.get(str(v)) or {}
            eyes = {int(c["img_idx"]): np.asarray(c["eye"], dtype=float) for c in grupo.get("images_sent", [])}
            for p in grupo.get("points_3d", []):
                if not p.get("hit") or int(p["img_idx"]) not in eyes:
                    continue
                d, theta = medir_punto(np.asarray(p["point_3d"], dtype=float), eyes[int(p["img_idx"])], planos, diag)
                filas.append({"object_id": objeto, "n_views": v, "img_idx": p["img_idx"],
                              "dist_norm": round(d, 6), "theta_deg": round(theta, 2)})

    print(f"plane_sym: {len(archivos)} objetos con puntos con malla, {sin_referencia} sin referencia o malla, "
          f"{len(filas)} puntos que impactaron la malla")
    for v in VISTAS:
        sub = [f for f in filas if f["n_views"] == v]
        if not sub:
            continue
        print(f"\n{v} vistas: {len(sub)} puntos")
        print(f"  {'theta (°)':12} {'n':>6} {'dist. mediana':>14} {f'frac < {CERCA}':>12}")
        for lo, hi in zip(RANGOS, RANGOS[1:]):
            r = [f["dist_norm"] for f in sub if lo <= f["theta_deg"] < hi]
            if r:
                print(f"  {f'{lo}-{min(hi, 90):.0f}':12} {len(r):>6} {np.median(r):14.4f} "
                      f"{np.mean(np.array(r) < CERCA):12.3f}")

        # Por objeto: distancia mediana de sus puntos frente al error del plano con malla
        errores = errores_por_objeto(eval_malla, str(v))
        por_objeto = {}
        for f in sub:
            por_objeto.setdefault(f["object_id"], []).append(f["dist_norm"])
        pares = [(float(np.median(d)), errores[o]) for o, d in por_objeto.items() if o in errores]
        if len(pares) >= 3:
            dist = np.array([d for d, _ in pares])
            cortes = np.percentile(dist, [100 / 3, 200 / 3])
            print(f"  {'tercil dist. mediana':26} {'n':>4} {'err. malla med.':>16}")
            for k, (lo, hi) in enumerate(((-np.inf, cortes[0]), (cortes[0], cortes[1]), (cortes[1], np.inf))):
                e = [err for d, err in pares if lo < d <= hi]
                print(f"  {k + 1} ({max(lo, 0):.4f}-{hi if np.isfinite(hi) else dist.max():.4f}){'':4} "
                      f"{len(e):>4} {np.median(e):16.1f}")

    if args.salida and filas:
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        with open(args.salida, "w", newline="", encoding="utf-8") as f:
            escritor = csv.DictWriter(f, fieldnames=list(filas[0]))
            escritor.writeheader()
            escritor.writerows(filas)
        print(f"\nCSV por punto: {args.salida}")


if __name__ == "__main__":
    main()
