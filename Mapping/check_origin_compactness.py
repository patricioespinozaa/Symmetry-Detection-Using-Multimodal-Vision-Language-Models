"""
Segundo chequeo para la discrepancia SDE vs. error angular (planar):
¿el SDE bajo de delegated pointing en objetos discordantes se explica porque
el plano predicho pasa cerca del centroide de un objeto compacto/convexo
(en vez de acertar la orientación real)?

Requiere:
  - El predicted_symmetry*.json de cada objeto (para origin/normal predichos).
  - La malla .obj de cada objeto (para centroide, bbox y una medida de
    compacidad = sphericity de la envolvente convexa).

Uso típico, encadenado con check_sde_vs_angular.py:

    python Mapping/check_sde_vs_angular.py \
        --eval-json /path/eval_delegated_results.json \
        --n-views 1 --label delegated \
        --out-csv scratch/discordant_delegated.csv

    python Mapping/check_origin_compactness.py \
        --discordant-csv scratch/discordant_delegated.csv \
        --renders-root /path/to/renders_root \
        --objects-root /path/to/objects_root \
        --symmetry-type plane_sym \
        --experiment-id plane_v04_1_flowC \
        --method ransac_svd_sde \
        --n-views 1 \
        --out-csv scratch/discordant_delegated_compactness.csv

Compara los objetos discordantes (ang alto, sde bajo) contra una muestra
aleatoria de objetos NO discordantes de la misma corrida, para ver si
compacidad / distancia-al-centroide difieren sistemáticamente.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

# Reusa las utilidades ya existentes del pipeline en vez de reimplementarlas.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_common.datasets import OBJECTS_SUBDIR, load_mesh_vertices  # noqa: E402
from pipeline_common.naming import exp_filename  # noqa: E402

PREDICTED_FILE = "predicted_symmetry.json"


def sphericity(vertices: np.ndarray) -> float:
    """
    Proxy de compacidad usando la envolvente convexa:
    sphericity = (36*pi*V^2)^(1/3) / A, en [0,1]; 1 = esfera perfecta.
    Requiere scipy.spatial.ConvexHull (parte de scipy, no de scikit-learn).
    """
    from scipy.spatial import ConvexHull  # import local: evitar costo si no se usa
    hull = ConvexHull(vertices)
    volume, area = hull.volume, hull.area
    if area <= 0:
        return float("nan")
    return float((36.0 * np.pi * volume ** 2) ** (1.0 / 3.0) / area)


def load_prediction(object_dir: Path, predicted_file: str,
                     n_views_key_variants: list[str], method: str) -> dict | None:
    path = object_dir / predicted_file
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    per_nviews = data.get("n_views_predictions", {})
    for key in n_views_key_variants:
        if key in per_nviews and method in per_nviews[key]:
            return per_nviews[key][method]
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--discordant-csv", required=True,
                   help="CSV de salida de check_sde_vs_angular.py.")
    p.add_argument("--renders-root", required=True)
    p.add_argument("--objects-root", required=True)
    p.add_argument("--symmetry-type", default="plane_sym", choices=["plane_sym"])
    p.add_argument("--experiment-id", default=None)
    p.add_argument("--method", required=True)
    p.add_argument("--n-views", default="1")
    p.add_argument("--n-control", type=int, default=None,
                   help="Tamaño de la muestra control no-discordante (default: "
                        "mismo tamaño que el grupo discordante).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-csv", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    symmetry_dir = Path(args.renders_root) / args.symmetry_type
    objects_dir  = Path(args.objects_root) / OBJECTS_SUBDIR[args.symmetry_type]
    predicted_file = exp_filename(PREDICTED_FILE, args.experiment_id)
    n_views_key_variants = [args.n_views, f"n_views_{args.n_views}"]

    with open(args.discordant_csv, encoding="utf-8") as f:
        discordant_ids = {row["object_id"] for row in csv.DictReader(f)}

    all_object_dirs = sorted(d.name for d in symmetry_dir.iterdir() if d.is_dir())
    control_pool = [oid for oid in all_object_dirs if oid not in discordant_ids]
    n_control = args.n_control or len(discordant_ids)
    control_ids = set(random.sample(control_pool, min(n_control, len(control_pool))))

    rows = []
    for obj_id in sorted(discordant_ids | control_ids):
        group = "discordant" if obj_id in discordant_ids else "control"

        obj_dir = symmetry_dir / obj_id
        pred = load_prediction(obj_dir, predicted_file, n_views_key_variants, args.method)
        mesh_path = objects_dir / f"{obj_id}.obj"
        if pred is None or not mesh_path.exists():
            continue

        vertices = load_mesh_vertices(mesh_path)
        centroid = vertices.mean(axis=0)
        origin = np.array(pred["origin"])
        bbox_diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))

        dist_origin_centroid = float(np.linalg.norm(origin - centroid)) / bbox_diag
        try:
            sph = sphericity(vertices)
        except Exception:
            sph = float("nan")

        rows.append({
            "object_id": obj_id,
            "group": group,
            "dist_origin_centroid_norm": round(dist_origin_centroid, 4),
            "sphericity": round(sph, 4) if sph == sph else "",  # NaN check
            "n_points": pred.get("n_points"),
        })

    if not rows:
        print("No se pudo cargar ningún objeto (revisa rutas / experiment-id / method).")
        return

    def summarize(group: str, field: str) -> tuple[float, float]:
        vals = [r[field] for r in rows if r["group"] == group and r[field] != ""]
        arr = np.array(vals, dtype=float)
        return (float(arr.mean()), float(np.median(arr))) if len(arr) else (float("nan"),) * 2

    print(f"\n{'':12}{'n':>5}  {'dist_origin_centroid (mean/median)':>36}  "
          f"{'sphericity (mean/median)':>28}")
    for group in ("discordant", "control"):
        n = sum(1 for r in rows if r["group"] == group)
        d_mean, d_med = summarize(group, "dist_origin_centroid_norm")
        s_mean, s_med = summarize(group, "sphericity")
        print(f"{group:12}{n:>5}  {d_mean:>16.4f} / {d_med:<16.4f}  "
              f"{s_mean:>12.4f} / {s_med:<12.4f}")

    print("\nInterpretación:")
    print("- dist_origin_centroid_norm más BAJO en 'discordant' que en 'control' sugiere")
    print("  que el SDE bajo viene de planos que pasan cerca del centroide (genéricos),")
    print("  no de acertar la orientación real.")
    print("- sphericity más ALTA en 'discordant' sugiere que el efecto se concentra en")
    print("  objetos compactos/convexos, donde casi cualquier plano central da SDE bajo.")

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
