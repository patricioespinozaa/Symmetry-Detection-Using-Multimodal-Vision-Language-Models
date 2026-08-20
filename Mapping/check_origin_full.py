"""
Versión "población completa" de check_origin_compactness.py: en vez de
comparar solo la cola discordante contra una muestra control, calcula
dist_origin_centroid_norm y sphericity para TODOS los objetos válidos de
cada estrategia (p.ej. los 641 de direct y los 712 de delegated) y compara
las distribuciones completas, además de su correlación con angular_error_deg
y sde dentro de cada grupo.

Uso (mismo patrón multi-archivo que check_sde_vs_angular.py):

    python Mapping/check_origin_full.py \
        --renders-root ../data/renders --objects-root ../data/objects \
        --symmetry-type plane_sym \
        --eval-json ../data/renders/plane_sym/eval_s224_flat_plane_v04_1_ransac_svd_sde_results.json \
                    ../data/renders/plane_sym/eval_s224_flat_plane_v04_1_flowC_ransac_svd_sde_results.json \
        --experiment-ids plane_v04_1 plane_v04_1_flowC \
        --labels direct delegated \
        --method ransac_svd_sde \
        --n-views 1 \
        --out-csv scratch/origin_full_direct_vs_delegated.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_common.datasets import OBJECTS_SUBDIR, load_mesh_vertices  # noqa: E402
from pipeline_common.naming import exp_filename  # noqa: E402

PREDICTED_FILE = "predicted_symmetry.json"


def sphericity(vertices: np.ndarray) -> float:
    from scipy.spatial import ConvexHull
    hull = ConvexHull(vertices)
    volume, area = hull.volume, hull.area
    if area <= 0:
        return float("nan")
    return float((36.0 * np.pi * volume ** 2) ** (1.0 / 3.0) / area)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def load_eval_pairs(eval_json_path: Path, n_views_key: str) -> dict[str, dict]:
    with open(eval_json_path, encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for obj_id, per_nviews in (data.get("objects") or {}).items():
        if not per_nviews:
            continue
        m = per_nviews.get(n_views_key)
        if not m or m.get("status") != "ok":
            continue
        out[obj_id] = {"angular_error_deg": m.get("angular_error_deg"), "sde": m.get("sde")}
    return out


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
    p.add_argument("--renders-root", required=True)
    p.add_argument("--objects-root", required=True)
    p.add_argument("--symmetry-type", default="plane_sym", choices=["plane_sym"])
    p.add_argument("--eval-json", nargs="+", required=True)
    p.add_argument("--experiment-ids", nargs="+", required=True,
                   help="Un experiment-id por --eval-json, mismo orden.")
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--n-views", default="1")
    p.add_argument("--out-csv", default=None,
                   help="CSV con una fila por objeto por grupo (todas las estrategias).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not (len(args.eval_json) == len(args.experiment_ids) == len(args.labels)):
        raise SystemExit("--eval-json, --experiment-ids y --labels deben tener el mismo largo")

    symmetry_dir = Path(args.renders_root) / args.symmetry_type
    objects_dir  = Path(args.objects_root) / OBJECTS_SUBDIR[args.symmetry_type]
    n_views_key_variants = [args.n_views, f"n_views_{args.n_views}"]

    all_rows = []

    for eval_path, exp_id, label in zip(args.eval_json, args.experiment_ids, args.labels):
        predicted_file = exp_filename(PREDICTED_FILE, exp_id)
        pairs = load_eval_pairs(Path(eval_path), args.n_views)
        if not pairs:
            pairs = load_eval_pairs(Path(eval_path), f"n_views_{args.n_views}")

        rows = []
        for obj_id, m in pairs.items():
            obj_dir = symmetry_dir / obj_id
            pred = load_prediction(obj_dir, predicted_file, n_views_key_variants, args.method)
            mesh_path = objects_dir / f"{obj_id}.obj"
            if pred is None or not mesh_path.exists() or m["angular_error_deg"] is None:
                continue
            try:
                vertices = load_mesh_vertices(mesh_path)
                centroid = vertices.mean(axis=0)
                origin = np.array(pred["origin"])
                bbox_diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
                if bbox_diag <= 0:
                    continue
                dist = float(np.linalg.norm(origin - centroid)) / bbox_diag
                sph = sphericity(vertices)
            except Exception:
                continue

            row = {
                "label": label,
                "object_id": obj_id,
                "angular_error_deg": m["angular_error_deg"],
                "sde": m["sde"],
                "dist_origin_centroid_norm": dist,
                "sphericity": sph,
            }
            rows.append(row)
            all_rows.append(row)

        ang = np.array([r["angular_error_deg"] for r in rows])
        sde = np.array([r["sde"] for r in rows if r["sde"] is not None])
        dist_arr = np.array([r["dist_origin_centroid_norm"] for r in rows])
        sph_arr = np.array([r["sphericity"] for r in rows if r["sphericity"] == r["sphericity"]])

        print(f"\n=== {label}  (n={len(rows)} objetos, n_views={args.n_views}) ===")
        print(f"  angular_error_deg : mean={ang.mean():.2f}  median={np.median(ang):.2f}")
        if len(sde):
            print(f"  sde               : mean={sde.mean():.4f}  median={np.median(sde):.4f}")
        print(f"  dist_origin_centroid_norm : mean={dist_arr.mean():.4f}  median={np.median(dist_arr):.4f}")
        if len(sph_arr):
            print(f"  sphericity                : mean={sph_arr.mean():.4f}  median={np.median(sph_arr):.4f}")
        print(f"  Pearson(dist_origin_centroid, angular_error) = {pearson(dist_arr, ang):+.3f}")
        if len(sde) == len(dist_arr):
            print(f"  Pearson(dist_origin_centroid, sde)           = {pearson(dist_arr, sde):+.3f}")

    if args.out_csv and all_rows:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nGuardado: {out_path}  ({len(all_rows)} filas)")

    print("\nInterpretación:")
    print("- Si 'dist_origin_centroid_norm' es sistemáticamente más BAJO en delegated que en")
    print("  direct (comparando medias/medianas arriba), el SDE agregado más bajo de delegated")
    print("  se explica por planos que caen más cerca del centroide en general, no solo en")
    print("  una cola de casos degenerados.")
    print("- Pearson(dist_origin_centroid, sde) alto y positivo dentro de cada grupo confirma")
    print("  que sde y la distancia al centroide están relacionados mecánicamente, como se")
    print("  espera de la fórmula de SDE del pipeline.")


if __name__ == "__main__":
    main()
