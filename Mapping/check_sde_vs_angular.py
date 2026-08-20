"""
Diagnóstico rápido: ¿por qué delegated pointing tiene el peor error angular
pero el mejor SDE (planar)?

No requiere volver a tocar mallas ni predicciones crudas: lee directamente
los `eval_*_results.json` ya producidos por evaluate.py (uno por estrategia),
extrae angular_error_deg y sde por objeto para un n_views dado, y reporta:

  1. Correlación (Pearson y Spearman-por-rangos) entre angular_error_deg y sde.
  2. Lista de objetos "discordantes": angular_error alto (>= --discord-angle,
     default 45°, el mismo corte de AUC_angular) pero sde bajo
     (< --discord-sde, default 0.05, el mismo SDE_THRESHOLD del pipeline).
  3. Un CSV con esos objetos discordantes para alimentar check_origin_compactness.py.

Uso:
    python Mapping/check_sde_vs_angular.py \
        --eval-json /path/eval_..._delegated_results.json \
        --n-views 1 \
        --label delegated \
        --out-csv scratch/discordant_delegated.csv

    # Comparar contra direct pointing en la misma corrida:
    python Mapping/check_sde_vs_angular.py \
        --eval-json /path/eval_direct_results.json /path/eval_delegated_results.json \
        --labels direct delegated \
        --n-views 1
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_via_ranks(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman = Pearson sobre rangos (evita depender de scipy)."""
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return pearson(rx.astype(float), ry.astype(float))


def load_pairs(eval_json_path: Path, n_views_key: str) -> dict[str, dict]:
    with open(eval_json_path, encoding="utf-8") as f:
        data = json.load(f)

    if data.get("symmetry_type") != "plane_sym":
        raise ValueError(
            f"{eval_json_path} es symmetry_type={data.get('symmetry_type')!r}; "
            "SDE solo existe para plane_sym en este pipeline."
        )

    out = {}
    for obj_id, per_nviews in (data.get("objects") or {}).items():
        if not per_nviews:
            continue
        m = per_nviews.get(n_views_key)
        if not m or m.get("status") != "ok":
            continue
        ang = m.get("angular_error_deg")
        sde = m.get("sde")
        if ang is None or sde is None:
            continue
        out[obj_id] = {
            "angular_error_deg": ang,
            "sde": sde,
            "translation_error": m.get("translation_error"),
            "n_points": m.get("n_points"),
        }
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-json", nargs="+", required=True,
                   help="Uno o más eval_*_results.json (uno por estrategia/config).")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Nombre corto por archivo (mismo orden que --eval-json). "
                        "Default: usa el nombre del archivo.")
    p.add_argument("--n-views", default="1",
                   help="Clave n_views a analizar tal como aparece en el JSON "
                        "(ej. '1', '6', '14', '26').")
    p.add_argument("--discord-angle", type=float, default=45.0,
                   help="Umbral de error angular 'alto' (grados). Default 45, "
                        "igual al corte de AUC_angular del pipeline.")
    p.add_argument("--discord-sde", type=float, default=0.05,
                   help="Umbral de SDE 'bajo' (bueno). Default 0.05, igual a "
                        "SDE_THRESHOLD del pipeline.")
    p.add_argument("--out-csv", default=None,
                   help="Si se indica, escribe ahí los objetos discordantes "
                        "(solo del último archivo de --eval-json).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    eval_paths = [Path(p) for p in args.eval_json]
    labels = args.labels or [p.stem for p in eval_paths]
    if len(labels) != len(eval_paths):
        raise SystemExit("--labels debe tener el mismo largo que --eval-json")

    n_views_key = args.n_views if args.n_views.startswith("n_views") else f"n_views_{args.n_views}"
    # evaluate.py guarda las claves tal como vienen de predicted_symmetry.json;
    # si tu formato real es distinto (p.ej. solo "1"), probamos ambas variantes.
    last_discordant: list[tuple[str, dict]] = []

    for label, eval_path in zip(labels, eval_paths):
        pairs = None
        for candidate_key in (n_views_key, args.n_views, f"{args.n_views}"):
            try:
                pairs = load_pairs(eval_path, candidate_key)
            except ValueError as e:
                print(f"[skip] {e}")
                pairs = None
                break
            if pairs:
                break
        if not pairs:
            print(f"[{label}] sin datos para n_views={args.n_views} en {eval_path.name}")
            continue

        ang = np.array([v["angular_error_deg"] for v in pairs.values()])
        sde = np.array([v["sde"] for v in pairs.values()])

        print(f"\n=== {label}  (n={len(pairs)} objetos, n_views={args.n_views}) ===")
        print(f"  angular_error_deg: mean={ang.mean():.2f}  median={np.median(ang):.2f}")
        print(f"  sde:               mean={sde.mean():.4f}  median={np.median(sde):.4f}")
        print(f"  Pearson(ang, sde)  = {pearson(ang, sde):+.3f}")
        print(f"  Spearman(ang, sde) = {spearman_via_ranks(ang, sde):+.3f}")
        print("  (si es ~0 o positivo débil, ang y sde miden cosas distintas: "
              "SDE bajo no implica orientación correcta)")

        discordant = {
            obj_id: v for obj_id, v in pairs.items()
            if v["angular_error_deg"] >= args.discord_angle and v["sde"] < args.discord_sde
        }
        print(f"  Objetos discordantes (ang>={args.discord_angle:.0f}°, "
              f"sde<{args.discord_sde}): {len(discordant)} / {len(pairs)} "
              f"({100*len(discordant)/len(pairs):.1f}%)")
        last_discordant = list(discordant.items())

    if args.out_csv and last_discordant:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["object_id", "angular_error_deg", "sde", "translation_error", "n_points"])
            for obj_id, v in last_discordant:
                w.writerow([obj_id, v["angular_error_deg"], v["sde"],
                            v["translation_error"], v["n_points"]])
        print(f"\nGuardado: {out_path}  ({len(last_discordant)} objetos discordantes)")


if __name__ == "__main__":
    main()
