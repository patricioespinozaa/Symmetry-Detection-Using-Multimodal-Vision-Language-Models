"""
export_viz_samples.py
---------------------
Selects the N objects with the best and worst symmetry predictions for a given
experiment, copies their pipeline JSONs to results/, and generates a README
with the exact visualize_rays.py commands for each object.

"Good" = N objects with lowest angular error (status=="ok")
"Bad"  = N objects with highest angular error (status=="ok")
"No prediction" = objects with no valid prediction for this n_views (listed separately)

Output structure
----------------
<results_dir>/<experiment_id>/
├── viz_samples/
│   ├── good/
│   │   └── <object_id>/
│   │       ├── predicted_symmetry[_EXP].json
│   │       └── <size>/<lighting>/
│   │           ├── molmo_multiview[_EXP].json
│   │           └── mapped_points_3d[_EXP].json
│   └── bad/
│       └── <object_id>/ (same structure)
├── objects_good.json     ← IDs + angular_error + n_points + sde
├── objects_bad.json
├── objects_no_pred.json  ← objects with no valid prediction
└── README.md             ← ready-to-run visualize_rays.py commands

Usage
-----
python Mapping/export_viz_samples.py \\
    --renders-root ../data/renders \\
    --objects-root ../data/objects \\
    --symmetry-type axis_sym \\
    --experiment-id axis_v00 \\
    --method svd \\
    --results-dir ../results \\
    --sizes 224 --lightings flat \\
    --n-samples 10

python Mapping/export_viz_samples.py \\
    --renders-root ../data/renders \\
    --objects-root ../data/objects \\
    --symmetry-type plane_sym \\
    --experiment-id plane_v00 \\
    --method svd \\
    --results-dir ../results
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import date
from pathlib import Path

OBJECTS_SUBDIR = {
    "axis_sym":  "curated_axis_sym_obj",
    "plane_sym": "curated_plane_sym_obj",
}

METHODS = ["svd", "ransac_svd", "svd_sde", "ransac_svd_sde"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _exp_filename(base: str, experiment_id: str | None) -> str:
    if not experiment_id:
        return base
    dot = base.rfind(".")
    return f"{base[:dot]}_{experiment_id}{base[dot:]}"


def _results_json_path(renders_root: Path, symmetry_type: str,
                       sizes: list[int], lightings: list[str],
                       experiment_id: str | None, method: str) -> Path:
    size_tag  = "s" + "_".join(str(s) for s in sizes)
    light_tag = "_".join(lightings)
    exp_part  = f"_{experiment_id}" if experiment_id else ""
    return (renders_root / symmetry_type
            / f"eval_{size_tag}_{light_tag}{exp_part}_{method}_results.json")


def _select_samples(objects: dict, nv_key: str, n: int
                    ) -> tuple[list[dict], list[dict], list[dict]]:
    """Returns (good, bad, no_pred). good/bad sorted by angular_error_deg."""
    ok, no_pred = [], []
    for obj_id, obj_data in objects.items():
        if obj_data is None or nv_key not in obj_data:
            no_pred.append({"object_id": obj_id, "reason": "no_data"})
            continue
        m = obj_data[nv_key]
        if m.get("status") != "ok":
            no_pred.append({
                "object_id": obj_id,
                "reason":    m.get("status", "unknown"),
                "n_points":  m.get("n_points", 0),
            })
            continue
        ok.append({
            "object_id":         obj_id,
            "angular_error_deg": m.get("angular_error_deg"),
            "translation_error": m.get("translation_error"),
            "n_points":          m.get("n_points"),
            "n_inliers":         m.get("n_inliers"),
            "sde":               m.get("sde"),
            "accepted":          m.get("accepted"),
        })
    ok.sort(key=lambda x: x["angular_error_deg"])
    return ok[:n], ok[-n:][::-1], no_pred


def _copy_object_jsons(
    object_id:     str,
    renders_root:  Path,
    symmetry_type: str,
    sizes:         list[int],
    lightings:     list[str],
    experiment_id: str | None,
    dest_dir:      Path,
) -> None:
    """Copy pipeline JSONs for one object to dest_dir, preserving subdirectory structure."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    obj_dir = renders_root / symmetry_type / object_id

    # Object-level JSON: predicted_symmetry[_EXP].json
    for base in ["predicted_symmetry.json"]:
        src = obj_dir / _exp_filename(base, experiment_id)
        if src.exists():
            shutil.copy2(src, dest_dir / src.name)

    # Size/lighting-level JSONs
    for size in sizes:
        for lighting in lightings:
            size_light_src = obj_dir / str(size) / lighting
            size_light_dst = dest_dir / str(size) / lighting
            size_light_dst.mkdir(parents=True, exist_ok=True)
            for base in ["molmo_multiview.json", "mapped_points_3d.json"]:
                src = size_light_src / _exp_filename(base, experiment_id)
                if src.exists():
                    shutil.copy2(src, size_light_dst / src.name)


def _generate_readme(
    out_path:       Path,
    good_objects:   list[dict],
    bad_objects:    list[dict],
    no_pred_count:  int,
    nv_key:         str,
    args:           argparse.Namespace,
) -> None:
    exp_id   = args.experiment_id or "(none)"
    method   = args.method
    sym_type = args.symmetry_type
    size     = args.sizes[0]
    lighting = args.lightings[0]

    def _cmd(obj_id: str, extra_flags: str = "") -> list[str]:
        lines = [
            "```bash",
            f"python Mapping/visualize_rays.py \\",
            f"    --object-id {obj_id} \\",
            f"    --renders-root {args.renders_root} \\",
            f"    --objects-root {args.objects_root} \\",
            f"    --symmetry-type {sym_type} \\",
            f"    --size {size} --lighting {lighting} \\",
            f"    --n-views {nv_key}",
        ]
        if args.experiment_id:
            lines[-1] += " \\"
            lines.append(f"    --experiment-id {args.experiment_id}")
        if extra_flags:
            lines[-1] += " \\"
            lines.append(f"    {extra_flags}")
        lines.append("```")
        return lines

    all_flags = f"--show-gt --show-predicted --pred-method {method} --show-clusters"

    blocks = [
        f"# Viz samples — {exp_id}",
        f"",
        f"| Campo             | Valor |",
        f"|---|---|",
        f"| Experimento       | `{exp_id}` |",
        f"| Tipo de simetría  | `{sym_type}` |",
        f"| Método de ranking | `{method}` |",
        f"| n\\_views           | {nv_key} |",
        f"| Objetos buenos    | {len(good_objects)} |",
        f"| Objetos malos     | {len(bad_objects)} |",
        f"| Sin predicción    | {no_pred_count} |",
        f"| Fecha             | {date.today()} |",
        f"",
        f"## Flags de visualización disponibles",
        f"",
        f"| Flag | Efecto |",
        f"|---|---|",
        f"| `--show-gt` | Muestra eje/plano GT (azul/verde) |",
        f"| `--show-predicted` | Muestra eje/plano predicho (rojo/magenta) |",
        f"| `--pred-method svd\\|ransac_svd\\|svd_sde\\|ransac_svd_sde` | Método a visualizar |",
        f"| `--show-clusters` | Muestra centroides de clusters (naranja-rojo) vs hits crudos (amarillo) |",
        f"| `--point-mode independent\\|midpoint` | Modo de colección de puntos (debe coincidir con estimate_symmetry.py) |",
        f"",
        f"---",
        f"",
        f"## Objetos con buena predicción (menor error angular)",
        f"",
    ]

    for obj in good_objects:
        oid = obj["object_id"]
        err = obj["angular_error_deg"]
        npt = obj["n_points"]
        sde = obj.get("sde")
        sde_s = f"  SDE={sde:.4f}" if sde is not None else ""
        blocks += [
            f"### `{oid}`",
            f"",
            f"> angular\\_error = **{err:.2f}°**  |  n\\_points = {npt}{sde_s}",
            f"",
        ] + _cmd(oid, all_flags) + [""]

    blocks += [
        f"---",
        f"",
        f"## Objetos con mala predicción (mayor error angular)",
        f"",
    ]

    for obj in bad_objects:
        oid = obj["object_id"]
        err = obj["angular_error_deg"]
        npt = obj["n_points"]
        sde = obj.get("sde")
        sde_s = f"  SDE={sde:.4f}" if sde is not None else ""
        blocks += [
            f"### `{oid}`",
            f"",
            f"> angular\\_error = **{err:.2f}°**  |  n\\_points = {npt}{sde_s}",
            f"",
        ] + _cmd(oid, all_flags) + [""]

    out_path.write_text("\n".join(blocks), encoding="utf-8")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export visualization samples (good/bad predictions) for an experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--renders-root",  required=True)
    p.add_argument("--objects-root",  required=True)
    p.add_argument("--symmetry-type", required=True, choices=["axis_sym", "plane_sym"])
    p.add_argument("--results-dir",   required=True,
                   help="Root results folder. Output goes to <results_dir>/<experiment_id>/")
    p.add_argument("--method",         default="svd", choices=METHODS,
                   help="Fitting method used to rank good/bad objects.")
    p.add_argument("--experiment-id",  default=None)
    p.add_argument("--sizes",     type=int, nargs="+", default=[224])
    p.add_argument("--lightings", type=str, nargs="+", default=["flat"],
                   choices=["flat", "darker", "brighter"])
    p.add_argument("--n-views",   type=int, default=None,
                   help="n_views group for ranking. Default: maximum available.")
    p.add_argument("--n-samples", type=int, default=10,
                   help="Number of good + bad objects to export.")
    return p.parse_args()


def main() -> None:
    args      = parse_args()
    root      = Path(args.renders_root)
    sym_type  = args.symmetry_type
    exp_id    = args.experiment_id
    folder    = exp_id if exp_id else f"{sym_type}_default"

    # ── Locate eval results JSON ───────────────────────────────────────────────
    results_json = _results_json_path(
        root, sym_type, args.sizes, args.lightings, exp_id, args.method
    )
    if not results_json.exists():
        raise SystemExit(f"[error] Results JSON not found: {results_json}\n"
                         f"        Run evaluate.py first.")

    with open(results_json, encoding="utf-8") as f:
        eval_data = json.load(f)

    objects = eval_data.get("objects", {})

    # ── Choose n_views ─────────────────────────────────────────────────────────
    all_nv: set[str] = set()
    for od in objects.values():
        if od:
            all_nv.update(od.keys())
    if not all_nv:
        raise SystemExit("[error] No n_views keys found in results JSON.")

    nv_key = str(args.n_views) if args.n_views else str(max(int(k) for k in all_nv))
    print(f"Using n_views = {nv_key}  (pass --n-views to override)")

    # ── Select samples ─────────────────────────────────────────────────────────
    good, bad, no_pred = _select_samples(objects, nv_key, args.n_samples)
    print(f"Good objects : {len(good)}")
    print(f"Bad objects  : {len(bad)}")
    print(f"No prediction: {len(no_pred)}")

    # ── Output directory ───────────────────────────────────────────────────────
    out_root = Path(args.results_dir) / folder
    out_root.mkdir(parents=True, exist_ok=True)

    # ── Copy JSONs ─────────────────────────────────────────────────────────────
    for category, sample_list in [("good", good), ("bad", bad)]:
        for obj in sample_list:
            oid      = obj["object_id"]
            dest_dir = out_root / "viz_samples" / category / oid
            _copy_object_jsons(
                oid, root, sym_type,
                args.sizes, args.lightings, exp_id, dest_dir,
            )
    print(f"JSONs copied → {out_root / 'viz_samples'}")

    # ── Save object lists ──────────────────────────────────────────────────────
    (out_root / "objects_good.json").write_text(
        json.dumps(good,    indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_root / "objects_bad.json").write_text(
        json.dumps(bad,     indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_root / "objects_no_pred.json").write_text(
        json.dumps(no_pred, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ── Generate README ────────────────────────────────────────────────────────
    readme_path = out_root / "README.md"
    _generate_readme(readme_path, good, bad, len(no_pred), nv_key, args)
    print(f"README       → {readme_path}")
    print(f"\nDone. Results in: {out_root}")


if __name__ == "__main__":
    main()
