#!/usr/bin/env python3
"""
audit_view_indices.py
----------------------
Audita todos los molmo_multiview*.json bajo <renders_root>/<symmetry_type>/
para saber, por cada (symmetry_type, prompt_id, experiment_id, n_views),
cuantas corridas usaron el patron SECUENCIAL (bug pre-fix: indices
[0, 1, ..., n_views-1]) vs el patron CORRECTO (linspace espaciado, el que
produce MolmoPointing/molmo_multiview_runner.py::get_n_views_entries desde el
commit a6ffeac, 2026-06-16).

Uso:
    python Mapping/audit_view_indices.py \\
        --renders-root ../data/renders \\
        --symmetry-type axis_sym plane_sym \\
        --out-summary audit_view_indices_summary.csv \\
        --out-raw audit_view_indices_raw.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def expected_indices_for(n_views: int, total: int) -> list[int]:
    if n_views >= total:
        return list(range(total))
    return sorted({int(round(i)) for i in np.linspace(0, total - 1, n_views)})


def classify(indices: list[int], n_views: int, total: int) -> str:
    if n_views >= total:
        return "n/a (n_views>=total)"
    if indices == expected_indices_for(n_views, total):
        return "correcto (espaciado)"
    if indices == list(range(n_views)):
        return "secuencial (bug)"
    return "otro/inesperado"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--renders-root", required=True, type=Path)
    p.add_argument("--symmetry-type", nargs="+", default=["axis_sym", "plane_sym"])
    p.add_argument("--default-total", type=int, default=114,
                   help="Total de vistas Fibonacci a asumir si no hay metadata_all.json (default: 114).")
    p.add_argument("--out-summary", default="audit_view_indices_summary.csv")
    p.add_argument("--out-raw", default="audit_view_indices_raw.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = []

    for symmetry_type in args.symmetry_type:
        sym_dir = args.renders_root / symmetry_type
        if not sym_dir.exists():
            print(f"[warn] no existe: {sym_dir}")
            continue

        for obj_dir in sorted(d for d in sym_dir.iterdir() if d.is_dir()):
            object_id = obj_dir.name
            for size_dir in sorted(d for d in obj_dir.iterdir() if d.is_dir()):
                for lighting_dir in sorted(d for d in size_dir.iterdir() if d.is_dir()):
                    total, total_source = args.default_total, "fallback_default"
                    meta_path = lighting_dir / "metadata_all.json"
                    if meta_path.exists():
                        try:
                            total = len(json.load(open(meta_path, encoding="utf-8")))
                            total_source = "metadata_all.json"
                        except Exception:
                            pass

                    for json_path in sorted(lighting_dir.glob("molmo_multiview*.json")):
                        try:
                            data = json.load(open(json_path, encoding="utf-8"))
                        except Exception as e:
                            print(f"[warn] no se pudo leer {json_path}: {e}")
                            continue

                        for n_views_str, entry in data.items():
                            if not isinstance(entry, dict) or "images_sent" not in entry:
                                continue
                            n_views = int(n_views_str)
                            indices = sorted(im["index"] for im in entry["images_sent"])
                            classification = classify(indices, n_views, total)

                            rows.append({
                                "symmetry_type": symmetry_type,
                                "object_id": object_id,
                                "size": size_dir.name,
                                "lighting": lighting_dir.name,
                                "file": json_path.name,
                                "prompt_id": entry.get("prompt_id") or "(default/produccion)",
                                "experiment_id": entry.get("experiment_id") or "(produccion)",
                                "flow": entry.get("flow", "a"),
                                "n_views": n_views,
                                "n_sent": len(indices),
                                "total_vistas": total,
                                "total_fuente": total_source,
                                "clasificacion": classification,
                            })

    if not rows:
        print("No se encontro ningun molmo_multiview*.json -- revisa --renders-root.")
        return

    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(args.out_raw, index=False)
    print(f"Detalle por archivo guardado en: {args.out_raw}  ({len(raw_df)} filas)")

    summary = (
        raw_df
        .groupby(["symmetry_type", "prompt_id", "experiment_id", "n_views", "clasificacion"])
        .size()
        .unstack("clasificacion", fill_value=0)
        .reset_index()
    )
    for col in ["secuencial (bug)", "correcto (espaciado)", "otro/inesperado", "n/a (n_views>=total)"]:
        if col not in summary.columns:
            summary[col] = 0

    summary["n_total_checkeado"] = (
        summary["secuencial (bug)"] + summary["correcto (espaciado)"] + summary["otro/inesperado"]
    )
    summary["pct_correcto"] = np.where(
        summary["n_total_checkeado"] > 0,
        (summary["correcto (espaciado)"] / summary["n_total_checkeado"] * 100).round(1),
        np.nan,
    )

    cols = [
        "symmetry_type", "prompt_id", "experiment_id", "n_views",
        "secuencial (bug)", "correcto (espaciado)", "otro/inesperado",
        "n_total_checkeado", "pct_correcto",
    ]
    summary = summary[cols].sort_values(
        ["symmetry_type", "prompt_id", "experiment_id", "n_views"]
    )
    summary.to_csv(args.out_summary, index=False)

    print(f"\nResumen guardado en: {args.out_summary}\n")
    print(summary.to_string(index=False))

    n_fallback = (raw_df["total_fuente"] == "fallback_default").sum()
    if n_fallback:
        print(f"\n[atencion] {n_fallback} filas usaron el total por defecto "
              f"({args.default_total}) porque no encontraron metadata_all.json -- "
              f"revisar esas carpetas si el resultado ahi parece raro.")


if __name__ == "__main__":
    main()
