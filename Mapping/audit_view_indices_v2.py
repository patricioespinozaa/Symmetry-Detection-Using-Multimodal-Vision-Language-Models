#!/usr/bin/env python3
"""
audit_view_indices_v2.py
--------------------------
Version mas completa de audit_view_indices.py: en vez de asumir la
estructura fija <renders_root>/<symmetry_type>/<object_id>/<size>/<lighting>/,
busca RECURSIVAMENTE cualquier molmo_multiview*.json (con o sin sufijo de
prompt/experimento) bajo --search-root, sin importar la carpeta en la que
este -- para no dejar afuera predicciones sueltas guardadas fuera de la
convencion oficial (como paso con carpetas tipo axis_sample/plane_sample,
que no siguen <renders_root>/<symmetry_type>/... ).

Para cada archivo encontrado:
- Busca metadata_all.json en la MISMA carpeta (asi se sabe el total real de
  vistas Fibonacci, sin asumir 114).
- Clasifica cada n_views dentro del JSON como:
    "correcto (espaciado)"  -> coincide con get_n_views_entries actual (linspace)
    "secuencial (bug)"      -> es exactamente [0, 1, ..., n_views-1]
    "otro/inesperado"       -> ninguno de los dos
    "n/a (n_views>=total)"  -> no aplica (usa todas las vistas, ambos patrones coinciden)
- Marca si la carpeta sigue o no la convencion estandar
  <root>/<symmetry_type>/<object_id>/<size>/<lighting>/ (columna "estandar"),
  para distinguir predicciones "de produccion" de corridas sueltas/manuales.

Uso (correrlo apuntando a la raiz de datos completa, no solo a renders/):
    python Mapping/audit_view_indices_v2.py \\
        --search-root ../data \\
        --symmetry-types axis_sym plane_sym \\
        --out-summary audit_view_indices_v2_summary.csv \\
        --out-raw audit_view_indices_v2_raw.csv
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


def infer_standard_path_info(json_path: Path, search_root: Path, symmetry_types: list[str]):
    """
    Intenta interpretar json_path como <search_root>/.../<symmetry_type>/<object_id>/<size>/<lighting>/archivo.json
    Devuelve (es_estandar, symmetry_type, object_id, size, lighting) -- los ultimos
    4 son None si no calzo el patron.
    """
    try:
        rel_parts = json_path.relative_to(search_root).parts
    except ValueError:
        rel_parts = json_path.parts

    for i, part in enumerate(rel_parts):
        if part in symmetry_types:
            remainder = rel_parts[i + 1:]
            if len(remainder) >= 4:  # object_id / size / lighting / archivo.json
                object_id, size, lighting = remainder[0], remainder[1], remainder[2]
                return True, part, object_id, size, lighting
            return False, part, None, None, None
    return False, None, None, None, None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--search-root", required=True, type=Path,
                   help="Raiz DESDE LA QUE BUSCAR (ej: ../data) -- no tiene que ser renders/, "
                        "se recorre todo recursivamente.")
    p.add_argument("--symmetry-types", nargs="+", default=["axis_sym", "plane_sym"],
                   help="Usado solo para reconocer la estructura estandar en las rutas encontradas.")
    p.add_argument("--default-total", type=int, default=114,
                   help="Total de vistas Fibonacci a asumir si no hay metadata_all.json junto al archivo (default: 114).")
    p.add_argument("--out-summary", default="audit_view_indices_v2_summary.csv")
    p.add_argument("--out-raw", default="audit_view_indices_v2_raw.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    search_root = args.search_root.resolve()
    if not search_root.exists():
        raise SystemExit(f"[error] no existe: {search_root}")

    json_paths = sorted(search_root.rglob("molmo_multiview*.json"))
    print(f"Archivos molmo_multiview*.json encontrados bajo {search_root}: {len(json_paths)}")

    rows = []
    for json_path in json_paths:
        is_std, symmetry_type, object_id, size, lighting = infer_standard_path_info(
            json_path, search_root, args.symmetry_types
        )

        total, total_source = args.default_total, "fallback_default"
        meta_path = json_path.parent / "metadata_all.json"
        if meta_path.exists():
            try:
                total = len(json.load(open(meta_path, encoding="utf-8")))
                total_source = "metadata_all.json"
            except Exception:
                pass

        try:
            data = json.load(open(json_path, encoding="utf-8"))
        except Exception as e:
            print(f"[warn] no se pudo leer {json_path}: {e}")
            continue

        rel_path = json_path.relative_to(search_root)

        for n_views_str, entry in data.items():
            if not isinstance(entry, dict) or "images_sent" not in entry:
                continue
            n_views = int(n_views_str)
            indices = sorted(im["index"] for im in entry["images_sent"])
            classification = classify(indices, n_views, total)

            rows.append({
                "estandar": is_std,
                "symmetry_type": symmetry_type or "(desconocido)",
                "object_id": object_id or "(desconocido)",
                "size": size or "",
                "lighting": lighting or "",
                "ruta_relativa": str(rel_path),
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
        print("No se encontro ningun molmo_multiview*.json -- revisa --search-root.")
        return

    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(args.out_raw, index=False)
    print(f"Detalle guardado en: {args.out_raw}  ({len(raw_df):,} filas)")

    # --- Veredicto global, separado por estandar vs no-estandar ---
    print("\n=== Veredicto global ===")
    for is_std_val, label in [(True, "DENTRO de la estructura estandar"), (False, "FUERA de la estructura estandar")]:
        sub = raw_df[raw_df["estandar"] == is_std_val]
        if sub.empty:
            print(f"\n{label}: no se encontro ningun archivo.")
            continue
        counts = sub["clasificacion"].value_counts()
        print(f"\n{label} ({len(sub):,} filas):")
        for k, v in counts.items():
            print(f"  {k:24s} {v:>8,}  ({v/len(sub)*100:5.1f}%)")

    n_bug_total = (raw_df["clasificacion"] == "secuencial (bug)").sum()
    n_otro_total = (raw_df["clasificacion"] == "otro/inesperado").sum()
    print()
    if n_bug_total == 0 and n_otro_total == 0:
        print("VEREDICTO FINAL: no se encontro el bug secuencial en NINGUN archivo, "
              "este dentro o fuera de la estructura estandar.")
    else:
        print(f"VEREDICTO FINAL: {n_bug_total:,} corridas con bug secuencial y "
              f"{n_otro_total:,} con patron inesperado -- ver columnas 'ruta_relativa' "
              f"en {args.out_raw} para ubicarlas exactamente.")

    # --- Resumen agregado (igual que v1, pero con la columna 'estandar' agregada) ---
    summary = (
        raw_df
        .groupby(["estandar", "symmetry_type", "prompt_id", "experiment_id", "n_views", "clasificacion"])
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
        "estandar", "symmetry_type", "prompt_id", "experiment_id", "n_views",
        "secuencial (bug)", "correcto (espaciado)", "otro/inesperado",
        "n_total_checkeado", "pct_correcto",
    ]
    summary = summary[cols].sort_values(
        ["estandar", "symmetry_type", "prompt_id", "experiment_id", "n_views"]
    )
    summary.to_csv(args.out_summary, index=False)
    print(f"\nResumen por prompt/experimento guardado en: {args.out_summary}")

    n_fallback = (raw_df["total_fuente"] == "fallback_default").sum()
    if n_fallback:
        print(f"\n[atencion] {n_fallback:,} filas usaron el total por defecto "
              f"({args.default_total}) porque no encontraron metadata_all.json junto al "
              f"archivo -- revisar esas rutas si el resultado ahi parece raro.")

    n_no_std = (~raw_df["estandar"]).sum()
    if n_no_std:
        print(f"\n[info] {n_no_std:,} filas vienen de rutas que NO siguen la convencion "
              f"<symmetry_type>/<object_id>/<size>/<lighting>/ -- revisar 'ruta_relativa' "
              f"en el CSV crudo para saber donde estan exactamente.")


if __name__ == "__main__":
    main()
