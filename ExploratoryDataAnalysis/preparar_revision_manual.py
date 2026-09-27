#!/usr/bin/env python3
"""
Prepara una revision manual de los 65 objetos ambiguos: copia 3 imagenes
representativas de cada uno (de las renders que ya genero el pipeline) y
arma un CSV con las categorias candidatas, sus estadisticas de apoyo, y una
columna vacia para que anotes tu decision.

Estructura de renders esperada (confirmada en el servidor):
    <renders-dir>/axis_sym/<object_id>/<resolucion>/<iluminacion>/IND_NN_AZ_AAA_EL_+EE.png
    <renders-dir>/plane_sym/<object_id>/<resolucion>/<iluminacion>/IND_NN_AZ_AAA_EL_+EE.png

  - <resolucion>: carpeta como "448", "224", "1136" (--resolution, default 448)
  - <iluminacion>: "flat" | "brighter" | "darker" (--lighting, default flat)
  - la subcarpeta axis_sym / plane_sym se elige automaticamente segun el
    tipo_simetria del objeto (axial -> axis_sym, planar -> plane_sym)

Seleccion de vistas: se ordenan los archivos IND_NN... numericamente y se
toman --n-images equiespaciadas (primera, intermedia(s), ultima) para
maximizar diversidad angular, en vez de las primeras N alfabeticas (que
quedarian todas muy cercanas en elevacion).

Requiere: pillow SOLO si usas --resize (por defecto se copian los archivos
tal cual, sin recodificar, ya que la carpeta de resolucion ya existe).

Uso:
    ./preparar_revision_manual.py \
        --categorias categorias_objetos.csv \
        --ambiguos categorias_objetos_ambiguos.csv \
        --taxonomy ~/taxonomy.json \
        --renders-dir ~/data/renders \
        --out-dir revision_ambiguos \
        --out revision_ambiguos.csv
"""
import argparse
import csv
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

IND_RE = re.compile(r"^IND_(\d+)_")


def norm_id(s):
    return s.strip().lower()


def load_taxonomy(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {norm_id(e["synsetId"]): e.get("name", "").split(",")[0].strip()
           for e in data if "synsetId" in e}


def load_ambiguous(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cands = [norm_id(c) for c in row["candidatos"].split(",")]
            rows.append({"object_id": norm_id(row["object_id"]),
                        "tipo_simetria": row["tipo_simetria"],
                        "candidatos": cands})
    return rows


def load_confident_counts(path, ambiguous_ids):
    total = Counter()
    by_type = Counter()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            oid = norm_id(row["object_id"])
            if oid in ambiguous_ids:
                continue
            synset = norm_id(row["synsetId"])
            tipo = row["tipo_simetria"]
            total[synset] += 1
            by_type[(synset, tipo)] += 1
    return total, by_type


def pick_spread_views(files, n):
    """files: lista de Path ya ordenados numericamente por IND. Devuelve n
    archivos equiespaciados (incluye siempre el primero y el ultimo)."""
    N = len(files)
    if N == 0:
        return []
    if N <= n:
        return files
    if n == 1:
        return [files[N // 2]]
    idxs = [round(i * (N - 1) / (n - 1)) for i in range(n)]
    seen = []
    for i in idxs:
        if i not in seen:
            seen.append(i)
    return [files[i] for i in seen]


def find_views(renders_dir, subdir, object_id, resolution, lighting):
    folder = Path(renders_dir) / subdir / object_id / str(resolution) / lighting
    if not folder.is_dir():
        return [], folder
    files = [p for p in folder.iterdir() if p.suffix.lower() == ".png"
            and IND_RE.match(p.name)]
    files.sort(key=lambda p: int(IND_RE.match(p.name).group(1)))
    return files, folder


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--categorias", required=True)
    ap.add_argument("--ambiguos", required=True)
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--renders-dir", required=True)
    ap.add_argument("--axis-subdir", default="axis_sym")
    ap.add_argument("--plane-subdir", default="plane_sym")
    ap.add_argument("--resolution", default="448")
    ap.add_argument("--lighting", default="flat", choices=["flat", "brighter", "darker"])
    ap.add_argument("--out-dir", default="revision_ambiguos")
    ap.add_argument("--out", default="revision_ambiguos.csv")
    ap.add_argument("--n-images", type=int, default=3)
    args = ap.parse_args()

    tax = load_taxonomy(args.taxonomy)
    amb_rows = load_ambiguous(args.ambiguos)
    ambiguous_ids = {r["object_id"] for r in amb_rows}
    total_counts, type_counts = load_confident_counts(args.categorias, ambiguous_ids)

    def nombre(s):
        return tax.get(s, s)

    os.makedirs(args.out_dir, exist_ok=True)
    out_rows = []
    sin_render = []
    max_cands = max(len(r["candidatos"]) for r in amb_rows)

    for r in amb_rows:
        oid, tipo, cands = r["object_id"], r["tipo_simetria"], r["candidatos"]
        info = sorted(
            ({"synset": c, "nombre": nombre(c), "n_total": total_counts.get(c, 0),
              "n_mismo_tipo": type_counts.get((c, tipo), 0)} for c in cands),
            key=lambda x: (x["n_total"], x["n_mismo_tipo"]), reverse=True)
        empate = len(info) > 1 and info[0]["n_total"] == info[1]["n_total"]
        recomendado = "" if empate else info[0]["synset"]

        subdir = args.axis_subdir if tipo == "axial" else args.plane_subdir
        files, folder = find_views(args.renders_dir, subdir, oid, args.resolution,
                                  args.lighting)
        chosen = pick_spread_views(files, args.n_images)

        obj_out_dir = os.path.join(args.out_dir, oid)
        if chosen:
            os.makedirs(obj_out_dir, exist_ok=True)
            for i, src in enumerate(chosen, start=1):
                dst = os.path.join(obj_out_dir, f"{oid}_{i}{src.suffix}")
                shutil.copyfile(src, dst)
        else:
            sin_render.append((oid, str(folder)))

        row = {
            "object_id": oid,
            "tipo_simetria": tipo,
            "carpeta_imagenes": obj_out_dir if chosen else "SIN_RENDERS",
            "n_imagenes_encontradas": len(chosen),
            "empate": "si" if empate else "no",
            "recomendado_auto": nombre(recomendado) if recomendado else "",
        }
        for i in range(max_cands):
            pref = f"candidato_{i+1}"
            if i < len(info):
                c = info[i]
                row[f"{pref}_synset"] = c["synset"]
                row[f"{pref}_nombre"] = c["nombre"]
                row[f"{pref}_n_total"] = c["n_total"]
                row[f"{pref}_n_mismo_tipo"] = c["n_mismo_tipo"]
            else:
                row[f"{pref}_synset"] = ""
                row[f"{pref}_nombre"] = ""
                row[f"{pref}_n_total"] = ""
                row[f"{pref}_n_mismo_tipo"] = ""
        row["categoria_definitiva"] = ""  # <- completar manualmente
        out_rows.append(row)

    fields = ["object_id", "tipo_simetria", "carpeta_imagenes", "n_imagenes_encontradas",
              "empate", "recomendado_auto"]
    for i in range(max_cands):
        pref = f"candidato_{i+1}"
        fields += [f"{pref}_synset", f"{pref}_nombre", f"{pref}_n_total", f"{pref}_n_mismo_tipo"]
    fields.append("categoria_definitiva")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    print(f"[info] {len(amb_rows)} objetos ambiguos procesados")
    print(f"[info] imagenes copiadas en: {args.out_dir}/<object_id>/  "
          f"(resolucion={args.resolution}, iluminacion={args.lighting})")
    print(f"[info] CSV para revision manual: {args.out}")
    if sin_render:
        print(f"\n[aviso] {len(sin_render)} objetos SIN carpeta de renders encontrada:")
        for oid, folder in sin_render:
            print(f"  {oid} -> se esperaba {folder}")


if __name__ == "__main__":
    main()
