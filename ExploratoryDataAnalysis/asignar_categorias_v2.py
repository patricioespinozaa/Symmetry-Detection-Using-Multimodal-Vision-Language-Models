#!/usr/bin/env python3
"""
Asigna la categoria de ShapeNetCore v2 a cada objeto del subconjunto curado
de simetrias, usando los archivos ya disponibles en el servidor:

  - synset_modelid.csv  : generado con `unzip -l ShapeNetCore.v2.zip`
                          (2 columnas SIN encabezado: synsetId,modelId)
  - taxonomy.json        : extraido de ShapeNetCore.v2.zip
  - curated_axis_sym_obj/, curated_plane_sym_obj/ : carpetas con .obj/.txt

NOTA IMPORTANTE:
  El synsetId de synset_modelid.csv es el que aparece directo en la ruta
  del zip (<synsetId>/<modelId>/models/...). En ShapeNetCore v2 eso
  corresponde a las 55 categorias principales, NO a una subcategoria mas
  fina del arbol de taxonomy.json. Por eso este script NO reporta
  subSynsetId ni split (ese ultimo solo viene en all.csv de SHREC16,
  que al momento de escribir este script no estaba accesible).

Uso:
    python asignar_categorias_v2.py \
        --axis-dir  data/objects/curated_axis_sym_obj \
        --plane-dir data/objects/curated_plane_sym_obj \
        --synset-modelid synset_modelid.csv \
        --taxonomy taxonomy.json \
        --out categorias_objetos.csv
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HEX_RE = re.compile(r"^[0-9a-f]+$")


def norm_id(s):
    return s.strip().lower()


def read_object_ids(folder):
    folder = Path(folder)
    if not folder.is_dir():
        sys.exit(f"[error] No existe la carpeta {folder}")
    obj_ids = sorted({norm_id(p.stem) for p in folder.iterdir()
                      if p.is_file() and p.suffix.lower() == ".obj"})
    bad = [i for i in obj_ids if not HEX_RE.match(i)]
    if bad:
        print(f"[aviso] {folder}: {len(bad)} nombres no hexadecimales, ej.: {bad[:3]}")
    return obj_ids


def read_synset_modelid(path):
    pairs = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            synset, model = norm_id(row[0]), norm_id(row[1])
            pairs[model].append(synset)
            pairs[model.lstrip("0")].append(synset)  # tambien indexado sin ceros iniciales
    lens = Counter(len(k) for k in pairs if k)
    print(f"[info] {path}: {sum(1 for k in pairs if k)} modelId indexados "
          f"(con y sin ceros iniciales); largos: {dict(sorted(lens.items()))}")
    return pairs


def load_taxonomy(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    names = {norm_id(e["synsetId"]): e.get("name", "") for e in data if "synsetId" in e}
    print(f"[info] taxonomy.json: {len(names)} synsets con nombre")
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axis-dir", required=True)
    ap.add_argument("--plane-dir", required=True)
    ap.add_argument("--synset-modelid", required=True)
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--out", default="categorias_objetos.csv")
    ap.add_argument("--nombre-completo", action="store_true",
                    help="usar todos los lemas (ej. 'airplane,aeroplane,plane')")
    args = ap.parse_args()

    idx = read_synset_modelid(args.synset_modelid)
    tax = load_taxonomy(args.taxonomy)

    objects = [(i, "axial") for i in read_object_ids(args.axis_dir)] + \
              [(i, "planar") for i in read_object_ids(args.plane_dir)]
    print(f"[info] Objetos leidos: {len(objects)}")

    out_rows, not_found, ambiguous = [], [], []
    for obj_id, sym in objects:
        cands = sorted(set(idx.get(obj_id, [])))
        if not cands:
            not_found.append({"object_id": obj_id, "tipo_simetria": sym})
            continue
        if len(cands) > 1:
            ambiguous.append({"object_id": obj_id, "tipo_simetria": sym,
                              "candidatos": ",".join(cands)})
        synset = cands[0]
        name = tax.get(synset, "")
        out_rows.append({
            "object_id": obj_id,
            "tipo_simetria": sym,
            "synsetId": synset,
            "nombre_categoria": name if args.nombre_completo else name.split(",")[0].strip(),
        })

    out = Path(args.out)
    fields = ["object_id", "tipo_simetria", "synsetId", "nombre_categoria"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    nf_path = out.with_name(out.stem + "_no_encontrados.csv")
    with open(nf_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["object_id", "tipo_simetria"])
        w.writeheader()
        w.writerows(not_found)

    if ambiguous:
        amb_path = out.with_name(out.stem + "_ambiguos.csv")
        with open(amb_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["object_id", "tipo_simetria", "candidatos"])
            w.writeheader()
            w.writerows(ambiguous)
        print(f"[aviso] {len(ambiguous)} objetos con mas de un synsetId candidato -> {amb_path}")

    print("\n===== RESUMEN =====")
    for sym in ("axial", "planar"):
        tot = sum(1 for _, s in objects if s == sym)
        ok = sum(1 for r in out_rows if r["tipo_simetria"] == sym)
        print(f"{sym:7s}: {ok}/{tot} encontrados")
    print(f"No encontrados: {len(not_found)} -> {nf_path}")

    sin_nombre = sum(1 for r in out_rows if not r["nombre_categoria"])
    if sin_nombre:
        print(f"[aviso] {sin_nombre} filas con synsetId sin nombre en taxonomy.json")

    print("\nObjetos por categoria y tipo:")
    cat = Counter((r["nombre_categoria"] or r["synsetId"], r["tipo_simetria"]) for r in out_rows)
    for (c, s), n in sorted(cat.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {c:25s} {s:7s} {n}")
    print(f"\nSalida: {out}")


if __name__ == "__main__":
    main()
