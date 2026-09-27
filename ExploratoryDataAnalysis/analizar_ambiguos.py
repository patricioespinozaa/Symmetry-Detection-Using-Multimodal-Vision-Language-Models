#!/usr/bin/env python3
"""
Para cada objeto ambiguo (mas de un synsetId candidato), muestra cuantos
objetos de TU dataset (los que ya quedaron sin ambiguedad) estan asignados
a cada categoria candidata. Asi puedes elegir el candidato mas representado
en vez de crear una categoria nueva con muy pocos objetos.

La logica:
  1. Se toman los conteos "de confianza" = categorias_objetos.csv EXCLUYENDO
     los object_id que aparecen en el csv de ambiguos (para no contarse a
     si mismos y sesgar la comparacion).
  2. Para cada objeto ambiguo, se compara cuantos objetos de confianza tiene
     cada synsetId candidato (en total, y dentro del mismo tipo de simetria).
  3. Se recomienda el candidato con mas objetos de confianza. Si ambos
     candidatos tienen 0 (ninguno esta representado aun), se marca como
     "SIN DATOS" para que la decidas manualmente.

Uso:
    ./analizar_ambiguos.py \
        --categorias categorias_objetos.csv \
        --ambiguos categorias_objetos_ambiguos.csv \
        --taxonomy taxonomy.json
"""
import argparse
import csv
import json
from collections import Counter


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
                continue  # no contar objetos que ya son ambiguos
            synset = norm_id(row["synsetId"])
            tipo = row["tipo_simetria"]
            total[synset] += 1
            by_type[(synset, tipo)] += 1
    return total, by_type


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--categorias", required=True)
    ap.add_argument("--ambiguos", required=True)
    ap.add_argument("--taxonomy", required=True)
    args = ap.parse_args()

    tax = load_taxonomy(args.taxonomy)
    amb_rows = load_ambiguous(args.ambiguos)
    ambiguous_ids = {r["object_id"] for r in amb_rows}
    total_counts, type_counts = load_confident_counts(args.categorias, ambiguous_ids)

    def nombre(s):
        return tax.get(s, s)

    print(f"[info] {len(amb_rows)} objetos ambiguos | conteos de confianza calculados "
          f"sobre {sum(total_counts.values())} objetos sin ambiguedad\n")

    resueltos = Counter()      # synset -> cuantos ambiguos se le asignarian
    sin_datos = []
    detalle = []

    for r in amb_rows:
        oid, tipo, cands = r["object_id"], r["tipo_simetria"], r["candidatos"]
        info_cands = []
        for c in cands:
            info_cands.append({
                "synset": c, "nombre": nombre(c),
                "n_total": total_counts.get(c, 0),
                "n_mismo_tipo": type_counts.get((c, tipo), 0),
            })
        # ordenar candidatos de mayor a menor representacion (total, luego mismo tipo)
        info_cands.sort(key=lambda x: (x["n_total"], x["n_mismo_tipo"]), reverse=True)
        mejor = info_cands[0]
        empate = len(info_cands) > 1 and mejor["n_total"] == info_cands[1]["n_total"]

        if mejor["n_total"] == 0:
            sin_datos.append((oid, tipo, info_cands))
        else:
            resueltos[mejor["synset"]] += 1

        detalle.append((oid, tipo, info_cands, mejor, empate))

    # ---- Detalle por objeto ----
    print("=" * 100)
    print("DETALLE POR OBJETO")
    print("=" * 100)
    for oid, tipo, info_cands, mejor, empate in detalle:
        cand_str = "  vs  ".join(
            f"{c['nombre']} ({c['synset']}): {c['n_total']} objetos "
            f"[{c['n_mismo_tipo']} del mismo tipo '{tipo}']"
            for c in info_cands
        )
        flag = " <-- EMPATE, revisar manualmente" if empate else ""
        rec = "SIN DATOS (ninguno representado aun)" if mejor["n_total"] == 0 \
            else f"-> recomendado: {mejor['nombre']} ({mejor['synset']})"
        print(f"{oid} [{tipo}]")
        print(f"    candidatos: {cand_str}")
        print(f"    {rec}{flag}")
        print()

    # ---- Resumen ----
    print("=" * 100)
    print("RESUMEN: a que categoria se sumaria cada objeto ambiguo resuelto")
    print("=" * 100)
    for synset, n in resueltos.most_common():
        base = total_counts.get(synset, 0)
        print(f"  {nombre(synset):25s} ({synset}): +{n} ambiguos  "
              f"({base} ya confirmados -> quedaria en {base + n})")

    if sin_datos:
        print(f"\n{'=' * 100}")
        print(f"SIN DATOS DE APOYO ({len(sin_datos)} objetos): ninguno de los candidatos "
              f"tiene objetos confirmados en tu dataset. Requieren decision manual.")
        print("=" * 100)
        for oid, tipo, info_cands in sin_datos:
            opciones = " / ".join(f"{c['nombre']} ({c['synset']})" for c in info_cands)
            print(f"  {oid} [{tipo}]: {opciones}")


if __name__ == "__main__":
    main()
