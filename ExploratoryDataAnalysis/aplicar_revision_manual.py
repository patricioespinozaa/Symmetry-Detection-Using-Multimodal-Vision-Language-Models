#!/usr/bin/env python3
"""
Aplica las decisiones escritas en la columna "categoria_definitiva" de
revision_ambiguos.csv sobre categorias_objetos.csv, y genera el CSV final
de categorias.

En "categoria_definitiva" puedes escribir:
  - el synsetId (ej. 02880940; tambien acepta 2880940 si Excel borro el cero)
  - o el nombre de la categoria (ej. bowl, pot, cellular telephone)

Si una fila de revision_ambiguos.csv tiene "categoria_definitiva" vacia:
  - si no hay empate, se usa la recomendacion automatica (mayoria)
  - si hay empate, se mantiene la categoria original y se marca "pendiente"
  (con --estricto, cualquier fila vacia detiene el script)

Columna nueva "resuelto_por" en la salida:
  no_ambiguo | manual | mayoria | pendiente

Uso:
    python3 aplicar_revision_manual.py \
        --categorias categorias_objetos.csv \
        --revision revision_ambiguos.csv \
        --taxonomy ~/taxonomy.json \
        --out categorias_objetos_final.csv
"""
import argparse
import csv
import json
import sys


def norm_id(s):
    return s.strip().lower()


def norm_synset(s):
    s = s.strip()
    return s.zfill(8) if s.isdigit() else s.lower()


def read_csv_any(path):
    """Lee CSV separado por ',' o ';' (Excel en configuracion regional de Chile)."""
    for enc in ("utf-8-sig", "latin-1"):
        try:
            with open(path, newline="", encoding=enc) as f:
                text = f.read()
            break
        except UnicodeDecodeError:
            continue
    first_line = text.splitlines()[0] if text else ""
    delim = ";" if first_line.count(";") > first_line.count(",") else ","
    rows = list(csv.DictReader(text.splitlines(), delimiter=delim))
    return rows, delim


def load_taxonomy(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    short, lemmas = {}, {}
    for e in data:
        if "synsetId" not in e:
            continue
        sid = norm_synset(e["synsetId"])
        names = [n.strip().lower() for n in e.get("name", "").split(",") if n.strip()]
        short[sid] = names[0] if names else sid
        for n in names:
            lemmas.setdefault(n, set()).add(sid)
    return short, lemmas


def resolve_value(value, cand_synsets, short, lemmas):
    """Traduce lo escrito por el usuario a un synsetId. Devuelve (synset, error)."""
    v = value.strip()
    if not v:
        return None, None
    if v.isdigit():
        sid = norm_synset(v)
        return (sid, None) if sid in short else (None, f"synsetId '{v}' no existe en taxonomy.json")
    name = v.lower()
    # 1) preferir un candidato de esa misma fila cuyo nombre calce
    for c in cand_synsets:
        if short.get(c, "") == name or c in lemmas.get(name, set()):
            return c, None
    # 2) si no, buscar en toda la taxonomia
    matches = lemmas.get(name, set())
    if len(matches) == 1:
        return next(iter(matches)), None
    if len(matches) > 1:
        return None, f"nombre '{v}' es ambiguo en taxonomy.json ({sorted(matches)}); usa el synsetId"
    return None, f"nombre '{v}' no encontrado en taxonomy.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--categorias", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--out", default="categorias_objetos_final.csv")
    ap.add_argument("--estricto", action="store_true",
                    help="exigir que todas las filas de la revision tengan categoria_definitiva")
    args = ap.parse_args()

    short, lemmas = load_taxonomy(args.taxonomy)
    base_rows, _ = read_csv_any(args.categorias)
    rev_rows, delim = read_csv_any(args.revision)
    print(f"[info] {args.revision}: {len(rev_rows)} filas (separador '{delim}')")

    decisiones, errores, vacias = {}, [], []
    for r in rev_rows:
        oid = norm_id(r["object_id"])
        cands = [norm_synset(r[k]) for k in r
                 if k.startswith("candidato_") and k.endswith("_synset") and r[k].strip()]
        valor = (r.get("categoria_definitiva") or "").strip()
        sid, err = resolve_value(valor, cands, short, lemmas)
        if err:
            errores.append(f"{oid}: {err}")
            continue
        if sid is None:
            vacias.append(oid)
            if r.get("empate", "").strip().lower() == "no":
                # usar el candidato mas representado (candidato_1 viene ordenado asi)
                decisiones[oid] = (cands[0], "mayoria")
            continue
        if sid not in cands:
            print(f"[aviso] {oid}: '{valor}' -> {sid} ({short.get(sid)}) no estaba entre "
                  f"los candidatos {cands}; se aplica igual")
        decisiones[oid] = (sid, "manual")

    if errores:
        print("\n[error] Valores que no se pudieron interpretar (corrigelos y vuelve a correr):")
        for e in errores:
            print(f"  {e}")
        sys.exit(1)
    if args.estricto and vacias:
        print(f"[error] --estricto: {len(vacias)} filas sin categoria_definitiva:")
        for oid in vacias:
            print(f"  {oid}")
        sys.exit(1)

    ids_base = {norm_id(r["object_id"]) for r in base_rows}
    no_encontrados = [oid for oid in (norm_id(r["object_id"]) for r in rev_rows)
                      if oid not in ids_base]
    if no_encontrados:
        print(f"[aviso] {len(no_encontrados)} object_id de la revision no estan en "
              f"{args.categorias} (¿Excel los altero?):")
        for oid in no_encontrados:
            print(f"  {oid}")

    ids_revision = {norm_id(r["object_id"]) for r in rev_rows}
    out_rows, cambios = [], []
    conteo = {"no_ambiguo": 0, "manual": 0, "mayoria": 0, "pendiente": 0}
    for r in base_rows:
        oid = norm_id(r["object_id"])
        original = norm_synset(r["synsetId"])
        if oid in decisiones:
            sid, fuente = decisiones[oid]
        elif oid in ids_revision:
            sid, fuente = original, "pendiente"
        else:
            sid, fuente = original, "no_ambiguo"
        conteo[fuente] += 1
        if sid != original:
            cambios.append((oid, r["tipo_simetria"], short.get(original, original),
                            short.get(sid, sid), fuente))
        out_rows.append({
            "object_id": r["object_id"],
            "tipo_simetria": r["tipo_simetria"],
            "synsetId": sid,
            "nombre_categoria": short.get(sid, sid),
            "resuelto_por": fuente,
        })

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["object_id", "tipo_simetria", "synsetId",
                                          "nombre_categoria", "resuelto_por"])
        w.writeheader()
        w.writerows(out_rows)

    print(f"\n[info] {len(out_rows)} objetos en la salida: {conteo}")
    if cambios:
        print(f"\n===== CAMBIOS respecto a {args.categorias} ({len(cambios)}) =====")
        for oid, tipo, antes, despues, fuente in cambios:
            print(f"  {oid} [{tipo}]: {antes} -> {despues}  ({fuente})")
    if conteo["pendiente"]:
        print(f"\n[aviso] {conteo['pendiente']} objetos empatados sin decision; "
              f"quedaron con su categoria original (resuelto_por=pendiente).")
    print(f"\nSalida: {args.out}")


if __name__ == "__main__":
    main()
