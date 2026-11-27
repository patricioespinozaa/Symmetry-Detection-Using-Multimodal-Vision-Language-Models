#!/usr/bin/env python3
# Resultados por categoría de objeto: AUC, error medio y mediano, P@10 y, en plano, F1 voraz de cada
# configuración restringidos a
# las categorías con suficientes instancias. Lee el error angular por objeto de los
# eval_*_triangulation_results.json que ya dejó evaluate.py (90 grados a los objetos sin predicción,
# igual que en las tablas) y la categoría de ExploratoryDataAnalysis/categorias_objetos_final.csv.
# El F1 voraz de una categoría acumula los conteos (tp, fp, fn) por umbral de sus objetos con
# predicción, igual que evaluate.py lo hace sobre todo el conjunto.
# La fila "todas" usa los 850 objetos y debe coincidir con las tablas del capítulo 4.
# Solo lee archivos existentes; no usa GPU.
#
#   python3 PlaneScoreData/diagnostico_categorias.py --salida resultados_finales/diagnosticos/categorias_resumen.csv
#   python3 PlaneScoreData/diagnostico_categorias.py --min-objetos 30 --salida /tmp/categorias.csv
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "PlaneScoreData"))
sys.path.insert(0, str(REPO / "Mapping"))
from compare_plane_scores import errores_por_objeto  # noqa: E402
from evaluate import AUC_ANGULAR_MAX, _f1_from_counts_by_threshold, auc_from_errors  # noqa: E402

VISTAS = ("6", "14", "26")
CATEGORIAS = REPO / "ExploratoryDataAnalysis" / "categorias_objetos_final.csv"
TIPO_CSV = {"axis_sym": "axial", "plane_sym": "planar"}
# Configuraciones principales: (etiqueta, id final de la evaluación)
PRINCIPALES = {
    "axis_sym": [("Molmo2 multiimagen", "axis_v04_s224_flat_ndcfix_expE_nomesh"),
                 ("Molmo2 una vista", "axis_v04_single_oct26_s224_flat_ndcfix_expE_nomesh"),
                 ("Qwen3-VL-8B", "axis_v04_qwen3vl_s224_flat_ndcfix_expE_nomesh"),
                 ("MolmoPoint-8B", "axis_v04_molmopoint_completo_s224_flat_ndcfix_expE_nomesh")],
    "plane_sym": [("Molmo2 multiimagen", "plane_v02_s224_flat_ndcfix_sdat_mp1_nomesh"),
                  ("Molmo2 una vista", "plane_v02_single_s224_flat_ndcfix_sdat_mp1_nomesh"),
                  ("Qwen3-VL-8B", "plane_v02_qwen3vl_s224_flat_ndcfix_sdat_mp1_nomesh"),
                  ("MolmoPoint-8B", "plane_v02_molmopoint_s224_flat_ndcfix_sdat_mp1_nomesh")],
}


def metricas(errores: list[float]) -> dict:
    """ AUC, error medio y mediano y P@10 de una lista de errores angulares. """
    e = np.asarray(errores, dtype=float)
    return {"n": len(e), "auc": round(auc_from_errors(list(e), AUC_ANGULAR_MAX), 4),
            "media": round(float(np.mean(e)), 2), "mediana": round(float(np.median(e)), 2),
            "p10": round(float(np.mean(e < 10)), 4)}


def conteos_f1(ruta: Path, vistas: str) -> dict[str, dict]:
    """ Conteos (tp, fp, fn) por umbral del F1 voraz de cada objeto con predicción (solo plano). """
    objetos = json.loads(ruta.read_text(encoding="utf-8"))["objects"]
    conteos = {}
    for obj, por_vistas in objetos.items():
        m = (por_vistas or {}).get(vistas, {})
        if m.get("status") == "ok" and m.get("f1_counts_ref") is not None:
            conteos[obj] = m["f1_counts_ref"]
    return conteos


def f1_grupo(conteos: dict[str, dict], objetos: list[str]) -> float | None:
    """ F1 voraz de un grupo de objetos: suma los conteos por umbral y promedia el F1 de los umbrales. """
    totales = defaultdict(lambda: [0, 0, 0])
    for o in objetos:
        for umbral, (tp, fp, fn) in conteos.get(o, {}).items():
            totales[float(umbral)][0] += tp
            totales[float(umbral)][1] += fp
            totales[float(umbral)][2] += fn
    return round(_f1_from_counts_by_threshold({u: tuple(c) for u, c in totales.items()}), 4) if totales else None


def main() -> None:
    """ Calcula las métricas por categoría para cada configuración principal y número de vistas. """
    ap = argparse.ArgumentParser(description="Resultados por categoría de objeto.")
    ap.add_argument("--renders-root", type=Path, default=Path("/data/pespinoz/renders"))
    ap.add_argument("--tipo", choices=["axis_sym", "plane_sym", "ambos"], default="ambos")
    ap.add_argument("--min-objetos", type=int, default=20, help="mínimo de objetos para reportar una categoría")
    ap.add_argument("--salida", type=Path, required=True)
    args = ap.parse_args()

    with open(CATEGORIAS, encoding="utf-8", newline="") as f:
        filas_cat = list(csv.DictReader(f))
    filas = []
    for tipo in (["axis_sym", "plane_sym"] if args.tipo == "ambos" else [args.tipo]):
        categoria = {r["object_id"]: r["nombre_categoria"] for r in filas_cat if r["tipo_simetria"] == TIPO_CSV[tipo]}
        conteo = defaultdict(int)
        for c in categoria.values():
            conteo[c] += 1
        grandes = sorted((c for c, n in conteo.items() if n >= args.min_objetos), key=lambda c: -conteo[c])
        print(f"\n=== {tipo}: {len(categoria)} objetos, {len(grandes)} categorías con >= {args.min_objetos} "
              f"({sum(conteo[c] for c in grandes)} objetos)")
        for etiqueta, exp in PRINCIPALES[tipo]:
            ruta = args.renders_root / tipo / f"eval_s224_flat_{exp}_triangulation_results.json"
            if not ruta.exists():
                print(f"  [aviso] falta {ruta.name}, se omite {etiqueta}")
                continue
            for v in VISTAS:
                errores = errores_por_objeto(ruta, v)
                conteos = conteos_f1(ruta, v) if tipo == "plane_sym" else {}
                # Objetos sin entrada en la evaluación: 90 grados, como en las tablas
                por_objeto = {o: errores.get(o, 90.0) for o in categoria}
                grupos = {"todas": list(por_objeto.values())}
                for c in grandes:
                    grupos[c] = [e for o, e in por_objeto.items() if categoria[o] == c]
                grupos["otras"] = [e for o, e in por_objeto.items() if categoria[o] not in grandes]
                miembros = {"todas": list(categoria), "otras": [o for o in categoria if categoria[o] not in grandes]}
                miembros.update({c: [o for o in categoria if categoria[o] == c] for c in grandes})
                for c, errs in grupos.items():
                    if errs:
                        filas.append({"tipo": tipo, "configuracion": etiqueta, "experiment": exp,
                                      "n_views": int(v), "categoria": c, **metricas(errs),
                                      "f1": f1_grupo(conteos, miembros[c]) if conteos else None})
            print(f"  {etiqueta:20} AUC por categoría (6 / 14 / 26 vistas):")
            for c in ["todas"] + grandes + ["otras"]:
                aucs = [f["auc"] for f in filas if f["experiment"] == exp and f["categoria"] == c]
                n = next((f["n"] for f in filas if f["experiment"] == exp and f["categoria"] == c), 0)
                if aucs:
                    print(f"    {c:18} n={n:>4}  " + "  ".join(f"{a:.3f}" for a in aucs))

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    with open(args.salida, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)
    print(f"\nResumen: {args.salida}")


if __name__ == "__main__":
    main()
