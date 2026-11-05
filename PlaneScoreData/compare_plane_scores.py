#!/usr/bin/env python3
# Compara dos experimentos de eje o plano evaluados por Mapping/evaluate.py: métricas agregadas de
# los CSV de Mapping/compare_results_no_mesh.py y diferencia objeto a objeto del error angular,
# imputando 90 grados a los objetos sin predicción (misma convención que evaluate.py).
import argparse
import csv
import json
from pathlib import Path

import numpy as np

METODO = "triangulation"
METRICAS = [
    ("n_objects", "Objetos", "{:.0f}"),
    ("auc_angular", "AUC45", "{:.3f}"),
    ("angular_error_median", "Mediana", "{:.1f}"),
    ("angular_error_mean", "Media", "{:.1f}"),
    ("precision_10deg", "P@10", "{:.3f}"),
    ("f1_ref", "F1 voraz", "{:.3f}"),
    ("f1_ref_hungarian", "F1 hung.", "{:.3f}"),
    ("sde_ref_mean", "SDE_ref", "{:.4f}"),
]
ERROR_IMPUTADO = 90.0
MARGEN_GRADOS = 1.0
N_REMUESTREOS = 10000
SEMILLA = 0


def errores_por_objeto(ruta: Path, n_views: str) -> dict[str, float]:
    """ Lee el error angular por objeto, con 90 grados si no hay predicción válida.

    Args:
        * ruta: eval_*_results.json de evaluate.py
        * n_views: grupo de vistas

    Returns:
        * dict[str, float]: error angular en grados por id de objeto

    """
    with open(ruta, encoding="utf-8") as archivo:
        objetos = json.load(archivo)["objects"]
    errores = {}
    for obj_id, por_vistas in objetos.items():
        metricas = (por_vistas or {}).get(n_views, {})
        valido = metricas.get("status") == "ok"
        errores[obj_id] = metricas["angular_error_deg"] if valido else ERROR_IMPUTADO
    return errores


def comparar_pareado(base: dict[str, float], nuevo: dict[str, float]) -> dict[str, float]:
    """ Resume la diferencia de error angular (nuevo - base) sobre los mismos objetos.

    Args:
        * base: error por objeto de la línea base
        * nuevo: error por objeto del método nuevo

    Returns:
        * dict[str, float]: tamaño, diferencia media e IC 95%, y objetos que mejoran o empeoran

    """
    if set(base) != set(nuevo):
        raise ValueError(f"objetos distintos: {len(set(base) ^ set(nuevo))} no coinciden")
    ids = sorted(base)
    diferencias = np.array([nuevo[i] - base[i] for i in ids])
    rng = np.random.default_rng(SEMILLA)
    medias = rng.choice(diferencias, (N_REMUESTREOS, len(ids)), replace=True).mean(axis=1)
    return {
        "n": len(ids),
        "dif_media": float(diferencias.mean()),
        "ic_inf": float(np.percentile(medias, 2.5)),
        "ic_sup": float(np.percentile(medias, 97.5)),
        "mejoran": int((diferencias < -MARGEN_GRADOS).sum()),
        "empeoran": int((diferencias > MARGEN_GRADOS).sum()),
    }


def main() -> None:
    """ Imprime la comparación y la guarda en un CSV. """
    parser = argparse.ArgumentParser(description="Compara un método nuevo con su línea base.")
    parser.add_argument("--renders-root", type=Path, required=True)
    parser.add_argument("--symmetry-type", choices=["axis_sym", "plane_sym"], default="plane_sym")
    parser.add_argument("--comparison-csv", type=Path, nargs="+", required=True,
                        help="CSV de compare_results_no_mesh.py que contienen ambos experimentos")
    parser.add_argument("--base-id", required=True)
    parser.add_argument("--new-id", required=True)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--lighting", default="flat")
    parser.add_argument("--salida", type=Path, required=True, help="CSV de salida")
    args = parser.parse_args()

    filas = {}
    for ruta_csv in args.comparison_csv:
        with open(ruta_csv, encoding="utf-8", newline="") as archivo:
            filas.update({(f["experiment"], int(f["n_views"])): f for f in csv.DictReader(archivo)
                          if f["method"] == METODO})
    vistas = sorted({n for exp, n in filas if exp == args.base_id}
                    & {n for exp, n in filas if exp == args.new_id})
    if not vistas:
        raise ValueError(f"{args.comparison_csv}: no hay vistas comunes a ambos experimentos")

    ruta_eval = args.renders_root / args.symmetry_type / "eval_s{}_{}_{}_{}_results.json"
    filas_salida = []
    for n_views in vistas:
        print(f"\n=== {n_views} vistas ===")
        print(f"{'Métrica':10s} {'Base':>10s} {'Nuevo':>10s}")
        fila_salida = {"n_views": n_views}
        for columna, nombre, formato in METRICAS:
            base = filas[(args.base_id, n_views)].get(columna)
            nuevo = filas[(args.new_id, n_views)].get(columna)
            if not base or not nuevo:  # F1 no existe en eje
                continue
            base, nuevo = float(base), float(nuevo)
            print(f"{nombre:10s} {formato.format(base):>10s} {formato.format(nuevo):>10s}")
            fila_salida[f"{columna}_base"] = base
            fila_salida[f"{columna}_nuevo"] = nuevo

        errores = [
            errores_por_objeto(Path(str(ruta_eval).format(args.size, args.lighting, exp, METODO)),
                               str(n_views))
            for exp in (args.base_id, args.new_id)
        ]
        pareado = comparar_pareado(errores[0], errores[1])
        print(f"Dif. de error (nuevo - base, n={pareado['n']}): {pareado['dif_media']:+.2f} "
              f"grados, IC 95% [{pareado['ic_inf']:+.2f}, {pareado['ic_sup']:+.2f}]")
        print(f"Objetos: mejoran {pareado['mejoran']}, empeoran {pareado['empeoran']} "
              f"(margen de {MARGEN_GRADOS:.0f} grado)")
        fila_salida.update(pareado)
        filas_salida.append(fila_salida)

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    with open(args.salida, "w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas_salida[0]))
        escritor.writeheader()
        escritor.writerows(filas_salida)
    print(f"\nGuardado: {args.salida}")


if __name__ == "__main__":
    main()
