#!/usr/bin/env python3
# Retroproyecta a la malla los puntos de plane_v02 (224px, flat, Flujo A multiimagen) y ajusta el
# plano con SVD y RANSAC+SVD, en las mismas 7 variantes que run_axis_malla.py: base (un rayo por
# punto), parche de 3x3 y 5x5 sub-rayos, agrupamiento voraz y HDBSCAN con min_samples 2, 3 y 5.
# Todo corre en CPU (trimesh, numpy, scipy, scikit-learn); no usa GPU ni vuelve a correr Molmo2.
#
#   python3 PlaneScoreData/run_plane_malla.py chequeo     # solo lee y valida; no escribe nada
#   python3 PlaneScoreData/run_plane_malla.py todo        # retroproyección, ajuste, evaluación
#   python3 PlaneScoreData/run_plane_malla.py todo --sin-referencia   # sin SDE_ref ni F1_ref
#
# Reutiliza las etapas de run_axis_malla.py cambiando solo el tipo de simetría, el prompt y la
# referencia sin malla (plane_v02 con S_dat y un plano por objeto). Las predicciones usan ids
# propios (plane_v02_s224_flat_ndcfix_mesh*), así que no pisan las del eje ni las sin malla.
# El pareado compara el error angular objeto a objeto contra plane_v02_s224_flat_ndcfix_sdat_mp1_nomesh.
import argparse
import datetime as dt
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "PlaneScoreData"))
import run_axis_malla as eje  # noqa: E402

# Las funciones de run_axis_malla.py leen estas constantes del módulo al ejecutarse.
eje.TIPO = "plane_sym"
eje.PROMPT = "plane_v02"
eje.ALIAS = f"{eje.PROMPT}_s{eje.SIZE}_{eje.LUZ}_ndcfix_mesh"
eje.ID_SIN_MALLA = f"{eje.PROMPT}_s{eje.SIZE}_{eje.LUZ}_ndcfix_sdat_mp1_nomesh"
eje.CSV_SIN_MALLA = (REPO / "resultados_finales" / "ndcfix_flujoA" / "experiments_05_11_2026"
                     / "plane_sym_final_comparison.csv")


def resumir(filas: dict) -> None:
    """ Imprime las variantes ordenadas por AUC promedio, con F1_ref, junto a la referencia sin malla. """
    sin_malla = eje.leer_csv(eje.CSV_SIN_MALLA) if eje.CSV_SIN_MALLA.exists() else {}

    def celdas_texto(c: dict) -> str:
        f1 = c.get("f1_ref") or "nan"
        return f"{c['n_objects']:>6} {float(c['auc_angular']):7.3f} {float(f1):6.3f}"

    print(f"\n{'variante':12} {'método':11} " + " ".join(
        f"{'obj' + str(v):>6} {'AUC' + str(v):>7} {'F1' + str(v):>6}" for v in eje.VISTAS))
    referencia = [sin_malla.get((eje.ID_SIN_MALLA, str(v))) for v in eje.VISTAS]
    if all(referencia):
        print(f"{'sin malla':12} {'S_dat':11} " + " ".join(celdas_texto(c) for c in referencia))
    orden = sorted(((v, m) for v in eje.VARIANTES for m in eje.METODOS),
                   key=lambda vm: -sum(float(filas[(eje.id_variante(vm[0]), vm[1], str(n))]
                                             ["auc_angular"]) for n in eje.VISTAS))
    for variante, metodo in orden:
        celdas = [filas[(eje.id_variante(variante), metodo, str(v))] for v in eje.VISTAS]
        print(f"{variante:12} {metodo:11} " + " ".join(celdas_texto(c) for c in celdas))


def main() -> None:
    """ Ejecuta el chequeo o el experimento completo y se detiene en el primer error. """
    parser = argparse.ArgumentParser(description="plane_v02 a 224px/flat retroproyectado a malla.")
    parser.add_argument("etapa", choices=["chequeo", "todo"])
    parser.add_argument("--renders-root", type=Path, default=Path("/data/pespinoz/renders"))
    parser.add_argument("--objects-root", type=Path, default=Path("/data/pespinoz/objects"))
    parser.add_argument("--carpeta", default="plane_malla_224",
                        help="subcarpeta de resultados_finales")
    parser.add_argument("--jobs", type=int, default=12, help="procesos de CPU simultáneos")
    parser.add_argument("--sin-referencia", action="store_true",
                        help="omite SDE_ref y F1_ref en evaluate.py, la etapa más lenta")
    args = parser.parse_args()
    args.salida = REPO / "resultados_finales" / args.carpeta

    eje.chequear(args)
    if args.etapa == "chequeo":
        return
    logs = args.salida / "logs"
    try:
        print(f"{eje.crear_alias(args.renders_root)} enlaces -> molmo_multiview_{eje.ALIAS}.json")
        for nombre, etapa in (("retroproyección", eje.retroproyectar), ("ajuste", eje.ajustar),
                              ("evaluación", eje.evaluar)):
            print(f"[{dt.datetime.now():%H:%M:%S}] {nombre}", flush=True)
            etapa(args, logs)
        ruta = eje.comparar(args, logs)
        filas = eje.leer_filas(ruta)
        faltan = [(eje.id_variante(v), m, n) for v in eje.VARIANTES for m in eje.METODOS
                  for n in eje.VISTAS if (eje.id_variante(v), m, str(n)) not in filas]
        if faltan:
            raise ValueError(f"faltan filas en {ruta}: {faltan}")
        eje.parear(args, filas)
    except (RuntimeError, ValueError) as error:
        sys.exit(f"[error] {error}")
    resumir(filas)
    print(f"\nResultados: {ruta}\nTerminó sin errores.")


if __name__ == "__main__":
    main()
