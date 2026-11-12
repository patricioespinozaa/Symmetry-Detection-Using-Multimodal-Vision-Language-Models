#!/usr/bin/env python3
# Experimento multiplano: con los mismos puntos de Molmo2 plane_v02 a 224px/flat (sin inferencia
# nueva, solo CPU), estima hasta 1, 2 y 3 planos por objeto con S_dat y consolidación secuencial,
# sin malla. Evalúa recall/precisión sobre el conjunto de planos GT y SDE_ref/F1_ref, compara las tres
# corridas y sube la carpeta a GitHub.
#
#   python3 PlaneScoreData/run_plane_multiplano.py chequeo     # solo lee
#   python3 PlaneScoreData/run_plane_multiplano.py todo
#
# max_planes 1 se incluye para comparar con la misma métrica de conjunto; su primer plano debe ser
# idéntico al de plane_v02_s224_flat_ndcfix_sdat_mp1_nomesh, y el runner lo verifica.
import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "PlaneScoreData"))
from run_plane_resolucion import correr, molmo_to_ndc  # noqa: E402
from run_plane_v02_224 import renombrar_comparacion, subir  # noqa: E402

TIPO = "plane_sym"
SIZE = 224
LUZ = "flat"
VISTAS = (6, 14, 26)
TOTAL_OBJETOS = 850
MAXIMOS = (1, 2, 3)
PY = sys.executable
ID_ENTRADA = f"plane_v02_s{SIZE}_{LUZ}_ndcfix_mp1_nomesh"     # enlace a los puntos de v02
ID_UN_PLANO = f"plane_v02_s{SIZE}_{LUZ}_ndcfix_sdat_mp1_nomesh"  # corrida de un plano existente
METODO = "triangulation_multiplane"
NOMBRE = "multiplano"


def id_salida(maximo: int) -> str:
    """ Id de las predicciones con hasta `maximo` planos. No pisa la corrida de un plano. """
    return f"plane_v02_s{SIZE}_{LUZ}_ndcfix_sdat_multi_mp{maximo}_nomesh"


def carpetas(renders: Path) -> list[Path]:
    """ Objetos planares con renders a 224px/flat. """
    return [d for d in sorted((renders / TIPO).iterdir())
            if (d / str(SIZE) / LUZ / "metadata_all.json").exists()]


def chequear(renders: Path) -> None:
    """ Revisa la corrección NDC, los 850 objetos y que existan las entradas y la corrida base. """
    if molmo_to_ndc(0, 0)[0] != 1.0:
        sys.exit("[error] pipeline_common/camera.py no tiene la corrección NDC")
    objetos = carpetas(renders)
    if len(objetos) != TOTAL_OBJETOS:
        sys.exit(f"[error] {len(objetos)} objetos con renders a {SIZE}px/{LUZ}")
    entrada = f"molmo_multiview_{ID_ENTRADA}.json"
    con_entrada = sum((d / str(SIZE) / LUZ / entrada).exists() for d in objetos)
    con_base = sum((d / f"predicted_symmetry_{ID_UN_PLANO}.json").exists() for d in objetos)
    print(f"Objetos: {len(objetos)}  con {entrada}: {con_entrada}  con predicción de un plano: "
          f"{con_base}")
    if con_entrada != TOTAL_OBJETOS:
        sys.exit(f"[error] faltan entradas {entrada}")


def verificar_primer_plano(renders: Path, maximo: int) -> None:
    """ El plano 1 de la corrida nueva debe ser el de la corrida de un plano existente. """
    distintos = revisados = 0
    for d in carpetas(renders):
        base, nuevo = (d / f"predicted_symmetry_{i}.json" for i in (ID_UN_PLANO, id_salida(maximo)))
        if not (base.exists() and nuevo.exists()):
            continue
        a = json.loads(base.read_text(encoding="utf-8"))["n_views_predictions"]
        b = json.loads(nuevo.read_text(encoding="utf-8"))["n_views_predictions"]
        for v in map(str, VISTAS):
            if v in a and v in b:
                revisados += 1
                planos = b[v][METODO]["planes"]
                distintos += planos[0]["normal"] != a[v]["triangulation"]["normal"]
    print(f"mp{maximo}: primer plano igual al de {ID_UN_PLANO} en {revisados - distintos}/"
          f"{revisados} grupos", flush=True)
    if distintos:
        raise ValueError(f"mp{maximo}: {distintos} grupos con un primer plano distinto")


def resumen(ruta: Path) -> None:
    """ Imprime las métricas de conjunto de las tres corridas. """
    with open(ruta, newline="", encoding="utf-8") as archivo:
        filas = [f for f in csv.DictReader(archivo) if f["method"] == METODO]
    columnas = [("n_objects", "obj", "{:>4.0f}"), ("n_planes_predicted_mean", "pred", "{:5.2f}"),
                ("n_true_planes_mean", "GT", "{:5.2f}"), ("recall_planes_mean", "recall", "{:6.3f}"),
                ("precision_planes_mean", "prec.", "{:6.3f}"), ("f1_ref", "F1", "{:6.3f}"),
                ("f1_ref_hungarian", "F1 hung", "{:7.3f}")]
    print(f"\n{'corrida':6} {'vistas':>6} " + " ".join(f"{n:>7}" for _, n, _ in columnas))
    for maximo in MAXIMOS:
        for v in VISTAS:
            f = next((f for f in filas if f["experiment"] == id_salida(maximo)
                      and f["n_views"] == str(v)), None)
            if f is None:
                raise ValueError(f"falta la fila de mp{maximo} con {v} vistas en {ruta}")
            valores = [fmt.format(float(f[c])) if f.get(c) else "-" for c, _, fmt in columnas]
            print(f"mp{maximo:<4} {v:>6} " + " ".join(f"{x:>7}" for x in valores))
    print("recall y precisión promedian solo los objetos con predicción (columna obj); "
          "F1 acumula TP/FP/FN sobre todo el conjunto.")


def main() -> None:
    """ Estima, verifica, evalúa, compara y sube las corridas de 1, 2 y 3 planos. """
    parser = argparse.ArgumentParser(description="plane_v02 a 224px/flat con hasta 1, 2 y 3 planos.")
    parser.add_argument("etapa", choices=["chequeo", "todo"])
    parser.add_argument("--renders-root", type=Path, default=Path("/data/pespinoz/renders"))
    parser.add_argument("--objects-root", type=Path, default=Path("/data/pespinoz/objects"))
    parser.add_argument("--jobs", type=int, default=12, help="procesos de S_dat")
    parser.add_argument("--sin-git", action="store_true", help="no sube los resultados")
    args = parser.parse_args()

    chequear(args.renders_root)
    if args.etapa == "chequeo":
        return
    salida = REPO / "resultados_finales" / f"plane_v02_224_{NOMBRE}"
    logs = salida / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    try:
        for maximo in MAXIMOS:
            final = id_salida(maximo)
            print(f"\n=== [{dt.datetime.now():%d/%m %H:%M}] hasta {maximo} planos -> {final}",
                  flush=True)
            correr([PY, "PlaneScoreData/plane_score_data.py", "--renders-root", args.renders_root,
                    "--input-id", ID_ENTRADA, "--output-id", final, "--criterio", "datos",
                    "--size", SIZE, "--lighting", LUZ, "--jobs", args.jobs,
                    "--max-planes", maximo, "--overwrite"], logs / f"est_{final}.log")
            verificar_primer_plano(args.renders_root, maximo)
            correr([PY, "Mapping/evaluate.py", "--renders-root", args.renders_root,
                    "--objects-root", args.objects_root, "--symmetry-type", TIPO, "--sizes", SIZE,
                    "--lightings", LUZ, "--experiment-id", final, "--method", METODO,
                    "--with-reference-metrics"], logs / f"eval_{final}.log")
        correr([PY, "Mapping/compare_results_no_mesh.py", "--renders-root", args.renders_root,
                "--symmetry-type", TIPO, "--sizes", SIZE, "--lightings", LUZ,
                "--experiment-id", *[id_salida(m) for m in MAXIMOS], "--csv-dir", salida,
                "--no-plots", "--total-objects", TOTAL_OBJETOS], logs / f"compare_{TIPO}.log")
        ruta = renombrar_comparacion(salida, TIPO)
        resumen(ruta)
        if not args.sin_git:
            subir(NOMBRE, salida)
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(f"[error] {NOMBRE}: {error}")
    print("\nMultiplano terminado sin errores")


if __name__ == "__main__":
    main()
