#!/usr/bin/env python3
# Diagnóstico de puntos repetidos entre vistas: ¿el VLM devuelve los mismos puntos en vistas distintas
# del mismo objeto? Las vistas están repartidas en la esfera de Fibonacci, así que la proyección real
# del eje o del plano cambia entre vistas; que varias vistas reciban exactamente los mismos puntos
# indica que el modelo copia la respuesta entre imágenes en vez de mirar cada una.
#
# Compara el envío multiimagen (una llamada con todas las vistas) con el envío de una vista por
# llamada, que sirve de control: ahí una repetición solo puede venir del sesgo del prompt (por ejemplo
# marcar siempre (500, 200) y (500, 800)), no de copiar entre imágenes. El modo se lee del campo
# prompt_mode de cada grupo del JSON: auto/multi con más de una vista = "multi", single = "single".
#
# Dos vistas son "idénticas" si tienen la misma cantidad de puntos y la distancia de Hausdorff entre
# sus conjuntos de puntos es 0 (mismas coordenadas, sin importar el obj_id), y "cercanas" si esa
# distancia es <= --tolerancia (unidades 0-1000 de Molmo). Por objeto y número de vistas reporta:
#   - frac_pares_identicos / frac_pares_cercanos: pares de vistas idénticas / cercanas sobre el total;
#   - frac_vistas_repetidas: vistas con puntos idénticos a los de al menos otra vista;
#   - grupo_max: mayor grupo de vistas con exactamente los mismos puntos, como fracción de las vistas
#     con puntos (13 de 26 vistas iguales = 0.5).
# Solo lee los molmo_multiview_<ID>.json existentes; no usa GPU. Las corridas que apuntan al mismo
# archivo (enlaces simbólicos de alias) se cuentan una sola vez.
#
#   python3 Pipeline_Experiments/diagnostics/diagnose_repeated_points.py --all \
#       --out-summary resultados_finales/diagnosticos/repetidos_resumen.csv
#   python3 Pipeline_Experiments/diagnostics/diagnose_repeated_points.py --tipo plane_sym \
#       --experiment-id plane_v02 plane_v02_single --out-summary /tmp/repetidos.csv
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import discover_experiment_ids  # noqa: E402

VISTAS = ("6", "14", "26")
TOLERANCIA = 5.0     # unidades 0-1000 de Molmo, ~1 px a 224 px


def modo_envio(grupo: dict, n_vistas: int) -> str:
    """ 'multi' si las vistas se enviaron juntas, 'single' si una por llamada (según prompt_mode). """
    modo = grupo.get("prompt_mode")
    if modo is None:
        return "desconocido"
    if modo == "single" or n_vistas == 1:
        return "single"
    return "multi" if modo in ("auto", "multi", "global") else modo


def hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    """ Distancia de Hausdorff entre dos conjuntos de puntos 2D. """
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


def grupo_mayor(n: int, pares: list[tuple[int, int]]) -> int:
    """ Tamaño de la mayor componente conexa de n vistas unidas por los pares dados (union-find). """
    padre = list(range(n))

    def raiz(i: int) -> int:
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i

    for i, j in pares:
        padre[raiz(i)] = raiz(j)
    tamanos = defaultdict(int)
    for i in range(n):
        tamanos[raiz(i)] += 1
    return max(tamanos.values()) if tamanos else 0


def medir_grupo(grupo: dict, tolerancia: float) -> dict | None:
    """ Repeticiones de puntos entre las vistas de un grupo (objeto x número de vistas).

    Args:
        * grupo: grupo de molmo_multiview con points_by_image
        * tolerancia: distancia de Hausdorff máxima para contar dos vistas como cercanas

    Returns:
        * dict | None: conteos del grupo, o None si hay menos de 2 vistas con puntos

    """
    vistas = [np.array([[p["x"], p["y"]] for p in pts], dtype=float)
              for _, pts in sorted(grupo.get("points_by_image", {}).items(), key=lambda kv: int(kv[0])) if pts]
    n = len(vistas)
    if n < 2:
        return None
    identicos, cercanos = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if len(vistas[i]) != len(vistas[j]):
                continue
            h = hausdorff(vistas[i], vistas[j])
            if h == 0:
                identicos.append((i, j))
            if h <= tolerancia:
                cercanos.append((i, j))
    repetidas = {k for par in identicos for k in par}
    return {"n_vistas_con_puntos": n, "n_pares": n * (n - 1) // 2,
            "n_pares_identicos": len(identicos), "n_pares_cercanos": len(cercanos),
            "n_vistas_repetidas": len(repetidas), "grupo_max": grupo_mayor(n, identicos)}


def diagnosticar(renders_root: Path, tipo: str, experiment_id: str, size: int, luz: str,
                 tolerancia: float) -> list[dict]:
    """ Filas por (objeto, número de vistas) de un experimento. """
    filas = []
    for carpeta in tqdm(sorted((renders_root / tipo).glob(f"*/{size}/{luz}")), unit="obj",
                        desc=f"{tipo}/{experiment_id}", dynamic_ncols=True):
        ruta = carpeta / f"molmo_multiview_{experiment_id}.json"
        if not ruta.exists():
            continue
        try:
            datos = json.loads(ruta.read_text(encoding="utf-8"))
        except ValueError:
            continue
        for v in VISTAS:
            grupo = datos.get(v)
            if not grupo:
                continue
            medida = medir_grupo(grupo, tolerancia)
            if medida is None:
                continue
            filas.append({"experiment_id": experiment_id, "tipo": tipo, "object_id": carpeta.parents[1].name,
                          "n_views": int(v), "modo": modo_envio(grupo, int(v)), **medida})
    return filas


def resumir(filas: list[dict]) -> list[dict]:
    """ Resumen por (tipo, experimento, modo, número de vistas). """
    grupos = defaultdict(list)
    for f in filas:
        grupos[(f["tipo"], f["experiment_id"], f["modo"], f["n_views"])].append(f)
    resumen = []
    for (tipo, exp, modo, v), fs in sorted(grupos.items()):
        pares = sum(f["n_pares"] for f in fs)
        vistas = sum(f["n_vistas_con_puntos"] for f in fs)
        frac_max = np.array([f["grupo_max"] / f["n_vistas_con_puntos"] for f in fs])
        resumen.append({
            "tipo": tipo, "experiment_id": exp, "modo": modo, "n_views": v, "n_objects": len(fs),
            "frac_pares_identicos": round(sum(f["n_pares_identicos"] for f in fs) / pares, 4),
            "frac_pares_cercanos": round(sum(f["n_pares_cercanos"] for f in fs) / pares, 4),
            "frac_vistas_repetidas": round(sum(f["n_vistas_repetidas"] for f in fs) / vistas, 4),
            "grupo_max_medio": round(float(frac_max.mean()), 4),
            "frac_objetos_mitad_repetida": round(float(np.mean(frac_max >= 0.5)), 4),
        })
    return resumen


def mismo_archivo(renders_root: Path, tipo: str, experiment_id: str, size: int, luz: str) -> str | None:
    """ Archivo real (resuelto) de la primera carpeta que tiene el experimento, para descartar alias. """
    for carpeta in sorted((renders_root / tipo).glob(f"*/{size}/{luz}")):
        ruta = carpeta / f"molmo_multiview_{experiment_id}.json"
        if ruta.exists():
            return str(ruta.resolve())
    return None


def escribir(ruta: Path, filas: list[dict]) -> None:
    """ CSV con las columnas de la primera fila. """
    if not filas:
        return
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)


def main() -> None:
    """ Recorre los experimentos pedidos (o todos) y resume las repeticiones por modo de envío. """
    p = argparse.ArgumentParser(description="Puntos repetidos entre vistas, por modo de envío.")
    p.add_argument("--renders-root", type=Path, default=Path("/data/pespinoz/renders"))
    p.add_argument("--tipo", choices=["axis_sym", "plane_sym", "ambos"], default="ambos")
    p.add_argument("--experiment-id", nargs="+", default=None, help="ids de molmo_multiview_<ID>.json")
    p.add_argument("--all", action="store_true", help="todos los ids encontrados en los renders")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--lighting", default="flat")
    p.add_argument("--tolerancia", type=float, default=TOLERANCIA)
    p.add_argument("--out-summary", type=Path, required=True)
    p.add_argument("--out-detail", type=Path, default=None, help="CSV por objeto (opcional, pesado)")
    args = p.parse_args()
    if not args.experiment_id and not args.all:
        sys.exit("[error] pasa --experiment-id <id> ... o --all")

    tipos = ["axis_sym", "plane_sym"] if args.tipo == "ambos" else [args.tipo]
    detalle = []
    for tipo in tipos:
        ids = discover_experiment_ids(args.renders_root, tipo) if args.all else args.experiment_id
        vistos, elegidos = set(), []
        for exp in ids:
            real = mismo_archivo(args.renders_root, tipo, exp, args.size, args.lighting)
            if real is None or real in vistos:      # no existe a este tamaño/luz, o es alias de otro
                continue
            vistos.add(real)
            elegidos.append(exp)
        print(f"{tipo}: {len(elegidos)} experimentos a {args.size}px/{args.lighting}", flush=True)
        for exp in elegidos:
            detalle += diagnosticar(args.renders_root, tipo, exp, args.size, args.lighting, args.tolerancia)

    resumen = resumir(detalle)
    escribir(args.out_summary, resumen)
    if args.out_detail:
        escribir(args.out_detail, detalle)
    print(f"\n{'tipo':9} {'experimento':48} {'modo':7} {'vistas':>6} {'obj':>4} {'pares id.':>9} "
          f"{'pares cerc.':>11} {'vistas rep.':>11} {'grupo máx.':>10}")
    for r in resumen:
        print(f"{r['tipo']:9} {r['experiment_id'][:48]:48} {r['modo']:7} {r['n_views']:>6} {r['n_objects']:>4} "
              f"{r['frac_pares_identicos']:9.3f} {r['frac_pares_cercanos']:11.3f} "
              f"{r['frac_vistas_repetidas']:11.3f} {r['grupo_max_medio']:10.3f}")
    print(f"\nResumen: {args.out_summary}")


if __name__ == "__main__":
    main()
