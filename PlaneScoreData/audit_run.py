#!/usr/bin/env python3
# Audita, solo leyendo, una corrida de post-procesamiento: que use los puntos que corresponden, que
# sus predicciones no vengan de otra corrida y que la evaluación haya leído esas predicciones.
# Termina con código 1 si encuentra algún problema.
#
#   python PlaneScoreData/audit_run.py --renders-root /data/pespinoz/renders \
#       --symmetry-type plane_sym --input-id plane_v05_1_s224_flat_ndcfix_mp1_nomesh \
#       --expected-input molmo_multiview_plane_v05_1_ndcfix_mp1_nomesh.json \
#       --output-id plane_v05_1_s224_flat_ndcfix_sdat_mp1_nomesh --estimador sdat \
#       --size 224 --lighting flat --desde "2026-11-05 00:00"
import argparse
import datetime as dt
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from plane_score_data import REPO, VISTAS_DEFECTO, estimate_plane, leer_grupos

from estimate_symmetry_no_mesh import (  # noqa: E402
    METHOD_KEY,
    MOLMO_JSON,
    OUTPUT_FILE,
    estimate_axis_no_mesh,
    estimate_plane_no_mesh,
)
from pipeline_common.naming import exp_filename  # noqa: E402
from pipeline_common.triangulation import get_point_by_obj_id  # noqa: E402

sys.path.insert(0, str(REPO / "Pipeline_Experiments"))
from triangulation_variants import get_variant  # noqa: E402

ESTIMADORES = {"axis_sym": ("base", "expE"), "plane_sym": ("base", "sdat", "camaras")}
TOLERANCIA = 1e-9
TODOS_LOS_GRUPOS = range(1000)
EJEMPLOS = 3


def recalcular(tipo: str, estimador: str, grupo: dict, fov: float, size: int) -> dict | None:
    """ Recalcula la predicción de un grupo de vistas con el estimador auditado.

    Args:
        * tipo: 'axis_sym' o 'plane_sym'
        * estimador: clave de ESTIMADORES[tipo]
        * grupo: entrada del JSON de puntos
        * fov: campo de visión en grados
        * size: lado de la imagen en píxeles

    Returns:
        * dict | None: predicción, o None si el estimador no pudo producir una

    """
    puntos, camaras = grupo.get("points_by_image", {}), grupo.get("images_sent", [])
    try:
        if tipo == "axis_sym":
            if estimador == "expE":
                return get_variant("expE").estimate_axis(puntos, camaras, fov, size)
            return estimate_axis_no_mesh(puntos, camaras, fov, size)
        if estimador == "base":
            return estimate_plane_no_mesh(puntos, camaras, fov, size)
        criterio = "datos" if estimador == "sdat" else "camaras"
        return estimate_plane(puntos, camaras, fov, size, criterio)
    except (ValueError, np.linalg.LinAlgError):
        return None


def coinciden(tipo: str, recalculada: dict | None, guardada: dict | None) -> bool:
    """ Indica si la predicción recalculada y la guardada son la misma.

    Args:
        * tipo: 'axis_sym' o 'plane_sym'
        * recalculada: salida de recalcular
        * guardada: predicción del archivo, o None si el grupo no existe

    Returns:
        * bool: True si ambas faltan o coinciden en dirección y origen

    """
    if recalculada is None or guardada is None:
        return recalculada is None and guardada is None
    clave = "direction" if tipo == "axis_sym" else "normal"
    coseno = abs(np.dot(recalculada[clave], guardada[clave]))
    distancia = np.linalg.norm(np.subtract(recalculada["origin"], guardada["origin"]))
    return bool(1.0 - coseno < TOLERANCIA and distancia < TOLERANCIA)


def estimable(tipo: str, grupo: dict) -> bool:
    """ Indica si el grupo tiene las vistas mínimas para que exista una predicción.

    Args:
        * tipo: 'axis_sym' o 'plane_sym'
        * grupo: entrada del JSON de puntos

    Returns:
        * bool: True si hay suficientes vistas con puntos

    """
    puntos = grupo.get("points_by_image", {}).values()
    if tipo == "axis_sym":
        return sum(len(p) >= 2 for p in puntos) >= 2
    validas = sum(get_point_by_obj_id(p, 1) is not None and get_point_by_obj_id(p, 2) is not None
                  for p in puntos)
    return validas >= 4


def procedencia_correcta(documento: dict, tipo: str, estimador: str, grupos: dict) -> bool:
    """ Revisa las marcas que deja cada estimador en su archivo de salida.

    Args:
        * documento: predicted_symmetry_<ID>.json
        * tipo: 'axis_sym' o 'plane_sym'
        * estimador: clave de ESTIMADORES[tipo]
        * grupos: n_views_predictions del documento

    Returns:
        * bool: True si las marcas corresponden al estimador auditado

    """
    if documento.get("object_id") is None or documento.get("symmetry_type") != tipo:
        return False
    if estimador == "expE":
        return documento.get("variant") == "expE"
    if estimador in ("sdat", "camaras"):
        esperado = "datos" if estimador == "sdat" else "camaras"
        return all(g.get(METHOD_KEY, {}).get("plane_score") == esperado for g in grupos.values())
    return "variant" not in documento and all(
        "plane_score" not in g.get(METHOD_KEY, {}) for g in grupos.values())


def auditar(args: argparse.Namespace) -> tuple[dict[str, list[str]], dict]:
    """ Revisa todos los objetos de una corrida y recalcula una muestra.

    Args:
        * args: argumentos de la línea de comandos

    Returns:
        * tuple[dict[str, list[str]], dict]: problemas por tipo y resumen de la corrida

    """
    tipo = args.symmetry_type
    raiz = args.renders_root / tipo
    objetos = sorted(d for d in raiz.iterdir() if d.is_dir())
    archivo_entrada = exp_filename(MOLMO_JSON, args.input_id)
    archivo_salida = exp_filename(OUTPUT_FILE, args.output_id)
    desde = dt.datetime.fromisoformat(args.desde).timestamp() if args.desde else None
    paso = max(1, len(objetos) // args.muestra) if args.muestra else 1

    problemas: dict[str, list[str]] = {}

    def anotar(clase: str, detalle: str) -> None:
        """ Registra un problema encontrado. """
        problemas.setdefault(clase, []).append(detalle)

    enlaces, cobertura, guardadas = Counter(), Counter(), {}
    n_entradas, n_salidas, n_recalculados, ultima = 0, 0, 0, 0.0
    for i, objeto in enumerate(objetos):
        render = objeto / str(args.size) / args.lighting
        entrada = render / archivo_entrada
        salida = objeto / archivo_salida
        hay_entrada = entrada.exists()
        n_entradas += hay_entrada

        if hay_entrada:
            if entrada.is_symlink():
                enlaces[f"{os.readlink(entrada)} -> {Path(os.path.realpath(entrada)).name}"] += 1
                if args.expected_input and os.readlink(entrada) != args.expected_input:
                    anotar("Entrada distinta de la esperada", objeto.name)
            elif args.expected_input and entrada.name != args.expected_input:
                anotar("Entrada distinta de la esperada", objeto.name)

        if not salida.exists():
            if not hay_entrada:
                continue
            documento, predichos = None, {}
        else:
            n_salidas += 1
            with open(salida, encoding="utf-8") as archivo:
                documento = json.load(archivo)
            predichos = documento.get("n_views_predictions", {})
            ultima = max(ultima, salida.stat().st_mtime)
            guardadas[objeto.name] = predichos
            for clave in predichos:
                cobertura[int(clave)] += 1
            if not hay_entrada:
                anotar("Predicción sin archivo de puntos en esta configuración", objeto.name)
                continue
            if not procedencia_correcta(documento, tipo, args.estimador, predichos):
                anotar("Predicción sin las marcas del estimador auditado", objeto.name)
            if desde is not None and salida.stat().st_mtime < desde:
                anotar("Predicción anterior a la fecha indicada", objeto.name)
            if salida.stat().st_mtime < os.stat(entrada).st_mtime:
                anotar("Predicción más antigua que sus puntos", objeto.name)

        todos, fov, size = leer_grupos(objeto, args.input_id, args.size, args.lighting,
                                       TODOS_LOS_GRUPOS)
        for clave in predichos:
            if clave not in todos:
                anotar("Predicción de un grupo de vistas que no está en los puntos",
                       f"{objeto.name}/{clave}")
            elif not estimable(tipo, todos[clave]):
                anotar("Predicción de un grupo sin vistas suficientes", f"{objeto.name}/{clave}")

        if i % paso == 0:
            n_recalculados += 1
            for clave, grupo in todos.items():
                if int(clave) not in args.vistas:
                    continue
                nueva = recalcular(tipo, args.estimador, grupo, fov, size)
                if not coinciden(tipo, nueva, predichos.get(clave, {}).get(METHOD_KEY)):
                    anotar("Predicción distinta de la recalculada", f"{objeto.name}/{clave}")

    ruta_eval = raiz / (f"eval_s{args.size}_{args.lighting}_{args.output_id}_"
                        f"{METHOD_KEY}_results.json")
    evaluados = Counter()
    if not ruta_eval.exists():
        anotar("Falta la evaluación", ruta_eval.name)
    else:
        if ruta_eval.stat().st_mtime < ultima:
            anotar("Evaluación más antigua que alguna predicción", ruta_eval.name)
        with open(ruta_eval, encoding="utf-8") as archivo:
            resultados = json.load(archivo)["objects"]
        for objeto, predichos in guardadas.items():
            por_vistas = resultados.get(objeto) or {}
            for clave in predichos:
                if por_vistas.get(clave, {}).get("status") == "ok":
                    evaluados[int(clave)] += 1
                else:
                    anotar("Predicción no evaluada correctamente", f"{objeto}/{clave}")
        for objeto, por_vistas in resultados.items():
            for clave in por_vistas or {}:
                if clave not in guardadas.get(objeto, {}):
                    anotar("Evaluación sin predicción actual", f"{objeto}/{clave}")

    resumen = {
        "objetos": len(objetos), "entradas": n_entradas, "salidas": n_salidas,
        "recalculados": n_recalculados, "enlaces": enlaces,
        "cobertura": cobertura, "evaluados": evaluados,
    }
    return problemas, resumen


def main() -> None:
    """ Imprime el informe y termina con código 1 si hay problemas. """
    parser = argparse.ArgumentParser(description="Audita una corrida de post-procesamiento.")
    parser.add_argument("--renders-root", type=Path, required=True)
    parser.add_argument("--symmetry-type", choices=list(ESTIMADORES), required=True)
    parser.add_argument("--input-id", required=True, help="id de molmo_multiview_<ID>.json")
    parser.add_argument("--expected-input", default=None,
                        help="archivo al que debe apuntar la entrada, si es un enlace")
    parser.add_argument("--output-id", required=True, help="id de predicted_symmetry_<ID>.json")
    parser.add_argument("--estimador", required=True, choices=["base", "expE", "sdat", "camaras"])
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--lighting", required=True)
    parser.add_argument("--desde", default=None,
                        help="fecha de inicio de la corrida, por ejemplo '2026-11-05 00:00'")
    parser.add_argument("--vistas", type=int, nargs="+", default=VISTAS_DEFECTO)
    parser.add_argument("--muestra", type=int, default=30,
                        help="objetos que se recalculan; 0 = todos")
    args = parser.parse_args()
    if args.estimador not in ESTIMADORES[args.symmetry_type]:
        parser.error(f"{args.estimador} no aplica a {args.symmetry_type}")

    problemas, resumen = auditar(args)
    print(f"=== {args.output_id} ({args.size}px, {args.lighting}, {args.estimador})")
    print(f"Objetos {resumen['objetos']} | con puntos {resumen['entradas']} | con predicción "
          f"{resumen['salidas']} | recalculados {resumen['recalculados']}")
    for enlace, n in resumen["enlaces"].most_common():
        print(f"Entrada: {enlace} ({n})")
    print(f"Predicciones por vistas: {sorted(resumen['cobertura'].items())}")
    print(f"Evaluadas por vistas   : {sorted(resumen['evaluados'].items())}")
    if not problemas:
        print("[ok] Sin problemas.")
        return
    for clase, detalles in problemas.items():
        ejemplos = ", ".join(detalles[:EJEMPLOS])
        print(f"[PROBLEMA] {clase}: {len(detalles)} (por ejemplo: {ejemplos})")
    sys.exit(1)


if __name__ == "__main__":
    main()
