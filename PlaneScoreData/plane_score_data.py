#!/usr/bin/env python3
# Estimador de plano sin malla con un criterio alternativo para elegir la normal candidata:
# 'datos' (S_dat, docs/hallazgo_seleccion_normal_plano.md) o 'camaras' (score_normal del flujo
# principal, usado como control). No modifica Mapping/: solo importa de ahí y escribe un
# predicted_symmetry_<ID>.json nuevo, legible por Mapping/evaluate.py.
import argparse
import json
import os
import sys
from collections import Counter
from functools import partial
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "Mapping"))
from estimate_symmetry_no_mesh import (  # noqa: E402
    DEFAULT_FOV,
    EDGE_ON_THRESH_DEFAULT,
    MANIFEST_FILE,
    METHOD_KEY,
    MOLMO_JSON,
    OUTPUT_FILE,
    _line_from_view_pair,
)
from pipeline_common.naming import exp_filename  # noqa: E402
from pipeline_common.triangulation import (  # noqa: E402
    get_point_by_obj_id,
    interpretation_plane_normal,
    ray_dir_for_point,
    view_forward_direction,
)

CRITERIOS = ("datos", "camaras")
VISTAS_DEFECTO = [6, 14, 26]
DIVISOR_FRACCION = 3  # "el tercio de vistas": fijado antes de ver los 850 objetos, no se ajusta


def normales_interpretacion(
    points_by_image: dict, images_sent: list[dict], view_idxs: list[int],
    fov_deg: float, image_size: int,
) -> np.ndarray:
    """ Calcula la normal del plano de interpretación de cada vista válida.

    Args:
        * points_by_image: puntos por índice de vista
        * images_sent: cámaras por índice de vista
        * view_idxs: vistas con los puntos obj_id 1 y 2
        * fov_deg: campo de visión en grados
        * image_size: lado de la imagen en píxeles

    Returns:
        * np.ndarray: matriz (v, 3) con una normal unitaria por vista no degenerada

    """
    normales = []
    for idx in view_idxs:
        pts = points_by_image[str(idx)]
        p1, p2 = get_point_by_obj_id(pts, 1), get_point_by_obj_id(pts, 2)
        cam = images_sent[idx]
        _, d1 = ray_dir_for_point(p1["x"], p1["y"], cam["R"], cam["T"], fov_deg, image_size)
        _, d2 = ray_dir_for_point(p2["x"], p2["y"], cam["R"], cam["T"], fov_deg, image_size)
        normal = interpretation_plane_normal(d1, d2)
        if normal is not None:
            normales.append(normal)
    return np.asarray(normales, dtype=np.float64).reshape(-1, 3)


def score_datos(candidatas: np.ndarray, normales: np.ndarray) -> np.ndarray:
    """ Calcula S_dat(c) = media de (1 - |n_i · c|) sobre el tercio de vistas con menor valor.

    Args:
        * candidatas: matriz (m, 3) de normales candidatas unitarias
        * normales: matriz (v, 3) de normales de interpretación unitarias

    Returns:
        * np.ndarray: vector (m,) de puntajes; menor es mejor

    """
    valores = 1.0 - np.abs(candidatas @ normales.T)
    k = max(2, normales.shape[0] // DIVISOR_FRACCION)
    return np.partition(valores, k - 1, axis=1)[:, :k].mean(axis=1)


def elegir_candidata(
    candidatas: list[np.ndarray], criterio: str, view_dirs: dict[int, np.ndarray],
    normales: np.ndarray,
) -> np.ndarray:
    """ Elige la normal candidata según el criterio.

    Args:
        * candidatas: normales candidatas, en el orden en que se generaron
        * criterio: 'datos' o 'camaras'
        * view_dirs: eje óptico de cada vista válida
        * normales: normales de interpretación de las vistas válidas

    Returns:
        * np.ndarray: candidata elegida; en un empate, la primera generada

    """
    if criterio == "datos":
        if normales.shape[0] < 2:
            raise ValueError("menos de 2 normales de interpretación válidas")
        return candidatas[int(np.argmin(score_datos(np.asarray(candidatas), normales)))]

    def score_normal(candidata: np.ndarray) -> float:
        """ Replica score_normal de estimate_plane_no_mesh. """
        valores = sorted(abs(np.dot(d, candidata)) for d in view_dirs.values())
        return float(np.mean(valores[:max(2, len(valores) // 2)]))

    return min(candidatas, key=score_normal)


def estimate_plane(
    points_by_image: dict, images_sent: list[dict], fov_deg: float, image_size: int,
    criterio: str, edge_on_thresh: float = EDGE_ON_THRESH_DEFAULT,
) -> dict:
    """ Estima un plano con los mismos pasos de estimate_plane_no_mesh, salvo la selección.

    Args:
        * points_by_image: puntos por índice de vista
        * images_sent: cámaras por índice de vista
        * fov_deg: campo de visión en grados
        * image_size: lado de la imagen en píxeles
        * criterio: 'datos' o 'camaras'
        * edge_on_thresh: |cos| bajo el cual una vista cuenta como de canto

    Returns:
        * dict: normal, origin, n_views_used, n_candidates y good_views

    """
    view_idxs = sorted(
        int(k) for k, pts in points_by_image.items()
        if get_point_by_obj_id(pts, 1) is not None and get_point_by_obj_id(pts, 2) is not None
    )
    if len(view_idxs) < 4:
        raise ValueError(f"se necesitan >=4 vistas válidas, hay {len(view_idxs)}")

    pair_lines = []
    for i, j in combinations(view_idxs, 2):
        linea = _line_from_view_pair(i, j, points_by_image, images_sent, fov_deg, image_size)
        if linea is not None:
            pair_lines.append(((i, j), linea[0], linea[1]))
    if len(pair_lines) < 2:
        raise ValueError("no hay suficientes rectas para generar candidatas")

    view_dirs = {
        idx: view_forward_direction(images_sent[idx]["R"], images_sent[idx]["T"], fov_deg,
                                    image_size)
        for idx in view_idxs
    }

    candidatas = []
    for a in range(len(pair_lines)):
        par_a, _, dir_a = pair_lines[a]
        for b in range(a + 1, len(pair_lines)):
            par_b, _, dir_b = pair_lines[b]
            if set(par_a) & set(par_b):
                continue
            normal = np.cross(dir_a, dir_b)
            norma = np.linalg.norm(normal)
            if norma >= 1e-6:
                candidatas.append(normal / norma)
    if not candidatas:
        raise ValueError("no se pudo generar ninguna candidata")

    normales = (normales_interpretacion(points_by_image, images_sent, view_idxs, fov_deg,
                                        image_size)
                if criterio == "datos" else np.empty((0, 3)))
    elegida = elegir_candidata(candidatas, criterio, view_dirs, normales)

    good_views = [i for i in view_idxs if abs(np.dot(view_dirs[i], elegida)) < edge_on_thresh]
    refit_dirs = [d for (par, _, d) in pair_lines if set(par).issubset(good_views)]
    refit_points = [p for (par, p, _) in pair_lines if set(par).issubset(good_views)]
    if len(refit_dirs) >= 2:
        _, _, vt = np.linalg.svd(np.asarray(refit_dirs))
        refinada = vt[-1] / np.linalg.norm(vt[-1])
        usados = refit_points
    else:
        refinada = elegida
        usados = [p for (_, p, _) in pair_lines]

    return {
        "normal": refinada.tolist(),
        "origin": np.mean(usados, axis=0).tolist(),
        "n_views_used": len(good_views),
        "n_candidates": len(candidatas),
        "good_views": good_views,
    }


def leer_grupos(
    object_dir: Path, id_entrada: str, size: int, lighting: str, vistas: list[int],
) -> tuple[dict, float, int]:
    """ Lee los grupos de vistas de Molmo2 y los parámetros de cámara de un objeto.

    Args:
        * object_dir: carpeta del objeto
        * id_entrada: id de molmo_multiview_<ID>.json
        * size: resolución de los renders
        * lighting: iluminación de los renders
        * vistas: grupos de vistas que se conservan

    Returns:
        * tuple[dict, float, int]: grupos por n_views, FOV y lado de la imagen

    """
    render_dir = object_dir / str(size) / lighting
    entrada = render_dir / exp_filename(MOLMO_JSON, id_entrada)
    if not entrada.exists():
        return {}, DEFAULT_FOV, size
    fov_deg, image_size = DEFAULT_FOV, size
    manifest = render_dir / MANIFEST_FILE
    if manifest.exists():
        with open(manifest, encoding="utf-8") as archivo:
            datos = json.load(archivo)
        image_size = datos.get("image_size", size)
        fov_deg = datos.get("fov", DEFAULT_FOV)
    with open(entrada, encoding="utf-8") as archivo:
        grupos = json.load(archivo)
    return {k: g for k, g in grupos.items() if int(k) in vistas}, fov_deg, image_size


def procesar_objeto(
    object_dir: Path, id_entrada: str, id_salida: str, size: int, lighting: str,
    criterio: str, vistas: list[int], overwrite: bool,
) -> str:
    """ Estima el plano de cada grupo de vistas y escribe predicted_symmetry_<id_salida>.json.

    Args:
        * object_dir: carpeta del objeto
        * id_entrada: id de molmo_multiview_<ID>.json
        * id_salida: id de predicted_symmetry_<ID>.json
        * size: resolución de los renders
        * lighting: iluminación de los renders
        * criterio: 'datos' o 'camaras'
        * vistas: grupos de vistas que se procesan
        * overwrite: recalcula aunque la salida ya exista

    Returns:
        * str: 'omitido', 'sin_prediccion' o 'escrito'

    """
    salida = object_dir / exp_filename(OUTPUT_FILE, id_salida)
    if salida.exists() and not overwrite:
        return "omitido"

    grupos, fov_deg, image_size = leer_grupos(object_dir, id_entrada, size, lighting, vistas)
    predicciones = {}
    for clave, grupo in grupos.items():
        points_by_image = grupo.get("points_by_image", {})
        if not points_by_image:
            continue
        try:
            plano = estimate_plane(points_by_image, grupo.get("images_sent", []), fov_deg,
                                   image_size, criterio)
        except (ValueError, np.linalg.LinAlgError):
            continue
        predicciones[clave] = {
            "n_points_raw": plano["n_views_used"],
            "n_points_fit": plano["n_views_used"],
            METHOD_KEY: {
                "normal": plano["normal"], "origin": plano["origin"],
                "n_points": plano["n_views_used"], "n_views_used": plano["n_views_used"],
                "n_candidates": plano["n_candidates"], "plane_score": criterio,
                "n_inliers": None, "sde": None, "accepted": None,
            },
        }
    if not predicciones:
        return "sin_prediccion"

    documento = {
        "object_id": object_dir.name, "symmetry_type": "plane_sym", "point_mode": METHOD_KEY,
        "clustering_method": "none", "hdbscan_min_samples": None,
        "n_views_predictions": predicciones,
    }
    # Se escribe en un temporal: una corrida interrumpida no debe dejar un JSON vacío o parcial.
    temporal = salida.with_name(salida.name + f".tmp{os.getpid()}")
    with open(temporal, "w", encoding="utf-8") as archivo:
        json.dump(documento, archivo, indent=2)
    os.replace(temporal, salida)
    return "escrito"


def main() -> None:
    """ Procesa en paralelo todos los objetos planares con el criterio pedido. """
    parser = argparse.ArgumentParser(description="Plano sin malla con selección S_dat.")
    parser.add_argument("--renders-root", type=Path, required=True)
    parser.add_argument("--input-id", required=True, help="id de molmo_multiview_<ID>.json")
    parser.add_argument("--output-id", required=True, help="id de predicted_symmetry_<ID>.json")
    parser.add_argument("--criterio", choices=CRITERIOS, default="datos")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--lighting", default="flat")
    parser.add_argument("--vistas", type=int, nargs="+", default=VISTAS_DEFECTO)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    objetos = sorted(d for d in (args.renders_root / "plane_sym").iterdir() if d.is_dir())
    if args.max_objects:
        objetos = objetos[:args.max_objects]
    print(f"Objetos   : {len(objetos)}")
    print(f"Entrada   : {exp_filename(MOLMO_JSON, args.input_id)}")
    print(f"Salida    : {exp_filename(OUTPUT_FILE, args.output_id)}")
    print(f"Criterio  : {args.criterio}  ({args.size}px, {args.lighting}, vistas {args.vistas})")

    tarea = partial(_procesar_con_args, args=args)
    with Pool(args.jobs) as pool:
        estados = Counter(tqdm(pool.imap_unordered(tarea, objetos), total=len(objetos),
                               unit="obj", dynamic_ncols=True))
    print(f"Estados   : {dict(estados)}")


def _procesar_con_args(object_dir: Path, args: argparse.Namespace) -> str:
    """ Adapta procesar_objeto a los argumentos de la línea de comandos. """
    return procesar_objeto(object_dir, args.input_id, args.output_id, args.size, args.lighting,
                           args.criterio, args.vistas, args.overwrite)


if __name__ == "__main__":
    main()
