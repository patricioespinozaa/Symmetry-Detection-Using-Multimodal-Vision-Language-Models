#!/usr/bin/env python3
# Control: con el criterio 'camaras', el código de esta carpeta debe reproducir las predicciones
# ya guardadas por Mapping/estimate_symmetry_no_mesh.py. Solo lee archivos, no escribe nada.
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from plane_score_data import VISTAS_DEFECTO, estimate_plane, leer_grupos

from estimate_symmetry_no_mesh import METHOD_KEY, OUTPUT_FILE  # noqa: E402
from pipeline_common.naming import exp_filename  # noqa: E402

TOLERANCIA = 1e-9


def verificar_objeto(
    object_dir: Path, id_entrada: str, id_referencia: str, size: int, lighting: str,
    vistas: list[int],
) -> tuple[int, int]:
    """ Compara las predicciones recalculadas de un objeto con las guardadas.

    Args:
        * object_dir: carpeta del objeto
        * id_entrada: id de molmo_multiview_<ID>.json
        * id_referencia: id de predicted_symmetry_<ID>.json
        * size: resolución de los renders
        * lighting: iluminación de los renders
        * vistas: grupos de vistas que se comparan

    Returns:
        * tuple[int, int]: grupos iguales y grupos distintos

    """
    ruta = object_dir / exp_filename(OUTPUT_FILE, id_referencia)
    guardadas = {}
    if ruta.exists():
        with open(ruta, encoding="utf-8") as archivo:
            guardadas = json.load(archivo).get("n_views_predictions", {})

    grupos, fov_deg, image_size = leer_grupos(object_dir, id_entrada, size, lighting, vistas)
    iguales, distintos = 0, 0
    for clave, grupo in grupos.items():
        referencia = guardadas.get(clave, {}).get(METHOD_KEY)
        try:
            plano = estimate_plane(grupo.get("points_by_image", {}), grupo.get("images_sent", []),
                                   fov_deg, image_size, "camaras")
        except (ValueError, np.linalg.LinAlgError):
            plano = None
        if plano is None or referencia is None:
            coincide = plano is None and referencia is None
        else:
            coseno = abs(np.dot(plano["normal"], referencia["normal"]))
            distancia = np.linalg.norm(np.subtract(plano["origin"], referencia["origin"]))
            coincide = 1.0 - coseno < TOLERANCIA and distancia < TOLERANCIA
        iguales += coincide
        distintos += not coincide
    return iguales, distintos


def main() -> None:
    """ Revisa una muestra de objetos y termina con código 1 si alguna predicción difiere. """
    parser = argparse.ArgumentParser(description="Control del criterio 'camaras'.")
    parser.add_argument("--renders-root", type=Path, required=True)
    parser.add_argument("--input-id", required=True, help="id de molmo_multiview_<ID>.json")
    parser.add_argument("--reference-id", required=True,
                        help="id de predicted_symmetry_<ID>.json del flujo principal")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--lighting", default="flat")
    parser.add_argument("--vistas", type=int, nargs="+", default=VISTAS_DEFECTO)
    parser.add_argument("--max-objects", type=int, default=30)
    args = parser.parse_args()

    objetos = sorted(d for d in (args.renders_root / "plane_sym").iterdir() if d.is_dir())
    iguales, distintos = 0, 0
    for objeto in tqdm(objetos[:args.max_objects], unit="obj"):
        resultado = verificar_objeto(objeto, args.input_id, args.reference_id, args.size,
                                     args.lighting, args.vistas)
        iguales += resultado[0]
        distintos += resultado[1]
    print(f"Grupos iguales: {iguales}  |  distintos: {distintos}")
    if distintos or not iguales:
        print("[error] El control falló: no uses la comparación hasta revisar la diferencia.")
        sys.exit(1)
    print("[ok] El código reproduce las predicciones del flujo principal.")


if __name__ == "__main__":
    main()
