"""
diagnose_view_geometry.py
--------------------------
Diagnostico complementario a diagnose_axis_conditioning.py: en vez de medir
que tan bien condicionados quedan los planos de interpretacion CONSTRUIDOS A
PARTIR de los puntos que devolvio Molmo2 (eso ya se descarto como causa --
ver docs/diagnostico_conditioning_axis.md), esto mide algo mas basico y
totalmente independiente de la respuesta de Molmo2: **que tan bien orientado
esta el propio SET DE CAMARAS (Fibonacci sphere sampling,
ImagesGenerator/) respecto al eje de simetria REAL (ground truth) de cada
objeto**, sin usar ninguna prediccion.

====================================================================
LA PREGUNTA Y POR QUE ES DISTINTA DE LO YA DESCARTADO
====================================================================
Cuando una camara mira casi exactamente a lo largo del eje de simetria real
del objeto ("de punta", end-on), el eje se proyecta casi a un punto en esa
imagen: el polo superior y el polo inferior (lo que pide axis_v00/v04/v05/v06)
caen casi en el mismo pixel. En esa vista, CUALQUIER par de puntos que
devuelva Molmo2 -- por buena que sea su localizacion en pixeles -- produce un
plano de interpretacion casi degenerado, porque la vista en si misma no
tiene informacion angular util sobre el eje.

Esto es DISTINTO de lo que midio diagnose_axis_conditioning.py: ese script
media el conditioning del sistema construido con los puntos que Molmo2
efectivamente devolvio (podia estar contaminado por identidad inconsistente
entre vistas). Este script mide el conditioning INTRINSECO del propio
muestreo de camaras contra el eje real, ANTES de que Molmo2 entre en juego
-- si el problema esta aca, ningun prompt nuevo lo va a arreglar, porque el
problema no es que Molmo2 senale mal: es que la vista que le mandamos no
tiene, geometricamente, la informacion que le estamos pidiendo.

Verificacion previa (hecha a mano sobre los .txt de
data/objects/curated_axis_sym_obj/): la direccion del eje GT SI varia
sustancialmente entre objetos (no estan todos alineados al mismo eje del
mundo) -- por lo tanto este es un predictor POR OBJETO valido, no una
constante del dataset.

====================================================================
METRICAS (por objeto x n_views, usando TODAS las vistas enviadas a Molmo2
en ese grupo, con o sin respuesta valida -- es una propiedad de la camara,
no de la respuesta)
====================================================================

- view_axis_angle_deg (por vista): angulo, sign-agnostic en [0,90], entre la
  direccion de vista de la camara (view_forward_direction) y la direccion
  del eje GT. 0 grados = camara mirando exactamente a lo largo del eje
  (maxima degeneracion, "end-on"). 90 grados = camara mirando exactamente
  perpendicular al eje (vista ideal para ver el eje proyectado como una
  linea larga).
- min_view_axis_angle_deg: el peor caso (la vista mas "de punta") entre las
  usadas para ese objeto x n_views.
- frac_degenerate: fraccion de vistas con view_axis_angle_deg por debajo de
  --degenerate-threshold-deg (default 20 grados).
- mean/median_view_axis_angle_deg: resumen general de que tan bien orientado
  estuvo, en promedio, el set de camaras para ESE eje en particular.

====================================================================
JUSTIFICACION / LITERATURA
====================================================================
Es el mismo argumento de conditioning por angulo de vista que
docs/pipeline_sin_malla.md S3.1 ya anticipa como "Riesgo conocido" (camaras
casi coplanares/colineales), pero medido aca contra la unica referencia que
importa geometricamente: el eje REAL, no la nube de normales derivadas de
las respuestas de Molmo2. El argumento de fondo (angulo de vista chico ->
triangulacion mal condicionada) es el mismo de Hartley & Zisserman (2004),
*Multiple View Geometry in Computer Vision*, cap. de triangulacion, ya
citado en docs/diagnostico_conditioning_axis.md S2.1/2.2 -- lo nuevo aca es
CONTRA QUE se mide el angulo (el eje GT, no las normales derivadas de la
respuesta del VLM), lo que aisla si el problema es de la CAMARA (fijo, no
arreglable con mejor prompt) o de la RESPUESTA (arreglable con mejor
prompt/mejor triangulacion).

Usage
-----
    python Mapping/diagnose_view_geometry.py \\
        --renders-root ../data/renders --objects-root ../data/objects \\
        --experiment-id axis_v06_nomesh \\
        --out ../results/diagnostics/axis_v06_view_geometry.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_common.datasets import OBJECTS_SUBDIR
from pipeline_common.naming import exp_filename
from pipeline_common.triangulation import view_forward_direction

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_axis_conditioning import load_angular_errors, pearson, spearman  # noqa: E402
from evaluate import parse_true_label  # noqa: E402

MOLMO_JSON    = "molmo_multiview.json"
MANIFEST_FILE = "manifest.json"
DEFAULT_FOV   = 60.0
EPS           = 1e-9


def sign_agnostic_angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    return float(np.degrees(np.arccos(np.clip(np.abs(np.dot(v1, v2)), 0.0, 1.0))))


def view_axis_angles_for_group(images_sent: list[dict], axis_dir: np.ndarray,
                                fov_deg: float, image_size: int) -> list[float]:
    """Angulo (grados) entre cada camara de este grupo de n_views y el eje GT
    -- usa TODAS las camaras enviadas, sin filtrar por si Molmo respondio."""
    angles = []
    for cam in images_sent:
        d = view_forward_direction(cam["R"], cam["T"], fov_deg, image_size)
        angles.append(sign_agnostic_angle_deg(d, axis_dir))
    return angles


def main() -> None:
    p = argparse.ArgumentParser(
        description="Diagnostico de conditioning del SET DE CAMARAS contra el eje GT "
                     "(independiente de la respuesta de Molmo2). Ver docstring del modulo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--renders-root", required=True)
    p.add_argument("--objects-root", required=True)
    p.add_argument("--experiment-id", required=True,
                   help="Ej: axis_v06_nomesh -- solo se usa para saber que objetos tienen "
                        "angular_error valido y que grupos de n_views existen; el resultado "
                        "de este script NO depende de que puntos devolvio Molmo2.")
    p.add_argument("--sizes", type=int, nargs="+", default=[224])
    p.add_argument("--lightings", type=str, nargs="+", default=["flat"])
    p.add_argument("--degenerate-threshold-deg", type=float, default=20.0,
                   help="Una vista con angulo camara-eje por debajo de este umbral se "
                        "considera 'de punta' / degenerada para ver el eje proyectado.")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    renders_root = Path(args.renders_root)
    objects_root = Path(args.objects_root)
    symmetry_dir = renders_root / "axis_sym"
    objects_dir  = objects_root / OBJECTS_SUBDIR["axis_sym"]

    angular_errors = load_angular_errors(renders_root, args.sizes, args.lightings, args.experiment_id)

    rows = []
    input_file = exp_filename(MOLMO_JSON, args.experiment_id)

    object_dirs = sorted(d for d in symmetry_dir.iterdir() if d.is_dir())
    for obj_dir in object_dirs:
        obj_id = obj_dir.name
        if obj_id not in angular_errors:
            continue

        txt_path = objects_dir / f"{obj_id}.txt"
        if not txt_path.exists():
            continue
        label = parse_true_label(txt_path)
        if label["type"] != "axis" or not label["elements"]:
            continue
        axis_dir = np.array(label["elements"][0]["direction"])

        for size in args.sizes:
            for lighting in args.lightings:
                render_dir = obj_dir / str(size) / lighting
                molmo_path = render_dir / input_file
                if not molmo_path.exists():
                    continue

                manifest_path = render_dir / MANIFEST_FILE
                if manifest_path.exists():
                    with open(manifest_path, encoding="utf-8") as f:
                        manifest = json.load(f)
                    image_size = manifest.get("image_size", size)
                    fov_deg    = manifest.get("fov", DEFAULT_FOV)
                else:
                    image_size = size
                    fov_deg    = DEFAULT_FOV

                with open(molmo_path, encoding="utf-8") as f:
                    molmo_data = json.load(f)

                for nv_key, group in molmo_data.items():
                    if nv_key not in angular_errors[obj_id]:
                        continue
                    images_sent = group.get("images_sent", [])
                    if not images_sent:
                        continue

                    angles = view_axis_angles_for_group(images_sent, axis_dir, fov_deg, image_size)
                    angles = np.array(angles)

                    rows.append({
                        "object_id":                 obj_id,
                        "n_views":                   int(nv_key),
                        "angular_error_deg":         angular_errors[obj_id][nv_key],
                        "min_view_axis_angle_deg":   float(np.min(angles)),
                        "mean_view_axis_angle_deg":  float(np.mean(angles)),
                        "median_view_axis_angle_deg": float(np.median(angles)),
                        "frac_degenerate":           float(np.mean(angles < args.degenerate_threshold_deg)),
                    })

    if not rows:
        print("[error] No se pudo calcular ninguna fila -- revisa --experiment-id y las rutas.")
        sys.exit(1)

    import csv as csv_mod
    fieldnames = list(rows[0].keys())
    out_path = Path(args.out) if args.out else Path(f"view_geometry_{args.experiment_id}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] {len(rows)} filas (objeto x n_views) guardadas en {out_path}\n")

    # ── Correlaciones ────────────────────────────────────────────────────────
    predictors = ["min_view_axis_angle_deg", "mean_view_axis_angle_deg",
                  "median_view_axis_angle_deg", "frac_degenerate"]
    expected_sign = {
        "min_view_axis_angle_deg": "negativa", "mean_view_axis_angle_deg": "negativa",
        "median_view_axis_angle_deg": "negativa", "frac_degenerate": "POSITIVA",
    }

    n_views_present = sorted({r["n_views"] for r in rows})
    print("Correlacion con angular_error_deg (Pearson / Spearman) -- signo esperado entre parentesis:")
    print("(negativa = a mas angulo camara-eje, MENOS error; POSITIVA = a mas fraccion de vistas "
          f"'de punta' (<{args.degenerate_threshold_deg} grados), MAS error)\n")

    header = f"{'predictor':<30}" + "".join(f"  n={nv:<10}" for nv in n_views_present) + "  pooled"
    print(header)
    print("-" * len(header))

    for pred in predictors:
        line = f"{pred + f' ({expected_sign[pred]})':<30}"
        for nv in n_views_present:
            x = np.array([r[pred] for r in rows if r["n_views"] == nv])
            y = np.array([r["angular_error_deg"] for r in rows if r["n_views"] == nv])
            if len(x) < 3:
                line += f"  {'n/a':<10}"
                continue
            pr, sr = pearson(x, y), spearman(x, y)
            line += f"  {pr:+.2f}/{sr:+.2f} "
        x_all = np.array([r[pred] for r in rows])
        y_all = np.array([r["angular_error_deg"] for r in rows])
        pr, sr = pearson(x_all, y_all), spearman(x_all, y_all)
        line += f"  {pr:+.2f}/{sr:+.2f}"
        print(line)

    # ── Comparacion de 2 grupos: tiene >=1 vista degenerada vs. no ────────────
    print(f"\nComparacion directa: objetos CON al menos una vista 'de punta' "
          f"(angulo < {args.degenerate_threshold_deg} grados) vs. SIN ninguna:\n")
    for nv in n_views_present:
        sub = [r for r in rows if r["n_views"] == nv]
        with_deg = [r["angular_error_deg"] for r in sub if r["frac_degenerate"] > 0]
        without  = [r["angular_error_deg"] for r in sub if r["frac_degenerate"] == 0]
        if len(with_deg) < 3 or len(without) < 3:
            print(f"  n_views={nv}: datos insuficientes en algun grupo (con={len(with_deg)}, sin={len(without)})")
            continue
        print(f"  n_views={nv:3d}:  CON vista degenerada -> n={len(with_deg):4d}  "
              f"media={np.mean(with_deg):.2f}  mediana={np.median(with_deg):.2f}   |   "
              f"SIN vista degenerada -> n={len(without):4d}  "
              f"media={np.mean(without):.2f}  mediana={np.median(without):.2f}")

    print(
        "\nLectura: |r| >= 0.3 con el signo esperado, o una diferencia de medias/medianas "
        "notoria en la comparacion de 2 grupos, es evidencia de que la orientacion del set "
        "de camaras (independiente de Molmo2) explica parte del error. Si es asi, la mejora "
        "correcta es re-disenar el muestreo de vistas (excluir/reponderar elevaciones "
        "'de punta' respecto al eje GT) o descartar del ajuste las vistas detectadas como "
        "degeneradas -- NO un prompt nuevo. Si no hay senal (igual que en "
        "diagnose_axis_conditioning.py), se refuerza la hipotesis de inconsistencia de "
        "identidad entre vistas como causa dominante."
    )


if __name__ == "__main__":
    main()
