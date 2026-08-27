"""
diagnose_point_localization.py
--------------------------------
Diagnostico complementario a diagnose_axis_conditioning.py / diagnose_view_geometry.py:
en vez de medir conditioning geometrico, mide algo mas basico y previo a
cualquier triangulacion -- **cuantos de los puntos 2D que devuelve Molmo2
caen efectivamente sobre el objeto renderizado (vs. sobre el fondo)**, y
cuantas vistas alcanzan siquiera el minimo de 2 puntos que el pipeline sin
malla necesita para construir un plano de interpretacion.

====================================================================
POR QUE ESTO Y NO OTRA COSA
====================================================================
Los dos diagnosticos anteriores (conditioning de las normales, angulo
camara-eje) midieron la CALIDAD GEOMETRICA de la triangulacion dado un
conjunto de puntos, y ninguno de los dos correlaciono con el error angular
(ver docs/diagnostico_conditioning_axis.md). Antes de seguir buscando
explicaciones mas sofisticadas (inconsistencia de identidad entre
estructuras validas), vale la pena descartar algo mas literal y barato de
chequear: que una fraccion no trivial de los puntos que Molmo2 devuelve
**ni siquiera caiga sobre el objeto** -- alucinaciones de coordenadas sobre
el fondo, sin filtro alguno en el pipeline sin malla (a diferencia del
pipeline con malla, donde `cast_ray` descarta implicitamente los puntos
cuyo rayo no interseca la malla real -- ver Mapping/map_to_3d.py).

Fondo confirmado (ImagesGenerator/export_fibonacci_views.py): PyTorch3D
`HardFlatShader` sin `BlendParams` explicito usa fondo blanco puro por
default, (1.0, 1.0, 1.0) -> RGB (255,255,255) tras la conversion a uint8.
Es CONSTANTE para todo objeto, tamano e iluminacion (flat/brighter/darker)
-- el objeto se sombrea en gris via `TexturesVertex`, nunca el fondo. Por
eso un umbral simple (todos los canales > ~250) alcanza para clasificar
fondo vs. objeto sin necesitar canal alpha (no existe: el render se guarda
como RGB plano, ver `save_image_tensor`).

====================================================================
METRICAS Y GRANULARIDAD (2 niveles -- no se exporta punto-a-punto)
====================================================================

Nivel B -- por (objeto x n_views): una fila por objeto, para cada n_views
group presente en su molmo_multiview_<EXP>.json:
  - n_views_validas:  cuantas de las n_views vistas tienen >=2 puntos
    devueltos (definicion agnostica a obj_id -- es lo minimo que necesita
    tanto `widest_pair` (eje) como `get_point_by_obj_id` (plano) para
    construir un plano de interpretacion).
  - n_puntos_molmo2:  total de puntos devueltos, sumando TODAS las vistas
    del grupo (incluidas las invalidas con 0-1 puntos).
  - n_puntos_en_objeto: de esos, cuantos caen sobre el objeto (no sobre el
    fondo blanco), reescalando x/y de la escala 0-1000 de Molmo2 al tamano
    real de la imagen renderizada.

Nivel C -- resumen por (experiment_id x n_views), agregando sobre TODOS los
objetos con datos:
  - n_objects_con_datos
  - views_validas_{min,mean,max}      (por objeto)
  - puntos_molmo2_{min,mean,max}      (por objeto)
  - puntos_en_objeto_{min,mean,max}   (por objeto)
  - pct_en_objeto_mean: promedio, por objeto, de
    n_puntos_en_objeto / n_puntos_molmo2 -- la metrica que de verdad permite
    comparar prompts entre si sin que el total de puntos (que depende de
    n_views) confunda la lectura.

Usage
-----
    python Mapping/diagnose_point_localization.py \\
        --renders-root ../data/renders --symmetry-type axis_sym \\
        --experiment-id axis_v06_nomesh \\
        --out-detail ../results/diagnostics/axis_v06_point_localization_detail.csv \\
        --out-summary ../results/diagnostics/axis_v06_point_localization_summary.csv

    python Mapping/diagnose_point_localization.py \\
        --renders-root ../data/renders --symmetry-type plane_sym \\
        --experiment-id plane_v02_nomesh \\
        --out-detail ../results/diagnostics/plane_v02_point_localization_detail.csv \\
        --out-summary ../results/diagnostics/plane_v02_point_localization_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_common.naming import exp_filename

MOLMO_JSON        = "molmo_multiview.json"
BACKGROUND_THRESH = 250  # canal > esto en los 3 canales RGB = fondo blanco (ver docstring)
MIN_POINTS_VALID  = 2    # una vista cuenta como "valida" si devuelve >=2 puntos


def is_on_object(pixel: np.ndarray) -> bool:
    """True si el pixel NO es fondo (blanco puro, con tolerancia de antialiasing)."""
    return not bool(np.all(pixel[:3] > BACKGROUND_THRESH))


def molmo_xy_to_pixel(x: float, y: float, img_w: int, img_h: int) -> tuple[int, int]:
    """Escala 0-1000 (esquina superior izquierda) -> indice de pixel entero,
    clampeado a los bordes de la imagen real."""
    px = int(round((x / 1000.0) * img_w))
    py = int(round((y / 1000.0) * img_h))
    px = min(max(px, 0), img_w - 1)
    py = min(max(py, 0), img_h - 1)
    return px, py


def process_object(render_dir: Path, molmo_data: dict, image_cache: dict) -> dict[str, dict]:
    """Para cada n_views_key del molmo_multiview_<EXP>.json de UN objeto,
    calcula n_views_validas / n_puntos_molmo2 / n_puntos_en_objeto.
    image_cache: dict compartido dentro de este objeto (filename -> np.ndarray),
    evita releer el mismo PNG si aparece en mas de un n_views group."""
    out = {}
    for n_views_key, group in molmo_data.items():
        images_sent     = group.get("images_sent", [])
        points_by_image = group.get("points_by_image", {})
        if not images_sent:
            continue

        n_views_validas   = 0
        n_puntos_molmo2   = 0
        n_puntos_objeto   = 0

        for img_idx_str, pts in points_by_image.items():
            n_puntos_molmo2 += len(pts)
            if len(pts) >= MIN_POINTS_VALID:
                n_views_validas += 1
            if not pts:
                continue

            cam      = images_sent[int(img_idx_str)]
            filename = cam["filename"]
            if filename not in image_cache:
                img_path = render_dir / filename
                if not img_path.exists():
                    image_cache[filename] = None
                else:
                    image_cache[filename] = np.array(Image.open(img_path).convert("RGB"))
            img = image_cache[filename]
            if img is None:
                continue  # PNG no encontrado -- no se puede clasificar, se omite del numerador
            img_h, img_w = img.shape[0], img.shape[1]

            for p in pts:
                px, py = molmo_xy_to_pixel(p["x"], p["y"], img_w, img_h)
                if is_on_object(img[py, px]):
                    n_puntos_objeto += 1

        out[n_views_key] = {
            "n_views_validas":    n_views_validas,
            "n_puntos_molmo2":    n_puntos_molmo2,
            "n_puntos_en_objeto": n_puntos_objeto,
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Cuenta puntos devueltos por Molmo2 y cuantos caen sobre el objeto "
                     "(vs. fondo blanco) -- ver docstring del modulo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--renders-root", required=True)
    p.add_argument("--symmetry-type", required=True, choices=["axis_sym", "plane_sym"])
    p.add_argument("--experiment-id", required=True, nargs="+",
                   help="Uno o mas experiment-id (ej. axis_v06_nomesh). Cada uno se procesa "
                        "por separado pero ambos niveles de salida quedan en el mismo CSV, "
                        "con una columna 'experiment_id' para distinguirlos.")
    p.add_argument("--sizes", type=int, nargs="+", default=[224])
    p.add_argument("--lightings", type=str, nargs="+", default=["flat"])
    p.add_argument("--out-detail", required=True, help="CSV nivel B (objeto x n_views).")
    p.add_argument("--out-summary", required=True, help="CSV nivel C (resumen por experiment_id x n_views).")
    args = p.parse_args()

    symmetry_dir = Path(args.renders_root) / args.symmetry_type
    detail_rows  = []

    for experiment_id in args.experiment_id:
        input_file = exp_filename(MOLMO_JSON, experiment_id)
        object_dirs = sorted(d for d in symmetry_dir.iterdir() if d.is_dir())

        for obj_dir in object_dirs:
            obj_id = obj_dir.name
            for size in args.sizes:
                for lighting in args.lightings:
                    render_dir = obj_dir / str(size) / lighting
                    molmo_path = render_dir / input_file
                    if not molmo_path.exists():
                        continue
                    with open(molmo_path, encoding="utf-8") as f:
                        molmo_data = json.load(f)

                    image_cache: dict = {}
                    per_nv = process_object(render_dir, molmo_data, image_cache)
                    for n_views_key, m in per_nv.items():
                        detail_rows.append({
                            "experiment_id":      experiment_id,
                            "object_id":          obj_id,
                            "n_views":            int(n_views_key),
                            "n_views_validas":    m["n_views_validas"],
                            "n_puntos_molmo2":    m["n_puntos_molmo2"],
                            "n_puntos_en_objeto": m["n_puntos_en_objeto"],
                        })

    if not detail_rows:
        print("[error] No se genero ninguna fila -- revisa --experiment-id y las rutas.")
        sys.exit(1)

    out_detail = Path(args.out_detail)
    out_detail.parent.mkdir(parents=True, exist_ok=True)
    with open(out_detail, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)
    print(f"[ok] Nivel B: {len(detail_rows)} filas guardadas en {out_detail}")

    # ── Nivel C: resumen por experiment_id x n_views ──────────────────────────
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in detail_rows:
        groups[(r["experiment_id"], r["n_views"])].append(r)

    summary_rows = []
    for (experiment_id, n_views), rows in sorted(groups.items()):
        views_validas = np.array([r["n_views_validas"] for r in rows])
        puntos_molmo2 = np.array([r["n_puntos_molmo2"] for r in rows])
        puntos_objeto = np.array([r["n_puntos_en_objeto"] for r in rows])
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = np.where(puntos_molmo2 > 0, puntos_objeto / np.maximum(puntos_molmo2, 1), np.nan)
        pct_validas = pct[puntos_molmo2 > 0]

        summary_rows.append({
            "experiment_id":            experiment_id,
            "n_views":                  n_views,
            "n_objects_con_datos":      len(rows),
            "views_validas_min":        int(views_validas.min()),
            "views_validas_mean":       round(float(views_validas.mean()), 3),
            "views_validas_max":        int(views_validas.max()),
            "puntos_molmo2_min":        int(puntos_molmo2.min()),
            "puntos_molmo2_mean":       round(float(puntos_molmo2.mean()), 3),
            "puntos_molmo2_max":        int(puntos_molmo2.max()),
            "puntos_en_objeto_min":     int(puntos_objeto.min()),
            "puntos_en_objeto_mean":    round(float(puntos_objeto.mean()), 3),
            "puntos_en_objeto_max":     int(puntos_objeto.max()),
            "pct_en_objeto_mean":       round(float(pct_validas.mean()), 4) if len(pct_validas) else None,
        })

    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    with open(out_summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[ok] Nivel C: {len(summary_rows)} filas guardadas en {out_summary}\n")

    print(f"{'experiment_id':<20} {'n_views':>7} {'n_obj':>6}  "
          f"{'views_validas(min/mean/max)':<28} {'puntos(min/mean/max)':<24} "
          f"{'en_objeto(min/mean/max)':<24} {'pct_obj':>8}")
    for r in summary_rows:
        vv = f"{r['views_validas_min']}/{r['views_validas_mean']}/{r['views_validas_max']}"
        pm = f"{r['puntos_molmo2_min']}/{r['puntos_molmo2_mean']}/{r['puntos_molmo2_max']}"
        po = f"{r['puntos_en_objeto_min']}/{r['puntos_en_objeto_mean']}/{r['puntos_en_objeto_max']}"
        pct_str = f"{r['pct_en_objeto_mean']*100:.1f}%" if r["pct_en_objeto_mean"] is not None else "n/a"
        print(f"{r['experiment_id']:<20} {r['n_views']:>7} {r['n_objects_con_datos']:>6}  "
              f"{vv:<28} {pm:<24} {po:<24} {pct_str:>8}")


if __name__ == "__main__":
    main()
