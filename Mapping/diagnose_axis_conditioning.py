"""
diagnose_axis_conditioning.py
------------------------------
Diagnostico barato (CERO llamadas nuevas a Molmo2) para decidir si vale la
pena invertir en un experimento de "4 puntos"/triangulacion alternativa para
axis_sym en el pipeline sin malla, o si el error angular alto que vemos en
TODOS los prompts de eje (~58-65 grados en el sweep, ver
results/experiments_*/axis_sym_nomesh_comparison.csv) tiene otra causa.

====================================================================
QUE SE EVALUA (3 metricas, una por mecanismo de ruido distinto)
====================================================================

Todas se calculan RE-PROCESANDO datos que ya existen
(molmo_multiview_<EXP>.json + eval_..._triangulation_results.json) -- no se
llama a Molmo2 de nuevo. Se correlaciona cada metrica, por objeto, contra el
angular_error_deg que ya reporto evaluate.py para ese objeto.

1. PIXEL_SEP (separacion en pixeles del par de puntos DENTRO de una vista)
   ---------------------------------------------------------------------
   Por que importa: cada vista aporta un plano de interpretacion (normal n_i)
   construido a partir de DOS puntos 2D + el centro de camara
   (pipeline_common.triangulation.interpretation_plane_normal). Si esos dos
   puntos estan muy juntos en la imagen, un mismo error de localizacion en
   pixeles de Molmo2 (siempre presente, cualquier VLM tiene error de
   localizacion) produce un error angular MUCHO mayor en la normal n_i
   resultante -- exactamente el argumento clasico de "baseline corto -> error
   de profundidad alto" en triangulacion estereo (Hartley & Zisserman,
   "Multiple View Geometry in Computer Vision", 2004, cap. de triangulacion:
   el error de triangulacion escala con 1/sin(theta), donde theta es el
   angulo/separacion entre las observaciones). Aca "theta chico" se traduce
   en PIXEL_SEP chico.
   Prediccion si esto domina: objetos con PIXEL_SEP promedio bajo (entre
   vistas) deberian tener angular_error sistematicamente mas alto.

2. COND_NUMBER (condicionamiento del sistema MULTI-VISTA, sigma1/sigma2)
   ---------------------------------------------------------------------
   Por que importa: triangulate_line (pipeline_common/triangulation.py)
   recupera la direccion del eje como el vector nulo (menor valor singular)
   de la matriz N (k normales apiladas, una por vista). En el caso ideal SIN
   ruido, N tiene rango 2 exacto (sigma3=0) y la direccion queda perfectamente
   determinada SIEMPRE QUE sigma2 (el segundo valor singular, el mas chico de
   los "no nulos") sea grande. Si las camaras usadas son casi coplanares o
   colineales -- el "Riesgo conocido" que ya anticipa
   docs/pipeline_sin_malla.md S3.1 -- las normales n_i tienden a ser casi
   paralelas entre si, sigma2 se acerca a sigma3, y la direccion (el vector
   nulo) queda pobremente determinada: pequenas perturbaciones de ruido en
   los puntos 2D pueden rotar mucho la direccion recuperada. Este es el
   argumento estandar de sensibilidad de soluciones por minimos cuadrados
   totales/SVD ante perturbaciones (Golub & Van Loan, "Matrix Computations",
   cap. de SVD y TLS: el numero de condicion sigma1/sigma2 acota la
   sensibilidad del vector nulo recuperado a ruido en la matriz de entrada).
   Se reporta como COND_NUMBER = sigma1/sigma2 (mayor = peor condicionado) y,
   de forma mas interpretable, como MEAN_PAIRWISE_ANGLE_DEG (angulo promedio
   entre pares de normales -- chico = normales casi paralelas = mismo problema
   dicho en grados en vez de un numero de condicion adimensional).
   Prediccion si esto domina: objetos con COND_NUMBER alto (o
   MEAN_PAIRWISE_ANGLE_DEG chico) deberian tener angular_error mas alto,
   independientemente de cuan separados esten los puntos DENTRO de cada vista.

3. AXIS_SPAN (el "lever arm" real, traducido a esta geometria)
   ---------------------------------------------------------------------
   Esta es la traduccion correcta de la hipotesis de "lever arm" que se
   discutio para el pipeline CON malla (ahi, se mide como la distancia 3D
   entre los dos puntos ya triangulados via ray-casting). Ese diagnostico NO
   aplica tal cual aca: el pipeline sin malla nunca produce puntos 3D
   individuales -- solo produce planos de interpretacion y, al final, UNA
   linea 3D (el eje ya ajustado). La traduccion correcta del argumento
   ("mas separacion espacial de las observaciones a lo largo del eje ->
   mejor determinada la direccion, para un mismo nivel de ruido angular por
   observacion") es el resultado clasico de regresion/TLS de que la varianza
   de la pendiente/direccion estimada es inversamente proporcional a la
   dispersion de los regresores a lo largo del eje de ajuste (Draper & Smith,
   "Applied Regression Analysis"; York, 1966, para TLS con error en ambas
   variables) -- el mismo principio por el que, en regresion lineal, puntos
   mas separados en X dan una pendiente mas precisa para un mismo ruido en Y.
   Aca, en vez de puntos 3D, usamos el eje YA AJUSTADO (direction, origin) y
   proyectamos sobre el, para cada punto 2D observado, el punto de la RECTA
   DEL EJE mas cercano al rayo camara-punto (formula estandar de punto mas
   cercano entre dos rectas). AXIS_SPAN = rango (maximo - minimo) de esas
   proyecciones a lo largo del eje, en unidades del mundo (mismas unidades
   que translation_error). Es un proxy geometrico razonable de "que tan lejos
   entre si, a lo largo del eje, caen las observaciones que sostienen el
   ajuste" -- sin necesitar reconstruir puntos 3D reales (no hay malla).
   Prediccion si esto domina: objetos con AXIS_SPAN chico (todas las
   observaciones concentradas cerca de una misma altura del eje) deberian
   tener angular_error mas alto, incluso si PIXEL_SEP y COND_NUMBER son
   buenos.

====================================================================
COMO SE USA EL RESULTADO
====================================================================
Se reporta correlacion de Pearson y de Spearman (via rangos, sin depender de
scipy -- ver docs/code-norms.md / CLAUDE.md sobre evitar scipy cuando se
pueda) entre angular_error_deg y cada una de las 3 metricas, por n_views y
pooled. Una correlacion fuerte y negativa (PIXEL_SEP, AXIS_SPAN) o positiva
(COND_NUMBER) en una metrica y debil en las otras dos indica CUAL mecanismo
domina el error -- y por lo tanto que tipo de cambio (prompt vs. ajuste
robusto vs. mas puntos bien espaciados) tiene sentido probar despues.

Usage
-----
    python Mapping/diagnose_axis_conditioning.py \\
        --renders-root ../data/renders \\
        --experiment-id axis_v06_nomesh \\
        --out ../results/diagnostics/axis_v06_conditioning.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_common.naming import exp_filename
from pipeline_common.triangulation import (
    interpretation_plane_normal,
    ray_dir_for_point,
    widest_pair,
)

MOLMO_JSON     = "molmo_multiview.json"
MANIFEST_FILE  = "manifest.json"
DEFAULT_FOV    = 60.0
EPS            = 1e-9


# ── Geometry helpers (nuevos, especificos de este diagnostico) ───────────────

def closest_point_on_line_to_line(
    p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray,
) -> float | None:
    """
    Parametro t tal que (p1 + t*d1) es el punto de la recta 1 mas cercano a
    la recta 2 (formula estandar de closest-point entre dos rectas en 3D).
    Usado aca con recta 1 = eje ya ajustado, recta 2 = rayo camara-punto 2D
    -- da la posicion, a lo largo del eje, "explicada" por esa observacion.
    Devuelve None si las rectas son (numericamente) paralelas.
    """
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)
    r = p1 - p2
    a = 1.0
    b = np.dot(d1, d2)
    c = 1.0
    d = np.dot(d1, r)
    e = np.dot(d2, r)
    denom = a * c - b * b
    if abs(denom) < EPS:
        return None
    t = (b * e - c * d) / denom
    return float(t)


def condition_number(normals: list[np.ndarray]) -> float | None:
    """sigma1/sigma2 de la matriz de normales apiladas (Golub & Van Loan) --
    mayor = peor condicionado el vector nulo (la direccion del eje) frente a
    ruido. None si hay <2 normales o sigma2 es numericamente cero."""
    N = np.asarray(normals, dtype=np.float64)
    if len(N) < 2:
        return None
    sing = np.linalg.svd(N, compute_uv=False)
    if len(sing) < 2 or sing[1] < EPS:
        return None
    return float(sing[0] / sing[1])


def mean_pairwise_angle_deg(normals: list[np.ndarray]) -> float | None:
    """Angulo promedio (grados, en [0,90], sign-agnostic) entre todos los
    pares de normales de interpretacion -- version interpretable de
    condition_number: chico = normales casi paralelas = mal condicionado."""
    n = len(normals)
    if n < 2:
        return None
    angles = []
    for i in range(n):
        for j in range(i + 1, n):
            cos_ang = abs(np.dot(normals[i], normals[j]))
            angles.append(np.degrees(np.arccos(np.clip(cos_ang, 0.0, 1.0))))
    return float(np.mean(angles))


# ── Correlacion sin scipy (Pearson + Spearman via rangos) ────────────────────

def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.std(x) < EPS or np.std(y) < EPS:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    """Spearman = Pearson sobre rangos (evita depender de scipy.stats, ver
    CLAUDE.md 'Evitar scipy en codigo de notebook si es evitable')."""
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return pearson(rx, ry)


# ── Por-objeto: recalcula normales/rayos desde el molmo_multiview crudo ──────

def diagnose_object(
    molmo_data: dict, fov_deg: float, image_size: int,
) -> dict[str, dict]:
    """Para cada n_views_key del molmo_multiview_<EXP>.json de un objeto,
    recalcula PIXEL_SEP / COND_NUMBER / MEAN_PAIRWISE_ANGLE. AXIS_SPAN se
    calcula despues, en main(), porque necesita el eje YA AJUSTADO
    (direction/origin) que viene de predicted_symmetry_<EXP>.json."""
    out = {}
    for n_views_key, group in molmo_data.items():
        images_sent     = group.get("images_sent", [])
        points_by_image = group.get("points_by_image", {})
        if not points_by_image:
            continue

        pixel_seps, normals, rays = [], [], []  # rays: (origin, direction) por punto usado
        for img_idx_str, pts in points_by_image.items():
            pair = widest_pair(pts)
            if pair is None:
                continue
            p_a, p_b = pair
            dx, dy = p_a["x"] - p_b["x"], p_a["y"] - p_b["y"]
            pixel_seps.append(float(np.hypot(dx, dy)))

            cam = images_sent[int(img_idx_str)]
            C, d_a = ray_dir_for_point(p_a["x"], p_a["y"], cam["R"], cam["T"], fov_deg, image_size)
            _, d_b = ray_dir_for_point(p_b["x"], p_b["y"], cam["R"], cam["T"], fov_deg, image_size)
            n = interpretation_plane_normal(d_a, d_b)
            if n is None:
                continue
            normals.append(n)
            rays.append((C, d_a))
            rays.append((C, d_b))

        if len(normals) < 2:
            continue

        out[n_views_key] = {
            "n_planes_used":          len(normals),
            "pixel_sep_mean":         float(np.mean(pixel_seps)) if pixel_seps else None,
            "pixel_sep_min":          float(np.min(pixel_seps)) if pixel_seps else None,
            "cond_number":            condition_number(normals),
            "mean_pairwise_angle_deg": mean_pairwise_angle_deg(normals),
            "_rays":                  rays,  # usado solo internamente para AXIS_SPAN
        }
    return out


def compute_axis_span(rays: list[tuple[np.ndarray, np.ndarray]],
                       axis_point: np.ndarray, axis_dir: np.ndarray) -> float | None:
    """Rango (max-min) de las proyecciones de cada rayo observado sobre el
    eje ya ajustado -- ver docstring del modulo, seccion 3 (AXIS_SPAN)."""
    ts = []
    for C, d in rays:
        t = closest_point_on_line_to_line(axis_point, axis_dir, C, d)
        if t is not None:
            ts.append(t)
    if len(ts) < 2:
        return None
    return float(np.max(ts) - np.min(ts))


# ── Carga de angular_error ya evaluado (evaluate.py) ──────────────────────────

def load_angular_errors(renders_root: Path, sizes: list[int], lightings: list[str],
                        experiment_id: str, method: str = "triangulation") -> dict:
    """Lee eval_..._<experiment_id>_<method>_results.json (ya escrito por
    Mapping/evaluate.py) y devuelve {object_id: {n_views_key: angular_error_deg}}
    solo para entradas con status == 'ok'."""
    size_tag  = "s" + "_".join(str(s) for s in sizes)
    light_tag = "_".join(lightings)
    path = renders_root / "axis_sym" / f"eval_{size_tag}_{light_tag}_{experiment_id}_{method}_results.json"
    if not path.exists():
        raise SystemExit(
            f"[error] No se encontro {path}. Corre antes: python Mapping/evaluate.py "
            f"--renders-root {renders_root} --objects-root <...> --symmetry-type axis_sym "
            f"--sizes {' '.join(map(str, sizes))} --lightings {' '.join(lightings)} "
            f"--experiment-id {experiment_id} --method {method}"
        )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    out = defaultdict(dict)
    for obj_id, per_nv in data.get("objects", {}).items():
        for nv_key, metrics in per_nv.items():
            if isinstance(metrics, dict) and metrics.get("status") == "ok":
                out[obj_id][nv_key] = metrics["angular_error_deg"]
    return out


def load_predicted_axes(object_dir: Path, experiment_id: str) -> dict:
    """Lee predicted_symmetry_<EXP>.json y devuelve {n_views_key: (origin, direction)}
    del metodo 'triangulation' -- el eje ya ajustado, necesario para AXIS_SPAN."""
    path = object_dir / exp_filename("predicted_symmetry.json", experiment_id)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for nv_key, preds in data.get("n_views_predictions", {}).items():
        pred = preds.get("triangulation")
        if pred is not None:
            out[nv_key] = (np.array(pred["origin"]), np.array(pred["direction"]))
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Diagnostico de conditioning geometrico para axis_sym, sin malla "
                     "(ver docstring del modulo para la justificacion completa).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--renders-root", required=True)
    p.add_argument("--experiment-id", required=True,
                   help="Ej: axis_v06_nomesh. Lee molmo_multiview_<ID>.json, "
                        "predicted_symmetry_<ID>.json y eval_..._<ID>_triangulation_results.json "
                        "-- los tres ya deben existir (no se llama a Molmo2 ni se re-estima nada).")
    p.add_argument("--sizes", type=int, nargs="+", default=[224])
    p.add_argument("--lightings", type=str, nargs="+", default=["flat"])
    p.add_argument("--out", default=None, help="CSV de salida (una fila por objeto x n_views).")
    args = p.parse_args()

    renders_root = Path(args.renders_root)
    symmetry_dir = renders_root / "axis_sym"

    angular_errors = load_angular_errors(renders_root, args.sizes, args.lightings, args.experiment_id)

    rows = []
    input_file = exp_filename(MOLMO_JSON, args.experiment_id)

    object_dirs = sorted(d for d in symmetry_dir.iterdir() if d.is_dir())
    for obj_dir in object_dirs:
        obj_id = obj_dir.name
        if obj_id not in angular_errors:
            continue

        predicted_axes = load_predicted_axes(obj_dir, args.experiment_id)

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

                diag = diagnose_object(molmo_data, fov_deg, image_size)

                for nv_key, metrics in diag.items():
                    if nv_key not in angular_errors[obj_id]:
                        continue
                    if nv_key not in predicted_axes:
                        continue
                    axis_point, axis_dir = predicted_axes[nv_key]
                    axis_span = compute_axis_span(metrics["_rays"], axis_point, axis_dir)

                    rows.append({
                        "object_id":               obj_id,
                        "n_views":                 int(nv_key),
                        "angular_error_deg":       angular_errors[obj_id][nv_key],
                        "n_planes_used":           metrics["n_planes_used"],
                        "pixel_sep_mean":          metrics["pixel_sep_mean"],
                        "pixel_sep_min":           metrics["pixel_sep_min"],
                        "cond_number":             metrics["cond_number"],
                        "mean_pairwise_angle_deg": metrics["mean_pairwise_angle_deg"],
                        "axis_span":               axis_span,
                    })

    if not rows:
        print("[error] No se pudo calcular ninguna fila -- revisa --experiment-id y las rutas.")
        sys.exit(1)

    import csv as csv_mod
    fieldnames = list(rows[0].keys())
    out_path = Path(args.out) if args.out else Path(f"diagnostic_{args.experiment_id}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] {len(rows)} filas (objeto x n_views) guardadas en {out_path}\n")

    # ── Reporte de correlaciones ──────────────────────────────────────────────
    predictors = ["pixel_sep_mean", "pixel_sep_min", "cond_number",
                  "mean_pairwise_angle_deg", "axis_span"]
    expected_sign = {
        "pixel_sep_mean": "negativa", "pixel_sep_min": "negativa",
        "cond_number": "POSITIVA", "mean_pairwise_angle_deg": "negativa",
        "axis_span": "negativa",
    }

    n_views_present = sorted({r["n_views"] for r in rows})
    print("Correlacion con angular_error_deg (Pearson / Spearman) -- signo esperado entre parentesis:")
    print("(negativa = a mas metrica, MENOS error; POSITIVA = a mas metrica, MAS error)\n")

    header = f"{'predictor':<26}" + "".join(f"  n={nv:<10}" for nv in n_views_present) + "  pooled"
    print(header)
    print("-" * len(header))

    y_all_by_nv = {nv: np.array([r["angular_error_deg"] for r in rows if r["n_views"] == nv])
                   for nv in n_views_present}
    y_pooled = np.array([r["angular_error_deg"] for r in rows])

    for pred in predictors:
        line = f"{pred + f' ({expected_sign[pred]})':<26}"
        for nv in n_views_present:
            sub = [r[pred] for r in rows if r["n_views"] == nv and r[pred] is not None]
            y_sub = np.array([r["angular_error_deg"] for r in rows if r["n_views"] == nv and r[pred] is not None])
            if len(sub) < 3:
                line += f"  {'n/a':<10}"
                continue
            x = np.array(sub)
            pr, sr = pearson(x, y_sub), spearman(x, y_sub)
            line += f"  {pr:+.2f}/{sr:+.2f} "
        x_pooled = np.array([r[pred] for r in rows if r[pred] is not None])
        y_p = np.array([r["angular_error_deg"] for r in rows if r[pred] is not None])
        pr, sr = pearson(x_pooled, y_p), spearman(x_pooled, y_p)
        line += f"  {pr:+.2f}/{sr:+.2f}"
        print(line)

    print(
        "\nLectura: |r| >= 0.3 con el signo esperado = evidencia de que ese mecanismo "
        "contribuye al error angular. Si SOLO cond_number/mean_pairwise_angle_deg "
        "correlaciona fuerte -> el problema es de diversidad de vistas (candidato: "
        "ajuste robusto/ponderado en triangulate_line, no un prompt nuevo). Si SOLO "
        "axis_span correlaciona fuerte -> confirma la hipotesis de 'lever arm' del "
        "pipeline con malla, traducida a este pipeline -> vale la pena un prompt de "
        "mas puntos bien anclados y bien separados a lo largo del eje. Si ninguna "
        "correlaciona -> el ruido probablemente viene de inconsistencia de IDENTIDAD "
        "entre vistas (Molmo senalando estructuras distintas en cada vista), no de "
        "conditioning geometrico -- ver la propuesta hibrida de verificacion cross-view."
    )


if __name__ == "__main__":
    main()
