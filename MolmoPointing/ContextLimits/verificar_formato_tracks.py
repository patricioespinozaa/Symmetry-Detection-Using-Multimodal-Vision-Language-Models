# Verifica si el cambio de formato de salida de Molmo2 (<points> -> <tracks>, su formato de
# tracking de video) es un problema de parseo o un cambio real en cómo predice el modelo.
# Ver contexto_tokens_molmo2.md §4. Responde tres preguntas, por llamada (objeto x size x
# lighting x n_views) del pipeline sin malla:
#
#   1. Índices de imagen: en <tracks>, ¿el primer número de cada grupo sigue siendo el índice
#      1-based de la imagen? (empieza en 0, fuera de rango, cubre 1..N, repetidos, orden).
#      Un corrimiento haría que parse_multi_coords asigne puntos a la cámara equivocada.
#   2. Sesgo de video: ¿las coordenadas de un mismo obj_id se mueven poco entre vistas
#      consecutivas, como en un video? Las vistas están repartidas en la esfera de Fibonacci,
#      así que la proyección real del eje/plano salta mucho entre vistas vecinas.
#   3. Desempeño: error angular (eval_*_results.json de evaluate.py) de <tracks> vs. <points>
#      a igual configuración, y comparación pareada del MISMO objeto entre iluminaciones
#      (224 px) cuando una corrida respondió <tracks> y la otra <points>.
#
# No usa GPU ni el modelo. Uso (desde la raíz del repo, en el servidor):
#   python MolmoPointing/ContextLimits/verificar_formato_tracks.py --renders-root ../data/renders
#
# Salidas en MolmoPointing/ContextLimits/resultados/:
#   tracks_llamadas.csv   una fila por llamada (formato, chequeos de índices, movimiento, error)
#   tracks_resumen.csv    por (tipo, size, lighting, n_views, formato)
#   tracks_pareado.csv    comparación pareada entre iluminaciones, por (tipo, n_views)
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Mapping"))
from evaluate import AUC_ANGULAR_MAX, auc_from_errors  # noqa: E402

SALIDA = Path(__file__).resolve().parent / "resultados"

# tipo -> (experiment_id de molmo_multiview_<ID>.json, experiment_id evaluado sin malla)
EXPERIMENTOS = {
    "axis_sym": ("axis_v04", "axis_v04_ndcfix_nomesh"),
    "plane_sym": ("plane_v05_1", "plane_v05_1_ndcfix_mp1_nomesh"),
}
METODO = "triangulation"
ERROR_SIN_PRED = 90.0     # misma imputación que evaluate.py
N_BOOTSTRAP = 2000

RE_COORDS = re.compile(r'coords=["\']([^"\']+)["\']')     # mismo regex que parse_multi_coords
RE_ETIQUETA = re.compile(r"<\s*(\w+)\s+coords=")


def parsear_salida(raw: str) -> tuple[str, list[tuple[int, list[tuple[int, float, float]]]]]:
    """ Clasifica la salida cruda y extrae sus grupos sin descartar índices fuera de rango.

    Args:
        * raw: raw_output de una llamada multi-imagen

    Returns:
        * tuple: (formato, grupos). formato es 'points', 'tracks', 'cortada', 'sin_coords' u
          'otro:<etiqueta>'; grupos es [(indice_crudo, [(obj_id, x, y), ...]), ...]

    """
    m = RE_COORDS.search(raw)
    etiqueta = RE_ETIQUETA.search(raw)
    if not m:
        return ("cortada" if "coords=" in raw else "sin_coords"), []
    nombre = etiqueta.group(1) if etiqueta else "?"
    formato = nombre if nombre in ("points", "tracks") else f"otro:{nombre}"

    grupos = []
    for grupo in m.group(1).split(";"):
        nums = grupo.split()
        if len(nums) < 4:
            continue
        try:
            idx = int(float(nums[0]))
            resto = [float(n) for n in nums[1:]]
        except ValueError:
            continue
        pts = [(int(resto[i]), resto[i + 1], resto[i + 2]) for i in range(0, len(resto) - 2, 3)]
        grupos.append((idx, pts))
    return formato, grupos


def chequear_indices(grupos: list, n_imagenes: int) -> dict:
    """ Chequeos de que el primer número de cada grupo sea el índice 1-based de la imagen.

    Args:
        * grupos: salida de parsear_salida()
        * n_imagenes: imágenes enviadas en la llamada

    Returns:
        * dict: n_grupos, min/max del índice y banderas de corrimiento, cobertura y orden

    """
    idx = [g[0] for g in grupos]
    if not idx:
        return {"n_grupos": 0}
    unicos = set(idx)
    return {
        "n_grupos": len(idx),
        "idx_min": min(idx),
        "idx_max": max(idx),
        "empieza_en_0": min(idx) == 0,
        "fuera_de_rango": any(i < 1 or i > n_imagenes for i in idx),
        "cubre_1_a_N": unicos == set(range(1, n_imagenes + 1)),
        "frac_imagenes_cubiertas": len(unicos & set(range(1, n_imagenes + 1))) / n_imagenes,
        "idx_repetidos": sum(1 for i in unicos if idx.count(i) > 1),
        "orden_creciente": all(a <= b for a, b in zip(idx, idx[1:])),
    }


def movimiento_entre_vistas(grupos: list) -> dict:
    """ Desplazamiento de cada obj_id entre imágenes consecutivas (escala 0-1000 de Molmo2).

    Args:
        * grupos: salida de parsear_salida()

    Returns:
        * dict: desplazamiento medio entre vistas consecutivas y fracción de pares con la
          misma coordenada exacta (NaN si no hay pares comparables)

    """
    por_imagen = {}
    for idx, pts in grupos:
        for obj_id, x, y in pts:
            por_imagen.setdefault(idx, {})[obj_id] = (x, y)
    imagenes = sorted(por_imagen)
    desplazamientos, identicos = [], 0
    for a, b in zip(imagenes, imagenes[1:]):
        for obj_id in por_imagen[a].keys() & por_imagen[b].keys():
            (xa, ya), (xb, yb) = por_imagen[a][obj_id], por_imagen[b][obj_id]
            d = float(np.hypot(xb - xa, yb - ya))
            desplazamientos.append(d)
            identicos += d == 0
    if not desplazamientos:
        return {"desplaz_medio": np.nan, "frac_pares_identicos": np.nan}
    return {"desplaz_medio": float(np.mean(desplazamientos)),
            "frac_pares_identicos": identicos / len(desplazamientos)}


def cargar_errores(renders_root: Path, tipo: str, exp_eval: str, size: int,
                   lighting: str) -> dict | None:
    """ Lee el error angular por objeto y n_views de un eval_*_results.json de evaluate.py.

    Args:
        * renders_root: raíz de renders
        * tipo: 'axis_sym' o 'plane_sym'
        * exp_eval: experiment_id evaluado (p. ej. axis_v04_ndcfix_nomesh)
        * size: resolución
        * lighting: iluminación

    Returns:
        * dict | None: {(object_id, n_views): angular_error_deg o None}, o None si no existe

    """
    ruta = renders_root / tipo / f"eval_s{size}_{lighting}_{exp_eval}_{METODO}_results.json"
    if not ruta.exists():
        print(f"  [aviso] no existe {ruta}: esas llamadas quedan sin error angular")
        return None
    with open(ruta, encoding="utf-8") as f:
        objetos = json.load(f)["objects"]
    errores = {}
    for obj, por_vistas in objetos.items():
        for nv, m in (por_vistas or {}).items():
            ok = isinstance(m, dict) and m.get("status") == "ok"
            errores[(obj, int(nv))] = m["angular_error_deg"] if ok else None
    print(f"  eval {ruta.name}: {len(objetos)} objetos")
    return errores


def recorrer(args: argparse.Namespace) -> pd.DataFrame:
    """ Construye la tabla por llamada recorriendo los molmo_multiview de cada configuración.

    Args:
        * args: argumentos de la línea de comandos

    Returns:
        * pd.DataFrame: una fila por (tipo, objeto, size, lighting, n_views)

    """
    filas = []
    for tipo in args.tipos:
        exp_molmo, exp_eval = EXPERIMENTOS[tipo]
        for size in args.sizes:
            for lighting in args.lightings:
                errores = cargar_errores(args.renders_root, tipo, exp_eval, size, lighting)
                rutas = sorted((args.renders_root / tipo).glob(
                    f"*/{size}/{lighting}/molmo_multiview_{exp_molmo}.json"))
                if not rutas:
                    continue
                print(f"  {tipo} {size}px {lighting}: {len(rutas)} molmo_multiview_{exp_molmo}.json")
                for ruta in rutas:
                    obj = ruta.parent.parent.parent.name
                    with open(ruta, encoding="utf-8") as f:
                        datos = json.load(f)
                    for nv in args.vistas:
                        entrada = datos.get(str(nv))
                        if not isinstance(entrada, dict) or not isinstance(entrada.get("raw_output"), str):
                            continue
                        n_img = len(entrada["images_sent"])
                        formato, grupos = parsear_salida(entrada["raw_output"])
                        fila = {"tipo": tipo, "objeto": obj, "size": size, "lighting": lighting,
                                "n_views": nv, "n_imagenes": n_img, "formato": formato,
                                "n_input_tokens": entrada.get("n_input_tokens"),
                                "n_imagenes_parseadas": len(entrada.get("points_by_image", {}))}
                        fila.update(chequear_indices(grupos, n_img))
                        fila.update(movimiento_entre_vistas(grupos))
                        if errores is not None:
                            err = errores.get((obj, nv))
                            fila["con_prediccion"] = err is not None
                            fila["error_angular"] = err if err is not None else ERROR_SIN_PRED
                        filas.append(fila)
    return pd.DataFrame(filas)


def bootstrap_ic(valores: np.ndarray, estadistico, rng: np.random.Generator) -> tuple[float, float]:
    """ Intervalo de confianza bootstrap al 95 % de un estadístico sobre una muestra.

    Args:
        * valores: muestra
        * estadistico: función np.ndarray -> float (p. ej. np.mean)
        * rng: generador aleatorio

    Returns:
        * tuple: (límite inferior, límite superior)

    """
    muestras = [estadistico(rng.choice(valores, len(valores))) for _ in range(N_BOOTSTRAP)]
    return float(np.percentile(muestras, 2.5)), float(np.percentile(muestras, 97.5))


def resumir(d: pd.DataFrame) -> pd.DataFrame:
    """ Resume por (tipo, size, lighting, n_views, formato): índices, movimiento y desempeño.

    Args:
        * d: DataFrame de recorrer()

    Returns:
        * pd.DataFrame: una fila por grupo

    """
    filas = []
    claves = ["tipo", "size", "lighting", "n_views", "formato"]
    for k, g in d.groupby(claves):
        fila = dict(zip(claves, k))
        fila["n_llamadas"] = len(g)
        fila["frac_del_total"] = len(g) / len(d[(d[claves[:4]] == pd.Series(k[:4], claves[:4])).all(axis=1)])
        for c in ["empieza_en_0", "fuera_de_rango", "cubre_1_a_N", "orden_creciente"]:
            fila[f"frac_{c}"] = g[c].mean() if c in g else np.nan
        for c in ["frac_imagenes_cubiertas", "idx_repetidos", "desplaz_medio", "frac_pares_identicos"]:
            fila[f"media_{c}" if not c.startswith("frac") else c] = g[c].mean() if c in g else np.nan
        if "error_angular" in g:
            e = g["error_angular"].to_numpy(dtype=float)
            fila.update({
                "frac_con_prediccion": g["con_prediccion"].mean(),
                "error_medio": e.mean(), "error_mediano": float(np.median(e)),
                "precision_10deg": float(np.mean(e < 10)),
                "auc_angular": auc_from_errors(list(e), AUC_ANGULAR_MAX),
            })
        filas.append(fila)
    return pd.DataFrame(filas)


def comparar_pareado(d: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """ Compara el MISMO objeto y n_views entre iluminaciones con formatos distintos (224 px).

    Para cada (objeto, n_views) con una corrida <tracks> y otra <points> (distinta
    iluminación), toma la diferencia de error angular tracks - points (promediando si hay
    varias de cada formato). Así se controla por la dificultad del objeto, que sí sesga la
    comparación no pareada.

    Args:
        * d: DataFrame de recorrer()
        * rng: generador aleatorio para el bootstrap

    Returns:
        * pd.DataFrame: una fila por (tipo, size, n_views) con al menos un par

    """
    if "error_angular" not in d:
        return pd.DataFrame()
    base = d[d["formato"].isin(["points", "tracks"])]
    medias = (base.groupby(["tipo", "size", "n_views", "objeto", "formato"])["error_angular"]
              .mean().unstack("formato").dropna())
    filas = []
    for (tipo, size, nv), g in medias.groupby(level=["tipo", "size", "n_views"]):
        diff = (g["tracks"] - g["points"]).to_numpy()
        ic_media = bootstrap_ic(diff, np.mean, rng)
        filas.append({
            "tipo": tipo, "size": size, "n_views": nv, "n_pares": len(diff),
            "error_points_medio": g["points"].mean(), "error_tracks_medio": g["tracks"].mean(),
            "diferencia_media": diff.mean(), "ic95_inf": ic_media[0], "ic95_sup": ic_media[1],
            "diferencia_mediana": float(np.median(diff)),
            "frac_tracks_peor": float(np.mean(diff > 0)),
            "frac_tracks_mejor": float(np.mean(diff < 0)),
        })
    return pd.DataFrame(filas)


def imprimir_conclusiones(res: pd.DataFrame, par: pd.DataFrame) -> None:
    """ Imprime un resumen legible de las tres verificaciones.

    Args:
        * res: DataFrame de resumir()
        * par: DataFrame de comparar_pareado()

    """
    tr = res[res["formato"] == "tracks"]
    print("\n=== 1. Índices de imagen en <tracks> ===")
    if tr.empty:
        print("  No hay llamadas <tracks>.")
    else:
        n = tr["n_llamadas"].sum()
        for c in ["frac_empieza_en_0", "frac_fuera_de_rango", "frac_cubre_1_a_N", "frac_orden_creciente"]:
            print(f"  {c:24s} {np.average(tr[c], weights=tr['n_llamadas']):.4f}  (sobre {n} llamadas)")
        print("  Cómo leerlo: empieza_en_0 y fuera_de_rango ~0 y cubre_1_a_N ~1: el índice es el 1-based de "
              "siempre y el parseo asigna cada punto a su cámara.")

    print("\n=== 2. Movimiento entre vistas consecutivas (0-1000) ===")
    for (tipo, size, light, nv), g in res.groupby(["tipo", "size", "lighting", "n_views"]):
        partes = [f"{f}: {r['media_desplaz_medio']:.0f} (idénticos {r['frac_pares_identicos']:.2f})"
                  for f, r in g.set_index("formato").iterrows() if f in ("points", "tracks")]
        if partes:
            print(f"  {tipo} {size}px {light} {nv} vistas | " + " | ".join(partes))
    print("  Cómo leerlo: desplazamiento mucho menor en <tracks> = el modelo sigue el punto como "
          "en un video.")

    if "error_medio" in res:
        print("\n=== 3a. Error angular por formato, misma configuración (no pareado) ===")
        for (tipo, size, light, nv), g in res.groupby(["tipo", "size", "lighting", "n_views"]):
            g = g.set_index("formato")
            if {"points", "tracks"} <= set(g.index):
                print(f"  {tipo} {size}px {light} {nv} vistas | "
                      + " | ".join(f"{f} n={int(g.loc[f, 'n_llamadas'])} "
                                   f"AUC={g.loc[f, 'auc_angular']:.3f} med={g.loc[f, 'error_mediano']:.1f}°"
                                   for f in ("points", "tracks")))
        print("  Ojo: no pareado; los objetos que disparan <tracks> pueden ser más difíciles.")
    if not par.empty:
        print("\n=== 3b. Pareado: mismo objeto, otra iluminación (tracks - points, grados) ===")
        for _, r in par.iterrows():
            print(f"  {r['tipo']} {r['size']}px {r['n_views']} vistas | {r['n_pares']} pares | "
                  f"dif. media {r['diferencia_media']:+.1f}° [IC95 {r['ic95_inf']:+.1f}, "
                  f"{r['ic95_sup']:+.1f}] | tracks peor en {r['frac_tracks_peor']:.0%}")
        print("  Cómo leerlo: IC95 entero > 0: <tracks> empeora la predicción del mismo objeto.")


def main() -> None:
    """ Corre las tres verificaciones y guarda los CSV. """
    p = argparse.ArgumentParser(description="Verifica el formato <tracks> de Molmo2.")
    p.add_argument("--renders-root", type=Path, required=True)
    p.add_argument("--tipos", nargs="+", default=list(EXPERIMENTOS), choices=list(EXPERIMENTOS))
    p.add_argument("--sizes", type=int, nargs="+", default=[224, 448, 1136])
    p.add_argument("--lightings", nargs="+", default=["flat", "darker", "brighter"])
    p.add_argument("--vistas", type=int, nargs="+", default=[6, 14, 26])
    p.add_argument("--salida", type=Path, default=SALIDA, help="carpeta de salida")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    d = recorrer(args)
    if d.empty:
        sys.exit(f"[error] no se encontraron molmo_multiview bajo {args.renders_root}")
    res = resumir(d)
    par = comparar_pareado(d, np.random.default_rng(args.seed))

    args.salida.mkdir(parents=True, exist_ok=True)
    for nombre, tabla in [("tracks_llamadas", d), ("tracks_resumen", res), ("tracks_pareado", par)]:
        ruta = args.salida / f"{nombre}.csv"
        tabla.round(4).to_csv(ruta, index=False)
        print(f"Guardado {ruta} ({len(tabla)} filas)")
    imprimir_conclusiones(res, par)


if __name__ == "__main__":
    main()
