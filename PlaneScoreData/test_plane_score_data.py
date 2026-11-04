#!/usr/bin/env python3
# Pruebas con cámaras y puntos sintéticos; no necesitan renders ni GPU.
#   python PlaneScoreData/test_plane_score_data.py
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from plane_score_data import VISTAS_DEFECTO, estimate_plane, procesar_objeto, score_datos

from estimate_symmetry_no_mesh import METHOD_KEY, estimate_plane_no_mesh  # noqa: E402
from evaluate import evaluate_object  # noqa: E402
from pipeline_common.camera import project_point  # noqa: E402
from verify_control import verificar_objeto

CARPETA = Path(__file__).resolve().parent
FOV, TAM = 60.0, 224
NORMAL_REAL = np.array([1.0, 0.0, 0.0])


def camaras_fibonacci(n_vistas: int) -> list[dict]:
    """ Crea cámaras sobre una esfera que miran al origen, con la convención de PyTorch3D. """
    camaras = []
    oro = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n_vistas):
        y = 1.0 - 2.0 * (i + 0.5) / n_vistas
        r = np.sqrt(1.0 - y * y)
        centro = np.array([r * np.cos(oro * i), y, r * np.sin(oro * i)]) * 2.0
        eje_z = -centro / np.linalg.norm(centro)
        eje_x = np.cross([0.0, 1.0, 0.0], eje_z)
        eje_x /= np.linalg.norm(eje_x)
        eje_y = np.cross(eje_z, eje_x)
        R = np.stack([eje_x, eje_y, eje_z], axis=1)
        camaras.append({"R": R.tolist(), "T": [0.0, 0.0, 2.0], "filename": f"v{i}.png",
                        "index": i})
    return camaras


def puntos_de_traza(camaras: list[dict], ruido: float, semilla: int = 0) -> dict:
    """ Proyecta el punto más alto y el más bajo de una traza elíptica sobre el plano x = 0. """
    rng = np.random.default_rng(semilla)
    angulos = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    traza = np.stack([np.zeros(400), 0.5 * np.sin(angulos), 0.8 * np.cos(angulos)], axis=1)
    puntos = {}
    for i, cam in enumerate(camaras):
        proyectados = np.array([project_point(p, cam["R"], cam["T"], FOV, TAM)[:2] for p in traza])
        proyectados = proyectados / TAM * 1000.0 + rng.normal(0, ruido, proyectados.shape)
        arriba = proyectados[proyectados[:, 1].argmin()]
        abajo = proyectados[proyectados[:, 1].argmax()]
        puntos[str(i)] = [{"obj_id": 1, "x": float(arriba[0]), "y": float(arriba[1])},
                          {"obj_id": 2, "x": float(abajo[0]), "y": float(abajo[1])}]
    return puntos


def error_grados(normal: list[float]) -> float:
    """ Calcula el ángulo entre una normal y la real, sin considerar el signo. """
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(normal, NORMAL_REAL)), 0.0, 1.0))))


def test_score_datos_premia_la_normal_paralela() -> None:
    """ Una candidata paralela al tercio de normales vale 0 y una perpendicular, 1. """
    normales = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1.0]])
    puntajes = score_datos(np.array([[1.0, 0, 0], [0.6, 0.8, 0]]), normales)
    assert puntajes[0] == 0.0
    assert puntajes[0] < puntajes[1]
    assert score_datos(np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]] * 3))[0] == 1.0


def test_camaras_reproduce_el_flujo_principal() -> None:
    """ El criterio 'camaras' debe dar exactamente lo mismo que estimate_plane_no_mesh. """
    for n_vistas in VISTAS_DEFECTO:
        for ruido in (0.0, 15.0):
            camaras = camaras_fibonacci(n_vistas)
            puntos = puntos_de_traza(camaras, ruido)
            nuevo = estimate_plane(puntos, camaras, FOV, TAM, "camaras")
            original = estimate_plane_no_mesh(puntos, camaras, FOV, TAM)
            assert nuevo["normal"] == original["normal"], (n_vistas, ruido)
            assert nuevo["origin"] == original["origin"], (n_vistas, ruido)
            assert nuevo["good_views"] == original["good_views"], (n_vistas, ruido)


def test_datos_no_empeora_y_mejora_con_muchas_vistas() -> None:
    """ Sin ruido, S_dat no debe empeorar y con 26 vistas debe superar al criterio actual.

    Esta escena tiene pocas vistas de canto con 6 y 14 vistas, y S_dat promedia al menos dos, así
    que ahí no se le exige recuperar el plano.
    """
    for n_vistas in VISTAS_DEFECTO:
        camaras = camaras_fibonacci(n_vistas)
        puntos = puntos_de_traza(camaras, 0.0)
        datos = error_grados(estimate_plane(puntos, camaras, FOV, TAM, "datos")["normal"])
        actual = error_grados(estimate_plane(puntos, camaras, FOV, TAM, "camaras")["normal"])
        print(f"  puntos perfectos, {n_vistas} vistas: S_dat {datos:.1f}, actual {actual:.1f}")
        assert datos <= actual + 1e-6, (n_vistas, datos, actual)
    assert datos < 5.0 and datos < actual / 2, (datos, actual)


def test_flujo_de_archivos() -> None:
    """ Prueba la escritura, el control y la comparación con un árbol de renders sintético. """
    with tempfile.TemporaryDirectory() as raiz:
        raiz = Path(raiz)
        objeto = raiz / "plane_sym" / "obj"
        render = objeto / "224" / "flat"
        render.mkdir(parents=True)
        grupos = {str(n): {"images_sent": camaras_fibonacci(n)} for n in VISTAS_DEFECTO + [42]}
        for n, grupo in grupos.items():
            grupo["points_by_image"] = puntos_de_traza(grupo["images_sent"], 5.0)
        (render / "molmo_multiview_plane_x.json").write_text(json.dumps(grupos), encoding="utf-8")

        assert procesar_objeto(objeto, "plane_x", "base", TAM, "flat", "camaras", VISTAS_DEFECTO,
                               False) == "escrito"
        subprocess.run(
            [sys.executable, str(CARPETA / "plane_score_data.py"), "--renders-root", str(raiz),
             "--input-id", "plane_x", "--output-id", "sdat", "--jobs", "2"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        assert procesar_objeto(objeto, "plane_x", "sdat", TAM, "flat", "datos", VISTAS_DEFECTO,
                               False) == "omitido"
        documento = json.loads((objeto / "predicted_symmetry_sdat.json").read_text("utf-8"))
        assert sorted(documento["n_views_predictions"], key=int) == ["6", "14", "26"]
        assert documento["n_views_predictions"]["6"]["triangulation"]["plane_score"] == "datos"
        assert not list(objeto.glob("*.tmp*"))

        assert verificar_objeto(objeto, "plane_x", "base", TAM, "flat", VISTAS_DEFECTO) == (3, 0)
        subprocess.run(
            [sys.executable, str(CARPETA / "verify_control.py"), "--renders-root", str(raiz),
             "--input-id", "plane_x", "--reference-id", "base"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        evaluadas = evaluate_object(
            objeto, {"elements": [{"normal": [1.0, 0.0, 0.0], "origin": [0.0, 0.0, 0.0]}]},
            "plane_sym", np.random.default_rng(0).normal(size=(200, 3)), METHOD_KEY,
            predicted_file="predicted_symmetry_sdat.json",
        )
        assert all(evaluadas[n]["status"] == "ok" for n in ("6", "14", "26")), evaluadas

        columnas = ["experiment", "method", "n_views", "n_objects", "auc_angular",
                    "angular_error_median", "angular_error_mean", "precision_10deg", "f1_ref",
                    "f1_ref_hungarian", "sde_ref_mean"]
        ruta_csv = raiz / "comparacion.csv"
        with open(ruta_csv, "w", encoding="utf-8", newline="") as archivo:
            escritor = csv.writer(archivo)
            escritor.writerow(columnas)
            for exp in ("base", "sdat"):
                escritor.writerow([exp, "triangulation", 6, 2, 0.5, 10, 20, 0.5, 0.3, 0.3, 0.01])
        for exp, error_a in (("base", 30.0), ("sdat", 10.0)):
            eval_json = {"objects": {"a": {"6": {"status": "ok", "angular_error_deg": error_a}},
                                     "b": None}}
            ruta = raiz / "plane_sym" / f"eval_s224_flat_{exp}_triangulation_results.json"
            ruta.write_text(json.dumps(eval_json), encoding="utf-8")
        salida = raiz / "salida.csv"
        subprocess.run(
            [sys.executable, str(CARPETA / "compare_plane_scores.py"), "--renders-root", str(raiz),
             "--comparison-csv", str(ruta_csv), "--base-id", "base", "--new-id", "sdat",
             "--salida", str(salida)],
            check=True, stdout=subprocess.DEVNULL,
        )
        with open(salida, encoding="utf-8", newline="") as archivo:
            fila = next(csv.DictReader(archivo))
        assert float(fila["dif_media"]) == -10.0
        assert int(fila["mejoran"]) == 1 and int(fila["empeoran"]) == 0


if __name__ == "__main__":
    for nombre, prueba in sorted(globals().items()):
        if nombre.startswith("test_"):
            prueba()
            print(f"ok {nombre}")
