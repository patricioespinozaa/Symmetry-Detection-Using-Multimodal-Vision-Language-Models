#!/usr/bin/env python3
# Prueba que audit_run.py detecte predicciones de otras corridas, usando un árbol sintético.
#   python PlaneScoreData/test_audit_run.py
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from plane_score_data import procesar_objeto

from test_plane_score_data import camaras_fibonacci, puntos_de_traza

CARPETA = Path(__file__).resolve().parent
ENTRADA = "plane_x"
SALIDA = "plane_x_sdat"
ARCHIVO_SALIDA = f"predicted_symmetry_{SALIDA}.json"
ARCHIVO_EVAL = f"eval_s224_flat_{SALIDA}_triangulation_results.json"


def crear_objeto(raiz: Path, nombre: str, size: int = 224, luz: str = "flat") -> Path:
    """ Crea un objeto con puntos de 6 vistas en una configuración. """
    objeto = raiz / "plane_sym" / nombre
    render = objeto / str(size) / luz
    render.mkdir(parents=True)
    camaras = camaras_fibonacci(6)
    grupos = {"6": {"images_sent": camaras, "points_by_image": puntos_de_traza(camaras, 5.0)}}
    (render / f"molmo_multiview_{ENTRADA}.json").write_text(json.dumps(grupos), encoding="utf-8")
    return objeto


def escribir_evaluacion(raiz: Path) -> None:
    """ Escribe una evaluación coherente con las predicciones actuales. """
    objetos = {}
    for objeto in sorted((raiz / "plane_sym").iterdir()):
        ruta = objeto / ARCHIVO_SALIDA
        if ruta.exists():
            grupos = json.loads(ruta.read_text("utf-8"))["n_views_predictions"]
            objetos[objeto.name] = {c: {"status": "ok", "angular_error_deg": 1.0} for c in grupos}
    (raiz / "plane_sym" / ARCHIVO_EVAL).write_text(json.dumps({"objects": objetos}), "utf-8")


def auditar(raiz: Path, *extra: str) -> tuple[int, str]:
    """ Ejecuta el auditor y devuelve el código de salida y el texto. """
    resultado = subprocess.run(
        [sys.executable, str(CARPETA / "audit_run.py"), "--renders-root", str(raiz),
         "--symmetry-type", "plane_sym", "--input-id", ENTRADA, "--output-id", SALIDA,
         "--estimador", "sdat", "--size", "224", "--lighting", "flat", "--muestra", "0", *extra],
        capture_output=True, text=True, encoding="utf-8",
        env=dict(os.environ, PYTHONIOENCODING="utf-8"), check=False,
    )
    return resultado.returncode, resultado.stdout + resultado.stderr


def preparar(raiz: Path) -> Path:
    """ Crea una corrida correcta y devuelve la ruta de la predicción de 'a'. """
    for nombre in ("a", "b"):
        assert procesar_objeto(crear_objeto(raiz, nombre), ENTRADA, SALIDA, 224, "flat", "datos",
                               [6, 14, 26], False) == "escrito"
    escribir_evaluacion(raiz)
    return raiz / "plane_sym" / "a" / ARCHIVO_SALIDA


def caso(descripcion: str, alterar, esperado: str) -> None:
    """ Altera una corrida correcta y comprueba que el auditor lo detecte. """
    with tempfile.TemporaryDirectory() as temporal:
        raiz = Path(temporal)
        prediccion = preparar(raiz)
        codigo, texto = auditar(raiz)
        assert codigo == 0, texto
        extra = alterar(raiz, prediccion) or ()
        codigo, texto = auditar(raiz, *extra)
        assert codigo == 1 and esperado in texto, (descripcion, texto)
        print(f"ok {descripcion}")


def predicciones_mezcladas(raiz: Path, prediccion: Path) -> None:
    """ Reemplaza la predicción de 'a' por la de otro criterio, como una corrida vieja. """
    documento = json.loads(prediccion.read_text("utf-8"))
    documento["n_views_predictions"]["6"]["triangulation"]["normal"] = [0.0, 0.0, 1.0]
    prediccion.write_text(json.dumps(documento), "utf-8")
    escribir_evaluacion(raiz)


def heredada_de_otra_configuracion(raiz: Path, _: Path) -> None:
    """ Simula el caso real: un objeto sin puntos aquí, con un archivo de otra configuración. """
    objeto = raiz / "plane_sym" / "c"
    objeto.mkdir()
    (objeto / ARCHIVO_SALIDA).write_text((raiz / "plane_sym" / "a" / ARCHIVO_SALIDA).read_text(
        "utf-8"), "utf-8")
    escribir_evaluacion(raiz)


def grupo_sin_vistas(raiz: Path, prediccion: Path) -> None:
    """ Deja sin puntos a la entrada de 'a', con su predicción aún en disco. """
    entrada = raiz / "plane_sym" / "a" / "224" / "flat" / f"molmo_multiview_{ENTRADA}.json"
    grupos = json.loads(entrada.read_text("utf-8"))
    grupos["6"]["points_by_image"] = {}
    entrada.write_text(json.dumps(grupos), "utf-8")
    os.utime(prediccion, (entrada.stat().st_mtime + 5, entrada.stat().st_mtime + 5))
    os.utime(raiz / "plane_sym" / ARCHIVO_EVAL,
             (entrada.stat().st_mtime + 9, entrada.stat().st_mtime + 9))


def prediccion_antigua(_: Path, prediccion: Path) -> tuple[str, str]:
    """ Fecha la predicción antes del inicio declarado de la corrida. """
    os.utime(prediccion, (1_000_000_000, 1_000_000_000))
    return "--desde", "2026-11-05 00:00"


def evaluacion_vieja(raiz: Path, prediccion: Path) -> None:
    """ Deja la evaluación con fecha anterior a una predicción. """
    ruta = raiz / "plane_sym" / ARCHIVO_EVAL
    os.utime(ruta, (prediccion.stat().st_mtime - 100, prediccion.stat().st_mtime - 100))


def evaluacion_de_mas(raiz: Path, _: Path) -> None:
    """ Agrega a la evaluación un objeto cuya predicción ya no existe. """
    ruta = raiz / "plane_sym" / ARCHIVO_EVAL
    datos = json.loads(ruta.read_text("utf-8"))
    datos["objects"]["fantasma"] = {"6": {"status": "ok"}}
    ruta.write_text(json.dumps(datos), "utf-8")


def marca_equivocada(raiz: Path, prediccion: Path) -> None:
    """ Marca la predicción de 'a' como producida por el criterio actual. """
    documento = json.loads(prediccion.read_text("utf-8"))
    documento["n_views_predictions"]["6"]["triangulation"]["plane_score"] = "camaras"
    prediccion.write_text(json.dumps(documento), "utf-8")
    escribir_evaluacion(raiz)


def test_overwrite_borra_lo_viejo() -> None:
    """ Con --overwrite, un objeto sin predicción no debe conservar el archivo anterior. """
    with tempfile.TemporaryDirectory() as temporal:
        raiz = Path(temporal)
        prediccion = preparar(raiz)
        entrada = raiz / "plane_sym" / "a" / "224" / "flat" / f"molmo_multiview_{ENTRADA}.json"
        grupos = json.loads(entrada.read_text("utf-8"))
        grupos["6"]["points_by_image"] = {}
        entrada.write_text(json.dumps(grupos), "utf-8")
        assert procesar_objeto(raiz / "plane_sym" / "a", ENTRADA, SALIDA, 224, "flat", "datos",
                               [6, 14, 26], True) == "sin_prediccion"
        assert not prediccion.exists()
        print("ok overwrite borra la predicción vieja")


if __name__ == "__main__":
    test_overwrite_borra_lo_viejo()
    caso("predicción alterada", predicciones_mezcladas, "distinta de la recalculada")
    caso("heredada de otra configuración", heredada_de_otra_configuracion,
         "sin archivo de puntos")
    caso("grupo sin vistas", grupo_sin_vistas, "sin vistas suficientes")
    caso("predicción antigua", prediccion_antigua, "anterior a la fecha")
    caso("evaluación vieja", evaluacion_vieja, "Evaluación más antigua")
    caso("evaluación de más", evaluacion_de_mas, "Evaluación sin predicción actual")
    caso("marca equivocada", marca_equivocada, "marcas del estimador")
