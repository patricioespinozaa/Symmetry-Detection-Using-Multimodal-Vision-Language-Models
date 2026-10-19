"""
Cuenta cuántos tokens recibe Molmo2 según la resolución y el número de vistas,
y extrae indicadores de saturación de las corridas ya hechas.

Dos modos (ver contexto_tokens_molmo2.md):

  procesador  (por defecto) Arma el mensaje EXACTAMENTE como
              molmo_multiview_runner.py::_call_model (texto primero, luego
              las N imágenes; prompt single si n_views == 1 y multi si
              n_views > 1, como --prompt-mode auto) y lo pasa por el
              procesador de Molmo2. No ejecuta el modelo: corre en CPU, sin
              GPU. Como el número de tokens depende del TAMAÑO de las
              imágenes y no de su contenido, usa imágenes grises sintéticas
              salvo que se pase --renders.

  json        No necesita transformers. Recorre los molmo_multiview_<EXP>.json
              ya generados bajo --renders-root y resume, por
              (symmetry_type, experimento, size, lighting, n_views), el campo
              real n_input_tokens que el runner guarda en cada llamada, junto
              con indicadores de saturación relevantes para el pipeline sin
              malla (vistas con >= 2 puntos, salidas sin bloque <points>,
              índices de imagen fuera de rango, puntos duplicados).

Uso:
    # Conteo con el prompt real del registro (prompts/axis/v04_*.txt)
    python contar_tokens_molmo2.py --prompt-id axis_v04
    python contar_tokens_molmo2.py --prompt-id plane_v05_1 --vistas 6 14 26

    # Control: un solo recorte por imagen
    python contar_tokens_molmo2.py --prompt-id axis_v04 --max-crops 1 --salida tokens_maxcrops1.csv

    # Con renders reales de un objeto (carpeta <obj>/<size>/<lighting>)
    python contar_tokens_molmo2.py --prompt-id axis_v04 --renders ruta/a/obj/448/flat --resoluciones 448

    # Tokens reales e indicadores de saturación desde las corridas existentes
    python contar_tokens_molmo2.py --modo json --renders-root /ruta/renders \
        --experimentos axis_v04 plane_v05_1 --salida saturacion_corridas.csv

Salida: CSV (una fila por resolución x número de vistas en modo procesador;
una fila por symmetry_type x experimento x size x lighting x n_views en modo json).
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Para importar prompts_registry (vive en MolmoPointing/, un nivel arriba).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL_ID = "allenai/Molmo2-8B"

# IDs de tokens especiales de imagen, tomados del config.json de Molmo2-8B.
IMAGE_START = 151936
IMAGE_END = 151937
IMAGE_PATCH = 151938          # parches de los recortes de alta resolución
IMAGE_COL = 151939            # separador de columnas entre recortes
LOW_RES_START = 151940
IMAGE_LOW_RES = 151942        # parches de la vista global de baja resolución
IDS_IMAGEN = {IMAGE_START, IMAGE_END, IMAGE_PATCH, IMAGE_COL,
              LOW_RES_START, IMAGE_LOW_RES}

LARGO_SFT = 16_384            # largo de secuencia del fine-tuning principal (paper, sec. 3.2)
CONTEXTO_MAX = 36_864         # max_position_embeddings (config.json)

# Mismos valores por defecto que molmo_multiview_runner.py (DEFAULT_VIEW_GROUPS /
# DEFAULT_IMAGE_SIZES). Copiados en vez de importados para no cargar torch.
VISTAS_POR_DEFECTO = [1, 6, 14, 26, 42, 62, 86, 114]
RESOLUCIONES_POR_DEFECTO = [224, 448, 1136]

# Mínimo de puntos por vista que necesita la triangulación del pipeline sin
# malla para definir el plano de interpretación (docs/pipeline_sin_malla.md §3).
MIN_PUNTOS_POR_VISTA = 2

RE_COORDS = re.compile(r'coords="([^"]*)"')


# ---------------------------------------------------------------------------
# Modo "procesador"
# ---------------------------------------------------------------------------

def cargar_prompts(prompt_id):
    """Devuelve (prompt_single, prompt_multi) desde prompts_registry."""
    from prompts_registry import get_prompt
    entrada = get_prompt(prompt_id)
    return entrada["single"], entrada["multi"]


def cargar_imagenes(n, resolucion, carpeta_renders=None):
    """Devuelve n imágenes de resolucion x resolucion."""
    from PIL import Image
    if carpeta_renders:
        rutas = sorted(Path(carpeta_renders).glob("*.png"))
        if len(rutas) < n:
            raise ValueError(f"Hay {len(rutas)} imágenes en {carpeta_renders}, se piden {n}.")
        # Mismo criterio que get_n_views_entries: n índices equiespaciados.
        idx = sorted({int(round(i)) for i in np.linspace(0, len(rutas) - 1, n)})
        return [Image.open(rutas[i]).convert("RGB").resize((resolucion, resolucion))
                for i in idx]
    return [Image.new("RGB", (resolucion, resolucion), (178, 178, 178)) for _ in range(n)]


def contar(processor, imagenes, prompt):
    """Arma el mensaje igual que molmo_multiview_runner.py::_call_model y cuenta tokens."""
    # Orden real del runner: primero el texto, después las imágenes.
    contenido = [{"type": "text", "text": prompt}]
    contenido.extend({"type": "image", "image": im} for im in imagenes)
    mensajes = [{"role": "user", "content": contenido}]

    entradas = processor.apply_chat_template(
        mensajes,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    ids = entradas["input_ids"][0].tolist()

    tokens_imagen = sum(1 for t in ids if t in IDS_IMAGEN)
    return {
        "tokens_total": len(ids),
        "tokens_imagen": tokens_imagen,
        "tokens_texto": len(ids) - tokens_imagen,
        # Cuántas imágenes llegaron realmente al modelo (detecta vistas descartadas).
        "imagenes_en_entrada": ids.count(LOW_RES_START) or ids.count(IMAGE_START),
        "parches_alta_res": ids.count(IMAGE_PATCH),
        "parches_baja_res": ids.count(IMAGE_LOW_RES),
    }


def modo_procesador(args):
    from transformers import AutoProcessor

    prompt_single, prompt_multi = cargar_prompts(args.prompt_id)

    # Mismos kwargs que get_model() en el runner.
    kwargs = {"trust_remote_code": True, "use_fast": True}
    processor = AutoProcessor.from_pretrained(MODEL_ID, **kwargs)
    if args.max_crops is not None:
        processor.image_processor.max_crops = args.max_crops
    print("max_crops efectivo:", getattr(processor.image_processor, "max_crops", "desconocido"))

    filas = []
    for res in args.resoluciones:
        for nv in args.vistas:
            imagenes = cargar_imagenes(nv, res, args.renders)
            prompt = prompt_single if nv == 1 else prompt_multi
            r = contar(processor, imagenes, prompt)
            r.update({
                "prompt_id": args.prompt_id,
                "resolucion": res,
                "vistas": nv,
                "max_crops": args.max_crops if args.max_crops is not None else "defecto",
                "tokens_por_imagen": round(r["tokens_imagen"] / nv, 1),
                "supera_largo_sft": r["tokens_total"] > LARGO_SFT,
                "supera_contexto_max": r["tokens_total"] > CONTEXTO_MAX,
                "imagenes_perdidas": nv - r["imagenes_en_entrada"],
            })
            filas.append(r)
            aviso = ""
            if r["supera_contexto_max"]:
                aviso = "  <-- SUPERA EL CONTEXTO MÁXIMO"
            elif r["supera_largo_sft"]:
                aviso = "  <-- supera el largo de entrenamiento"
            print(f"{res:>5} px | {nv:>3} vistas | {r['tokens_total']:>7} tokens "
                  f"({r['tokens_por_imagen']} por imagen){aviso}")

    columnas = ["prompt_id", "resolucion", "vistas", "max_crops", "tokens_total",
                "tokens_imagen", "tokens_texto", "tokens_por_imagen",
                "imagenes_en_entrada", "imagenes_perdidas", "parches_alta_res",
                "parches_baja_res", "supera_largo_sft", "supera_contexto_max"]
    return filas, columnas


# ---------------------------------------------------------------------------
# Modo "json"
# ---------------------------------------------------------------------------

def indices_crudos(raw_output):
    """
    Índices de imagen (1-based) que el modelo escribió en su salida cruda, antes
    del filtro de parse_multi_coords (que descarta en silencio los fuera de rango).
    """
    indices = []
    for bloque in RE_COORDS.findall(raw_output):
        for grupo in bloque.split(";"):
            tokens = grupo.split()
            if tokens and tokens[0].isdigit():
                indices.append(int(tokens[0]))
    return indices


def indicadores_llamada(entrada):
    """Indicadores de saturación de una llamada (una clave n_views del JSON)."""
    n_imagenes = len(entrada.get("images_sent", []))
    puntos = entrada.get("points_by_image", {})
    raw = entrada.get("raw_output", "")
    # --prompt-mode single guarda una lista de salidas (una por imagen).
    raws = raw if isinstance(raw, list) else [raw]

    vistas_utiles = sum(1 for pts in puntos.values() if len(pts) >= MIN_PUNTOS_POR_VISTA)
    n_pts = sum(len(pts) for pts in puntos.values())
    duplicados = sum(
        len(pts) - len({(p["x"], p["y"]) for p in pts}) for pts in puntos.values()
    )
    sin_bloque = sum(1 for r in raws if "<points" not in r)
    fuera_rango = 0
    if not isinstance(raw, list) and n_imagenes > 1:
        fuera_rango = sum(1 for i in indices_crudos(raw) if i < 1 or i > n_imagenes)

    return {
        "n_input_tokens": entrada.get("n_input_tokens"),
        "flow": entrada.get("flow", "a"),
        "prompt_mode": entrada.get("prompt_mode"),
        "n_imagenes": n_imagenes,
        "frac_vistas_con_puntos": entrada.get("n_images_with_points", len(puntos)) / max(n_imagenes, 1),
        "frac_vistas_utiles": vistas_utiles / max(n_imagenes, 1),
        "puntos_por_vista": n_pts / max(n_imagenes, 1),
        "frac_puntos_duplicados": duplicados / max(n_pts, 1),
        "salida_sin_bloque": sin_bloque > 0,
        "grupos_fuera_de_rango": fuera_rango,
    }


def experimento_de(nombre_archivo):
    """molmo_multiview_axis_v04.json -> 'axis_v04'; molmo_multiview.json -> '(produccion)'."""
    base = nombre_archivo[: -len(".json")]
    return base[len("molmo_multiview_"):] if base != "molmo_multiview" else "(produccion)"


def modo_json(args):
    raiz = Path(args.renders_root)
    grupos = defaultdict(list)
    for sym in args.symmetry_types:
        # <renders_root>/<symmetry_type>/<object_id>/<size>/<lighting>/molmo_multiview*.json
        for ruta in sorted((raiz / sym).glob("*/*/*/molmo_multiview*.json")):
            exp = experimento_de(ruta.name)
            if args.experimentos and exp not in args.experimentos:
                continue
            size, lighting = ruta.parent.parent.name, ruta.parent.name
            with open(ruta, encoding="utf-8") as f:
                datos = json.load(f)
            for n_views, entrada in datos.items():
                if not isinstance(entrada, dict) or "images_sent" not in entrada:
                    continue
                grupos[(sym, exp, int(size), lighting, int(n_views))].append(
                    indicadores_llamada(entrada))

    if not grupos:
        print(f"[aviso] No se encontraron molmo_multiview*.json bajo {raiz}")

    filas = []
    for (sym, exp, size, lighting, nv), llamadas in sorted(grupos.items()):
        tokens = [c["n_input_tokens"] for c in llamadas if c["n_input_tokens"] is not None]
        flows = sorted({c["flow"] for c in llamadas})
        media = lambda k: round(float(np.mean([c[k] for c in llamadas])), 4)
        fila = {
            "symmetry_type": sym, "experimento": exp, "size": size,
            "lighting": lighting, "n_views": nv, "n_llamadas": len(llamadas),
            "flow": "/".join(flows),
            "prompt_mode": "/".join(sorted({str(c["prompt_mode"]) for c in llamadas})),
            "tokens_media": round(float(np.mean(tokens))) if tokens else "",
            "tokens_min": min(tokens) if tokens else "",
            "tokens_max": max(tokens) if tokens else "",
            "frac_supera_largo_sft": round(float(np.mean([t > LARGO_SFT for t in tokens])), 4) if tokens else "",
            "frac_supera_contexto_max": round(float(np.mean([t > CONTEXTO_MAX for t in tokens])), 4) if tokens else "",
            "frac_vistas_con_puntos": media("frac_vistas_con_puntos"),
            "frac_vistas_utiles": media("frac_vistas_utiles"),
            "puntos_por_vista": media("puntos_por_vista"),
            "frac_puntos_duplicados": media("frac_puntos_duplicados"),
            "frac_salidas_sin_bloque": media("salida_sin_bloque"),
            "frac_llamadas_fuera_de_rango": round(float(np.mean(
                [c["grupos_fuera_de_rango"] > 0 for c in llamadas])), 4),
        }
        filas.append(fila)
        aviso = "  (flow b/c: n_input_tokens incluye la llamada previa de descripción)" \
            if flows != ["a"] else ""
        print(f"{sym:<9} {exp:<14} {size:>5} px {lighting:<8} {nv:>3} vistas | "
              f"{fila['tokens_media']:>7} tokens | vistas útiles {fila['frac_vistas_utiles']:.2f}{aviso}")

    columnas = ["symmetry_type", "experimento", "size", "lighting", "n_views",
                "n_llamadas", "flow", "prompt_mode", "tokens_media", "tokens_min",
                "tokens_max", "frac_supera_largo_sft", "frac_supera_contexto_max",
                "frac_vistas_con_puntos", "frac_vistas_utiles", "puntos_por_vista",
                "frac_puntos_duplicados", "frac_salidas_sin_bloque",
                "frac_llamadas_fuera_de_rango"]
    return filas, columnas


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modo", choices=["procesador", "json"], default="procesador")
    # Modo procesador
    p.add_argument("--prompt-id", default="axis_v04",
                   help="Variante de prompts_registry (p. ej. axis_v04, plane_v05_1).")
    p.add_argument("--resoluciones", type=int, nargs="+", default=RESOLUCIONES_POR_DEFECTO)
    p.add_argument("--vistas", type=int, nargs="+", default=VISTAS_POR_DEFECTO)
    p.add_argument("--max-crops", type=int, default=None,
                   help="Sobrescribe processor.image_processor.max_crops (por defecto 8).")
    p.add_argument("--renders", default=None,
                   help="Carpeta <obj>/<size>/<lighting> con renders reales (opcional).")
    # Modo json
    p.add_argument("--renders-root", default=None,
                   help="Raíz de renders con los molmo_multiview*.json (modo json).")
    p.add_argument("--symmetry-types", nargs="+", default=["axis_sym", "plane_sym"])
    p.add_argument("--experimentos", nargs="+", default=None,
                   help="Filtra por experiment_id (p. ej. axis_v04 plane_v05_1). Por defecto, todos.")
    p.add_argument("--salida", default=None,
                   help="CSV de salida (por defecto tokens_molmo2.csv o saturacion_corridas.csv).")
    args = p.parse_args()

    if args.modo == "json":
        if not args.renders_root:
            p.error("--modo json requiere --renders-root")
        filas, columnas = modo_json(args)
        salida = args.salida or "saturacion_corridas.csv"
    else:
        filas, columnas = modo_procesador(args)
        salida = args.salida or "tokens_molmo2.csv"

    with open(salida, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columnas)
        w.writeheader()
        w.writerows(filas)
    print(f"\nGuardado en {os.path.abspath(salida)}")


if __name__ == "__main__":
    main()
