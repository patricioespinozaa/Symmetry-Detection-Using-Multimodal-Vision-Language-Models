#!/usr/bin/env python3
"""
Precision del pointing 2D de un VLM respecto a la linea de simetria planar verdadera,
en funcion de cuan de canto ve cada camara al plano.

Separa la capacidad del modelo (marcar la linea de simetria en la imagen) del metodo 3D
(triangulacion + S_dat): no reconstruye nada, solo mide en 2D.

Por cada objeto con un solo plano y cada imagen enviada al modelo (grupos 6/14/26):
  * rasteriza la malla con la camara guardada (PyTorch3D, mismas R/T/fov del render) y
    marca como "linea de simetria" los pixeles visibles cuya superficie esta a menos de
    --tol del plano GT;
  * mide la distancia de cada punto marcado a esa linea (transformada de distancia),
    normalizada por la diagonal del objeto en la imagen;
  * delta = angulo entre la direccion de la camara y el plano (0 = de canto);
  * referencias: (a) un punto al azar sobre el objeto, (b) el atajo "columna central"
    (x = 500 en la escala 0-1000, a la altura del punto marcado).

Salida: un CSV por punto (<out>.csv) y un resumen por tramo de delta en pantalla.

Uso:
  python3 pointing_2d_canto.py --renders-root /data/pespinoz/renders_rot_so3 \\
      --objects-root /data/pespinoz/objects_rot_so3 \\
      --molmo-json molmo_multiview_plane_v02_rot_so3.json --out pointing2d_rot_so3
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import FoVPerspectiveCameras, MeshRasterizer, RasterizationSettings
from scipy.ndimage import distance_transform_edt

TRAMOS = [(0, 5), (5, 15), (15, 30), (30, 60), (60, 90.01)]


def planos(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """ (normal unitaria, origen) de cada linea 'plane' de una etiqueta. """
    out = []
    for linea in path.read_text(encoding="utf-8").splitlines():
        t = linea.split()
        if t and t[0] == "plane" and len(t) >= 7:
            n = np.array([float(x) for x in t[1:4]])
            out.append((n / np.linalg.norm(n), np.array([float(x) for x in t[4:7]])))
    return out


def main() -> None:
    """ Mide la distancia 2D de los puntos del modelo a la linea de simetria GT. """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--renders-root", type=Path, required=True)
    ap.add_argument("--objects-root", type=Path, required=True)
    ap.add_argument("--molmo-json", required=True, help="nombre del JSON de puntos en cada carpeta 224/flat")
    ap.add_argument("--out", required=True, help="prefijo del CSV de salida")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--tol", type=float, default=0.01, help="distancia al plano (unidades del bbox)")
    ap.add_argument("--grupos", nargs="+", default=["6", "14", "26"])
    ap.add_argument("--max-objects", type=int, default=None)
    ap.add_argument("--semilla", type=int, default=0)
    args = ap.parse_args()

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    S = args.size
    raster = MeshRasterizer(raster_settings=RasterizationSettings(
        image_size=S, faces_per_pixel=1, bin_size=0, cull_backfaces=True))
    rng = np.random.default_rng(args.semilla)
    obj_dir = args.objects_root / "curated_plane_sym_obj"
    ids = sorted(p.stem for p in obj_dir.glob("*.txt") if len(planos(p)) == 1)
    if args.max_objects:
        ids = ids[:args.max_objects]

    filas = []
    for k, oid in enumerate(ids, 1):
        rdir = args.renders_root / "plane_sym" / oid / str(S) / "flat"
        jpath = rdir / args.molmo_json
        if not jpath.exists():
            continue
        (n, p0), = planos(obj_dir / f"{oid}.txt")
        datos = json.loads(jpath.read_text(encoding="utf-8"))
        mesh = load_objs_as_meshes([str(obj_dir / f"{oid}.obj")], load_textures=False, device=dev)
        fv = mesh.verts_packed()[mesh.faces_packed()]
        cache = {}                                            # indice de vista -> mapas
        for g in args.grupos:
            grupo = datos.get(g)
            if not isinstance(grupo, dict):
                continue
            for pos, cam_e in enumerate(grupo["images_sent"]):
                idx = cam_e["index"]
                if idx not in cache:
                    cam = FoVPerspectiveCameras(R=torch.tensor([cam_e["R"]], device=dev),
                                                T=torch.tensor([cam_e["T"]], device=dev),
                                                fov=60.0, device=dev)
                    fr = raster(mesh, cameras=cam)
                    p2f = fr.pix_to_face[0, ..., 0]
                    ok = p2f >= 0
                    pts = torch.zeros(S, S, 3, device=dev)
                    pts[ok] = (fr.bary_coords[0, ..., 0, :][ok][..., None] * fv[p2f[ok]]).sum(1)
                    objeto = ok.cpu().numpy()
                    curva = objeto & (np.abs((pts.cpu().numpy() - p0) @ n) < args.tol)
                    if curva.sum() < 2:
                        cache[idx] = None
                        continue
                    ys, xs = np.nonzero(objeto)
                    diag = float(np.hypot(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
                    eye = np.array(cam_e["eye"]) / np.linalg.norm(cam_e["eye"])
                    delta = float(np.degrees(np.arcsin(min(1.0, abs(eye @ n)))))
                    cache[idx] = (distance_transform_edt(~curva), objeto, ys, xs, diag, delta)
                if cache[idx] is None:
                    continue
                dist, objeto, ys, xs, diag, delta = cache[idx]
                for p in grupo["points_by_image"].get(str(pos), []):
                    px = min(S - 1, max(0, int(p["x"] / 1000 * S)))
                    py = min(S - 1, max(0, int(p["y"] / 1000 * S)))
                    j = rng.integers(len(ys))
                    filas.append({
                        "object_id": oid, "grupo": g, "vista": idx, "delta": round(delta, 2),
                        "diag_px": round(diag, 1),
                        "d_modelo": round(dist[py, px] / diag, 4),
                        "sobre_objeto": int(objeto[py, px]),
                        "d_azar": round(dist[ys[j], xs[j]] / diag, 4),
                        "d_columna": round(dist[py, S // 2] / diag, 4),
                    })
        if k % 100 == 0:
            print(f"  {k}/{len(ids)}", flush=True)

    with open(f"{args.out}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)

    d = {c: np.array([r[c] for r in filas], float) for c in ("delta", "d_modelo", "d_azar", "d_columna")}
    print(f"\n{args.out}: {len(filas)} puntos de {len({r['object_id'] for r in filas})} objetos")
    print("acierto = punto a <= 3% de la diagonal del objeto de la linea de simetria GT")
    print(f"{'delta':>9} {'puntos':>7} {'modelo med':>10} {'acierto':>8} {'azar med':>9} "
          f"{'acierto':>8} {'columna med':>11} {'acierto':>8}")
    for lo, hi in TRAMOS + [(0, 90.01)]:
        m = (d["delta"] >= lo) & (d["delta"] < hi)
        if not m.any():
            continue
        cel = lambda c: f"{np.median(d[c][m]):10.3f} {np.mean(d[c][m] <= 0.03):8.2f}"
        etiqueta = "todas" if (lo, hi) == (0, 90.01) else f"{lo:.0f}-{min(hi, 90):.0f}"
        print(f"{etiqueta:>9} {m.sum():>7} {cel('d_modelo')} {cel('d_azar').replace('  ', ' ', 1):>18} "
              f"{cel('d_columna')}")


if __name__ == "__main__":
    main()
