#!/usr/bin/env python3
"""
Verifica la conversión Molmo2 -> NDC -> rayo de pipeline_common/camera.py
contra la referencia independiente: la cámara de PyTorch3D construida con los
mismos R, T y FoV con que se generaron los renders, y los PNG reales.

Compara la versión corregida (2026-10-05, +X a la izquierda) con la anterior
(convención OpenGL, +X a la derecha).

Pruebas:
  1. Ejes de la cámara a partir de R guardado (sin usar camera.py): ¿+x_cam
     apunta a la derecha o a la izquierda del observador?
  2. PyTorch3D vs PNG: la silueta proyectada por la cámara de PyTorch3D debe
     coincidir con el render (valida que PyTorch3D sea una referencia fiable y
     que el PNG no se guardó volteado).
  3. Proyección: posición en la imagen según PyTorch3D vs. project_point()
     (corregida y anterior), sobre miles de puntos 3D.
  4. Rayos: dirección del rayo de un píxel según PyTorch3D (unproject) vs.
     build_camera_rays(molmo_to_ndc(...)) (corregida y anterior).
  5. De punta a punta: puntos perfectos sobre el eje de referencia, proyectados
     con PyTorch3D, triangulados con pipeline_common/triangulation.py. La
     versión correcta debe recuperar el eje con error ~0°.

Uso (desde la raíz del repo, con PyTorch3D instalado):
  python utils/verificar_convencion_ndc.py
  python utils/verificar_convencion_ndc.py --renders <carpeta flat> --obj <.obj> --txt <.txt>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pipeline_common.camera import build_camera_rays, molmo_to_ndc, project_point  # noqa: E402
from pipeline_common.triangulation import interpretation_plane_normal, triangulate_line  # noqa: E402
import trimesh  # noqa: E402


def _importar_camaras_p3d():
    """Importa FoVPerspectiveCameras sin ejecutar pytorch3d/renderer/__init__.py,
    que carga la extensión compilada _C (no necesaria para las cámaras, y que
    en algunos Windows queda bloqueada). cameras.py es Python puro."""
    try:
        from pytorch3d.renderer import FoVPerspectiveCameras
        return FoVPerspectiveCameras
    except ImportError:
        import types
        import pytorch3d
        pkg = types.ModuleType("pytorch3d.renderer")
        pkg.__path__ = [str(Path(pytorch3d.__file__).parent / "renderer")]
        sys.modules["pytorch3d.renderer"] = pkg
        from pytorch3d.renderer.cameras import FoVPerspectiveCameras
        return FoVPerspectiveCameras


FoVPerspectiveCameras = _importar_camaras_p3d()

FOV = 60.0


# --- versión anterior (OpenGL), reproducida aquí solo para comparar
def molmo_to_ndc_anterior(x, y):
    return (x / 1000.0) * 2.0 - 1.0, 1.0 - (y / 1000.0) * 2.0


def project_point_anterior(p, R, T, fov, size):
    px, py, behind = project_point(p, R, T, fov, size)
    return size - px, py, behind          # la anterior es el espejo horizontal


def cargar(renders):
    m = json.loads((renders / "metadata_all.json").read_text())
    return m if isinstance(m, list) else next(v for v in m.values() if isinstance(v, list))


def camara_p3d(e, H, W):
    R = torch.tensor([e["R"]], dtype=torch.float32)
    T = torch.tensor([e["T"]], dtype=torch.float32)
    return FoVPerspectiveCameras(R=R, T=T, fov=FOV), (H, W)


def p3d_a_molmo(cam, hw, pts):
    """Puntos 3D -> coordenadas Molmo2 (0-1000, origen arriba-izquierda) según PyTorch3D."""
    s = cam.transform_points_screen(torch.tensor(pts, dtype=torch.float32)[None],
                                    image_size=(hw,))[0, :, :2].numpy()
    return s[:, 0] / hw[1] * 1000.0, s[:, 1] / hw[0] * 1000.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--renders", default=str(REPO / "Examples/renders/axis_sym/axis_sym_obj/448/flat"))
    ap.add_argument("--obj", default=str(REPO / "Examples/objects/axis_sym_obj.obj"))
    ap.add_argument("--txt", default=str(REPO / "Examples/objects/axis_sym_obj.txt"))
    args = ap.parse_args()

    renders = Path(args.renders)
    meta = cargar(renders)
    mesh = trimesh.load(args.obj, force="mesh")
    vistas = [e for e in meta if abs(e["elevation"]) < 70][::7]
    rng = np.random.default_rng(0)

    # ------------------------------------------------------------- prueba 1
    print("=" * 74)
    print("1. Ejes de cámara desde R guardado (sin camera.py)")
    print("=" * 74)
    signos = []
    for e in vistas:
        R = np.array(e["R"]); eye = np.asarray(e["eye"], dtype=float).reshape(-1)[:3]
        f = -eye / np.linalg.norm(eye)                    # mira al origen
        derecha = np.cross(f, [0, 1, 0]); derecha /= np.linalg.norm(derecha)
        signos.append(float(R[:, 0] @ derecha))           # columna 0 de R = +x_cam en el mundo
    print(f"   x_cam · (derecha del observador) en {len(signos)} vistas: "
          f"min {min(signos):+.4f}, max {max(signos):+.4f}")
    print("   -> -1 significa que +x_cam apunta a la IZQUIERDA de la imagen")

    # ------------------------------------------------------------- prueba 2
    print("\n" + "=" * 74)
    print("2. PyTorch3D vs PNG real (IoU de siluetas)")
    print("=" * 74)
    verts = np.asarray(mesh.sample(30000))
    from scipy.ndimage import binary_dilation, binary_fill_holes
    for e in vistas[:6]:
        img = np.asarray(Image.open(renders / e["filename"]).convert("L"))
        H, W = img.shape
        real = img < 245
        cam, hw = camara_p3d(e, H, W)
        mx, my = p3d_a_molmo(cam, hw, verts)
        pred = np.zeros_like(real)
        px = (mx / 1000 * W).astype(int); py = (my / 1000 * H).astype(int)
        ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        pred[py[ok], px[ok]] = True
        pred = binary_fill_holes(binary_dilation(pred, iterations=2))
        iou = lambda a, b: (a & b).sum() / max(1, (a | b).sum())
        print(f"   vista {e['index']:3d}: IoU directo {iou(pred, real):.3f} | "
              f"con espejo {iou(pred, real[:, ::-1]):.3f}")

    # ------------------------------------------------------------- prueba 3
    print("\n" + "=" * 74)
    print("3. Proyección: PyTorch3D vs project_point (unidades Molmo2, 0-1000)")
    print("=" * 74)
    pts = rng.uniform(-0.5, 0.5, size=(2000, 3))
    for nombre, fn in (("corregida", project_point), ("anterior ", project_point_anterior)):
        errs = []
        for e in vistas:
            cam, hw = camara_p3d(e, 448, 448)
            mx, my = p3d_a_molmo(cam, hw, pts)
            for p, x, y in zip(pts, mx, my):
                qx, qy, _ = fn(p, e["R"], e["T"], FOV, 448)
                errs.append(np.hypot(qx / 448 * 1000 - x, qy / 448 * 1000 - y))
        errs = np.array(errs)
        print(f"   {nombre}: error medio {errs.mean():9.4f}  máximo {errs.max():9.4f}  "
              f"({len(errs)} puntos x vistas)")

    # ------------------------------------------------------------- prueba 4
    print("\n" + "=" * 74)
    print("4. Rayos: los puntos del rayo de build_camera_rays, proyectados por")
    print("   PyTorch3D, deben caer en el mismo píxel de origen (unidades Molmo2)")
    print("=" * 74)
    xy = rng.uniform(20, 980, size=(500, 2))
    for nombre, conv in (("corregida", molmo_to_ndc), ("anterior ", molmo_to_ndc_anterior)):
        errs = []
        for e in vistas:
            cam, hw = camara_p3d(e, 448, 448)
            for x, y in xy:
                C, d = build_camera_rays(*conv(x, y), e["R"], e["T"], FOV, 448)
                sobre_rayo = np.array([C + lam * d for lam in (0.5, 1.0, 2.0)])
                mx, my = p3d_a_molmo(cam, hw, sobre_rayo)
                errs.extend(np.hypot(mx - x, my - y))
        errs = np.array(errs)
        print(f"   {nombre}: error medio {errs.mean():9.4f}  máximo {errs.max():9.4f}")

    # ------------------------------------------------------------- prueba 5
    print("\n" + "=" * 74)
    print("5. De punta a punta: puntos perfectos sobre el eje de referencia")
    print("=" * 74)
    tok = next(l.split() for l in Path(args.txt).read_text().splitlines()
               if l.split() and l.split()[0] == "axis")
    u = np.array([float(t) for t in tok[1:4]]); u /= np.linalg.norm(u)
    o = np.array([float(t) for t in tok[4:7]])
    eje_pts = np.array([o + 0.25 * u, o - 0.25 * u])
    for nombre, conv in (("corregida", molmo_to_ndc), ("anterior ", molmo_to_ndc_anterior)):
        Cs, Ns = [], []
        for e in vistas:
            cam, hw = camara_p3d(e, 448, 448)
            mx, my = p3d_a_molmo(cam, hw, eje_pts)
            Ca, da = build_camera_rays(*conv(mx[0], my[0]), e["R"], e["T"], FOV, 448)
            _, db = build_camera_rays(*conv(mx[1], my[1]), e["R"], e["T"], FOV, 448)
            n = interpretation_plane_normal(da, db)
            if n is not None:
                Cs.append(Ca); Ns.append(n)
        _, d = triangulate_line(Cs, Ns)
        err = np.degrees(np.arccos(min(1.0, abs(d @ u))))
        print(f"   {nombre}: error angular del eje triangulado = {err:.4f}°  ({len(Ns)} vistas)")


if __name__ == "__main__":
    main()
