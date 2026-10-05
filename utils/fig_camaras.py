"""
Figura: renderizado multivista con muestreo de Fibonacci sobre una esfera.
(a) 114 cámaras distribuidas sobre la esfera, todas orientadas hacia el origen.
(b) Geometría de una cámara: centro C = -R T, ejes locales (columnas de R),
    frustum (FoV = 60°), azimut y elevación, y parámetros guardados por vista.

Usa la misma convención que ImagesGenerator/export_fibonacci_views.py y
PyTorch3D (look_at_view_transform):
  - eje vertical del mundo: +Y ("arriba" de todas las cámaras);
  - Fibonacci con y de 1 a -1, x = cos(theta) r_y, z = sin(theta) r_y;
  - azimut = atan2(z, x), elevación = asin(y / r);
  - ejes de cámara: z_cam = eje óptico (hacia el origen),
    x_cam = up x z_cam, y_cam = z_cam x x_cam; son las columnas de R en el
    convenio de vector-fila p_cam = p_world R + T.

Genera PDF vectorial (para LaTeX) y PNG.

Uso:
  python utils/fig_camaras.py                       # salida por defecto
  python utils/fig_camaras.py --out-dir <carpeta>   # otra carpeta
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
})

N_VIEWS = 114
FOV_DEG = 60.0
RADIUS = 1.0
UP = np.array([0.0, 1.0, 0.0])          # "arriba" de las cámaras (PyTorch3D)

REPO = Path(__file__).resolve().parents[1]
OUT_DEFAULT = REPO / "Tesis_DCC_MDS" / "imgs" / "chapter_3"


# ---------------------------------------------------------------- utilidades
def fibonacci_sphere(n, r=1.0):
    """Igual que sample_fibonacci_viewpoints(): y va de 1 a -1."""
    i = np.arange(n)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - 2.0 * i / (n - 1)
    rho = np.sqrt(1.0 - y ** 2)
    theta = golden_angle * i
    return r * np.stack([rho * np.cos(theta), y, rho * np.sin(theta)], axis=1)


def look_at_axes(c):
    """Ejes de una cámara en c que mira al origen (convención PyTorch3D)."""
    z = -c / np.linalg.norm(c)                 # eje óptico, hacia el origen
    up = UP if abs(np.dot(z, UP)) < 0.99 else np.array([0.0, 0.0, 1.0])
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return x, y, z


def frustum_faces(c, depth, fov_deg=FOV_DEG):
    x, y, z = look_at_axes(c)
    h = depth * np.tan(np.radians(fov_deg) / 2)
    center = c + depth * z
    corners = [center + sx * h * x + sy * h * y
               for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    sides = [[c, corners[k], corners[(k + 1) % 4]] for k in range(4)]
    return sides, corners


def draw_frustum(ax, c, depth, color, alpha=0.25, lw=0.6):
    sides, corners = frustum_faces(c, depth)
    ax.add_collection3d(Poly3DCollection(
        sides, facecolor=color, edgecolor=color, alpha=alpha, linewidths=lw))
    ax.add_collection3d(Poly3DCollection(
        [corners], facecolor=color, edgecolor=color, alpha=alpha + 0.15,
        linewidths=lw))


def draw_box(ax, s=0.18, color="0.55"):
    """Cubo que representa el objeto en el origen."""
    v = np.array([[x, y, z] for x in (-s, s) for y in (-s, s) for z in (-s, s)])
    faces = [[v[0], v[1], v[3], v[2]], [v[4], v[5], v[7], v[6]],
             [v[0], v[1], v[5], v[4]], [v[2], v[3], v[7], v[6]],
             [v[0], v[2], v[6], v[4]], [v[1], v[3], v[7], v[5]]]
    ax.add_collection3d(Poly3DCollection(
        faces, facecolor=color, edgecolor="0.3", alpha=0.6, linewidths=0.5))


def sphere_wire(ax, r=1.0, color="0.85"):
    """Malla de la esfera con los polos en el eje Y."""
    uu, vv = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    ax.plot_wireframe(r * np.cos(uu) * np.sin(vv), r * np.cos(vv),
                      r * np.sin(uu) * np.sin(vv), color=color, linewidth=0.3)


def clean_axes(ax, lim):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()


def make_figure():
    pts = fibonacci_sphere(N_VIEWS, RADIUS)
    fig = plt.figure(figsize=(11, 5.0))

    # ------------------------ (a) distribución de cámaras sobre la esfera
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    sphere_wire(ax1, RADIUS)
    draw_box(ax1)
    sc = ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 1], cmap="viridis",
                     s=14, depthshade=True, edgecolors="k", linewidths=0.2)
    # Unas pocas cámaras con su frustum, para mostrar que miran al origen
    for idx in [5, 23, 40, 57, 74, 91, 108]:
        draw_frustum(ax1, pts[idx], depth=0.28, color="tab:red", alpha=0.18)
    ax1.view_init(elev=18, azim=35, vertical_axis="y")
    clean_axes(ax1, 1.05)
    cb = fig.colorbar(sc, ax=ax1, shrink=0.5, pad=0.0)
    cb.set_label("altura $y$ de la cámara")

    # ------------------------------------- (b) geometría de una cámara
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    az, el = np.radians(40), np.radians(30)
    D = 1.35
    # azimut = atan2(z, x), elevación = asin(y / r)
    C = D * np.array([np.cos(el) * np.cos(az), np.sin(el), np.cos(el) * np.sin(az)])

    # ejes del mundo
    L = 0.8
    for vec, lab in zip(np.eye(3), ["$x$", "$y$", "$z$"]):
        ax2.quiver(0, 0, 0, *(L * vec), color="0.3", arrow_length_ratio=0.08,
                   linewidth=1)
        ax2.text(*(1.1 * L * vec), lab, color="0.2")
    draw_box(ax2, s=0.1)

    # vector del origen a la cámara y su proyección en el plano horizontal xz
    ax2.plot([0, C[0]], [0, C[1]], [0, C[2]], color="tab:blue", lw=1.4)
    proj = np.array([C[0], 0, C[2]])
    ax2.plot([0, proj[0]], [0, 0], [0, proj[2]], "--", color="tab:blue", lw=0.9)
    ax2.plot([proj[0], C[0]], [0, C[1]], [proj[2], C[2]], ":", color="tab:blue",
             lw=0.9)
    ax2.scatter(*C, color="tab:red", s=30, zorder=5)
    ax2.text(C[0] + 0.06, C[1] + 0.10, C[2], r"$\mathbf{C} = -R\,T$",
             color="tab:red", fontsize=11)

    # arco de azimut (plano xz, desde +x) y de elevación (desde el plano xz)
    t = np.linspace(0, az, 40)
    ra = 0.35
    ax2.plot(ra * np.cos(t), 0 * t, ra * np.sin(t), color="tab:green", lw=1.3)
    ax2.text(0.42 * np.cos(az / 2), -0.16, 0.42 * np.sin(az / 2),
             r"azimut $=\mathrm{atan2}(z,x)$", color="tab:green", fontsize=9)
    s = np.linspace(0, el, 40)
    re = 0.6
    ax2.plot(re * np.cos(s) * np.cos(az), re * np.sin(s), re * np.cos(s) * np.sin(az),
             color="tab:purple", lw=1.3)
    # a la derecha de la línea punteada vertical, fuera del frustum
    ax2.text(1.06 * proj[0], 0.16, 1.06 * proj[2],
             "elevación\n" r"$=\arcsin(y/r)$", color="tab:purple", fontsize=9)

    # frustum y ejes locales de la cámara (columnas de R)
    draw_frustum(ax2, C, depth=0.4, color="tab:red", alpha=0.15)
    x, y, z = look_at_axes(C)
    for vec, col in [(z, "tab:red"), (x, "0.4"), (y, "0.4")]:
        ax2.quiver(*C, *(0.35 * vec), color=col, arrow_length_ratio=0.2, lw=1)
    ax2.text(*(C + 0.42 * x), r"$x_{cam}$", color="0.3", fontsize=10)
    ax2.text(*(C + 0.42 * y), r"$y_{cam}$", color="0.3", fontsize=10)
    ax2.text(*(C + 0.22 * z - 0.20 * y), r"$z_{cam}$", color="tab:red",
             fontsize=10)
    mid = C + 0.4 * z
    ax2.text(*(mid + 0.32 * y - 0.30 * x), r"FoV $=60^\circ$", color="tab:red",
             fontsize=10)

    ax2.view_init(elev=16, azim=-60, vertical_axis="y")
    # encuadre corrido hacia la cámara para dejar espacio a las etiquetas
    ctr = C / 2 + 0.12 * proj / np.linalg.norm(proj) + np.array([0.0, -0.08, 0.0])
    hr = 0.62
    ax2.set_xlim(ctr[0] - hr, ctr[0] + hr)
    ax2.set_ylim(ctr[1] - hr, ctr[1] + hr)
    ax2.set_zlim(ctr[2] - hr, ctr[2] + hr)
    ax2.set_box_aspect([1, 1, 1]); ax2.set_axis_off()

    plt.subplots_adjust(left=0.0, right=1.0, wspace=0.05, bottom=0.02, top=0.95)
    fig.text(0.24, 0.93, f"(a) {N_VIEWS} vistas con muestreo de Fibonacci",
             ha="center", fontsize=12)
    fig.text(0.74, 0.93, "(b) Geometría de una cámara virtual", ha="center",
             fontsize=12)

    # parámetros que se guardan por vista (mismas notaciones que la tesis)
    ax2.text2D(
        0.5, 0.0,
        "Por vista se guarda: "
        r"$R\in\mathbb{R}^{3\times 3}$ (columnas $x_{cam}, y_{cam}, z_{cam}$), "
        r"$T\in\mathbb{R}^{3}$, $\mathbf{C}=-R\,T$,"
        "\n"
        "azimut y elevación;  " r"$z_{cam}$" " = eje óptico.  "
        r"Convenio de vector-fila: $p_{\mathrm{cam}} = p_{\mathrm{world}}\,R + T$",
        transform=ax2.transAxes, ha="center", va="bottom", fontsize=9,
        linespacing=1.6,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="0.97", edgecolor="0.7"))
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT),
                    help=f"carpeta de salida (por defecto {OUT_DEFAULT})")
    ap.add_argument("--name", default="fig_camaras_fibonacci",
                    help="nombre base de los archivos")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig = make_figure()
    fig.savefig(out / f"{args.name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{args.name}.png", dpi=600, bbox_inches="tight")
    print(f"ok: {out / args.name}.pdf / .png")


if __name__ == "__main__":
    main()
