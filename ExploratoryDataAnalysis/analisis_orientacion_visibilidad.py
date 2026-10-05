#!/usr/bin/env python3
"""
Estadisticas y figuras de (1) la orientacion de los elementos de simetria de
referencia (ejes y normales de plano) y (2) su visibilidad desde las vistas
del pipeline.

No recalcula nada geometrico: consume las salidas de visibilidad_simetria.py.

Entradas (mismo --prefix que se paso como --out-prefix a visibilidad_simetria.py):
  <prefix>_elementos.csv     direccion (ux,uy,uz) de cada eje/normal
  <prefix>_objetos.csv       delta_min / n_buenas / frac_buenas por objeto y n_v

Orientacion. Para una direccion sin signo distribuida uniformemente en la
esfera, cada componente absoluta |u.X|, |u.Y|, |u.Z| es Uniforme[0,1]
(teorema de Arquimedes) y el eje del mundo mas cercano es X, Y o Z con
probabilidad 1/3. Se contrasta:
  - KS de |u.X|, |u.Y|, |u.Z| contra U[0,1] (p asintotico de Kolmogorov).
  - chi-cuadrado (gl=2) del eje del mundo mas cercano contra (1/3,1/3,1/3);
    con gl=2 el p-valor es exactamente exp(-chi2/2).
  - % de direcciones a <=5 y <=15 grados de un eje del mundo, frente al valor
    esperado bajo uniformidad (Monte Carlo con semilla fija).

Visibilidad. delta = arcsin(|u.w|): 0 = la vista ve el eje/plano de canto
(ideal), 90 = lo ve de frente. Una vista es "buena" si delta <= umbral (el
mismo umbral usado al correr visibilidad_simetria.py; por defecto 15).

Salidas en --out-dir:
  orientacion_resumen.csv / .tex          estadisticos por tipo de simetria
  orientacion_por_categoria.csv           alineacion por categoria
  visibilidad_resumen.csv / .tex          delta_min y vistas buenas por n_v
  fig_orientacion_esfera.pdf/.png         proyeccion de igual area (centro = +Y)
  fig_orientacion_componentes.pdf/.png    histogramas de |u.X|, |u.Y|, |u.Z|
  fig_orientacion_eje_cercano.pdf/.png    ECDF del angulo al eje mas cercano
  fig_visibilidad_delta.pdf/.png          delta_min (mediana, p10-p90) vs n_v
  fig_visibilidad_buenas.pdf/.png         distribucion de vistas buenas vs n_v

Uso:
  python3 analisis_orientacion_visibilidad.py --prefix visibilidad \\
      --out-dir figuras/orientacion_visibilidad
"""
import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paleta (validada con dataviz/validate_palette.js, modo claro: ALL PASS)
COLOR = {"axial": "#2a78d6", "planar": "#eb6834"}
ETIQUETA = {"axial": "Axial (ejes)", "planar": "Planar (normales)"}
TXT_1, TXT_2, GRID, REF = "#0b0b0b", "#52514e", "#dddcd8", "#8a8985"
# Rampa secuencial ordinal (azul, pasos 250-650) para 0 / 1 / 2 / >=3 vistas buenas
RAMPA = ["#86b6ef", "#3987e5", "#256abf", "#104281"]
AXES = ["X", "Y", "Z"]
TIPOS = ("axial", "planar")

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": TXT_2, "axes.titlecolor": TXT_1,
    "xtick.color": TXT_2, "ytick.color": TXT_2, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


# ----------------------------------------------------------------- estadistica
def ks_uniforme(x):
    """KS de una muestra contra U[0,1]. Devuelve (D, p asintotico)."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0:
        return float("nan"), float("nan")
    i = np.arange(1, n + 1)
    d = max(np.max(i / n - x), np.max(x - (i - 1) / n))
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    p = 2 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return float(d), float(min(1.0, max(0.0, p)))


def chi2_tercios(conteos):
    """chi-cuadrado contra (1/3,1/3,1/3); gl=2 => p = exp(-chi2/2)."""
    c = np.asarray(conteos, dtype=float)
    e = c.sum() / 3
    chi2 = float(((c - e) ** 2 / e).sum())
    return chi2, math.exp(-chi2 / 2)


def angulo_eje_cercano(u):
    """u: (N,3) unitarios. Angulo (grados) al eje del mundo mas cercano."""
    return np.degrees(np.arccos(np.clip(np.abs(u).max(axis=1), 0, 1)))


def referencia_uniforme(n=1_000_000, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return angulo_eje_cercano(v)


def fmt_p(p):
    """p-valor para LaTeX (modo matematico: '<' en texto OT1 se imprime como '¡')."""
    return "$<0.001$" if p < 1e-3 else f"{p:.3f}"


# ----------------------------------------------------------------- orientacion
def resumen_orientacion(el, ang_unif):
    filas = []
    esp5 = 100 * np.mean(ang_unif <= 5)
    esp15 = 100 * np.mean(ang_unif <= 15)
    for tipo in TIPOS:
        d = el[el.tipo_simetria == tipo]
        u = d[["ux", "uy", "uz"]].to_numpy()
        ang = angulo_eje_cercano(u)
        cerc = np.abs(u).argmax(axis=1)
        conteo = [int((cerc == k).sum()) for k in range(3)]
        chi2, p_chi = chi2_tercios(conteo)
        fila = {
            "tipo_simetria": tipo,
            "n_objetos": d.object_id.nunique(),
            "n_elementos": len(d),
            "pct_a_5deg": round(100 * np.mean(ang <= 5), 1),
            "pct_a_5deg_esperado_uniforme": round(esp5, 1),
            "pct_a_15deg": round(100 * np.mean(ang <= 15), 1),
            "pct_a_15deg_esperado_uniforme": round(esp15, 1),
            "angulo_eje_cercano_mediana": round(float(np.median(ang)), 1),
            "cercano_X": conteo[0], "cercano_Y": conteo[1], "cercano_Z": conteo[2],
            "chi2_eje_cercano": round(chi2, 1), "p_chi2_eje_cercano": p_chi,
        }
        for k, ax in enumerate(AXES):
            comp = np.abs(u[:, k])
            dks, pks = ks_uniforme(comp)
            fila[f"media_abs_u{ax}"] = round(float(comp.mean()), 3)
            fila[f"ks_D_{ax}"] = round(dks, 3)
            fila[f"ks_p_{ax}"] = pks
        filas.append(fila)
    return pd.DataFrame(filas)


def orientacion_por_categoria(el):
    u = el[["ux", "uy", "uz"]].to_numpy()
    el = el.assign(ang=angulo_eje_cercano(u),
                   eje=[AXES[k] for k in np.abs(u).argmax(axis=1)])
    filas = []
    for (tipo, cat), d in el.groupby(["tipo_simetria", "nombre_categoria"]):
        al = d[d.ang <= 5]
        filas.append({
            "tipo_simetria": tipo, "categoria": cat,
            "n_objetos": d.object_id.nunique(), "n_elementos": len(d),
            "pct_a_5deg": round(100 * len(al) / len(d), 1),
            "alineados_X": int((al.eje == "X").sum()),
            "alineados_Y": int((al.eje == "Y").sum()),
            "alineados_Z": int((al.eje == "Z").sum()),
        })
    return (pd.DataFrame(filas)
            .sort_values(["tipo_simetria", "n_objetos"], ascending=[True, False]))


def tex_orientacion(res, path):
    r = {t: res[res.tipo_simetria == t].iloc[0] for t in TIPOS}
    filas = [
        ("Objetos / elementos", lambda x: f"{x.n_objetos} / {x.n_elementos}"),
        ("A $\\leq 5^\\circ$ de un eje del mundo (\\%)",
         lambda x: f"{x.pct_a_5deg} ({x.pct_a_5deg_esperado_uniforme})"),
        ("A $\\leq 15^\\circ$ de un eje del mundo (\\%)",
         lambda x: f"{x.pct_a_15deg} ({x.pct_a_15deg_esperado_uniforme})"),
        ("Eje del mundo más cercano: $X$ / $Y$ / $Z$",
         lambda x: f"{x.cercano_X} / {x.cercano_Y} / {x.cercano_Z}"),
        ("$\\chi^2$ eje más cercano vs.\\ $(\\tfrac13,\\tfrac13,\\tfrac13)$, $p$",
         lambda x: f"{x.chi2_eje_cercano}, {fmt_p(x.p_chi2_eje_cercano)}"),
    ] + [
        (f"Media $|\\hat{{u}}\\cdot\\hat{{{ax}}}|$ (uniforme: 0.5); KS $p$",
         lambda x, ax=ax: f"{x[f'media_abs_u{ax}']}; {fmt_p(x[f'ks_p_{ax}'])}")
        for ax in AXES
    ]
    lineas = [
        "\\begin{table}[htbp]", "\\centering", "\\small",
        "\\caption{Orientación de los elementos de simetría de referencia. Entre "
        "paréntesis, el valor esperado para direcciones uniformes en la esfera.}",
        "\\label{tab:orientacion_gt}",
        "\\begin{tabular}{lcc}", "\\toprule",
        " & Axial (ejes) & Planar (normales) \\\\", "\\midrule",
    ]
    lineas += [f"{nom} & {f(r['axial'])} & {f(r['planar'])} \\\\" for nom, f in filas]
    lineas += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    Path(path).write_text("\n".join(lineas), encoding="utf-8")


def fig_esfera(el, out):
    """Proyeccion azimutal de igual area centrada en +Y (direcciones sin signo
    plegadas al hemisferio uy >= 0). Uniforme en la esfera => uniforme en el disco."""
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 3.3))
    R = math.sqrt(2)
    for ax, tipo in zip(axs, TIPOS):
        u = el.loc[el.tipo_simetria == tipo, ["ux", "uy", "uz"]].to_numpy().copy()
        u[u[:, 1] < 0] *= -1
        theta = np.arccos(np.clip(u[:, 1], -1, 1))
        rho = 2 * np.sin(theta / 2)
        h = np.hypot(u[:, 0], u[:, 2])
        h[h == 0] = 1
        x, y = rho * u[:, 0] / h, rho * u[:, 2] / h
        t = np.linspace(0, 2 * np.pi, 361)
        for elev in (15, 45, 75):  # anillos de elevacion sobre el plano XZ
            r = 2 * math.sin(math.radians(90 - elev) / 2)
            ax.plot(r * np.cos(t), r * np.sin(t), color=GRID, lw=0.6, zorder=1)
            ax.text(r * math.cos(math.radians(60)), r * math.sin(math.radians(60)),
                    f"{elev}°", color=TXT_2, fontsize=7, ha="left", va="bottom")
        ax.plot(R * np.cos(t), R * np.sin(t), color=TXT_2, lw=0.8, zorder=1)
        ax.plot([-R, R], [0, 0], color=GRID, lw=0.6, zorder=1)
        ax.plot([0, 0], [-R, R], color=GRID, lw=0.6, zorder=1)
        ax.scatter(x, y, s=9, color=COLOR[tipo], alpha=0.55, linewidths=0, zorder=3)
        for lbl, (lx, ly) in {"X": (R + 0.08, 0), "Z": (0, R + 0.08),
                              "Y": (0.06, 0.06)}.items():
            ax.text(lx, ly, lbl, color=TXT_1, fontsize=8.5, fontweight="bold",
                    ha="left" if lbl != "Z" else "center",
                    va="center" if lbl != "Z" else "bottom")
        ax.set_title(f"{ETIQUETA[tipo]}  (n = {len(u)})")
        ax.set_aspect("equal")
        ax.set_xlim(-R - 0.25, R + 0.3)
        ax.set_ylim(-R - 0.15, R + 0.3)
        ax.axis("off")
    fig.text(0.5, 0.04, "Centro: dirección vertical (+Y, el «arriba» de las cámaras). "
             "Borde: direcciones horizontales. Anillos: elevación sobre el plano XZ.",
             ha="center", color=TXT_2, fontsize=7.5)
    guardar(fig, out, "fig_orientacion_esfera")


def fig_componentes(el, out):
    fig, axs = plt.subplots(2, 3, figsize=(6.3, 3.8), sharex=True)
    bins = np.linspace(0, 1, 21)
    for i, tipo in enumerate(TIPOS):
        u = el.loc[el.tipo_simetria == tipo, ["ux", "uy", "uz"]].to_numpy()
        for k, axn in enumerate(AXES):
            ax = axs[i, k]
            ax.hist(np.abs(u[:, k]), bins=bins, density=True, color=COLOR[tipo],
                    edgecolor="white", linewidth=0.6, zorder=2)
            ax.axhline(1.0, color=REF, lw=1.2, ls="--", zorder=3)
            ax.set_xlim(0, 1)
            ax.grid(axis="x", visible=False)
            if i == 0:
                ax.set_title(f"$|\\hat{{u}}\\cdot\\hat{{{axn}}}|$")
            if k == 0:
                ax.set_ylabel(f"{ETIQUETA[tipo]}\ndensidad")
            if i == 1:
                ax.set_xlabel("componente absoluta")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.text(0.99, 0.985, "- - -  densidad esperada para direcciones uniformes (= 1)",
             color=REF, fontsize=7.5, ha="right", va="top")
    guardar(fig, out, "fig_orientacion_componentes")


def fig_eje_cercano(el, ang_unif, out):
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    tmax = math.degrees(math.acos(1 / math.sqrt(3)))  # 54.74: maximo posible
    xs = np.linspace(0, tmax, 400)
    ref = np.searchsorted(np.sort(ang_unif), xs, side="right") / len(ang_unif)
    ax.plot(xs, ref, color=REF, lw=1.4, ls="--", zorder=2,
            label="Direcciones uniformes (referencia)")
    for tipo in TIPOS:
        u = el.loc[el.tipo_simetria == tipo, ["ux", "uy", "uz"]].to_numpy()
        a = np.sort(angulo_eje_cercano(u))
        y = np.arange(1, len(a) + 1) / len(a)
        ax.step(np.r_[0, a], np.r_[0, y], where="post", color=COLOR[tipo], lw=2,
                label=ETIQUETA[tipo], zorder=3)
    ax.axvline(5, color=GRID, lw=0.8, zorder=1)
    ax.axvline(15, color=GRID, lw=0.8, zorder=1)
    ax.set_xlim(0, tmax)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("ángulo al eje del mundo más cercano (°)")
    ax.set_ylabel("fracción acumulada")
    ax.legend(loc="upper left", bbox_to_anchor=(0.06, 0.98), fontsize=8)
    guardar(fig, out, "fig_orientacion_eje_cercano")


# ----------------------------------------------------------------- visibilidad
def claves_nv(obj, n_total):
    """Claves de columna por n_v, en orden. 'all' se omite si --nv ya incluia
    n_total (serian columnas identicas)."""
    ks = [c[len("delta_min_"):] for c in obj.columns if c.startswith("delta_min_")]
    num = sorted(int(k) for k in ks if k != "all")
    claves = [str(n) for n in num]
    if "all" in ks and n_total not in num:
        claves.append("all")
    return claves


def resumen_visibilidad(obj, n_total):
    claves = claves_nv(obj, n_total)
    filas = []
    for tipo in TIPOS:
        d = obj[obj.tipo_simetria == tipo]
        for k in claves:
            dm = d[f"delta_min_{k}"].to_numpy()
            nb = d[f"n_buenas_{k}"].to_numpy()
            filas.append({
                "tipo_simetria": tipo, "n_v": n_total if k == "all" else int(k),
                "n_objetos": len(d),
                "delta_min_mediana": round(float(np.median(dm)), 1),
                "delta_min_p90": round(float(np.percentile(dm, 90)), 1),
                "pct_obj_con_1_buena": round(100 * np.mean(nb >= 1), 1),
                "pct_obj_con_2_buenas": round(100 * np.mean(nb >= 2), 1),
                "frac_buenas_media": round(float(d[f"frac_buenas_{k}"].mean()), 3),
            })
    return pd.DataFrame(filas)


def tex_visibilidad(res, umbral, path):
    lineas = [
        "\\begin{table}[htbp]", "\\centering", "\\small",
        f"\\caption{{Visibilidad de los elementos de simetría desde las vistas "
        f"seleccionadas. $\\delta_{{\\min}}$: mejor desviación entre las $n_v$ vistas "
        f"($0^\\circ$ = el eje o plano se ve de canto). Vista buena: "
        f"$\\delta \\leq {umbral:g}^\\circ$.}}",
        "\\label{tab:visibilidad}",
        "\\begin{tabular}{llrrrr}", "\\toprule",
        "Tipo & $n_v$ & $\\delta_{\\min}$ mediana & $\\delta_{\\min}$ p90 & "
        "\\% obj.\\ $\\geq 1$ buena & \\% vistas buenas \\\\", "\\midrule",
    ]
    for i, tipo in enumerate(TIPOS):
        d = res[res.tipo_simetria == tipo]
        for j, r in enumerate(d.itertuples()):
            lineas.append(
                f"{tipo if j == 0 else ''} & {r.n_v} & {r.delta_min_mediana}° & "
                f"{r.delta_min_p90}° & {r.pct_obj_con_1_buena} & "
                f"{round(100 * r.frac_buenas_media, 1)} \\\\")
        if i == 0:
            lineas.append("\\midrule")
    lineas += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    Path(path).write_text("\n".join(lineas), encoding="utf-8")


def fig_visibilidad_delta(obj, n_total, umbral, out):
    claves = claves_nv(obj, n_total)
    nvs = [n_total if k == "all" else int(k) for k in claves]
    xs = list(range(len(nvs)))  # niveles del diseno: equiespaciados
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for tipo in TIPOS:
        d = obj[obj.tipo_simetria == tipo]
        q = np.array([np.percentile(d[f"delta_min_{k}"], [10, 50, 90]) for k in claves])
        ax.fill_between(xs, q[:, 0], q[:, 2], color=COLOR[tipo], alpha=0.15,
                        linewidth=0, zorder=2)
        ax.plot(xs, q[:, 1], color=COLOR[tipo], lw=2, marker="o", ms=4.5,
                markeredgecolor="white", markeredgewidth=1, label=ETIQUETA[tipo], zorder=3)
    ax.axhline(umbral, color=REF, lw=1.2, ls="--", zorder=1)
    ax.text(xs[-1], umbral, f"umbral vista buena ({umbral:g}°)", color=REF,
            fontsize=7.5, ha="right", va="bottom")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(n) for n in nvs])
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, None)
    ax.set_xlabel("número de vistas $n_v$")
    ax.set_ylabel("$\\delta_{\\min}$ (°): mediana y p10–p90")
    ax.legend(loc="upper right")
    guardar(fig, out, "fig_visibilidad_delta")


def fig_visibilidad_buenas(obj, n_total, umbral, out):
    claves = claves_nv(obj, n_total)
    etiquetas_nv = [str(n_total) if k == "all" else k for k in claves]
    grupos = ["0", "1", "2", "≥3"]
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.8), sharey=True)
    for ax, tipo in zip(axs, TIPOS):
        d = obj[obj.tipo_simetria == tipo]
        izq = np.zeros(len(claves))
        for g, col in zip(range(4), RAMPA):
            vals = np.array([
                100 * np.mean((d[f"n_buenas_{k}"] >= 3) if g == 3 else (d[f"n_buenas_{k}"] == g))
                for k in claves])
            ax.barh(etiquetas_nv, vals, left=izq, color=col, edgecolor="white",
                    linewidth=1, height=0.65, label=grupos[g], zorder=2)
            izq += vals
        ax.set_xlim(0, 100)
        ax.grid(axis="y", visible=False)
        ax.set_title(ETIQUETA[tipo])
        ax.set_xlabel("% de objetos")
    axs[0].set_ylabel("número de vistas $n_v$")
    axs[1].legend(title=f"vistas buenas ($\\delta \\leq {umbral:g}°$)", ncol=4,
                  loc="upper center", bbox_to_anchor=(-0.05, -0.22), fontsize=7.5,
                  title_fontsize=7.5)
    guardar(fig, out, "fig_visibilidad_buenas")


# ----------------------------------------------------------------- util / main
def guardar(fig, out, nombre):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{nombre}.{ext}")
    plt.close(fig)
    print(f"[fig] {out / nombre}.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="visibilidad",
                    help="mismo valor que --out-prefix de visibilidad_simetria.py")
    ap.add_argument("--out-dir", default="figuras/orientacion_visibilidad")
    ap.add_argument("--umbral", type=float, default=15.0,
                    help="umbral (grados) usado al correr visibilidad_simetria.py")
    ap.add_argument("--n-total", type=int, default=114,
                    help="numero total de vistas renderizadas (fila 'all')")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    el = pd.read_csv(f"{args.prefix}_elementos.csv", dtype={"object_id": str})
    obj = pd.read_csv(f"{args.prefix}_objetos.csv", dtype={"object_id": str})
    el["nombre_categoria"] = el["nombre_categoria"].fillna("(sin categoria)")
    u = el[["ux", "uy", "uz"]].to_numpy()
    el[["ux", "uy", "uz"]] = u / np.linalg.norm(u, axis=1, keepdims=True)  # deshacer redondeo

    ang_unif = referencia_uniforme()

    res_o = resumen_orientacion(el, ang_unif)
    res_o.to_csv(out / "orientacion_resumen.csv", index=False)
    tex_orientacion(res_o, out / "orientacion_resumen.tex")
    orientacion_por_categoria(el).to_csv(out / "orientacion_por_categoria.csv", index=False)

    res_v = resumen_visibilidad(obj, args.n_total)
    res_v.to_csv(out / "visibilidad_resumen.csv", index=False)
    tex_visibilidad(res_v, args.umbral, out / "visibilidad_resumen.tex")

    fig_esfera(el, out)
    fig_componentes(el, out)
    fig_eje_cercano(el, ang_unif, out)
    fig_visibilidad_delta(obj, args.n_total, args.umbral, out)
    fig_visibilidad_buenas(obj, args.n_total, args.umbral, out)

    pd.set_option("display.width", 200)
    print("\n=== ORIENTACION ===")
    print(res_o.T.to_string(header=False))
    print("\n=== VISIBILIDAD ===")
    print(res_v.to_string(index=False))
    print(f"\nSalidas en {out}/")


if __name__ == "__main__":
    main()
