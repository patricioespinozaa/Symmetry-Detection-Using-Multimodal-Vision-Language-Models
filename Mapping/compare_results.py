"""
compare_results.py
------------------
Lee todos los CSVs de evaluación para un symmetry_type y genera:
  1. Tabla de consola: experiment × method × n_views
  2. Líneas: angular error mean, translation error mean, AUC angular,
     precision@5°, precision@10° vs n_views (eje X discreto), por método
  3. Curva de precisión angular continua (precision@θ, θ ∈ [0°,90°]) para el mayor n_views
  4. Tasa de aceptación SDE (% accepted=True) para métodos svd_sde y ransac_svd_sde
  5. % objetos válidos por prompt vs n_views
  6. Para plane_sym: también métricas SDE

Usage
-----
    python Mapping/compare_results.py --renders-root ../data/renders --symmetry-type axis_sym
    python Mapping/compare_results.py --renders-root ../data/renders --symmetry-type plane_sym --save-dir ../results/plots
    python Mapping/compare_results.py --renders-root ../data/renders --symmetry-type axis_sym --csv-dir ../results

    python Mapping/compare_results.py \
    --renders-root ../data/renders \
    --symmetry-type plane_sym \
    --save-dir ../results/plots \
    --csv-dir ../results

    python Mapping/compare_results.py \
    --renders-root ../data/renders \
    --symmetry-type axis_sym \
    --save-dir ../results/plots \
    --csv-dir ../results
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = ["ransac_svd_sde", "svd_sde", "ransac_svd", "svd"]

METHOD_COLORS = {
    "svd":            "#4C72B0",
    "ransac_svd":     "#DD8452",
    "svd_sde":        "#55A868",
    "ransac_svd_sde": "#C44E52",
}

METHOD_LABELS = {
    "svd":            "SVD",
    "ransac_svd":     "RANSAC+SVD",
    "svd_sde":        "SVD+SDE",
    "ransac_svd_sde": "RANSAC+SVD+SDE",
}

# Academic style suitable for thesis / conference papers
_ACADEMIC_STYLE: dict = {
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   9,
    "legend.framealpha": 0.92,
    "legend.edgecolor":  "0.75",
    "lines.linewidth":   1.8,
    "lines.markersize":  6,
    "patch.linewidth":   0.6,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linewidth":    0.6,
    "figure.dpi":        150,
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_csvs(renders_root: Path, symmetry_type: str, sizes: list[int],
              lightings: list[str]) -> pd.DataFrame:
    size_tag    = "s" + "_".join(str(s) for s in sizes)
    light_tag   = "_".join(lightings)
    prefix      = f"eval_{size_tag}_{light_tag}_"
    sym_prefix  = "axis_v" if symmetry_type == "axis_sym" else "plane_v"
    root        = renders_root / symmetry_type

    frames = []
    for csv in sorted(root.glob(f"{prefix}{sym_prefix}*_summary.csv")):
        stem   = csv.stem.replace(prefix, "").removesuffix("_summary")
        method = next((m for m in METHODS if stem.endswith("_" + m)), "unknown")
        exp    = stem[: -(len(method) + 1)]
        df     = pd.read_csv(csv)
        df.insert(0, "method", method)
        df.insert(0, "experiment", exp)
        frames.append(df)

    if not frames:
        raise SystemExit(f"[error] No CSVs found in {root} matching '{prefix}{sym_prefix}*'")

    combined = pd.concat(frames, ignore_index=True)
    combined["n_views"] = combined["n_views"].astype(int)
    return combined


# ── Console table ─────────────────────────────────────────────────────────────

def print_table(df: pd.DataFrame, symmetry_type: str) -> None:
    is_plane = symmetry_type == "plane_sym"
    cols = ["experiment", "method", "n_views", "n_objects",
            "angular_error_mean", "angular_error_median",
            "auc_angular", "precision_5deg", "precision_10deg"]
    if is_plane:
        sde_cols = [c for c in ["sde_mean", "auc_sde", "precision_sde_010"] if c in df.columns]
        cols += sde_cols

    available = [c for c in cols if c in df.columns]
    print("\n" + df[available].sort_values(["experiment", "method", "n_views"])
                               .to_string(index=False))


# ── Plot 1: métricas vs n_views por método ────────────────────────────────────

def plot_metrics_by_nviews(df: pd.DataFrame, symmetry_type: str,
                           save_dir: Path | None) -> None:
    methods = [m for m in METHODS if m in df["method"].unique()]
    exps    = sorted(df["experiment"].unique())
    n_cols  = len(methods)

    metrics = [
        ("angular_error_mean",      "Angular error mean (°)",       False),
        ("translation_error_mean",  "Translation error mean",       False),
        ("auc_angular",             "AUC angular",                  True),
        ("precision_5deg",          "Precision @ 5°",               True),
        ("precision_10deg",         "Precision @ 10°",              True),
    ]
    if symmetry_type == "plane_sym":
        if "sde_mean" in df.columns:
            metrics.append(("sde_mean", "SDE mean", False))
        if "auc_sde" in df.columns:
            metrics.append(("auc_sde",  "AUC SDE",  True))

    cmap        = plt.get_cmap("tab10")
    colors      = {e: cmap(i) for i, e in enumerate(exps)}
    n_views_all = sorted(df["n_views"].unique())

    for metric, ylabel, higher_better in metrics:
        if metric not in df.columns:
            continue

        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4.5), sharey=True)
        if n_cols == 1:
            axes = [axes]

        for ax, method in zip(axes, methods):
            sub = df[df["method"] == method].sort_values("n_views")
            for exp in exps:
                edf = sub[sub["experiment"] == exp]
                if edf.empty:
                    continue
                ax.plot(edf["n_views"], edf[metric],
                        marker="o", label=exp, color=colors[exp])
            ax.set_title(METHOD_LABELS[method])
            ax.set_xlabel("n_views")
            ax.set_xticks(n_views_all)
            ax.grid()
            if ax is axes[0]:
                ax.set_ylabel(ylabel)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(len(exps), 6),
                   bbox_to_anchor=(0.5, -0.10))
        direction = "↑ mejor" if higher_better else "↓ mejor"
        fig.suptitle(f"{symmetry_type} — {ylabel}  ({direction})", y=1.02)
        fig.tight_layout(rect=[0, 0.08, 1, 1])
        _save_or_show(fig, save_dir, f"{symmetry_type}_{metric}.png")


# ── Plot 3: curva de precisión continua ──────────────────────────────────────

def _results_json_path(renders_root: Path, symmetry_type: str,
                       sizes: list[int], lightings: list[str],
                       exp: str, method: str) -> Path:
    size_tag  = "s" + "_".join(str(s) for s in sizes)
    light_tag = "_".join(lightings)
    return renders_root / symmetry_type / f"eval_{size_tag}_{light_tag}_{exp}_{method}_results.json"


def plot_precision_curve(df: pd.DataFrame, renders_root: Path, symmetry_type: str,
                         sizes: list[int], lightings: list[str],
                         save_dir: Path | None) -> None:
    """Curva de precisión angular continua (precision@θ para θ ∈ [0°, 90°]).
    Se usa el mayor n_views disponible (resultado final del sistema).
    Un subplot por método, curvas coloreadas por experimento."""
    exps    = sorted(df["experiment"].unique())
    methods = [m for m in METHODS if m in df["method"].unique()]
    nv_max  = int(df["n_views"].max())
    nv_key  = str(nv_max)

    cmap        = plt.get_cmap("tab10")
    colors      = {e: cmap(i) for i, e in enumerate(exps)}
    thresholds  = np.linspace(0, 90, 181)   # 0° a 90° en pasos de 0.5°

    n_cols = len(methods)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4.5), sharey=True)
    if n_cols == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        for exp in exps:
            path = _results_json_path(renders_root, symmetry_type, sizes, lightings, exp, method)
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            errors = [
                obj[nv_key]["angular_error_deg"]
                for obj in data.get("objects", {}).values()
                if obj is not None
                and nv_key in obj
                and obj[nv_key].get("status") == "ok"
            ]
            if not errors:
                continue

            arr        = np.array(errors)
            precisions = [(arr < t).mean() for t in thresholds]
            ax.plot(thresholds, precisions, label=exp, color=colors[exp])

        ax.set_title(METHOD_LABELS[method], fontsize=10)
        ax.set_xlabel("Umbral angular (°)")
        ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("Precision")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(exps), 6),
               bbox_to_anchor=(0.5, -0.10))
    fig.suptitle(
        f"{symmetry_type} — Curva de Precisión Angular  (n_views={nv_max})  ↑ mejor",
        y=1.02,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    _save_or_show(fig, save_dir, f"{symmetry_type}_precision_curve_nv{nv_max}.png")


# ── Plot 4: tasa de aceptación SDE ───────────────────────────────────────────

def plot_acceptance_rate(df: pd.DataFrame, renders_root: Path, symmetry_type: str,
                         save_dir: Path | None) -> None:
    """% de predicciones con accepted=True (SDE ≤ umbral) por experimento y n_views.
    Solo se grafican los métodos SDE (svd_sde, ransac_svd_sde).
    Lee predicted_symmetry_<EXP>.json directamente de cada objeto."""
    sde_methods = [m for m in ["svd_sde", "ransac_svd_sde"] if m in df["method"].unique()]
    if not sde_methods:
        return

    exps        = sorted(df["experiment"].unique())
    n_views_all = sorted(df["n_views"].unique())
    sym_dir     = renders_root / symmetry_type
    cmap        = plt.get_cmap("tab10")
    colors      = {e: cmap(i) for i, e in enumerate(exps)}

    n_cols = len(sde_methods)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4.5), sharey=True)
    if n_cols == 1:
        axes = [axes]

    for ax, method in zip(axes, sde_methods):
        for exp in exps:
            json_name      = f"predicted_symmetry_{exp}.json"
            total_by_nv    = {nv: 0 for nv in n_views_all}
            accepted_by_nv = {nv: 0 for nv in n_views_all}

            for obj_dir in sym_dir.iterdir():
                if not obj_dir.is_dir():
                    continue
                pth = obj_dir / json_name
                if not pth.exists():
                    continue
                try:
                    with open(pth, encoding="utf-8") as f:
                        pred = json.load(f)
                except Exception:
                    continue

                preds = pred.get("n_views_predictions", {})
                for nv in n_views_all:
                    entry = preds.get(str(nv), {}).get(method)
                    if entry is None or entry.get("accepted") is None:
                        continue   # SDE no computado para este objeto
                    total_by_nv[nv] += 1
                    if entry["accepted"] is True:
                        accepted_by_nv[nv] += 1

            rates = [
                accepted_by_nv[nv] / total_by_nv[nv] * 100
                if total_by_nv[nv] > 0 else float("nan")
                for nv in n_views_all
            ]
            ax.plot(n_views_all, rates, marker="o", label=exp, color=colors[exp])

        ax.set_title(METHOD_LABELS[method], fontsize=10)
        ax.set_xlabel("n_views")
        ax.set_xticks(n_views_all)
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("% predicciones aceptadas (SDE ≤ umbral)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(exps), 6),
               bbox_to_anchor=(0.5, -0.10))
    fig.suptitle(f"{symmetry_type} — Tasa de aceptación SDE  ↑ mejor", y=1.02)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    _save_or_show(fig, save_dir, f"{symmetry_type}_acceptance_rate.png")


# ── Plot 5: % objetos válidos por prompt (un gráfico por prompt) ──────────────

def plot_valid_by_prompt(df: pd.DataFrame, symmetry_type: str,
                         save_dir: Path | None,
                         total_objects: int | None = None) -> None:
    """Un único gráfico de líneas con una curva por experimento/prompt.
    Eje X: n_views (discreto). Eje Y: % objetos con predicción válida.
    Como todos los métodos dan el mismo n_objects, se agrupan y se toma el primer valor.
    Se guarda en save_dir/valid/."""
    exps        = sorted(df["experiment"].unique())
    n_views_all = sorted(df["n_views"].unique())
    valid_dir   = (save_dir / "valid") if save_dir else None

    cmap   = plt.get_cmap("tab10")
    colors = {e: cmap(i) for i, e in enumerate(exps)}

    agg = (df.groupby(["experiment", "n_views"])["n_objects"]
             .first()
             .reset_index())
    denom = total_objects or int(agg["n_objects"].max())

    fig, ax = plt.subplots(figsize=(7, 4))

    for exp in exps:
        edf  = agg[agg["experiment"] == exp].sort_values("n_views")
        if edf.empty:
            continue
        pcts = edf["n_objects"] / denom * 100
        ax.plot(edf["n_views"].tolist(), pcts.tolist(),
                marker="o", label=exp, color=colors[exp])

    ax.set_xticks(n_views_all)
    ax.set_xlabel("n_views")
    ax.set_ylabel("Objetos con predicción válida (%)")
    ax.set_ylim(0, 105)
    title_n = f"  [N={denom}]" if total_objects else ""
    ax.set_title(f"{symmetry_type}  —  Objetos con predicción válida{title_n}")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    ax.grid()
    fig.tight_layout()
    _save_or_show(fig, valid_dir, f"{symmetry_type}_valid.png")


# ── Helper ────────────────────────────────────────────────────────────────────

def _save_or_show(fig: plt.Figure, save_dir: Path | None, filename: str) -> None:
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / filename
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
        plt.close(fig)
    else:
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compara resultados de evaluación entre experimentos y métodos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--renders-root",  required=True)
    p.add_argument("--symmetry-type", required=True, choices=["axis_sym", "plane_sym"])
    p.add_argument("--sizes",     type=int, nargs="+", default=[224])
    p.add_argument("--lightings", type=str, nargs="+", default=["flat"],
                   choices=["flat", "darker", "brighter"])
    p.add_argument("--save-dir",  default=None,
                   help="Directorio donde guardar los plots. Si se omite, se muestran en pantalla.")
    p.add_argument("--csv-dir",   default=None,
                   help="Directorio base donde guardar el CSV combinado. "
                        "Se crea la subcarpeta experiments_DD_MM_YYYY/ automáticamente. "
                        "Ejemplo: ../results")
    p.add_argument("--total-objects", type=int, default=None,
                   help="Total de objetos evaluados. Si se indica, muestra el porcentaje "
                        "sobre cada barra en el gráfico n_objects (e.g. 100).")
    p.add_argument("--no-plots",  action="store_true",
                   help="Solo imprime la tabla, sin generar gráficos.")
    return p.parse_args()


def main() -> None:
    args     = parse_args()
    root     = Path(args.renders_root)
    save_dir = Path(args.save_dir) if args.save_dir else None

    plt.rcParams.update(_ACADEMIC_STYLE)

    df = load_csvs(root, args.symmetry_type, args.sizes, args.lightings)
    print_table(df, args.symmetry_type)

    if args.csv_dir:
        today   = date.today().strftime("%d_%m_%Y")
        out_dir = Path(args.csv_dir) / f"experiments_{today}"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{args.symmetry_type}_comparison.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

    if args.no_plots:
        return

    plot_metrics_by_nviews(df, args.symmetry_type, save_dir)
    plot_precision_curve(df, root, args.symmetry_type, args.sizes, args.lightings, save_dir)
    plot_acceptance_rate(df, root, args.symmetry_type, save_dir)
    plot_valid_by_prompt(df, args.symmetry_type, save_dir, total_objects=args.total_objects)


if __name__ == "__main__":
    main()
