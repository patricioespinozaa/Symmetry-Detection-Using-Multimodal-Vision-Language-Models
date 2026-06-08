"""
compare_results.py
------------------
Lee todos los CSVs de evaluación para un symmetry_type y genera:
  1. Tabla de consola: experiment × method × n_views
  2. Gráfico de barras: n_objects detectados por experimento/método
  3. Líneas: métricas clave vs n_views, desglosadas por método
  4. Para plane_sym: también métricas SDE

Usage
-----
    python Mapping/compare_results.py --renders-root ../data/renders --symmetry-type axis_sym
    python Mapping/compare_results.py --renders-root ../data/renders --symmetry-type plane_sym --save-dir ../results/plots
    python Mapping/compare_results.py --renders-root ../data/renders --symmetry-type axis_sym --csv-dir ../results
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
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
        stem   = csv.stem.replace(prefix, "").replace("_summary", "")
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


# ── Plot 1: n_objects detectados ──────────────────────────────────────────────

def plot_n_objects(df: pd.DataFrame, symmetry_type: str, save_dir: Path | None) -> None:
    exps    = sorted(df["experiment"].unique())
    methods = [m for m in METHODS if m in df["method"].unique()]
    n_views = sorted(df["n_views"].unique())

    # Use the n_views with max objects (typically the last/largest group)
    nv      = max(n_views)
    sub     = df[df["n_views"] == nv]

    x       = range(len(exps))
    width   = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(max(8, len(exps) * 1.5), 5))
    for i, method in enumerate(methods):
        vals = [sub.loc[sub["experiment"] == e, "n_objects"]
                    .values[0] if e in sub[sub["method"] == method]["experiment"].values else 0
                for e in exps]
        offset = (i - len(methods) / 2 + 0.5) * width
        ax.bar([xi + offset for xi in x], vals, width,
               label=METHOD_LABELS[method], color=METHOD_COLORS[method], alpha=0.85)

    ax.set_xticks(list(x))
    ax.set_xticklabels(exps, rotation=20, ha="right")
    ax.set_ylabel("n_objects detectados")
    ax.set_title(f"{symmetry_type} — Objetos con predicción válida (n_views={nv})")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, save_dir, f"{symmetry_type}_n_objects.png")


# ── Plot 2: métricas vs n_views por método ────────────────────────────────────

def plot_metrics_by_nviews(df: pd.DataFrame, symmetry_type: str,
                           save_dir: Path | None) -> None:
    methods = [m for m in METHODS if m in df["method"].unique()]
    exps    = sorted(df["experiment"].unique())
    n_cols  = len(methods)

    metrics = [
        ("precision_5deg",       "Precision @ 5°",         True),
        ("auc_angular",          "AUC angular",             True),
        ("angular_error_mean",   "Angular error mean (°)",  False),
    ]
    if symmetry_type == "plane_sym":
        if "sde_mean" in df.columns:
            metrics.append(("sde_mean", "SDE mean", False))
        if "auc_sde" in df.columns:
            metrics.append(("auc_sde",  "AUC SDE",  True))

    cmap   = plt.get_cmap("tab10")
    colors = {e: cmap(i) for i, e in enumerate(exps)}

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
            ax.set_title(METHOD_LABELS[method], fontsize=10)
            ax.set_xlabel("n_views")
            ax.grid(alpha=0.3)
            if ax is axes[0]:
                ax.set_ylabel(ylabel)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(len(exps), 4),
                   bbox_to_anchor=(0.5, -0.02), fontsize=8)
        direction = "↑ mejor" if higher_better else "↓ mejor"
        fig.suptitle(f"{symmetry_type} — {ylabel}  ({direction})", y=1.01)
        fig.tight_layout()
        _save_or_show(fig, save_dir, f"{symmetry_type}_{metric}.png")


# ── Plot 3: mejor método por experimento ──────────────────────────────────────

def plot_best_method(df: pd.DataFrame, symmetry_type: str, save_dir: Path | None) -> None:
    """Heatmap: precision@5° por experimento × método, para el mayor n_views."""
    nv   = df["n_views"].max()
    sub  = df[df["n_views"] == nv]
    exps = sorted(sub["experiment"].unique())
    mths = [m for m in METHODS if m in sub["method"].unique()]

    matrix = pd.DataFrame(index=exps, columns=mths, dtype=float)
    for exp in exps:
        for m in mths:
            row = sub[(sub["experiment"] == exp) & (sub["method"] == m)]
            matrix.loc[exp, m] = row["precision_5deg"].values[0] if not row.empty else float("nan")

    fig, ax = plt.subplots(figsize=(len(mths) * 1.8 + 1, len(exps) * 0.7 + 1.5))
    im = ax.imshow(matrix.values.astype(float), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Precision @ 5°")

    ax.set_xticks(range(len(mths)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in mths], rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(exps)))
    ax.set_yticklabels(exps, fontsize=9)

    for i, exp in enumerate(exps):
        for j, m in enumerate(mths):
            val = matrix.loc[exp, m]
            if not pd.isna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                        color="black" if 0.3 < val < 0.8 else "white")

    ax.set_title(f"{symmetry_type} — Precision@5° (n_views={nv})")
    fig.tight_layout()
    _save_or_show(fig, save_dir, f"{symmetry_type}_heatmap_precision5.png")


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
                        "Se crea la subcarpeta experiments_YYYYMMDD/ automáticamente. "
                        "Ejemplo: ../results")
    p.add_argument("--no-plots",  action="store_true",
                   help="Solo imprime la tabla, sin generar gráficos.")
    return p.parse_args()


def main() -> None:
    args     = parse_args()
    root     = Path(args.renders_root)
    save_dir = Path(args.save_dir) if args.save_dir else None

    df = load_csvs(root, args.symmetry_type, args.sizes, args.lightings)
    print_table(df, args.symmetry_type)

    if args.csv_dir:
        today   = date.today().strftime("%Y%m%d")
        out_dir = Path(args.csv_dir) / f"experiments_{today}"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{args.symmetry_type}_comparison.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

    if args.no_plots:
        return

    plot_n_objects(df, args.symmetry_type, save_dir)
    plot_metrics_by_nviews(df, args.symmetry_type, save_dir)
    plot_best_method(df, args.symmetry_type, save_dir)


if __name__ == "__main__":
    main()
