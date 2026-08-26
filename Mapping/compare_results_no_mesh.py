"""
compare_results_no_mesh.py
---------------------------
Version de compare_results.py restringida a los metodos del pipeline SIN
malla (`triangulation`, `triangulation_multiplane` -- ver
`Mapping/estimate_symmetry_no_mesh.py` / `docs/pipeline_sin_malla.md`).

compare_results.py no sirve tal cual para estas corridas: tiene METHODS
hardcodeado a los 4 metodos con malla (svd/ransac_svd/svd_sde/ransac_svd_sde)
y el parseo del experiment_id desde el nombre de archivo asume esos sufijos.
Ademas, `triangulation_multiplane` tiene una forma de metricas totalmente
distinta (recall/precision sobre el conjunto de planos GT, sin
angular_error/AUC/precision@theta -- ver `evaluate.py::write_csv`), por lo
que no puede compartir los mismos plots que `triangulation`.

Solo lee corridas cuyo experiment_id termina en "_nomesh" (la convencion
usada en los comandos de estimate_symmetry_no_mesh.py/evaluate.py), y solo
los metodos "triangulation"/"triangulation_multiplane".

Usage
-----
    python Mapping/compare_results_no_mesh.py --renders-root ../data/renders --symmetry-type axis_sym
    python Mapping/compare_results_no_mesh.py --renders-root ../data/renders --symmetry-type plane_sym --save-dir ../results/plots_nomesh --csv-dir ../results

    # Un solo experimento (exact match):
    python Mapping/compare_results_no_mesh.py \
        --renders-root ../data/renders --symmetry-type axis_sym \
        --experiment-id axis_v00_nomesh \
        --save-dir ../results/axis_sym/per_experiment/axis_v00_nomesh/plots \
        --csv-dir ../results/axis_sym/per_experiment/axis_v00_nomesh
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = ["triangulation_multiplane", "triangulation"]  # orden importa: ver load_csvs

METHOD_COLORS = {
    "triangulation":            "#4C72B0",
    "triangulation_multiplane": "#C44E52",
}

METHOD_LABELS = {
    "triangulation":            "Triangulacion (sin malla)",
    "triangulation_multiplane": "Triangulacion multi-plano (sin malla)",
}

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
              lightings: list[str],
              experiment_ids: list[str] | None = None) -> pd.DataFrame:
    """Solo experiment_id que terminan en '_nomesh' y metodo in METHODS.
    Igual que compare_results.py, el match de --experiment-id es exacto
    (nunca por prefijo de archivo)."""
    size_tag   = "s" + "_".join(str(s) for s in sizes)
    light_tag  = "_".join(lightings)
    prefix     = f"eval_{size_tag}_{light_tag}_"
    sym_prefix = "axis_v" if symmetry_type == "axis_sym" else "plane_v"
    root       = renders_root / symmetry_type
    wanted     = set(experiment_ids) if experiment_ids else None

    frames     = []
    found_exps = set()
    for csv in sorted(root.glob(f"{prefix}{sym_prefix}*_summary.csv")):
        stem = csv.stem.replace(prefix, "").removesuffix("_summary")
        # METHODS ordenado con "triangulation_multiplane" primero para que
        # el sufijo mas largo/especifico se pruebe antes que "triangulation".
        method = next((m for m in METHODS if stem.endswith("_" + m)), None)
        if method is None:
            continue
        exp = stem[: -(len(method) + 1)]
        if not exp.endswith("_nomesh"):
            continue
        found_exps.add(exp)
        if wanted is not None and exp not in wanted:
            continue
        df = pd.read_csv(csv)
        df.insert(0, "method", method)
        df.insert(0, "experiment", exp)
        frames.append(df)

    if not frames:
        if wanted is not None:
            raise SystemExit(
                f"[error] None of --experiment-id {sorted(wanted)} found in {root}. "
                f"Available (_nomesh only): {sorted(found_exps)}"
            )
        raise SystemExit(
            f"[error] No CSVs '_nomesh' found in {root} matching '{prefix}{sym_prefix}*' "
            f"for methods {METHODS}."
        )

    if wanted is not None:
        missing = wanted - found_exps
        if missing:
            print(f"[warn] --experiment-id not found, skipped: {sorted(missing)}")

    combined = pd.concat(frames, ignore_index=True)
    combined["n_views"] = combined["n_views"].astype(int)
    return combined


# ── Console table ─────────────────────────────────────────────────────────────

def print_table(df: pd.DataFrame) -> None:
    single = df[df["method"] == "triangulation"]
    multi  = df[df["method"] == "triangulation_multiplane"]

    if not single.empty:
        cols = ["experiment", "method", "n_views", "n_objects",
                "angular_error_mean", "angular_error_median",
                "auc_angular", "precision_5deg", "precision_10deg"]
        available = [c for c in cols if c in single.columns]
        print("\n--- triangulation (eje/plano unico) ---")
        print(single[available].sort_values(["experiment", "n_views"]).to_string(index=False))

    if not multi.empty:
        cols = ["experiment", "method", "n_views", "n_objects",
                "n_planes_predicted_mean", "n_true_planes_mean",
                "n_planes_matched_mean", "recall_planes_mean", "precision_planes_mean"]
        available = [c for c in cols if c in multi.columns]
        print("\n--- triangulation_multiplane (recall/precision sobre planos GT) ---")
        print(multi[available].sort_values(["experiment", "n_views"]).to_string(index=False))


# ── Plot 1: triangulation -- metricas vs n_views ──────────────────────────────

def plot_metrics_by_nviews(df: pd.DataFrame, symmetry_type: str,
                           save_dir: Path | None,
                           out_prefix: str | None = None) -> None:
    sub = df[df["method"] == "triangulation"]
    if sub.empty:
        return
    prefix = out_prefix or symmetry_type
    exps   = sorted(sub["experiment"].unique())

    metrics = [
        ("angular_error_mean",     "Angular error mean (°)", False),
        ("translation_error_mean", "Translation error mean", False),
        ("auc_angular",            "AUC angular",            True),
        ("precision_5deg",         "Precision @ 5°",         True),
        ("precision_10deg",        "Precision @ 10°",        True),
    ]

    cmap        = plt.get_cmap("tab10")
    colors      = {e: cmap(i) for i, e in enumerate(exps)}
    n_views_all = sorted(sub["n_views"].unique())

    for metric, ylabel, higher_better in metrics:
        if metric not in sub.columns:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for exp in exps:
            edf = sub[sub["experiment"] == exp].sort_values("n_views")
            if edf.empty or edf[metric].isna().all():
                continue
            ax.plot(edf["n_views"], edf[metric], marker="o", label=exp, color=colors[exp])
        ax.set_xlabel("n_views")
        ax.set_xticks(n_views_all)
        ax.set_ylabel(ylabel)
        ax.grid()
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
        direction = "↑ mejor" if higher_better else "↓ mejor"
        ax.set_title(f"{symmetry_type} (sin malla) — {ylabel}  ({direction})")
        fig.tight_layout()
        _save_or_show(fig, save_dir, f"{prefix}_triangulation_{metric}.png")


# ── Plot 2: triangulation_multiplane -- recall/precision vs n_views ──────────

def plot_multiplane_metrics(df: pd.DataFrame, symmetry_type: str,
                            save_dir: Path | None,
                            out_prefix: str | None = None) -> None:
    sub = df[df["method"] == "triangulation_multiplane"]
    if sub.empty:
        return
    prefix = out_prefix or symmetry_type
    exps   = sorted(sub["experiment"].unique())

    metrics = [
        ("recall_planes_mean",     "Recall (planos GT)",      True),
        ("precision_planes_mean",  "Precision (planos GT)",   True),
        ("n_planes_matched_mean",  "N planos matcheados (prom.)", True),
        ("n_planes_predicted_mean", "N planos predichos (prom.)", None),
    ]

    cmap        = plt.get_cmap("tab10")
    colors      = {e: cmap(i) for i, e in enumerate(exps)}
    n_views_all = sorted(sub["n_views"].unique())

    for metric, ylabel, higher_better in metrics:
        if metric not in sub.columns:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for exp in exps:
            edf = sub[sub["experiment"] == exp].sort_values("n_views")
            if edf.empty or edf[metric].isna().all():
                continue
            ax.plot(edf["n_views"], edf[metric], marker="o", label=exp, color=colors[exp])
        ax.set_xlabel("n_views")
        ax.set_xticks(n_views_all)
        ax.set_ylabel(ylabel)
        ax.grid()
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
        direction = "" if higher_better is None else ("  (↑ mejor)" if higher_better else "  (↓ mejor)")
        ax.set_title(f"{symmetry_type} (sin malla, multi-plano) — {ylabel}{direction}")
        fig.tight_layout()
        _save_or_show(fig, save_dir, f"{prefix}_multiplane_{metric}.png")


# ── Plot 3: curva de precision continua (solo triangulation) ─────────────────

def _results_json_path(renders_root: Path, symmetry_type: str,
                       sizes: list[int], lightings: list[str],
                       exp: str, method: str) -> Path:
    size_tag  = "s" + "_".join(str(s) for s in sizes)
    light_tag = "_".join(lightings)
    return renders_root / symmetry_type / f"eval_{size_tag}_{light_tag}_{exp}_{method}_results.json"


def plot_precision_curve(df: pd.DataFrame, renders_root: Path, symmetry_type: str,
                         sizes: list[int], lightings: list[str],
                         save_dir: Path | None,
                         out_prefix: str | None = None) -> None:
    """Curva de precision angular continua (precision@theta, theta in [0,90]),
    solo para 'triangulation' -- 'triangulation_multiplane' no tiene un unico
    angular_error por objeto (ver plot_multiplane_metrics en su lugar)."""
    sub = df[df["method"] == "triangulation"]
    if sub.empty:
        return
    prefix = out_prefix or symmetry_type
    exps   = sorted(sub["experiment"].unique())
    nv_max = int(sub["n_views"].max())
    nv_key = str(nv_max)

    cmap       = plt.get_cmap("tab10")
    colors     = {e: cmap(i) for i, e in enumerate(exps)}
    thresholds = np.linspace(0, 90, 181)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for exp in exps:
        path = _results_json_path(renders_root, symmetry_type, sizes, lightings, exp, "triangulation")
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

    ax.set_xlabel("Umbral angular (°)")
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Precision")
    ax.grid(alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    ax.set_title(f"{symmetry_type} (sin malla) — Curva de Precision Angular  (n_views={nv_max})  ↑ mejor")
    fig.tight_layout()
    _save_or_show(fig, save_dir, f"{prefix}_triangulation_precision_curve_nv{nv_max}.png")


# ── Plot 4: % objetos validos por prompt ──────────────────────────────────────

def plot_valid_by_prompt(df: pd.DataFrame, symmetry_type: str,
                         save_dir: Path | None,
                         total_objects: int | None = None,
                         out_prefix: str | None = None) -> None:
    prefix    = out_prefix or symmetry_type
    valid_dir = (save_dir / "valid") if save_dir else None

    for method in ("triangulation", "triangulation_multiplane"):
        sub = df[df["method"] == method]
        if sub.empty:
            continue

        exps        = sorted(sub["experiment"].unique())
        n_views_all = sorted(sub["n_views"].unique())
        cmap        = plt.get_cmap("tab10")
        colors      = {e: cmap(i) for i, e in enumerate(exps)}

        agg   = (sub.groupby(["experiment", "n_views"])["n_objects"].first().reset_index())
        denom = total_objects or int(agg["n_objects"].max())

        fig, ax = plt.subplots(figsize=(7, 4))
        for exp in exps:
            edf = agg[agg["experiment"] == exp].sort_values("n_views")
            if edf.empty:
                continue
            pcts = edf["n_objects"] / denom * 100
            ax.plot(edf["n_views"].tolist(), pcts.tolist(), marker="o", label=exp, color=colors[exp])

        ax.set_xticks(n_views_all)
        ax.set_xlabel("n_views")
        ax.set_ylabel("Objetos con predicción válida (%)")
        ax.set_ylim(0, 105)
        title_n = f"  [N={denom}]" if total_objects else ""
        ax.set_title(f"{symmetry_type} (sin malla, {METHOD_LABELS[method]}) — Objetos válidos{title_n}")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
        ax.grid()
        fig.tight_layout()
        _save_or_show(fig, valid_dir, f"{prefix}_{method}_valid.png")


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
        description="Compara resultados de evaluacion del pipeline SIN malla "
                     "(triangulation / triangulation_multiplane, experiment_id '_nomesh').",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--renders-root",  required=True)
    p.add_argument("--symmetry-type", required=True, choices=["axis_sym", "plane_sym"])
    p.add_argument("--experiment-id", nargs="+", default=None, metavar="EXP_ID",
                   help="Filtra a uno o mas experimentos '_nomesh' especificos (exact match). "
                        "Ej.: --experiment-id axis_v00_nomesh axis_v05_nomesh")
    p.add_argument("--sizes",     type=int, nargs="+", default=[224])
    p.add_argument("--lightings", type=str, nargs="+", default=["flat"],
                   choices=["flat", "darker", "brighter"])
    p.add_argument("--save-dir",  default=None,
                   help="Directorio donde guardar los plots. Si se omite, se muestran en pantalla.")
    p.add_argument("--csv-dir",   default=None,
                   help="Directorio base donde guardar el CSV combinado. "
                        "Se crea la subcarpeta experiments_DD_MM_YYYY/ automaticamente.")
    p.add_argument("--total-objects", type=int, default=None,
                   help="Total de objetos evaluados, para mostrar porcentaje en plot_valid_by_prompt.")
    p.add_argument("--no-plots",  action="store_true",
                   help="Solo imprime la tabla, sin generar graficos.")
    return p.parse_args()


def main() -> None:
    args     = parse_args()
    root     = Path(args.renders_root)
    save_dir = Path(args.save_dir) if args.save_dir else None

    plt.rcParams.update(_ACADEMIC_STYLE)

    df = load_csvs(root, args.symmetry_type, args.sizes, args.lightings,
                    experiment_ids=args.experiment_id)
    print_table(df)

    if args.experiment_id:
        ids = sorted(args.experiment_id)
        tag = "_".join(ids) if len(ids) <= 3 else f"{len(ids)}exps"
        out_prefix = f"{args.symmetry_type}_{tag}"
    else:
        out_prefix = f"{args.symmetry_type}_nomesh"

    if args.csv_dir:
        today    = date.today().strftime("%d_%m_%Y")
        out_dir  = Path(args.csv_dir) / f"experiments_{today}"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{out_prefix}_comparison.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

    if args.no_plots:
        return

    plot_metrics_by_nviews(df, args.symmetry_type, save_dir, out_prefix=out_prefix)
    plot_multiplane_metrics(df, args.symmetry_type, save_dir, out_prefix=out_prefix)
    plot_precision_curve(df, root, args.symmetry_type, args.sizes, args.lightings, save_dir,
                          out_prefix=out_prefix)
    plot_valid_by_prompt(df, args.symmetry_type, save_dir, total_objects=args.total_objects,
                          out_prefix=out_prefix)


if __name__ == "__main__":
    main()
