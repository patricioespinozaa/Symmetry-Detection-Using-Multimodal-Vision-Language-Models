# LaTeX tables (booktabs + adjustbox, same style as utils/generar_tablas_latex.py)
# with the random-baseline scores on the full dataset: one table for axial and
# one for planar symmetry. The random predictor answers every object.
from __future__ import annotations

from collections import Counter

import numpy as np

from simulation import ANGULAR_THRESHOLDS, AUC_ANGULAR_MAX, GroundTruth

HEADER = ("% Generado por RandomBaseline/run_baseline.py latex. "
          "Requiere booktabs y adjustbox.\n\n")
# metric -> (row label, number format)
ROWS = {
    "auc_angular": (rf"AUC$_{{{AUC_ANGULAR_MAX:g}^\circ}}$", "{:.3f}"),
    "angular_error_mean": (r"Error angular medio ($^\circ$)", "{:.1f}"),
    "angular_error_median": (r"Error angular mediano ($^\circ$)", "{:.1f}"),
    "angular_error_std": (r"Desv.\ estándar del error ($^\circ$)", "{:.1f}"),
    **{f"precision_{t}deg": (rf"Precisión @{t}$^\circ$", "{:.3f}") for t in ANGULAR_THRESHOLDS},
    "f1_ref": (r"$F1_{\mathrm{ref}}$ voraz", "{:.3f}"),
    "f1_ref_hungarian": (r"$F1_{\mathrm{ref}}$ húngaro", "{:.3f}"),
}
LABEL = {"axis_sym": "tab:axis_azar", "plane_sym": "tab:plane_azar"}
LABEL_MULTIPLANE = "tab:plane_azar_multi"
TRIVIAL_ANGULAR_METRICS = ("auc_angular", "angular_error_mean", "angular_error_median",
                           *[f"precision_{t}deg" for t in ANGULAR_THRESHOLDS])
# metric -> row label; the adjusted metrics equal the raw ones at full coverage
MULTIPLANE_ROWS = {
    "recall_planes_mean": r"Recall a $15^\circ$",
    "precision_planes_mean": r"Precisión a $15^\circ$",
    "f1_ref": ROWS["f1_ref"][0],
    "f1_ref_hungarian": ROWS["f1_ref_hungarian"][0],
}


def _wrap(caption: str, label: str, alignment: str, header: str, body: list[str]) -> str:
    """ Wrap header and rows in table + adjustbox + tabular with booktabs rules.

    Args:
        * caption: text of \\caption
        * label: text of \\label
        * alignment: tabular column spec, e.g. 'lrrr'
        * header: header row, already formatted
        * body: data rows, already formatted

    Returns:
        * str: complete LaTeX table

    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{adjustbox}{max width=\textwidth}",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        header,
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def _interval(values: np.ndarray, fmt: str) -> str:
    """ 95% interval (percentiles 2.5 and 97.5) formatted as [low, high]. """
    low, high = np.percentile(values, [2.5, 97.5])
    return f"[{fmt.format(low)}, {fmt.format(high)}]"


def _profile_text(gt: GroundTruth) -> str:
    """ Spanish description of how many objects have 1, 2 or 3 GT planes. """
    counts = Counter(int(k) for k in gt.n_elements)
    parts = [f"{counts[k]} con {k} {'plano' if k == 1 else 'planos'}" for k in sorted(counts)]
    return ", ".join(parts[:-1]) + " y " + parts[-1] if len(parts) > 1 else parts[0]


def axis_table(gt: GroundTruth, simulated: dict[str, np.ndarray],
               closed_form: dict[str, float], repetitions: int) -> str:
    """ Random-baseline table for axial symmetry: closed form, simulation and 95% interval.

    Args:
        * gt: axis_sym ground truth (for the object count)
        * simulated: simulate_random_baseline output at full coverage
        * closed_form: chance.analytic_summary output at coverage 1
        * repetitions: number of simulated datasets

    Returns:
        * str: LaTeX table

    """
    body = []
    for metric, (label, fmt) in ROWS.items():
        if metric not in simulated:
            continue
        values = simulated[metric]
        body.append(f"{label} & {fmt.format(closed_form[metric])} & "
                    f"{fmt.format(values.mean())} & {_interval(values, fmt)} " + r"\\")
    caption = (
        "Línea base aleatoria para simetría axial: la dirección del eje se elige uniforme "
        f"en la esfera, sin mirar las imágenes, para los {gt.n_objects} objetos (todos "
        "reciben predicción). Fórmula cerrada: valor teórico con un eje verdadero por "
        f"objeto. Simulación: media de {repetitions} conjuntos de datos simulados, evaluados "
        r"con las mismas métricas que el método; IC 95\,\%: percentiles 2.5 y 97.5."
    )
    header = r"Métrica & Fórmula cerrada & Simulación & IC 95\,\% \\"
    return _wrap(caption, LABEL["axis_sym"], "lrrr", header, body)


def plane_table(gt: GroundTruth, simulated: dict[str, np.ndarray], repetitions: int) -> str:
    """ Random-baseline table for planar symmetry: simulation and 95% interval.

    The single-plane closed form is left out on purpose: the error is measured
    against the closest GT plane, so objects with several planes are easier to
    hit by chance and the closed form overstates the chance error.

    Args:
        * gt: plane_sym ground truth (object count and planes per object)
        * simulated: simulate_random_baseline output at full coverage
        * repetitions: number of simulated datasets

    Returns:
        * str: LaTeX table

    """
    body = []
    for metric, (label, fmt) in ROWS.items():
        if metric not in simulated:
            continue
        values = simulated[metric]
        body.append(f"{label} & {fmt.format(values.mean())} & {_interval(values, fmt)} " + r"\\")
    caption = (
        "Línea base aleatoria para simetría planar: la normal del plano se elige uniforme "
        "en la esfera y el plano pasa por el centro del objeto ($d=0$), sin mirar las "
        f"imágenes, para los {gt.n_objects} objetos (todos reciben predicción; "
        f"{_profile_text(gt)}). El error se mide contra el plano verdadero más cercano, "
        "igual que para el método, por lo que el azar es más favorable que el valor "
        r"teórico de un solo plano ($57.3^\circ$). Simulación: media de "
        f"{repetitions} conjuntos de datos simulados; " + r"IC 95\,\%: percentiles 2.5 y 97.5."
    )
    header = r"Métrica & Simulación & IC 95\,\% \\"
    return _wrap(caption, LABEL["plane_sym"], "lrr", header, body)


def trivial_angular_table(scores: dict[str, dict[str, float]], caption: str, label: str) -> str:
    """ Angular metrics of fixed world-axis predictors, one row per predictor.

    Args:
        * scores: predictor label -> simulation.summarize_errors output
        * caption: text of \\caption
        * label: text of \\label

    Returns:
        * str: LaTeX table

    """
    metrics = [m for m in ROWS if m in TRIVIAL_ANGULAR_METRICS]
    body = []
    for name, values in scores.items():
        cells = [ROWS[m][1].format(values[m]) for m in metrics]
        body.append(f"{name} & " + " & ".join(cells) + r" \\")
    header = "Predictor & " + " & ".join(ROWS[m][0] for m in metrics) + r" \\"
    return _wrap(caption, label, "l" + "r" * len(metrics), header, body)


def trivial_plane_set_table(rows: list[dict], caption: str, label: str) -> str:
    """ Plane-set metrics of fixed world-plane predictors, one row per predictor.

    Args:
        * rows: trivial_baselines.score_plane_predictor outputs, 'predictor' already in LaTeX
        * caption: text of \\caption
        * label: text of \\label

    Returns:
        * str: LaTeX table

    """
    metrics = list(MULTIPLANE_ROWS)
    body = []
    for row in rows:
        cells = [f"{row[m]:.3f}" for m in metrics]
        body.append(f"{row['predictor']} & {row['n_planes_predicted']} & "
                    + " & ".join(cells) + r" \\")
    columns = ["Predictor", "Planos", *MULTIPLANE_ROWS.values()]
    return _wrap(caption, label, "lr" + "r" * len(metrics), " & ".join(columns) + r" \\", body)


def plane_multi_table(gt: GroundTruth, simulated: dict[int, dict[str, np.ndarray]],
                      repetitions: int, min_separation_deg: float) -> str:
    """ Random-baseline table for multi-plane predictions, one column per planes per object.

    Args:
        * gt: plane_sym ground truth (object count and planes per object)
        * simulated: planes per object -> simulate_random_multiplane output at full coverage
        * repetitions: number of simulated datasets
        * min_separation_deg: smallest angle allowed between planes of one object

    Returns:
        * str: LaTeX table

    """
    fmt = "{:.3f}"
    body = []
    for metric, label in MULTIPLANE_ROWS.items():
        cells = [f"{fmt.format(s[metric].mean())} {_interval(s[metric], fmt)}"
                 for s in simulated.values()]
        body.append(f"{label} & " + " & ".join(cells) + r" \\")
    caption = (
        "Línea base aleatoria para simetría planar en modo multiplano: cada objeto recibe "
        "un número fijo de planos con normales uniformes en la esfera, que pasan por el "
        rf"centro del objeto ($d=0$) y están separados al menos ${min_separation_deg:g}^\circ$ "
        f"entre sí, sin mirar las imágenes, para los {gt.n_objects} objetos "
        f"({_profile_text(gt)}). Recall y precisión usan el emparejamiento voraz a "
        r"$15^\circ$ de la evaluación multiplano. Cada celda: media de "
        f"{repetitions} conjuntos de datos simulados " + r"e IC 95\,\% (percentiles 2.5 y 97.5)."
    )
    columns = [f"{k} {'plano' if k == 1 else 'planos'} por objeto" for k in simulated]
    header = "Métrica & " + " & ".join(columns) + r" \\"
    return _wrap(caption, LABEL_MULTIPLANE, "l" + "r" * len(columns), header, body)
