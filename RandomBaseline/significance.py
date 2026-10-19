# Decides whether a method beats the random baseline (docs/literature/
# baseline_aleatorio.md section 5, option A): the method's summary value vs.
# the simulated distribution of the random predictor, with its 95% interval, a
# one-sided empirical p-value and a z-score, for every summary metric.
from __future__ import annotations

import math

import numpy as np

from simulation import ANGULAR_THRESHOLDS

# metric -> True when higher is better
COMPARED_METRICS: dict[str, bool] = {
    "angular_error_mean": False,
    "angular_error_median": False,
    "auc_angular": True,
    **{f"precision_{t}deg": True for t in ANGULAR_THRESHOLDS},
    "f1_ref": True,
    "f1_ref_hungarian": True,
}
CI_PERCENTILES = (2.5, 97.5)
ALPHA = 0.025


def compare_metric(method_value: float, random_values: np.ndarray,
                   higher_is_better: bool, alpha: float = ALPHA) -> dict:
    """ Compare one method metric against its simulated random-baseline distribution.

    Args:
        * method_value: the method's summary value (e.g. angular_error_mean)
        * random_values: (R,) simulated values of the same metric for the random predictor
        * higher_is_better: orientation of the metric
        * alpha: one-sided significance level

    Returns:
        * dict: random mean/std/95% interval, z = (method - random_mean) / random_std,
          one-sided p-values (empirical with add-one correction, and normal
          approximation) and whether the method is better than random

    """
    r = np.asarray(random_values, dtype=float)
    mean = float(r.mean())
    std = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    if higher_is_better:
        n_as_good = int((r >= method_value).sum())
    else:
        n_as_good = int((r <= method_value).sum())
    p_empirical = (1 + n_as_good) / (len(r) + 1)

    if std > 0:
        z = (method_value - mean) / std
        z_better = z if higher_is_better else -z
        p_normal = 0.5 * math.erfc(z_better / math.sqrt(2.0))
    else:
        z, p_normal = float("nan"), float("nan")

    return {
        "random_mean": mean,
        "random_std": std,
        "random_ci_low": float(np.percentile(r, CI_PERCENTILES[0])),
        "random_ci_high": float(np.percentile(r, CI_PERCENTILES[1])),
        "z": z,
        "p_empirical": p_empirical,
        "p_normal": p_normal,
        "better_than_random": p_empirical < alpha,
    }
