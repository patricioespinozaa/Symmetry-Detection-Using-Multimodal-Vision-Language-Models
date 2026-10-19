# Closed-form scores of a predictor that draws its direction uniformly on the
# sphere, against ONE true direction (axis_sym, or a plane_sym object with a
# single GT plane). By Archimedes' theorem |cos(theta)| is uniform on [0, 1],
# so P(theta <= t) = 1 - cos(t); everything below follows from that CDF.
# Objects without a prediction are imputed with 90 deg (same rule as
# Mapping/evaluate.py), so with coverage c the error is a mixture:
# c * (random direction) + (1 - c) * (point mass at 90 deg).
# See docs/literature/baseline_aleatorio.md sections 2 and 3.1.
from __future__ import annotations

import math


def mean_error_deg(coverage: float = 1.0) -> float:
    """ Expected angular error in degrees (1 rad = 57.30 deg at full coverage). """
    return coverage * math.degrees(1.0) + (1.0 - coverage) * 90.0


def median_error_deg(coverage: float = 1.0) -> float:
    """ Median angular error in degrees (60 deg at full coverage, 90 deg below 50% coverage). """
    if coverage < 0.5:
        return 90.0
    return math.degrees(math.acos(1.0 - 0.5 / coverage))


def std_error_deg(coverage: float = 1.0) -> float:
    """ Standard deviation of the angular error in degrees (sqrt(pi - 3) rad at full coverage). """
    half_pi = math.pi / 2.0
    mean_rad = coverage * 1.0 + (1.0 - coverage) * half_pi
    second_moment = coverage * (math.pi - 2.0) + (1.0 - coverage) * half_pi ** 2
    return math.degrees(math.sqrt(second_moment - mean_rad ** 2))


def precision_at(threshold_deg: float, coverage: float = 1.0) -> float:
    """ Fraction of objects with angular error below threshold_deg: c * (1 - cos t). """
    return coverage * (1.0 - math.cos(math.radians(threshold_deg)))


def auc_angular(max_error_deg: float, coverage: float = 1.0) -> float:
    """ Normalized area under precision@t for t in [0, max_error_deg]: c * (1 - sin(T) / T). """
    t = math.radians(max_error_deg)
    return coverage * (1.0 - math.sin(t) / t)


def analytic_summary(thresholds_deg: list[float], auc_max_deg: float,
                     coverage: float = 1.0) -> dict[str, float]:
    """ All closed-form chance scores, keyed with Mapping/evaluate.py's summary column names.

    Args:
        * thresholds_deg: thresholds for precision@t (evaluate.ANGULAR_THRESHOLDS)
        * auc_max_deg: AUC integration limit (evaluate.AUC_ANGULAR_MAX)
        * coverage: fraction of objects that receive a prediction, in [0, 1]

    Returns:
        * dict[str, float]: metric name -> chance value

    """
    summary = {
        "angular_error_mean": mean_error_deg(coverage),
        "angular_error_median": median_error_deg(coverage),
        "angular_error_std": std_error_deg(coverage),
        "auc_angular": auc_angular(auc_max_deg, coverage),
    }
    for t in thresholds_deg:
        summary[f"precision_{t}deg"] = precision_at(t, coverage)
    return summary
