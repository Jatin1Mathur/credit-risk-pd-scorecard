"""PD calibration and the rating master scale."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .config import MASTER_SCALE


def intercept_shift_for(log_odds: np.ndarray, target_mean_pd: float) -> float:
    """Log-odds shift so the average PD equals the target central tendency.

    Keeps the rank order (and therefore Gini) unchanged; only the PD level moves.
    """

    def gap(shift: float) -> float:
        return float(np.mean(1 / (1 + np.exp(-(log_odds + shift)))) - target_mean_pd)

    return float(optimize.brentq(gap, -20, 20))


def assign_grade(pd_: np.ndarray) -> np.ndarray:
    bounds = np.array([b for _, b in MASTER_SCALE])
    names = np.array([g for g, _ in MASTER_SCALE])
    idx = np.searchsorted(bounds, pd_, side="left")
    return names[np.minimum(idx, len(names) - 1)]


def grade_report(y: np.ndarray, pd_: np.ndarray, grades: np.ndarray | None = None) -> pd.DataFrame:
    """Per grade: obligors, mean PD, observed default rate and a one-sided binomial test.

    `grades` lets you assign grades with the long-run calibrated PD while testing
    against PDs at the sample's own default level (`pd_`). The binomial p-value tests
    H0 "the grade PD is not underestimated"; a small value flags a grade whose
    observed defaults exceed what its PD allows.
    """
    grades = assign_grade(pd_) if grades is None else grades
    df = pd.DataFrame({"grade": grades, "y": y, "pd": pd_})
    rows = []
    for grade, _ in MASTER_SCALE:
        g = df[df["grade"] == grade]
        if g.empty:
            continue
        n, d, p = len(g), int(g["y"].sum()), float(g["pd"].mean())
        rows.append(
            {
                "grade": grade,
                "n": n,
                "defaults": d,
                "mean_pd": p,
                "observed_dr": d / n,
                "binom_p": float(stats.binomtest(d, n, p, alternative="greater").pvalue),
            }
        )
    return pd.DataFrame(rows)


def is_monotonic(series: pd.Series) -> bool:
    return bool(series.is_monotonic_increasing)
