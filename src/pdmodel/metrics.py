"""Discrimination, calibration and stability metrics used in credit-risk validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, roc_auc_score


def gini(y: np.ndarray, score: np.ndarray) -> float:
    """Gini = 2 * AUC - 1 (accuracy ratio)."""
    return 2.0 * roc_auc_score(y, score) - 1.0


def ks(y: np.ndarray, score: np.ndarray) -> float:
    """Kolmogorov-Smirnov: max distance between the score CDFs of bads and goods."""
    y = np.asarray(y)
    score = np.asarray(score)
    return float(stats.ks_2samp(score[y == 1], score[y == 0]).statistic)


def brier(y: np.ndarray, pd_: np.ndarray) -> float:
    return float(brier_score_loss(y, pd_))


def hosmer_lemeshow(y: np.ndarray, pd_: np.ndarray, groups: int = 10) -> tuple[float, float]:
    """Hosmer-Lemeshow goodness-of-fit test on PD deciles. Returns (statistic, p-value)."""
    df = pd.DataFrame({"y": y, "p": pd_})
    df["g"] = pd.qcut(df["p"].rank(method="first"), groups, labels=False)
    agg = df.groupby("g").agg(n=("y", "size"), obs=("y", "sum"), exp=("p", "sum"))
    exp = agg["exp"].to_numpy()
    obs = agg["obs"].to_numpy()
    n = agg["n"].to_numpy()
    hl = float(np.sum((obs - exp) ** 2 / (exp * (1 - exp / n))))
    return hl, float(stats.chi2.sf(hl, groups - 2))


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two score distributions.

    < 0.10 stable, 0.10-0.25 monitor, > 0.25 significant shift.
    """
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def bootstrap_ci(
    y: np.ndarray, score: np.ndarray, fn=gini, n: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """95 % bootstrap confidence interval for a metric."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    score = np.asarray(score)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        vals.append(fn(y[idx], score[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
