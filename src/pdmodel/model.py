"""Scorecard model: feature selection, logistic regression on WoE, scaling to points."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .binning import WoETransformer
from .config import BASE_ODDS, BASE_SCORE, MAX_CORR, MAX_PVALUE, MIN_IV, PDO


def select_features(woe: pd.DataFrame, iv: pd.Series) -> tuple[list[str], dict[str, str]]:
    """IV filter, then correlation filter. Returns kept features and the reason for each drop."""
    dropped: dict[str, str] = {}
    candidates = [c for c in iv.index if iv[c] >= MIN_IV]
    for c in iv.index:
        if c not in candidates:
            dropped[c] = f"IV {iv[c]:.3f} < {MIN_IV}"
    kept: list[str] = []
    for c in candidates:  # iv is sorted descending, so stronger features win
        corr = [abs(np.corrcoef(woe[c], woe[k])[0, 1]) for k in kept]
        if corr and max(corr) > MAX_CORR:
            dropped[c] = f"|corr| {max(corr):.2f} > {MAX_CORR} with stronger feature"
        else:
            kept.append(c)
    return kept, dropped


def fit_logit(woe: pd.DataFrame, y: pd.Series, features: list[str]):
    """Backward elimination: drop features with p > MAX_PVALUE or a wrong (negative) sign."""
    feats = list(features)
    dropped: dict[str, str] = {}
    while True:
        res = sm.Logit(y.to_numpy(), sm.add_constant(woe[feats])).fit(disp=0)
        params = res.params.drop("const")
        pvals = res.pvalues.drop("const")
        wrong_sign = params[params <= 0]
        if len(wrong_sign):
            f = wrong_sign.index[0]
            dropped[f] = f"coefficient {params[f]:.3f} has wrong sign"
        elif pvals.max() > MAX_PVALUE:
            f = pvals.idxmax()
            dropped[f] = f"p-value {pvals[f]:.3f} > {MAX_PVALUE}"
        else:
            return res, feats, dropped
        feats.remove(f)


@dataclass
class Scorecard:
    woe: WoETransformer
    features: list[str]
    result: object  # statsmodels LogitResults
    intercept_shift: float = 0.0  # log-odds adjustment from calibration
    dropped: dict[str, str] = field(default_factory=dict)

    @property
    def coef(self) -> pd.Series:
        return self.result.params

    def log_odds(self, X: pd.DataFrame) -> np.ndarray:
        """Log-odds of default (bad:good)."""
        W = self.woe.transform(X, self.features)
        return (
            self.coef["const"] + self.intercept_shift + W.to_numpy() @ self.coef[self.features].to_numpy()
        )

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.log_odds(X)))

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Scorecard points: higher = safer. Uses the uncalibrated model so points are stable."""
        factor = PDO / np.log(2)
        offset = BASE_SCORE - factor * np.log(BASE_ODDS)
        good_odds_log = -(self.log_odds(X) - self.intercept_shift)
        return offset + factor * good_odds_log

    def points_table(self) -> pd.DataFrame:
        """Points per bin, the familiar scorecard format."""
        factor = PDO / np.log(2)
        offset = BASE_SCORE - factor * np.log(BASE_ODDS)
        n = len(self.features)
        rows = []
        for f in self.features:
            b = self.woe.binnings[f].table
            beta = self.coef[f]
            for _, r in b.iterrows():
                pts = -(beta * r["woe"] + self.coef["const"] / n) * factor + offset / n
                rows.append(
                    {
                        "feature": f,
                        "bin": r["label"],
                        "count": r["count"],
                        "bad_rate": round(r["bad_rate"], 3),
                        "woe": round(r["woe"], 3),
                        "points": int(round(pts)),
                    }
                )
        return pd.DataFrame(rows)


def fit_scorecard(X: pd.DataFrame, y: pd.Series) -> Scorecard:
    woe = WoETransformer().fit(X, y)
    W = woe.transform(X)
    kept, dropped = select_features(W, woe.iv_table())
    result, feats, dropped_logit = fit_logit(W, y, kept)
    return Scorecard(woe=woe, features=feats, result=result, dropped={**dropped, **dropped_logit})
