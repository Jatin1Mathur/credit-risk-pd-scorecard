"""Weight-of-Evidence (WoE) binning and Information Value (IV).

WoE is defined as ln(%bad / %good) per bin, so a higher WoE means riskier and the
logistic-regression coefficients on WoE features are expected to be positive.

Numeric features start from quantile bins; adjacent bins are then merged until
(a) every bin holds at least MIN_BIN_SHARE of the sample and (b) the bad rate is
monotonic across bins. Categorical features keep one bin per category; rare
categories are merged with the category whose bad rate is closest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import MAX_BINS, MIN_BIN_SHARE, WOE_SMOOTHING


@dataclass
class FeatureBinning:
    name: str
    kind: str  # "numeric" or "categorical"
    edges: list[float] = field(default_factory=list)       # numeric: inner cut points
    groups: dict[str, int] = field(default_factory=dict)   # categorical: category -> bin id
    table: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def iv(self) -> float:
        return float(self.table["iv"].sum())

    def assign(self, x: pd.Series) -> np.ndarray:
        """Bin id for every value."""
        if self.kind == "numeric":
            return np.searchsorted(np.asarray(self.edges), x.to_numpy(dtype=float), side="right")
        # unseen categories go to the riskiest bin (conservative)
        fallback = int(self.table["woe"].idxmax())
        return x.map(self.groups).fillna(fallback).astype(int).to_numpy()

    def transform(self, x: pd.Series) -> np.ndarray:
        return self.table["woe"].to_numpy()[self.assign(x)]


def _woe_table(bins: np.ndarray, y: np.ndarray, n_bins: int) -> pd.DataFrame:
    total_bad = y.sum()
    total_good = len(y) - total_bad
    rows = []
    for b in range(n_bins):
        mask = bins == b
        bad = y[mask].sum()
        good = mask.sum() - bad
        p_bad = (bad + WOE_SMOOTHING) / (total_bad + WOE_SMOOTHING * n_bins)
        p_good = (good + WOE_SMOOTHING) / (total_good + WOE_SMOOTHING * n_bins)
        woe = np.log(p_bad / p_good)
        rows.append(
            {
                "bin": b,
                "count": int(mask.sum()),
                "bad": int(bad),
                "bad_rate": bad / mask.sum() if mask.sum() else np.nan,
                "woe": woe,
                "iv": (p_bad - p_good) * woe,
            }
        )
    return pd.DataFrame(rows).set_index("bin")


def _is_monotonic(rates: list[float]) -> bool:
    d = np.diff(rates)
    return bool(np.all(d >= 0) or np.all(d <= 0))


def _fit_numeric(name: str, x: pd.Series, y: np.ndarray) -> FeatureBinning:
    values = x.to_numpy(dtype=float)
    qs = np.unique(np.quantile(values, np.linspace(0, 1, MAX_BINS + 1)[1:-1]))
    edges = list(qs)

    def stats(edges_: list[float]) -> tuple[np.ndarray, np.ndarray]:
        b = np.searchsorted(np.asarray(edges_), values, side="right")
        n = len(edges_) + 1
        counts = np.bincount(b, minlength=n)
        bads = np.bincount(b, weights=y, minlength=n)
        return counts, bads

    # merge until every bin is large enough and the bad rate is monotonic
    while edges:
        counts, bads = stats(edges)
        rates = bads / np.maximum(counts, 1)
        small = counts / len(values) < MIN_BIN_SHARE
        if not small.any() and _is_monotonic(list(rates)):
            break
        if small.any():
            i = int(np.argmin(counts))
        else:
            # merge the adjacent pair with the closest bad rates
            i = int(np.argmin(np.abs(np.diff(rates))))
        # bin i spans (edges[i-1], edges[i]]; drop the edge shared with its neighbour
        drop = min(i, len(edges) - 1)
        edges.pop(drop)

    bins = np.searchsorted(np.asarray(edges), values, side="right")
    table = _woe_table(bins, y, len(edges) + 1)
    labels = []
    bounds = [-np.inf] + edges + [np.inf]
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        labels.append(f"[{lo:g}, {hi:g})")
    table.insert(0, "label", labels)
    return FeatureBinning(name, "numeric", edges=edges, table=table)


def _fit_categorical(name: str, x: pd.Series, y: np.ndarray) -> FeatureBinning:
    df = pd.DataFrame({"x": x.astype(str).to_numpy(), "y": y})
    stats = df.groupby("x")["y"].agg(["count", "mean"])
    groups = {cat: [cat] for cat in stats.index}

    # merge rare categories into the group with the nearest bad rate
    def group_stats() -> pd.DataFrame:
        rows = {}
        for key, cats in groups.items():
            sub = stats.loc[cats]
            n = sub["count"].sum()
            rows[key] = {"count": n, "rate": (sub["count"] * sub["mean"]).sum() / n}
        return pd.DataFrame(rows).T

    while len(groups) > 2:
        g = group_stats()
        share = g["count"] / len(df)
        if (share >= MIN_BIN_SHARE).all():
            break
        rare = share.idxmin()
        others = g.drop(index=rare)
        target = (others["rate"] - g.loc[rare, "rate"]).abs().idxmin()
        groups[target].extend(groups.pop(rare))

    # order bins by bad rate so bin ids are readable
    g = group_stats().sort_values("rate")
    mapping: dict[str, int] = {}
    labels = []
    for bin_id, key in enumerate(g.index):
        for cat in groups[key]:
            mapping[cat] = bin_id
        labels.append(" | ".join(groups[key]))

    bins = df["x"].map(mapping).to_numpy()
    table = _woe_table(bins, y, len(labels))
    table.insert(0, "label", labels)
    return FeatureBinning(name, "categorical", groups=mapping, table=table)


def fit_binning(x: pd.Series, y: np.ndarray | pd.Series) -> FeatureBinning:
    y = np.asarray(y, dtype=float)
    if pd.api.types.is_numeric_dtype(x) and x.nunique() > MAX_BINS:
        return _fit_numeric(x.name, x, y)
    return _fit_categorical(x.name, x.astype(str), y)


class WoETransformer:
    """Fits one binning per feature; transforms raw data into WoE values."""

    def __init__(self) -> None:
        self.binnings: dict[str, FeatureBinning] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> WoETransformer:
        self.binnings = {c: fit_binning(X[c], y) for c in X.columns}
        return self

    def transform(self, X: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        cols = columns or list(self.binnings)
        out = {}
        for c in cols:
            x = X[c] if self.binnings[c].kind == "numeric" else X[c].astype(str)
            out[c] = self.binnings[c].transform(x)
        return pd.DataFrame(out, index=X.index)

    def iv_table(self) -> pd.Series:
        return pd.Series({c: b.iv for c, b in self.binnings.items()}).sort_values(ascending=False)
