import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from pdmodel import calibration, metrics
from pdmodel.binning import WoETransformer, fit_binning
from pdmodel.config import CENTRAL_TENDENCY, EXCLUDED_FEATURES, MIN_BIN_SHARE, TARGET
from pdmodel.data import feature_columns, load, validate
from pdmodel.model import fit_scorecard


@pytest.fixture(scope="module")
def data():
    df = load()
    X, y = df[feature_columns(df)], df[TARGET]
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)


@pytest.fixture(scope="module")
def scorecard(data):
    X_tr, _, y_tr, _ = data
    return fit_scorecard(X_tr, y_tr)


# ---------- data ----------

def test_load_encodes_target_and_drops_protected_features():
    df = load()
    assert set(df[TARGET].unique()) == {0, 1}
    assert df[TARGET].mean() == pytest.approx(0.30)
    for c in EXCLUDED_FEATURES:
        assert c not in df.columns


def test_validate_rejects_missing_values():
    df = pd.DataFrame({"creditability": ["good"] * 200, "x": [1.0] * 199 + [np.nan]})
    with pytest.raises(ValueError, match="missing values"):
        validate(df)


def test_validate_rejects_unknown_labels():
    df = pd.DataFrame({"creditability": ["good"] * 199 + ["maybe"], "x": range(200)})
    with pytest.raises(ValueError, match="unexpected target"):
        validate(df)


# ---------- binning ----------

def test_numeric_bins_are_monotonic_and_large_enough():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=2000), name="x")
    y = (rng.random(2000) < 1 / (1 + np.exp(-x))).astype(int)
    b = fit_binning(x, y)
    rates = b.table["bad_rate"].to_numpy()
    assert np.all(np.diff(rates) >= 0) or np.all(np.diff(rates) <= 0)
    assert (b.table["count"] / len(x) >= MIN_BIN_SHARE).all()
    assert b.table["woe"].is_monotonic_increasing  # risk rises with x


def test_woe_sign_convention_higher_is_riskier():
    x = pd.Series(["a"] * 500 + ["b"] * 500, name="cat")
    y = np.array([0] * 450 + [1] * 50 + [0] * 250 + [1] * 250)
    t = fit_binning(x, y).table.set_index("label")
    assert t.loc["b", "woe"] > 0 > t.loc["a", "woe"]


def test_rare_categories_are_merged():
    x = pd.Series(["a"] * 600 + ["b"] * 380 + ["rare"] * 20, name="cat")
    y = np.r_[np.zeros(600), np.ones(380), np.ones(20)]
    b = fit_binning(x, y)
    assert (b.table["count"] / len(x) >= MIN_BIN_SHARE).all()
    assert b.groups["rare"] == b.groups["b"]


def test_uninformative_feature_has_near_zero_iv():
    rng = np.random.default_rng(1)
    x = pd.Series(rng.normal(size=5000), name="noise")
    y = rng.integers(0, 2, 5000)
    assert fit_binning(x, y).iv < 0.02


def test_unseen_category_maps_to_riskiest_bin(data):
    X_tr, _, y_tr, _ = data
    woe = WoETransformer().fit(X_tr[["purpose"]], y_tr)
    out = woe.transform(pd.DataFrame({"purpose": ["never seen before"]}))
    assert out.iloc[0, 0] == woe.binnings["purpose"].table["woe"].max()


# ---------- model ----------

def test_scorecard_coefficients_positive_and_significant(scorecard):
    params = scorecard.coef.drop("const")
    pvals = scorecard.result.pvalues.drop("const")
    assert (params > 0).all()
    assert (pvals <= 0.05).all()
    assert 4 <= len(scorecard.features) <= 15


def test_test_set_discrimination(data, scorecard):
    _, X_te, _, y_te = data
    g = metrics.gini(y_te, scorecard.predict_pd(X_te))
    assert g > 0.40  # typical published range for German Credit is 0.45-0.60


def test_higher_score_means_lower_pd(data, scorecard):
    _, X_te, _, _ = data
    s = scorecard.score(X_te)
    p = scorecard.predict_pd(X_te)
    assert np.corrcoef(s, p)[0, 1] < -0.95


def test_points_table_reproduces_score(data, scorecard):
    _, X_te, _, _ = data
    row = X_te.iloc[[0]]
    pts = scorecard.points_table()
    total = 0
    for f in scorecard.features:
        b = scorecard.woe.binnings[f]
        x = row[f] if b.kind == "numeric" else row[f].astype(str)
        label = b.table["label"].iloc[b.assign(x)[0]]
        total += pts[(pts.feature == f) & (pts.bin == label)]["points"].iloc[0]
    assert total == pytest.approx(scorecard.score(row)[0], abs=len(scorecard.features))


# ---------- metrics ----------

def test_gini_and_ks_extremes():
    y = np.array([0, 0, 1, 1])
    assert metrics.gini(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert metrics.ks(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert metrics.gini(y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(-1.0)


def test_psi_zero_for_identical_and_large_for_shift():
    rng = np.random.default_rng(2)
    a = rng.normal(size=5000)
    assert metrics.psi(a, a) == pytest.approx(0.0, abs=1e-9)
    assert metrics.psi(a, a + 1.0) > 0.25


def test_hosmer_lemeshow_accepts_well_calibrated_pd():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.01, 0.6, 20000)
    y = (rng.random(20000) < p).astype(int)
    _, pval = metrics.hosmer_lemeshow(y, p)
    assert pval > 0.01


# ---------- calibration & rating scale ----------

def test_intercept_shift_hits_central_tendency_and_keeps_ranking(data, scorecard):
    X_tr, X_te, _, y_te = data
    shift = calibration.intercept_shift_for(scorecard.log_odds(X_tr), CENTRAL_TENDENCY)
    before = scorecard.predict_pd(X_te)
    scorecard.intercept_shift = shift
    try:
        assert scorecard.predict_pd(X_tr).mean() == pytest.approx(CENTRAL_TENDENCY, abs=1e-6)
        after = scorecard.predict_pd(X_te)
    finally:
        scorecard.intercept_shift = 0.0
    assert metrics.gini(y_te, after) == pytest.approx(metrics.gini(y_te, before))


def test_master_scale_assignment():
    g = calibration.assign_grade(np.array([0.001, 0.005, 0.0051, 0.03, 0.5]))
    assert list(g) == ["R1", "R1", "R2", "R4", "R7"]


def test_grade_default_rates_monotonic_on_test(data, scorecard):
    X_tr, X_te, _, y_te = data
    scorecard.intercept_shift = calibration.intercept_shift_for(
        scorecard.log_odds(X_tr), CENTRAL_TENDENCY
    )
    grades = calibration.assign_grade(scorecard.predict_pd(X_te))
    scorecard.intercept_shift = 0.0
    tbl = calibration.grade_report(y_te.to_numpy(), scorecard.predict_pd(X_te), grades)
    assert tbl["mean_pd"].is_monotonic_increasing
    assert calibration.is_monotonic(tbl["observed_dr"])
