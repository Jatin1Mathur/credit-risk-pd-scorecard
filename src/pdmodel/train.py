"""End-to-end run: split, cross-validate, fit, calibrate, grade, benchmark, report.

Usage:  pd-train            (writes everything to reports/)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402

from . import calibration, metrics  # noqa: E402
from .config import CENTRAL_TENDENCY, CV_FOLDS, SEED, TARGET, TEST_SIZE  # noqa: E402
from .data import feature_columns, load  # noqa: E402
from .model import fit_scorecard  # noqa: E402


def cross_validate(X: pd.DataFrame, y: pd.Series) -> dict:
    """Refit the WHOLE pipeline (binning + selection + logit) inside each fold.

    Fitting the WoE bins on all data first would leak target information into the
    validation folds and overstate Gini.
    """
    skf = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
    ginis, n_feats = [], []
    for tr, va in skf.split(X, y):
        sc = fit_scorecard(X.iloc[tr], y.iloc[tr])
        ginis.append(metrics.gini(y.iloc[va], sc.predict_pd(X.iloc[va])))
        n_feats.append(len(sc.features))
    return {"gini_mean": float(np.mean(ginis)), "gini_std": float(np.std(ginis)), "n_features": n_feats}


def benchmark_gbm(X_tr, y_tr, X_te, y_te) -> dict:
    """Challenger model: gradient boosting on raw features (native categorical support)."""
    cat_cols = [c for c in X_tr.columns if not pd.api.types.is_numeric_dtype(X_tr[c])]

    def encode(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in cat_cols:
            out[c] = pd.Categorical(out[c], categories=sorted(X_tr[c].unique()))
        return out

    def model():
        return HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
            categorical_features="from_dtype", random_state=SEED,
        )

    skf = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
    Xe = encode(X_tr)
    cv = []
    for tr, va in skf.split(Xe, y_tr):
        m = model().fit(Xe.iloc[tr], y_tr.iloc[tr])
        cv.append(metrics.gini(y_tr.iloc[va], m.predict_proba(Xe.iloc[va])[:, 1]))
    m = model().fit(Xe, y_tr)
    p = m.predict_proba(encode(X_te))[:, 1]
    return {
        "cv_gini_mean": float(np.mean(cv)),
        "cv_gini_std": float(np.std(cv)),
        "test_gini": metrics.gini(y_te, p),
        "test_auc": float(roc_auc_score(y_te, p)),
    }


def plots(out: Path, y_te, pd_te, gbm_auc, grades: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    fpr, tpr, _ = roc_curve(y_te, pd_te)
    axes[0].plot(fpr, tpr, label=f"Scorecard (AUC {roc_auc_score(y_te, pd_te):.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="grey", label="Random")
    axes[0].set(title="ROC, test set", xlabel="False positive rate", ylabel="True positive rate")
    axes[0].legend(loc="lower right")

    df = pd.DataFrame({"y": y_te, "p": pd_te})
    df["decile"] = pd.qcut(df["p"].rank(method="first"), 10, labels=False)
    agg = df.groupby("decile").agg(pred=("p", "mean"), obs=("y", "mean"))
    axes[1].plot(agg["pred"], agg["obs"], "o-", label="Scorecard")
    axes[1].plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    axes[1].set(
        title="Calibration by PD decile, test set", xlabel="Predicted PD", ylabel="Observed default rate"
    )
    axes[1].legend(loc="upper left")

    x = np.arange(len(grades))
    axes[2].bar(x - 0.2, grades["observed_dr"], 0.4, label="Observed default rate")
    axes[2].bar(x + 0.2, grades["mean_pd"], 0.4, label="Mean PD (sample level)")
    axes[2].set_xticks(x, grades["grade"])
    axes[2].set(title="Default rate by rating grade, test set", ylabel="Default rate")
    axes[2].legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out / "validation.png", dpi=120)
    plt.close(fig)


def run(out_dir: str | Path = "reports") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load()
    X, y = df[feature_columns(df)], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )

    cv = cross_validate(X_tr, y_tr)
    sc = fit_scorecard(X_tr, y_tr)

    pd_te = sc.predict_pd(X_te)
    hl_stat, hl_p = metrics.hosmer_lemeshow(y_te.to_numpy(), pd_te)
    gini_lo, gini_hi = metrics.bootstrap_ci(y_te.to_numpy(), pd_te)

    # Long-run calibration: shift the intercept to the central tendency, then grade.
    shift = calibration.intercept_shift_for(sc.log_odds(X_tr), CENTRAL_TENDENCY)
    sc.intercept_shift = shift
    pd_te_lr = sc.predict_pd(X_te)
    grades_te = calibration.assign_grade(pd_te_lr)
    sc.intercept_shift = 0.0
    grade_tbl = calibration.grade_report(y_te.to_numpy(), pd_te, grades=grades_te)

    gbm = benchmark_gbm(X_tr, y_tr, X_te, y_te)

    results = {
        "data": {"rows": len(df), "default_rate": float(y.mean()), "train": len(X_tr), "test": len(X_te)},
        "features": {
            "candidates": len(X.columns),
            "selected": sc.features,
            "dropped": sc.dropped,
            "iv": {k: round(v, 4) for k, v in sc.woe.iv_table().items()},
        },
        "coefficients": {k: round(float(v), 4) for k, v in sc.coef.items()},
        "p_values": {k: round(float(v), 4) for k, v in sc.result.pvalues.items()},
        "cv": cv,
        "test": {
            "auc": float(roc_auc_score(y_te, pd_te)),
            "gini": metrics.gini(y_te, pd_te),
            "gini_95ci": [gini_lo, gini_hi],
            "ks": metrics.ks(y_te.to_numpy(), pd_te),
            "brier": metrics.brier(y_te, pd_te),
            "hosmer_lemeshow": {"stat": hl_stat, "p_value": hl_p},
            "psi_train_vs_test": metrics.psi(sc.score(X_tr), sc.score(X_te)),
        },
        "calibration": {
            "central_tendency": CENTRAL_TENDENCY,
            "intercept_shift": shift,
            "grades_monotonic": calibration.is_monotonic(grade_tbl["observed_dr"]),
            "grades_failing_binomial_5pct": grade_tbl.loc[grade_tbl["binom_p"] < 0.05, "grade"].tolist(),
        },
        "benchmark_gbm": gbm,
    }

    (out / "results.json").write_text(json.dumps(results, indent=2))
    sc.points_table().to_csv(out / "scorecard_points.csv", index=False)
    sc.woe.iv_table().rename("iv").to_csv(out / "information_value.csv")
    grade_tbl.round(4).to_csv(out / "grades_test.csv", index=False)
    plots(out, y_te.to_numpy(), pd_te, gbm["test_auc"], grade_tbl)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Train and validate the PD scorecard")
    ap.add_argument("--out", default="reports")
    r = run(ap.parse_args().out)
    t = r["test"]
    print(f"Selected {len(r['features']['selected'])} features: {r['features']['selected']}")
    print(f"CV Gini {r['cv']['gini_mean']:.3f} +/- {r['cv']['gini_std']:.3f}")
    print(f"Test AUC {t['auc']:.3f} | Gini {t['gini']:.3f} | KS {t['ks']:.3f} | Brier {t['brier']:.3f}")
    print(f"Hosmer-Lemeshow p = {t['hosmer_lemeshow']['p_value']:.3f} | PSI {t['psi_train_vs_test']:.3f}")
    print(f"GBM challenger test Gini {r['benchmark_gbm']['test_gini']:.3f}")


if __name__ == "__main__":
    main()
