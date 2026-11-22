"""Central configuration for the PD scorecard pipeline."""

from __future__ import annotations

TARGET_RAW = "creditability"
TARGET = "bad"  # 1 = default ("bad"), 0 = non-default ("good")

# Excluded on fairness / regulatory grounds: sex and marital status, nationality.
# They are dropped before any binning so they cannot leak into the model.
EXCLUDED_FEATURES = ["personal_status_and_sex", "foreign_worker"]

SEED = 42
TEST_SIZE = 0.30
CV_FOLDS = 5

# Binning
MAX_BINS = 5          # initial quantile bins for numeric features
MIN_BIN_SHARE = 0.05  # every bin must hold at least 5 % of the sample
WOE_SMOOTHING = 0.5   # Laplace smoothing so empty cells do not give +/- inf

# Feature selection
MIN_IV = 0.02         # drop "useless" predictors (IV < 0.02)
MAX_CORR = 0.70       # drop the weaker of two WoE features correlated above this
MAX_PVALUE = 0.05     # backward elimination threshold

# Scorecard scaling: 600 points at good:bad odds of 50:1, 20 points to double the odds
BASE_SCORE = 600
BASE_ODDS = 50
PDO = 20

# Calibration: the German Credit sample over-represents defaults (30 %).
# Rating models are calibrated to a long-run central tendency instead.
CENTRAL_TENDENCY = 0.05

# Rating master scale: upper PD bound of each grade (last grade is open-ended)
MASTER_SCALE = [
    ("R1", 0.005),
    ("R2", 0.010),
    ("R3", 0.020),
    ("R4", 0.040),
    ("R5", 0.080),
    ("R6", 0.160),
    ("R7", 1.000),
]
