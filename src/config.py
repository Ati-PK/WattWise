# src/config.py
# ─────────────────────────────────────────────────────────────
# Central configuration for the BESS price forecasting project.
# All constants, paths, and shared settings live here.
# ─────────────────────────────────────────────────────────────

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent
DATA_PATH   = ROOT_DIR / "data" / "df_features.csv"
RESULTS_DIR = ROOT_DIR / "results" / "scores"
PLOTS_DIR   = ROOT_DIR / "results" / "plots"

# ── Time splits ──────────────────────────────────────────────
VAL_START  = "2025-01-01"
TEST_START = "2025-07-01"

# ── Target & leakage ─────────────────────────────────────────
TARGET = "price"

FEATURES_TO_DROP = [
    "actual_wind_offshore",
    "actual_wind_onshore",
    "actual_solar",
    "actual_load",
    "is_negative_price",
    "is_high_price_regime",
    "timestamp",
]

# ── Regime weights ───────────────────────────────────────────
REGIME_WEIGHTS = {
    ("2019-01-01", "2020-12-31"): 0.6,   # pre-crisis
    ("2021-01-01", "2022-12-31"): 0.3,   # energy crisis
    ("2023-01-01", "2025-06-30"): 1.0,   # post-crisis
}