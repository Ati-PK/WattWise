# src/models/naive.py
# ─────────────────────────────────────────────────────────────
# Naive day-ahead price forecast baseline.
# Predicts next day's 24-hour price vector by copying
# historical prices — lag-24h or lag-168h depending on weekday.
#
# Rule:
#   Mon, Sat → copy same day last week (lag-168h)
#   All other days → copy yesterday (lag-24h)
#
# No training required. No features used. Price Series only.
# ─────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np


def predict_naive(y: pd.Series) -> pd.Series:
    """
    Generate naive day-ahead price forecasts.

    Args:
        y : hourly price Series with DatetimeIndex

    Returns:
        pd.Series of predicted prices, same index as y
    """
    lag_24  = y.shift(24)   # yesterday same hour
    lag_168 = y.shift(168)  # last week same hour

    # Monday=0, Saturday=5
    mon_sat = y.index.dayofweek.isin([0, 5])

    predictions = pd.Series(
        np.where(mon_sat, lag_168, lag_24),
        index=y.index,
        name="naive_forecast"
    )

    return predictions