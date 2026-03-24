# src/weights.py
# ─────────────────────────────────────────────────────────────
# Creates sample weights for model training.
# Applied during .fit() only — not during metric evaluation.
# ─────────────────────────────────────────────────────────────

import pandas as pd
from src.config import REGIME_WEIGHTS


def create_regime_weights(index: pd.DatetimeIndex) -> pd.Series:
    """
    Assign a weight to each observation based on its time regime.
    
    Regime weights are defined in config.py:
        2019–2020 : 0.6  (pre-crisis, older market structure)
        2021–2022 : 0.3  (energy crisis, anomalous period)
        2023–2024 : 1.0  (post-crisis, most representative)

    Args:
        index : DatetimeIndex of the training DataFrame

    Returns:
        pd.Series of weights aligned to the input index
    """
    weights = pd.Series(1.0, index=index)

    for (start, end), weight in REGIME_WEIGHTS.items():
        mask = (index >= start) & (index <= end)
        weights.loc[mask] = weight

    return weights