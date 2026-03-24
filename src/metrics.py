# src/metrics.py
# ─────────────────────────────────────────────────────────────
# Metric functions for model evaluation.
# Single source of truth — all models import from here.
# Weights are NOT applied here (unweighted evaluation throughout).
# ─────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from datetime import datetime


def mae(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Mean Absolute Error in EUR/MWh."""
    return float(np.mean(np.abs(y_true - y_pred)))


def smape(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error in %.
    Zero-denominator hours are excluded (handles near-zero prices).
    """
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask  = denom != 0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def dae(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """
    Daily Average Error in EUR/MWh.
    Sums actuals and predictions per day first, then takes
    absolute difference.
    Only includes days with all 24 hours present.
    """
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred, index=y_true.index)

    df = pd.DataFrame({"true": y_true, "pred": y_pred})
    df["date"] = df.index.floor("D")

    complete = df.groupby("date").filter(
        lambda g: g.index.hour.nunique() == 24
    )

    daily = complete.groupby("date")[["true", "pred"]].sum()
    return float(np.abs(daily["true"] - daily["pred"]).mean())


def dae_norm(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """
    Daily Average Error in EUR/MWh.
    Computes mean of 24 hourly prices per day for actual and predicted,
    then averages the absolute differences across all days.
    Equivalent to DAE_sum / 24 — directly interpretable as 
    average hourly price level error per day.
    Only includes days with all 24 hours present.
    """
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred, index=y_true.index)

    df = pd.DataFrame({"true": y_true, "pred": y_pred})
    df["date"] = df.index.floor("D")

    complete = df.groupby("date").filter(
        lambda g: g.index.hour.nunique() == 24
    )

    daily  = complete.groupby("date")[["true", "pred"]].mean() # mean not sum
    denom  = (np.abs(daily["true"]) + np.abs(daily["pred"])) / 2
    mask   = denom != 0
    errors = np.abs(daily["true"][mask] - daily["pred"][mask]) / denom[mask]
    return float(errors.mean() * 100)

def rmae(y_true: pd.Series, y_pred: np.ndarray,
         naive_mae: float) -> float:
    """
    Relative MAE — model MAE normalised by naive baseline MAE.
    
    RMAE = 1.0  → same performance as naive
    RMAE = 0.5  → 50% better than naive
    RMAE = 0.0  → perfect forecast
    
    Args:
        y_true     : actual prices
        y_pred     : predicted prices
        naive_mae  : MAE of naive model on same period
    """
    return float(mae(y_true, y_pred) / naive_mae)

def evaluate_all(
    y_true     : pd.Series,
    y_pred     : np.ndarray,
    model_name : str,
    split      : str,
    naive_mae  : float = None,
) -> dict:
    """
    Compute all metrics for a given model and return as a
    structured dictionary ready for a results table.

    Args:
        y_true     : actual prices
        y_pred     : predicted prices
        model_name : e.g. "naive", "lasso", "xgboost"
        split      : "validation" or "test"
        naive_mae  : MAE of naive model on same period (optional)
                     if provided, RMAE is computed

    Returns:
        dict with model, split, timestamp, MAE, DAE, DAE_norm,
        SMAPE, and optionally RMAE
    """
    result = {
        "model"    : model_name,
        "split"    : split,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "MAE"      : round(mae(y_true, y_pred),      4),
        "DAE"      : round(dae(y_true, y_pred),      4),
        "DAE_norm" : round(dae_norm(y_true, y_pred), 4),
        "SMAPE"    : round(smape(y_true, y_pred),    4),
    }

    if naive_mae is not None:
        result["RMAE"] = round(rmae(y_true, y_pred, naive_mae), 4)

    return result