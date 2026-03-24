# src/models/trees.py
# ─────────────────────────────────────────────────────────────
# Tree-based ensemble models: Random Forest, XGBoost,
# LightGBM, and CatBoost.
#
# No feature scaling needed — tree models are scale-invariant.
# Weights applied during training only.
# Hyperparameters tuned via TimeSeriesSplit CV.
# ─────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from xgboost                 import XGBRegressor
from lightgbm                import LGBMRegressor
from catboost                import CatBoostRegressor


# ── Hyperparameter grids ──────────────────────────────────────
PARAM_GRIDS = {

    "random_forest": {
        "n_estimators": [200, 500],
        "max_depth"   : [3, 5, 7, 10],   # removed None, added smaller values
        "max_features": [0.3, 0.5, 0.7], # added 0.3 for more aggressive feature subsampling
},

    "xgboost": {
        "n_estimators" : [200, 500],
        "learning_rate": [0.05, 0.1],
        "max_depth"    : [4, 6],
        "subsample"    : [0.7, 1.0],
    },

    "lightgbm": {
        "n_estimators" : [200, 500],
        "learning_rate": [0.05, 0.1],
        "max_depth"    : [4, 6],
        "subsample"    : [0.7, 1.0],
    },

    "catboost": {
        "iterations"   : [200, 500],
        "learning_rate": [0.05, 0.1],
        "depth"        : [4, 6],
    },
}


# ── Helper: run GridSearchCV with TimeSeriesSplit ─────────────
def _tune_model(
    model,
    param_grid : dict,
    X_train    : pd.DataFrame,
    y_train    : pd.Series,
    w_train    : pd.Series,
    fit_params : dict = None,
    n_splits   : int  = 5,
) -> tuple:
    """
    Search best hyperparameters, refit on full training set.
    Returns (best_estimator, best_params).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    search = GridSearchCV(
        model,
        param_grid,
        cv      = tscv,
        scoring = "neg_mean_absolute_error",
        refit   = True,
        n_jobs  = -1,
        verbose = 0,
    )

    fit_params = fit_params or {}
    search.fit(X_train, y_train,
               sample_weight=w_train,
               **fit_params)

    return search.best_estimator_, search.best_params_


# ── Public functions ──────────────────────────────────────────
def fit_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
) -> tuple:
    """Fit Random Forest. Returns (fitted_model, best_params)."""
    return _tune_model(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        PARAM_GRIDS["random_forest"],
        X_train, y_train, w_train,
    )


def fit_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
) -> tuple:
    """Fit XGBoost. Returns (fitted_model, best_params)."""
    return _tune_model(
        XGBRegressor(random_state=42, verbosity=0),
        PARAM_GRIDS["xgboost"],
        X_train, y_train, w_train,
    )


def fit_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
) -> tuple:
    """Fit LightGBM. Returns (fitted_model, best_params)."""
    return _tune_model(
        LGBMRegressor(random_state=42, verbosity=-1),
        PARAM_GRIDS["lightgbm"],
        X_train, y_train, w_train,
    )


def fit_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
) -> tuple:
    """
    Fit CatBoost. Returns (fitted_model, best_params).
    Note: categorical features not passed explicitly —
    all features treated as numeric. CatBoost native
    categorical handling requires string types, but our
    binary/ordinal features are already numeric and work
    better as-is.
    """
    return _tune_model(
        CatBoostRegressor(
            random_state = 42,
            verbose      = 0,
        ),
        PARAM_GRIDS["catboost"],
        X_train, y_train, w_train,
    )
