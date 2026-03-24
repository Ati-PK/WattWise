# src/models/linear.py
# ─────────────────────────────────────────────────────────────
# Regularized linear models: Ridge, Lasso, ElasticNet
# ─────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from sklearn.linear_model    import Ridge, Lasso, ElasticNet
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV


# ── Hyperparameter grids ──────────────────────────────────────
ALPHA_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

PARAM_GRIDS = {
    "ridge"      : {"model__alpha": ALPHA_GRID},
    "lasso"      : {"model__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]},
    "elasticnet" : {
        "model__alpha"   : [0.01, 0.1, 1.0, 10.0],  # focused range
        "model__l1_ratio": [0.7, 0.9],               # focused on L1-heavy
    },
}


# ── Helper: build and tune a pipeline ────────────────────────
def _fit_linear_model(
    model,
    param_grid: dict,
    X_train:    pd.DataFrame,
    y_train:    pd.Series,
    w_train:    pd.Series,
    n_splits:   int = 5,
) -> tuple:
    """
    Build a Pipeline (scaler + model), search for best
    hyperparameters using TimeSeriesSplit CV, refit on
    full training set, and return fitted pipeline + best params.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  model),
    ])

    tscv = TimeSeriesSplit(n_splits=n_splits)

    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        refit=True,       # refit best model on full X_train
        n_jobs=-1,        # use all CPU cores
        verbose=0,
    )

    search.fit(X_train, y_train, model__sample_weight=w_train)

    return search.best_estimator_, search.best_params_


# ── Public functions ──────────────────────────────────────────
def fit_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
) -> tuple:
    """Fit Ridge regression. Returns (fitted_pipeline, best_params)."""
    return _fit_linear_model(
        Ridge(),
        PARAM_GRIDS["ridge"],
        X_train, y_train, w_train,
    )


def fit_lasso(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
) -> tuple:
    """Fit Lasso regression. Returns (fitted_pipeline, best_params)."""
    return _fit_linear_model(
        Lasso(max_iter=10000, tol=1e-3),
        PARAM_GRIDS["lasso"],
        X_train, y_train, w_train,
    )


def fit_elasticnet(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
) -> tuple:
    """Fit ElasticNet regression. Returns (fitted_pipeline, best_params)."""
    return _fit_linear_model(
        ElasticNet(max_iter=10000, tol=1e-3),
        PARAM_GRIDS["elasticnet"],
        X_train, y_train, w_train,
    )
