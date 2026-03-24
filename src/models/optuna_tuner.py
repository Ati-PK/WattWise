# src/models/optuna_tuner.py
# ─────────────────────────────────────────────────────────────
# Optuna-based hyperparameter tuning for XGBoost and CatBoost.
# Uses TimeSeriesSplit to respect temporal order.
# Replaces GridSearchCV for finer hyperparameter search.
# ─────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import optuna
from tqdm.auto import tqdm
from optuna.samplers        import TPESampler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics         import mean_absolute_error
from xgboost                 import XGBRegressor
from catboost                import CatBoostRegressor

# Suppress Optuna logging — we'll print our own summary
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── XGBoost ───────────────────────────────────────────────────
def tune_xgboost(
    X_train      : pd.DataFrame,
    y_train      : pd.Series,
    w_train      : pd.Series,
    n_trials     : int = 50,
    n_splits     : int = 5,
    cv_start_date: str = None,
) -> tuple:
    """
    Tune XGBoost hyperparameters using Optuna TPE sampler.
    Returns (fitted_model, best_params, best_mae).

    Args:
        n_trials      : number of Optuna trials
        n_splits      : TimeSeriesSplit folds
        cv_start_date : if provided, only use data from this date
                        onwards for CV fold construction.
                        Aligns CV distribution with validation period.
                        Example: "2023-01-01"
    """
    # Filter to recent data for CV if requested
    if cv_start_date is not None:
        X_cv = X_train[X_train.index >= cv_start_date]
        y_cv = y_train[y_train.index >= cv_start_date]
        w_cv = w_train[w_train.index >= cv_start_date]
        print(f"  CV restricted to {cv_start_date} onwards: "
              f"{len(X_cv):,} rows")
    else:
        X_cv, y_cv, w_cv = X_train, y_train, w_train

    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial):
        params = {
            "n_estimators"     : trial.suggest_int(
                                   "n_estimators", 100, 1000),
            "learning_rate"    : trial.suggest_float(
                                   "learning_rate", 0.01, 0.3,
                                   log=True),
            "max_depth"        : trial.suggest_int(
                                   "max_depth", 3, 8),
            "subsample"        : trial.suggest_float(
                                   "subsample", 0.5, 1.0),
            "colsample_bytree" : trial.suggest_float(
                                   "colsample_bytree", 0.5, 1.0),
            "min_child_weight" : trial.suggest_int(
                                   "min_child_weight", 1, 10),
            "reg_alpha"        : trial.suggest_float(
                                   "reg_alpha", 1e-8, 1.0,
                                   log=True),
            "reg_lambda"       : trial.suggest_float(
                                   "reg_lambda", 1e-8, 1.0,
                                   log=True),
            "random_state"     : 42,
            "verbosity"        : 0,
        }

        fold_maes = []
        for train_idx, val_idx in tscv.split(X_cv):
            X_fold_train = X_cv.iloc[train_idx]
            y_fold_train = y_cv.iloc[train_idx]
            w_fold_train = w_cv.iloc[train_idx]
            X_fold_val   = X_cv.iloc[val_idx]
            y_fold_val   = y_cv.iloc[val_idx]

            model = XGBRegressor(**params)
            model.fit(
                X_fold_train, y_fold_train,
                sample_weight = w_fold_train,
                verbose       = False,
            )
            preds = model.predict(X_fold_val)
            fold_maes.append(mean_absolute_error(y_fold_val, preds))

        return np.mean(fold_maes)

    sampler = TPESampler(seed=42)
    study   = optuna.create_study(
                direction = "minimize",
                sampler   = sampler,
              )
    study.optimize(objective, n_trials=n_trials,
                   show_progress_bar=True)

    # Refit on FULL training data — not just CV subset
    best_params = study.best_params
    best_model  = XGBRegressor(
                    **best_params,
                    random_state = 42,
                    verbosity    = 0,
                  )
    best_model.fit(X_train, y_train, sample_weight=w_train)

    return best_model, best_params, round(study.best_value, 4)

# ── CatBoost ──────────────────────────────────────────────────
def tune_catboost(
    X_train   : pd.DataFrame,
    y_train   : pd.Series,
    w_train   : pd.Series,
    n_trials  : int = 50,
    n_splits  : int = 5,
) -> tuple:
    """
    Tune CatBoost hyperparameters using Optuna TPE sampler.
    Returns (fitted_model, best_params, best_mae).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial):
        params = {
            "iterations"        : trial.suggest_int(
                                    "iterations", 100, 1000),
            "learning_rate"     : trial.suggest_float(
                                    "learning_rate", 0.01, 0.3,
                                    log=True),
            "depth"             : trial.suggest_int(
                                    "depth", 3, 8),
            "l2_leaf_reg"       : trial.suggest_float(
                                    "l2_leaf_reg", 1e-8, 10.0,
                                    log=True),
            "bagging_temperature": trial.suggest_float(
                                    "bagging_temperature", 0.0, 1.0),
            "random_strength"   : trial.suggest_float(
                                    "random_strength", 1e-8, 10.0,
                                    log=True),
            "random_seed"       : 42,
            "verbose"           : 0,
        }

        fold_maes = []
        for train_idx, val_idx in tscv.split(X_train):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            w_fold_train = w_train.iloc[train_idx]
            X_fold_val   = X_train.iloc[val_idx]
            y_fold_val   = y_train.iloc[val_idx]

            model = CatBoostRegressor(**params)
            model.fit(
                X_fold_train, y_fold_train,
                sample_weight = w_fold_train,
            )
            preds = model.predict(X_fold_val)
            fold_maes.append(mean_absolute_error(y_fold_val, preds))

        return np.mean(fold_maes)

    sampler = TPESampler(seed=42)
    study   = optuna.create_study(
                direction = "minimize",
                sampler   = sampler,
              )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_model  = CatBoostRegressor(
                    **best_params,
                    random_seed = 42,
                    verbose     = 0,
                  )
    best_model.fit(X_train, y_train, sample_weight=w_train)

    return best_model, best_params, round(study.best_value, 4)
