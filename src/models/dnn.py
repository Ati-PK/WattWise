# src/models/dnn.py
# ─────────────────────────────────────────────────────────────
# Shallow Deep Neural Network for day-ahead electricity price
# forecasting using TensorFlow/Keras.
#
# Architecture:
#   Input → Dense(128, ReLU) → Dense(64, ReLU) → Dense(32, ReLU)
#   → Dense(1, Linear)
#
# Key design decisions:
#   - StandardScaler on X (same as linear models)
#   - StandardScaler on y (stabilises gradient descent)
#   - Early stopping to prevent overfitting
#   - Sample weights via class_weight equivalent
# ─────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow              import keras
from tensorflow.keras        import layers, callbacks
from sklearn.preprocessing   import StandardScaler


def build_dnn(input_dim: int) -> keras.Model:
    """
    Build shallow DNN architecture.

    Args:
        input_dim: number of input features

    Returns:
        Compiled Keras model
    """
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    model = keras.Sequential([
        # Input layer
        keras.Input(shape=(input_dim,)),

        # Hidden layer 1
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        # Hidden layer 2
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        # Hidden layer 3
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.1),

        # Output layer — linear activation for regression
        layers.Dense(1, activation="linear"),
    ])

    model.compile(
        optimizer = keras.optimizers.legacy.Adam(learning_rate=1e-3),
        loss      = "mae",       # optimise directly for MAE
        metrics   = ["mae"],
    )

    return model


def fit_dnn(
    X_train   : pd.DataFrame,
    y_train   : pd.Series,
    w_train   : pd.Series,
    X_val     : pd.DataFrame,
    y_val     : pd.Series,
    epochs    : int = 200,
    batch_size: int = 256,
) -> tuple:
    """
    Fit shallow DNN with early stopping.

    Returns:
        (model, scaler_X, scaler_y, history, y_mean, y_std)
    """
    # ── Scale features ────────────────────────────────────────
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled   = scaler_X.transform(X_val)

    # ── Scale target ──────────────────────────────────────────
    y_mean = float(y_train.mean())
    y_std  = float(y_train.std())
    y_train_scaled = (y_train.values - y_mean) / y_std
    y_val_scaled   = (y_val.values   - y_mean) / y_std

    # ── Normalise sample weights to sum to 1 ──────────────────
    w_normalised = w_train.values / w_train.values.sum()

    # ── Build model ───────────────────────────────────────────
    model = build_dnn(input_dim=X_train.shape[1])

    # ── Callbacks ─────────────────────────────────────────────
    early_stop = callbacks.EarlyStopping(
        monitor              = "val_loss",
        patience             = 20,        # wait 20 epochs before stopping
        restore_best_weights = True,      # revert to best epoch
        verbose              = 1,
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor  = "val_loss",
        factor   = 0.5,         # halve learning rate when stuck
        patience = 10,          # wait 10 epochs before reducing
        min_lr   = 1e-6,
        verbose  = 1,
    )

    # ── Train ─────────────────────────────────────────────────
    history = model.fit(
        X_train_scaled, y_train_scaled,
        validation_data = (X_val_scaled, y_val_scaled),
        epochs          = epochs,
        batch_size      = batch_size,
        sample_weight   = w_normalised,
        callbacks       = [early_stop, reduce_lr],
        verbose         = 1,
    )

    return model, scaler_X, y_mean, y_std, history


def predict_dnn(
    model    : keras.Model,
    scaler_X : StandardScaler,
    y_mean   : float,
    y_std    : float,
    X        : pd.DataFrame,
) -> np.ndarray:
    """
    Generate predictions and inverse-transform to EUR/MWh.

    Args:
        model    : fitted Keras model
        scaler_X : fitted StandardScaler for features
        y_mean   : mean of training target
        y_std    : std of training target
        X        : feature matrix to predict on

    Returns:
        predictions in EUR/MWh
    """
    X_scaled      = scaler_X.transform(X)
    y_pred_scaled = model.predict(X_scaled, verbose=0).flatten()
    return y_pred_scaled * y_std + y_mean
