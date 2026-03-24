# src/data_loader.py
# ─────────────────────────────────────────────────────────────
# Loads, cleans, and splits the feature dataset.
# All split dates and feature drops come from config.py.
# ─────────────────────────────────────────────────────────────

import pandas as pd
from src.config import DATA_PATH, VAL_START, TEST_START, FEATURES_TO_DROP, TARGET


def load_data() -> pd.DataFrame:
    """
    Load raw feature CSV, parse timestamps, set index.
    Returns the full cleaned DataFrame.
    """
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")
    df = df.drop(columns=[c for c in FEATURES_TO_DROP if c in df.columns])
    return df


def split_data(df: pd.DataFrame) -> tuple:
    """
    Temporally split DataFrame into train, validation, and test sets.
    
    Returns:
        train_df : Jan 2019 – Dec 2024
        val_df   : Jan 2025 – Jun 2025
        test_df  : Jul 2025 – Mar 2026
    """
    train_df = df[df.index < VAL_START]
    val_df   = df[(df.index >= VAL_START) & (df.index < TEST_START)]
    test_df  = df[df.index >= TEST_START]

    return train_df, val_df, test_df


def get_X_y(df: pd.DataFrame) -> tuple:
    """
    Separate feature matrix X and target vector y from a DataFrame.
    
    Returns:
        X : DataFrame of all features (target excluded)
        y : Series of target variable (price)
    """
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    return X, y



