"""
src/data_preprocessing.py
Handles data loading, cleaning, splitting, and feature transformation.
"""

import os
import pickle
import logging

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

logger = logging.getLogger(__name__)


# ── Loading ────────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Read the raw CSV and return a DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file not found at '{path}'.\n"
            "Download it from: https://www.kaggle.com/datasets/mirichoi0218/insurance\n"
            "and save it to data/raw/insurance.csv"
        )
    df = pd.read_csv(path)
    logger.info("Loaded %d rows × %d columns from '%s'", *df.shape, path)
    return df


# ── Cleaning ───────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
      - Drop exact duplicate rows.
      - Assert no nulls remain (the insurance dataset is clean by default).
    """
    original_len = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dropped = original_len - len(df)
    if dropped:
        logger.info("Dropped %d duplicate rows.", dropped)

    null_counts = df.isnull().sum()
    if null_counts.any():
        logger.warning("Null values found:\n%s", null_counts[null_counts > 0])
    else:
        logger.info("No null values detected.")

    return df


# ── Splitting ──────────────────────────────────────────────────────────────────

def split_features_target(
    df: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(
        "Train: %d samples | Test: %d samples", len(X_train), len(X_test)
    )
    return X_train, X_test, y_train, y_test


# ── Preprocessing ──────────────────────────────────────────────────────────────

def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> make_column_transformer:
    """
    Returns an *unfitted* ColumnTransformer that:
      - MinMaxScales numeric columns.
      - OneHotEncodes categorical columns.
    """
    preprocessor = make_column_transformer(
        (MinMaxScaler(), numeric_features),
        (OneHotEncoder(handle_unknown="ignore"), categorical_features),
    )
    return preprocessor


def fit_transform_data(
    preprocessor,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit on train, transform both train and test."""
    preprocessor.fit(X_train)
    X_train_t = preprocessor.transform(X_train)
    X_test_t  = preprocessor.transform(X_test)
    logger.info(
        "Preprocessed shapes — train: %s | test: %s",
        X_train_t.shape, X_test_t.shape,
    )
    return X_train_t, X_test_t


def save_preprocessor(preprocessor, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(preprocessor, f)
    logger.info("Preprocessor saved to '%s'.", path)


def load_preprocessor(path: str):
    with open(path, "rb") as f:
        preprocessor = pickle.load(f)
    logger.info("Preprocessor loaded from '%s'.", path)
    return preprocessor


# ── Convenience pipeline ───────────────────────────────────────────────────────

def run_preprocessing_pipeline(
    raw_path: str,
    preprocessor_save_path: str,
    target: str,
    numeric_features: list[str],
    categorical_features: list[str],
    test_size: float = 0.20,
    random_state: int = 42,
) -> dict:
    """
    End-to-end helper: load → clean → split → preprocess.
    Returns a dict with all train/test arrays and the fitted preprocessor.
    """
    df = load_data(raw_path)
    df = clean_data(df)

    X, y = split_features_target(df, target)
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train_t, X_test_t = fit_transform_data(preprocessor, X_train, X_test)

    save_preprocessor(preprocessor, preprocessor_save_path)

    return {
        "X_train": X_train_t,
        "X_test":  X_test_t,
        "y_train": y_train,
        "y_test":  y_test,
        "preprocessor": preprocessor,
    }
