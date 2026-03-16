"""
src/predict.py
Inference helpers — load saved artifacts and predict on new data.
"""

import logging

import numpy as np
import pandas as pd

from src.data_preprocessing import load_preprocessor
from src.model_training import load_model

logger = logging.getLogger(__name__)


def load_artifacts(preprocessor_path: str, model_path: str) -> tuple:
    """Load and return (preprocessor, model, poly)."""
    preprocessor = load_preprocessor(preprocessor_path)
    model, poly   = load_model(model_path)
    return preprocessor, model, poly


def predict(
    input_data: pd.DataFrame | dict,
    preprocessor,
    model,
    poly=None,
) -> np.ndarray:
    """
    Run end-to-end prediction on raw input.

    Parameters
    ----------
    input_data : pd.DataFrame or dict
        Raw feature values (age, sex, bmi, children, smoker, region).
    preprocessor : fitted ColumnTransformer
    model : fitted sklearn/XGBoost model
    poly : optional PolynomialFeatures transformer

    Returns
    -------
    np.ndarray of predicted charges
    """
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])

    X = preprocessor.transform(input_data)

    if poly is not None:
        X = poly.transform(X)

    predictions = model.predict(X)
    return predictions


def predict_from_paths(
    input_data: pd.DataFrame | dict,
    preprocessor_path: str,
    model_path: str,
) -> np.ndarray:
    """Convenience wrapper: load artifacts then predict."""
    preprocessor, model, poly = load_artifacts(preprocessor_path, model_path)
    return predict(input_data, preprocessor, model, poly)
