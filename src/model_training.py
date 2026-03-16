"""
src/model_training.py
Handles model instantiation, training, saving, and loading.
"""

import importlib
import logging
import os
import pickle

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)


# ── Model instantiation ────────────────────────────────────────────────────────

def instantiate_model(class_path: str, params: dict):
    """
    Dynamically import a class and return an instance.
    e.g. class_path = 'sklearn.ensemble.GradientBoostingRegressor'
    """
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**params)


# ── Single model training ──────────────────────────────────────────────────────

def train_model(
    model,
    X_train: np.ndarray,
    y_train,
    poly_degree: int | None = None,
) -> tuple:
    """
    Fit a model on training data.
    If poly_degree is set, first apply PolynomialFeatures and return
    (fitted_model, poly_transformer); otherwise return (fitted_model, None).
    """
    poly = None
    if poly_degree is not None:
        poly = PolynomialFeatures(degree=poly_degree)
        X_train = poly.fit_transform(X_train)
        logger.info("Applied PolynomialFeatures(degree=%d).", poly_degree)

    model.fit(X_train, y_train)
    logger.info("Trained %s.", type(model).__name__)
    return model, poly


# ── Save / load ────────────────────────────────────────────────────────────────

def save_model(model, path: str, poly=None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"model": model, "poly": poly}
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    logger.info("Model saved to '%s'.", path)


def load_model(path: str) -> tuple:
    """Returns (model, poly) — poly is None unless Polynomial Regression."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    model = payload["model"]
    poly  = payload.get("poly")
    logger.info("Model loaded from '%s'.", path)
    return model, poly


# ── Train all models ───────────────────────────────────────────────────────────

def train_all_models(
    model_configs: dict,
    X_train: np.ndarray,
    y_train,
    models_dir: str,
) -> dict:
    """
    Train every model defined in model_configs, save each one, and return
    a dict of { name: (model, poly) }.
    """
    trained = {}
    for name, cfg in model_configs.items():
        logger.info("── Training %s ──", name)
        model = instantiate_model(cfg["class"], cfg.get("params", {}))
        poly_degree = cfg.get("poly_degree")
        fitted_model, poly = train_model(model, X_train, y_train, poly_degree)

        model_path = os.path.join(models_dir, f"{name}.pkl")
        save_model(fitted_model, model_path, poly)

        trained[name] = (fitted_model, poly)

    logger.info("All %d models trained and saved.", len(trained))
    return trained
