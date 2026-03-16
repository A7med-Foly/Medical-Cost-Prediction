"""
src/model_evaluation.py
Computes and displays regression metrics, and selects the best model.
"""

import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


# ── Single model metrics ───────────────────────────────────────────────────────

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test,
    poly=None,
    model_name: str = "",
) -> dict:
    """
    Return a dict of metrics for one model.
    poly: optional PolynomialFeatures transformer (Polynomial Regression only).
    """
    if poly is not None:
        X_test = poly.transform(X_test)

    y_pred   = model.predict(X_test)
    r2       = r2_score(y_test, y_pred)
    mse      = mean_squared_error(y_test, y_pred)
    rmse     = np.sqrt(mse)
    mae      = mean_absolute_error(y_test, y_pred)

    metrics = {
        "Model":    model_name or type(model).__name__,
        "R2":       round(r2,   4),
        "MSE":      round(mse,  2),
        "RMSE":     round(rmse, 2),
        "MAE":      round(mae,  2),
    }
    return metrics


# ── Evaluate all models ────────────────────────────────────────────────────────

def evaluate_all_models(
    trained_models: dict,   # { name: (model, poly) }
    X_test: np.ndarray,
    y_test,
) -> pd.DataFrame:
    """
    Run evaluate_model for every entry, collect into a DataFrame sorted by R2.
    """
    rows = []
    for name, (model, poly) in trained_models.items():
        metrics = evaluate_model(model, X_test, y_test, poly, model_name=name)
        rows.append(metrics)
        logger.info(
            "%-22s | R2=%.4f | RMSE=%.2f | MAE=%.2f",
            name, metrics["R2"], metrics["RMSE"], metrics["MAE"],
        )

    results_df = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    return results_df


# ── Best model selection ───────────────────────────────────────────────────────

def select_best_model(
    trained_models: dict,
    X_test: np.ndarray,
    y_test,
) -> tuple:
    """
    Return (best_name, best_model, best_poly) based on highest R2 score.
    """
    best_name  = None
    best_r2    = -np.inf
    best_model = None
    best_poly  = None

    for name, (model, poly) in trained_models.items():
        metrics = evaluate_model(model, X_test, y_test, poly)
        if metrics["R2"] > best_r2:
            best_r2    = metrics["R2"]
            best_name  = name
            best_model = model
            best_poly  = poly

    logger.info("Best model: %s (R2=%.4f)", best_name, best_r2)
    return best_name, best_model, best_poly


# ── Persistence ────────────────────────────────────────────────────────────────

def save_results(results_df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    results_df.to_csv(path, index=False)
    logger.info("Comparison results saved to '%s'.", path)


def print_results(results_df: pd.DataFrame) -> None:
    """Pretty-print the comparison table to stdout."""
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON RESULTS (sorted by R²)")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print("=" * 70 + "\n")
