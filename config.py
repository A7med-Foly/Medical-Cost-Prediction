"""
Configuration file for the Medical Cost Prediction project.
Centralizes all hyperparameters, paths, and settings.
"""

import os

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
RAW_DIR    = os.path.join(DATA_DIR, "raw")
PROC_DIR   = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR   = os.path.join(BASE_DIR, "logs")

RAW_DATA_PATH       = os.path.join(RAW_DIR, "insurance.csv")
PREPROCESSOR_PATH   = os.path.join(MODELS_DIR, "preprocessor.pkl")
BEST_MODEL_PATH     = os.path.join(MODELS_DIR, "best_model.pkl")
RESULTS_PATH        = os.path.join(LOGS_DIR, "model_comparison.csv")

# ── Data settings ─────────────────────────────────────────────────────────────
TARGET_COLUMN   = "charges"
NUMERIC_FEATURES  = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
TEST_SIZE       = 0.20
RANDOM_STATE    = 42

# ── Model hyperparameters ─────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "LinearRegression": {
        "class": "sklearn.linear_model.LinearRegression",
        "params": {}
    },
    "PolynomialRegression": {
        "class": "sklearn.linear_model.LinearRegression",
        "params": {},
        "poly_degree": 2          # special flag handled in training
    },
    "GradientBoosting": {
        "class": "sklearn.ensemble.GradientBoostingRegressor",
        "params": {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 4,
            "random_state": RANDOM_STATE,
        }
    },
    "XGBoost": {
        "class": "xgboost.XGBRegressor",
        "params": {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 4,
            "random_state": RANDOM_STATE,
            "verbosity": 0,
        }
    },
    "RandomForest": {
        "class": "sklearn.ensemble.RandomForestRegressor",
        "params": {
            "n_estimators": 200,
            "max_depth": None,
            "random_state": RANDOM_STATE,
        }
    },
    "KNN": {
        "class": "sklearn.neighbors.KNeighborsRegressor",
        "params": {
            "n_neighbors": 5,
        }
    },
}
