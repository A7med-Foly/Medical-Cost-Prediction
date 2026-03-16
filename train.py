"""
train.py
─────────────────────────────────────────────────────────────────────────────
End-to-end training pipeline for Medical Cost Prediction.

Usage
-----
    python train.py

What it does
------------
  1. Loads and cleans insurance.csv
  2. Splits into train / test sets
  3. Fits and saves the preprocessor (MinMaxScaler + OneHotEncoder)
  4. Trains all configured models (Linear, Gradient Boosting, XGBoost, ...)
  5. Evaluates every model and prints a comparison table
  6. Saves the best model as models/best_model.pkl
  7. Writes results to logs/model_comparison.csv
"""

import os
import sys

# Allow imports from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    BEST_MODEL_PATH,
    CATEGORICAL_FEATURES,
    MODELS_DIR,
    MODEL_CONFIGS,
    NUMERIC_FEATURES,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    RAW_DATA_PATH,
    RESULTS_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data_preprocessing import run_preprocessing_pipeline
from src.logger import setup_logger
from src.model_evaluation import (
    evaluate_all_models,
    print_results,
    save_results,
    select_best_model,
)
from src.model_training import save_model, train_all_models


def main() -> None:
    setup_logger(log_dir=os.path.dirname(RESULTS_PATH))

    import logging
    logger = logging.getLogger(__name__)
    logger.info("═" * 60)
    logger.info("  MEDICAL COST PREDICTION — TRAINING PIPELINE")
    logger.info("═" * 60)

    # ── Step 1: Preprocessing ──────────────────────────────────────────────
    logger.info("Step 1/4 — Loading & preprocessing data …")
    data = run_preprocessing_pipeline(
        raw_path=RAW_DATA_PATH,
        preprocessor_save_path=PREPROCESSOR_PATH,
        target=TARGET_COLUMN,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    X_train = data["X_train"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_test  = data["y_test"]

    # ── Step 2: Training ────────────────────────────────────────────────────
    logger.info("Step 2/4 — Training %d models …", len(MODEL_CONFIGS))
    trained_models = train_all_models(
        model_configs=MODEL_CONFIGS,
        X_train=X_train,
        y_train=y_train,
        models_dir=MODELS_DIR,
    )

    # ── Step 3: Evaluation ──────────────────────────────────────────────────
    logger.info("Step 3/4 — Evaluating models …")
    results_df = evaluate_all_models(trained_models, X_test, y_test)
    print_results(results_df)
    save_results(results_df, RESULTS_PATH)

    # ── Step 4: Save best model ─────────────────────────────────────────────
    logger.info("Step 4/4 — Selecting and saving best model …")
    best_name, best_model, best_poly = select_best_model(
        trained_models, X_test, y_test
    )
    save_model(best_model, BEST_MODEL_PATH, best_poly)
    logger.info("✓ Best model '%s' saved to '%s'.", best_name, BEST_MODEL_PATH)
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
