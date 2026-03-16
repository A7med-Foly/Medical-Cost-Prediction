"""
predict_cli.py
─────────────────────────────────────────────────────────────────────────────
Command-line interface for predicting medical insurance charges.

Usage examples
--------------
Single prediction (interactive defaults):
    python predict_cli.py

Pass values directly as arguments:
    python predict_cli.py \
        --age 35 --sex male --bmi 28.5 \
        --children 1 --smoker no --region southwest

Batch prediction from a CSV file:
    python predict_cli.py --csv path/to/new_patients.csv
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import BEST_MODEL_PATH, PREPROCESSOR_PATH
from src.logger import setup_logger
from src.predict import predict_from_paths


# ── CLI parsing ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict medical insurance charges."
    )
    parser.add_argument("--age",      type=int,   help="Age of the patient (18–64)")
    parser.add_argument("--sex",      type=str,   choices=["male", "female"],
                        help="Sex of the patient")
    parser.add_argument("--bmi",      type=float, help="Body mass index")
    parser.add_argument("--children", type=int,   default=0,
                        help="Number of dependent children (0–5)")
    parser.add_argument("--smoker",   type=str,   choices=["yes", "no"],
                        help="Smoker status")
    parser.add_argument("--region",   type=str,
                        choices=["southwest", "southeast", "northwest", "northeast"],
                        help="Residential region")
    parser.add_argument("--csv",      type=str,
                        help="Path to a CSV file with multiple patients")
    parser.add_argument("--model",    type=str,   default=BEST_MODEL_PATH,
                        help="Path to a saved model .pkl (default: best_model.pkl)")
    parser.add_argument("--preprocessor", type=str, default=PREPROCESSOR_PATH,
                        help="Path to preprocessor .pkl")
    return parser.parse_args()


def interactive_input() -> dict:
    """Prompt the user for patient info when no args provided."""
    print("\n── Medical Cost Prediction ──")
    print("Enter patient details (press Enter to accept defaults):\n")

    age      = int(input("  Age [30]: ") or 30)
    sex      = input("  Sex (male/female) [male]: ").strip() or "male"
    bmi      = float(input("  BMI [27.5]: ") or 27.5)
    children = int(input("  Number of children [0]: ") or 0)
    smoker   = input("  Smoker (yes/no) [no]: ").strip() or "no"
    region   = input(
        "  Region (southwest/southeast/northwest/northeast) [southeast]: "
    ).strip() or "southeast"

    return {
        "age": age, "sex": sex, "bmi": bmi,
        "children": children, "smoker": smoker, "region": region,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    setup_logger()
    args = parse_args()

    if args.csv:
        # ── Batch mode ────────────────────────────────────────────────────
        df = pd.read_csv(args.csv)
        preds = predict_from_paths(df, args.preprocessor, args.model)
        df["predicted_charges"] = preds.round(2)
        print("\nBatch Predictions:")
        print(df.to_string(index=False))

    elif args.age is not None:
        # ── Single patient from CLI flags ─────────────────────────────────
        patient = {
            "age":      args.age,
            "sex":      args.sex,
            "bmi":      args.bmi,
            "children": args.children,
            "smoker":   args.smoker,
            "region":   args.region,
        }
        pred = predict_from_paths(patient, args.preprocessor, args.model)
        _print_prediction(patient, pred[0])

    else:
        # ── Interactive mode ──────────────────────────────────────────────
        patient = interactive_input()
        pred = predict_from_paths(patient, args.preprocessor, args.model)
        _print_prediction(patient, pred[0])


def _print_prediction(patient: dict, charge: float) -> None:
    print("\n" + "─" * 45)
    print("  PATIENT DETAILS")
    print("─" * 45)
    for k, v in patient.items():
        print(f"  {k:<12}: {v}")
    print("─" * 45)
    print(f"  Predicted Charges: ${charge:,.2f}")
    print("─" * 45 + "\n")


if __name__ == "__main__":
    main()
