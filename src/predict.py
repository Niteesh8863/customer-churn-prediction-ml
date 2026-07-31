"""Command-line prediction script for the trained churn model.

Example
-------
    python -m src.predict --gender Male --SeniorCitizen 0 --Partner Yes \\
        --Dependents No --tenure 12 --PhoneService Yes --MultipleLines No \\
        --InternetService "Fiber optic" --OnlineSecurity No --OnlineBackup Yes \\
        --DeviceProtection No --TechSupport No --StreamingTV Yes \\
        --StreamingMovies Yes --Contract Month-to-month --PaperlessBilling Yes \\
        --PaymentMethod "Electronic check" --MonthlyCharges 85.5 --TotalCharges 1020.5

Outputs the predicted class (Churn / No Churn) and the churn probability.
"""

from __future__ import annotations

import argparse
import sys

import joblib
import pandas as pd

from src.config import MODEL_FILE
from src.feature_engineering import add_engineered_features
from src.utils import get_logger

logger = get_logger(__name__)

YES_NO = ["Yes", "No"]
YES_NO_NA = ["Yes", "No", "No internet service"]
YES_NO_PHONE_NA = ["Yes", "No", "No phone service"]


def build_arg_parser() -> argparse.ArgumentParser:
    """Define the CLI schema mirroring the raw Telco churn feature columns."""
    parser = argparse.ArgumentParser(
        description="Predict customer churn from raw customer attributes."
    )
    parser.add_argument("--gender", required=True, choices=["Male", "Female"])
    parser.add_argument("--SeniorCitizen", required=True, type=int, choices=[0, 1])
    parser.add_argument("--Partner", required=True, choices=YES_NO)
    parser.add_argument("--Dependents", required=True, choices=YES_NO)
    parser.add_argument("--tenure", required=True, type=int, help="Tenure in months")
    parser.add_argument("--PhoneService", required=True, choices=YES_NO)
    parser.add_argument("--MultipleLines", required=True, choices=YES_NO_PHONE_NA)
    parser.add_argument(
        "--InternetService", required=True, choices=["DSL", "Fiber optic", "No"]
    )
    parser.add_argument("--OnlineSecurity", required=True, choices=YES_NO_NA)
    parser.add_argument("--OnlineBackup", required=True, choices=YES_NO_NA)
    parser.add_argument("--DeviceProtection", required=True, choices=YES_NO_NA)
    parser.add_argument("--TechSupport", required=True, choices=YES_NO_NA)
    parser.add_argument("--StreamingTV", required=True, choices=YES_NO_NA)
    parser.add_argument("--StreamingMovies", required=True, choices=YES_NO_NA)
    parser.add_argument(
        "--Contract", required=True, choices=["Month-to-month", "One year", "Two year"]
    )
    parser.add_argument("--PaperlessBilling", required=True, choices=YES_NO)
    parser.add_argument(
        "--PaymentMethod",
        required=True,
        choices=[
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    parser.add_argument("--MonthlyCharges", required=True, type=float)
    parser.add_argument("--TotalCharges", required=True, type=float)
    return parser


def build_input_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    """Turn parsed CLI args into a single-row, feature-engineered DataFrame."""
    raw_record = {
        "gender": args.gender,
        "SeniorCitizen": args.SeniorCitizen,
        "Partner": args.Partner,
        "Dependents": args.Dependents,
        "tenure": args.tenure,
        "PhoneService": args.PhoneService,
        "MultipleLines": args.MultipleLines,
        "InternetService": args.InternetService,
        "OnlineSecurity": args.OnlineSecurity,
        "OnlineBackup": args.OnlineBackup,
        "DeviceProtection": args.DeviceProtection,
        "TechSupport": args.TechSupport,
        "StreamingTV": args.StreamingTV,
        "StreamingMovies": args.StreamingMovies,
        "Contract": args.Contract,
        "PaperlessBilling": args.PaperlessBilling,
        "PaymentMethod": args.PaymentMethod,
        "MonthlyCharges": args.MonthlyCharges,
        "TotalCharges": args.TotalCharges,
    }
    df = pd.DataFrame([raw_record])
    return add_engineered_features(df)


def predict_churn(input_df: pd.DataFrame, model_path=MODEL_FILE) -> tuple[str, float]:
    """Load the saved pipeline and predict churn label + probability.

    Returns
    -------
    tuple[str, float]
        ``("Churn" | "No Churn", probability_of_churn)``
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run `python -m src.train` first."
        )

    pipeline = joblib.load(model_path)
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0, 1]
    label = "Churn" if prediction == 1 else "No Churn"
    return label, float(probability)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        input_df = build_input_dataframe(args)
        label, probability = predict_churn(input_df)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 - surface any unexpected failure clearly
        logger.error("Prediction failed: %s", exc)
        sys.exit(1)

    print("\n=== Churn Prediction ===")
    print(f"Prediction : {label}")
    print(f"Probability: {probability:.2%} chance of churn")


if __name__ == "__main__":
    main()
