"""Automated test suite for the Customer Churn Prediction ML pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure root workspace directory is in python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api import app
from src.config import MODEL_FILE
from src.feature_engineering import add_engineered_features
from src.predict import predict_churn
from src.preprocessing import clean_raw_data


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "customerID": "7590-VHVEG",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": "29.85",
                "Churn": "No",
            }
        ]
    )


def test_clean_raw_data(sample_raw_df: pd.DataFrame):
    cleaned = clean_raw_data(sample_raw_df)
    assert "customerID" not in cleaned.columns
    assert cleaned["Churn"].iloc[0] == 0
    assert cleaned["TotalCharges"].dtype in ("float64", "float32", "int64")


def test_feature_engineering(sample_raw_df: pd.DataFrame):
    cleaned = clean_raw_data(sample_raw_df)
    engineered = add_engineered_features(cleaned)
    expected_cols = [
        "TenureGroup",
        "ContractLengthCategory",
        "AvgMonthlySpend",
        "TotalServicesCount",
        "HasInternetService",
        "IsSeniorCitizen",
    ]
    for col in expected_cols:
        assert col in engineered.columns, f"Missing feature: {col}"

    assert engineered["IsSeniorCitizen"].iloc[0] == "No"
    assert engineered["ContractLengthCategory"].iloc[0] == "Short-term"


def test_predict_churn_func(sample_raw_df: pd.DataFrame):
    cleaned = clean_raw_data(sample_raw_df)
    engineered = add_engineered_features(cleaned)
    X = engineered.drop(columns=["Churn"])
    label, proba = predict_churn(X)
    assert label in ("Churn", "No Churn")
    assert 0.0 <= proba <= 1.0


def test_fastapi_endpoint(sample_raw_df: pd.DataFrame):
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }
    post_res = client.post("/predict", json=payload)
    assert post_res.status_code == 200
    data = post_res.json()
    assert "prediction" in data
    assert "churn_probability" in data
