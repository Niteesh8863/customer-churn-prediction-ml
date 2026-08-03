"""FastAPI REST service for customer churn prediction."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure repo root is on import path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import MODEL_FILE
from src.feature_engineering import add_engineered_features

app = FastAPI(
    title="Customer Churn Prediction API",
    description="REST API for predicting telecom customer churn probability.",
    version="1.0.0",
)


class CustomerData(BaseModel):
    gender: str = Field(..., json_schema_extra={"example": "Male"})
    SeniorCitizen: int = Field(..., json_schema_extra={"example": 0})
    Partner: str = Field(..., json_schema_extra={"example": "Yes"})
    Dependents: str = Field(..., json_schema_extra={"example": "No"})
    tenure: int = Field(..., json_schema_extra={"example": 12})
    PhoneService: str = Field(..., json_schema_extra={"example": "Yes"})
    MultipleLines: str = Field(..., json_schema_extra={"example": "No"})
    InternetService: str = Field(..., json_schema_extra={"example": "Fiber optic"})
    OnlineSecurity: str = Field(..., json_schema_extra={"example": "No"})
    OnlineBackup: str = Field(..., json_schema_extra={"example": "Yes"})
    DeviceProtection: str = Field(..., json_schema_extra={"example": "No"})
    TechSupport: str = Field(..., json_schema_extra={"example": "No"})
    StreamingTV: str = Field(..., json_schema_extra={"example": "Yes"})
    StreamingMovies: str = Field(..., json_schema_extra={"example": "Yes"})
    Contract: str = Field(..., json_schema_extra={"example": "Month-to-month"})
    PaperlessBilling: str = Field(..., json_schema_extra={"example": "Yes"})
    PaymentMethod: str = Field(..., json_schema_extra={"example": "Electronic check"})
    MonthlyCharges: float = Field(..., json_schema_extra={"example": 85.5})
    TotalCharges: float = Field(..., json_schema_extra={"example": 1020.5})


class PredictionResponse(BaseModel):
    prediction: str
    churn_probability: float
    churn_probability_percentage: str


@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Customer Churn Prediction API",
        "endpoints": {"predict": "/predict", "docs": "/docs"},
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData):
    if not MODEL_FILE.exists():
        raise HTTPException(
            status_code=500,
            detail="Trained model artifact not found. Please train model first.",
        )

    model = joblib.load(MODEL_FILE)
    df = pd.DataFrame([customer.model_dump()])
    engineered_df = add_engineered_features(df)

    pred = model.predict(engineered_df)[0]
    proba = float(model.predict_proba(engineered_df)[0, 1])
    label = "Churn" if pred == 1 else "No Churn"

    return PredictionResponse(
        prediction=label,
        churn_probability=round(proba, 4),
        churn_probability_percentage=f"{proba:.2%}",
    )
