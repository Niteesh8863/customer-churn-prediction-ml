"""Streamlit web app for interactive customer churn prediction.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Allow running via `streamlit run app/streamlit_app.py` from the repo root.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import FEATURE_IMPORTANCE_FILE, MODEL_FILE
from src.evaluate import get_feature_importance
from src.feature_engineering import add_engineered_features

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")


@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        return None
    return joblib.load(MODEL_FILE)


def sidebar_inputs() -> dict:
    st.sidebar.header("Customer Attributes")

    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
    partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)

    st.sidebar.subheader("Services")
    phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    st.sidebar.subheader("Account")
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.sidebar.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=0.5)
    total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0, float(monthly_charges) * tenure, step=1.0)

    return {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }


def main() -> None:
    st.title("📉 Customer Churn Prediction")
    st.write(
        "Predict the likelihood that a telecom customer will churn, using a "
        "model trained on the IBM Telco Customer Churn dataset."
    )

    model = load_model()
    if model is None:
        st.error(
            "No trained model found. Run `python -m src.train` first to train "
            "and save `models/churn_model.pkl`."
        )
        return

    inputs = sidebar_inputs()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Prediction")
        if st.button("Predict Churn", type="primary"):
            input_df = pd.DataFrame([inputs])
            input_df = add_engineered_features(input_df)

            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0, 1]

            label = "⚠️ Likely to Churn" if prediction == 1 else "✅ Likely to Stay"
            st.metric("Prediction", label)
            st.metric("Churn Probability", f"{probability:.1%}")
            st.progress(min(max(probability, 0.0), 1.0))

    with col2:
        st.subheader("Feature Importance (Random Forest)")
        if FEATURE_IMPORTANCE_FILE.exists():
            st.image(str(FEATURE_IMPORTANCE_FILE))
        else:
            st.info("Feature importance plot not found. Run training to generate it.")

    st.divider()
    st.caption(
        "Model: scikit-learn pipeline (preprocessing + SMOTE + best tuned "
        "classifier), selected automatically by ROC-AUC. See the project "
        "README for full methodology."
    )


if __name__ == "__main__":
    main()
