"""Streamlit Web Application: TelePredict Pro — AI Customer Churn Intelligence."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Allow running via `streamlit run app/streamlit_app.py` from repo root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (
    CONFUSION_MATRIX_FILE,
    CORRELATION_HEATMAP_FILE,
    FEATURE_IMPORTANCE_FILE,
    MODEL_COMPARISON_FILE,
    MODEL_FILE,
    ROC_CURVE_FILE,
)
from src.feature_engineering import add_engineered_features

# --------------------------------------------------------------------------- #
# Streamlit Page Configuration & Modern Theme Styling
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="TelePredict Pro | AI Churn Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Modern CSS Injection
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
        color: #F8FAFC;
    }

    .stAppHeader {
        background: transparent !important;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }

    .hero-title {
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.1rem;
        margin-bottom: 1.8rem;
    }

    .metric-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .badge-high-risk {
        background: rgba(239, 68, 68, 0.2);
        color: #FCA5A5;
        border: 1px solid #EF4444;
    }

    .badge-low-risk {
        background: rgba(34, 197, 94, 0.2);
        color: #86EFAC;
        border: 1px solid #22C55E;
    }

    .stat-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(15, 23, 42, 0.6);
        border-radius: 12px;
        padding: 16px 20px;
        margin-top: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stat-label {
        color: #94A3B8;
        font-size: 0.9rem;
        font-weight: 500;
    }

    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #F8FAFC;
    }

    /* Style Streamlit Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #3B82F6 0%, #6366F1 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 28px;
        font-weight: 700;
        font-size: 1rem;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px rgba(59, 130, 246, 0.4);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6);
    }

    /* Custom Divider */
    hr {
        border-color: rgba(255, 255, 255, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        return None
    return joblib.load(MODEL_FILE)


def sidebar_inputs() -> dict:
    st.sidebar.markdown(
        "<h2 style='color:#38BDF8; font-size:1.4rem;'>⚙️ Customer Parameters</h2>",
        unsafe_allow_html=True,
    )

    with st.sidebar.expander("👤 Demographics", expanded=True):
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (Months)", 0, 72, 12, help="Months with the company")

    with st.sidebar.expander("📡 Telecom Services", expanded=False):
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    with st.sidebar.expander("💳 Account & Billing", expanded=True):
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 75.0, step=1.0)
        total = st.number_input(
            "Total Charges ($)", 0.0, 10000.0, float(monthly) * tenure, step=10.0
        )

    return {
        "gender": gender,
        "SeniorCitizen": senior,
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
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }


def main() -> None:
    st.markdown("<h1 class='hero-title'>🔮 TelePredict Pro</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='hero-subtitle'>Production Machine Learning Intelligence for Customer Retention & Churn Prevention</p>",
        unsafe_allow_html=True,
    )

    model = load_model()
    if model is None:
        st.error("⚠️ No trained model pipeline found. Run `python -m src.train` to generate `models/churn_model.pkl`.")
        return

    inputs = sidebar_inputs()

    tab1, tab2, tab3 = st.tabs(["🎯 Real-Time Prediction", "📊 Model Insights & Analytics", "📘 Model Architecture"])

    with tab1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Predict Customer Retention Risk")
        st.caption("Configure customer profile parameters in the left sidebar and trigger real-time AI scoring below.")

        if st.button("🚀 Calculate Churn Probability", type="primary"):
            input_df = pd.DataFrame([inputs])
            input_df = add_engineered_features(input_df)

            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0, 1]

            res_col1, res_col2, res_col3 = st.columns([1.2, 1, 1])

            with res_col1:
                if pred == 1:
                    st.markdown(
                        "<div style='margin-top:10px;'><span class='metric-badge badge-high-risk'>🚨 HIGH CHURN RISK</span></div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("<h3 style='color:#FCA5A5; margin-top:10px;'>At Risk of Leaving</h3>", unsafe_allow_html=True)
                else:
                    st.markdown(
                        "<div style='margin-top:10px;'><span class='metric-badge badge-low-risk'>✅ RETAINED / LOW RISK</span></div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("<h3 style='color:#86EFAC; margin-top:10px;'>Likely to Stay</h3>", unsafe_allow_html=True)

            with res_col2:
                st.metric("Probability of Churn", f"{prob:.1%}")
                st.progress(float(prob))

            with res_col3:
                st.metric("Estimated Monthly Value", f"${inputs['MonthlyCharges']:.2f}")
                st.caption(f"Contract: **{inputs['Contract']}**")

            st.divider()

            # Retention Recommendations
            st.markdown("#### 💡 Retention Action Recommendations")
            if pred == 1:
                st.warning(
                    "• **Offer Long-Term Contract Discount**: High-risk churn is correlated with month-to-month contracts.\n"
                    "• **Bundle Tech Support & Security**: Adding security and tech support services decreases churn probability significantly.\n"
                    "• **Switch Payment Method**: Customers on Electronic Check churn at higher rates than automated bank transfer/credit card."
                )
            else:
                st.success("Customer profile is stable. Maintain engagement and evaluate cross-selling opportunities for add-on services.")

        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.subheader("📈 Model Analytics & Feature Drivers")
        insight_col1, insight_col2 = st.columns(2)

        with insight_col1:
            if FEATURE_IMPORTANCE_FILE.exists():
                st.markdown("**Top Churn Drivers (Feature Importance)**")
                st.image(str(FEATURE_IMPORTANCE_FILE), width="stretch")

        with insight_col2:
            if MODEL_COMPARISON_FILE.exists():
                st.markdown("**Cross-Model ROC-AUC Performance**")
                st.image(str(MODEL_COMPARISON_FILE), width="stretch")

        st.divider()
        diag_col1, diag_col2 = st.columns(2)
        with diag_col1:
            if CONFUSION_MATRIX_FILE.exists():
                st.markdown("**Test Set Confusion Matrix**")
                st.image(str(CONFUSION_MATRIX_FILE), width="stretch")
        with diag_col2:
            if ROC_CURVE_FILE.exists():
                st.markdown("**Test Set ROC Curve**")
                st.image(str(ROC_CURVE_FILE), width="stretch")

    with tab3:
        st.subheader("⚙️ End-to-End Pipeline Architecture")
        st.markdown(
            """
            - **Preprocessing Pipeline**: Imputer -> OneHotEncoder (`drop='if_binary'`) -> StandardScaler
            - **Class Imbalance Handling**: **SMOTE** oversampling applied exclusively inside CV folds to eliminate data leakage.
            - **Hyperparameter Optimization**: `GridSearchCV` evaluated using Stratified 5-Fold Cross Validation (`roc_auc`).
            - **Model Serving**: Serialized single artifact pipeline (`churn_model.pkl`) containing preprocessing, transformation, and classifier.
            """
        )


if __name__ == "__main__":
    main()
