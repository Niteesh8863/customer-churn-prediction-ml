"""TelePredict Pro Enterprise — AI Customer Churn Intelligence & Revenue Risk Platform.

A high-performance, enterprise-grade ML web application featuring:
  • Executive KPI Metrics Header & Financial Risk Calculator
  • Real-Time Single-Customer AI Scoring with Prescriptive Retention Actions
  • Batch CSV Processing & Downloadable Churn Risk Reports
  • Seamless Dual Theme System (Ultra-Dark Glassmorphism & Clean Executive Light)
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Ensure repository root is on sys.path
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
# Streamlit Page Configuration
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="TelePredict Pro | Executive ML Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "Dark"


# --------------------------------------------------------------------------- #
# Dynamic CSS Injection (Enterprise Dark & Light Themes)
# --------------------------------------------------------------------------- #
def inject_custom_css(theme: str) -> None:
    if theme == "Light":
        css = """
        <style>
        @import url('https://cdn.jsdelivr.net/gh/mishra-ankit/gilroy-free-webfont@main/gilroy-webfont.css');
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Gilroy', 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }

        .stApp {
            background: linear-gradient(145deg, #F8FAFC 0%, #F1F5F9 50%, #E2E8F0 100%) !important;
            color: #0F172A !important;
        }

        .stAppHeader { background: transparent !important; }

        /* Glassmorphic Cards Light */
        .enterprise-card {
            background: rgba(255, 255, 255, 0.92) !important;
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid #E2E8F0 !important;
            border-radius: 20px !important;
            padding: 28px !important;
            margin-bottom: 24px !important;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.05) !important;
        }

        .kpi-card {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 16px !important;
            padding: 18px 24px !important;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03) !important;
        }

        .kpi-title {
            color: #64748B !important;
            font-size: 0.85rem !important;
            font-weight: 700 !important;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .kpi-value {
            color: #0F172A !important;
            font-size: 1.8rem !important;
            font-weight: 800 !important;
            margin-top: 4px;
        }

        .hero-title {
            background: linear-gradient(90deg, #1E3A8A 0%, #2563EB 50%, #4F46E5 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3.1rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            margin-bottom: 0.2rem;
        }

        .hero-subtitle {
            color: #475569 !important;
            font-size: 1.15rem;
            margin-bottom: 2rem;
        }

        .badge-danger {
            background: #FEE2E2 !important;
            color: #991B1B !important;
            border: 1px solid #FCA5A5 !important;
            padding: 8px 18px;
            border-radius: 30px;
            font-weight: 800;
            font-size: 0.9rem;
        }

        .badge-success {
            background: #DCFCE7 !important;
            color: #166534 !important;
            border: 1px solid #86EFAC !important;
            padding: 8px 18px;
            border-radius: 30px;
            font-weight: 800;
            font-size: 0.9rem;
        }

        .stButton>button {
            background: linear-gradient(90deg, #2563EB 0%, #4F46E5 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 14px !important;
            padding: 14px 28px !important;
            font-weight: 700 !important;
            font-size: 1.05rem !important;
            box-shadow: 0 4px 16px rgba(37, 99, 235, 0.25) !important;
        }

        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(37, 99, 235, 0.4) !important;
        }

        hr { border-color: #E2E8F0 !important; }
        </style>
        """
    else:
        css = """
        <style>
        @import url('https://cdn.jsdelivr.net/gh/mishra-ankit/gilroy-free-webfont@main/gilroy-webfont.css');
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Gilroy', 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }

        .stApp {
            background: linear-gradient(135deg, #0B0F19 0%, #111827 50%, #0F172A 100%) !important;
            color: #F8FAFC !important;
        }

        .stAppHeader { background: transparent !important; }

        /* Glassmorphism Cards Dark */
        .enterprise-card {
            background: rgba(17, 24, 39, 0.75) !important;
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 20px !important;
            padding: 28px !important;
            margin-bottom: 24px !important;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.45) !important;
        }

        .kpi-card {
            background: rgba(15, 23, 42, 0.7) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 16px !important;
            padding: 18px 24px !important;
            text-align: center;
            backdrop-filter: blur(12px);
        }

        .kpi-title {
            color: #94A3B8 !important;
            font-size: 0.85rem !important;
            font-weight: 700 !important;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .kpi-value {
            color: #F8FAFC !important;
            font-size: 1.8rem !important;
            font-weight: 800 !important;
            margin-top: 4px;
        }

        .hero-title {
            background: linear-gradient(90deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3.1rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            margin-bottom: 0.2rem;
        }

        .hero-subtitle {
            color: #94A3B8 !important;
            font-size: 1.15rem;
            margin-bottom: 2rem;
        }

        .badge-danger {
            background: rgba(239, 68, 68, 0.2) !important;
            color: #FCA5A5 !important;
            border: 1px solid #EF4444 !important;
            padding: 8px 18px;
            border-radius: 30px;
            font-weight: 800;
            font-size: 0.9rem;
        }

        .badge-success {
            background: rgba(34, 197, 94, 0.2) !important;
            color: #86EFAC !important;
            border: 1px solid #22C55E !important;
            padding: 8px 18px;
            border-radius: 30px;
            font-weight: 800;
            font-size: 0.9rem;
        }

        .stButton>button {
            background: linear-gradient(90deg, #3B82F6 0%, #6366F1 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 14px !important;
            padding: 14px 28px !important;
            font-weight: 700 !important;
            font-size: 1.05rem !important;
            box-shadow: 0 4px 16px rgba(59, 130, 246, 0.4) !important;
        }

        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(59, 130, 246, 0.6) !important;
        }

        hr { border-color: rgba(255, 255, 255, 0.08) !important; }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        return None
    return joblib.load(MODEL_FILE)


# --------------------------------------------------------------------------- #
# Sidebar Parameter Form & Presets
# --------------------------------------------------------------------------- #
def sidebar_controls() -> tuple[dict, str]:
    st.sidebar.markdown("### 🎨 Theme Controller")
    theme_choice = st.sidebar.radio(
        "Theme Choice",
        ["🌙 Ultra Dark", "☀️ Executive Light"],
        index=0 if st.session_state["theme_mode"] == "Dark" else 1,
        label_visibility="collapsed",
    )
    st.session_state["theme_mode"] = "Dark" if "Dark" in theme_choice else "Light"

    st.sidebar.divider()
    st.sidebar.markdown("### ⚡ Parameter Profiles")

    preset = st.sidebar.selectbox(
        "Load Preset Profile",
        ["Custom Manual Entry", "🚨 High-Risk Profile Example", "✅ Loyal Account Example"],
    )

    if preset == "🚨 High-Risk Profile Example":
        def_gender, def_senior, def_partner, def_dep = "Female", 0, "No", "No"
        def_tenure, def_contract, def_payment = 2, "Month-to-month", "Electronic check"
        def_internet, def_sec, def_back, def_tech = "Fiber optic", "No", "No", "No"
        def_monthly, def_total = 95.0, 190.0
    elif preset == "✅ Loyal Account Example":
        def_gender, def_senior, def_partner, def_dep = "Male", 0, "Yes", "Yes"
        def_tenure, def_contract, def_payment = 60, "Two year", "Bank transfer (automatic)"
        def_internet, def_sec, def_back, def_tech = "DSL", "Yes", "Yes", "Yes"
        def_monthly, def_total = 65.0, 3900.0
    else:
        def_gender, def_senior, def_partner, def_dep = "Male", 0, "Yes", "No"
        def_tenure, def_contract, def_payment = 12, "Month-to-month", "Electronic check"
        def_internet, def_sec, def_back, def_tech = "Fiber optic", "No", "Yes", "No"
        def_monthly, def_total = 75.0, 900.0

    with st.sidebar.expander("👤 Customer Demographics", expanded=True):
        gender = st.selectbox("Gender", ["Male", "Female"], index=["Male", "Female"].index(def_gender))
        senior = st.selectbox("Senior Citizen", [0, 1], index=def_senior, format_func=lambda x: "Yes" if x else "No")
        partner = st.selectbox("Partner", ["Yes", "No"], index=["Yes", "No"].index(def_partner))
        dependents = st.selectbox("Dependents", ["Yes", "No"], index=["Yes", "No"].index(def_dep))
        tenure = st.slider("Tenure (Months)", 0, 72, def_tenure)

    with st.sidebar.expander("📡 Telecom Services", expanded=False):
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], index=["Fiber optic", "DSL", "No"].index(def_internet))
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(def_sec))
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], index=["Yes", "No", "No internet service"].index(def_back))
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(def_tech))
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    with st.sidebar.expander("💳 Account & Financials", expanded=True):
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(def_contract))
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(def_payment),
        )
        monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, def_monthly, step=1.0)
        total = st.number_input("Total Charges ($)", 0.0, 10000.0, def_total, step=10.0)

    params = {
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
    return params, preset


# --------------------------------------------------------------------------- #
# Main Application Dashboard
# --------------------------------------------------------------------------- #
def main() -> None:
    inject_custom_css(st.session_state["theme_mode"])

    # Hero Banner
    st.markdown("<h1 class='hero-title'>⚡ TelePredict Pro Enterprise</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='hero-subtitle'>Production Customer Retention Analytics, Financial Risk Assessment & Prescriptive AI Platform</p>",
        unsafe_allow_html=True,
    )

    # Executive KPI Summary Bar
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.markdown(
            "<div class='kpi-card'><div class='kpi-title'>Dataset Volume</div><div class='kpi-value'>7,043</div></div>",
            unsafe_allow_html=True,
        )
    with kpi_col2:
        st.markdown(
            "<div class='kpi-card'><div class='kpi-title'>Model ROC-AUC</div><div class='kpi-value'>0.8403</div></div>",
            unsafe_allow_html=True,
        )
    with kpi_col3:
        st.markdown(
            "<div class='kpi-card'><div class='kpi-title'>Churn Recall Rate</div><div class='kpi-value'>76.6%</div></div>",
            unsafe_allow_html=True,
        )
    with kpi_col4:
        st.markdown(
            "<div class='kpi-card'><div class='kpi-title'>Pipeline Architecture</div><div class='kpi-value'>SMOTE + LR</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    model = load_model()
    if model is None:
        st.error("⚠️ Model pipeline artifact not found. Please execute `python -m src.train` first.")
        return

    params, preset = sidebar_controls()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🎯 Single Customer Risk Scoring",
            "📁 Batch CSV Processing",
            "📈 Machine Learning Analytics",
            "⚙️ Technical Architecture",
        ]
    )

    # TAB 1: Single Scoring & Financial Risk
    with tab1:
        st.markdown("<div class='enterprise-card'>", unsafe_allow_html=True)
        st.subheader("Real-Time Churn Scoring & Financial Risk Calculator")
        st.caption("Adjust parameters in the left sidebar or select preset profiles to calculate real-time retention risk and financial exposure.")

        if st.button("🚀 Calculate Retention & Risk Assessment", type="primary"):
            input_df = pd.DataFrame([params])
            engineered_df = add_engineered_features(input_df)

            pred = model.predict(engineered_df)[0]
            prob = float(model.predict_proba(engineered_df)[0, 1])

            res_col1, res_col2, res_col3 = st.columns([1.2, 1, 1.1])

            with res_col1:
                if pred == 1:
                    st.markdown(
                        "<div style='margin-top:8px;'><span class='badge-danger'>🚨 HIGH CHURN EXPOSURE</span></div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("<h3 style='color:#EF4444; margin-top:12px;'>Customer At Risk</h3>", unsafe_allow_html=True)
                else:
                    st.markdown(
                        "<div style='margin-top:8px;'><span class='badge-success'>✅ LOW CHURN RISK</span></div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("<h3 style='color:#22C55E; margin-top:12px;'>Customer Retained</h3>", unsafe_allow_html=True)

            with res_col2:
                st.metric("Churn Probability", f"{prob:.1%}")
                st.progress(prob)

            with res_col3:
                acv = params["MonthlyCharges"] * 12
                st.metric("Annual Contract Value (ACV)", f"${acv:,.2f}")
                st.caption(f"Contract: **{params['Contract']}** | Payment: **{params['PaymentMethod']}**")

            st.divider()

            # Financial Risk Exposure Breakdown
            st.markdown("#### 💰 Financial Exposure & ROI Retention Analysis")
            fin_col1, fin_col2, fin_col3 = st.columns(3)
            with fin_col1:
                st.metric("3-Month Revenue Exposure", f"${params['MonthlyCharges'] * 3:,.2f}")
            with fin_col2:
                st.metric("Annual Revenue at Risk (ACV)", f"${params['MonthlyCharges'] * 12:,.2f}")
            with fin_col3:
                intervention_cost = 15.0 * 12
                net_savings = (params["MonthlyCharges"] * 12) - intervention_cost
                st.metric("Estimated Intervention Net ROI", f"${net_savings:,.2f}")

            st.divider()

            # Prescriptive Recommendations
            st.markdown("#### 💡 Executive Prescriptive Recommendations")
            if pred == 1:
                st.error(
                    "1. **Contract Conversion Incentive**: Offer a $15/month discount in exchange for a 1-Year or 2-Year contract commitment.\n"
                    "2. **Security & Tech Support Bundle**: Upgrade the account with complimentary Online Security & Tech Support for 3 months.\n"
                    "3. **Payment Method Migration**: Prompt customer to switch from Electronic Check to Automated Bank Transfer to reduce friction."
                )
            else:
                st.success(
                    "Customer stability is high. Recommended Action: Maintain standard engagement and evaluate cross-sell campaigns for streaming add-ons."
                )

        st.markdown("</div>", unsafe_allow_html=True)

    # TAB 2: Batch Processing
    with tab2:
        st.markdown("<div class='enterprise-card'>", unsafe_allow_html=True)
        st.subheader("📁 Batch Customer Churn Processing")
        st.caption("Upload a CSV file of customer records to process predictions and export a risk report.")

        uploaded_file = st.file_uploader("Upload Raw Customer CSV", type=["csv"])

        if uploaded_file is not None:
            batch_raw_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(batch_raw_df)}** records for batch evaluation.")

            if st.button("⚡ Process Batch Predictions", type="primary"):
                batch_clean_df = batch_raw_df.copy()
                if "TotalCharges" in batch_clean_df.columns:
                    batch_clean_df["TotalCharges"] = pd.to_numeric(batch_clean_df["TotalCharges"], errors="coerce").fillna(0)
                if "customerID" in batch_clean_df.columns:
                    batch_clean_df = batch_clean_df.drop(columns=["customerID"])
                if "Churn" in batch_clean_df.columns:
                    batch_clean_df = batch_clean_df.drop(columns=["Churn"])

                engineered_batch = add_engineered_features(batch_clean_df)

                batch_preds = model.predict(engineered_batch)
                batch_probs = model.predict_proba(engineered_batch)[:, 1]

                results_df = batch_raw_df.copy()
                results_df["Predicted_Churn"] = ["Churn" if p == 1 else "No Churn" for p in batch_preds]
                results_df["Churn_Probability"] = [round(float(p), 4) for p in batch_probs]
                results_df["Churn_Probability_Pct"] = [f"{p:.1%}" for p in batch_probs]

                st.subheader("Batch Scoring Summary")
                n_high_risk = (batch_preds == 1).sum()
                n_low_risk = (batch_preds == 0).sum()

                sum_col1, sum_col2, sum_col3 = st.columns(3)
                sum_col1.metric("Total Customers Scored", len(results_df))
                sum_col2.metric("High Churn Risk Customers", n_high_risk, delta=f"{n_high_risk / len(results_df):.1%}")
                sum_col3.metric("Low Churn Risk Customers", n_low_risk, delta=f"{n_low_risk / len(results_df):.1%}")

                st.dataframe(results_df.head(20), use_container_width=True)

                # CSV Export
                csv_buffer = io.StringIO()
                results_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Full Batch Risk Report (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="customer_churn_predictions_report.csv",
                    mime="text/csv",
                )
        else:
            st.info("Upload a CSV file formatted like `data/raw/Telco-Customer-Churn.csv` to run batch scoring.")
        st.markdown("</div>", unsafe_allow_html=True)

    # TAB 3: Model Analytics & Visualizations
    with tab3:
        st.subheader("📊 Model Performance Benchmarks & Diagnostic Plots")

        diag_col1, diag_col2 = st.columns(2)
        with diag_col1:
            if FEATURE_IMPORTANCE_FILE.exists():
                st.markdown("**Feature Importance Drivers (Random Forest)**")
                st.image(str(FEATURE_IMPORTANCE_FILE), width="stretch")
        with diag_col2:
            if MODEL_COMPARISON_FILE.exists():
                st.markdown("**Cross-Model ROC-AUC Benchmark**")
                st.image(str(MODEL_COMPARISON_FILE), width="stretch")

        st.divider()

        diag_col3, diag_col4 = st.columns(2)
        with diag_col3:
            if CONFUSION_MATRIX_FILE.exists():
                st.markdown("**Confusion Matrix (Test Set)**")
                st.image(str(CONFUSION_MATRIX_FILE), width="stretch")
        with diag_col4:
            if ROC_CURVE_FILE.exists():
                st.markdown("**ROC Curve (Test Set)**")
                st.image(str(ROC_CURVE_FILE), width="stretch")

    # TAB 4: Architecture
    with tab4:
        st.subheader("⚙️ Pipeline & Architectural Specifications")
        st.markdown(
            """
            #### End-to-End Pipeline Workflow:
            1. **Preprocessing Transformer**: Scikit-Learn `ColumnTransformer` (SimpleImputer, OneHotEncoder `drop='if_binary'`, StandardScaler).
            2. **SMOTE Oversampling**: Embedded via `imblearn.pipeline.Pipeline` to guarantee resampling only occurs inside training folds.
            3. **Hyperparameter Tuning**: 5-Fold Stratified Cross-Validation (`GridSearchCV`) targeting ROC-AUC.
            4. **Serialization**: Full inference pipeline serialized to `models/churn_model.pkl`.
            """
        )


if __name__ == "__main__":
    main()
