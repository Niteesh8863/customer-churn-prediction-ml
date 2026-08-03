"""Streamlit Web Application: TelePredict Pro — AI Customer Churn Intelligence with Dual Theme Mode."""

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
# Streamlit Page Configuration
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="TelePredict Pro | AI Churn Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Theme Toggle State & Dynamic CSS Injection
# --------------------------------------------------------------------------- #
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "Dark"

def inject_custom_css(theme: str) -> None:
    if theme == "Light":
        css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        .stApp {
            background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 50%, #F1F5F9 100%) !important;
            color: #0F172A !important;
        }

        .stAppHeader {
            background: transparent !important;
        }

        /* Glassmorphism Cards Light */
        .glass-card {
            background: rgba(255, 255, 255, 0.9) !important;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid #E2E8F0 !important;
            border-radius: 18px !important;
            padding: 26px !important;
            margin-bottom: 22px !important;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.06) !important;
        }

        .hero-title {
            background: linear-gradient(90deg, #1E40AF 0%, #3B82F6 50%, #6D28D9 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.9rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin-bottom: 0.2rem;
        }

        .hero-subtitle {
            color: #475569 !important;
            font-size: 1.15rem;
            margin-bottom: 1.8rem;
        }

        .metric-badge {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 30px;
            font-weight: 800;
            font-size: 0.95rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .badge-high-risk {
            background: #FEE2E2 !important;
            color: #991B1B !important;
            border: 1px solid #FCA5A5 !important;
        }

        .badge-low-risk {
            background: #DCFCE7 !important;
            color: #166534 !important;
            border: 1px solid #86EFAC !important;
        }

        .stat-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: #F1F5F9 !important;
            border-radius: 12px;
            padding: 16px 20px;
            margin-top: 12px;
            border: 1px solid #CBD5E1 !important;
        }

        .stat-label {
            color: #475569 !important;
            font-size: 0.95rem;
            font-weight: 600;
        }

        .stat-value {
            font-size: 1.5rem;
            font-weight: 800;
            color: #0F172A !important;
        }

        .stButton>button {
            background: linear-gradient(90deg, #2563EB 0%, #4F46E5 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 14px !important;
            padding: 14px 28px !important;
            font-weight: 700 !important;
            font-size: 1.05rem !important;
            width: 100% !important;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3) !important;
        }

        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4) !important;
        }

        hr {
            border-color: #E2E8F0 !important;
        }
        </style>
        """
    else:
        css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        .stApp {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%) !important;
            color: #F8FAFC !important;
        }

        .stAppHeader {
            background: transparent !important;
        }

        /* Glassmorphism Cards Dark */
        .glass-card {
            background: rgba(30, 41, 59, 0.75) !important;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 18px !important;
            padding: 26px !important;
            margin-bottom: 22px !important;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35) !important;
        }

        .hero-title {
            background: linear-gradient(90deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.9rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin-bottom: 0.2rem;
        }

        .hero-subtitle {
            color: #94A3B8 !important;
            font-size: 1.15rem;
            margin-bottom: 1.8rem;
        }

        .metric-badge {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 30px;
            font-weight: 800;
            font-size: 0.95rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .badge-high-risk {
            background: rgba(239, 68, 68, 0.25) !important;
            color: #FCA5A5 !important;
            border: 1px solid #EF4444 !important;
        }

        .badge-low-risk {
            background: rgba(34, 197, 94, 0.25) !important;
            color: #86EFAC !important;
            border: 1px solid #22C55E !important;
        }

        .stat-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(15, 23, 42, 0.6) !important;
            border-radius: 12px;
            padding: 16px 20px;
            margin-top: 12px;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
        }

        .stat-label {
            color: #94A3B8 !important;
            font-size: 0.95rem;
            font-weight: 600;
        }

        .stat-value {
            font-size: 1.5rem;
            font-weight: 800;
            color: #F8FAFC !important;
        }

        .stButton>button {
            background: linear-gradient(90deg, #3B82F6 0%, #6366F1 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 14px !important;
            padding: 14px 28px !important;
            font-weight: 700 !important;
            font-size: 1.05rem !important;
            width: 100% !important;
            box-shadow: 0 4px 14px rgba(59, 130, 246, 0.4) !important;
        }

        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6) !important;
        }

        hr {
            border-color: rgba(255, 255, 255, 0.1) !important;
        }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        return None
    return joblib.load(MODEL_FILE)


def sidebar_inputs() -> dict:
    # ----------------------------------------------------------------------- #
    # Dynamic Light / Dark Theme Switcher Control
    # ----------------------------------------------------------------------- #
    st.sidebar.markdown("### 🎨 Theme Selector")
    theme_choice = st.sidebar.radio(
        "Appearance",
        ["🌙 Dark Mode", "☀️ Light Mode"],
        index=0 if st.session_state["theme_mode"] == "Dark" else 1,
        label_visibility="collapsed",
    )
    st.session_state["theme_mode"] = "Dark" if "Dark" in theme_choice else "Light"

    st.sidebar.divider()

    st.sidebar.markdown(
        "<h2 style='font-size:1.3rem; margin-bottom:10px;'>⚙️ Customer Parameters</h2>",
        unsafe_allow_html=True,
    )

    # Preset profile loader for quick testing
    preset = st.sidebar.selectbox(
        "Load Preset Customer Profile",
        ["Custom Input", "High-Risk Customer Example", "Loyal Customer Example"],
    )

    if preset == "High-Risk Customer Example":
        def_gender, def_senior, def_partner, def_dep = "Female", 0, "No", "No"
        def_tenure, def_contract, def_payment = 2, "Month-to-month", "Electronic check"
        def_internet, def_sec, def_back, def_tech = "Fiber optic", "No", "No", "No"
        def_monthly, def_total = 95.0, 190.0
    elif preset == "Loyal Customer Example":
        def_gender, def_senior, def_partner, def_dep = "Male", 0, "Yes", "Yes"
        def_tenure, def_contract, def_payment = 48, "Two year", "Bank transfer (automatic)"
        def_internet, def_sec, def_back, def_tech = "DSL", "Yes", "Yes", "Yes"
        def_monthly, def_total = 65.0, 3120.0
    else:
        def_gender, def_senior, def_partner, def_dep = "Male", 0, "Yes", "No"
        def_tenure, def_contract, def_payment = 12, "Month-to-month", "Electronic check"
        def_internet, def_sec, def_back, def_tech = "Fiber optic", "No", "Yes", "No"
        def_monthly, def_total = 75.0, 900.0

    with st.sidebar.expander("👤 Demographics", expanded=True):
        gender = st.selectbox("Gender", ["Male", "Female"], index=["Male", "Female"].index(def_gender))
        senior = st.selectbox("Senior Citizen", [0, 1], index=def_senior, format_func=lambda x: "Yes" if x else "No")
        partner = st.selectbox("Partner", ["Yes", "No"], index=["Yes", "No"].index(def_partner))
        dependents = st.selectbox("Dependents", ["Yes", "No"], index=["Yes", "No"].index(def_dep))
        tenure = st.slider("Tenure (Months)", 0, 72, def_tenure, help="Months customer has stayed with company")

    with st.sidebar.expander("📡 Services Subscribed", expanded=False):
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], index=["Fiber optic", "DSL", "No"].index(def_internet))
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(def_sec))
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], index=["Yes", "No", "No internet service"].index(def_back))
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(def_tech))
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    with st.sidebar.expander("💳 Billing & Contract", expanded=True):
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(def_contract))
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(def_payment),
        )
        monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, def_monthly, step=1.0)
        total = st.number_input(
            "Total Charges ($)", 0.0, 10000.0, def_total, step=10.0
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
    # Inject Theme Styles dynamically
    inject_custom_css(st.session_state["theme_mode"])

    st.markdown("<h1 class='hero-title'>🔮 TelePredict Pro</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='hero-subtitle'>Production Machine Learning Intelligence for Customer Retention & Churn Risk Analytics</p>",
        unsafe_allow_html=True,
    )

    model = load_model()
    if model is None:
        st.error("⚠️ No trained model pipeline found. Run `python -m src.train` to generate `models/churn_model.pkl`.")
        return

    inputs = sidebar_inputs()

    tab1, tab2, tab3 = st.tabs(["🎯 Real-Time Churn Scoring", "📊 Machine Learning Insights", "📘 Pipeline Specification"])

    with tab1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Predict Customer Retention Risk")
        st.caption("Select customer profile parameters or load preset examples from the left sidebar to generate real-time predictions.")

        if st.button("🚀 Calculate Churn Risk Score", type="primary"):
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
                    st.markdown("<h3 style='color:#EF4444; margin-top:10px;'>At Risk of Leaving</h3>", unsafe_allow_html=True)
                else:
                    st.markdown(
                        "<div style='margin-top:10px;'><span class='metric-badge badge-low-risk'>✅ RETAINED / LOW RISK</span></div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("<h3 style='color:#22C55E; margin-top:10px;'>Likely to Stay</h3>", unsafe_allow_html=True)

            with res_col2:
                st.metric("Churn Risk Probability", f"{prob:.1%}")
                st.progress(float(prob))

            with res_col3:
                st.metric("Estimated Monthly Value", f"${inputs['MonthlyCharges']:.2f}")
                st.caption(f"Contract: **{inputs['Contract']}**")

            st.divider()

            # Retention Recommendations
            st.markdown("#### 💡 Prescriptive Retention Actions")
            if pred == 1:
                st.warning(
                    "• **Offer Long-Term Contract Incentives**: Month-to-month contracts are the #1 predictor of churn. Offer a 1-year or 2-year discount.\n"
                    "• **Bundle Tech Support & Cyber Security**: Adding TechSupport and OnlineSecurity reduces churn probability by up to 35%.\n"
                    "• **Promote Automated Direct Payment**: Encourage switching from Electronic Check to Bank Transfer or Credit Card."
                )
            else:
                st.success("Customer profile is highly stable. Maintain engagement and explore up-selling streaming or device protection add-ons.")

        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.subheader("📈 Model Analytics & Feature Drivers")
        insight_col1, insight_col2 = st.columns(2)

        with insight_col1:
            if FEATURE_IMPORTANCE_FILE.exists():
                st.markdown("**Top Churn Drivers (Random Forest Feature Importance)**")
                st.image(str(FEATURE_IMPORTANCE_FILE), width="stretch")

        with insight_col2:
            if MODEL_COMPARISON_FILE.exists():
                st.markdown("**Model Performance Benchmark (Test ROC-AUC)**")
                st.image(str(MODEL_COMPARISON_FILE), width="stretch")

        st.divider()
        diag_col1, diag_col2 = st.columns(2)
        with diag_col1:
            if CONFUSION_MATRIX_FILE.exists():
                st.markdown("**Held-Out Test Set Confusion Matrix**")
                st.image(str(CONFUSION_MATRIX_FILE), width="stretch")
        with diag_col2:
            if ROC_CURVE_FILE.exists():
                st.markdown("**Receiver Operating Characteristic (ROC Curve)**")
                st.image(str(ROC_CURVE_FILE), width="stretch")

    with tab3:
        st.subheader("⚙️ Machine Learning Pipeline Architecture")
        st.markdown(
            """
            - **Data Cleaning & Imputation**: `TotalCharges` coerced to numeric; median imputation for numerical columns; most-frequent imputation for categorical features.
            - **Preprocessing Pipeline**: `ColumnTransformer` with `StandardScaler` for numeric columns and `OneHotEncoder(drop='if_binary')` for categoricals.
            - **Class Imbalance Management**: **SMOTE** oversampling embedded inside `imblearn.pipeline.Pipeline` to guarantee zero data leakage into validation folds.
            - **Hyperparameter Tuning**: 5-Fold Stratified Cross-Validation (`GridSearchCV`) optimized on ROC-AUC.
            - **Artifact Production**: Serialized end-to-end pipeline stored in `models/churn_model.pkl`.
            """
        )


if __name__ == "__main__":
    main()
