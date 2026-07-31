"""Central configuration for the customer churn prediction pipeline.

All paths, column groups, and hyperparameter-search grids used across the
project are defined here so that every module (and notebook) shares a single
source of truth.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_STATE: int = 42

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

RAW_DATA_FILE: Path = RAW_DATA_DIR / "Telco-Customer-Churn.csv"
PROCESSED_TRAIN_FILE: Path = PROCESSED_DATA_DIR / "train.csv"
PROCESSED_TEST_FILE: Path = PROCESSED_DATA_DIR / "test.csv"

MODELS_DIR: Path = PROJECT_ROOT / "models"
MODEL_FILE: Path = MODELS_DIR / "churn_model.pkl"
PREPROCESSOR_FILE: Path = MODELS_DIR / "preprocessor.pkl"

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
METRICS_FILE: Path = REPORTS_DIR / "metrics.json"
CONFUSION_MATRIX_FILE: Path = REPORTS_DIR / "confusion_matrix.png"
ROC_CURVE_FILE: Path = REPORTS_DIR / "roc_curve.png"
FEATURE_IMPORTANCE_FILE: Path = REPORTS_DIR / "feature_importance.png"
CORRELATION_HEATMAP_FILE: Path = REPORTS_DIR / "correlation_heatmap.png"
TARGET_DISTRIBUTION_FILE: Path = REPORTS_DIR / "target_distribution.png"
MODEL_COMPARISON_FILE: Path = REPORTS_DIR / "model_comparison.png"
SMOTE_COMPARISON_FILE: Path = REPORTS_DIR / "smote_class_distribution.png"

LOGS_DIR: Path = PROJECT_ROOT / "logs"

# Dataset source used by ``src.data_loader`` when data/raw is empty.
DATASET_URL: str = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

# --------------------------------------------------------------------------- #
# Dataset schema
# --------------------------------------------------------------------------- #
TARGET_COLUMN: str = "Churn"
ID_COLUMN: str = "customerID"

NUMERICAL_FEATURES: list[str] = ["tenure", "MonthlyCharges", "TotalCharges"]

BINARY_CATEGORICAL_FEATURES: list[str] = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]

MULTI_CATEGORICAL_FEATURES: list[str] = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]

# Engineered features created in ``src.feature_engineering``.
ENGINEERED_NUMERICAL_FEATURES: list[str] = [
    "AvgMonthlySpend",
    "TotalServicesCount",
]

ENGINEERED_CATEGORICAL_FEATURES: list[str] = [
    "TenureGroup",
    "ContractLengthCategory",
    "HasInternetService",
    "IsSeniorCitizen",
]

ALL_CATEGORICAL_FEATURES: list[str] = (
    BINARY_CATEGORICAL_FEATURES
    + MULTI_CATEGORICAL_FEATURES
    + ENGINEERED_CATEGORICAL_FEATURES
)
ALL_NUMERICAL_FEATURES: list[str] = NUMERICAL_FEATURES + ENGINEERED_NUMERICAL_FEATURES

# --------------------------------------------------------------------------- #
# Train / test split & imbalance handling
# --------------------------------------------------------------------------- #
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5
SMOTE_SAMPLING_STRATEGY: str = "auto"

# --------------------------------------------------------------------------- #
# Model hyperparameter search grids
# --------------------------------------------------------------------------- #
MODEL_PARAM_GRIDS: dict = {
    "logistic_regression": {
        "estimator__C": [0.01, 0.1, 1, 10],
        "estimator__penalty": ["l2"],
        "estimator__solver": ["lbfgs"],
        "estimator__max_iter": [1000],
    },
    "decision_tree": {
        "estimator__max_depth": [3, 5, 7, 10, None],
        "estimator__min_samples_split": [2, 5, 10],
        "estimator__min_samples_leaf": [1, 2, 4],
        "estimator__criterion": ["gini", "entropy"],
    },
    "random_forest": {
        "estimator__n_estimators": [100, 200, 300],
        "estimator__max_depth": [5, 10, 15, None],
        "estimator__min_samples_split": [2, 5],
        "estimator__min_samples_leaf": [1, 2],
        "estimator__max_features": ["sqrt", "log2"],
    },
}

SCORING_METRIC: str = "roc_auc"
