"""Domain-driven feature engineering for the Telco churn dataset.

All transformations here are row-wise and computed purely from raw column
values (no fitted statistics), so they are safe to apply identically to
training data, test data, and single prediction requests without risk of
data leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

ADDON_SERVICE_COLUMNS: list[str] = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

TENURE_BINS = [-np.inf, 12, 24, 48, 60, np.inf]
TENURE_LABELS = ["0-12", "13-24", "25-48", "49-60", "61-72"]


def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket ``tenure`` (months) into human-readable groups."""
    df["TenureGroup"] = pd.cut(df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS)
    return df


def add_contract_length_category(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the 3-level ``Contract`` column into Short vs Long term."""
    mapping = {
        "Month-to-month": "Short-term",
        "One year": "Long-term",
        "Two year": "Long-term",
    }
    df["ContractLengthCategory"] = df["Contract"].map(mapping)
    return df


def add_avg_monthly_spend(df: pd.DataFrame) -> pd.DataFrame:
    """Average historical monthly spend = TotalCharges / max(tenure, 1).

    Using ``max(tenure, 1)`` avoids division by zero for brand-new customers
    (tenure == 0), whose TotalCharges is also 0.
    """
    safe_tenure = df["tenure"].replace(0, 1)
    df["AvgMonthlySpend"] = (df["TotalCharges"] / safe_tenure).round(2)
    return df


def add_total_services_count(df: pd.DataFrame) -> pd.DataFrame:
    """Count how many add-on/telecom services each customer subscribes to."""
    df["TotalServicesCount"] = (df[ADDON_SERVICE_COLUMNS] == "Yes").sum(axis=1)
    return df


def add_has_internet_service_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flag: does the customer have any internet service at all."""
    df["HasInternetService"] = np.where(
        df["InternetService"] == "No", "No", "Yes"
    )
    return df


def add_senior_citizen_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Readable Yes/No version of the raw 0/1 ``SeniorCitizen`` column."""
    df["IsSeniorCitizen"] = np.where(df["SeniorCitizen"] == 1, "Yes", "No")
    df = df.drop(columns=["SeniorCitizen"])
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full set of engineered features to a cleaned DataFrame.

    Parameters
    ----------
    df:
        A cleaned DataFrame (see ``src.preprocessing.clean_raw_data``) that
        still contains the original raw columns needed to derive new ones.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional engineered columns appended.
    """
    df = df.copy()
    logger.info("Adding engineered features...")
    df = add_tenure_group(df)
    df = add_contract_length_category(df)
    df = add_avg_monthly_spend(df)
    df = add_total_services_count(df)
    df = add_has_internet_service_flag(df)
    df = add_senior_citizen_flag(df)
    logger.info("Feature engineering complete. New shape: %s", df.shape)
    return df
