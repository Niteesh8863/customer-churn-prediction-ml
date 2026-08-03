"""Data cleaning, encoding, scaling, and train/test splitting.

This module implements the non-domain-specific preprocessing steps of the
pipeline: missing-value handling, type coercion, target encoding, and the
scikit-learn ``ColumnTransformer`` used for one-hot encoding + scaling.
Domain feature engineering lives in :mod:`src.feature_engineering`.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    ALL_CATEGORICAL_FEATURES,
    ALL_NUMERICAL_FEATURES,
    ID_COLUMN,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.utils import get_logger

logger = get_logger(__name__)


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Telco churn DataFrame.

    Steps:
        1. Coerce ``TotalCharges`` to numeric (it ships as a string with
           blank values for customers with zero tenure) and fill the
           resulting NaNs with 0.
        2. Drop the ``customerID`` identifier column (not predictive).
        3. Encode the target ``Churn`` column from Yes/No to 1/0.
        4. Drop exact duplicate rows, if any.

    Parameters
    ----------
    df:
        Raw DataFrame as loaded from the source CSV.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for feature engineering.
    """
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_missing = df["TotalCharges"].isna().sum()
    if n_missing:
        logger.info(
            "Found %d missing TotalCharges values (new customers with 0 "
            "tenure); filling with 0.",
            n_missing,
        )
        df["TotalCharges"] = df["TotalCharges"].fillna(0)

    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    if not pd.api.types.is_numeric_dtype(df[TARGET_COLUMN]):
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0}).astype(int)

    before = len(df)
    df = df.drop_duplicates()
    if len(df) != before:
        logger.info("Dropped %d duplicate rows.", before - len(df))

    return df


def get_feature_target_split(
    df: pd.DataFrame, target_col: str = TARGET_COLUMN
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a fully-engineered DataFrame into features ``X`` and target ``y``."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def build_preprocessor(
    numerical_features: list[str] = ALL_NUMERICAL_FEATURES,
    categorical_features: list[str] = ALL_CATEGORICAL_FEATURES,
) -> ColumnTransformer:
    """Build the scikit-learn ``ColumnTransformer`` for encoding + scaling.

    Numerical features are median-imputed and standardized. Categorical
    features are most-frequent-imputed and one-hot encoded (binary features
    collapse to a single dropped-first column; unseen categories at
    inference time are safely ignored).
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numerical_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
    return preprocessor


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split preserving the churn class ratio."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(
        "Split data into train=%s and test=%s (stratified on target).",
        X_train.shape,
        X_test.shape,
    )
    return X_train, X_test, y_train, y_test
