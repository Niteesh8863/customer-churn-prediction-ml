"""Model evaluation utilities and a standalone CLI to score a saved model.

The core :func:`compute_metrics` and :func:`get_feature_importance` helpers
are imported by :mod:`src.train` during training. Running this file directly
re-evaluates an already-trained, saved model against the held-out processed
test split (useful for CI checks without retraining).
"""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score

from src.config import (
    CONFUSION_MATRIX_FILE,
    CV_FOLDS,
    METRICS_FILE,
    MODEL_FILE,
    PROCESSED_TEST_FILE,
    ROC_CURVE_FILE,
    TARGET_COLUMN,
)
from src.utils import get_logger, save_json
from src.visualization import plot_confusion_matrix, plot_roc_curve

logger = get_logger(__name__)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Compute the full evaluation metric suite for a binary classifier.

    Returns
    -------
    dict
        accuracy, precision, recall, f1, roc_auc, confusion_matrix (list of
        lists) and the full sklearn classification report (as a dict).
    """
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
    }


def compute_cross_val_scores(
    pipeline, X, y, cv: int = CV_FOLDS, scoring: str = "roc_auc"
) -> dict:
    """Run k-fold cross-validation for an (unfitted) pipeline and summarize it."""
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        "cv_scoring": scoring,
        "cv_folds": cv,
        "cv_scores": [round(s, 4) for s in scores],
        "cv_mean": round(float(np.mean(scores)), 4),
        "cv_std": round(float(np.std(scores)), 4),
    }


def get_feature_importance(pipeline) -> tuple[list[str], np.ndarray]:
    """Extract feature names and importances from a fitted tree-based pipeline.

    Assumes the pipeline has a ``preprocessor`` step exposing
    ``get_feature_names_out`` and an ``estimator`` step exposing
    ``feature_importances_`` (e.g. RandomForest, DecisionTree).
    """
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["estimator"].feature_importances_
    return list(feature_names), importances


def evaluate_saved_model() -> dict:
    """Reload the persisted best model and score it against the processed test set."""
    logger.info("Loading model from %s", MODEL_FILE)
    pipeline = joblib.load(MODEL_FILE)

    test_df = pd.read_csv(PROCESSED_TEST_FILE)
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba)
    logger.info("Re-evaluation metrics: %s", {k: v for k, v in metrics.items() if k != "classification_report"})

    plot_confusion_matrix(y_test, y_pred, save_path=CONFUSION_MATRIX_FILE)
    plot_roc_curve(y_test, y_proba, save_path=ROC_CURVE_FILE)
    save_json(metrics, METRICS_FILE)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-evaluate the saved churn model against the processed test split."
    )
    parser.parse_args()
    metrics = evaluate_saved_model()
    print("\n=== Evaluation Metrics ===")
    for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        print(f"{key:>10}: {metrics[key]}")


if __name__ == "__main__":
    main()
