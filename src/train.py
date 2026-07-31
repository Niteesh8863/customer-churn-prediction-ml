"""End-to-end training pipeline: preprocessing, SMOTE, tuning, model selection.

Usage
-----
    python -m src.train

This script:
    1. Loads and cleans the raw Telco churn data.
    2. Applies feature engineering.
    3. Splits into train/test (stratified).
    4. Builds an imbalanced-learn pipeline per model (preprocessing -> SMOTE
       -> estimator) so oversampling only ever touches training folds.
    5. Tunes each model with ``GridSearchCV`` (scoring = ROC-AUC).
    6. Evaluates all tuned models on the held-out test set.
    7. Selects the best model by test ROC-AUC and saves it with ``joblib``.
    8. Saves comparison tables, plots, and a ``metrics.json`` report.
"""

from __future__ import annotations

import time

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.config import (
    CONFUSION_MATRIX_FILE,
    CORRELATION_HEATMAP_FILE,
    CV_FOLDS,
    FEATURE_IMPORTANCE_FILE,
    METRICS_FILE,
    MODEL_COMPARISON_FILE,
    MODEL_FILE,
    MODEL_PARAM_GRIDS,
    PROCESSED_TEST_FILE,
    PROCESSED_TRAIN_FILE,
    RANDOM_STATE,
    ROC_CURVE_FILE,
    SCORING_METRIC,
    SMOTE_COMPARISON_FILE,
    TARGET_COLUMN,
    TARGET_DISTRIBUTION_FILE,
)
from src.data_loader import load_raw_data
from src.evaluate import compute_metrics, get_feature_importance
from src.feature_engineering import add_engineered_features
from src.preprocessing import build_preprocessor, clean_raw_data, get_feature_target_split, split_data
from src.utils import get_logger, save_json, set_seed
from src.visualization import (
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_model_comparison,
    plot_roc_curve,
    plot_smote_class_distribution,
    plot_target_distribution,
)

logger = get_logger(__name__, log_file="train.log")

MODEL_REGISTRY: dict[str, tuple[str, object]] = {
    "Logistic Regression": ("logistic_regression", LogisticRegression(random_state=RANDOM_STATE)),
    "Decision Tree": ("decision_tree", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    "Random Forest": ("random_forest", RandomForestClassifier(random_state=RANDOM_STATE)),
}


def build_pipeline(estimator) -> ImbPipeline:
    """Assemble preprocessing -> SMOTE -> estimator as a single fittable pipeline.

    Because this uses ``imblearn``'s ``Pipeline``, SMOTE resampling is applied
    only during ``fit`` (and only to the training folds inside cross-
    validation) — never to validation/test data, which avoids information
    leakage into evaluation.
    """
    return ImbPipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("estimator", estimator),
        ]
    )


def train_and_tune(
    name: str, estimator, param_grid: dict, X_train: pd.DataFrame, y_train: pd.Series
) -> GridSearchCV:
    """Run GridSearchCV for one model and return the fitted search object."""
    pipeline = build_pipeline(estimator)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=SCORING_METRIC,
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )

    logger.info("Tuning %s with %d-fold CV (scoring=%s)...", name, CV_FOLDS, SCORING_METRIC)
    start = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start
    logger.info(
        "%s tuning complete in %.1fs. Best CV %s = %.4f. Best params: %s",
        name,
        elapsed,
        SCORING_METRIC,
        search.best_score_,
        search.best_params_,
    )
    return search


def main() -> None:
    set_seed()

    # 1. Load & clean raw data --------------------------------------------------
    raw_df = load_raw_data()
    clean_df = clean_raw_data(raw_df)

    # 2. Feature engineering ------------------------------------------------------
    engineered_df = add_engineered_features(clean_df)

    # EDA artifacts generated as a byproduct of training, for convenience.
    plot_target_distribution(clean_df, save_path=TARGET_DISTRIBUTION_FILE)
    plot_correlation_heatmap(engineered_df, save_path=CORRELATION_HEATMAP_FILE)

    # 3. Train/test split ----------------------------------------------------------
    X, y = get_feature_target_split(engineered_df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    PROCESSED_TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([X_train, y_train], axis=1).to_csv(PROCESSED_TRAIN_FILE, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(PROCESSED_TEST_FILE, index=False)

    # 4. Visualize class imbalance before/after SMOTE (illustrative only) ---------
    preview_preprocessor = build_preprocessor()
    X_train_transformed = preview_preprocessor.fit_transform(X_train, y_train)
    X_res, y_res = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train_transformed, y_train)
    plot_smote_class_distribution(y_train, y_res, save_path=SMOTE_COMPARISON_FILE)
    logger.info(
        "Class distribution before SMOTE: %s | after SMOTE: %s",
        y_train.value_counts().to_dict(),
        pd.Series(y_res).value_counts().to_dict(),
    )

    # 5. Train, tune & evaluate every candidate model ------------------------------
    results: dict[str, dict] = {}
    fitted_pipelines: dict[str, ImbPipeline] = {}

    for display_name, (grid_key, estimator) in MODEL_REGISTRY.items():
        search = train_and_tune(
            display_name, estimator, MODEL_PARAM_GRIDS[grid_key], X_train, y_train
        )
        best_pipeline = search.best_estimator_
        fitted_pipelines[display_name] = best_pipeline

        y_pred = best_pipeline.predict(X_test)
        y_proba = best_pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)
        metrics["cv_best_score"] = round(search.best_score_, 4)
        metrics["best_params"] = search.best_params_
        results[display_name] = metrics

        logger.info(
            "%s test metrics: accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%.4f",
            display_name,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1_score"],
            metrics["roc_auc"],
        )

    # 6. Model comparison table & chart --------------------------------------------
    comparison_table = pd.DataFrame(
        {
            name: {
                "Accuracy": m["accuracy"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1 Score": m["f1_score"],
                "ROC AUC": m["roc_auc"],
                "CV ROC AUC": m["cv_best_score"],
            }
            for name, m in results.items()
        }
    ).T
    logger.info("\n=== Model Comparison ===\n%s", comparison_table.to_string())
    print("\n=== Model Comparison (Test Set) ===")
    print(comparison_table.to_string())

    plot_model_comparison(results, metric="roc_auc", save_path=MODEL_COMPARISON_FILE)

    # 7. Select the best model by test ROC-AUC ---------------------------------------
    best_model_name = max(results, key=lambda name: results[name]["roc_auc"])
    best_pipeline = fitted_pipelines[best_model_name]
    logger.info(
        "Best model: %s (ROC-AUC=%.4f)", best_model_name, results[best_model_name]["roc_auc"]
    )
    print(f"\nBest model selected: {best_model_name} (ROC-AUC = {results[best_model_name]['roc_auc']:.4f})")

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_FILE)
    logger.info("Saved best model pipeline to %s", MODEL_FILE)

    # 8. Diagnostic plots for the best model -----------------------------------------
    y_pred_best = best_pipeline.predict(X_test)
    y_proba_best = best_pipeline.predict_proba(X_test)[:, 1]
    plot_confusion_matrix(y_test, y_pred_best, save_path=CONFUSION_MATRIX_FILE)
    plot_roc_curve(y_test, y_proba_best, save_path=ROC_CURVE_FILE)

    # Feature importance is always reported for Random Forest, regardless of
    # whether it happened to win overall, since it's the most interpretable
    # of the tree-based candidates.
    rf_pipeline = fitted_pipelines["Random Forest"]
    feature_names, importances = get_feature_importance(rf_pipeline)
    plot_feature_importance(feature_names, importances, save_path=FEATURE_IMPORTANCE_FILE)

    # 9. Persist full metrics report --------------------------------------------------
    report = {
        "best_model": best_model_name,
        "random_state": RANDOM_STATE,
        "test_size": len(X_test) / (len(X_test) + len(X_train)),
        "models": results,
    }
    save_json(report, METRICS_FILE)
    logger.info("Saved full metrics report to %s", METRICS_FILE)
    print(f"\nAll reports saved to the reports/ directory. Best model saved to {MODEL_FILE}")


if __name__ == "__main__":
    main()
