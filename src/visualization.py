"""Reusable plotting functions for EDA and model-evaluation reports.

Every function saves its figure to disk (when given a ``save_path``) and
returns the ``matplotlib`` ``Figure`` so it can also be displayed inline in
a notebook.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix

from src.utils import get_logger

logger = get_logger(__name__)

sns.set_theme(style="whitegrid", palette="viridis")


def _save(fig: plt.Figure, save_path: Path | None) -> None:
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info("Saved figure to %s", save_path)


def plot_target_distribution(
    df: pd.DataFrame, target_col: str = "Churn", save_path: Path | None = None
) -> plt.Figure:
    """Bar + pie chart of the churn target class balance."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    counts = df[target_col].value_counts()
    labels = [str(c) for c in counts.index]

    sns.barplot(x=labels, y=counts.values, ax=axes[0], hue=labels, legend=False)
    axes[0].set_title("Churn Class Counts")
    axes[0].set_xlabel(target_col)
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 20, str(v), ha="center")

    axes[1].pie(
        counts.values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=sns.color_palette("viridis", len(counts)),
    )
    axes[1].set_title("Churn Class Proportion")

    fig.suptitle("Target Distribution: Customer Churn", fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_missing_values(df: pd.DataFrame, save_path: Path | None = None) -> plt.Figure:
    """Bar chart of missing-value counts per column (if any)."""
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    if missing.empty:
        ax.text(0.5, 0.5, "No missing values found", ha="center", va="center")
        ax.axis("off")
    else:
        sns.barplot(x=missing.values, y=missing.index, ax=ax, hue=missing.index, legend=False)
        ax.set_xlabel("Missing count")
    ax.set_title("Missing Values by Column")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame, save_path: Path | None = None
) -> plt.Figure:
    """Correlation heatmap of numeric features."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="viridis", ax=ax, square=True)
    ax.set_title("Correlation Heatmap (Numeric Features)", fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_categorical_distributions(
    df: pd.DataFrame,
    categorical_cols: list[str],
    target_col: str = "Churn",
    save_path: Path | None = None,
) -> plt.Figure:
    """Grid of count plots for categorical features, split by churn."""
    n_cols = 3
    n_rows = int(np.ceil(len(categorical_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(categorical_cols):
        sns.countplot(data=df, x=col, hue=target_col, ax=axes[i])
        axes[i].set_title(col)
        axes[i].tick_params(axis="x", rotation=30)

    for j in range(len(categorical_cols), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Categorical Feature Distributions by Churn", fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_numerical_distributions(
    df: pd.DataFrame,
    numerical_cols: list[str],
    target_col: str = "Churn",
    save_path: Path | None = None,
) -> plt.Figure:
    """Grid of histograms (KDE overlay) for numeric features, split by churn."""
    fig, axes = plt.subplots(1, len(numerical_cols), figsize=(6 * len(numerical_cols), 4.5))
    if len(numerical_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, numerical_cols):
        sns.histplot(data=df, x=col, hue=target_col, kde=True, ax=ax, element="step")
        ax.set_title(f"Distribution of {col}")

    fig.suptitle("Numerical Feature Distributions by Churn", fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_boxplots(
    df: pd.DataFrame,
    numerical_cols: list[str],
    target_col: str = "Churn",
    save_path: Path | None = None,
) -> plt.Figure:
    """Boxplots of numeric features grouped by churn, for outlier inspection."""
    fig, axes = plt.subplots(1, len(numerical_cols), figsize=(5 * len(numerical_cols), 4.5))
    if len(numerical_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, numerical_cols):
        sns.boxplot(data=df, x=target_col, y=col, ax=ax, hue=target_col, legend=False)
        ax.set_title(f"{col} by Churn")

    fig.suptitle("Boxplots of Numerical Features by Churn", fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_pairplot(
    df: pd.DataFrame,
    numerical_cols: list[str],
    target_col: str = "Churn",
    save_path: Path | None = None,
):
    """Pairwise scatter/KDE grid of numeric features colored by churn."""
    grid = sns.pairplot(df[numerical_cols + [target_col]], hue=target_col, corner=True)
    grid.fig.suptitle("Pairplot of Numerical Features", y=1.02, fontweight="bold")
    _save(grid.fig, save_path)
    return grid.fig


def plot_smote_class_distribution(
    y_before: pd.Series, y_after: pd.Series, save_path: Path | None = None
) -> plt.Figure:
    """Side-by-side class balance bar charts, before vs. after SMOTE."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, y, title in zip(axes, [y_before, y_after], ["Before SMOTE", "After SMOTE"]):
        counts = pd.Series(y).value_counts().sort_index()
        labels = [str(c) for c in counts.index]
        sns.barplot(x=labels, y=counts.values, ax=ax, hue=labels, legend=False)
        ax.set_title(title)
        ax.set_xlabel("Churn")
        ax.set_ylabel("Count")
        for i, v in enumerate(counts.values):
            ax.text(i, v + 5, str(v), ha="center")

    fig.suptitle("Class Distribution: Before vs. After SMOTE", fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_confusion_matrix(
    y_true, y_pred, save_path: Path | None = None, labels=("No Churn", "Churn")
) -> plt.Figure:
    """Confusion matrix heatmap for a trained classifier's predictions."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="viridis", colorbar=False)
    ax.set_title("Confusion Matrix", fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_roc_curve(y_true, y_proba, save_path: Path | None = None) -> plt.Figure:
    """ROC curve for a trained classifier's predicted probabilities."""
    fig, ax = plt.subplots(figsize=(6, 5.5))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax, name="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_title("ROC Curve", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    top_n: int = 15,
    save_path: Path | None = None,
) -> plt.Figure:
    """Horizontal bar chart of the top-N most important features."""
    order = np.argsort(importances)[::-1][:top_n]
    top_features = np.array(feature_names)[order]
    top_importances = importances[order]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * top_n)))
    sns.barplot(x=top_importances, y=top_features, ax=ax, hue=top_features, legend=False)
    ax.set_title(f"Top {top_n} Feature Importances (Random Forest)", fontweight="bold")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_model_comparison(
    results: dict[str, dict[str, float]],
    metric: str = "roc_auc",
    save_path: Path | None = None,
) -> plt.Figure:
    """Bar chart comparing a given metric across trained models."""
    models = list(results.keys())
    scores = [results[m][metric] for m in models]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(x=models, y=scores, ax=ax, hue=models, legend=False)
    ax.set_title(f"Model Comparison — {metric.replace('_', ' ').upper()}", fontweight="bold")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_ylim(0, 1)
    for i, v in enumerate(scores):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
    fig.tight_layout()
    _save(fig, save_path)
    return fig
