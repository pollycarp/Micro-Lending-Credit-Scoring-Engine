"""
Model evaluation utilities.

Metrics used
------------
roc_auc        : Area under the ROC curve — overall discrimination ability.
avg_precision  : Area under the Precision-Recall curve — better than AUC when
                 classes are imbalanced (we have 32% defaults).
log_loss       : Penalises confident wrong predictions — measures calibration.
brier_score    : Mean squared error of predicted probabilities — lower is better.

These four together give a complete picture:
  - Can the model rank borrowers correctly?   → roc_auc, avg_precision
  - Are the probabilities trustworthy?        → log_loss, brier_score
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    log_loss,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
    threshold: float = 0.5,
) -> dict:
    """
    Compute all evaluation metrics for a fitted model.

    Parameters
    ----------
    model      : fitted scikit-learn compatible estimator
    X_test     : feature matrix (test set)
    y_test     : true labels (test set)
    model_name : label used in printed output
    threshold  : decision threshold for binary classification

    Returns
    -------
    dict with keys: roc_auc, avg_precision, log_loss, brier_score
    """
    y_prob  = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_prob >= threshold).astype(int)

    metrics = {
        "roc_auc":       round(roc_auc_score(y_test, y_prob), 4),
        "avg_precision": round(average_precision_score(y_test, y_prob), 4),
        "log_loss":      round(log_loss(y_test, y_prob), 4),
        "brier_score":   round(brier_score_loss(y_test, y_prob), 4),
    }

    return metrics


def print_report(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
    threshold: float = 0.5,
) -> None:
    """Print a detailed evaluation report to stdout."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = evaluate_model(model, X_test, y_test, model_name, threshold)

    print(f"\n{'═' * 55}")
    print(f"  Evaluation Report — {model_name}")
    print(f"{'═' * 55}")
    print(f"  ROC-AUC          : {metrics['roc_auc']:.4f}")
    print(f"  Avg Precision    : {metrics['avg_precision']:.4f}")
    print(f"  Log Loss         : {metrics['log_loss']:.4f}")
    print(f"  Brier Score      : {metrics['brier_score']:.4f}")
    print(f"\n  Classification Report (threshold = {threshold}):")
    print(classification_report(y_test, y_pred, target_names=["Repaid", "Default"]))

    # Score distribution by actual class
    prob_series = pd.Series(y_prob, index=y_test.index, name="pd_score")
    df = pd.DataFrame({"pd_score": prob_series, "actual": y_test})
    print("  PD Score distribution by actual class:")
    print(df.groupby("actual")["pd_score"].describe().round(3).to_string())
    print(f"{'═' * 55}\n")


def get_roc_data(model, X_test, y_test) -> pd.DataFrame:
    """Return FPR/TPR values for plotting the ROC curve."""
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})


def get_pr_data(model, X_test, y_test) -> pd.DataFrame:
    """Return Precision/Recall values for plotting the PR curve."""
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    return pd.DataFrame({
        "precision": precision[:-1],
        "recall":    recall[:-1],
        "threshold": thresholds,
    })
