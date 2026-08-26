"""
Bộ chỉ số đánh giá toàn diện: Classification + Probabilistic + Calibration.

Trích xuất từ notebook gốc (cell 52).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, log_loss, precision_score,
    recall_score, roc_auc_score,
)


def compute_full_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    target_names: tuple[str, str] = ("Non-Default", "Default"),
) -> dict:
    """
    Tính toàn bộ metrics: Accuracy, Precision, Recall, F1, Specificity,
    ROC-AUC, PR-AUC, Log Loss, confusion matrix, và classification report.

    Returns
    -------
    dict chứa toàn bộ giá trị scalar + 'confusion_matrix' (tn, fp, fn, tp)
    + 'classification_report' (str).
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    ll = log_loss(y_true, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    report = classification_report(y_true, y_pred, target_names=list(target_names))

    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1,
        "specificity": specificity, "sensitivity": sensitivity,
        "roc_auc": roc_auc, "pr_auc": pr_auc, "log_loss": ll,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "classification_report": report,
    }


def metrics_to_dataframe(metrics_by_model: dict[str, dict]) -> pd.DataFrame:
    """
    Gộp kết quả compute_full_metrics() của nhiều mô hình thành 1 bảng so sánh
    (Accuracy, Precision, Recall, Specificity, F1, AUC-ROC, AUC-PR).
    """
    rows = []
    for name, m in metrics_by_model.items():
        rows.append({
            "Model": name,
            "Accuracy": m["accuracy"], "Precision": m["precision"],
            "Recall": m["recall"], "Specificity": m["specificity"],
            "F1": m["f1_score"], "AUC-ROC": m["roc_auc"], "AUC-PR": m["pr_auc"],
        })
    return pd.DataFrame(rows).set_index("Model")
