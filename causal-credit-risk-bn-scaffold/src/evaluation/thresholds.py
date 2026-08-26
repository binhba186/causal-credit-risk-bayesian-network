"""
Ngưỡng phân loại tối ưu từ đường ROC + Decision Curve Analysis (DCA).

3 tiêu chí chọn ngưỡng:
  - Youden Index    : J(c) = Se(c) + Sp(c) - 1        -> tối đa hoá
  - Closest to (0,1): D(c) = sqrt((1-Se)^2+(1-Sp)^2)   -> tối thiểu hoá
  - Symmetry Point  : |Se + FPR - 1| / sqrt(2)          -> tối thiểu hoá

Decision Curve Analysis (Vickers & Elkin, 2006):
  NB(c) = p*Se(c) - (1-p)*(1-Sp(c))*[c/(1-c)]

Trích xuất từ notebook gốc (cells 54, 55).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.metrics import auc, roc_curve


def compute_roc_thresholds(y_val: np.ndarray, y_prob_val: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """
    Tính đường ROC và 3 ngưỡng tối ưu (Youden / Closest-to-(0,1) / Symmetry).

    Returns
    -------
    (threshold_summary, roc_data)
        threshold_summary : DataFrame 3 dòng {Criterion, Notation, Threshold,
            Sensitivity, Specificity, FPR, TPR}.
        roc_data : dict chứa fpr_raw, tpr_raw, thresholds_raw, roc_auc — cần
            cho decision_curve_analysis() và vẽ đồ thị.
    """
    y_val = np.asarray(y_val).astype(int)
    y_prob_val = np.clip(np.asarray(y_prob_val).astype(float), 0, 1)

    fpr_raw, tpr_raw, thresholds_raw = roc_curve(y_val, y_prob_val)
    roc_auc_val = auc(fpr_raw, tpr_raw)

    fpr_unique, unique_idx = np.unique(fpr_raw, return_index=True)
    tpr_unique = tpr_raw[unique_idx]

    tpr_of_fpr = interp1d(
        fpr_unique, tpr_unique, kind="linear",
        bounds_error=False, fill_value=(tpr_unique[0], tpr_unique[-1]),
    )

    fine_fpr = np.linspace(0.0, 1.0, 10_000)
    fine_tpr = tpr_of_fpr(fine_fpr)
    fine_specificity = 1.0 - fine_fpr
    fine_sensitivity = fine_tpr

    fine_roc_df = pd.DataFrame({
        "FPR": fine_fpr, "Sensitivity": fine_sensitivity, "Specificity": fine_specificity,
    })
    fine_roc_df["Youden_Index"] = fine_sensitivity + fine_specificity - 1.0
    fine_roc_df["Distance_to_01"] = np.sqrt((1.0 - fine_sensitivity) ** 2 + (1.0 - fine_specificity) ** 2)
    fine_roc_df["Symmetry_Gap"] = np.abs(fine_sensitivity + fine_fpr - 1.0) / np.sqrt(2)

    idx_youden = fine_roc_df["Youden_Index"].idxmax()
    idx_closest = fine_roc_df["Distance_to_01"].idxmin()
    idx_symmetry = fine_roc_df["Symmetry_Gap"].idxmin()

    point_youden = fine_roc_df.loc[idx_youden]
    point_closest = fine_roc_df.loc[idx_closest]
    point_symmetry = fine_roc_df.loc[idx_symmetry]

    valid_mask = thresholds_raw <= 1.0
    fpr_valid, thr_valid = fpr_raw[valid_mask], thresholds_raw[valid_mask]
    fpr_for_thr, thr_for_fpr = np.flip(fpr_valid), np.flip(thr_valid)
    fpr_u2, u2_idx = np.unique(fpr_for_thr, return_index=True)
    thr_u2 = thr_for_fpr[u2_idx]

    threshold_of_fpr = interp1d(
        fpr_u2, thr_u2, kind="linear", bounds_error=False,
        fill_value=(thr_u2[0], thr_u2[-1]),
    )

    def get_threshold(fpr_val: float) -> float:
        return float(threshold_of_fpr(fpr_val))

    threshold_summary = pd.DataFrame([
        {"Criterion": "Youden Index", "Notation": "c_J",
         "Threshold": get_threshold(point_youden["FPR"]),
         "Sensitivity": float(point_youden["Sensitivity"]),
         "Specificity": float(point_youden["Specificity"]),
         "FPR": float(point_youden["FPR"]), "TPR": float(point_youden["Sensitivity"])},
        {"Criterion": "Closest to (0,1)", "Notation": "c_D",
         "Threshold": get_threshold(point_closest["FPR"]),
         "Sensitivity": float(point_closest["Sensitivity"]),
         "Specificity": float(point_closest["Specificity"]),
         "FPR": float(point_closest["FPR"]), "TPR": float(point_closest["Sensitivity"])},
        {"Criterion": "Symmetry Point", "Notation": "c_S",
         "Threshold": get_threshold(point_symmetry["FPR"]),
         "Sensitivity": float(point_symmetry["Sensitivity"]),
         "Specificity": float(point_symmetry["Specificity"]),
         "FPR": float(point_symmetry["FPR"]), "TPR": float(point_symmetry["Sensitivity"])},
    ])

    roc_data = {
        "fpr_raw": fpr_raw, "tpr_raw": tpr_raw,
        "thresholds_raw": thresholds_raw, "roc_auc": roc_auc_val,
    }
    return threshold_summary, roc_data


def decision_curve_analysis(
    y_val: np.ndarray,
    roc_data: dict,
    threshold_summary: pd.DataFrame,
    c_min: float = 0.0,
    c_max: float = 1.0,
    n_points: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Decision Curve Analysis (Vickers & Elkin, 2006).

    So sánh Net Benefit của mô hình với 2 chiến lược baseline:
      - Treat All  : NB_all(c)  = p - (1-p)*[c/(1-c)]
      - Treat None : NB_none(c) = 0

    Returns
    -------
    (dca_df, nb_summary)
        dca_df     : lưới NB theo c cho Model / Treat All / Treat None.
        nb_summary : NB tại 3 ngưỡng ROC tối ưu (bảng tham chiếu phụ,
            KHÔNG phải "ngưỡng do DCA chọn" — DCA đánh giá lợi ích trên
            toàn vùng c, không tự chọn ngưỡng).
    """
    prevalence = float(np.mean(y_val))
    thresholds_raw, fpr_raw, tpr_raw = (
        roc_data["thresholds_raw"], roc_data["fpr_raw"], roc_data["tpr_raw"]
    )

    valid_mask = thresholds_raw <= 1.0
    thr_valid, fpr_valid, tpr_valid = thresholds_raw[valid_mask], fpr_raw[valid_mask], tpr_raw[valid_mask]
    thr_asc, fpr_asc, tpr_asc = np.flip(thr_valid), np.flip(fpr_valid), np.flip(tpr_valid)
    thr_u, u_idx = np.unique(thr_asc, return_index=True)
    fpr_u, tpr_u = fpr_asc[u_idx], tpr_asc[u_idx]

    se_of_c = interp1d(thr_u, tpr_u, kind="linear", bounds_error=False, fill_value=(tpr_u[0], tpr_u[-1]))
    fpr_of_c = interp1d(thr_u, fpr_u, kind="linear", bounds_error=False, fill_value=(fpr_u[0], fpr_u[-1]))

    def sp_of_c(c):
        return 1.0 - fpr_of_c(c)

    def net_benefit_model(c, prevalence):
        c = np.clip(c, 1e-6, 1.0 - 1e-6)
        return prevalence * se_of_c(c) - (1.0 - prevalence) * (1.0 - sp_of_c(c)) * (c / (1.0 - c))

    def net_benefit_treat_all(c, prevalence):
        c = np.clip(c, 1e-6, 1.0 - 1e-6)
        return prevalence - (1.0 - prevalence) * (c / (1.0 - c))

    c_grid = np.linspace(c_min, c_max, n_points)
    dca_df = pd.DataFrame({
        "c": c_grid,
        "Net_Benefit_Model": net_benefit_model(c_grid, prevalence),
        "Net_Benefit_Treat_All": net_benefit_treat_all(c_grid, prevalence),
        "Net_Benefit_Treat_None": 0.0,
    })

    nb_records = []
    for _, row in threshold_summary.iterrows():
        c = float(row["Threshold"])
        nb_m = float(net_benefit_model(c, prevalence))
        nb_all = float(net_benefit_treat_all(c, prevalence))
        baseline_label = "vs Treat All" if nb_all > 0 else "vs Treat None"
        delta_nb = (nb_m - nb_all) if nb_all > 0 else nb_m
        nb_records.append({
            "Criterion": row["Criterion"], "Notation": row["Notation"],
            "Threshold (c)": c, "NB Model": nb_m, "NB Treat All": nb_all,
            "NB Treat None": 0.0, "Delta NB (baseline)": baseline_label, "Delta NB": delta_nb,
        })

    nb_summary = pd.DataFrame(nb_records)
    return dca_df, nb_summary


def evaluate_at_threshold(y_val: np.ndarray, y_prob_val: np.ndarray, threshold: float) -> dict:
    """Tính Accuracy/Sensitivity/Specificity/Precision/F1 tại một ngưỡng cụ thể."""
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score

    y_pred = (y_prob_val >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_val, y_pred),
        "Sensitivity": sensitivity, "Specificity": specificity,
        "Precision": precision_score(y_val, y_pred, zero_division=0),
        "F1_score": f1_score(y_val, y_pred, zero_division=0),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
    }
