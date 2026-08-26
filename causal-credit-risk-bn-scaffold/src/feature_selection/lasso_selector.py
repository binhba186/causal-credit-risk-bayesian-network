"""
Chọn lọc đặc trưng bằng Hồi quy Lasso (L_Pro).

Kết hợp 3 tiêu chí để chọn C* (hệ số phạt L1) tối ưu:
  1. Max Cross-Validation AUC
  2. Max BIC-penalized AUC (phạt số lượng đặc trưng)
  3. Elbow point trên đường cong N_features theo log(C) (Kneedle algorithm)
C* cuối cùng = trung bình hình học (geometric mean) của 3 tiêu chí.

Trích xuất từ notebook gốc (cells 28, 29, 31).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Chuẩn hoá numeric (StandardScaler) + one-hot categorical."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_features),
        ]
    )


def find_elbow(x: np.ndarray, y: np.ndarray) -> int:
    """
    Kneedle algorithm đơn giản: tìm điểm có khoảng cách vuông góc lớn nhất
    tới đường thẳng nối điểm đầu và điểm cuối (sau khi normalize [0,1]).
    """
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    x_n = (x - x.min()) / (x.max() - x.min() + 1e-10)
    y_n = (y - y.min()) / (y.max() - y.min() + 1e-10)
    d = np.array([x_n[-1] - x_n[0], y_n[-1] - y_n[0]])
    d = d / np.linalg.norm(d)
    dists = [abs((xi - x_n[0]) * d[1] - (yi - y_n[0]) * d[0]) for xi, yi in zip(x_n, y_n)]
    return int(np.argmax(dists))


def select_optimal_C(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    c_grid: np.ndarray | None = None,
    n_cv: int = 5,
    random_state: int = 42,
) -> tuple[float, pd.DataFrame]:
    """
    Quét lưới C, tính CV AUC + BIC-AUC + elbow, trả về C* tối ưu tổng hợp.

    Returns
    -------
    (C_optimal, cv_df) : hệ số C tối ưu và bảng kết quả chi tiết theo từng C.
    """
    if c_grid is None:
        c_grid = np.logspace(-3, 1, 40)

    preprocessor.fit(X_train)
    n_all_feats = len(preprocessor.get_feature_names_out())

    skf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=random_state)
    cv_results = []

    for c_val in c_grid:
        pipe = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("selector", SelectFromModel(
                LogisticRegression(penalty="l1", solver="liblinear", C=c_val,
                                    random_state=random_state, max_iter=5000)
            )),
            ("model", LogisticRegression(penalty="l1", solver="liblinear", C=c_val,
                                          random_state=random_state, max_iter=5000)),
        ])
        try:
            aucs = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="roc_auc")
        except ValueError as e:
            if "Found array with 0 feature(s)" in str(e):
                aucs = np.full(n_cv, np.nan)
            else:
                raise

        n_feats_per_fold = []
        for tr_idx, _ in skf.split(X_train, y_train):
            X_tr_fold, y_tr_fold = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
            p_tmp = Pipeline([
                ("preprocessor", clone(preprocessor)),
                ("selector", SelectFromModel(
                    LogisticRegression(penalty="l1", solver="liblinear", C=c_val,
                                        random_state=random_state, max_iter=5000)
                )),
            ])
            try:
                p_tmp.fit(X_tr_fold, y_tr_fold)
                n_feats_per_fold.append(p_tmp.named_steps["selector"].get_support().sum())
            except ValueError as e:
                if "Found array with 0 feature(s)" in str(e):
                    n_feats_per_fold.append(0)
                else:
                    raise

        cv_results.append({
            "C": c_val, "log_C": np.log10(c_val),
            "AUC_mean": np.nanmean(aucs), "AUC_std": np.nanstd(aucs),
            "N_features": np.mean(n_feats_per_fold),
        })

    cv_df = pd.DataFrame(cv_results)

    n_train = len(X_train)
    cv_df["BIC_AUC"] = cv_df["AUC_mean"] - (cv_df["N_features"] / (2 * n_train)) * np.log(n_train)

    elbow_idx = find_elbow(cv_df["log_C"].values, cv_df["N_features"].values)
    C_elbow = float(cv_df.iloc[elbow_idx]["C"])
    C_best_auc = float(cv_df.loc[cv_df["AUC_mean"].idxmax(), "C"])
    C_best_bic = float(cv_df.loc[cv_df["BIC_AUC"].idxmax(), "C"])

    C_optimal = float(np.exp(np.mean([np.log(C_best_auc), np.log(C_best_bic), np.log(C_elbow)])))
    C_optimal = float(c_grid[np.argmin(np.abs(c_grid - C_optimal))])

    return C_optimal, cv_df


def apply_lasso_selection(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    numeric_features: list[str],
    categorical_features: list[str],
    C_optimal: float,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Áp dụng C* để trích xuất tập đặc trưng gốc cuối cùng (map ngược từ
    one-hot encoded feature names về tên cột gốc).

    Returns
    -------
    (X_train_selected, X_test_selected, ordered_selected_features)
    """
    lasso_optimal = LogisticRegression(
        penalty="l1", solver="liblinear", C=C_optimal,
        random_state=random_state, max_iter=5000,
    )
    pipeline_optimal = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("selector", SelectFromModel(lasso_optimal)),
    ])
    pipeline_optimal.fit(X_train, y_train)

    all_features = preprocessor.get_feature_names_out()
    support_mask = pipeline_optimal.named_steps["selector"].get_support()
    selected_features_raw = all_features[support_mask]

    selected_original_set: set[str] = set()
    sorted_cat_features = sorted(categorical_features, key=len, reverse=True)

    for sf in selected_features_raw:
        clean_name = sf.split("__")[-1] if "__" in sf else sf
        if clean_name in numeric_features:
            selected_original_set.add(clean_name)
        else:
            for oc in sorted_cat_features:
                if clean_name.startswith(oc + "_") or clean_name == oc:
                    selected_original_set.add(oc)
                    break

    ordered_selected_features = [c for c in X_train.columns if c in selected_original_set]

    X_train_selected = X_train[ordered_selected_features]
    X_test_selected = X_test[ordered_selected_features]

    return X_train_selected, X_test_selected, ordered_selected_features
