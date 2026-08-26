"""
Baseline ML models + Stacking Ensemble để đối chiếu hiệu năng với Causal BN.

Kiến trúc Stacking (Wolpert, 1992):
  Level-0 (base learners): Random Forest, XGBoost, Logistic Regression
  Level-1 (meta-learner) : Logistic Regression
  Cross-fitting: 5-fold Stratified CV (out-of-fold predictions, tránh
  data leakage giữa base learner và meta-learner).

Trích xuất từ notebook gốc (cells 56, 58).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def build_baseline_models(y_train: pd.Series, n_train: int, random_state: int = 42) -> dict:
    """
    Định nghĩa các baseline model dùng để so sánh với Causal BN.

    - Logistic Regression: L1-penalized, dùng làm linear baseline ổn định.
    - Random Forest: bagging, giảm variance.
    - XGBoost: boosting, giảm bias; scale_pos_weight tự động xử lý
      class imbalance dựa trên tỉ lệ lớp trong y_train.
    - KNN: non-parametric, k = sqrt(n_train), distance-weighted.
    """
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(penalty="l1", solver="liblinear",
                                        C=0.1, random_state=random_state, max_iter=5000)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=10,
            random_state=random_state, n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", random_state=random_state, n_jobs=-1,
        ),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(
                n_neighbors=int(np.sqrt(n_train)), weights="distance",
                metric="euclidean", n_jobs=-1,
            )),
        ]),
    }


def fit_baseline_models(
    models: dict, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame
) -> dict[str, np.ndarray]:
    """
    Huấn luyện toàn bộ baseline models và trả về P(default=1) trên X_test.

    Returns
    -------
    dict {model_name: y_prob (ndarray shape (n_test,))}
    """
    y_probs = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_probs[name] = model.predict_proba(X_test)[:, 1]
    return y_probs


def build_stacking_ensemble(
    y_train: pd.Series, random_state: int = 42, cv_folds: int = 5
) -> StackingClassifier:
    """
    Stacking Ensemble: RF + XGBoost + Logistic Regression (level-0),
    meta-learner Logistic Regression (level-1), cross-fit qua 5-fold CV
    để tạo out-of-fold predictions (tránh leakage base -> meta).
    """
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    base_estimators = [
        ("rf", RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=10,
            random_state=random_state, n_jobs=-1,
        )),
        ("xgb", XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", random_state=random_state, n_jobs=-1,
        )),
        ("lr", Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(penalty="l1", solver="liblinear",
                                        C=0.1, random_state=random_state, max_iter=5000)),
        ])),
    ]

    meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=random_state)

    return StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1,
    )


def fit_and_predict_stacking(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, random_state: int = 42
) -> tuple[StackingClassifier, np.ndarray]:
    """Fit Stacking Ensemble và trả về (model, P(default=1) trên X_test)."""
    stacking_clf = build_stacking_ensemble(y_train, random_state=random_state)
    stacking_clf.fit(X_train, y_train)
    y_prob_stack = stacking_clf.predict_proba(X_test)[:, 1]
    return stacking_clf, y_prob_stack


def get_meta_learner_weights(stacking_clf: StackingClassifier, base_names: list[str]) -> dict[str, float]:
    """
    Trích xuất trọng số (coefficient) mà meta-learner gán cho từng base
    model — cho biết mức độ "tin tưởng" tương đối của ensemble vào mỗi
    base learner.
    """
    meta_coefs = stacking_clf.final_estimator_.coef_[0]
    return dict(sorted(zip(base_names, meta_coefs), key=lambda x: abs(x[1]), reverse=True))
