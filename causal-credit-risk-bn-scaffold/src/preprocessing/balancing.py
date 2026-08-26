"""
Cân bằng dữ liệu mất cân bằng lớp (class imbalance) bằng SMOTENC.

Trích xuất từ notebook gốc (cells 23, 24) — S_Pro trong pipeline.
SMOTENC được chọn thay vì SMOTE thường vì dataset có trộn lẫn
biến liên tục (Income, Debt...) và biến định tính (Gender, Employed...).
"""

from __future__ import annotations

import pandas as pd
from imblearn.over_sampling import SMOTENC


DEFAULT_CATEGORICAL_COLS = [
    "Gender", "BankCustomer", "EducationLevel", "Ethnicity",
    "CreditHistory", "Employed", "DriversLicense", "Citizen",
]


def balance_with_smotenc(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    categorical_cols: list[str] | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Áp dụng SMOTENC để oversample lớp thiểu số trên tập train.

    QUAN TRỌNG: chỉ fit trên tập train, không bao giờ áp dụng lên
    tập test/validation để tránh data leakage.

    Parameters
    ----------
    X_train, y_train : dữ liệu huấn luyện gốc (trước cân bằng).
    categorical_cols : list[str], optional
        Danh sách cột định tính. Mặc định dùng bộ cột của Australian dataset.

    Returns
    -------
    (X_train_balanced, y_train_balanced)
    """
    if categorical_cols is None:
        categorical_cols = DEFAULT_CATEGORICAL_COLS

    cat_idx = [X_train.columns.get_loc(col) for col in categorical_cols]

    smote = SMOTENC(categorical_features=cat_idx, random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res
