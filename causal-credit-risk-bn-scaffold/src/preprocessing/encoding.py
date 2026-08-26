"""
Nạp dữ liệu và tiền xử lý cơ bản cho bộ Australian Credit Approval.

Trích xuất và làm sạch từ notebook gốc (cells 5, 8, 13, 14).
"""

from __future__ import annotations

import pandas as pd

# Tên cột chuẩn của bộ Australian Credit Approval (UCI ML Repository)
AUSTRALIAN_COLUMNS = [
    "Gender", "Age", "Debt", "BankCustomer", "EducationLevel", "Ethnicity",
    "YearsEmployed", "CreditHistory", "Employed", "CreditScore",
    "DriversLicense", "Citizen", "ZipCode", "Income", "target",
]


def load_australian_dataset(path: str = "data/raw/australian.dat") -> pd.DataFrame:
    """
    Đọc file .dat gốc (phân tách bởi khoảng trắng), gán tên cột chuẩn.

    Parameters
    ----------
    path : str
        Đường dẫn tới file australian.dat.

    Returns
    -------
    pd.DataFrame
        DataFrame đã gán tên cột, cột 'target' vẫn ở dạng gốc (chưa đảo nhãn).
    """
    df = pd.read_csv(path, header=None, sep=r"\s+")
    df.columns = AUSTRALIAN_COLUMNS
    return df


def flip_target_label(df: pd.DataFrame, target_col: str = "target") -> pd.DataFrame:
    """
    Đảo nhãn target: 1 = Default (rủi ro), 0 = Non-default.

    Bộ Australian gốc mã hoá target ngược so với quy ước credit-risk chuẩn
    (1 = được duyệt vay). Hàm này đảo lại để '1' luôn có nghĩa 'rủi ro/default'.
    """
    df = df.copy()
    df[target_col] = 1 - df[target_col]
    return df


def split_features_target(
    df: pd.DataFrame,
    target_col: str = "target",
    drop_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Tách biến độc lập (X) và biến phụ thuộc (y).

    Parameters
    ----------
    drop_cols : list[str], optional
        Các cột cần loại bỏ khỏi X ngoài target (mặc định: ['ZipCode'] —
        ZipCode không mang ý nghĩa nhân quả trực tiếp và dễ leak thông tin
        định danh cá nhân).
    """
    if drop_cols is None:
        drop_cols = ["ZipCode"]

    X = df.drop(columns=[target_col, *drop_cols])
    y = df[target_col]
    return X, y


def load_and_prepare(path: str = "data/raw/australian.dat") -> tuple[pd.DataFrame, pd.Series]:
    """Pipeline gộp: đọc file -> đảo nhãn -> tách X/y."""
    df = load_australian_dataset(path)
    df = flip_target_label(df)
    return split_features_target(df)
