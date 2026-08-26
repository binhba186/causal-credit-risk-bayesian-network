"""
Rời rạc hoá đặc trưng liên tục bằng K-means (K_Pro) cho Bayesian Network.

Bayesian Network rời rạc (pgmpy DiscreteBayesianNetwork) yêu cầu mọi biến
ở dạng categorical/ordinal. Module này chuyển các đặc trưng liên tục
(Income, Debt, Age...) sang các bin rời rạc bằng K-means, sau khi chuẩn hoá
phân phối bằng Power Transform (Yeo-Johnson) để giảm lệch (skewness).

Trích xuất từ notebook gốc (cells 34, 35, 36).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer, PowerTransformer


def discretize_kmeans(
    X_train_selected: pd.DataFrame,
    X_test_selected: pd.DataFrame,
    continuous_cols: list[str],
    n_bins: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chuẩn hoá (Yeo-Johnson) rồi rời rạc hoá (K-means binning) các cột liên tục.

    Parameters
    ----------
    X_train_selected, X_test_selected : DataFrame đã qua bước chọn đặc trưng.
    continuous_cols : danh sách cột số liên tục cần rời rạc hoá.
    n_bins : số bin K-means cho mỗi biến (mặc định 4).

    Returns
    -------
    (X_train_kpro, X_test_kpro) : DataFrame toàn bộ cột ở dạng int,
    sẵn sàng cho pgmpy.
    """
    pt = PowerTransformer(method="yeo-johnson")
    kbins = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="kmeans")

    X_train_kpro = X_train_selected.copy()
    X_test_kpro = X_test_selected.copy()

    if continuous_cols:
        # Bước 1: chuẩn hoá phân phối (không mất dữ liệu như capping)
        X_train_kpro[continuous_cols] = pt.fit_transform(X_train_selected[continuous_cols])
        X_test_kpro[continuous_cols] = pt.transform(X_test_selected[continuous_cols])

        # Bước 2: phân cụm K-means trên không gian đã chuẩn hoá
        X_train_kpro[continuous_cols] = kbins.fit_transform(X_train_kpro[continuous_cols])
        X_test_kpro[continuous_cols] = kbins.transform(X_test_kpro[continuous_cols])

    # pgmpy yêu cầu không có float
    X_train_kpro = X_train_kpro.round().astype(int)
    X_test_kpro = X_test_kpro.round().astype(int)

    assert X_train_kpro.isnull().sum().sum() == 0, "Còn NaN sau discretize!"
    assert (X_train_kpro < 0).sum().sum() == 0, "Còn giá trị âm sau discretize!"

    return X_train_kpro, X_test_kpro


def build_bin_labels_map(
    X_train_selected: pd.DataFrame,
    X_train_kpro: pd.DataFrame,
    continuous_cols: list[str],
) -> dict[str, dict[float, str]]:
    """
    Xây dựng ánh xạ bin -> khoảng giá trị gốc thực tế (không dùng inverse
    transform vì dễ sai số). Dùng min/max thực tế của từng bin, đảm bảo
    liên tục giữa các bin liền kề.

    Returns
    -------
    dict {feature: {bin_id: "lo ~ hi"}}
    """
    bin_labels_map: dict[str, dict[float, str]] = {}

    for col in continuous_cols:
        x_orig = X_train_selected[col].values
        x_binned = X_train_kpro[col].values
        is_int = (
            pd.api.types.is_integer_dtype(X_train_selected[col])
            or np.all(x_orig == x_orig.astype(int))
        )

        def fmt(v: float) -> str:
            return str(int(round(v))) if is_int else f"{v:.2f}"

        bin_ids = sorted(np.unique(x_binned))

        bounds: dict[float, list[float]] = {}
        for b in bin_ids:
            mask = x_binned == b
            bounds[b] = [x_orig[mask].min(), x_orig[mask].max()]

        # Khớp liên tục: max(bin k) = min(bin k+1)
        for i in range(len(bin_ids) - 1):
            b_cur, b_next = bin_ids[i], bin_ids[i + 1]
            bounds[b_cur][1] = bounds[b_next][0]

        bin_labels_map[col] = {}
        for b in bin_ids:
            lo, hi = bounds[b]
            label = f"= {fmt(lo)}" if lo == hi else f"{fmt(lo)} ~ {fmt(hi)}"
            bin_labels_map[col][float(b)] = label

    return bin_labels_map
