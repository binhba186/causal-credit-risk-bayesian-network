"""
Lọc đặc trưng bằng Information Value (IV) & Weight of Evidence (WoE).

Tiêu chuẩn vàng trong Credit Scoring để loại bỏ biến yếu, dùng như bước
lọc bổ sung sau Lasso selection (module lasso_selector.py).

Trích xuất từ notebook gốc (cell 32).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Ngưỡng phân loại sức mạnh biến theo IV (chuẩn ngành Credit Scoring)
IV_THRESHOLDS = {
    "useless": 0.02,     # IV < 0.02: vô dụng
    "weak": 0.04,        # 0.02 <= IV < 0.04: yếu
    "medium": 0.3,       # 0.05 <= IV < 0.3: trung bình
    "strong": 0.5,       # 0.3 <= IV < 0.5: mạnh
}                        # IV >= 0.5: cực mạnh (cảnh báo dễ leakage)


def calculate_woe_iv(dataset: pd.DataFrame, feature: str, target: str) -> float:
    """
    Tính Information Value (IV) tổng của một biến.

    Biến liên tục (>10 giá trị duy nhất) được tự động chia thành 10
    quantile bins (decile) trước khi tính WoE/IV.

    IV(feature) = Σ (%Goods - %Bads) * WoE
    WoE(bin)    = ln(%Goods / %Bads)

    Sử dụng Laplace smoothing (+0.5) để tránh chia cho 0 / log(0).
    """
    df = dataset[[feature, target]].copy()

    if pd.api.types.is_numeric_dtype(df[feature]) and df[feature].nunique() > 10:
        df[feature] = pd.qcut(df[feature], q=10, duplicates="drop")

    grouped = df.groupby(feature, observed=False)[target].agg(["count", "sum"])
    grouped.columns = ["Total", "Bads"]
    grouped["Goods"] = grouped["Total"] - grouped["Bads"]

    grouped["Goods"] = grouped["Goods"].replace(0, 0.5)
    grouped["Bads"] = grouped["Bads"].replace(0, 0.5)

    total_goods = grouped["Goods"].sum()
    total_bads = grouped["Bads"].sum()

    grouped["%Goods"] = grouped["Goods"] / total_goods
    grouped["%Bads"] = grouped["Bads"] / total_bads

    grouped["WoE"] = np.log(grouped["%Goods"] / grouped["%Bads"])
    grouped["IV"] = (grouped["%Goods"] - grouped["%Bads"]) * grouped["WoE"]

    return grouped["IV"].sum()


def filter_features_by_iv(
    X_train_selected: pd.DataFrame,
    y_train: pd.Series,
    candidate_features: list[str],
    min_iv_threshold: float = 0.04,
) -> tuple[list[str], pd.DataFrame]:
    """
    Tính IV cho từng candidate feature (thường là output của Lasso selection)
    và loại các biến có IV dưới ngưỡng tối thiểu.

    Returns
    -------
    (final_features, iv_df) : danh sách feature giữ lại, và bảng IV đầy đủ
    (sắp xếp giảm dần) để tham khảo/log.
    """
    df_iv_calc = X_train_selected.copy()
    df_iv_calc["target_for_iv"] = y_train.values

    iv_results = []
    for feat in candidate_features:
        try:
            iv_val = calculate_woe_iv(df_iv_calc, feat, "target_for_iv")
            iv_results.append({"Feature": feat, "IV": iv_val})
        except Exception:
            continue

    iv_df = pd.DataFrame(iv_results).sort_values(by="IV", ascending=False).reset_index(drop=True)
    final_features = iv_df.loc[iv_df["IV"] >= min_iv_threshold, "Feature"].tolist()

    return final_features, iv_df
