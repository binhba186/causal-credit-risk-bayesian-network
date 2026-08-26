"""
Exact Inference bằng Variable Elimination (VE) trên Bayesian Network đã học.

Thuật toán VE:
    P(Q | E=e) ~ Sum_hidden Prod_i phi_i(Xi, pa(Xi))
    1. Lấy tất cả factor (CPD) từ BN
    2. Gán evidence E=e
    3. Lần lượt eliminate từng biến ẩn X: nhóm factor chứa X, nhân lại,
       marginalize (sum-out) X
    4. Nhân các factor còn lại -> h(Q)
    5. Normalize: P(Q|E=e) = h(Q) / Sum_Q h(Q)

Trích xuất từ notebook gốc (cells 42, 51, 57).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork


def build_inference_engine(bn_model: DiscreteBayesianNetwork) -> VariableElimination:
    """Khởi tạo engine Variable Elimination cho exact inference."""
    return VariableElimination(bn_model)


def get_markov_blanket_info(
    bn_model: DiscreteBayesianNetwork, target: str
) -> tuple[list[str], list[str], list[str]]:
    """
    Markov Blanket(target) = parents U children U parents-of-children.
    Đây là tập biến tối thiểu đủ để dự đoán target (d-separates target
    khỏi phần còn lại của mạng).

    Returns
    -------
    (parents, children, markov_blanket)
    """
    parents = list(bn_model.get_parents(target))
    children = list(bn_model.get_children(target))
    markov_blanket = list(bn_model.get_markov_blanket(target))
    return parents, children, markov_blanket


def nearest_valid_state(value, valid_states: list) -> int:
    """
    Ánh xạ L1-nearest: đưa giá trị về state gần nhất trong không gian state
    đã biết. Giải quyết vấn đề Out-Of-Vocabulary (OOV) khi giá trị test
    không xuất hiện trong tập train.
    """
    value = int(round(value))
    return value if value in valid_states else min(valid_states, key=lambda s: abs(s - value))


def prepare_bn_features(
    X_raw: pd.DataFrame | np.ndarray,
    feature_cols: list[str],
    state_names: dict[str, list],
) -> pd.DataFrame:
    """Chuẩn hoá input: chọn đúng cột, map OOV values, ép kiểu category."""
    X = pd.DataFrame(X_raw, columns=feature_cols) if isinstance(X_raw, np.ndarray) else X_raw.copy()
    X = X[feature_cols]
    for col in feature_cols:
        X[col] = X[col].apply(lambda v: nearest_valid_state(v, state_names[col]))
    return X.astype("category")


def bn_predict_proba(
    bn_model: DiscreteBayesianNetwork,
    X_raw: pd.DataFrame | np.ndarray,
    feature_cols: list[str],
    state_names: dict[str, list],
    target: str,
    prior_default: float,
) -> np.ndarray:
    """
    Batch Variable Elimination prediction: P(target=0), P(target=1) cho
    toàn bộ dataset qua pgmpy's `predict_probability` (batch VE).
    Fallback về prior khi evidence OOV không xử lý được.

    Returns
    -------
    ndarray shape (n_samples, 2) : [P(target=0), P(target=1)]
    """
    X_bn = prepare_bn_features(X_raw, feature_cols, state_names)
    probs = bn_model.predict_probability(X_bn)

    t1_col, t0_col = f"{target}_1", f"{target}_0"
    probs[t1_col] = probs[t1_col].fillna(prior_default)
    if t0_col not in probs.columns:
        probs[t0_col] = 1 - probs[t1_col]
    probs[t0_col] = probs[t0_col].fillna(1 - prior_default)

    return probs[[t0_col, t1_col]].values


def query_single_sample_proba(
    infer: VariableElimination,
    evidence: dict,
    target: str,
    prior_default: float,
) -> float:
    """
    Query VE cho một mẫu đơn lẻ (dùng cho What-If / sensitivity analysis
    nơi cần truy vấn lặp lại với evidence thay đổi liên tục).

    Trả về P(target=1 | evidence), fallback về prior nếu query lỗi
    (ví dụ CPD thiếu do tổ hợp evidence hiếm gặp).
    """
    try:
        q = infer.query(variables=[target], evidence=evidence, show_progress=False)
        states = list(q.state_names[target])
        idx1 = states.index(1) if 1 in states else states.index("1")
        return float(q.values[idx1])
    except Exception:
        return float(prior_default)
