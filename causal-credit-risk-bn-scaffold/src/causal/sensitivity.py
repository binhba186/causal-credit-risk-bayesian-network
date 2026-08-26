"""
Sensitivity Analysis + Backward (Diagnostic) Inference cho Bayesian Network.

- Sensitivity: đo mức thay đổi P(target=1) khi ép một feature về từng state.
    delta(Xi=s) = P(target=1 | Xi=s) - P(target=1)_prior
  Feature có max|delta| lớn nhất = causal driver quan trọng nhất.

- Backward inference: P(Xi | target=class) — suy diễn ngược từ kết quả
  quan sát về phân phối đặc trưng, dùng Likelihood Ratio để tìm risk
  discriminator.
    LR(Xi=s) = P(Xi=s | default=1) / P(Xi=s | default=0)

Trích xuất từ notebook gốc (cells 61, 65).
"""

from __future__ import annotations

import pandas as pd
from pgmpy.inference import VariableElimination


def sensitivity_analysis(
    infer: VariableElimination,
    features_to_analyze: list[str],
    state_names: dict[str, list],
    target: str,
    prior_default: float,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Tính delta(Xi=s) cho từng feature/state trong danh sách (thường là
    Markov Blanket của target).

    Returns
    -------
    (sensitivity_df, feat_sensitivity)
        sensitivity_df    : bảng chi tiết theo từng (feature, state).
        feat_sensitivity  : Series max|delta| theo feature, sort giảm dần.
    """
    records = []
    for feat in features_to_analyze:
        if feat == target or feat not in state_names:
            continue
        for state in state_names[feat]:
            try:
                q = infer.query(variables=[target], evidence={feat: state}, show_progress=False)
                p1 = float(q.values[1])
                records.append({
                    "feature": feat, "state": state,
                    "P(default=1|feat)": p1,
                    "delta": p1 - prior_default,
                    "abs_delta": abs(p1 - prior_default),
                })
            except Exception:
                continue

    sensitivity_df = pd.DataFrame(records)
    if sensitivity_df.empty:
        return sensitivity_df, pd.Series(dtype=float)

    feat_sensitivity = (
        sensitivity_df.groupby("feature")["abs_delta"].max().sort_values(ascending=False)
    )
    return sensitivity_df, feat_sensitivity


def backward_inference(
    infer: VariableElimination,
    target: str,
    target_value: int,
    query_features: list[str],
    state_names: dict[str, list],
) -> dict[str, dict]:
    """
    Suy diễn ngược P(Xi | target=target_value) cho danh sách feature.

    Returns
    -------
    dict {feature: {state: probability}}
    """
    results: dict[str, dict] = {}
    for feat in query_features:
        if feat == target:
            continue
        try:
            q = infer.query(variables=[feat], evidence={target: target_value}, show_progress=False)
            results[feat] = {
                st: float(q.values[i]) for i, st in enumerate(state_names.get(feat, []))
            }
        except Exception:
            results[feat] = {}
    return results


def likelihood_ratio_table(
    bd_default: dict[str, dict],
    bd_nondefault: dict[str, dict],
    state_names: dict[str, list],
) -> pd.DataFrame:
    """
    Tổng hợp bảng Likelihood Ratio LR(Xi=s) = P(Xi=s|default=1) / P(Xi=s|default=0)
    từ kết quả backward_inference() cho hai class.

    LR > 1: state gặp nhiều hơn khi default -> risk discriminator.
    """
    rows = []
    for feat in bd_default:
        if feat not in bd_nondefault:
            continue
        for st in state_names.get(feat, []):
            p1 = bd_default[feat].get(st, 0)
            p0 = bd_nondefault[feat].get(st, 0)
            lr = p1 / p0 if p0 > 1e-6 else float("inf")
            rows.append({
                "feature": feat, "state": st,
                "P(.|default=1)": p1, "P(.|default=0)": p0, "LR": lr,
            })
    return pd.DataFrame(rows)
