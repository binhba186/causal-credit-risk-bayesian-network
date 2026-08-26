"""
What-If Scenario Analysis — mô phỏng can thiệp phản thực tế (counterfactual).

Trả lời câu hỏi: "Nếu can thiệp thay đổi feature Xi của một cá nhân,
P(default=1) thay đổi ra sao?"

    delta_P(default=1) = P(default=1 | Xi=s_new, rest) - P(default=1 | original)

Đây là xấp xỉ soft-evidence / posterior update của do-calculus:
    P(Y | do(Xi=s)) != P(Y | Xi=s)  (trong trường hợp tổng quát)
Bayesian Network ở đây thực hiện cập nhật hậu nghiệm (posterior update),
không phải hard do-calculus đầy đủ — cần lưu ý khi diễn giải kết quả.

Trích xuất từ notebook gốc (cell 69).
"""

from __future__ import annotations

import pandas as pd
from pgmpy.inference import VariableElimination


def whatif_scenario(
    infer: VariableElimination,
    base_evidence: dict,
    feature: str,
    target: str,
    state_names: dict[str, list],
    prior_default: float,
    new_states: list | None = None,
) -> tuple[pd.DataFrame, float]:
    """
    Mô phỏng kịch bản: thay đổi `feature` lần lượt về từng state, đo
    delta_P(default) so với baseline (profile gốc `base_evidence`).

    Ứng dụng thực tế: "Nếu borrower tăng Income thêm 1 bracket, rủi ro
    default giảm bao nhiêu?" — hỗ trợ thiết kế can thiệp chính sách tín dụng.

    Parameters
    ----------
    base_evidence : dict {feature: state} — profile gốc của cá nhân
        (nên giữ tối giản, chỉ các biến "persona anchor" không đổi được,
        để tránh evidence quá chi tiết làm sai lệch phản thực tế).
    feature : biến cần thử nghiệm thay đổi.
    new_states : danh sách state cần thử (mặc định: toàn bộ state của feature).

    Returns
    -------
    (result_df, p1_base) : bảng {state, P(default=1), delta, is_baseline}
    và xác suất baseline.
    """
    if new_states is None:
        new_states = state_names.get(feature, [])

    try:
        q_base = infer.query(variables=[target], evidence=base_evidence, show_progress=False)
        p1_base = float(q_base.values[1])
    except Exception:
        p1_base = prior_default

    records = []
    for s in new_states:
        mod_evidence = {**base_evidence, feature: s}
        try:
            q = infer.query(variables=[target], evidence=mod_evidence, show_progress=False)
            p1_new = float(q.values[1])
        except Exception:
            p1_new = prior_default
        records.append({
            "state": s,
            "P(default=1)": p1_new,
            "delta": p1_new - p1_base,
            "is_baseline": (s == base_evidence.get(feature)),
        })

    return pd.DataFrame(records), p1_base


def run_whatif_for_top_features(
    infer: VariableElimination,
    base_evidence_full: dict,
    top_features: list[str],
    target: str,
    state_names: dict[str, list],
    prior_default: float,
    persona_keys: list[str] | None = None,
) -> dict[str, tuple[pd.DataFrame, float]]:
    """
    Chạy what-if cho một danh sách feature (thường là top sensitivity
    features), sử dụng chiến lược "Persona": chỉ giữ lại một tập nhỏ
    biến nhân khẩu học không thể can thiệp (persona_keys, ví dụ Age,
    Citizen) làm evidence cố định, để kịch bản phản thực tế không bị
    "nhiễu" bởi quá nhiều evidence chi tiết từ hồ sơ gốc.

    Returns
    -------
    dict {feature: (result_df, p1_base)}
    """
    if persona_keys is None:
        persona_keys = []

    results: dict[str, tuple[pd.DataFrame, float]] = {}
    for feat in top_features:
        if feat not in base_evidence_full or feat not in state_names:
            continue
        trimmed_evidence = {
            k: v for k, v in base_evidence_full.items()
            if k in persona_keys and k != feat
        }
        df_wi, p_base = whatif_scenario(
            infer=infer, base_evidence=trimmed_evidence, feature=feat,
            target=target, state_names=state_names, prior_default=prior_default,
        )
        results[feat] = (df_wi, p_base)

    return results
