"""
Structure Learning cho Bayesian Network: Hill-Climb + Tabu Search + BIC score.

Học cấu trúc DAG G* tối ưu điểm số BIC:
    G* = argmax_G  BIC(G, D) = log L(theta_hat_G | D) - (log N / 2) * dim(G)

Chạy nhiều lần (N_RUNS) với random-start DAG khác nhau để tránh hội tụ
vào local optimum, giữ lại DAG có BIC cao nhất.

Trích xuất từ notebook gốc (cell 39).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pgmpy.base import DAG
from pgmpy.estimators import HillClimbSearch

try:
    from pgmpy.estimators import BIC
except ImportError:  # pragma: no cover - tương thích phiên bản pgmpy cũ
    from pgmpy.estimators.StructureScore import BIC

from pgmpy.models import DiscreteBayesianNetwork


def make_random_start_dag(
    nodes: list[str],
    rng: np.random.Generator,
    edge_prob: float = 0.3,
    max_indegree: int = 5,
) -> DAG:
    """
    Tạo DAG ngẫu nhiên làm điểm khởi đầu cho mỗi lần chạy Tabu Search.

    Random start giúp Tabu Search khám phá nhiều vùng khác nhau của
    không gian cấu trúc, giảm rủi ro hội tụ vào local optimum.
    Đảm bảo tính acyclic: chỉ nối cạnh theo thứ tự topo ngẫu nhiên.
    """
    nodes = list(nodes)
    order = nodes.copy()
    rng.shuffle(order)

    dag = DAG()
    dag.add_nodes_from(nodes)
    indegree = {node: 0 for node in nodes}

    for i, parent in enumerate(order):
        for child in order[i + 1:]:
            if indegree[child] >= max_indegree:
                continue
            if rng.random() < edge_prob:
                dag.add_edge(parent, child)
                indegree[child] += 1

    return dag


def learn_structure(
    train_data_cat: pd.DataFrame,
    state_names: dict[str, list],
    n_runs: int = 50,
    tabu_length: int = 100,
    max_indegree: int = 5,
    edge_prob: float = 0.3,
    max_iter: int = 1000,
    random_state: int = 42,
) -> tuple[DAG, float, pd.DataFrame]:
    """
    Chạy N_RUNS lần Hill-Climb + Tabu Search với random-start DAG khác nhau,
    chấm điểm mỗi kết quả bằng BIC, giữ lại DAG tốt nhất.

    Parameters
    ----------
    train_data_cat : DataFrame dạng category (bao gồm cả cột target).
    state_names : dict {column: [state1, state2, ...]}.

    Returns
    -------
    (best_dag, best_bic, training_summary_df)
    """
    rng = np.random.default_rng(random_state)

    hc = HillClimbSearch(train_data_cat, state_names=state_names)
    bic_score = BIC(train_data_cat, state_names=state_names)

    best_dag: DAG | None = None
    best_bic = -np.inf
    training_records = []

    for run in range(n_runs):
        start_dag = make_random_start_dag(
            nodes=list(train_data_cat.columns), rng=rng,
            edge_prob=edge_prob, max_indegree=max_indegree,
        )

        learned_dag = hc.estimate(
            scoring_method="bic-d",
            start_dag=start_dag,
            tabu_length=tabu_length,
            max_indegree=max_indegree,
            max_iter=max_iter,
            show_progress=(run == 0),
        )

        candidate_bn = DiscreteBayesianNetwork()
        candidate_bn.add_nodes_from(train_data_cat.columns)
        candidate_bn.add_edges_from(learned_dag.edges())
        score = bic_score.score(candidate_bn)

        training_records.append({
            "run": run + 1, "bic_score": score,
            "n_edges": len(candidate_bn.edges()),
            "edges": list(candidate_bn.edges()),
        })

        if score > best_bic:
            best_bic = score
            best_dag = learned_dag

    training_summary_df = (
        pd.DataFrame(training_records)
        .sort_values("bic_score", ascending=False)
        .reset_index(drop=True)
    )

    assert best_dag is not None, "Structure learning thất bại: không có DAG hợp lệ."
    return best_dag, best_bic, training_summary_df
