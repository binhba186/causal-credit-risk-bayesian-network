"""
Parameter Learning: ước lượng bảng xác suất có điều kiện (CPD) bằng MLE.

theta*_ijk = m_ijk / sum_k(m_ijk)   (frequency estimate)
theta_ijk  = P(Xi = k | pa(Xi) = j)

Trích xuất từ notebook gốc (cell 41).
"""

from __future__ import annotations

from pgmpy.base import DAG
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import DiscreteBayesianNetwork


def build_and_fit_bn(
    dag: DAG,
    train_data_cat,
    state_names: dict[str, list] | None = None,
) -> DiscreteBayesianNetwork:
    """
    Khởi tạo DiscreteBayesianNetwork từ DAG đã học (structure_learning.py),
    sau đó học tham số CPD bằng Maximum Likelihood Estimation.

    Parameters
    ----------
    dag : DAG đã học từ learn_structure().
    train_data_cat : DataFrame category chứa cả target.
    state_names : dict {column: [states]}, giúp đảm bảo mọi state được
        khai báo dù không xuất hiện đủ trong dữ liệu train.

    Returns
    -------
    DiscreteBayesianNetwork đã fit đầy đủ CPDs, đã qua check_model().
    """
    bn_model = DiscreteBayesianNetwork()
    bn_model.add_nodes_from(dag.nodes())
    bn_model.add_edges_from(dag.edges())

    try:
        mle = MaximumLikelihoodEstimator(model=bn_model, data=train_data_cat, state_names=state_names)
    except TypeError:
        mle = MaximumLikelihoodEstimator(model=bn_model, data=train_data_cat)

    cpds = mle.get_parameters()
    bn_model.add_cpds(*cpds)

    assert bn_model.check_model(), "BN model không hợp lệ. Kiểm tra lại DAG hoặc CPDs."
    return bn_model
