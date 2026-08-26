"""
Trực quan hoá cấu trúc Bayesian Network (DAG) đã học.

Cung cấp:
  - build_nx_graph_from_bn(): chuyển pgmpy model -> networkx DiGraph
  - get_bn_layout(): layout Graphviz 'dot' (ưu tiên) hoặc spring layout
    (fallback nếu máy chưa cài graphviz/pygraphviz/pydot)
  - get_target_centered_layout(): layout thủ công đặt target ở tâm,
    parents bên trái, children bên phải, spouses (co-parents) phía dưới
  - plot_global_bn_structure(): vẽ toàn bộ DAG

Trích xuất từ notebook gốc (cells 43, 46).
"""

from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from pgmpy.models import DiscreteBayesianNetwork


def build_nx_graph_from_bn(bn_model: DiscreteBayesianNetwork) -> nx.DiGraph:
    """Chuyển đổi pgmpy DiscreteBayesianNetwork sang networkx.DiGraph."""
    G = nx.DiGraph()
    G.add_nodes_from(list(bn_model.nodes()))
    G.add_edges_from(list(bn_model.edges()))
    return G


def wrap_node_labels(nodes, width: int = 14) -> dict[str, str]:
    """Ngắt dòng label dài để hiển thị gọn trong node ellipse."""
    return {node: "\n".join(textwrap.wrap(str(node), width=width)) for node in nodes}


def get_bn_layout(G: nx.DiGraph, seed: int = 42) -> dict:
    """
    Layout phân tầng (hierarchical) bằng Graphviz 'dot' nếu có cài đặt;
    fallback về spring_layout nếu môi trường chưa có graphviz/pygraphviz/pydot.
    """
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        return graphviz_layout(G, prog="dot")
    except Exception:
        pass

    try:
        from networkx.drawing.nx_pydot import graphviz_layout
        return graphviz_layout(G, prog="dot")
    except Exception:
        pass

    return nx.spring_layout(G, seed=seed, k=1.4, iterations=300)


def get_target_centered_layout(G: nx.DiGraph, target: str) -> dict:
    """
    Layout thủ công: target ở tâm (0,0), parents bên trái (x=-3),
    children bên phải (x=3), spouses (co-parents của children) phía dưới.
    Giúp trực quan hoá rõ Markov Blanket của target.
    """
    parents = list(G.predecessors(target))
    children = list(G.successors(target))

    spouses = set()
    for child in children:
        spouses.update(set(G.predecessors(child)))
    spouses = list(spouses - set(parents) - {target})

    pos = {target: (0.0, 0.0)}

    def place_vertical(nodes, x, y_top=1.8, y_bottom=-1.8):
        if not nodes:
            return
        ys = [0.0] if len(nodes) == 1 else np.linspace(y_top, y_bottom, len(nodes))
        for node, y in zip(nodes, ys):
            pos[node] = (x, y)

    place_vertical(parents, x=-3.0)
    place_vertical(children, x=3.0)
    place_vertical(spouses, x=1.2, y_top=-2.2, y_bottom=-4.0)

    remaining_nodes = [n for n in G.nodes() if n not in pos]
    place_vertical(remaining_nodes, x=0.0, y_top=3.0, y_bottom=2.2)

    return pos


def plot_global_bn_structure(
    bn_model: DiscreteBayesianNetwork,
    target: str,
    title: str = "(a) Global Bayesian Network Structure",
    save_path: str | None = None,
    figsize: tuple[float, float] | None = None,
    centered_on_target: bool = False,
):
    """
    Vẽ cấu trúc DAG của Bayesian Network.

    Parameters
    ----------
    centered_on_target : nếu True, dùng get_target_centered_layout() để
        làm nổi bật Markov Blanket của target; nếu False dùng layout
        phân tầng Graphviz/spring mặc định.
    save_path : nếu có, lưu hình ra file (dpi=150) thay vì chỉ hiển thị.
    """
    G = build_nx_graph_from_bn(bn_model)

    if figsize is None:
        n_nodes = len(G.nodes())
        figsize = (max(12, n_nodes * 0.75), max(8, n_nodes * 0.45))

    pos = (
        get_target_centered_layout(G, target)
        if centered_on_target
        else get_bn_layout(G)
    )
    labels = wrap_node_labels(G.nodes())

    node_colors = ["#d7191c" if n == target else "#2c7bb6" for n in G.nodes()]

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1800,
                            edgecolors="white", linewidths=1.5, ax=ax, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color="#888888", arrows=True,
                            arrowsize=15, width=1.4, ax=ax,
                            connectionstyle="arc3,rad=0.05")
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8,
                             font_color="white", font_weight="bold", ax=ax)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Target",
               markerfacecolor="#d7191c", markersize=12),
        Line2D([0], [0], marker="o", color="w", label="Feature",
               markerfacecolor="#2c7bb6", markersize=12),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    return fig, ax
