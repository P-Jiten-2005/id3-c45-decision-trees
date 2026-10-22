"""Graphviz rendering of trees and Matplotlib charts for evaluation."""
from __future__ import annotations

import graphviz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .tree import DecisionTreeBase, Node, edge_label  # noqa: E402

# Colour-blind-safe palette (Okabe–Ito) used for class colours.
CLASS_COLOURS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
                 "#D55E00", "#F0E442", "#999999"]
MODEL_COLOURS = {"ID3": "#0072B2", "C4.5": "#E69F00",
                 "CART (scikit-learn)": "#999999"}


def tree_to_graphviz(model: DecisionTreeBase, max_depth: int | None = None,
                     show_distribution: bool = True) -> graphviz.Digraph:
    """Build a Graphviz ``Digraph`` of a fitted tree.

    Internal nodes show the tested attribute and its gain (and gain ratio for
    C4.5); leaves show the predicted class and are filled with that class's
    colour, with opacity proportional to the leaf's purity. ``max_depth``
    truncates very large trees for display (cut subtrees appear as "…").
    """
    model._check_fitted()
    dot = graphviz.Digraph(model.algorithm_name, format="svg")
    dot.attr("graph", rankdir="TB", nodesep="0.25", ranksep="0.45",
             fontname="Helvetica", bgcolor="transparent")
    dot.attr("node", fontname="Helvetica", fontsize="11", style="rounded,filled",
             shape="box", color="#555555", penwidth="1")
    dot.attr("edge", fontname="Helvetica", fontsize="10", color="#777777",
             fontcolor="#333333")
    classes = [str(c) for c in model.classes_]

    def dist_text(node: Node) -> str:
        parts = [f"{c}: {v:g}" if float(v).is_integer() else f"{c}: {v:.1f}"
                 for c, v in zip(classes, node.class_dist) if v > 1e-9]
        return ", ".join(parts)

    def add(node: Node):
        nid = f"n{node.node_id}"
        if node.is_leaf or (max_depth is not None and node.depth >= max_depth):
            cls = node.majority
            purity = node.class_dist.max() / max(node.n_samples, 1e-12)
            colour = CLASS_COLOURS[cls % len(CLASS_COLOURS)]
            alpha = int(40 + 180 * purity)
            label = f"<<b>{_esc(classes[cls])}</b>"
            label += f"<br/><font point-size='9'>n = {node.n_samples:.4g}</font>"
            if show_distribution:
                label += f"<br/><font point-size='9'>{_esc(dist_text(node))}</font>"
            if not node.is_leaf:
                label += "<br/><font point-size='9'><i>… subtree hidden</i></font>"
            label += ">"
            dot.node(nid, label, fillcolor=f"{colour}{alpha:02x}", shape="ellipse")
            return
        label = f"<<b>{_esc(node.feature)}</b>"
        if node.threshold is not None:
            label += "<br/><font point-size='9'>(continuous split)</font>"
        stats = f"gain = {node.gain:.3f}"
        if node.gain_ratio is not None:
            stats += f", ratio = {node.gain_ratio:.3f}"
        label += f"<br/><font point-size='9'>{stats}</font>"
        label += f"<br/><font point-size='9'>n = {node.n_samples:.4g}</font>>"
        dot.node(nid, label, fillcolor="#F2F2F2")
        for key, child in node.children.items():
            add(child)
            dot.edge(nid, f"n{child.node_id}", label=_esc(edge_label(node, key)))

    add(model.root_)
    return dot


def render_tree_png(model: DecisionTreeBase, path: str, max_depth: int | None = None) -> str:
    """Render to a file with the Graphviz ``dot`` binary (must be installed)."""
    dot = tree_to_graphviz(model, max_depth=max_depth)
    dot.format = "png"
    dot.attr("graph", dpi="150", bgcolor="white")
    return dot.render(path.removesuffix(".png"), cleanup=True)


def _esc(text) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


# ------------------------------------------------------------------ charts
def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="#333333")
    ax.yaxis.label.set_color("#333333")
    ax.xaxis.label.set_color("#333333")


def plot_confusion_matrix(cm: np.ndarray, labels: list, title: str, cmap: str = "Blues"):
    n = len(labels)
    size = max(3.6, 0.55 * n + 2.4)
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    ax.imshow(cm, cmap=cmap)
    ax.set_xticks(range(n), labels=labels, rotation=45 if n > 4 else 0,
                  ha="right" if n > 4 else "center")
    ax.set_yticks(range(n), labels=labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(title, fontsize=11, loc="left")
    vmax = cm.max() if cm.size else 0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=10,
                    color="white" if cm[i, j] > vmax / 2 else "#222222")
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    return fig


def plot_metric_comparison(table: pd.DataFrame):
    """Grouped bars: one group per metric, one bar per model.

    ``table`` has models as rows and metric names as columns.
    """
    metrics = list(table.columns)
    models = list(table.index)
    x = np.arange(len(metrics))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for i, m in enumerate(models):
        vals = table.loc[m].to_numpy(dtype=float)
        bars = ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width * 0.92,
                      label=m, color=MODEL_COLOURS.get(m, CLASS_COLOURS[i % 8]))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8, color="#333333")
    ax.set_xticks(x, labels=metrics)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.legend(frameon=False, ncol=len(models), loc="upper center",
              bbox_to_anchor=(0.5, 1.15))
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_class_distribution(y: pd.Series):
    counts = y.astype(str).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, max(1.8, 0.35 * len(counts) + 0.8)))
    ax.barh(counts.index, counts.values, color="#0072B2")
    for i, v in enumerate(counts.values):
        ax.text(v, i, f" {v}", va="center", fontsize=9, color="#333333")
    ax.invert_yaxis()
    ax.set_xlabel("Rows")
    _style(ax)
    fig.tight_layout()
    return fig


def plot_feature_importance(series_by_model: dict[str, pd.Series]):
    df = pd.DataFrame(series_by_model).fillna(0.0)
    df = df.loc[df.max(axis=1).sort_values().index]
    df = df.tail(15)
    fig, ax = plt.subplots(figsize=(7, max(2.2, 0.35 * len(df) + 1)))
    y = np.arange(len(df))
    h = 0.8 / len(df.columns)
    for i, m in enumerate(df.columns):
        ax.barh(y + (i - (len(df.columns) - 1) / 2) * h, df[m], h * 0.92, label=m,
                color=MODEL_COLOURS.get(m, CLASS_COLOURS[i % 8]))
    ax.set_yticks(y, labels=df.index)
    ax.set_xlabel("Share of total information gain")
    ax.legend(frameon=False, loc="lower right")
    _style(ax)
    fig.tight_layout()
    return fig
