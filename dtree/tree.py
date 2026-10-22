"""Tree data structure and the behaviour shared by ID3 and C4.5.

Both algorithms grow the same kind of tree:

* a categorical test creates one branch per observed value (multi-way split);
* a continuous test (C4.5 only) creates two branches, ``<= t`` and ``> t``.

Every node stores the weighted class distribution of the training samples that
reached it, so any node can act as a leaf (used for unseen categories, missing
values at prediction time, and pruning).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

LE = "<="
GT = ">"


@dataclass
class Node:
    """One node of a decision tree."""

    class_dist: np.ndarray                     # weighted class counts at this node
    depth: int = 0
    feature: str | None = None                 # attribute tested (None for a leaf)
    threshold: float | None = None             # set only for continuous tests
    children: dict[Any, "Node"] = field(default_factory=dict)
    branch_weights: dict[Any, float] = field(default_factory=dict)
    gain: float = 0.0                          # information gain of the chosen split
    gain_ratio: float | None = None            # C4.5 only
    node_id: int = 0

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def n_samples(self) -> float:
        return float(self.class_dist.sum())

    @property
    def majority(self) -> int:
        return int(np.argmax(self.class_dist))

    def make_leaf(self) -> None:
        self.feature = None
        self.threshold = None
        self.children = {}
        self.branch_weights = {}
        self.gain = 0.0
        self.gain_ratio = None


class DecisionTreeBase:
    """Common fit/predict plumbing. Subclasses implement ``_build``."""

    algorithm_name = "DecisionTree"
    _fallback_description = "predicted from this node's class distribution"

    def __init__(self, max_depth: int | None = None, min_samples_split: int = 2,
                 min_gain: float = 1e-7):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.root_: Node | None = None
        self.classes_: np.ndarray | None = None
        self.feature_names_: list[str] = []
        self.feature_types_: dict[str, str] = {}
        self.fit_time_: float = 0.0
        self._next_id = 0

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray,
            feature_types: dict[str, str] | None = None) -> "DecisionTreeBase":
        """Grow the tree.

        Parameters
        ----------
        X : DataFrame of features.
        y : class labels (any hashable values).
        feature_types : optional ``{column: "categorical" | "continuous"}``.
            Columns not listed are inferred from their dtype.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = X.reset_index(drop=True)
        y = pd.Series(np.asarray(y)).reset_index(drop=True)
        if len(X) != len(y):
            raise ValueError("X and y have different numbers of rows")
        if len(X) == 0:
            raise ValueError("Cannot fit a tree on an empty dataset")

        self.feature_names_ = list(X.columns)
        self.feature_types_ = infer_feature_types(X, feature_types)
        self._validate_feature_types()
        self.classes_, y_enc = np.unique(y if pd.api.types.is_numeric_dtype(y) else y.astype(str),
                                         return_inverse=True)
        self._n_classes = len(self.classes_)
        self._next_id = 0

        start = time.perf_counter()
        weights = np.ones(len(X), dtype=float)
        self.root_ = self._build(X, y_enc, weights, list(self.feature_names_), depth=0)
        self._post_process()
        self.fit_time_ = time.perf_counter() - start
        return self

    def _validate_feature_types(self) -> None:  # overridden by ID3
        pass

    def _post_process(self) -> None:            # overridden by C4.5 (pruning)
        pass

    def _build(self, X, y, w, features, depth) -> Node:  # pragma: no cover
        raise NotImplementedError

    def _new_node(self, dist: np.ndarray, depth: int) -> Node:
        node = Node(class_dist=dist, depth=depth, node_id=self._next_id)
        self._next_id += 1
        return node

    def _class_dist(self, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        return np.bincount(y, weights=w, minlength=self._n_classes).astype(float)

    def _should_stop(self, dist: np.ndarray, depth: int, features: list[str]) -> bool:
        total = dist.sum()
        return (
            np.count_nonzero(dist > 1e-9) <= 1
            or not features
            or (self.max_depth is not None and depth >= self.max_depth)
            or total < self.min_samples_split
        )

    # -------------------------------------------------------------- predict
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        X = self._align(X)
        rows = X.to_dict(orient="records")
        return np.vstack([self._proba_row(self.root_, r) for r in rows])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def _proba_row(self, node: Node, row: dict) -> np.ndarray:
        """Route one sample to a leaf and return its class probabilities.

        If the tested value is missing (or a category never seen during
        training) the sample is sent down *every* branch and the results are
        averaged using the training weight of each branch, which is how C4.5
        classifies incomplete cases. ID3 falls back to the node's own
        distribution for unseen categories.
        """
        while not node.is_leaf:
            value = row.get(node.feature)
            child = self._route(node, value)
            if child is None:
                return self._distribute(node, row)
            node = child
        return _normalise(node.class_dist)

    def _route(self, node: Node, value) -> Node | None:
        if _is_missing(value):
            return None
        if node.threshold is not None:
            try:
                v = float(value)
            except (TypeError, ValueError):
                return None
            return node.children[LE] if v <= node.threshold else node.children[GT]
        return node.children.get(_as_key(value))

    def _distribute(self, node: Node, row: dict) -> np.ndarray:
        return _normalise(node.class_dist)

    def _align(self, X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_)
        missing = [c for c in self.feature_names_ if c not in X.columns]
        if missing:
            raise ValueError(f"Input is missing feature columns: {missing}")
        return X[self.feature_names_].reset_index(drop=True)

    def _check_fitted(self) -> None:
        if self.root_ is None:
            raise RuntimeError("The tree has not been fitted yet; call fit() first")

    # ---------------------------------------------------------- inspection
    def iter_nodes(self):
        stack = [self.root_] if self.root_ else []
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(list(node.children.values())))

    @property
    def n_nodes(self) -> int:
        return sum(1 for _ in self.iter_nodes())

    @property
    def n_leaves(self) -> int:
        return sum(1 for n in self.iter_nodes() if n.is_leaf)

    @property
    def depth(self) -> int:
        return max((n.depth for n in self.iter_nodes()), default=0)

    def feature_importances(self) -> pd.Series:
        """Sum of weighted information gain contributed by each attribute."""
        self._check_fitted()
        total = self.root_.n_samples
        imp = dict.fromkeys(self.feature_names_, 0.0)
        for node in self.iter_nodes():
            if not node.is_leaf:
                imp[node.feature] += node.gain * node.n_samples / total
        s = pd.Series(imp, dtype=float)
        if s.sum() > 0:
            s = s / s.sum()
        return s.sort_values(ascending=False)

    def to_rules(self) -> list[str]:
        """Flatten the tree into IF ... THEN ... rules (one per leaf)."""
        self._check_fitted()
        rules: list[str] = []

        def walk(node: Node, conds: list[str]):
            if node.is_leaf:
                cls = self.classes_[node.majority]
                conf = node.class_dist.max() / max(node.n_samples, 1e-12)
                body = " AND ".join(conds) if conds else "TRUE"
                rules.append(f"IF {body} THEN class = {cls}  "
                             f"[n={node.n_samples:.1f}, confidence={conf:.2f}]")
                return
            for key, child in node.children.items():
                walk(child, conds + [edge_label(node, key, with_feature=True)])

        walk(self.root_, [])
        return rules

    def decision_path(self, row: dict) -> list[str]:
        """Human-readable list of tests a single sample passes through."""
        self._check_fitted()
        steps, node = [], self.root_
        while not node.is_leaf:
            child = self._route(node, row.get(node.feature))
            if child is None:
                steps.append(f"{node.feature} is missing or was not seen in training "
                             f"({self._fallback_description})")
                break
            key = next(k for k, c in node.children.items() if c is child)
            steps.append(edge_label(node, key, with_feature=True))
            node = child
        return steps

    def summary(self) -> dict:
        return {
            "algorithm": self.algorithm_name,
            "depth": self.depth,
            "nodes": self.n_nodes,
            "leaves": self.n_leaves,
            "fit_time_s": self.fit_time_,
        }


# ---------------------------------------------------------------- helpers
def infer_feature_types(X: pd.DataFrame,
                        overrides: dict[str, str] | None = None) -> dict[str, str]:
    overrides = overrides or {}
    types = {}
    for col in X.columns:
        if col in overrides:
            types[col] = overrides[col]
        elif pd.api.types.is_bool_dtype(X[col]):
            types[col] = "categorical"
        elif pd.api.types.is_numeric_dtype(X[col]):
            types[col] = "continuous"
        else:
            types[col] = "categorical"
    return types


def edge_label(node: Node, key, with_feature: bool = False) -> str:
    if node.threshold is not None:
        txt = f"{key} {format_number(node.threshold)}"
    else:
        txt = f"= {key}"
    return f"{node.feature} {txt}" if with_feature else txt


def format_number(x: float) -> str:
    return f"{x:.4g}"


def _normalise(dist: np.ndarray) -> np.ndarray:
    total = dist.sum()
    if total <= 0:
        return np.full_like(dist, 1.0 / len(dist))
    return dist / total


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _as_key(value):
    """Categorical branch keys are always strings, so 1, 1.0 and "1" agree."""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)
