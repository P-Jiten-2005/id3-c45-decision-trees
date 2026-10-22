"""ID3 (Iterative Dichotomiser 3), Quinlan 1986.

Algorithm
---------
1. If every sample at the node has the same class, or no attributes are left,
   create a leaf labelled with the majority class.
2. Otherwise pick the attribute A with the highest information gain
   ``Gain(S, A) = H(S) - sum_v |S_v|/|S| H(S_v)``.
3. Create one branch for every value of A seen in S and recurse on each subset
   with A removed from the candidate list (an attribute is used at most once on
   any root-to-leaf path).

Classic ID3 only understands categorical attributes and has no notion of
missing values, so the preprocessing pipeline discretises continuous columns
into bins and imputes missing values before training. This class refuses
continuous features to make that requirement explicit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .criteria import information_gain
from .tree import DecisionTreeBase, Node, _as_key


class ID3Classifier(DecisionTreeBase):
    algorithm_name = "ID3"

    def _validate_feature_types(self) -> None:
        cont = [c for c, t in self.feature_types_.items() if t != "categorical"]
        if cont:
            raise ValueError(
                "ID3 only supports categorical features; discretise these columns "
                f"first: {cont}"
            )

    def fit(self, X, y, feature_types=None):
        X = pd.DataFrame(X)
        if X.isna().any().any():
            raise ValueError("ID3 cannot handle missing values; impute them first")
        # Treat every column as categorical and compare values as strings.
        X = X.apply(lambda col: col.map(_as_key))
        ft = {c: "categorical" for c in X.columns}
        return super().fit(X, y, ft)

    def _build(self, X: pd.DataFrame, y: np.ndarray, w: np.ndarray,
               features: list[str], depth: int) -> Node:
        dist = self._class_dist(y, w)
        node = self._new_node(dist, depth)
        if self._should_stop(dist, depth, features):
            return node

        best_feature, best_gain, best_groups = None, -1.0, None
        for feat in features:
            col = X[feat].to_numpy()
            values = np.unique(col)
            if len(values) < 2:
                continue
            groups = {v: col == v for v in values}
            children = [self._class_dist(y[m], w[m]) for m in groups.values()]
            gain = information_gain(dist, children)
            # Ties are broken by column order, which keeps results reproducible.
            if gain > best_gain + 1e-12:
                best_feature, best_gain, best_groups = feat, gain, groups

        if best_feature is None or best_gain <= self.min_gain:
            return node

        node.feature = best_feature
        node.gain = best_gain
        remaining = [f for f in features if f != best_feature]
        for value, mask in best_groups.items():
            key = str(value)
            node.branch_weights[key] = float(w[mask].sum())
            node.children[key] = self._build(
                X[mask].reset_index(drop=True), y[mask], w[mask], remaining, depth + 1
            )
        return node
