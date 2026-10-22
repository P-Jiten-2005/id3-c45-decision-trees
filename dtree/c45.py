"""C4.5, Quinlan 1993 — the successor of ID3.

What C4.5 adds on top of ID3
----------------------------
* **Gain ratio** instead of raw information gain. ``GainRatio = Gain /
  SplitInfo`` penalises attributes that split the data into many small
  branches (e.g. an ID column). Following Quinlan, only attributes whose gain is
  at least the average gain of all candidates are eligible, which stops a tiny
  SplitInfo from inflating the ratio of a useless attribute.
* **Continuous attributes.** For a numeric attribute the samples are sorted and
  every midpoint between two consecutive distinct values is a candidate
  threshold ``t``; the test becomes the binary split ``A <= t`` / ``A > t``.
  The best threshold is the one with the highest gain, and the gain is reduced
  by ``log2(#candidates) / |S|`` (the MDL correction of C4.5 Release 8) so that
  numeric attributes are not favoured just because they offer many thresholds.
  A numeric attribute may be tested again further down the tree.
* **Missing values.** Gain is computed on the samples whose value is known and
  multiplied by the known fraction. When splitting, a sample with a missing
  value is sent down every branch with its weight multiplied by that branch's
  share of the known samples. At prediction time an unknown value is handled
  the same way and the leaf distributions are combined.
* **Pruning.** After growing, subtrees are replaced by leaves bottom-up when
  the pessimistic error estimate of the leaf (upper limit of a binomial
  confidence interval, confidence factor ``cf``, default 0.25) is no worse than
  that of the subtree.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .criteria import entropy, gain_ratio, pessimistic_errors, split_information
from .tree import GT, LE, DecisionTreeBase, Node, _as_key, _normalise


class C45Classifier(DecisionTreeBase):
    algorithm_name = "C4.5"
    _fallback_description = ("missing: all branches combined by training share; "
                             "unseen category: this node's class distribution")

    def __init__(self, max_depth: int | None = None, min_samples_split: int = 2,
                 min_samples_leaf: float = 2, prune: bool = True,
                 confidence_factor: float = 0.25, min_gain: float = 1e-7):
        super().__init__(max_depth=max_depth, min_samples_split=min_samples_split,
                         min_gain=min_gain)
        if not 0 < confidence_factor < 1:
            raise ValueError("confidence_factor must be in (0, 1)")
        self.min_samples_leaf = min_samples_leaf
        self.prune = prune
        self.confidence_factor = confidence_factor
        self.n_nodes_before_pruning_: int | None = None

    # ----------------------------------------------------------------- fit
    def fit(self, X, y, feature_types=None):
        X = pd.DataFrame(X).copy()
        types = feature_types or {}
        for col in X.columns:
            t = types.get(col)
            if t == "continuous":
                X[col] = pd.to_numeric(X[col], errors="coerce")
            elif t == "categorical" or not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].map(lambda v: np.nan if pd.isna(v) else _as_key(v))
        return super().fit(X, y, feature_types)

    def _build(self, X: pd.DataFrame, y: np.ndarray, w: np.ndarray,
               features: list[str], depth: int) -> Node:
        dist = self._class_dist(y, w)
        node = self._new_node(dist, depth)
        if self._should_stop(dist, depth, features) or dist.sum() < 2 * self.min_samples_leaf:
            return node

        candidates = []
        for feat in features:
            if self.feature_types_[feat] == "continuous":
                res = self._eval_continuous(X[feat].to_numpy(dtype=float), y, w, dist)
            else:
                res = self._eval_categorical(X[feat].to_numpy(dtype=object), y, w, dist)
            if res is not None and res["gain"] > self.min_gain:
                res["feature"] = feat
                candidates.append(res)
        if not candidates:
            return node

        avg_gain = float(np.mean([c["gain"] for c in candidates]))
        eligible = [c for c in candidates if c["gain"] >= avg_gain - 1e-12] or candidates
        best = max(eligible, key=lambda c: (c["gain_ratio"], c["gain"]))

        node.feature = best["feature"]
        node.gain = best["gain"]
        node.gain_ratio = best["gain_ratio"]
        node.threshold = best.get("threshold")

        col = X[node.feature]
        missing = col.isna().to_numpy()
        if node.threshold is not None:
            vals = col.to_numpy(dtype=float)
            branches = {LE: (~missing) & (vals <= node.threshold),
                        GT: (~missing) & (vals > node.threshold)}
        else:
            vals = col.to_numpy(dtype=object)
            branches = {k: (~missing) & (vals == k) for k in best["values"]}

        known_w = sum(float(w[m].sum()) for m in branches.values())
        # Continuous attributes stay available deeper in the tree; categorical
        # ones are exhausted once tested (every branch has a single value).
        remaining = features if node.threshold is not None else \
            [f for f in features if f != node.feature]

        for key, mask in branches.items():
            share = float(w[mask].sum()) / known_w
            node.branch_weights[key] = share
            sub_w = np.where(mask, w, np.where(missing, w * share, 0.0))
            keep = sub_w > 1e-9
            node.children[key] = self._build(
                X[keep].reset_index(drop=True), y[keep], sub_w[keep], remaining, depth + 1
            )
        return node

    # ---------------------------------------------------- split evaluation
    def _eval_categorical(self, col: np.ndarray, y, w, dist):
        known = ~pd.isna(col)
        if known.sum() == 0:
            return None
        total_w = float(w.sum())
        kw = w[known]
        kcol, ky = col[known], y[known]
        known_w = float(kw.sum())
        values = sorted(set(kcol))
        if len(values) < 2:
            return None
        child_dists, part_w = [], []
        for v in values:
            m = kcol == v
            child_dists.append(self._class_dist(ky[m], kw[m]))
            part_w.append(float(kw[m].sum()))
        # require at least two branches carrying min_samples_leaf weight
        if sum(pw >= self.min_samples_leaf for pw in part_w) < 2:
            return None
        known_dist = self._class_dist(ky, kw)
        frac = known_w / total_w
        gain = frac * (entropy(known_dist)
                       - sum(pw / known_w * entropy(cd) for pw, cd in zip(part_w, child_dists)))
        missing_w = total_w - known_w
        split_parts = part_w + ([missing_w] if missing_w > 1e-9 else [])
        si = split_information(split_parts)
        return {"gain": gain, "gain_ratio": gain_ratio(gain, si), "values": values}

    def _eval_continuous(self, col: np.ndarray, y, w, dist):
        known = ~np.isnan(col)
        if known.sum() < 2:
            return None
        total_w = float(w.sum())
        kv, ky, kw = col[known], y[known], w[known]
        order = np.argsort(kv, kind="mergesort")
        kv, ky, kw = kv[order], ky[order], kw[order]
        known_w = float(kw.sum())

        # cumulative class weights left of each split position
        onehot = np.zeros((len(kv), self._n_classes))
        onehot[np.arange(len(kv)), ky] = kw
        left_cum = np.cumsum(onehot, axis=0)
        known_dist = left_cum[-1]

        # candidate split positions: between i and i+1 where value changes
        pos = np.nonzero(kv[1:] > kv[:-1] + 1e-12)[0]
        if len(pos) == 0:
            return None
        left = left_cum[pos]
        right = known_dist - left
        lw, rw = left.sum(axis=1), right.sum(axis=1)
        valid = (lw >= self.min_samples_leaf) & (rw >= self.min_samples_leaf)
        if not valid.any():
            return None
        pos, left, right, lw, rw = pos[valid], left[valid], right[valid], lw[valid], rw[valid]

        h_known = entropy(known_dist)
        h_left = _row_entropy(left)
        h_right = _row_entropy(right)
        info_gain = h_known - (lw * h_left + rw * h_right) / known_w
        best = int(np.argmax(info_gain))

        frac = known_w / total_w
        n_candidates = len(pos)
        gain = frac * info_gain[best] - math.log2(n_candidates) / total_w if n_candidates > 1 \
            else frac * info_gain[best]
        if gain <= 0:
            return None
        i = pos[best]
        threshold = float((kv[i] + kv[i + 1]) / 2.0)
        missing_w = total_w - known_w
        split_parts = [lw[best], rw[best]] + ([missing_w] if missing_w > 1e-9 else [])
        si = split_information(split_parts)
        return {"gain": gain, "gain_ratio": gain_ratio(gain, si), "threshold": threshold}

    # ------------------------------------------------------------- pruning
    def _post_process(self) -> None:
        self.n_nodes_before_pruning_ = self.n_nodes
        if self.prune:
            self._prune(self.root_)

    def _prune(self, node: Node) -> float:
        """Bottom-up subtree replacement; returns the node's estimated errors."""
        n = node.n_samples
        leaf_errors = pessimistic_errors(n, n - node.class_dist.max(), self.confidence_factor)
        if node.is_leaf:
            return leaf_errors
        subtree_errors = sum(self._prune(child) for child in node.children.values())
        if leaf_errors <= subtree_errors + 0.1:
            node.make_leaf()
            return leaf_errors
        return subtree_errors

    # ---------------------------------------------------------- prediction
    def _distribute(self, node: Node, row: dict) -> np.ndarray:
        value = row.get(node.feature)
        if node.threshold is None and not _missing(value):
            # unseen category: fall back to this node's class distribution
            return _normalise(node.class_dist)
        out = np.zeros(self._n_classes)
        for key, child in node.children.items():
            out += node.branch_weights.get(key, 0.0) * self._proba_row(child, row)
        return _normalise(out)


def _row_entropy(dists: np.ndarray) -> np.ndarray:
    totals = dists.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(totals > 0, dists / totals, 0.0)
        logs = np.where(p > 0, np.log2(np.where(p > 0, p, 1.0)), 0.0)
    return -(p * logs).sum(axis=1)


def _missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
