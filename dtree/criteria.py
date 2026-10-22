"""Impurity and split-quality measures used by ID3 and C4.5.

All functions work on *weighted class distributions*: a 1-D NumPy array whose
k-th entry is the (possibly fractional) weight of samples belonging to class k.
Fractional weights are what let C4.5 handle missing values, and plain integer
counts are just the special case where every weight is 1.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

_EPS = 1e-12


def entropy(dist: np.ndarray) -> float:
    """Shannon entropy H(S) = -sum p_k log2 p_k of a class distribution."""
    dist = np.asarray(dist, dtype=float)
    total = dist.sum()
    if total <= _EPS:
        return 0.0
    p = dist[dist > _EPS] / total
    return float(-(p * np.log2(p)).sum())


def information_gain(parent: np.ndarray, children: list[np.ndarray]) -> float:
    """Gain(S, A) = H(S) - sum_v |S_v|/|S| * H(S_v)."""
    total = float(np.sum(parent))
    if total <= _EPS:
        return 0.0
    remainder = sum(float(c.sum()) / total * entropy(c) for c in children)
    return entropy(parent) - remainder


def split_information(partition_weights: list[float] | np.ndarray) -> float:
    """SplitInfo(S, A) = -sum_v |S_v|/|S| log2(|S_v|/|S|).

    This is the entropy of the partition itself (ignoring classes) and is the
    denominator of C4.5's gain ratio. It grows when an attribute fragments the
    data into many small branches, which is exactly the bias ID3 suffers from.
    """
    return entropy(np.asarray(partition_weights, dtype=float))


def gain_ratio(gain: float, split_info: float) -> float:
    """GainRatio = Gain / SplitInfo (0 when SplitInfo is ~0)."""
    if split_info <= _EPS:
        return 0.0
    return gain / split_info


def pessimistic_errors(n: float, e: float, cf: float = 0.25) -> float:
    """Upper-confidence estimate of the errors made by a leaf (C4.5 pruning).

    A leaf that covers ``n`` training samples and misclassifies ``e`` of them is
    treated as ``e`` successes in ``n`` binomial trials. C4.5 replaces the
    observed error rate by the upper limit of its ``cf`` confidence interval and
    returns ``n * U_cf(e, n)``. Smaller ``cf`` means more pessimism and
    therefore more aggressive pruning.

    This is a direct port of ``AddErrs`` from Quinlan's C4.5 Release 8 source
    (``stats.c``), returning total estimated errors rather than the increment.
    """
    if n <= _EPS:
        return 0.0
    return e + _add_errs(n, e, cf)


def _add_errs(n: float, e: float, cf: float) -> float:
    if e < 1e-6:
        return n * (1.0 - math.exp(math.log(cf) / n))
    if e < 0.9999:
        v0 = n * (1.0 - math.exp(math.log(cf) / n))
        return v0 + e * (_add_errs(n, 1.0, cf) - v0)
    if e + 0.5 >= n:
        return 0.67 * (n - e)
    z = NormalDist().inv_cdf(1.0 - cf)
    coeff = z * z
    pr = (
        e + 0.5 + coeff / 2.0
        + math.sqrt(coeff * ((e + 0.5) * (1.0 - (e + 0.5) / n) + coeff / 4.0))
    ) / (n + coeff)
    return n * pr - e
