"""Train, evaluate and compare the algorithms on a prepared dataset."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from .c45 import C45Classifier
from .id3 import ID3Classifier
from .metrics import Evaluation, evaluate
from .preprocessing import PreparedData, PreprocessConfig, Preprocessor


@dataclass
class ModelParams:
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 2          # C4.5
    prune: bool = True                 # C4.5
    confidence_factor: float = 0.25    # C4.5
    include_sklearn: bool = True


@dataclass
class ModelResult:
    name: str
    model: object
    y_pred: np.ndarray
    proba: np.ndarray | None
    evaluation: Evaluation
    train_accuracy: float
    fit_time: float
    complexity: dict = field(default_factory=dict)


def build_id3(p: ModelParams) -> ID3Classifier:
    return ID3Classifier(max_depth=p.max_depth, min_samples_split=p.min_samples_split)


def build_c45(p: ModelParams) -> C45Classifier:
    return C45Classifier(max_depth=p.max_depth, min_samples_split=p.min_samples_split,
                         min_samples_leaf=p.min_samples_leaf, prune=p.prune,
                         confidence_factor=p.confidence_factor)


class SklearnCART:
    """scikit-learn's optimised CART tree with the entropy criterion.

    Included only as a reference point. It needs one-hot encoded categorical
    features and always makes binary splits.
    """

    algorithm_name = "CART (scikit-learn)"

    def __init__(self, p: ModelParams, random_state: int = 42):
        self.clf = DecisionTreeClassifier(criterion="entropy", max_depth=p.max_depth,
                                          min_samples_split=max(2, p.min_samples_split),
                                          random_state=random_state)
        self.encoder = None
        self.cat_cols: list[str] = []
        self.num_cols: list[str] = []

    def _encode(self, X: pd.DataFrame, fit: bool) -> np.ndarray:
        parts = []
        if self.num_cols:
            parts.append(X[self.num_cols].to_numpy(dtype=float))
        if self.cat_cols:
            cats = X[self.cat_cols].astype(str)
            if fit:
                self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                self.encoder.fit(cats)
            parts.append(self.encoder.transform(cats))
        return np.hstack(parts)

    def fit(self, X: pd.DataFrame, y, feature_types: dict[str, str]):
        self.num_cols = [c for c in X.columns if feature_types[c] == "continuous"]
        self.cat_cols = [c for c in X.columns if feature_types[c] != "continuous"]
        start = time.perf_counter()
        self.clf.fit(self._encode(X, True), np.asarray(y).astype(str))
        self.fit_time_ = time.perf_counter() - start
        self.classes_ = self.clf.classes_
        return self

    def predict(self, X):
        return self.clf.predict(self._encode(X, False))

    def predict_proba(self, X):
        return self.clf.predict_proba(self._encode(X, False))

    def summary(self) -> dict:
        return {"algorithm": self.algorithm_name, "depth": int(self.clf.get_depth()),
                "nodes": int(self.clf.tree_.node_count),
                "leaves": int(self.clf.get_n_leaves()), "fit_time_s": self.fit_time_}


def _fit_eval(name, model, pre: Preprocessor, algo_input: str, data: PreparedData,
              average: str, labels) -> ModelResult:
    Xtr = pre.transform(data.X_train_raw, algo_input)
    Xte = pre.transform(data.X_test_raw, algo_input)
    ft = pre.feature_types_for(algo_input)
    model.fit(Xtr, data.y_train, ft)
    y_pred = model.predict(Xte)
    proba = model.predict_proba(Xte)
    train_acc = float(np.mean(np.asarray(model.predict(Xtr)).astype(str)
                              == data.y_train.astype(str).to_numpy()))
    ev = evaluate(data.y_test, y_pred, average=average, labels=labels)
    summ = model.summary()
    return ModelResult(name, model, y_pred, proba, ev, train_acc, summ["fit_time_s"], summ)


def run_all(data: PreparedData, params: ModelParams, average: str = "weighted") -> dict:
    """Fit ID3, C4.5 (and optionally scikit-learn CART) and evaluate on test."""
    pre = data.preprocessor
    labels = sorted(set(data.y_train.astype(str)) | set(data.y_test.astype(str)))
    results = {
        "ID3": _fit_eval("ID3", build_id3(params), pre, "id3", data, average, labels),
        "C4.5": _fit_eval("C4.5", build_c45(params), pre, "c45", data, average, labels),
    }
    if params.include_sklearn:
        # CART always sees imputed, non-discretised data.
        results["CART (scikit-learn)"] = _fit_eval(
            "CART (scikit-learn)", SklearnCART(params, pre.config.random_state), pre,
            "cart", data, average, labels)
    return results


def comparison_table(results: dict) -> pd.DataFrame:
    rows = {}
    for name, r in results.items():
        rows[name] = {**r.evaluation.as_dict(), "Train accuracy": r.train_accuracy,
                      "Depth": r.complexity["depth"], "Nodes": r.complexity["nodes"],
                      "Leaves": r.complexity["leaves"],
                      "Fit time (ms)": r.fit_time * 1000}
    return pd.DataFrame(rows).T


def cross_validate(df: pd.DataFrame, config: PreprocessConfig, params: ModelParams,
                   k: int = 5, average: str = "weighted") -> pd.DataFrame:
    """k-fold CV: preprocessing is re-fitted inside every fold (no leakage)."""
    data = df[df[config.target].notna()].reset_index(drop=True)
    feats = [c for c in data.columns if c != config.target and c not in config.drop_columns]
    if config.missing_strategy == "drop":
        data = data.dropna(subset=feats).reset_index(drop=True)
    y = data[config.target].astype(str)
    X = data.drop(columns=[config.target])
    counts = y.value_counts()
    splitter = StratifiedKFold(k, shuffle=True, random_state=config.random_state) \
        if counts.min() >= k else KFold(k, shuffle=True, random_state=config.random_state)
    records = []
    for fold, (tr, te) in enumerate(splitter.split(X, y), start=1):
        pd_ = PreparedData(X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True),
                           y.iloc[tr].reset_index(drop=True), y.iloc[te].reset_index(drop=True),
                           Preprocessor(config).fit(X.iloc[tr]), 0, 0, True)
        res = run_all(pd_, params, average)
        for name, r in res.items():
            records.append({"fold": fold, "model": name, **r.evaluation.as_dict()})
    return pd.DataFrame(records)


def agreement(results: dict, a: str = "ID3", b: str = "C4.5") -> float:
    pa = np.asarray(results[a].y_pred).astype(str)
    pb = np.asarray(results[b].y_pred).astype(str)
    return float(np.mean(pa == pb))
