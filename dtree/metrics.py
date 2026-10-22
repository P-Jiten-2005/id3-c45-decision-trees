"""Evaluation helpers built on scikit-learn's metric functions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)


@dataclass
class Evaluation:
    labels: list
    confusion: np.ndarray
    accuracy: float
    precision: float
    recall: float
    f1: float
    average: str
    report: pd.DataFrame

    def as_dict(self) -> dict:
        return {"Accuracy": self.accuracy, "Precision": self.precision,
                "Recall": self.recall, "F1-score": self.f1}


def evaluate(y_true, y_pred, average: str = "weighted", labels=None) -> Evaluation:
    """Confusion matrix, accuracy and averaged precision / recall / F1.

    ``average`` is one of ``"macro"``, ``"weighted"`` or ``"micro"``. Classes
    that are never predicted get precision 0 instead of raising a warning.
    """
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    labels = [str(l) for l in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=average, zero_division=0
    )
    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True,
                                zero_division=0)
    report = pd.DataFrame(rep).T
    report = report.rename(columns={"precision": "Precision", "recall": "Recall",
                                    "f1-score": "F1-score", "support": "Support"})
    return Evaluation(labels, cm, float(accuracy_score(y_true, y_pred)),
                      float(p), float(r), float(f), average, report)
