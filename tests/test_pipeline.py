"""Tests for preprocessing, evaluation and the experiment runner."""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dtree.experiment import ModelParams, comparison_table, cross_validate, run_all
from dtree.metrics import evaluate
from dtree.preprocessing import (PreprocessConfig, Preprocessor, detect_feature_types,
                                 prepare, read_csv, suggest_id_columns, target_problems)
from dtree.visualize import tree_to_graphviz

DATA = Path(__file__).resolve().parent.parent / "data"


def test_read_csv_normalises_missing_tokens_and_numeric_text():
    raw = b"a;b;c\n1;x;?\n2; y ;3.5\n?;NA;4\n"
    df = read_csv(io.BytesIO(raw))
    assert list(df.columns) == ["a", "b", "c"]
    assert pd.api.types.is_numeric_dtype(df["a"]) and df["a"].isna().sum() == 1
    assert df["b"].tolist()[:2] == ["x", "y"] and pd.isna(df["b"].iloc[2])
    assert pd.api.types.is_numeric_dtype(df["c"])


def test_detect_feature_types():
    df = pd.DataFrame({"num": np.linspace(0, 1, 30), "flag": [0, 1] * 15,
                       "txt": ["a", "b", "c"] * 10})
    t = detect_feature_types(df)
    assert t == {"num": "continuous", "flag": "categorical", "txt": "categorical"}


def test_suggest_id_columns():
    df = pd.read_csv(DATA / "loan_approval.csv")
    assert "applicant_id" in suggest_id_columns(df)


def test_target_problems():
    assert target_problems(pd.Series(["a"] * 5))
    assert target_problems(pd.Series(np.random.default_rng(0).normal(size=100)))
    assert not target_problems(pd.Series(["a", "b"] * 10))


def test_preprocessor_fitted_on_train_only():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, np.nan], "c": ["p", "p", "q", None],
                          "y": list("abab")})
    test = pd.DataFrame({"x": [np.nan, 100.0], "c": [None, "zzz"], "y": list("ab")})
    cfg = PreprocessConfig(target="y", n_bins=2, feature_types={"x": "continuous"})
    pre = Preprocessor(cfg).fit(train)
    assert pre.fill_values_ == {"x": 2.0, "c": "p"}
    out = pre.transform(test, "c45")
    assert out["x"].tolist() == [2.0, 100.0]
    assert out["c"].tolist() == ["p", "zzz"]
    id3 = pre.transform(test, "id3")
    assert id3["x"].iloc[1].startswith(">")       # 100 lands in the open top bin


def test_native_missing_strategy_keeps_nans_for_c45_only():
    df = pd.read_csv(DATA / "loan_approval.csv")
    cfg = PreprocessConfig(target="approved", drop_columns=["applicant_id"],
                           missing_strategy="native")
    data = prepare(df, cfg)
    pre = data.preprocessor
    assert pre.transform(data.X_train_raw, "c45").isna().any().any()
    assert not pre.transform(data.X_train_raw, "id3").isna().any().any()
    assert not pre.transform(data.X_train_raw, "cart").isna().any().any()


def test_prepare_drops_missing_targets_and_stratifies():
    df = pd.read_csv(DATA / "iris.csv")
    df.loc[[0, 50], "species"] = np.nan
    data = prepare(df, PreprocessConfig(target="species", test_size=0.2))
    assert data.n_dropped_target == 2
    assert data.stratified
    assert len(data.y_train) + len(data.y_test) == 148


def test_evaluate_metrics():
    ev = evaluate(["a", "a", "b", "b"], ["a", "b", "b", "b"], average="macro")
    assert ev.accuracy == pytest.approx(0.75)
    assert ev.confusion.tolist() == [[1, 1], [0, 2]]
    assert ev.precision == pytest.approx((1.0 + 2 / 3) / 2)
    assert ev.recall == pytest.approx((0.5 + 1.0) / 2)


@pytest.mark.parametrize("fname,target,drop", [
    ("play_tennis.csv", "PlayTennis", ["Day"]),
    ("iris.csv", "species", []),
    ("loan_approval.csv", "approved", ["applicant_id"]),
])
@pytest.mark.parametrize("strategy", ["impute", "native", "drop"])
def test_run_all_end_to_end(fname, target, drop, strategy):
    df = read_csv(DATA / fname)
    data = prepare(df, PreprocessConfig(target=target, drop_columns=drop,
                                        missing_strategy=strategy))
    results = run_all(data, ModelParams())
    table = comparison_table(results)
    assert set(table.index) == {"ID3", "C4.5", "CART (scikit-learn)"}
    assert ((table[["Accuracy", "Precision", "Recall", "F1-score"]] >= 0).all().all())
    for name in ("ID3", "C4.5"):
        dot = tree_to_graphviz(results[name].model)
        assert "digraph" in dot.source


def test_iris_accuracy_is_reasonable():
    df = read_csv(DATA / "iris.csv")
    data = prepare(df, PreprocessConfig(target="species"))
    res = run_all(data, ModelParams(include_sklearn=False))
    assert res["ID3"].evaluation.accuracy > 0.8
    assert res["C4.5"].evaluation.accuracy > 0.85


def test_cross_validation_shape():
    df = read_csv(DATA / "iris.csv")
    cv = cross_validate(df, PreprocessConfig(target="species"),
                        ModelParams(include_sklearn=False), k=3)
    assert len(cv) == 6 and set(cv.model) == {"ID3", "C4.5"}
