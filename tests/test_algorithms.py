"""Unit tests for the impurity measures and the ID3 / C4.5 classifiers."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dtree import C45Classifier, ID3Classifier
from dtree.criteria import (entropy, gain_ratio, information_gain, pessimistic_errors,
                            split_information)

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture
def tennis():
    df = pd.read_csv(DATA / "play_tennis.csv")
    return df.drop(columns=["Day", "PlayTennis"]), df["PlayTennis"]


# ------------------------------------------------------------ criteria
def test_entropy_known_values():
    assert entropy(np.array([9, 5])) == pytest.approx(0.940, abs=1e-3)   # Mitchell, ch. 3
    assert entropy(np.array([7, 7])) == pytest.approx(1.0)
    assert entropy(np.array([10, 0])) == 0.0
    assert entropy(np.array([0, 0])) == 0.0


def test_information_gain_outlook_matches_textbook(tennis):
    X, y = tennis
    parent = y.value_counts().reindex(["No", "Yes"]).to_numpy()
    children = [y[X.Outlook == v].value_counts().reindex(["No", "Yes"], fill_value=0)
                .to_numpy() for v in ["Sunny", "Overcast", "Rain"]]
    assert information_gain(parent, children) == pytest.approx(0.2467, abs=1e-3)


def test_split_information_and_ratio():
    assert split_information([5, 4, 5]) == pytest.approx(1.577, abs=1e-3)
    assert gain_ratio(0.2467, 1.577) == pytest.approx(0.156, abs=1e-3)
    assert gain_ratio(0.5, 0.0) == 0.0


def test_pessimistic_errors_exceed_observed():
    # Quinlan (1993), ch. 4: a leaf with 6 cases and 0 errors -> U = 0.206, 1.24 errors
    assert pessimistic_errors(6, 0, 0.25) == pytest.approx(6 * 0.206, abs=1e-2)
    for n, e in [(10, 1), (50, 5), (3, 2)]:
        assert pessimistic_errors(n, e) > e
    # lower confidence factor -> more pessimistic estimate
    assert pessimistic_errors(20, 2, 0.1) > pessimistic_errors(20, 2, 0.4)


# ----------------------------------------------------------------- ID3
def test_id3_reproduces_classic_tennis_tree(tennis):
    X, y = tennis
    m = ID3Classifier().fit(X, y)
    assert m.root_.feature == "Outlook"
    assert m.root_.children["Overcast"].is_leaf
    assert m.root_.children["Sunny"].feature == "Humidity"
    assert m.root_.children["Rain"].feature == "Wind"
    assert m.n_leaves == 5 and m.depth == 2
    assert (m.predict(X) == y.to_numpy()).all()


def test_id3_rejects_missing_values():
    with pytest.raises(ValueError, match="missing"):
        ID3Classifier().fit(pd.DataFrame({"a": ["p", None, "q"]}), ["x", "y", "x"])


def test_id3_unseen_category_falls_back_to_node_distribution(tennis):
    X, y = tennis
    m = ID3Classifier().fit(X, y)
    row = X.iloc[[0]].copy()
    row["Outlook"] = "Snow"
    proba = m.predict_proba(row)[0]
    assert proba == pytest.approx([5 / 14, 9 / 14])


def test_max_depth_limits_tree(tennis):
    X, y = tennis
    m = ID3Classifier(max_depth=1).fit(X, y)
    assert m.depth == 1


def test_rules_cover_every_leaf(tennis):
    X, y = tennis
    m = ID3Classifier().fit(X, y)
    rules = m.to_rules()
    assert len(rules) == m.n_leaves
    assert any("Outlook = Overcast" in r and "Yes" in r for r in rules)


# ---------------------------------------------------------------- C4.5
def test_c45_prefers_gain_ratio_over_id_column(tennis):
    X, y = tennis
    X = X.assign(Day=[f"D{i}" for i in range(14)])
    id3 = ID3Classifier().fit(X, y)
    c45 = C45Classifier(prune=False).fit(X, y)     # default: >= 2 cases in 2 branches
    assert id3.root_.feature == "Day"           # ID3's bias towards many-valued attributes
    assert c45.root_.feature != "Day"


def test_c45_finds_continuous_threshold():
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0]})
    y = ["a"] * 4 + ["b"] * 4
    m = C45Classifier(min_samples_leaf=1).fit(X, y)
    assert m.root_.threshold == pytest.approx(7.0)
    assert list(m.predict(pd.DataFrame({"x": [0.0, 6.9, 7.1, 50.0]}))) == ["a", "a", "b", "b"]


def test_c45_reuses_continuous_attribute():
    X = pd.DataFrame({"x": np.arange(1, 91, dtype=float)})
    y = ["a"] * 30 + ["b"] * 30 + ["a"] * 30
    m = C45Classifier(min_samples_leaf=1, prune=False).fit(X, y)
    used = [n.feature for n in m.iter_nodes() if not n.is_leaf]
    assert used.count("x") >= 2
    assert (m.predict(X) == np.array(y)).all()


def test_c45_handles_missing_values_in_training_and_prediction():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, 200)
    y = np.where(x > 5, "high", "low")
    X = pd.DataFrame({"x": x, "noise": rng.choice(["p", "q"], 200)})
    X.loc[rng.random(200) < 0.15, "x"] = np.nan
    m = C45Classifier().fit(X, y)
    assert m.root_.feature == "x"
    assert m.root_.threshold == pytest.approx(5, abs=0.5)
    # weights are conserved: root distribution sums to the number of rows
    assert m.root_.n_samples == pytest.approx(200)
    proba = m.predict_proba(pd.DataFrame({"x": [np.nan], "noise": ["p"]}))[0]
    assert proba.sum() == pytest.approx(1.0)
    assert 0.2 < proba.max() < 0.8              # genuinely uncertain when x is unknown


def test_pruning_shrinks_noisy_tree():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300),
                      "c": rng.choice(list("pqrs"), 300)})
    y = np.where(X.a + rng.normal(scale=1.0, size=300) > 0, "y", "n")
    unpruned = C45Classifier(prune=False).fit(X, y)
    pruned = C45Classifier(prune=True).fit(X, y)
    harsh = C45Classifier(prune=True, confidence_factor=0.05).fit(X, y)
    assert pruned.n_nodes < unpruned.n_nodes
    assert harsh.n_nodes <= pruned.n_nodes
    assert pruned.n_nodes_before_pruning_ == unpruned.n_nodes


def test_predict_proba_rows_sum_to_one(tennis):
    X, y = tennis
    for m in (ID3Classifier().fit(X, y), C45Classifier().fit(X, y)):
        p = m.predict_proba(X)
        assert p.shape == (14, 2)
        assert np.allclose(p.sum(axis=1), 1.0)


def test_feature_importances_normalised(tennis):
    X, y = tennis
    imp = ID3Classifier().fit(X, y).feature_importances()
    assert imp.sum() == pytest.approx(1.0)
    # Wind and Humidity each purify a 5-sample branch completely (gain 0.971),
    # which outweighs Outlook's root gain of 0.247 over 14 samples.
    assert imp["Wind"] == pytest.approx(imp["Humidity"])
    assert imp["Outlook"] > 0
    assert imp["Temperature"] == 0


def test_numeric_class_labels_round_trip():
    X = pd.DataFrame({"f": ["a", "a", "b", "b", "c", "c"]})
    y = np.array([0, 0, 1, 1, 2, 2])
    m = ID3Classifier().fit(X, y)
    assert (m.predict(X) == y).all()


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        ID3Classifier().predict(pd.DataFrame({"a": ["x"]}))
