"""End-to-end smoke tests that drive the Streamlit app headlessly."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _trained_app(sample_index: int) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.sidebar.radio[0].set_value("Sample dataset").run()
    sel = at.sidebar.selectbox[0]
    sel.set_value(sel.options[sample_index]).run()
    at.button[0].click().run()
    return at


def test_app_starts_without_data():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("Upload a CSV" in i.value for i in at.info)


@pytest.mark.parametrize("sample_index", [0, 1, 2])
def test_train_on_each_sample(sample_index):
    at = _trained_app(sample_index)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    labels = [m.label for m in at.metric]
    assert labels.count("Accuracy") == 2          # ID3 tab + C4.5 tab


def test_single_prediction_and_cross_validation():
    at = _trained_app(1)                            # iris
    predict = next(b for b in at.button if b.label == "Predict")
    predict.click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(m.label == "C4.5" for m in at.metric)
    cv = next(b for b in at.button if b.label == "Run cross-validation")
    cv.click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("folds" in md.value for md in at.markdown)
