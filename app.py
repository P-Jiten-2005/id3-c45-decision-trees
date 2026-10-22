"""Streamlit front end for the Decision Tree Classification System.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from dtree.experiment import (ModelParams, agreement, comparison_table, cross_validate,
                              run_all)
from dtree.preprocessing import (PreprocessConfig, detect_feature_types, prepare, read_csv,
                                 suggest_id_columns, target_problems)
from dtree.visualize import (plot_class_distribution, plot_confusion_matrix,
                             plot_feature_importance, plot_metric_comparison,
                             tree_to_graphviz)

DATA_DIR = Path(__file__).parent / "data"
SAMPLES = {
    "Play Tennis (categorical, 14 rows)": ("play_tennis.csv", "PlayTennis"),
    "Iris (continuous, 150 rows)": ("iris.csv", "species"),
    "Loan approval (mixed + missing values, synthetic)": ("loan_approval.csv", "approved"),
}
TREE_MODELS = ["ID3", "C4.5"]
METRICS = ["Accuracy", "Precision", "Recall", "F1-score"]

st.set_page_config(page_title="Decision Tree Classifier", page_icon="🌳", layout="wide")


# ----------------------------------------------------------------- helpers
@st.cache_data(show_spinner=False)
def load_csv(raw: bytes) -> pd.DataFrame:
    return read_csv(io.BytesIO(raw))


def show_fig(fig):
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def fingerprint(*parts) -> str:
    return hashlib.md5(repr(parts).encode()).hexdigest()


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("1 · Dataset")
    source = st.radio("Source", ["Upload CSV", "Sample dataset"], horizontal=True,
                      label_visibility="collapsed")
    df, default_target, raw = None, None, None
    if source == "Upload CSV":
        up = st.file_uploader("Classification dataset (.csv)", type=["csv", "txt"])
        if up is not None:
            raw = up.getvalue()
            data_name = up.name
    else:
        choice = st.selectbox("Sample", list(SAMPLES))
        fname, default_target = SAMPLES[choice]
        raw = (DATA_DIR / fname).read_bytes()
        data_name = fname
    if raw is not None:
        try:
            df = load_csv(raw)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read the file: {exc}")

    st.header("2 · Model settings")
    limit_depth = st.checkbox("Limit tree depth", value=False)
    max_depth = st.slider("Max depth", 1, 20, 6, disabled=not limit_depth)
    min_split = st.number_input("Min samples to split a node", 2, 100, 2)
    with st.expander("C4.5 options", expanded=True):
        prune = st.checkbox("Pessimistic error pruning", value=True)
        cf = st.slider("Confidence factor (lower = more pruning)", 0.01, 0.50, 0.25, 0.01,
                       disabled=not prune)
        min_leaf = st.number_input("Min samples per branch", 1, 50, 2)
    st.header("3 · Evaluation")
    average = st.selectbox("Averaging for precision / recall / F1",
                           ["weighted", "macro", "micro"],
                           help="weighted: by class support · macro: every class equal · "
                                "micro: global counts (equals accuracy for single-label data)")
    include_cart = st.checkbox("Also train scikit-learn CART as a reference", value=True)

params = ModelParams(max_depth=max_depth if limit_depth else None,
                     min_samples_split=int(min_split), min_samples_leaf=int(min_leaf),
                     prune=prune, confidence_factor=float(cf), include_sklearn=include_cart)

# ------------------------------------------------------------------- header
st.title("🌳 Decision Tree Classification System")
st.caption("Upload a classification dataset, choose the target, and compare from-scratch "
           "**ID3** and **C4.5** implementations on the same train/test split.")

if df is None:
    st.info("Upload a CSV file or pick a sample dataset in the sidebar to begin.")
    with st.expander("What does the app expect?"):
        st.markdown(
            "- A CSV with a header row; each row is one example.\n"
            "- One column holds the **class label** (the target).\n"
            "- Features can be categorical (text) or numeric; missing cells may be empty, "
            "`?`, `NA` or `null`.\n"
            "- ID3 needs categorical inputs, so numeric columns are discretised into bins "
            "for it. C4.5 uses numeric columns directly with threshold splits.")
    st.stop()

# --------------------------------------------------------------- data view
st.subheader("Dataset overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", df.shape[1])
c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
c4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

tab_preview, tab_cols = st.tabs(["Preview", "Column summary"])
with tab_preview:
    st.dataframe(df.head(200), width="stretch", height=260)
with tab_cols:
    detected = detect_feature_types(df)
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "detected type": pd.Series(detected),
        "distinct": df.nunique(),
        "missing": df.isna().sum(),
        "example": [str(df[c].dropna().iloc[0]) if df[c].notna().any() else "" for c in df],
    })
    st.dataframe(summary, width="stretch")

# ---------------------------------------------------- target & preprocessing
st.subheader("Target and preprocessing")
cols = list(df.columns)
default_idx = cols.index(default_target) if default_target in cols else len(cols) - 1
left, right = st.columns([1, 1])
with left:
    target = st.selectbox("Target attribute (class label)", cols, index=default_idx)
    for msg in target_problems(df[target]):
        st.warning(msg)
    suggested = [c for c in suggest_id_columns(df) if c != target]
    drop_cols = st.multiselect("Columns to exclude", [c for c in cols if c != target],
                               default=suggested,
                               help="Identifiers and free-text columns should be excluded: "
                                    "ID3 would happily split on a unique ID and memorise "
                                    "the training set.")
    st.markdown("**Class distribution**")
    show_fig(plot_class_distribution(df[target].dropna()))

feature_cols = [c for c in cols if c != target and c not in drop_cols]
with right:
    missing_strategy = st.radio(
        "Missing values",
        ["impute", "native", "drop"],
        format_func={"impute": "Impute (mode / median, fitted on training data)",
                     "native": "C4.5 native handling (fractional weights); ID3 still imputes",
                     "drop": "Drop rows with missing feature values"}.get,
    )
    b1, b2 = st.columns(2)
    n_bins = b1.slider("Bins for numeric features (ID3)", 2, 10, 4)
    binning = b2.selectbox("Binning", ["quantile", "uniform"],
                           format_func={"quantile": "Equal frequency",
                                        "uniform": "Equal width"}.get)
    s1, s2 = st.columns(2)
    test_size = s1.slider("Test set size", 0.1, 0.5, 0.3, 0.05)
    seed = s2.number_input("Random seed", 0, 10_000, 42)

    with st.expander("Feature types (override detection)"):
        st.caption("Numeric integer columns with ≤ 10 distinct values are treated as "
                   "categorical by default.")
        ft_df = pd.DataFrame({"feature": feature_cols,
                              "type": [detected[c] for c in feature_cols]})
        edited = st.data_editor(
            ft_df, hide_index=True, width="stretch", disabled=["feature"],
            column_config={"type": st.column_config.SelectboxColumn(
                options=["categorical", "continuous"], required=True)},
            key=f"types_{target}_{len(feature_cols)}")
        feature_types = dict(zip(edited["feature"], edited["type"]))
        bad = [c for c, t in feature_types.items()
               if t == "continuous" and pd.to_numeric(df[c], errors="coerce").notna().sum()
               < df[c].notna().sum()]
        if bad:
            st.warning(f"Non-numeric values in {bad} will be treated as missing.")

config = PreprocessConfig(target=target, drop_columns=drop_cols, feature_types=feature_types,
                          missing_strategy=missing_strategy, n_bins=n_bins, binning=binning,
                          test_size=float(test_size), random_state=int(seed))
current_fp = fingerprint(data_name, len(df), config, params, average)

can_train = bool(feature_cols) and df[target].nunique() >= 2
if not feature_cols:
    st.error("Select at least one feature column.")
train_clicked = st.button("Train ID3 and C4.5", type="primary", disabled=not can_train,
                          width="stretch")

if train_clicked:
    try:
        with st.spinner("Preprocessing data and growing trees…"):
            prepared = prepare(df, config)
            results = run_all(prepared, params, average)
        st.session_state.update(prepared=prepared, results=results, fp=current_fp,
                                config=config, params=params, cv=None, average=average)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Training failed: {exc}")
        st.stop()

if "results" not in st.session_state:
    st.stop()

if st.session_state["fp"] != current_fp:
    st.warning("Settings changed since the models were trained. Click **Train** to refresh "
               "the results below.")

prepared = st.session_state["prepared"]
results = st.session_state["results"]
pre = prepared.preprocessor
train_config = st.session_state["config"]
train_avg = st.session_state["average"]

# ------------------------------------------------------------------ results
st.divider()
st.subheader("Results")
notes = [f"{len(prepared.y_train)} training rows · {len(prepared.y_test)} test rows",
         "stratified split" if prepared.stratified else "random (non-stratified) split"]
if prepared.n_dropped_target:
    notes.append(f"{prepared.n_dropped_target} rows dropped (missing target)")
if prepared.n_dropped_missing:
    notes.append(f"{prepared.n_dropped_missing} rows dropped (missing features)")
st.caption(" · ".join(notes))

tabs = st.tabs(["Preprocessing", "ID3", "C4.5", "Comparison", "Predict"])

with tabs[0]:
    st.markdown("Statistics below were learned from the **training split only** and applied "
                "unchanged to the test split and to new data.")
    st.dataframe(pre.describe(), width="stretch", hide_index=True)
    a, b = st.columns(2)
    with a:
        st.markdown("**ID3 input (first rows of the training set)**")
        st.dataframe(pre.transform(prepared.X_train_raw, "id3").head(15),
                     width="stretch")
    with b:
        st.markdown("**C4.5 input (first rows of the training set)**")
        st.dataframe(pre.transform(prepared.X_train_raw, "c45").head(15),
                     width="stretch")


def model_tab(name: str):
    r = results[name]
    ev = r.evaluation
    m = st.columns(5)
    m[0].metric("Accuracy", pct(ev.accuracy))
    m[1].metric(f"Precision ({train_avg})", pct(ev.precision))
    m[2].metric(f"Recall ({train_avg})", pct(ev.recall))
    m[3].metric(f"F1-score ({train_avg})", pct(ev.f1))
    m[4].metric("Train accuracy", pct(r.train_accuracy),
                delta=f"{(r.train_accuracy - ev.accuracy) * 100:+.1f} pts vs test",
                delta_color="off")
    model = r.model
    info = (f"Depth **{model.depth}** · **{model.n_nodes}** nodes · **{model.n_leaves}** leaves"
            f" · fitted in {r.fit_time * 1000:.1f} ms")
    if name == "C4.5" and model.prune:
        info += f" · pruning removed {model.n_nodes_before_pruning_ - model.n_nodes} nodes"
    st.markdown(info)

    st.markdown("#### Decision tree")
    show_depth = model.depth
    if model.depth > 3:
        show_depth = st.slider("Levels to display", 1, model.depth, min(model.depth, 6),
                               key=f"depth_{name}")
    dot = tree_to_graphviz(model, max_depth=show_depth)
    st.graphviz_chart(dot, width="stretch")
    d1, d2, d3 = st.columns(3)
    full = tree_to_graphviz(model)
    d1.download_button("Download .dot", full.source, f"{name.lower()}_tree.dot",
                       key=f"dot_{name}")
    try:
        d2.download_button("Download SVG", full.pipe(format="svg"),
                           f"{name.lower()}_tree.svg", "image/svg+xml", key=f"svg_{name}")
        full.attr("graph", bgcolor="white", dpi="150")
        d3.download_button("Download PNG", full.pipe(format="png"),
                           f"{name.lower()}_tree.png", "image/png", key=f"png_{name}")
    except Exception:  # noqa: BLE001 - graphviz system binary not installed
        d2.caption("Install the Graphviz system package to enable SVG/PNG export.")

    with st.expander(f"Decision rules ({model.n_leaves})"):
        st.code("\n".join(model.to_rules()), language="text")

    left, right = st.columns([1, 1.2])
    with left:
        st.markdown("#### Confusion matrix")
        show_fig(plot_confusion_matrix(ev.confusion, ev.labels, f"{name} — test set",
                                       "Blues" if name == "ID3" else "Oranges"))
    with right:
        st.markdown("#### Per-class report")
        st.dataframe(ev.report.style.format("{:.3f}", subset=["Precision", "Recall",
                                                              "F1-score"])
                     .format("{:.0f}", subset=["Support"]), width="stretch")

    st.markdown("#### Test-set predictions")
    out = prepared.X_test_raw.copy()
    out[f"actual_{train_config.target}"] = prepared.y_test.values
    out["predicted"] = r.y_pred
    out["confidence"] = r.proba.max(axis=1).round(3)
    out["correct"] = out["predicted"].astype(str) == out[f"actual_{train_config.target}"].astype(str)
    only_wrong = st.toggle("Show misclassified rows only", key=f"wrong_{name}")
    st.dataframe(out[~out["correct"]] if only_wrong else out, width="stretch",
                 height=280)
    st.download_button("Download predictions (CSV)", out.to_csv(index=False),
                       f"{name.lower()}_predictions.csv", "text/csv", key=f"csv_{name}")


with tabs[1]:
    model_tab("ID3")
with tabs[2]:
    model_tab("C4.5")

with tabs[3]:
    table = comparison_table(results)
    best = table["Accuracy"].astype(float).idxmax()
    id3_acc, c45_acc = table.loc["ID3", "Accuracy"], table.loc["C4.5", "Accuracy"]
    if abs(id3_acc - c45_acc) < 1e-9:
        verdict = "ID3 and C4.5 reach the same test accuracy"
    else:
        win = "C4.5" if c45_acc > id3_acc else "ID3"
        verdict = f"**{win}** is ahead by {abs(c45_acc - id3_acc) * 100:.1f} accuracy points"
    smaller = "C4.5" if table.loc["C4.5", "Nodes"] < table.loc["ID3", "Nodes"] else "ID3"
    st.markdown(f"{verdict} on this split. {smaller} builds the smaller tree "
                f"({int(table.loc[smaller, 'Nodes'])} vs "
                f"{int(table.loc['ID3' if smaller == 'C4.5' else 'C4.5', 'Nodes'])} nodes). "
                f"The two models agree on {pct(agreement(results))} of test predictions.")

    show_fig(plot_metric_comparison(table[METRICS].astype(float)))
    fmt = {c: "{:.3f}" for c in METRICS + ["Train accuracy"]}
    fmt.update({"Depth": "{:.0f}", "Nodes": "{:.0f}", "Leaves": "{:.0f}",
                "Fit time (ms)": "{:.1f}"})
    st.dataframe(table.style.format(fmt).highlight_max(subset=METRICS, color="#d9ead3"),
                 width="stretch")

    a, b = st.columns(2)
    with a:
        st.markdown("#### Attribute importance")
        st.caption("Share of total weighted information gain contributed by each attribute.")
        show_fig(plot_feature_importance({n: results[n].model.feature_importances()
                                          for n in TREE_MODELS}))
    with b:
        st.markdown("#### How the algorithms differ")
        st.markdown(
            "| | ID3 | C4.5 |\n|---|---|---|\n"
            "| Split criterion | Information gain | Gain ratio |\n"
            "| Numeric attributes | Discretised into bins | Binary threshold splits |\n"
            "| Missing values | Must be imputed | Fractional weights |\n"
            "| Pruning | None | Pessimistic error pruning |\n"
            "| Overfitting risk | High (grows until pure) | Lower |")

    st.markdown("#### k-fold cross-validation")
    st.caption("A single split can be noisy. Cross-validation re-runs preprocessing and "
               "training inside every fold for a more reliable comparison.")
    k = st.slider("Folds", 3, 10, 5)
    if st.button("Run cross-validation"):
        with st.spinner(f"Running {k}-fold cross-validation…"):
            try:
                st.session_state["cv"] = (k, cross_validate(df, train_config,
                                                            st.session_state["params"],
                                                            k, train_avg))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Cross-validation failed: {exc}")
    if st.session_state.get("cv") is not None:
        k_done, cv = st.session_state["cv"]
        agg = cv.groupby("model")[METRICS].agg(["mean", "std"])
        disp = pd.DataFrame({m: [f"{agg.loc[i, (m, 'mean')]:.3f} ± {agg.loc[i, (m, 'std')]:.3f}"
                                 for i in agg.index] for m in METRICS}, index=agg.index)
        st.markdown(f"Mean ± standard deviation over {k_done} folds")
        st.dataframe(disp, width="stretch")
        show_fig(plot_metric_comparison(agg.xs("mean", axis=1, level=1)[METRICS]))

with tabs[4]:
    st.markdown("Enter attribute values to classify a new example. Leave a value as "
                "*(missing)* to see how each algorithm copes with unknown data.")
    with st.form("predict_one"):
        inputs = {}
        grid = st.columns(3)
        for i, col in enumerate(pre.features_):
            with grid[i % 3]:
                if pre.types_[col] == "continuous":
                    lo, hi, med = pre.numeric_ranges_[col]
                    missing = st.checkbox(f"{col} missing", key=f"miss_{col}")
                    val = st.number_input(col, value=float(med), key=f"in_{col}",
                                          help=f"Training range {lo:g} – {hi:g}")
                    inputs[col] = np.nan if missing else val
                else:
                    opts = ["(missing)"] + [str(v) for v in pre.categories_.get(col, [])]
                    default = opts.index(str(pre.fill_values_[col])) \
                        if str(pre.fill_values_[col]) in opts else 0
                    val = st.selectbox(col, opts, index=default, key=f"in_{col}")
                    inputs[col] = np.nan if val == "(missing)" else val
        submitted = st.form_submit_button("Predict", type="primary")
    if submitted:
        row = pd.DataFrame([inputs])
        cols_out = st.columns(len(results))
        for (name, r), slot in zip(results.items(), cols_out):
            algo = {"ID3": "id3", "C4.5": "c45"}.get(name, "cart")
            Xr = pre.transform(row, algo)
            proba = r.model.predict_proba(Xr)[0]
            classes = [str(c) for c in r.model.classes_]
            with slot:
                st.metric(name, classes[int(np.argmax(proba))],
                          f"{proba.max() * 100:.0f}% confidence", delta_color="off")
                st.dataframe(pd.DataFrame({"probability": proba}, index=classes)
                             .sort_values("probability", ascending=False)
                             .style.format("{:.3f}").bar(color="#9fc5e8", vmin=0, vmax=1),
                             width="stretch")
                if hasattr(r.model, "decision_path"):
                    path = r.model.decision_path(Xr.iloc[0].to_dict())
                    st.caption("Path: " + (" → ".join(path) if path else "root is a leaf"))

    st.markdown("#### Batch prediction")
    batch = st.file_uploader("CSV with the same feature columns", type=["csv"],
                             key="batch_upload")
    if batch is not None:
        try:
            new = read_csv(batch)
            missing_cols = [c for c in pre.features_ if c not in new.columns]
            if missing_cols:
                st.error(f"Missing columns: {missing_cols}")
            else:
                out = new.copy()
                for name, r in results.items():
                    algo = {"ID3": "id3", "C4.5": "c45"}.get(name, "cart")
                    out[f"pred_{name}"] = r.model.predict(pre.transform(new, algo))
                st.dataframe(out, width="stretch", height=280)
                st.download_button("Download predictions", out.to_csv(index=False),
                                   "batch_predictions.csv", "text/csv")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not score the file: {exc}")
