# Architecture

## Overview

```
          ┌──────────────┐
CSV ───►  │ preprocessing│  read_csv → detect types → prepare() (split)
          │              │  Preprocessor.fit(train) → transform(split, algorithm)
          └──────┬───────┘
                 │ id3 view (binned, imputed)   c45 view (numeric, NaN allowed)
        ┌────────┴────────┐
        ▼                 ▼
   ID3Classifier     C45Classifier   (+ SklearnCART reference)
        │                 │
        └────────┬────────┘
                 ▼
     experiment.run_all ─► metrics.evaluate ─► comparison_table / cross_validate
                 │
                 ▼
     visualize (Graphviz tree, Matplotlib charts)  ─►  app.py (Streamlit UI)
```

The algorithm library in `dtree/` has no Streamlit dependency, so it can be imported in
notebooks, scripts and tests; `app.py` is only a presentation layer.

## Modules

| Module | Responsibility |
|---|---|
| `criteria.py` | Pure functions on weighted class distributions: entropy, information gain, split information, gain ratio, C4.5's pessimistic error bound. |
| `tree.py` | `Node` dataclass and `DecisionTreeBase`: input validation, label encoding, prediction by tree traversal, rules, decision paths, feature importances, size statistics. |
| `id3.py` | `ID3Classifier._build`: information-gain multi-way splits on categorical data. |
| `c45.py` | `C45Classifier`: gain-ratio selection, threshold search, weighted missing-value handling, pruning, weighted prediction for missing values. |
| `preprocessing.py` | Robust CSV reading, type detection, ID-column suggestions, target checks, the leakage-free `Preprocessor`, and `prepare()` for the stratified split. |
| `metrics.py` | Wraps scikit-learn's confusion matrix and precision/recall/F1 into one `Evaluation` object. |
| `visualize.py` | Graphviz `Digraph` export and Matplotlib figures. |
| `experiment.py` | Runs every model on the same split, builds the comparison table, k-fold cross-validation, and the scikit-learn CART reference. |

## Design decisions

**Weighted class distributions everywhere.** Every node stores a vector of class weights
rather than a list of row indices. Integer counts are the special case of weight 1, so ID3
and C4.5 share one representation, and C4.5's fractional weights for missing values need no
special data structures. Any node can act as a leaf, which simplifies pruning,
depth-limited display and fallback predictions.

**One shared base class.** Fitting, prediction, rule extraction and statistics live in
`DecisionTreeBase`; the subclasses only implement `_build` (how to grow) and optionally
`_post_process` (C4.5 pruning) and `_distribute` (C4.5 missing-value prediction).

**Preprocessing fitted on training data only.** Imputation values and bin edges come
from the training split and are reused for the test split, cross-validation folds,
single predictions and batch scoring. This avoids leakage and guarantees new data is
encoded exactly like the training data.

**Each algorithm gets the input it was designed for.** ID3 receives binned, imputed,
categorical data; C4.5 receives raw numeric columns (and optionally NaNs). Comparing them
this way shows the real effect of C4.5's improvements rather than comparing both on
pre-discretised data.

**Categorical keys are strings.** Values are normalised to strings (`1`, `1.0` and `"1"`
are the same category), so CSV parsing quirks don't create spurious branches.

**Reproducibility.** Ties between attributes are broken by column order, threshold search
uses a stable sort, and the split uses a configurable seed.

## Complexity

With *n* samples, *m* attributes and tree depth *d*:

* ID3: O(*m · n*) per level → O(*d · m · n*).
* C4.5: sorting for thresholds adds a log factor → O(*d · m · n* log *n*).
* Pruning is a single O(nodes) bottom-up pass.

## Extending

* A new split criterion (e.g. Gini) is a function in `criteria.py` plus a subclass that
  calls it in `_build`.
* A new algorithm only needs `fit`, `predict`, `predict_proba` and `summary`; add it to
  `experiment.run_all` and it appears in the comparison automatically.
