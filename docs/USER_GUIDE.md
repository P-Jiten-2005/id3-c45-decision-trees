# User guide

Start the app with `streamlit run app.py` and open <http://localhost:8501>.

## 1. Load a dataset

In the sidebar choose **Upload CSV** and select a file, or **Sample dataset** to try one of
the bundled examples.

CSV requirements:

* A header row with column names; one row per example.
* One column containing the class label.
* Comma, semicolon, tab or pipe delimiters are detected automatically.
* Empty cells and the tokens `?`, `NA`, `N/A`, `null`, `none`, `nan`, `-` are read as
  missing values. A text column that only contains numbers and such tokens is converted
  to numeric.

The **Dataset overview** shows row and column counts, missing cells, duplicate rows, a
preview, and a column summary with the detected type of each column.

## 2. Choose the target and preprocessing

* **Target attribute** — the column to predict. The app warns if it looks continuous,
  has a single class, or has classes with only one example (which prevents a stratified
  split).
* **Columns to exclude** — identifiers are pre-selected. Leave them excluded: ID3 would
  split on a unique ID and memorise the training data.
* **Missing values**
  * *Impute* — mode for categorical, median for numeric, computed on the training split.
  * *C4.5 native handling* — C4.5 keeps missing values and uses fractional weights; ID3
    (and the CART reference) still use imputation.
  * *Drop* — remove rows that have any missing feature value.
* **Bins for numeric features (ID3)** and **Binning** — ID3 needs categorical inputs, so
  numeric columns are cut into this many intervals. *Equal frequency* puts roughly the same
  number of training rows in each bin; *equal width* splits the value range evenly.
* **Test set size** and **Random seed** — control the train/test split (stratified by
  class when possible).
* **Feature types** — override the automatic categorical / continuous detection, e.g. to
  treat a numeric code as categorical.

## 3. Model settings (sidebar)

| Setting | Applies to | Effect |
|---|---|---|
| Limit tree depth / Max depth | both | Stop growing below this depth |
| Min samples to split a node | both | Nodes with fewer samples become leaves |
| Pessimistic error pruning | C4.5 | Turn pruning on or off |
| Confidence factor | C4.5 | Lower values prune more aggressively (default 0.25) |
| Min samples per branch | C4.5 | A split needs at least two branches this large (default 2) |
| Averaging | metrics | How per-class precision/recall/F1 are combined |
| scikit-learn CART reference | comparison | Adds an optimised library tree for context |

Click **Train ID3 and C4.5**. If you change any setting afterwards, a banner reminds you to
retrain.

## 4. Read the results

### Preprocessing tab
The fitted imputation values and bin edges, plus the first rows exactly as ID3 and C4.5
receive them.

### ID3 and C4.5 tabs
* **Metric cards** — test accuracy, precision, recall, F1 and training accuracy. A large
  gap between training and test accuracy indicates overfitting.
* **Decision tree** — grey boxes are tests (attribute, gain, gain ratio for C4.5, number of
  samples); coloured ellipses are leaves (predicted class, samples, class counts). Stronger
  colour means a purer leaf. For deep trees use **Levels to display**; hidden subtrees are
  marked. Download the full tree as DOT, SVG or PNG.
* **Decision rules** — every root-to-leaf path as an IF–THEN rule with its support and
  confidence.
* **Confusion matrix** and **per-class report**.
* **Test-set predictions** — every test row with actual class, prediction, confidence
  and correctness; filter to misclassified rows and download as CSV.

### Comparison tab
* A one-line verdict, the metrics chart and a full table including tree size and training
  time.
* **Attribute importance** — the share of total information gain each attribute
  contributes in each tree.
* **k-fold cross-validation** — re-runs preprocessing and training in every fold and
  reports mean ± standard deviation, which is more reliable than a single split,
  especially on small datasets.

### Predict tab
* Fill in the form to classify one example. Mark values as missing to see how each model
  copes; the decision path explains each prediction.
* Upload a CSV with the same feature columns for batch prediction and download the results.

## Troubleshooting

| Problem | Fix |
|---|---|
| "The target looks continuous" | Pick a column with discrete class labels, or bin the values first. |
| ID3 tree is huge and test accuracy is low | Exclude ID columns, reduce the number of bins, or limit the depth. |
| SVG/PNG download unavailable | Install the Graphviz system package (see README). |
| Non-stratified split notice | Some class has fewer than two examples; consider removing it. |
