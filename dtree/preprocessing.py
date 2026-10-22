"""Data preparation for the decision-tree algorithms.

The :class:`Preprocessor` is *fitted on the training split only* and then
applied unchanged to the test split and to new data, so statistics such as
imputation values and bin edges never leak information from the test set.

Pipeline
--------
1. Drop rows whose target is missing; drop user-selected columns.
2. Normalise categorical columns to stripped strings and treat empty strings
   or common tokens such as ``"?"`` / ``"NA"`` as missing.
3. Missing values: impute (mode for categorical, median for continuous) or,
   for C4.5 only, keep them so the algorithm can use fractional weights.
4. For ID3: discretise continuous columns into ``n_bins`` intervals using
   equal-width or equal-frequency (quantile) binning.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .tree import format_number

MISSING_TOKENS = {"", "?", "na", "n/a", "nan", "null", "none", "-"}


def read_csv(file) -> pd.DataFrame:
    """Read a CSV, sniffing the delimiter, and normalise missing tokens."""
    df = pd.read_csv(file, sep=None, engine="python", skipinitialspace=True)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed: \d+$")]
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            s = df[col].astype("string").str.strip()
            s = s.mask(s.str.lower().isin(MISSING_TOKENS))
            numeric = pd.to_numeric(s, errors="coerce")
            # A text column that is really numeric (e.g. had "?" placeholders)
            if s.notna().sum() > 0 and numeric.notna().sum() == s.notna().sum():
                df[col] = numeric
            else:
                df[col] = s.astype(object).where(s.notna(), np.nan)
    return df


def detect_feature_types(df: pd.DataFrame, max_categories_for_int: int = 10) -> dict[str, str]:
    """Guess categorical vs continuous for each column.

    Integer columns with only a handful of distinct values (e.g. 0/1 flags or
    1–5 ratings) are treated as categorical; other numeric columns as
    continuous; everything else as categorical.
    """
    types = {}
    for col in df.columns:
        s = df[col].dropna()
        if pd.api.types.is_bool_dtype(s):
            types[col] = "categorical"
        elif pd.api.types.is_numeric_dtype(s):
            is_int = len(s) > 0 and np.all(np.mod(s, 1) == 0)
            types[col] = "categorical" if is_int and s.nunique() <= max_categories_for_int \
                else "continuous"
        else:
            types[col] = "categorical"
    return types


def suggest_id_columns(df: pd.DataFrame) -> list[str]:
    """Columns that look like row identifiers (unique per row or named *id)."""
    out = []
    n = len(df)
    for col in df.columns:
        name = col.lower()
        unique_ratio = df[col].nunique(dropna=True) / max(n, 1)
        if name in {"id", "index", "row", "rowid"} or name.endswith("_id") or \
                (unique_ratio > 0.95 and not pd.api.types.is_float_dtype(df[col]) and n > 20):
            out.append(col)
    return out


def target_problems(y: pd.Series) -> list[str]:
    """Warnings about a column chosen as a classification target."""
    msgs = []
    y = y.dropna()
    k = y.nunique()
    if k < 2:
        msgs.append("The target has fewer than two classes, so there is nothing to classify.")
    if pd.api.types.is_float_dtype(y) and k > 20:
        msgs.append("The target looks continuous (many distinct decimal values); "
                    "decision-tree *classification* needs discrete class labels.")
    elif k > 50:
        msgs.append(f"The target has {k} distinct classes; results will be hard to interpret.")
    counts = y.value_counts()
    if (counts < 2).any():
        msgs.append("Some classes have only one sample; the split cannot be stratified.")
    return msgs


@dataclass
class PreprocessConfig:
    target: str
    drop_columns: list[str] = field(default_factory=list)
    feature_types: dict[str, str] = field(default_factory=dict)
    missing_strategy: str = "impute"         # "impute" | "native" (C4.5) | "drop"
    n_bins: int = 4
    binning: str = "quantile"                # "quantile" | "uniform"
    test_size: float = 0.3
    random_state: int = 42


class Preprocessor:
    """Fit on training data, then transform any split consistently."""

    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.features_: list[str] = []
        self.types_: dict[str, str] = {}
        self.fill_values_: dict[str, object] = {}
        self.bin_edges_: dict[str, np.ndarray] = {}
        self.bin_labels_: dict[str, list[str]] = {}
        self.categories_: dict[str, list[str]] = {}
        self.numeric_ranges_: dict[str, tuple[float, float, float]] = {}

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        cfg = self.config
        self.features_ = [c for c in X.columns if c != cfg.target and c not in cfg.drop_columns]
        detected = detect_feature_types(X[self.features_])
        self.types_ = {c: cfg.feature_types.get(c, detected[c]) for c in self.features_}
        Xc = self._coerce(X[self.features_])
        for col, t in self.types_.items():
            s = Xc[col].dropna()
            if t == "continuous":
                med = float(s.median()) if len(s) else 0.0
                self.fill_values_[col] = med
                lo = float(s.min()) if len(s) else 0.0
                hi = float(s.max()) if len(s) else 0.0
                self.numeric_ranges_[col] = (lo, hi, med)
                self._fit_bins(col, s)
            else:
                mode = s.mode()
                self.fill_values_[col] = mode.iloc[0] if len(mode) else "missing"
                self.categories_[col] = sorted(s.unique().tolist(), key=str)
        return self

    def _fit_bins(self, col: str, s: pd.Series) -> None:
        cfg = self.config
        k = max(2, int(cfg.n_bins))
        if s.nunique() <= 1:
            edges = np.array([-np.inf, np.inf])
        elif cfg.binning == "uniform":
            inner = np.linspace(s.min(), s.max(), k + 1)[1:-1]
            edges = np.concatenate([[-np.inf], np.unique(inner), [np.inf]])
        else:
            inner = np.quantile(s, np.linspace(0, 1, k + 1)[1:-1])
            edges = np.concatenate([[-np.inf], np.unique(inner), [np.inf]])
        self.bin_edges_[col] = edges
        labels = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            if np.isinf(lo):
                labels.append(f"<= {format_number(hi)}" if not np.isinf(hi) else "all")
            elif np.isinf(hi):
                labels.append(f"> {format_number(lo)}")
            else:
                labels.append(f"({format_number(lo)}, {format_number(hi)}]")
        self.bin_labels_[col] = labels

    def _coerce(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, t in self.types_.items():
            if t == "continuous":
                X[col] = pd.to_numeric(X[col], errors="coerce")
            else:
                X[col] = X[col].map(_cat_value)
        return X

    def transform(self, X: pd.DataFrame, for_algorithm: str) -> pd.DataFrame:
        """Return features ready for ``"id3"``, ``"c45"`` or ``"cart"``.

        * id3  — missing values imputed, continuous columns discretised.
        * c45  — continuous columns kept numeric; missing values kept when the
                 strategy is ``"native"``, imputed otherwise.
        * cart — imputed and numeric (scikit-learn cannot take NaN here).
        """
        Xc = self._coerce(X[self.features_])
        keep_missing = for_algorithm == "c45" and self.config.missing_strategy == "native"
        if not keep_missing:
            for col in self.features_:
                Xc[col] = Xc[col].fillna(self.fill_values_[col])
        if for_algorithm == "id3":
            for col, t in self.types_.items():
                if t == "continuous":
                    Xc[col] = pd.cut(Xc[col], bins=self.bin_edges_[col],
                                     labels=self.bin_labels_[col]).astype(str)
        return Xc.reset_index(drop=True)

    def feature_types_for(self, algorithm: str) -> dict[str, str]:
        if algorithm == "id3":
            return {c: "categorical" for c in self.features_}
        return dict(self.types_)

    def describe(self) -> pd.DataFrame:
        rows = []
        for col in self.features_:
            t = self.types_[col]
            id3 = (f"{len(self.bin_labels_[col])} bins: " + ", ".join(self.bin_labels_[col])) \
                if t == "continuous" else f"{len(self.categories_.get(col, []))} categories"
            rows.append({"feature": col, "type": t, "fill value": str(self.fill_values_[col]),
                         "ID3 representation": id3})
        return pd.DataFrame(rows)


def _cat_value(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    s = str(v).strip()
    return np.nan if s.lower() in MISSING_TOKENS else s


@dataclass
class PreparedData:
    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: Preprocessor
    n_dropped_target: int
    n_dropped_missing: int
    stratified: bool


def prepare(df: pd.DataFrame, config: PreprocessConfig) -> PreparedData:
    """Clean the dataset, split it, and fit a :class:`Preprocessor` on train."""
    if config.target not in df.columns:
        raise ValueError(f"Target column '{config.target}' not found")
    data = df.copy()
    before = len(data)
    data = data[data[config.target].notna()]
    n_dropped_target = before - len(data)

    feature_cols = [c for c in data.columns
                    if c != config.target and c not in config.drop_columns]
    if not feature_cols:
        raise ValueError("No feature columns left after dropping columns")

    n_dropped_missing = 0
    if config.missing_strategy == "drop":
        b = len(data)
        data = data.dropna(subset=feature_cols)
        n_dropped_missing = b - len(data)
    if len(data) < 4:
        raise ValueError("Not enough rows left to train and test a model")

    y = data[config.target]
    y = y if pd.api.types.is_numeric_dtype(y) else y.map(_cat_value).astype(object)
    X = data.drop(columns=[config.target])

    counts = y.value_counts()
    stratify = y if (counts.min() >= 2 and len(counts) > 1) else None
    n_test = max(1, int(round(len(data) * config.test_size)))
    if stratify is not None and n_test < len(counts):
        stratify = None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state, stratify=stratify
    )
    pre = Preprocessor(config).fit(X_tr)
    return PreparedData(X_tr.reset_index(drop=True), X_te.reset_index(drop=True),
                        y_tr.reset_index(drop=True), y_te.reset_index(drop=True),
                        pre, n_dropped_target, n_dropped_missing, stratify is not None)
