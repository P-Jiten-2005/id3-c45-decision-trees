"""Regenerate the sample datasets in data/ (iris and a synthetic loan dataset)."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

DATA = Path(__file__).resolve().parent.parent / "data"


def iris():
    ds = load_iris(as_frame=True)
    df = ds.frame.rename(columns=lambda c: c.replace(" (cm)", "").replace(" ", "_"))
    df["species"] = df.pop("target").map(dict(enumerate(ds.target_names)))
    df.to_csv(DATA / "iris.csv", index=False)


def loan(n: int = 400, seed: int = 7):
    """Synthetic, mixed-type dataset with ~5% missing values (not real data)."""
    rng = np.random.default_rng(seed)
    income = rng.normal(55, 18, n).clip(12, 150).round(1)          # thousands
    credit = rng.normal(680, 60, n).clip(450, 850).round()
    employment = rng.choice(["Salaried", "Self-employed", "Unemployed"], n, p=[.6, .3, .1])
    education = rng.choice(["Graduate", "Not Graduate"], n, p=[.65, .35])
    property_area = rng.choice(["Urban", "Semiurban", "Rural"], n)
    dependents = rng.choice([0, 1, 2, 3], n, p=[.45, .25, .2, .1])
    loan_amt = (income * rng.uniform(1.5, 4.5, n)).round(1)
    score = ((credit - 650) / 40 + (income - 50) / 25 - (loan_amt / income - 3)
             + np.where(employment == "Unemployed", -2.5, 0)
             + np.where(education == "Graduate", 0.4, 0)
             + np.where(property_area == "Semiurban", 0.3, 0)
             - 0.2 * dependents + rng.normal(0, 0.8, n))
    df = pd.DataFrame({
        "applicant_id": [f"A{i:04d}" for i in range(1, n + 1)],
        "income_k": income, "credit_score": credit, "employment": employment,
        "education": education, "property_area": property_area,
        "dependents": dependents, "loan_amount_k": loan_amt,
        "approved": np.where(score > 0, "Yes", "No"),
    })
    for col in ["income_k", "credit_score", "employment", "property_area"]:
        df.loc[rng.random(n) < 0.05, col] = np.nan
    df.to_csv(DATA / "loan_approval.csv", index=False)


if __name__ == "__main__":
    iris()
    loan()
    print("Wrote", sorted(p.name for p in DATA.glob("*.csv")))
