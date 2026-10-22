"""Regenerate the figures used in README.md and docs/ (needs Graphviz installed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt  # noqa: E402

from dtree import ID3Classifier  # noqa: E402
from dtree.experiment import ModelParams, comparison_table, run_all
from dtree.preprocessing import PreprocessConfig, prepare, read_csv
from dtree.visualize import plot_confusion_matrix, plot_metric_comparison, render_tree_png

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "docs" / "images"


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    tennis = read_csv(ROOT / "data" / "play_tennis.csv")
    m = ID3Classifier().fit(tennis.drop(columns=["Day", "PlayTennis"]), tennis["PlayTennis"])
    render_tree_png(m, str(IMG / "id3_play_tennis.png"))

    iris = read_csv(ROOT / "data" / "iris.csv")
    res = run_all(prepare(iris, PreprocessConfig(target="species")), ModelParams())
    render_tree_png(res["C4.5"].model, str(IMG / "c45_iris.png"))

    loan = read_csv(ROOT / "data" / "loan_approval.csv")
    cfg = PreprocessConfig(target="approved", drop_columns=["applicant_id"])
    res = run_all(prepare(loan, cfg), ModelParams())
    table = comparison_table(res)
    fig = plot_metric_comparison(table[["Accuracy", "Precision", "Recall", "F1-score"]])
    fig.savefig(IMG / "comparison_loan.png", dpi=150, facecolor="white")
    plt.close(fig)
    ev = res["C4.5"].evaluation
    fig = plot_confusion_matrix(ev.confusion, ev.labels, "C4.5 — loan approval test set",
                                "Oranges")
    fig.savefig(IMG / "confusion_c45_loan.png", dpi=150, facecolor="white")
    plt.close(fig)
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
