"""dtree — from-scratch ID3 and C4.5 decision-tree classifiers."""
from .c45 import C45Classifier
from .id3 import ID3Classifier
from .metrics import evaluate
from .preprocessing import PreprocessConfig, Preprocessor, prepare, read_csv

__all__ = ["ID3Classifier", "C45Classifier", "evaluate", "PreprocessConfig",
           "Preprocessor", "prepare", "read_csv"]
__version__ = "1.0.0"
