import numpy as np
import pandas as pd

from src.classifier import FraudSpikeClassifier


def test_classifier_evaluates_binary_predictions():
    X = pd.DataFrame({"a": [0.1, 0.2, 0.3, 0.8, 0.9, 0.7], "b": [0.2, 0.3, 0.1, 0.9, 0.8, 0.7]})
    y = np.array([0, 0, 0, 1, 1, 1])
    model = FraudSpikeClassifier().fit(X, y)
    metrics = model.evaluate(X, y)
    assert set(metrics) == {"auroc", "auprc", "precision", "recall", "f2"}


def test_cross_validate_reports_confusion_matrix_and_positives():
    y = pd.Series([0, 0, 0, 0, 1, 1, 0, 0] * 2)
    X = pd.DataFrame({"a": np.arange(len(y))})
    res = FraudSpikeClassifier.cross_validate(X, y, cv=4)
    assert set(res["avg"]) >= {"precision", "recall", "f2", "total_positives"}
    assert res["avg"]["total_positives"] == int(y.sum())
    fold_positives = [f["fold_positives"] for f in res["folds"]]
    assert len(fold_positives) == 4
    assert sum(fold_positives) == int(y.sum())  # every positive evaluated once
    cm = res["confusion_matrix"]
    assert cm[1][1] + cm[1][0] == int(y.sum())  # all positives accounted for in CM
