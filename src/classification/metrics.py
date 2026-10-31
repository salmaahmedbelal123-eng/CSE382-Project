"""One function that computes every metric the project asks for.

Author: Lujy (classification part)
"""
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)


def evaluate(pipeline, X, y):
    """Return all classification metrics (class 1 = canceled)."""
    pred = pipeline.predict(X)
    proba = pipeline.predict_proba(X)[:, 1]          # probability of "canceled"
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred), 4),
        "recall": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "roc_auc": round(roc_auc_score(y, proba), 4),
        "pr_auc": round(average_precision_score(y, proba), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
