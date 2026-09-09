"""Metrics for the 3-class recyclability head."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, recall_score)


def _collapse(y):
    return np.where(np.asarray(y) == "recyclable", "recyclable", "not")


def score(y_true, y_pred, labels) -> dict:
    out = {
        "acc": accuracy_score(y_true, y_pred),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "bin_acc": accuracy_score(_collapse(y_true), _collapse(y_pred)),
        "bin_recyc_recall": recall_score(_collapse(y_true), _collapse(y_pred),
                                         pos_label="recyclable", zero_division=0),
    }
    for lab, r in zip(labels, recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)):
        out[f"recall[{lab}]"] = r
    return out


def confusion(y_true, y_pred, labels) -> str:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    w = max(len(x) for x in labels) + 2
    rows = [" " * w + "".join(f"{l[:8]:>10}" for l in labels)]
    rows += [f"{labels[i]:>{w}}" + "".join(f"{v:>10}" for v in cm[i]) for i in range(len(labels))]
    return "\n".join(rows)
