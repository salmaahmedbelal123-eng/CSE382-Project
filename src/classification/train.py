"""Milestone 1: train UNTUNED (default settings) classification models and score them
on the VALIDATION set. The test set is not used.

Run from the project root folder:
    python -m src.classification.train

Author: Lujy (classification part)
"""
import time

import pandas as pd

from src.classification.features import ROOT, build_pipeline, load_split
from src.classification.metrics import evaluate
from src.classification.models import get_models

OUT_DIR = ROOT / "outputs" / "classification"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    print("Train:", X_train.shape, "| Val:", X_val.shape)
    print("Columns used:", list(X_train.columns))
    print("Cancellation rate in train:", round(y_train.mean(), 3))
    print("Accuracy of always guessing 'not canceled':", round(1 - y_val.mean(), 3))

    rows = []
    for name, model in get_models().items():
        pipe = build_pipeline(model, X_train)
        start = time.time()
        pipe.fit(X_train, y_train)                   # learns ONLY from the training rows
        seconds = round(time.time() - start, 1)
        scores = evaluate(pipe, X_val, y_val)        # scored on rows it has never seen
        rows.append({"model": name, "version": "baseline", **scores, "train_seconds": seconds})

    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "baseline_results.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
