# Classification part (IsCanceled): Lujy, Milestone 1

Untuned Logistic Regression and Random Forest, scored on the validation set.

## Run (from the project root, after `python -m src.data.split` has created `data/processed/train.csv` and `val.csv`)
```
pip install pandas scikit-learn
python -m src.classification.train
```
Output: `outputs/classification/baseline_results.csv`

## Files
| File | What it does |
| --- | --- |
| src/classification/features.py | loads the data, drops leaking columns, builds the preprocessing + pipeline |
| src/classification/models.py | the models |
| src/classification/metrics.py | accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix |
| src/classification/train.py | trains the models and scores them on validation |
| docs/classification_milestone1.md | prediction point, feature choices, preliminary leakage audit, plan |
