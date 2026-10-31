"""The classification models for Milestone 1 (default settings = untuned baseline).

Author: Lujy (classification part)
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

SEED = 42


def get_models():
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "random_forest": RandomForestClassifier(random_state=SEED, n_jobs=-1),
    }
