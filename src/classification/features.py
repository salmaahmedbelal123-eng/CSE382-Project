"""Load the train/validation data and build the preprocessing for the classification task.

Author: Lujy (classification part)
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]      # the project root folder
PROCESSED = ROOT / "data" / "processed"         # where train.csv / val.csv / test.csv live
TARGET = "IsCanceled"

# PREDICTION POINT: we predict at the moment the booking is made.
# So we remove every column that is only known LATER (see docs/classification_milestone1.md).
LEAKY_COLUMNS = [
    "ReservationStatus",         # says "Canceled" / "Check-Out" -> it IS the answer
    "ReservationStatusDate",     # the date of that final status -> same problem
    "AssignedRoomType",          # decided at check-in; canceled bookings never get there
    "BookingChanges",            # counts changes until check-in OR cancellation
    "RequiredCarParkingSpaces",  # only guests who really arrive end up with parking
    "TotalOfSpecialRequests",    # grows during the stay; final value is stored
]


def clean_agent(a):
    """Missing agent = booking made without an agent -> its own category 'NoAgent'."""
    if pd.isna(a):
        return "NoAgent"
    return str(int(float(a)))


def load_split(name):
    """Read data/processed/<name>.csv and return X (inputs) and y (the answer)."""
    df = pd.read_csv(PROCESSED / f"{name}.csv")
    if "Agent" in df.columns:          # skip if the team already encoded Agent
        df["Agent"] = df["Agent"].apply(clean_agent)
    y = df[TARGET]
    X = df.drop(columns=[TARGET] + LEAKY_COLUMNS)
    return X, y


def build_preprocessor(X):
    """Fill missing values, scale numbers, one-hot encode text columns.

    Nothing is learned here. It is learned later, inside pipeline.fit(), on the
    TRAINING rows only (this is what prevents data leakage).
    """
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    numbers = Pipeline([
        ("fill", SimpleImputer(strategy="mean")),            # e.g. Children: mean of train
        ("scale", StandardScaler()),
    ])
    texts = Pipeline([
        ("fill", SimpleImputer(strategy="most_frequent")),   # e.g. Country: most frequent in train
        ("onehot", OneHotEncoder(
            min_frequency=0.01,                    # rare categories -> one "infrequent" column
            handle_unknown="infrequent_if_exist",  # unseen category later -> no crash
            sparse_output=False)),
    ])
    return ColumnTransformer([("num", numbers, num_cols), ("cat", texts, cat_cols)])


def build_pipeline(model, X):
    """preprocessing -> model, together in ONE object."""
    return Pipeline([("prep", build_preprocessor(X)), ("model", model)])
