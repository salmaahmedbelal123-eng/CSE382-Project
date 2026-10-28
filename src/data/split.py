"""Data pipeline: read cleaned data -> split into train/val/test -> save.

Run from the project root:
    python -m src.data.split

Input:
    data/interim/cleaned.csv   (created by the cleaning cell in 01_eda.ipynb)

Outputs (not tracked by git, see .gitignore):
    data/processed/train.csv
    data/processed/val.csv
    data/processed/test.csv
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loader import PROJECT_ROOT

SEED = 42                  # fixed seed so everyone gets the identical split
TARGET = "IsCanceled"
TEST_SIZE = 0.15
VAL_SIZE = 0.15

CLEAN_PATH = PROJECT_ROOT / "data" / "interim" / "cleaned.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed"


def load_clean() -> pd.DataFrame:
    if not CLEAN_PATH.is_file():
        raise FileNotFoundError(
            f"{CLEAN_PATH} not found. Run the cleaning cell in notebooks/01_eda.ipynb "
            "to create it, or pull it from git."
        )
    return pd.read_csv(CLEAN_PATH)


def make_splits(df: pd.DataFrame):
    """Stratified 70/15/15 split on the target."""
    # Step 1: split off the TEST set. Nobody looks at it until the end.
    train_val, test = train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED, stratify=df[TARGET]
    )
    # Step 2: split the rest into TRAIN and VALIDATION.
    # 15% of the original data = 15/85 of what is left.
    train, val = train_test_split(
        train_val,
        test_size=VAL_SIZE / (1 - TEST_SIZE),
        random_state=SEED,
        stratify=train_val[TARGET],
    )
    return train, val, test


def check_splits(df, train, val, test) -> None:
    assert len(train) + len(val) + len(test) == len(df), "rows were lost or duplicated"
    overlaps = (
        len(set(train.index) & set(val.index)),
        len(set(train.index) & set(test.index)),
        len(set(val.index) & set(test.index)),
    )
    assert overlaps == (0, 0, 0), f"splits overlap: {overlaps}"


def main() -> None:
    df = load_clean()

    train, val, test = make_splits(df)
    check_splits(df, train, val, test)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(f"{name:5s} rows={len(part):6d}  cancel rate={part[TARGET].mean():.4f}")

    print(f"Total rows: {len(df)}")
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()