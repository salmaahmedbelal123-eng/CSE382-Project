# H2.csv loader - preprocessing branch
from pathlib import Path
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)

def load_h2() -> pd.DataFrame:
    path = PROJECT_ROOT / load_config()["data"]["raw_path"]
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. See README, 'Getting the data'.")
    return pd.read_csv(path)