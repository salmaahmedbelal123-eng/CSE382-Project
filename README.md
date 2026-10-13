# City Hotel Booking Demand — Regression, Classification & Clustering
CSE382 – Introduction to Machine Learning | End-to-End ML Project (Dataset: H2.csv, City Hotel)

## Submission info
| Milestone | GitHub URL | Commit SHA |
|-----------|-----------|------------|
| 1 | https://github.com/<org>/<repo> | `<paste SHA>` |
| 2 | https://github.com/<org>/<repo> | `<paste SHA>` |

## Team
| Name | Student ID | GitHub account | Main contributions |
|------|-----------|----------------|--------------------|
|      |           | @              |                    |
|      |           | @              |                    |
|      |           | @              |                    |
|      |           | @              |                    |

## Setup
```bash
git clone https://github.com/<org>/<repo>.git
cd <repo>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Get the data
Place H2.csv in `data/raw/` (link in `config.yaml`).

## Run commands
```bash
python main.py analyze
python main.py train --task regression --model random_forest
python main.py train --task classification --model xgboost
python main.py tune --task regression --model xgboost
python main.py evaluate --task classification --model all
python main.py cluster --algorithm kmeans
python main.py predict --task regression
python main.py predict --task classification
```
(Update this list to match what you actually implement.)

## Project structure
See the tree in the course spec, section 15. Reports are in `reports/`, design decisions in `docs/`.

## Reproducibility
Fixed seed = 42 (see `config.yaml`). Trained artifacts are NOT in git; regenerate with the train/tune commands.
