# ml-forge

A feature engineering pipeline system for training, evaluating, and tracking ML models.

## Stack

- **API** — FastAPI
- **Database** — PostgreSQL + SQLAlchemy 2.0
- **ML** — scikit-learn, XGBoost
- **Tracking** — MLflow
- **Infra** — Docker Compose

---

## Quick start

### Option 1 — Docker (recommended)

```bash
docker-compose up -d
```

| Service  | URL                        |
|----------|----------------------------|
| API      | http://localhost:8000      |
| API docs | http://localhost:8000/docs |
| MLflow   | http://localhost:5000      |

### Option 2 — Local

**Prerequisites**: Python 3.11+, PostgreSQL running locally.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
export DATABASE_URL=postgresql://localhost/ml_forge
export MLFLOW_TRACKING_URI=http://localhost:5000   # optional

# 3. Run
python main.py
```

API available at http://localhost:8000/docs

---

## Run tests

```bash
pytest
```

---

## Project structure

```
ml-forge/
├── api/            # FastAPI endpoints
├── data/           # Data loading and preprocessing
├── db/             # SQLAlchemy models and schema
├── evaluation/     # Metrics, comparison, error analysis
├── features/       # Feature engineering and selection
├── models/         # Model definitions and trainer
├── tracking/       # MLflow integration
└── tests/          # pytest test suite
```

---

## API overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/projects/{id}/load_data` | Upload CSV or Parquet dataset |
| `GET`  | `/projects/{id}/data_profile` | EDA statistics |
| `GET`  | `/projects/{id}/missing_values` | Missing value report |
| `POST` | `/projects/{id}/feature_sets` | Create a feature set from spec |
| `GET`  | `/projects/{id}/feature_sets` | List feature sets |
| `GET`  | `/projects/{id}/feature_sets/{id}/preview` | Preview engineered features |
| `GET`  | `/projects/{id}/feature_importance` | Feature correlation scores |
| `POST` | `/projects/{id}/select_features` | Run feature selection |
| `GET`  | `/projects/{id}/feature_stats` | Correlation matrix |
| `POST` | `/projects/{id}/train_run` | Train one or more models |
| `GET`  | `/projects/{id}/runs` | List training runs |
| `GET`  | `/projects/{id}/runs/{id}/progress` | Run status and metrics |
| `GET`  | `/projects/{id}/comparison` | Leaderboard across all runs |
| `GET`  | `/projects/{id}/best_model` | Best model with confidence interval |
| `GET`  | `/projects/{id}/runs/{id}/analysis` | Error analysis |
| `GET`  | `/projects/{id}/experiments` | MLflow experiment details |
| `GET`  | `/projects/{id}/experiments/best` | Best MLflow run |
