# ml-forge

A production-grade feature engineering pipeline system for training, evaluating, and tracking ML models — built across 8 phases.

## Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI + Pydantic |
| Database | PostgreSQL + SQLAlchemy 2.0 |
| ML | scikit-learn, XGBoost |
| Tracking | MLflow |
| Infra | Docker Compose, AWS ECS |

---

## Quick start

### Option 1 — Docker (recommended)

```bash
git clone https://github.com/bythebug/ml-forge
cd ml-forge
docker-compose up -d
```

| Service | URL |
|---------|-----|
| API + interactive docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |

### Option 2 — Local

```bash
pip install -r requirements.txt

export DATABASE_URL=postgresql://localhost/ml_forge
export MLFLOW_TRACKING_URI=http://localhost:5000   # optional

python main.py        # API at http://localhost:8000
```

### Run the pipeline directly

```bash
python pipeline.py --dataset data/churn.csv --target churn --models random_forest,xgboost
```

### Run tests

```bash
pytest                  # all tests
pytest tests/test_integration.py   # end-to-end only
```

---

## Data format

The pipeline accepts **CSV** or **Parquet** files.

```
age,income,tenure,churned
25,52000,12,0
34,78000,36,1
...
```

- All feature columns must be numeric (or encodable via the feature spec)
- Target column can be binary (0/1) for classification or continuous for regression
- Missing values are handled automatically (default: mean imputation)

### Feature spec format

Define custom feature engineering via JSON:

```json
{
  "features": [
    {"name": "age",              "type": "numeric",  "source": "raw"},
    {"name": "age_squared",      "type": "numeric",  "source": "polynomial", "base": "age", "degree": 2},
    {"name": "age_x_income",     "type": "numeric",  "source": "interaction", "operands": ["age", "income"]},
    {"name": "tenure_log",       "type": "numeric",  "source": "log",        "base": "tenure"},
    {"name": "signup_month",     "type": "categorical", "source": "time",    "base": "signup_date", "component": "month"}
  ]
}
```

---

## API overview

### Data management
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/projects/{id}/load_data` | Upload CSV or Parquet |
| `GET` | `/projects/{id}/data_profile` | Shape, dtypes, missing values, distributions |
| `GET` | `/projects/{id}/missing_values` | Per-column missing report for visualisation |

### Feature engineering
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/projects/{id}/feature_sets` | Create feature set from JSON spec |
| `GET` | `/projects/{id}/feature_sets` | List all feature sets |
| `GET` | `/projects/{id}/feature_sets/{id}/preview` | Preview engineered features |
| `GET` | `/projects/{id}/feature_importance` | Correlation scores with target |
| `POST` | `/projects/{id}/select_features` | Run selection (correlation / variance / forward / RFE) |
| `GET` | `/projects/{id}/feature_stats` | Correlation matrix + distribution stats |

### Training
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/projects/{id}/train_run` | Train one or more models |
| `GET` | `/projects/{id}/runs` | List all training runs |
| `GET` | `/projects/{id}/runs/{id}/progress` | Run status and metrics |

### Evaluation
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/projects/{id}/comparison` | Leaderboard across all runs |
| `GET` | `/projects/{id}/best_model` | Best model + confidence interval |
| `GET` | `/projects/{id}/runs/{id}/analysis` | Confusion matrix, feature importance, error examples |

### Experiment tracking
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/projects/{id}/experiments` | MLflow experiment details + UI link |
| `GET` | `/projects/{id}/experiments/best` | Best MLflow run by metric |

---

## Project structure

```
ml-forge/
├── pipeline.py         ← end-to-end orchestration (start here)
├── main.py             ← API entry point
├── api/                ← FastAPI route handlers
├── data/               ← data loading and preprocessing
├── db/                 ← SQLAlchemy models and PostgreSQL DDL
├── evaluation/         ← metrics, comparison, error analysis
├── features/           ← 12 engineering techniques, selection, scaling
├── models/             ← 5 model families + trainer
├── tracking/           ← MLflow integration
├── tests/              ← pytest suite (213 tests, all passing)
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Deploy to AWS

```bash
chmod +x deploy.sh
AWS_REGION=us-east-1 ENVIRONMENT=prod ./deploy.sh
```

Requires: AWS CLI configured, ECR repository and ECS cluster already provisioned.

---

## Architecture

```
Client
  │
  ▼
FastAPI (port 8000)
  ├── data/         load + preprocess
  ├── features/     engineer + select + scale
  ├── models/       train (LR, SVM, RF, XGBoost, NN)
  ├── evaluation/   metrics + compare + analyse
  └── tracking/     MLflow logging
        │
        ▼
  MLflow server (port 5000)
        │
  PostgreSQL (port 5432)
```
