# ml-forge

A production-grade ML pipeline — feature engineering, model training, evaluation, and experiment tracking — with a full React dashboard for visualising results.

## Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI + Pydantic |
| Database | PostgreSQL + SQLAlchemy 2.0 |
| ML | scikit-learn, XGBoost |
| Tracking | MLflow |
| Frontend | React 18 + Vite + TypeScript + Tailwind CSS + Recharts |
| Infra | Docker Compose, nginx, AWS ECS |

---

## Quick start

### Docker (recommended)

```bash
git clone https://github.com/bythebug/ml-forge
cd ml-forge
docker compose up -d
```

| Service | URL |
|---------|-----|
| **Dashboard** | http://localhost:3000 |
| **API docs** | http://localhost:8000/docs |
| **MLflow UI** | http://localhost:5001 |

### Local (API only)

```bash
pip install -r requirements.txt

export DATABASE_URL=postgresql://localhost/ml_forge
export MLFLOW_TRACKING_URI=http://localhost:5001   # optional

python main.py        # API at http://localhost:8000
```

### Run the pipeline directly (CLI)

```bash
python pipeline.py --dataset data/churn.csv --target churn --models random_forest,xgboost
```

### Run tests

```bash
pytest                             # all 213 tests
pytest tests/test_integration.py  # end-to-end only
```

---

## Dashboard

The frontend covers the full ML workflow in one place:

| Page | Path | What it shows |
|------|------|---------------|
| Dashboard | `/` | All projects, create new |
| Data Explorer | `/projects/:id/data` | Upload CSV/Parquet, EDA stats, missing values chart |
| Feature Engineering | `/projects/:id/features` | Feature spec editor, correlation heatmap, importance bars |
| Training | `/projects/:id/training` | Model selector, run history with live status |
| Results | `/projects/:id/results` | Leaderboard, winner card with CI, accuracy vs time chart |
| Error Analysis | `/projects/:id/analysis/:runId` | Confusion matrix, per-class error rates, feature importance |

---

## Data format

CSV or Parquet. All feature columns numeric (or encodable via feature spec). Target: binary 0/1 for classification or continuous for regression.

```
age,income,tenure,churned
25,52000,12,0
34,78000,36,1
```

### Feature spec (JSON)

```json
{
  "features": [
    {"name": "age_squared",  "type": "numeric", "source": "polynomial", "base": "age", "degree": 2},
    {"name": "tenure_log",   "type": "numeric", "source": "log",        "base": "tenure"},
    {"name": "age_x_income", "type": "numeric", "source": "interaction", "operands": ["age", "income"]}
  ]
}
```

---

## API reference

### Projects
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/projects` | List all projects |
| `POST` | `/projects` | Create project |

### Data
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/projects/{id}/load_data` | Upload CSV or Parquet |
| `GET` | `/projects/{id}/data_profile` | Shape, dtypes, distributions |
| `GET` | `/projects/{id}/missing_values` | Per-column missing report |

### Feature engineering
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/projects/{id}/feature_sets` | Create feature set from JSON spec |
| `GET` | `/projects/{id}/feature_sets` | List feature sets |
| `GET` | `/projects/{id}/feature_sets/{id}/preview` | Preview engineered features (first 5 rows) |
| `GET` | `/projects/{id}/feature_importance` | Correlation scores with target |
| `POST` | `/projects/{id}/select_features` | Run selection (correlation / variance / forward / RFE / IV) |
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
| `GET` | `/projects/{id}/comparison` | Leaderboard with McNemar significance test |
| `GET` | `/projects/{id}/best_model` | Best model + 95% Wilson confidence interval |
| `GET` | `/projects/{id}/runs/{id}/analysis` | Confusion matrix, feature importance, error examples |

### Experiment tracking
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/projects/{id}/experiments` | MLflow experiment + UI link |
| `GET` | `/projects/{id}/experiments/best` | Best MLflow run by metric |

---

## Project structure

```
ml-forge/
├── frontend/           ← React dashboard (Vite + TS + Tailwind + Recharts)
│   ├── src/
│   │   ├── api/        ← Axios client + typed endpoint functions
│   │   ├── components/ ← charts/, layout/, ui/ primitives
│   │   ├── pages/      ← 6 pages
│   │   └── types/      ← TypeScript types matching backend responses
│   ├── Dockerfile      ← multi-stage: node builder → nginx
│   └── nginx.conf      ← serves static files, proxies /projects to API
├── api/app.py          ← FastAPI route handlers (all endpoints)
├── pipeline.py         ← 7-step E2E orchestrator (CLI entry point)
├── main.py             ← API entry point
├── data/               ← loading and preprocessing
├── db/                 ← SQLAlchemy models and PostgreSQL DDL
├── evaluation/         ← metrics, comparison, error analysis
├── features/           ← 12 engineering techniques, selection, scaling
├── models/             ← 5 model families + trainer
├── tracking/           ← MLflow integration
├── tests/              ← 213 passing pytest tests
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Architecture

```
Browser (port 3000)
  │
  ▼
nginx (frontend container)
  ├── /           → React SPA (static files)
  └── /projects   → proxy → FastAPI (port 8000)
                        │
                        ├── data/       load + preprocess
                        ├── features/   engineer + select + scale
                        ├── models/     train (LR, SVM, RF, XGBoost, NN)
                        ├── evaluation/ metrics + compare + analyse
                        └── tracking/   MLflow logging
                                │
                          MLflow (port 5001)
                                │
                          PostgreSQL (port 5432)
```

---

## Deploy to AWS

```bash
chmod +x deploy.sh
AWS_REGION=us-east-1 ENVIRONMENT=prod ./deploy.sh
```

Requires: AWS CLI configured, ECR repository and ECS cluster provisioned.
