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

| Page | What it shows |
|------|---------------|
| Dashboard | All projects, create new |
| Data Explorer | Upload CSV/Parquet, EDA stats, missing values chart |
| Feature Engineering | Feature spec editor, correlation heatmap, importance bars |
| Training | Model selector, run history with live status |
| Results | Leaderboard, winner card with confidence interval, accuracy vs training time chart |
| Error Analysis | Confusion matrix, per-class error rates, feature importance breakdown |

---

## 12 Feature Engineering Techniques

Implemented in `features/features.py`. Every function is pure — returns a new DataFrame, never mutates input. Driven by a JSON spec stored in `FeatureSet.features_list`.

| # | Technique | What it does |
|---|-----------|--------------|
| 1 | **Numerical passthrough** | Selects and validates numeric columns as-is |
| 2 | **Polynomial features** | Raises columns to powers: `age²`, `age³`, etc. |
| 3 | **Interaction features** | Multiplies column pairs: `age × income` |
| 4 | **Lag features** | Time-series look-back: `revenue_lag_1`, `_lag_7`, `_lag_30` |
| 5 | **Rolling window** | Rolling mean/std/sum over configurable windows |
| 6 | **Categorical encoding** | Label encoding (ordinal int) or one-hot encoding |
| 7 | **Domain-specific features** | Business logic: `avg_order_value`, `recency_score`, `revenue_per_visit`, `purchase_frequency`, `customer_lifetime_value` |
| 8 | **Time / date features** | Extracts `day_of_week`, `month`, `quarter`, `hour`, `week_of_year`, `is_month_start`, `is_month_end`, `is_weekend` |
| 9 | **Binning / discretization** | Uniform (equal-width) or quantile (equal-frequency) bins |
| 10 | **Statistical / group aggregation** | Group-by aggregates (`mean`, `std`, `count`) broadcast back to every row |
| 11 | **Ratio features** | `numerator / denominator` pairs with division-by-zero safety |
| 12 | **Log transform** | `log1p` to reduce right skew; clips negatives before transforming |

### Feature spec format

```json
{
  "features": [
    {"name": "age",          "source": "raw",         "type": "numeric"},
    {"name": "age_squared",  "source": "polynomial",  "base": "age",    "degree": 2},
    {"name": "age_x_income", "source": "interaction", "operands": ["age", "income"]},
    {"name": "tenure_log",   "source": "log",         "base": "tenure"},
    {"name": "signup_month", "source": "time",        "base": "signup_date", "component": "month"}
  ]
}
```

---

## 6 Feature Selection Methods

Implemented in `features/feature_selector.py`. All methods return a ranked list of selected columns.

| Method | How it works |
|--------|--------------|
| **Variance threshold** | Drops columns whose variance falls below a threshold |
| **Statistical selection** | Ranks by absolute Pearson correlation with the target |
| **Forward selection** | Greedily adds the feature that most improves cross-val score |
| **Backward elimination** | Starts with all features, removes the weakest one at each step |
| **Recursive feature elimination (RFE)** | Uses a model's `coef_` or `feature_importances_` to prune |
| **Information value (IV)** | WOE-based binning; IV > 0.02 = weak, > 0.1 = medium, > 0.3 = strong |

---

## 5 Model Families

Implemented in `models/model_definitions.py`. All support both classification and regression. Hyperparameters are fully configurable via the API; defaults shown below.

### 1. Logistic Regression
Linear model with L2 regularisation (default). Fast, interpretable.
```
penalty=l2 · C=1.0 · solver=lbfgs · max_iter=1000
```

### 2. Support Vector Machine
RBF kernel (non-linear), soft-margin with probability estimates enabled.
```
kernel=rbf · C=1.0 · gamma=scale · probability=True
```

### 3. Random Forest
Ensemble of 100 decision trees, feature subsampling per split, fully parallel.
```
n_estimators=100 · max_depth=None · max_features=sqrt · n_jobs=-1
```

### 4. XGBoost
Gradient boosted trees with row and column subsampling per round.
```
n_estimators=100 · max_depth=6 · learning_rate=0.1 · subsample=0.8 · colsample_bytree=0.8
```

### 5. Neural Network (MLP)
3-layer perceptron with adaptive learning rate and early stopping.
```
hidden_layers=(128, 64, 32) · activation=relu · learning_rate=adaptive · early_stopping=True
```

---

## Evaluation

Implemented in `evaluation/evaluator.py`, `comparator.py`, and `analysis.py`.

**Classification metrics:** accuracy, precision, recall, F1, ROC-AUC, confusion matrix, per-class error rates

**Regression metrics:** R², RMSE, MAE, residual distribution

**Model comparison:**
- McNemar's test for pairwise statistical significance
- 95% Wilson confidence interval on accuracy
- Leaderboard with rank, metric value, and margin over second place

**Error analysis:**
- Confusion matrix with per-class breakdown
- Feature importance (mean decrease in impurity for tree models)
- Error examples — rows the model misclassified

---

## MLflow Tracking

Every training run is optionally logged to MLflow:
- Hyperparameters as run params
- Metrics (accuracy, F1, ROC-AUC, training time)
- Model artifact saved to the MLflow artifact store
- Silent fallback if MLflow is unavailable — the pipeline never breaks

Access the MLflow UI at **http://localhost:5001** when running via Docker.

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
| `GET` | `/projects/{id}/data_profile` | Shape, dtypes, missing values, distributions |
| `GET` | `/projects/{id}/missing_values` | Per-column missing report |

### Feature engineering
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/projects/{id}/feature_sets` | Create feature set from JSON spec |
| `GET` | `/projects/{id}/feature_sets` | List feature sets |
| `GET` | `/projects/{id}/feature_sets/{id}/preview` | Preview engineered features (first 5 rows) |
| `GET` | `/projects/{id}/feature_importance` | Correlation scores with target |
| `POST` | `/projects/{id}/select_features` | Run feature selection |
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
| `GET` | `/projects/{id}/experiments` | MLflow experiment details + UI link |
| `GET` | `/projects/{id}/experiments/best` | Best MLflow run by metric |

---

## Project structure

```
ml-forge/
├── frontend/                   ← React dashboard (Vite + TS + Tailwind + Recharts)
│   ├── src/
│   │   ├── api/                ← Axios client + typed endpoint functions
│   │   ├── components/
│   │   │   ├── charts/         ← ConfusionMatrix, FeatureImportanceBar, CorrelationHeatmap, ...
│   │   │   ├── layout/         ← Sidebar, TopBar
│   │   │   └── ui/             ← Card, Badge, Spinner
│   │   ├── pages/              ← Dashboard, DataExplorer, FeatureEngineering, Training, Results, Analysis
│   │   └── types/api.ts        ← TypeScript interfaces matching backend responses
│   ├── Dockerfile              ← multi-stage: node builder → nginx
│   └── nginx.conf              ← serves SPA, proxies /projects to API container
├── api/app.py                  ← FastAPI route handlers (all endpoints + CORS + lifespan)
├── pipeline.py                 ← 7-step E2E orchestrator (CLI entry point)
├── main.py                     ← API entry point
├── data/
│   ├── data_loader.py          ← load_dataset() — CSV, Parquet, SQL; dataset_statistics()
│   └── preprocessor.py        ← handle_missing_values, detect_outliers, normalize, encode
├── db/
│   ├── models.py               ← SQLAlchemy 2.0 ORM (Mapped / mapped_column)
│   ├── schema.sql              ← PostgreSQL DDL — source of truth
│   └── session.py             ← engine + get_db() dependency
├── features/
│   ├── features.py             ← 12 feature engineering functions
│   ├── feature_builder.py      ← build_feature_set() + validate_feature_definitions()
│   ├── feature_selector.py     ← 6 selection methods
│   └── normalizer.py          ← stateful Scaler (fit/transform separated)
├── models/
│   ├── model_definitions.py    ← 5 model families + hyperparameter defaults
│   └── trainer.py             ← train_multiple(), save_model(), load_model()
├── evaluation/
│   ├── evaluator.py            ← classification + regression metrics, Wilson CI
│   ├── comparator.py          ← McNemar test, leaderboard, tiebreak logic
│   └── analysis.py            ← confusion matrix, feature importance, residuals
├── tracking/
│   └── mlflow_integration.py  ← MLflowTracker class, silent fallback
├── tests/                     ← 213 passing pytest tests
│   ├── test_data_cleaning.py
│   ├── test_features.py
│   └── test_integration.py    ← reproducibility + feature impact E2E tests
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
  ├── /           → React SPA (static build)
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
