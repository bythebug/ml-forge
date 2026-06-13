-- ml-forge schema
-- Requires PostgreSQL 13+ (JSONB, indexes)

-- ─────────────────────────────────────────
-- Users
-- ─────────────────────────────────────────
CREATE TABLE users (
    id          SERIAL          PRIMARY KEY,
    email       VARCHAR(255)    NOT NULL UNIQUE,
    created_at  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_users_email ON users (email);


-- ─────────────────────────────────────────
-- Projects
-- ─────────────────────────────────────────
CREATE TABLE projects (
    id              SERIAL          PRIMARY KEY,
    user_id         INTEGER         NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    name            VARCHAR(255)    NOT NULL,
    description     TEXT,
    dataset_path    VARCHAR(512),
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_projects_user_id ON projects (user_id);


-- ─────────────────────────────────────────
-- Feature Sets
-- JSONB shape:
--   {
--     "features": [
--       {"name": "age",                   "type": "numeric",  "source": "raw"},
--       {"name": "age_squared",           "type": "numeric",  "source": "polynomial",  "base": "age"},
--       {"name": "interaction_age_income","type": "numeric",  "source": "interaction", "operands": ["age","income"]}
--     ]
--   }
-- ─────────────────────────────────────────
CREATE TABLE feature_sets (
    id              SERIAL          PRIMARY KEY,
    project_id      INTEGER         NOT NULL REFERENCES projects (id) ON DELETE CASCADE,
    name            VARCHAR(255)    NOT NULL,
    features_list   JSONB           NOT NULL,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_feature_sets_project_id  ON feature_sets (project_id);
-- GIN index enables fast JSONB containment queries (@>, ?)
CREATE INDEX idx_feature_sets_features    ON feature_sets USING GIN (features_list);


-- ─────────────────────────────────────────
-- Training Runs
-- JSONB shape (metrics):
--   {"accuracy": 0.95, "f1": 0.93, "roc_auc": 0.97, "train_time_s": 42.1}
-- ─────────────────────────────────────────
CREATE TABLE training_runs (
    id              SERIAL          PRIMARY KEY,
    project_id      INTEGER         NOT NULL REFERENCES projects (id) ON DELETE CASCADE,
    feature_set_id  INTEGER         NOT NULL REFERENCES feature_sets (id) ON DELETE CASCADE,
    model_type      VARCHAR(100)    NOT NULL,
    metrics         JSONB,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_training_runs_project_id     ON training_runs (project_id);
CREATE INDEX idx_training_runs_feature_set_id ON training_runs (feature_set_id);
-- Allows filtering/sorting runs by a specific metric, e.g. metrics->>'accuracy'
CREATE INDEX idx_training_runs_metrics        ON training_runs USING GIN (metrics);


-- ─────────────────────────────────────────
-- Feature History
-- ─────────────────────────────────────────
CREATE TABLE feature_history (
    id                  SERIAL          PRIMARY KEY,
    feature_set_id      INTEGER         NOT NULL REFERENCES feature_sets (id) ON DELETE CASCADE,
    feature_name        VARCHAR(255)    NOT NULL,
    importance_score    FLOAT
);

CREATE INDEX idx_feature_history_feature_set_id ON feature_history (feature_set_id);
CREATE INDEX idx_feature_history_feature_name   ON feature_history (feature_name);
