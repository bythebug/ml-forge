export interface Project {
  id: number;
  name: string;
  description: string | null;
  dataset_path: string | null;
  run_count: number;
  best_accuracy: number | null;
  recent_model_types: string[];
  created_at: string;
}

export interface ProjectCreate {
  name: string;
  description?: string;
}

export interface LoadDataResponse {
  project_id: number;
  dataset_path: string;
  rows: number;
  columns: number;
  column_names: string[];
}

export interface ColumnStats {
  count: number;
  null_count: number;
  null_pct: number;
  dtype: string;
  unique_count?: number;
  mean?: number;
  std?: number;
  min?: number;
  max?: number;
}

export interface DataProfileResponse {
  project_id: number;
  shape: { rows: number; columns: number };
  columns: string[];
  dtypes: Record<string, string>;
  missing: Record<string, number>;
  missing_pct: Record<string, number>;
  duplicates: number;
  numeric_stats: Record<string, Record<string, number>>;
  categorical_stats: Record<string, { unique: number; top: string | null; freq: number }>;
}

export interface MissingColumn {
  column: string;
  missing_count: number;
  missing_pct: number;
}

export interface MissingValuesResponse {
  project_id: number;
  total_rows: number;
  columns: MissingColumn[];
}

export interface FeatureSet {
  id: number;
  name: string;
  feature_count: number;
  created_at: string;
}

export interface FeaturePreview {
  feature_set_id: number;
  feature_set_name: string;
  original_columns: string[];
  engineered_columns: string[];
  preview: Record<string, unknown>[];
}

export interface ImportanceScore {
  feature: string;
  score: number;
}

export interface FeatureImportanceResponse {
  project_id: number;
  target: string;
  method: string;
  importance_scores: Record<string, number>;
}

export interface FeatureStatsResponse {
  project_id: number;
  numeric_columns: string[];
  correlation_matrix: Record<string, Record<string, number | null>>;
  distribution_stats: Record<string, {
    mean: number;
    std: number;
    skewness: number;
    kurtosis: number;
    min: number;
    max: number;
  }>;
}

export interface RunSummary {
  run_id: number;
  model_type: string;
  feature_set_id: number;
  status: string;
  primary_metric: Record<string, number> | null;
  training_time_s: number | null;
  created_at: string;
}

export interface TrainRunResult {
  run_id: number;
  model_type: string;
  metrics: Record<string, unknown>;
  created_at: string;
}

export interface TrainRunResponse {
  project_id: number;
  feature_set_id: number;
  runs: TrainRunResult[];
}

export interface RunProgress {
  run_id: number;
  project_id: number;
  model_type: string;
  feature_set_id: number;
  status: string;
  metrics: Record<string, unknown> | null;
  created_at: string;
}

export interface ComparisonRun {
  run_id: number;
  model_type: string;
  feature_set_id: number;
  metrics: Record<string, number>;
  rank: number;
}

export interface ComparisonResponse {
  project_id: number;
  primary_metric: string;
  leaderboard: ComparisonRun[];
  winner: ComparisonRun | null;
  pairwise_significance: Record<string, unknown>[] | null;
  total_runs: number;
}

export interface BestModelResponse {
  project_id: number;
  best_run: ComparisonRun;
  recommendation: string;
  margin: number | null;
  confidence_interval_95?: { lower: number; upper: number };
}

export interface ConfusionMatrixAnalysis {
  confusion_matrix: number[][];
  total_samples: number;
  total_correct: number;
  overall_error_rate: number;
  error_rate_per_class: Record<string, {
    total: number;
    correct: number;
    errors: number;
    error_rate: number;
  }>;
  hardest_class: string;
  most_confused_pairs: unknown[];
}

export interface FeatureImportanceResult {
  method: string;
  total_features: number;
  top_features: { feature: string; importance: number; importance_pct: number; cumulative_pct: number }[];
  features_for_80pct: number;
  features_for_95pct: number;
}

export interface AnalysisResponse {
  run_id: number;
  model_type: string;
  task: string;
  n_test_samples: number;
  confusion_matrix_analysis?: ConfusionMatrixAnalysis;
  residual_analysis?: Record<string, unknown>;
  error_examples: Record<string, unknown>[];
  feature_importance: FeatureImportanceResult;
}
