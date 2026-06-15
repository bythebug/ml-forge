import client from "./client";
import type {
  AnalysisResponse,
  BestModelResponse,
  ComparisonResponse,
  DataProfileResponse,
  FeatureImportanceResponse,
  FeaturePreview,
  FeatureSet,
  FeatureStatsResponse,
  LoadDataResponse,
  MissingValuesResponse,
  Project,
  ProjectCreate,
  RunProgress,
  RunSummary,
  TrainRunResponse,
} from "../types/api";

const p = (id: number) => `/projects/${id}`;

export const api = {
  projects: {
    list: () => client.get<Project[]>("/projects").then((r) => r.data),
    create: (body: ProjectCreate) =>
      client.post<Project>("/projects", body).then((r) => r.data),
  },

  data: {
    upload: (projectId: number, file: File) => {
      const form = new FormData();
      form.append("file", file);
      return client
        .post<LoadDataResponse>(`${p(projectId)}/load_data`, form, {
          headers: { "Content-Type": "multipart/form-data" },
        })
        .then((r) => r.data);
    },
    profile: (projectId: number) =>
      client.get<DataProfileResponse>(`${p(projectId)}/data_profile`).then((r) => r.data),
    missing: (projectId: number) =>
      client.get<MissingValuesResponse>(`${p(projectId)}/missing_values`).then((r) => r.data),
  },

  features: {
    list: (projectId: number) =>
      client.get<FeatureSet[]>(`${p(projectId)}/feature_sets`).then((r) => r.data),
    create: (projectId: number, name: string, features_spec: object) =>
      client
        .post<FeatureSet>(`${p(projectId)}/feature_sets`, { name, features_spec })
        .then((r) => r.data),
    preview: (projectId: number, featureSetId: number) =>
      client
        .get<FeaturePreview>(`${p(projectId)}/feature_sets/${featureSetId}/preview`)
        .then((r) => r.data),
    importance: (projectId: number, target: string, method = "correlation") =>
      client
        .get<FeatureImportanceResponse>(`${p(projectId)}/feature_importance`, {
          params: { target, method },
        })
        .then((r) => r.data),
    stats: (projectId: number) =>
      client.get<FeatureStatsResponse>(`${p(projectId)}/feature_stats`).then((r) => r.data),
  },

  training: {
    run: (
      projectId: number,
      body: {
        feature_set_id: number;
        target_col: string;
        models: { type: string; hyperparams: Record<string, unknown> }[];
        task?: "classification" | "regression";
        test_size?: number;
      }
    ) => client.post<TrainRunResponse>(`${p(projectId)}/train_run`, body).then((r) => r.data),
    list: (projectId: number) =>
      client.get<RunSummary[]>(`${p(projectId)}/runs`).then((r) => r.data),
    progress: (projectId: number, runId: number) =>
      client
        .get<RunProgress>(`${p(projectId)}/runs/${runId}/progress`)
        .then((r) => r.data),
  },

  results: {
    comparison: (projectId: number, metric = "accuracy") =>
      client
        .get<ComparisonResponse>(`${p(projectId)}/comparison`, { params: { metric } })
        .then((r) => r.data),
    bestModel: (projectId: number, metric = "accuracy") =>
      client
        .get<BestModelResponse>(`${p(projectId)}/best_model`, { params: { metric } })
        .then((r) => r.data),
  },

  analysis: {
    run: (projectId: number, runId: number, target_col: string, task = "classification") =>
      client
        .get<AnalysisResponse>(`${p(projectId)}/runs/${runId}/analysis`, {
          params: { target_col, task },
        })
        .then((r) => r.data),
  },
};
