import { useState } from "react";
import { useParams } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Plus, Eye, AlertCircle } from "lucide-react";
import { api } from "../api/endpoints";
import Card from "../components/ui/Card";
import Spinner from "../components/ui/Spinner";
import CorrelationHeatmap from "../components/charts/CorrelationHeatmap";
import FeatureImportanceBar from "../components/charts/FeatureImportanceBar";

const DEFAULT_SPEC = JSON.stringify(
  {
    features: [
      { name: "age_squared", type: "polynomial", source: "age", degree: 2 },
      { name: "income_log", type: "log", source: "income" },
    ],
  },
  null,
  2
);

export default function FeatureEngineering() {
  const { id } = useParams<{ id: string }>();
  const projectId = Number(id);
  const qc = useQueryClient();

  const [specText, setSpecText] = useState(DEFAULT_SPEC);
  const [specName, setSpecName] = useState("feature_set_v1");
  const [specError, setSpecError] = useState<string | null>(null);
  const [selectedFs, setSelectedFs] = useState<number | null>(null);
  const [importTarget, setImportTarget] = useState("");

  const { data: featureSets, isLoading: fsLoading } = useQuery({
    queryKey: ["feature_sets", projectId],
    queryFn: () => api.features.list(projectId),
  });

  const { data: stats } = useQuery({
    queryKey: ["feature_stats", projectId],
    queryFn: () => api.features.stats(projectId),
    retry: false,
  });

  const { data: preview } = useQuery({
    queryKey: ["feature_preview", projectId, selectedFs],
    queryFn: () => api.features.preview(projectId, selectedFs!),
    enabled: selectedFs != null,
  });

  const { data: importance } = useQuery({
    queryKey: ["feature_importance", projectId, importTarget],
    queryFn: () => api.features.importance(projectId, importTarget),
    enabled: !!importTarget,
  });

  const createFs = useMutation({
    mutationFn: () => {
      let parsed: object;
      try {
        parsed = JSON.parse(specText);
      } catch {
        throw new Error("Invalid JSON in feature spec.");
      }
      return api.features.create(projectId, specName, parsed);
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["feature_sets", projectId] });
      setSpecError(null);
    },
    onError: (e) => setSpecError((e as Error).message),
  });

  const columns = stats?.numeric_columns ?? [];

  return (
    <div className="p-8 max-w-6xl space-y-6">
      <h1 className="text-2xl font-bold text-slate-800">Feature Engineering</h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Spec editor */}
        <Card title="Feature Spec Editor">
          <div className="space-y-3">
            <input
              className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={specName}
              onChange={(e) => setSpecName(e.target.value)}
              placeholder="Feature set name"
            />
            <textarea
              className="w-full border border-slate-200 rounded-lg px-3 py-2 text-xs font-mono focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
              rows={14}
              value={specText}
              onChange={(e) => setSpecText(e.target.value)}
            />
            {specError && (
              <p className="text-red-500 text-xs flex items-center gap-1">
                <AlertCircle className="w-3.5 h-3.5" /> {specError}
              </p>
            )}
            <button
              onClick={() => createFs.mutate()}
              disabled={createFs.isPending || !specName.trim()}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white text-sm rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50"
            >
              {createFs.isPending ? <Spinner size={4} /> : <Plus className="w-4 h-4" />}
              Save Feature Set
            </button>
          </div>
        </Card>

        {/* Feature sets list */}
        <Card title="Saved Feature Sets">
          {fsLoading ? (
            <div className="flex justify-center py-10"><Spinner /></div>
          ) : featureSets?.length === 0 ? (
            <p className="text-slate-400 text-sm text-center py-10">No feature sets yet.</p>
          ) : (
            <div className="space-y-2">
              {featureSets?.map((fs) => (
                <div
                  key={fs.id}
                  className={`flex items-center justify-between p-3 rounded-lg border cursor-pointer transition-colors ${
                    selectedFs === fs.id
                      ? "border-blue-300 bg-blue-50"
                      : "border-slate-200 hover:border-slate-300 hover:bg-slate-50"
                  }`}
                  onClick={() => setSelectedFs(fs.id === selectedFs ? null : fs.id)}
                >
                  <div>
                    <p className="text-sm font-medium text-slate-700">{fs.name}</p>
                    <p className="text-xs text-slate-400">
                      {fs.feature_count} feature{fs.feature_count !== 1 ? "s" : ""} ·{" "}
                      {new Date(fs.created_at).toLocaleDateString()}
                    </p>
                  </div>
                  <Eye className="w-4 h-4 text-slate-400" />
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>

      {/* Preview table */}
      {preview && (
        <Card title={`Preview — ${preview.feature_set_name}`}>
          <div className="overflow-auto">
            <table className="text-xs w-full">
              <thead>
                <tr>
                  {Object.keys(preview.preview[0] ?? {}).map((col) => (
                    <th
                      key={col}
                      className={`text-left pb-2 pr-3 font-semibold text-xs uppercase tracking-wide ${
                        preview.engineered_columns.includes(col)
                          ? "text-green-600"
                          : "text-slate-500"
                      }`}
                    >
                      {col}
                      {preview.engineered_columns.includes(col) && " ✦"}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {preview.preview.map((row, i) => (
                  <tr key={i} className="border-t border-slate-50">
                    {Object.entries(row).map(([col, val]) => (
                      <td
                        key={col}
                        className={`py-1.5 pr-3 ${
                          preview.engineered_columns.includes(col)
                            ? "text-green-700 font-medium bg-green-50"
                            : "text-slate-600"
                        }`}
                      >
                        {val == null ? <span className="text-slate-300">null</span> : String(val)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Correlation heatmap */}
      {stats && stats.numeric_columns.length > 1 && (
        <Card title="Correlation Heatmap">
          <CorrelationHeatmap
            matrix={stats.correlation_matrix}
            columns={stats.numeric_columns}
          />
        </Card>
      )}

      {/* Feature importance */}
      {columns.length > 0 && (
        <Card title="Feature Importance vs Target">
          <div className="flex items-center gap-3 mb-4">
            <select
              className="border border-slate-200 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={importTarget}
              onChange={(e) => setImportTarget(e.target.value)}
            >
              <option value="">Select target column…</option>
              {columns.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>
          {importance && (
            <FeatureImportanceBar
              items={Object.entries(importance.importance_scores).map(([feature, score]) => ({
                feature,
                importance: score,
              }))}
            />
          )}
          {!importTarget && (
            <p className="text-slate-400 text-sm text-center py-6">
              Select a target column to compute correlation scores.
            </p>
          )}
        </Card>
      )}
    </div>
  );
}
