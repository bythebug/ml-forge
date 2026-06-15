import { useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, AlertCircle } from "lucide-react";
import { api } from "../api/endpoints";
import Card from "../components/ui/Card";
import Spinner from "../components/ui/Spinner";
import ConfusionMatrix from "../components/charts/ConfusionMatrix";
import FeatureImportanceBar from "../components/charts/FeatureImportanceBar";

export default function Analysis() {
  const { id, runId } = useParams<{ id: string; runId: string }>();
  const projectId = Number(id);
  const navigate = useNavigate();

  const [targetCol, setTargetCol] = useState("");
  const [task, setTask] = useState<"classification" | "regression">("classification");
  const [submitted, setSubmitted] = useState(false);

  const { data: profile } = useQuery({
    queryKey: ["profile", projectId],
    queryFn: () => api.data.profile(projectId),
    retry: false,
  });

  const { data: analysis, isLoading, error, refetch } = useQuery({
    queryKey: ["analysis", projectId, runId, targetCol, task],
    queryFn: () => api.analysis.run(projectId, Number(runId), targetCol, task),
    enabled: submitted && !!targetCol,
  });

  const columns = profile?.columns ?? [];

  const cm = analysis?.confusion_matrix_analysis;
  const featureImportance = analysis?.feature_importance?.top_features ?? [];
  const perClass = cm?.error_rate_per_class ?? {};
  const overallAccuracy = cm ? 1 - cm.overall_error_rate : null;
  const worstClass = cm?.hardest_class ?? null;

  return (
    <div className="p-8 max-w-6xl space-y-6">
      <div className="flex items-center gap-3">
        <button
          onClick={() => navigate(`/projects/${projectId}/results`)}
          className="text-slate-400 hover:text-slate-600"
        >
          <ArrowLeft className="w-5 h-5" />
        </button>
        <div>
          <h1 className="text-2xl font-bold text-slate-800">Error Analysis</h1>
          <p className="text-sm text-slate-500">Run #{runId}</p>
        </div>
      </div>

      {/* Config */}
      <Card title="Analysis Configuration">
        <div className="flex items-end gap-4">
          <div>
            <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1">
              Target Column
            </label>
            <select
              className="border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={targetCol}
              onChange={(e) => { setTargetCol(e.target.value); setSubmitted(false); }}
            >
              <option value="">Select…</option>
              {columns.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1">
              Task
            </label>
            <select
              className="border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={task}
              onChange={(e) => setTask(e.target.value as "classification" | "regression")}
            >
              <option value="classification">Classification</option>
              <option value="regression">Regression</option>
            </select>
          </div>
          <button
            disabled={!targetCol}
            onClick={() => { setSubmitted(true); refetch(); }}
            className="px-4 py-2 bg-blue-600 text-white text-sm rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50"
          >
            Run Analysis
          </button>
        </div>
      </Card>

      {isLoading && (
        <div className="flex justify-center py-12"><Spinner size={8} /></div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-red-700 text-sm flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0" />
          {(error as Error).message}
        </div>
      )}

      {analysis && (
        <>
          {/* Summary */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              { label: "Test Samples", value: analysis.n_test_samples },
              {
                label: "Overall Accuracy",
                value: overallAccuracy != null ? `${(overallAccuracy * 100).toFixed(1)}%` : "—",
              },
              { label: "Classes", value: cm ? Object.keys(perClass).length : "—" },
              {
                label: "Worst Class",
                value: worstClass ?? "—",
              },
            ].map(({ label, value }) => (
              <div key={label} className="bg-white border border-slate-200 rounded-xl p-4 text-center shadow-sm">
                <div className="text-xl font-bold text-slate-800">{value}</div>
                <div className="text-xs text-slate-500 mt-1">{label}</div>
              </div>
            ))}
          </div>

          {worstClass && perClass[worstClass] && (
            <div className="flex items-center gap-2 p-4 bg-orange-50 border border-orange-200 rounded-xl text-sm text-orange-800">
              <AlertCircle className="w-4 h-4 shrink-0 text-orange-500" />
              Model struggles most with class <strong>{worstClass}</strong> — error rate{" "}
              {(perClass[worstClass].error_rate * 100).toFixed(0)}%,{" "}
              {perClass[worstClass].errors} errors out of {perClass[worstClass].total} samples
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Confusion matrix */}
            {cm && (
              <Card title="Confusion Matrix">
                <ConfusionMatrix data={cm} />
              </Card>
            )}

            {/* Per-class metrics */}
            {cm && Object.keys(perClass).length > 0 && (
              <Card title="Per-Class Error Rates">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-100 text-xs text-slate-500 font-semibold uppercase tracking-wide text-left">
                      <th className="pb-2 pr-4">Class</th>
                      <th className="pb-2 pr-4">Total</th>
                      <th className="pb-2 pr-4">Correct</th>
                      <th className="pb-2 pr-4">Errors</th>
                      <th className="pb-2">Error Rate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(perClass).map(([cls, m]) => (
                      <tr key={cls} className="border-b border-slate-50">
                        <td className="py-2 pr-4 font-medium text-slate-700">{cls}</td>
                        <td className="py-2 pr-4 text-slate-600">{m.total}</td>
                        <td className="py-2 pr-4 text-slate-600">{m.correct}</td>
                        <td className="py-2 pr-4 text-slate-600">{m.errors}</td>
                        <td className="py-2 text-slate-500">{(m.error_rate * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Card>
            )}
          </div>

          {/* Feature importance */}
          {featureImportance.length > 0 && (
            <Card title="Feature Importance (Top 15)">
              <FeatureImportanceBar items={featureImportance} topN={15} />
            </Card>
          )}

          {/* Error examples */}
          {analysis.error_examples.length > 0 && (
            <Card title="Hardest Misclassified Examples">
              <div className="overflow-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-slate-100 text-left text-slate-500 font-semibold uppercase tracking-wide">
                      {Object.keys(analysis.error_examples[0]).map((k) => (
                        <th key={k} className="pb-2 pr-3">{k}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {analysis.error_examples.slice(0, 10).map((row, i) => (
                      <tr key={i} className="border-b border-slate-50 hover:bg-slate-50">
                        {Object.values(row).map((v, j) => (
                          <td key={j} className="py-1.5 pr-3 text-slate-600">
                            {v == null ? "—" : String(v)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
