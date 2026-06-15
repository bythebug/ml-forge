import { useState } from "react";
import { useParams } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Play, XCircle, Clock } from "lucide-react";
import { api } from "../api/endpoints";
import Card from "../components/ui/Card";
import Badge from "../components/ui/Badge";
import Spinner from "../components/ui/Spinner";

const MODEL_TYPES = [
  { type: "logistic_regression", label: "Logistic Regression" },
  { type: "svm", label: "Support Vector Machine" },
  { type: "random_forest", label: "Random Forest" },
  { type: "xgboost", label: "XGBoost" },
  { type: "neural_network", label: "Neural Network" },
];

function statusBadge(status: string) {
  if (status === "completed") return <Badge label="Completed" variant="green" />;
  if (status === "failed") return <Badge label="Failed" variant="red" />;
  if (status === "running") return <Badge label="Running" variant="yellow" pulse />;
  return <Badge label={status} variant="gray" />;
}

export default function Training() {
  const { id } = useParams<{ id: string }>();
  const projectId = Number(id);
  const qc = useQueryClient();

  const [selected, setSelected] = useState<string[]>(["random_forest"]);
  const [targetCol, setTargetCol] = useState("");
  const [task, setTask] = useState<"classification" | "regression">("classification");
  const [featureSetId, setFeatureSetId] = useState<number | "">("");

  const { data: featureSets } = useQuery({
    queryKey: ["feature_sets", projectId],
    queryFn: () => api.features.list(projectId),
  });

  const { data: profile } = useQuery({
    queryKey: ["profile", projectId],
    queryFn: () => api.data.profile(projectId),
    retry: false,
  });

  const { data: runs, isLoading: runsLoading } = useQuery({
    queryKey: ["runs", projectId],
    queryFn: () => api.training.list(projectId),
    refetchInterval: 3000,
  });

  const trainMut = useMutation({
    mutationFn: () =>
      api.training.run(projectId, {
        feature_set_id: Number(featureSetId),
        target_col: targetCol,
        task,
        models: selected.map((type) => ({ type, hyperparams: {} })),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["runs", projectId] }),
  });

  const columns = profile ? Object.keys(profile.columns) : [];
  const canTrain = selected.length > 0 && targetCol && featureSetId !== "";

  return (
    <div className="p-8 max-w-6xl space-y-6">
      <h1 className="text-2xl font-bold text-slate-800">Training</h1>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Config panel */}
        <div className="lg:col-span-2 space-y-4">
          <Card title="Configuration">
            <div className="space-y-4">
              {/* Task type */}
              <div>
                <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
                  Task
                </label>
                <div className="flex rounded-lg border border-slate-200 overflow-hidden">
                  {(["classification", "regression"] as const).map((t) => (
                    <button
                      key={t}
                      onClick={() => setTask(t)}
                      className={`flex-1 py-2 text-sm font-medium transition-colors ${
                        task === t
                          ? "bg-blue-600 text-white"
                          : "text-slate-600 hover:bg-slate-50"
                      }`}
                    >
                      {t.charAt(0).toUpperCase() + t.slice(1)}
                    </button>
                  ))}
                </div>
              </div>

              {/* Feature set */}
              <div>
                <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
                  Feature Set
                </label>
                <select
                  className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  value={featureSetId}
                  onChange={(e) => setFeatureSetId(Number(e.target.value) || "")}
                >
                  <option value="">Select feature set…</option>
                  {featureSets?.map((fs) => (
                    <option key={fs.id} value={fs.id}>{fs.name}</option>
                  ))}
                </select>
              </div>

              {/* Target column */}
              <div>
                <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
                  Target Column
                </label>
                <select
                  className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  value={targetCol}
                  onChange={(e) => setTargetCol(e.target.value)}
                >
                  <option value="">Select target…</option>
                  {columns.map((c) => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                </select>
              </div>
            </div>
          </Card>

          {/* Model selector */}
          <Card title="Models">
            <div className="space-y-2">
              {MODEL_TYPES.map(({ type, label }) => (
                <label
                  key={type}
                  className="flex items-center gap-3 p-3 rounded-lg border border-slate-200 cursor-pointer hover:bg-slate-50 transition-colors"
                >
                  <input
                    type="checkbox"
                    checked={selected.includes(type)}
                    onChange={(e) =>
                      setSelected((prev) =>
                        e.target.checked ? [...prev, type] : prev.filter((t) => t !== type)
                      )
                    }
                    className="w-4 h-4 text-blue-600 rounded"
                  />
                  <span className="text-sm font-medium text-slate-700">{label}</span>
                </label>
              ))}
            </div>

            <button
              disabled={!canTrain || trainMut.isPending}
              onClick={() => trainMut.mutate()}
              className="mt-4 w-full flex items-center justify-center gap-2 py-2.5 bg-blue-600 text-white text-sm rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {trainMut.isPending ? (
                <><Spinner size={4} /> Training…</>
              ) : (
                <><Play className="w-4 h-4" /> Start Training</>
              )}
            </button>

            {trainMut.isError && (
              <p className="text-red-500 text-xs mt-2 flex items-center gap-1">
                <XCircle className="w-3.5 h-3.5" />
                {(trainMut.error as Error).message}
              </p>
            )}
          </Card>
        </div>

        {/* Run history */}
        <div className="lg:col-span-3">
          <Card title="Run History">
            {runsLoading && (
              <div className="flex justify-center py-10"><Spinner /></div>
            )}
            {!runsLoading && (!runs || runs.length === 0) && (
              <p className="text-slate-400 text-sm text-center py-10">
                No runs yet. Configure and start training.
              </p>
            )}
            {runs && runs.length > 0 && (
              <div className="overflow-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-100 text-left text-xs text-slate-500 font-semibold uppercase tracking-wide">
                      <th className="pb-2 pr-4">Model</th>
                      <th className="pb-2 pr-4">Status</th>
                      <th className="pb-2 pr-4">Metric</th>
                      <th className="pb-2 pr-4">Time</th>
                      <th className="pb-2">Date</th>
                    </tr>
                  </thead>
                  <tbody>
                    {runs.map((run) => {
                      const metricEntry = run.primary_metric
                        ? Object.entries(run.primary_metric)[0]
                        : null;
                      return (
                        <tr key={run.run_id} className="border-b border-slate-50 hover:bg-slate-50">
                          <td className="py-3 pr-4 font-medium text-slate-700">
                            {run.model_type.replace(/_/g, " ")}
                          </td>
                          <td className="py-3 pr-4">{statusBadge(run.status)}</td>
                          <td className="py-3 pr-4 text-slate-600">
                            {metricEntry ? (
                              <span>
                                <span className="text-slate-400 text-xs">{metricEntry[0]}: </span>
                                <span className="font-semibold">
                                  {(metricEntry[1] * 100).toFixed(1)}%
                                </span>
                              </span>
                            ) : (
                              "—"
                            )}
                          </td>
                          <td className="py-3 pr-4 text-slate-500 text-xs">
                            {run.training_time_s != null ? (
                              <span className="flex items-center gap-1">
                                <Clock className="w-3 h-3" />
                                {run.training_time_s.toFixed(1)}s
                              </span>
                            ) : (
                              "—"
                            )}
                          </td>
                          <td className="py-3 text-slate-400 text-xs">
                            {new Date(run.created_at).toLocaleString()}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}
