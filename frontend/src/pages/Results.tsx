import { useParams, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Trophy, TrendingUp, Zap } from "lucide-react";
import { ScatterChart, Scatter, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ZAxis } from "recharts";
import { api } from "../api/endpoints";
import Card from "../components/ui/Card";
import Spinner from "../components/ui/Spinner";
import AccuracyComparison from "../components/charts/AccuracyComparison";

const MEDALS = ["🥇", "🥈", "🥉"];

function fmt(v: number | undefined) {
  return v != null ? (v * 100).toFixed(2) + "%" : "—";
}

export default function Results() {
  const { id } = useParams<{ id: string }>();
  const projectId = Number(id);
  const navigate = useNavigate();

  const { data: comparison, isLoading: cmpLoading, error: cmpError } = useQuery({
    queryKey: ["comparison", projectId],
    queryFn: () => api.results.comparison(projectId),
  });

  const { data: best } = useQuery({
    queryKey: ["best_model", projectId],
    queryFn: () => api.results.bestModel(projectId),
    enabled: !!comparison,
  });

  if (cmpLoading) {
    return (
      <div className="p-8 flex justify-center">
        <Spinner size={8} />
      </div>
    );
  }

  if (cmpError) {
    return (
      <div className="p-8">
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-6 text-amber-700 text-sm">
          No training runs found yet. Go to <strong>Training</strong> and train some models first.
        </div>
      </div>
    );
  }

  const ranked = comparison?.leaderboard ?? [];

  const scatterData = ranked.map((r) => ({
    name: r.model_type.replace(/_/g, " "),
    x: r.metrics.training_time_s ?? 0,
    y: r.metrics.accuracy != null ? +(r.metrics.accuracy * 100).toFixed(2) : 0,
    z: 100,
  }));

  return (
    <div className="p-8 max-w-6xl space-y-6">
      <h1 className="text-2xl font-bold text-slate-800">Results & Comparison</h1>

      {/* Winner card */}
      {best && (
        <div className="bg-gradient-to-r from-blue-600 to-indigo-600 rounded-2xl p-6 text-white shadow-lg">
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <Trophy className="w-5 h-5 text-yellow-300" />
                <span className="text-sm font-medium text-blue-200">Best Model</span>
              </div>
              <h2 className="text-2xl font-bold mb-1">
                {best.best_run.model_type.replace(/_/g, " ")}
              </h2>
              <div className="flex items-center gap-4 text-sm text-blue-100 mt-3">
                <span>
                  Accuracy:{" "}
                  <strong className="text-white">
                    {fmt(best.best_run.metrics.accuracy)}
                  </strong>
                </span>
                {best.confidence_interval_95 && (
                  <span className="text-blue-200 text-xs">
                    CI: [{(best.confidence_interval_95.lower * 100).toFixed(1)}% –{" "}
                    {(best.confidence_interval_95.upper * 100).toFixed(1)}%]
                  </span>
                )}
                {best.margin != null && best.margin > 0 && (
                  <span className="flex items-center gap-1">
                    <TrendingUp className="w-3.5 h-3.5 text-green-300" />
                    <span className="text-green-300">+{(best.margin * 100).toFixed(2)}% over #2</span>
                  </span>
                )}
              </div>
            </div>
            <button
              onClick={() =>
                navigate(`/projects/${projectId}/analysis/${best.best_run.run_id}`)
              }
              className="flex items-center gap-2 px-4 py-2 bg-white/20 hover:bg-white/30 text-white text-sm rounded-lg font-medium transition-colors"
            >
              <Zap className="w-4 h-4" /> Analyse
            </button>
          </div>
        </div>
      )}

      {/* Leaderboard */}
      <Card title="Leaderboard">
        <div className="overflow-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-100 text-left text-xs text-slate-500 font-semibold uppercase tracking-wide">
                <th className="pb-2 pr-3">Rank</th>
                <th className="pb-2 pr-4">Model</th>
                <th className="pb-2 pr-4">Accuracy</th>
                <th className="pb-2 pr-4">F1</th>
                <th className="pb-2 pr-4">AUC</th>
                <th className="pb-2 pr-4">R²</th>
                <th className="pb-2 pr-4">Time</th>
                <th className="pb-2">Actions</th>
              </tr>
            </thead>
            <tbody>
              {ranked.map((run, i) => (
                <tr
                  key={run.run_id}
                  className={`border-b border-slate-50 hover:bg-slate-50 ${i === 0 ? "bg-blue-50/50" : ""}`}
                >
                  <td className="py-3 pr-3 text-lg">{MEDALS[i] ?? `#${i + 1}`}</td>
                  <td className="py-3 pr-4 font-semibold text-slate-800">
                    {run.model_type.replace(/_/g, " ")}
                  </td>
                  <td className="py-3 pr-4 text-slate-700">{fmt(run.metrics.accuracy)}</td>
                  <td className="py-3 pr-4 text-slate-700">{fmt(run.metrics.f1)}</td>
                  <td className="py-3 pr-4 text-slate-700">{fmt(run.metrics.roc_auc)}</td>
                  <td className="py-3 pr-4 text-slate-700">
                    {run.metrics.r2 != null ? run.metrics.r2.toFixed(3) : "—"}
                  </td>
                  <td className="py-3 pr-4 text-slate-500 text-xs">
                    {run.metrics.training_time_s != null
                      ? `${Number(run.metrics.training_time_s).toFixed(1)}s`
                      : "—"}
                  </td>
                  <td className="py-3">
                    <button
                      onClick={() =>
                        navigate(`/projects/${projectId}/analysis/${run.run_id}`)
                      }
                      className="text-xs text-blue-600 hover:text-blue-800 font-medium"
                    >
                      Analyse →
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Charts row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title="Accuracy / F1 / AUC Comparison">
          {ranked.length > 0 ? (
            <AccuracyComparison runs={ranked} />
          ) : (
            <p className="text-slate-400 text-sm text-center py-10">No data.</p>
          )}
        </Card>

        <Card title="Accuracy vs Training Time">
          <ResponsiveContainer width="100%" height={280}>
            <ScatterChart margin={{ top: 4, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis
                dataKey="x"
                name="Time (s)"
                unit="s"
                tick={{ fontSize: 11 }}
                label={{ value: "Training time (s)", position: "insideBottom", offset: -4, fontSize: 11 }}
              />
              <YAxis
                dataKey="y"
                name="Accuracy"
                unit="%"
                domain={["auto", "auto"]}
                tick={{ fontSize: 11 }}
              />
              <ZAxis dataKey="z" range={[60, 120]} />
              <Tooltip
                cursor={{ strokeDasharray: "3 3" }}
                content={({ payload }) => {
                  if (!payload?.length) return null;
                  const d = payload[0].payload;
                  return (
                    <div className="bg-white border border-slate-200 rounded-lg p-3 text-xs shadow-md">
                      <p className="font-semibold text-slate-800">{d.name}</p>
                      <p className="text-slate-500">Accuracy: {d.y}%</p>
                      <p className="text-slate-500">Time: {d.x}s</p>
                    </div>
                  );
                }}
              />
              <Scatter data={scatterData} fill="#3b82f6" />
            </ScatterChart>
          </ResponsiveContainer>
        </Card>
      </div>
    </div>
  );
}
