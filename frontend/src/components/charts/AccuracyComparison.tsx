import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import type { ComparisonRun } from "../../types/api";

interface Props {
  runs: ComparisonRun[];
}

export default function AccuracyComparison({ runs }: Props) {
  const data = runs.map((r) => ({
    name: r.model_type.replace(/_/g, " "),
    accuracy: r.metrics.accuracy != null ? +(r.metrics.accuracy * 100).toFixed(1) : undefined,
    f1: r.metrics.f1 != null ? +(r.metrics.f1 * 100).toFixed(1) : undefined,
    auc: r.metrics.roc_auc != null ? +(r.metrics.roc_auc * 100).toFixed(1) : undefined,
  }));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data} margin={{ top: 4, right: 16 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
        <XAxis dataKey="name" tick={{ fontSize: 11 }} />
        <YAxis
          domain={[50, 100]}
          tickFormatter={(v) => `${v}%`}
          tick={{ fontSize: 11 }}
        />
        <Tooltip formatter={(v) => `${v}%`} />
        <Legend />
        <Bar dataKey="accuracy" fill="#3b82f6" name="Accuracy" radius={[3, 3, 0, 0]} />
        <Bar dataKey="f1" fill="#8b5cf6" name="F1" radius={[3, 3, 0, 0]} />
        <Bar dataKey="auc" fill="#10b981" name="AUC" radius={[3, 3, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
