import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import type { MissingColumn } from "../../types/api";

interface Props {
  columns: MissingColumn[];
}

export default function MissingValuesBar({ columns }: Props) {
  const sorted = [...columns]
    .filter((c) => c.missing_pct > 0)
    .sort((a, b) => b.missing_pct - a.missing_pct)
    .slice(0, 20);

  if (sorted.length === 0) {
    return (
      <div className="flex items-center justify-center h-40 text-slate-400 text-sm">
        No missing values — dataset is complete.
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={Math.max(200, sorted.length * 28)}>
      <BarChart data={sorted} layout="vertical" margin={{ left: 16, right: 32 }}>
        <XAxis
          type="number"
          domain={[0, 100]}
          tickFormatter={(v) => `${v}%`}
          tick={{ fontSize: 11 }}
        />
        <YAxis
          type="category"
          dataKey="column"
          width={120}
          tick={{ fontSize: 11 }}
        />
        <Tooltip formatter={(v) => `${Number(v).toFixed(1)}%`} />
        <Bar dataKey="missing_pct" radius={[0, 4, 4, 0]}>
          {sorted.map((entry) => (
            <Cell
              key={entry.column}
              fill={entry.missing_pct > 30 ? "#ef4444" : entry.missing_pct > 10 ? "#f97316" : "#3b82f6"}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
