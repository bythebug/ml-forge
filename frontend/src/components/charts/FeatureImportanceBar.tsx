import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";

interface Props {
  items: { feature: string; importance: number }[];
  topN?: number;
}

const COLORS = [
  "#3b82f6", "#6366f1", "#8b5cf6", "#a855f7", "#ec4899",
  "#f43f5e", "#f97316", "#eab308", "#22c55e", "#14b8a6",
];

export default function FeatureImportanceBar({ items, topN = 15 }: Props) {
  const sorted = [...items]
    .sort((a, b) => b.importance - a.importance)
    .slice(0, topN);

  return (
    <ResponsiveContainer width="100%" height={Math.max(200, sorted.length * 30)}>
      <BarChart data={sorted} layout="vertical" margin={{ left: 8, right: 32 }}>
        <XAxis type="number" tick={{ fontSize: 11 }} />
        <YAxis
          type="category"
          dataKey="feature"
          width={140}
          tick={{ fontSize: 11 }}
        />
        <Tooltip formatter={(v) => Number(v).toFixed(4)} />
        <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
          {sorted.map((_, i) => (
            <Cell key={i} fill={COLORS[i % COLORS.length]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
