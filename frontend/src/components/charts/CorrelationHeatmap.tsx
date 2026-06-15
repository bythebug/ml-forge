interface Props {
  matrix: Record<string, Record<string, number | null>>;
  columns: string[];
}

function cellColor(v: number | null): string {
  if (v === null) return "#f8fafc";
  const abs = Math.abs(v);
  return v >= 0 ? `rgba(59,130,246,${(abs * 0.85).toFixed(2)})` : `rgba(239,68,68,${(abs * 0.85).toFixed(2)})`;
}

export default function CorrelationHeatmap({ matrix, columns }: Props) {
  const cols = columns.slice(0, 12);

  return (
    <div className="overflow-auto">
      <table className="text-xs border-collapse" style={{ minWidth: cols.length * 52 }}>
        <thead>
          <tr>
            <th className="w-28" />
            {cols.map((c) => (
              <th
                key={c}
                className="text-slate-500 font-medium p-1"
                style={{ writingMode: "vertical-rl", maxHeight: 80 }}
              >
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {cols.map((row) => (
            <tr key={row}>
              <td className="text-slate-600 font-medium pr-2 text-right whitespace-nowrap max-w-28 overflow-hidden text-ellipsis">
                {row}
              </td>
              {cols.map((col) => {
                const v = matrix[row]?.[col] ?? null;
                return (
                  <td
                    key={col}
                    title={`${row} × ${col}: ${v?.toFixed(2) ?? "n/a"}`}
                    className="border border-white"
                    style={{
                      width: 44,
                      height: 36,
                      background: cellColor(v),
                      textAlign: "center",
                      color: Math.abs(v ?? 0) > 0.5 ? "#fff" : "#334155",
                      fontWeight: row === col ? 700 : 400,
                    }}
                  >
                    {v !== null ? v.toFixed(2) : ""}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
