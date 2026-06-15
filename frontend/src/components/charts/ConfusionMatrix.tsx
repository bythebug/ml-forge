import type { ConfusionMatrixAnalysis } from "../../types/api";

interface Props {
  data: ConfusionMatrixAnalysis;
}

export default function ConfusionMatrix({ data }: Props) {
  const matrix = data.confusion_matrix;
  const labels = Object.keys(data.error_rate_per_class);
  const rowMax = matrix.map((row) => Math.max(...row));

  return (
    <div className="overflow-auto">
      <table className="text-sm border-collapse">
        <thead>
          <tr>
            <th className="text-slate-400 font-normal text-xs p-1" colSpan={2} rowSpan={2} />
            <th
              className="text-slate-500 text-xs font-semibold pb-1 text-center"
              colSpan={labels.length}
            >
              Predicted
            </th>
          </tr>
          <tr>
            {labels.map((l) => (
              <th key={String(l)} className="text-slate-500 text-xs font-medium p-1 text-center min-w-12">
                {String(l)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, ri) => (
            <tr key={ri}>
              {ri === 0 && (
                <td
                  className="text-slate-500 text-xs font-semibold text-center"
                  rowSpan={matrix.length}
                  style={{ writingMode: "vertical-rl" }}
                >
                  Actual
                </td>
              )}
              <td className="text-slate-500 text-xs font-medium p-1 text-right pr-2">
                {String(labels[ri])}
              </td>
              {row.map((cell, ci) => {
                const intensity = rowMax[ri] > 0 ? cell / rowMax[ri] : 0;
                const isDiag = ri === ci;
                const bg = isDiag
                  ? `rgba(34,197,94,${(0.15 + intensity * 0.7).toFixed(2)})`
                  : cell > 0
                  ? `rgba(239,68,68,${(0.1 + intensity * 0.6).toFixed(2)})`
                  : "#f8fafc";
                return (
                  <td
                    key={ci}
                    className="border border-white text-center font-semibold"
                    style={{ background: bg, width: 52, height: 44, color: intensity > 0.5 ? "#fff" : "#1e293b" }}
                  >
                    {cell}
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
