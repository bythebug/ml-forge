import { useState, useRef } from "react";
import { useParams } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { UploadCloud, FileText, AlertCircle } from "lucide-react";
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { api } from "../api/endpoints";
import Card from "../components/ui/Card";
import Spinner from "../components/ui/Spinner";
import MissingValuesBar from "../components/charts/MissingValuesBar";

function ColumnTypeBreakdown({ dtypes }: { dtypes: Record<string, string> }) {
  const counts: Record<string, number> = {};
  Object.values(dtypes).forEach((dtype) => {
    const kind =
      dtype.includes("int") || dtype.includes("float")
        ? "Numeric"
        : dtype.includes("datetime")
        ? "Datetime"
        : "Categorical";
    counts[kind] = (counts[kind] ?? 0) + 1;
  });

  const data = Object.entries(counts).map(([name, value]) => ({ name, value }));
  const COLORS = ["#3b82f6", "#10b981", "#f59e0b"];

  return (
    <ResponsiveContainer width="100%" height={180}>
      <PieChart>
        <Pie data={data} cx="50%" cy="50%" outerRadius={65} dataKey="value" label={({ name, value }) => `${name}: ${value}`} labelLine={false}>
          {data.map((_, i) => (
            <Cell key={i} fill={COLORS[i % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip />
        <Legend />
      </PieChart>
    </ResponsiveContainer>
  );
}

export default function DataExplorer() {
  const { id } = useParams<{ id: string }>();
  const projectId = Number(id);
  const qc = useQueryClient();
  const fileRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const { data: profile, isLoading: profileLoading } = useQuery({
    queryKey: ["profile", projectId],
    queryFn: () => api.data.profile(projectId),
    retry: false,
  });

  const { data: missing } = useQuery({
    queryKey: ["missing", projectId],
    queryFn: () => api.data.missing(projectId),
    enabled: !!profile,
  });

  const upload = useMutation({
    mutationFn: (file: File) => api.data.upload(projectId, file),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["profile", projectId] });
      qc.invalidateQueries({ queryKey: ["missing", projectId] });
    },
  });

  const handleFile = (file: File) => upload.mutate(file);

  return (
    <div className="p-8 max-w-6xl space-y-6">
      <h1 className="text-2xl font-bold text-slate-800">Data Explorer</h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Upload */}
        <Card title="Dataset" className="lg:col-span-1">
          <div
            className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-colors ${
              dragOver ? "border-blue-400 bg-blue-50" : "border-slate-200 hover:border-slate-300"
            }`}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              const file = e.dataTransfer.files[0];
              if (file) handleFile(file);
            }}
            onClick={() => fileRef.current?.click()}
          >
            <input
              ref={fileRef}
              type="file"
              accept=".csv,.parquet,.pq"
              className="hidden"
              onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
            />
            {upload.isPending ? (
              <div className="flex flex-col items-center gap-2">
                <Spinner size={8} />
                <p className="text-sm text-slate-500">Uploading...</p>
              </div>
            ) : (
              <>
                <UploadCloud className="w-10 h-10 text-slate-300 mx-auto mb-2" />
                <p className="text-sm font-medium text-slate-600">Drop CSV or Parquet</p>
                <p className="text-xs text-slate-400 mt-1">or click to browse</p>
              </>
            )}
          </div>

          {upload.isSuccess && (
            <div className="mt-3 p-3 bg-green-50 rounded-lg text-xs text-green-700 flex items-center gap-2">
              <FileText className="w-4 h-4" />
              {upload.data.rows.toLocaleString()} rows × {upload.data.columns} columns loaded
            </div>
          )}

          {upload.isError && (
            <div className="mt-3 p-3 bg-red-50 rounded-lg text-xs text-red-700 flex items-center gap-2">
              <AlertCircle className="w-4 h-4" />
              {(upload.error as Error).message}
            </div>
          )}
        </Card>

        {/* Profile stats */}
        <Card title="Dataset Overview" className="lg:col-span-2">
          {profileLoading && (
            <div className="flex justify-center py-10">
              <Spinner size={6} />
            </div>
          )}
          {!profile && !profileLoading && (
            <p className="text-slate-400 text-sm text-center py-10">
              Upload a dataset to see statistics.
            </p>
          )}
          {profile && (
            <div className="grid grid-cols-3 gap-4 mb-6">
              {[
                { label: "Rows", value: profile.shape.rows.toLocaleString() },
                { label: "Columns", value: profile.shape.columns },
                { label: "Duplicates", value: profile.duplicates },
              ].map(({ label, value }) => (
                <div key={label} className="text-center p-4 bg-slate-50 rounded-xl">
                  <div className="text-2xl font-bold text-slate-800">{value}</div>
                  <div className="text-xs text-slate-500 mt-1">{label}</div>
                </div>
              ))}
            </div>
          )}
          {profile && <ColumnTypeBreakdown dtypes={profile.dtypes} />}
        </Card>
      </div>

      {/* Missing values */}
      {missing && (
        <Card title="Missing Values by Column">
          <MissingValuesBar columns={missing.columns} />
        </Card>
      )}

      {/* Column details */}
      {profile && (
        <Card title="Column Details">
          <div className="overflow-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-100 text-left text-xs text-slate-500 font-semibold uppercase tracking-wide">
                  <th className="pb-2 pr-4">Column</th>
                  <th className="pb-2 pr-4">Type</th>
                  <th className="pb-2 pr-4">Missing</th>
                  <th className="pb-2 pr-4">Missing %</th>
                  <th className="pb-2 pr-4">Mean</th>
                  <th className="pb-2">Std</th>
                </tr>
              </thead>
              <tbody>
                {profile.columns.map((col) => {
                  const ns = profile.numeric_stats[col];
                  return (
                    <tr key={col} className="border-b border-slate-50 hover:bg-slate-50">
                      <td className="py-2 pr-4 font-medium text-slate-700">{col}</td>
                      <td className="py-2 pr-4">
                        <span className="px-2 py-0.5 bg-blue-50 text-blue-700 text-xs rounded">
                          {profile.dtypes[col]}
                        </span>
                      </td>
                      <td className="py-2 pr-4 text-slate-600">{profile.missing[col]}</td>
                      <td className="py-2 pr-4 text-slate-600">{profile.missing_pct[col]}%</td>
                      <td className="py-2 pr-4 text-slate-600">
                        {ns?.mean != null ? Number(ns.mean).toFixed(3) : "—"}
                      </td>
                      <td className="py-2 text-slate-600">
                        {ns?.std != null ? Number(ns.std).toFixed(3) : "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}
