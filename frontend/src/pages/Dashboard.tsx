import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { Plus, Database, TrendingUp, Clock, X } from "lucide-react";
import { api } from "../api/endpoints";
import Spinner from "../components/ui/Spinner";
import type { ProjectCreate } from "../types/api";

function CreateProjectModal({ onClose }: { onClose: () => void }) {
  const [name, setName] = useState("");
  const [desc, setDesc] = useState("");
  const qc = useQueryClient();
  const mut = useMutation({
    mutationFn: (body: ProjectCreate) => api.projects.create(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["projects"] });
      onClose();
    },
  });

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
      <div className="bg-white rounded-2xl shadow-xl w-full max-w-md p-6">
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-lg font-semibold text-slate-800">New Project</h2>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-600">
            <X className="w-5 h-5" />
          </button>
        </div>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Name</label>
            <input
              className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Churn Prediction"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">
              Description <span className="text-slate-400">(optional)</span>
            </label>
            <textarea
              className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              rows={3}
              value={desc}
              onChange={(e) => setDesc(e.target.value)}
              placeholder="What are you predicting?"
            />
          </div>
          {mut.error && (
            <p className="text-red-500 text-sm">{(mut.error as Error).message}</p>
          )}
          <div className="flex justify-end gap-3 pt-2">
            <button
              onClick={onClose}
              className="px-4 py-2 text-sm text-slate-600 hover:text-slate-800"
            >
              Cancel
            </button>
            <button
              disabled={!name.trim() || mut.isPending}
              onClick={() => mut.mutate({ name: name.trim(), description: desc || undefined })}
              className="px-4 py-2 bg-blue-600 text-white text-sm rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {mut.isPending && <Spinner size={4} />}
              Create Project
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function Dashboard() {
  const [showCreate, setShowCreate] = useState(false);
  const navigate = useNavigate();
  const { data: projects, isLoading, error } = useQuery({
    queryKey: ["projects"],
    queryFn: api.projects.list,
  });

  return (
    <div className="p-8 max-w-5xl">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-slate-800">Projects</h1>
          <p className="text-slate-500 text-sm mt-1">
            ML experiments — upload data, train models, compare results
          </p>
        </div>
        <button
          onClick={() => setShowCreate(true)}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white text-sm rounded-lg font-medium hover:bg-blue-700"
        >
          <Plus className="w-4 h-4" /> New Project
        </button>
      </div>

      {isLoading && (
        <div className="flex justify-center py-20">
          <Spinner size={8} />
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-6 text-red-700 text-sm">
          <strong>Could not connect to the API.</strong> Make sure the FastAPI backend is running on
          port 8000.
          <br />
          <code className="text-xs mt-1 block">{(error as Error).message}</code>
        </div>
      )}

      {projects && projects.length === 0 && (
        <div className="flex flex-col items-center justify-center py-24 text-slate-400">
          <Database className="w-12 h-12 mb-4 opacity-40" />
          <p className="text-lg font-medium text-slate-500">No projects yet</p>
          <p className="text-sm mt-1">Create your first project to get started</p>
          <button
            onClick={() => setShowCreate(true)}
            className="mt-6 flex items-center gap-2 px-4 py-2 bg-blue-600 text-white text-sm rounded-lg font-medium hover:bg-blue-700"
          >
            <Plus className="w-4 h-4" /> New Project
          </button>
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {projects?.map((project) => (
          <button
            key={project.id}
            onClick={() => navigate(`/projects/${project.id}/data`)}
            className="text-left bg-white rounded-xl border border-slate-200 shadow-sm p-5 hover:shadow-md hover:border-blue-300 transition-all"
          >
            <div className="flex items-start justify-between mb-3">
              <h3 className="font-semibold text-slate-800 leading-tight">{project.name}</h3>
              <span className="text-xs text-slate-400 shrink-0 ml-2">#{project.id}</span>
            </div>

            {project.description && (
              <p className="text-xs text-slate-500 mb-4 line-clamp-2">{project.description}</p>
            )}

            <div className="space-y-2">
              <div className="flex items-center gap-2 text-xs text-slate-500">
                <TrendingUp className="w-3.5 h-3.5 text-slate-400" />
                {project.run_count} run{project.run_count !== 1 ? "s" : ""}
                {project.best_accuracy != null && (
                  <span className="ml-auto font-semibold text-green-600">
                    {(project.best_accuracy * 100).toFixed(1)}% best
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2 text-xs text-slate-400">
                <Clock className="w-3.5 h-3.5" />
                {new Date(project.created_at).toLocaleDateString()}
              </div>
            </div>

            {project.recent_model_types.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-3">
                {project.recent_model_types.map((t) => (
                  <span
                    key={t}
                    className="px-1.5 py-0.5 bg-slate-100 text-slate-500 text-xs rounded"
                  >
                    {t}
                  </span>
                ))}
              </div>
            )}
          </button>
        ))}
      </div>

      {showCreate && <CreateProjectModal onClose={() => setShowCreate(false)} />}
    </div>
  );
}
