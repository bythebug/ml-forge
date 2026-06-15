import { useLocation, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "../../api/endpoints";

const segmentLabel: Record<string, string> = {
  data: "Data Explorer",
  features: "Features",
  training: "Training",
  results: "Results",
  analysis: "Analysis",
};

export default function TopBar() {
  const { id } = useParams<{ id: string }>();
  const location = useLocation();
  const segments = location.pathname.split("/").filter(Boolean);

  const { data: projects } = useQuery({
    queryKey: ["projects"],
    queryFn: api.projects.list,
    enabled: !!id,
  });

  const project = projects?.find((p) => p.id === Number(id));
  const page = segments.find((s) => segmentLabel[s]);

  return (
    <header className="h-14 bg-white border-b border-slate-200 flex items-center px-6 gap-2 text-sm text-slate-500">
      <span className="font-medium text-slate-800">
        {project ? project.name : "ml-forge"}
      </span>
      {page && (
        <>
          <span className="text-slate-300">/</span>
          <span>{segmentLabel[page]}</span>
        </>
      )}
    </header>
  );
}
