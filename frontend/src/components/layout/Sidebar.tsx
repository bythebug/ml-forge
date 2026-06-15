import { NavLink, useParams } from "react-router-dom";
import {
  BarChart2,
  Database,
  FlaskConical,
  Home,
  LineChart,
  Microscope,
  Zap,
} from "lucide-react";

const logo = (
  <div className="flex items-center gap-2 px-6 py-5 border-b border-slate-200">
    <Zap className="w-6 h-6 text-blue-600" />
    <span className="font-bold text-slate-800 text-lg tracking-tight">ml-forge</span>
  </div>
);

const navCls = ({ isActive }: { isActive: boolean }) =>
  `flex items-center gap-3 px-4 py-2.5 rounded-lg text-sm font-medium transition-colors ${
    isActive
      ? "bg-blue-50 text-blue-700"
      : "text-slate-600 hover:bg-slate-100 hover:text-slate-800"
  }`;

export default function Sidebar() {
  const { id } = useParams<{ id: string }>();

  return (
    <aside className="w-56 shrink-0 bg-white border-r border-slate-200 flex flex-col min-h-screen">
      {logo}
      <nav className="flex-1 p-3 space-y-0.5">
        <NavLink to="/" end className={navCls}>
          <Home className="w-4 h-4" /> Dashboard
        </NavLink>
        {id && (
          <>
            <div className="pt-4 pb-1 px-4 text-xs font-semibold text-slate-400 uppercase tracking-widest">
              Project
            </div>
            <NavLink to={`/projects/${id}/data`} className={navCls}>
              <Database className="w-4 h-4" /> Data Explorer
            </NavLink>
            <NavLink to={`/projects/${id}/features`} className={navCls}>
              <FlaskConical className="w-4 h-4" /> Features
            </NavLink>
            <NavLink to={`/projects/${id}/training`} className={navCls}>
              <BarChart2 className="w-4 h-4" /> Training
            </NavLink>
            <NavLink to={`/projects/${id}/results`} className={navCls}>
              <LineChart className="w-4 h-4" /> Results
            </NavLink>
            <NavLink
              to={`/projects/${id}/analysis`}
              className={({ isActive }) =>
                navCls({ isActive }) + " opacity-60 pointer-events-none"
              }
            >
              <Microscope className="w-4 h-4" /> Analysis
            </NavLink>
          </>
        )}
      </nav>
      <div className="p-4 border-t border-slate-200">
        <p className="text-xs text-slate-400">ml-forge v0.1.0</p>
      </div>
    </aside>
  );
}
