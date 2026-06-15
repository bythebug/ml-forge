import { BrowserRouter, Routes, Route, Outlet } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import Sidebar from "./components/layout/Sidebar";
import TopBar from "./components/layout/TopBar";
import Dashboard from "./pages/Dashboard";
import DataExplorer from "./pages/DataExplorer";
import FeatureEngineering from "./pages/FeatureEngineering";
import Training from "./pages/Training";
import Results from "./pages/Results";
import Analysis from "./pages/Analysis";

const qc = new QueryClient({
  defaultOptions: { queries: { staleTime: 30_000, retry: 1 } },
});

function Layout() {
  return (
    <div className="flex min-h-screen bg-slate-50">
      <Sidebar />
      <div className="flex-1 flex flex-col min-w-0">
        <TopBar />
        <main className="flex-1 overflow-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<Dashboard />} />
            <Route path="/projects/:id/data" element={<DataExplorer />} />
            <Route path="/projects/:id/features" element={<FeatureEngineering />} />
            <Route path="/projects/:id/training" element={<Training />} />
            <Route path="/projects/:id/results" element={<Results />} />
            <Route path="/projects/:id/analysis/:runId" element={<Analysis />} />
            <Route
              path="/projects/:id/analysis"
              element={
                <div className="p-8 text-slate-500">
                  Select a run from <strong>Results</strong> to analyse.
                </div>
              }
            />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
