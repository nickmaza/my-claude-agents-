import { useCallback, useEffect, useState } from "react";
import { api, Job, Options } from "./api";
import JobDetail from "./components/JobDetail";
import JobList from "./components/JobList";
import NewJob from "./components/NewJob";

function useHashId(): [string | null, (id: string | null) => void] {
  const read = () => window.location.hash.replace(/^#\/?/, "") || null;
  const [id, setId] = useState<string | null>(read);
  useEffect(() => {
    const onChange = () => setId(read());
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return [id, (next) => (window.location.hash = next ? `/${next}` : "")];
}

export default function App() {
  const [options, setOptions] = useState<Options | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [selected, select] = useHashId();

  const refresh = useCallback(() => {
    api.jobs().then(setJobs, (e: Error) => setError(e.message));
  }, []);

  useEffect(() => {
    api.options().then(setOptions, (e: Error) => setError(`Cannot reach the API server: ${e.message}`));
    refresh();
  }, [refresh]);

  const active = jobs.some((j) => j.status === "queued" || j.status === "running");
  useEffect(() => {
    if (!active) return;
    const timer = window.setInterval(refresh, 1500);
    return () => window.clearInterval(timer);
  }, [active, refresh]);

  return (
    <div className="app">
      <header className="topbar">
        <a className="brand" href="#/" onClick={() => select(null)}>
          <span className="logo" aria-hidden>
            ⟲
          </span>
          footage2robot
        </a>
        <span className="tagline">Human video → robot training data</span>
        {options && <span className="version">v{options.version}</span>}
      </header>

      {error && (
        <div className="banner error" role="alert">
          {error}
          <button className="link" onClick={() => setError(null)}>
            dismiss
          </button>
        </div>
      )}

      <main className="layout">
        <aside className="sidebar">
          <button className={`new-btn ${selected ? "" : "current"}`} onClick={() => select(null)}>
            + New conversion
          </button>
          <JobList jobs={jobs} selected={selected} onSelect={select} />
        </aside>
        <section className="content">
          {selected ? (
            <JobDetail
              key={selected}
              id={selected}
              options={options}
              onDeleted={() => {
                select(null);
                refresh();
              }}
            />
          ) : options ? (
            <NewJob
              options={options}
              onCreated={(job) => {
                refresh();
                select(job.id);
              }}
            />
          ) : (
            !error && <p className="muted">Loading…</p>
          )}
        </section>
      </main>
    </div>
  );
}
