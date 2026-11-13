import { useCallback, useEffect, useState } from "react";
import { api, ApiError, Job, Options, setTokenProvider } from "./api";
import type { AuthApi, Session } from "./auth";
import JobDetail from "./components/JobDetail";
import JobList from "./components/JobList";
import NewJob from "./components/NewJob";
import SignIn from "./components/SignIn";

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

type AuthState =
  | { phase: "loading" }
  | { phase: "open" } // server does not require sign-in
  | { phase: "signed-out"; auth: AuthApi }
  | { phase: "signed-in"; auth: AuthApi; session: Session };

export default function App() {
  const [authState, setAuthState] = useState<AuthState>({ phase: "loading" });
  const [options, setOptions] = useState<Options | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [selected, select] = useHashId();

  useEffect(() => {
    let unsubscribe = () => {};
    let cancelled = false;
    (async () => {
      try {
        const cfg = await api.config();
        if (!cfg.auth_required) return setAuthState({ phase: "open" });
        if (!cfg.firebase) throw new Error("Sign-in is required, but the server has no Firebase web config set.");
        const { initAuth } = await import("./auth");
        const auth = await initAuth(cfg.firebase);
        if (cancelled) return;
        setTokenProvider(auth.getToken);
        unsubscribe = auth.onChange((session) =>
          setAuthState(session ? { phase: "signed-in", auth, session } : { phase: "signed-out", auth }),
        );
      } catch (e) {
        setError(`Cannot start the app: ${(e as Error).message}`);
      }
    })();
    return () => {
      cancelled = true;
      unsubscribe();
    };
  }, []);

  const ready = authState.phase === "open" || authState.phase === "signed-in";

  const refresh = useCallback(() => {
    api.jobs().then(setJobs, (e: Error) => {
      if (!(e instanceof ApiError && e.status === 401)) setError(e.message);
    });
  }, []);

  useEffect(() => {
    if (!ready) {
      setJobs([]);
      return;
    }
    api.options().then(setOptions, (e: Error) => setError(e.message));
    refresh();
  }, [ready, refresh]);

  const active = jobs.some((j) => j.status === "queued" || j.status === "running");
  useEffect(() => {
    if (!active || !ready) return;
    const timer = window.setInterval(refresh, 1500);
    return () => window.clearInterval(timer);
  }, [active, ready, refresh]);

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
        <span className="spacer" />
        {authState.phase === "signed-in" && (
          <span className="account">
            <span className="muted small email">{authState.session.email}</span>
            <button className="link" onClick={() => void authState.session.signOut()}>
              Sign out
            </button>
          </span>
        )}
      </header>

      {error && (
        <div className="banner error" role="alert">
          {error}
          <button className="link" onClick={() => setError(null)}>
            dismiss
          </button>
        </div>
      )}

      {authState.phase === "loading" && !error && <p className="muted pad">Loading…</p>}
      {authState.phase === "signed-out" && <SignIn auth={authState.auth} />}

      {ready && (
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
                onChanged={refresh}
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
      )}
    </div>
  );
}
