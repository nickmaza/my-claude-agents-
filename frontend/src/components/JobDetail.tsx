import { useCallback, useEffect, useState } from "react";
import { api, Job, Options, Overlay, Preview } from "../api";
import { StatusBadge, timeAgo } from "./JobList";
import Player from "./Player";
import SignalChart from "./SignalChart";

interface Props {
  id: string;
  options: Options | null;
  onDeleted: () => void;
}

const pct = (v: number) => `${Math.round(v * 100)}%`;

export default function JobDetail({ id, options, onDeleted }: Props) {
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [clip, setClip] = useState(0);
  const [embodiment, setEmbodiment] = useState<string | null>(null);
  const [overlay, setOverlay] = useState<Overlay | null>(null);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [frame, setFrame] = useState(0);

  const active = !job || job.status === "queued" || job.status === "running";
  useEffect(() => {
    let stop = false;
    let timer = 0;
    const poll = () =>
      api.job(id).then(
        (j) => {
          if (stop) return;
          setJob(j);
          if (j.status === "queued" || j.status === "running") timer = window.setTimeout(poll, 1000);
        },
        (e: Error) => !stop && setError(e.message),
      );
    void poll();
    return () => {
      stop = true;
      window.clearTimeout(timer);
    };
  }, [id]);

  const datasets = job?.summary?.datasets ?? [];
  const usable = datasets.filter((d) => d.num_episodes);
  const current = usable.find((d) => d.embodiment === embodiment) ?? usable[0];
  const hasTracking = Boolean(job?.summary);

  useEffect(() => {
    if (!hasTracking) return;
    setOverlay(null);
    setFrame(0);
    api.overlay(id, clip).then(setOverlay, () => setOverlay(null));
  }, [id, clip, hasTracking]);

  const currentKey = current?.embodiment;
  useEffect(() => {
    if (!currentKey) return;
    setPreview(null);
    api.preview(id, currentKey).then(setPreview, () => setPreview(null));
  }, [id, currentKey]);

  const onFrame = useCallback((f: number) => setFrame(f), []);

  if (error) return <div className="banner error">{error}</div>;
  if (!job) return <p className="muted">Loading…</p>;

  const summary = job.summary;
  const episode = preview?.episodes.find((e) => e.clip === clip);
  const isSynthetic = job.clips.length === 1 && job.clips[0] === "synthetic demo";
  const formats = job.formats;

  const remove = async () => {
    if (!window.confirm("Delete this conversion and all of its exported datasets?")) return;
    try {
      await api.remove(id);
      onDeleted();
    } catch (e) {
      setError((e as Error).message);
    }
  };

  return (
    <div className="detail">
      <div className="detail-head">
        <div>
          <h1>{job.task}</h1>
          <p className="muted small">
            <StatusBadge status={job.status} /> {job.clips.length} clip{job.clips.length === 1 ? "" : "s"} · started{" "}
            {timeAgo(job.created_at)}
            {summary && ` · processed in ${summary.elapsed_s.toFixed(1)}s`}
          </p>
        </div>
        {!active && (
          <button className="secondary danger" onClick={() => void remove()}>
            Delete
          </button>
        )}
      </div>

      {active && (
        <div className="card">
          <div className="bar big">
            <span style={{ width: pct(job.progress) }} />
          </div>
          <p className="muted">
            {pct(job.progress)} — {job.message}
          </p>
        </div>
      )}

      {job.status === "failed" && (
        <div className="banner error" role="alert">
          <strong>Conversion failed.</strong> {job.error}
          {!summary && " Check that the files are valid videos."}
        </div>
      )}

      {summary && summary.warnings.length > 0 && job.status === "done" && (
        <div className="banner warn">
          <strong>Warnings</strong>
          <ul>
            {summary.warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </div>
      )}

      {summary && (
        <>
          <section className="card">
            <h2>Tracking</h2>
            {summary.clips.length > 1 && (
              <div className="tabs">
                {summary.clips.map((c, i) => (
                  <button key={i} className={i === clip ? "on" : ""} onClick={() => setClip(i)} title={c.name}>
                    {i + 1}. {c.name}
                  </button>
                ))}
              </div>
            )}
            <div className="tracking">
              {overlay ? (
                <Player
                  key={clip}
                  overlay={overlay}
                  videoUrl={isSynthetic ? null : api.videoUrl(id, clip)}
                  frame={Math.min(frame, overlay.body.length - 1)}
                  onFrame={onFrame}
                />
              ) : (
                <p className="muted">Loading tracking…</p>
              )}
              <dl className="stats">
                <dt>Frames</dt>
                <dd>
                  {summary.clips[clip].frames} @ {summary.clips[clip].fps.toFixed(1)} fps
                </dd>
                <dt>Duration</dt>
                <dd>{summary.clips[clip].duration_s.toFixed(2)} s</dd>
                {(["body", "left_hand", "right_hand"] as const).map((k) => (
                  <div key={k} className="cov">
                    <dt>{k.replace("_", " ")} tracked</dt>
                    <dd>
                      <span className="bar">
                        <span style={{ width: pct(summary.clips[clip].coverage[k]) }} />
                      </span>
                      {pct(summary.clips[clip].coverage[k])}
                    </dd>
                  </div>
                ))}
              </dl>
            </div>
          </section>

          {usable.length > 0 && current && (
            <section className="card">
              <h2>Robot trajectory</h2>
              <div className="tabs caps">
                {usable.map((d) => (
                  <button
                    key={d.embodiment}
                    className={d.embodiment === current.embodiment ? "on" : ""}
                    onClick={() => setEmbodiment(d.embodiment)}
                  >
                    {d.embodiment.replace("_", " ")}
                  </button>
                ))}
              </div>
              <p className="muted small">
                {current.description} · {current.state_dim}-D state · {current.num_episodes} episode
                {current.num_episodes === 1 ? "" : "s"} · {current.total_frames} frames
              </p>
              {preview && episode && overlay ? (
                <SignalChart
                  key={`${current.embodiment}-${clip}`}
                  names={preview.state_names}
                  state={episode.state}
                  offset={episode.offset}
                  clipFrames={overlay.body.length}
                  frame={frame}
                  fps={overlay.fps}
                  onSeek={setFrame}
                />
              ) : preview && !episode ? (
                <p className="muted">This clip did not have enough tracking for this embodiment.</p>
              ) : (
                <p className="muted">Loading…</p>
              )}
              {current.episodes
                ?.filter((e) => e.clip === clip && e.defaulted_dims.length > 0)
                .map((e, i) => (
                  <p key={i} className="muted small">
                    Not visible in this clip, set to neutral (0): {e.defaulted_dims.join(", ")}
                  </p>
                ))}
            </section>
          )}

          {job.status === "done" && (
            <section className="card">
              <div className="card-head">
                <h2>Datasets</h2>
                <a className="primary btn" href={api.downloadUrl(id)}>
                  Download everything (.zip)
                </a>
              </div>
              <div className="table-wrap">
                <table className="downloads">
                  <thead>
                    <tr>
                      <th>Embodiment</th>
                      {formats.map((f) => (
                        <th key={f} title={options?.formats[f]}>
                          {f}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {datasets.map((d) => (
                      <tr key={d.embodiment}>
                        <th>
                          {d.embodiment.replace("_", " ")}
                          <small className="muted">
                            {d.num_episodes ? `${d.state_dim}-D · ${d.num_episodes} ep` : "no usable tracking"}
                          </small>
                        </th>
                        {formats.map((f) => (
                          <td key={f}>
                            {d.exports[f] ? (
                              <a href={api.downloadUrl(id, d.embodiment, f)}>.zip</a>
                            ) : (
                              <span className="fail" title={d.errors[f] ?? d.errors.all ?? "Not exported"}>
                                failed
                              </span>
                            )}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}
        </>
      )}
    </div>
  );
}
