import { useCallback, useEffect, useState } from "react";
import { api, ClipEdit, Job, Options, Overlay, Preview } from "../api";
import EpisodeEditor from "./EpisodeEditor";
import { StatusBadge, timeAgo } from "./JobList";
import Player from "./Player";
import SignalChart from "./SignalChart";

interface Props {
  id: string;
  options: Options | null;
  onChanged: () => void;
  onDeleted: () => void;
}

const pct = (v: number) => `${Math.round(v * 100)}%`;
const label = (key: string) => key.replace(/_/g, " ");

export default function JobDetail({ id, options, onChanged, onDeleted }: Props) {
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [clip, setClip] = useState(0);
  const [embodiment, setEmbodiment] = useState<string | null>(null);
  const [overlay, setOverlay] = useState<Overlay | null>(null);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [frame, setFrame] = useState(0);
  const [edits, setEdits] = useState<Record<number, ClipEdit>>({});
  const [picking, setPicking] = useState(false);
  const [saving, setSaving] = useState(false);
  const [pollKey, setPollKey] = useState(0);

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
  }, [id, pollKey]);

  const datasets = job?.summary?.datasets ?? [];
  const usable = datasets.filter((d) => d.num_episodes);
  const current = usable.find((d) => d.embodiment === embodiment) ?? usable[0];
  // refetch tracking/previews whenever a run finishes
  const version = job && !active ? `${job.finished_at}` : null;
  const hasTracking = Boolean(job?.summary) && !active;

  useEffect(() => {
    if (!hasTracking) return;
    setOverlay(null);
    setFrame(0);
    setPicking(false);
    api.overlay(id, clip).then(setOverlay, () => setOverlay(null));
  }, [id, clip, hasTracking, version]);

  const currentKey = current?.embodiment;
  useEffect(() => {
    if (!currentKey || !hasTracking) return;
    setPreview(null);
    api.preview(id, currentKey).then(setPreview, () => setPreview(null));
  }, [id, currentKey, hasTracking, version]);

  const onFrame = useCallback((f: number) => setFrame(f), []);

  if (error && !job) return <div className="banner error">{error}</div>;
  if (!job) return <p className="muted">Loading…</p>;

  const summary = job.summary;
  const episodes = preview?.episodes.filter((e) => e.clip === clip) ?? [];
  const clipInfo = summary?.clips[clip];
  const formats = job.formats;
  const draft = edits[clip] ?? {};
  const segments = draft.segments ?? overlay?.segments ?? [];
  const prompt = draft.object_prompt !== undefined ? draft.object_prompt : (overlay?.object_prompt ?? null);
  const dirty = Object.keys(edits).length > 0;
  const setDraft = (patch: ClipEdit) => setEdits((e) => ({ ...e, [clip]: { ...e[clip], ...patch } }));

  const remove = async () => {
    if (!window.confirm("Delete this conversion and all of its exported datasets?")) return;
    try {
      await api.remove(id);
      onDeleted();
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const apply = async () => {
    setSaving(true);
    setError(null);
    try {
      const body = Object.fromEntries(Object.entries(edits).map(([k, v]) => [String(k), v]));
      setJob(await api.edit(id, body));
      setEdits({});
      setPollKey((k) => k + 1);
      onChanged();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  };

  const ik = current?.episodes?.find((e) => e.clip === clip);

  return (
    <div className="detail">
      <div className="detail-head">
        <div>
          <h1>{job.task}</h1>
          <p className="muted small">
            <StatusBadge status={job.status} /> {job.clips.length} clip{job.clips.length === 1 ? "" : "s"} · started{" "}
            {timeAgo(job.created_at)}
            {summary && !active && ` · processed in ${summary.elapsed_s.toFixed(1)}s`}
          </p>
        </div>
        {!active && (
          <button className="secondary danger" onClick={() => void remove()}>
            Delete
          </button>
        )}
      </div>

      {error && (
        <div className="banner error" role="alert">
          {error}
        </div>
      )}

      {active && (
        <div className="card">
          <div className="bar big">
            <span style={{ width: pct(job.progress) }} />
          </div>
          <p className="muted">
            {job.status === "queued" ? "Waiting for a free worker…" : `${pct(job.progress)} — ${job.message}`}
            {job.stage === "rebuild" && " (re-exporting with your changes)"}
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

      {summary && !active && clipInfo && (
        <>
          <section className="card">
            <h2>Tracking and episodes</h2>
            {summary.clips.length > 1 && (
              <div className="tabs">
                {summary.clips.map((c, i) => (
                  <button key={i} className={i === clip ? "on" : ""} onClick={() => setClip(i)} title={c.name}>
                    {i + 1}. {c.name}
                    {edits[i] ? " •" : ""}
                  </button>
                ))}
              </div>
            )}
            <div className="tracking">
              {overlay ? (
                <div>
                  <Player
                    key={`${clip}-${version}`}
                    overlay={overlay}
                    videoUrl={clipInfo.has_video ? api.videoUrl(job, clip) : null}
                    frame={Math.min(frame, overlay.body.length - 1)}
                    onFrame={onFrame}
                    prompt={prompt}
                    onPick={
                      picking
                        ? (x, y) => {
                            setDraft({ object_prompt: [Math.min(frame, overlay.body.length - 1), x, y] });
                            setPicking(false);
                          }
                        : undefined
                    }
                  />
                  <EpisodeEditor
                    segments={segments}
                    total={overlay.body.length}
                    frame={Math.min(frame, overlay.body.length - 1)}
                    fps={overlay.fps}
                    onChange={(s) => setDraft({ segments: s })}
                    onSeek={setFrame}
                  />
                </div>
              ) : (
                <p className="muted">Loading tracking…</p>
              )}
              <div>
                <dl className="stats">
                  <dt>Frames</dt>
                  <dd>
                    {clipInfo.frames} @ {clipInfo.fps.toFixed(1)} fps
                  </dd>
                  <dt>Duration</dt>
                  <dd>{clipInfo.duration_s.toFixed(2)} s</dd>
                  {(["body", "left_hand", "right_hand"] as const).map((k) => (
                    <div key={k} className="cov">
                      <dt>{label(k)} tracked</dt>
                      <dd>
                        <span className="bar">
                          <span style={{ width: pct(clipInfo.coverage[k]) }} />
                        </span>
                        {pct(clipInfo.coverage[k])}
                      </dd>
                    </div>
                  ))}
                  <dt>Scale</dt>
                  <dd>
                    {clipInfo.calibration.method
                      ? `×${clipInfo.calibration.scale.toFixed(2)} from ${clipInfo.calibration.method === "height" ? "height" : "arm length"}`
                      : "not calibrated"}
                  </dd>
                  <dt>Workspace</dt>
                  <dd>
                    {clipInfo.calibration.marker
                      ? `marker, ${clipInfo.calibration.marker_distance_m?.toFixed(2)} m from camera`
                      : "relative to the body"}
                  </dd>
                  <dt>Object</dt>
                  <dd>{clipInfo.object_tracked ? "tracked" : prompt ? "point set" : "not tracked"}</dd>
                </dl>
                {clipInfo.has_video && (
                  <div className="object-tools">
                    <button type="button" className={picking ? "primary" : "secondary"} onClick={() => setPicking((p) => !p)}>
                      {picking ? "Click the object in the video…" : prompt ? "Pick a different object point" : "Track an object"}
                    </button>
                    {prompt && (
                      <button type="button" className="link" onClick={() => setDraft({ object_prompt: null })}>
                        stop tracking
                      </button>
                    )}
                    <p className="muted small">
                      Pause on a frame where the object is clearly visible, then click it.
                    </p>
                  </div>
                )}
              </div>
            </div>
            {dirty && (
              <div className="apply">
                <span>You have unsaved changes to episodes or the tracked object.</span>
                <button className="link" onClick={() => setEdits({})}>
                  discard
                </button>
                <button className="primary" disabled={saving} onClick={() => void apply()}>
                  {saving ? "Saving…" : "Apply and re-export"}
                </button>
              </div>
            )}
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
                    {label(d.embodiment)}
                  </button>
                ))}
              </div>
              <p className="muted small">
                {current.description} · {current.state_dim}-D state · {current.num_episodes} episode
                {current.num_episodes === 1 ? "" : "s"} · {current.total_frames} frames
              </p>
              {preview && episodes.length > 0 && overlay ? (
                <SignalChart
                  key={`${current.embodiment}-${clip}`}
                  names={preview.state_names}
                  episodes={episodes}
                  clipFrames={overlay.body.length}
                  frame={frame}
                  fps={overlay.fps}
                  onSeek={setFrame}
                />
              ) : preview && episodes.length === 0 ? (
                <p className="muted">This clip did not have enough tracking for this embodiment.</p>
              ) : (
                <p className="muted">Loading…</p>
              )}
              {ik?.ik_error_mean_m !== undefined && (
                <p className="muted small">
                  Inverse kinematics: mean position error {(ik.ik_error_mean_m * 1000).toFixed(1)} mm, worst{" "}
                  {((ik.ik_error_max_m ?? 0) * 1000).toFixed(1)} mm, {pct(ik.ik_reachable_fraction ?? 0)} of frames
                  reachable. {ik.mapping && `Mapping: ${ik.mapping}.`}
                </p>
              )}
              {ik?.limit_clipped_fraction !== undefined && (
                <p className="muted small">{pct(ik.limit_clipped_fraction)} of joint values were clipped to the robot&apos;s limits.</p>
              )}
              {ik?.notes && <p className="muted small">{ik.notes}</p>}
              {current.episodes
                ?.filter((e) => e.clip === clip && e.defaulted_dims.length > 0)
                .slice(0, 1)
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
                <a className="primary btn" href={api.downloadUrl(job)}>
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
                          {f.replace("_", " ")}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {datasets.map((d) => (
                      <tr key={d.embodiment}>
                        <th>
                          {label(d.embodiment)}
                          <small className="muted">
                            {d.num_episodes
                              ? `${d.state_dim}-D · ${d.num_episodes} ep${d.env_state_names?.length ? " · object" : ""}`
                              : "no usable tracking"}
                          </small>
                        </th>
                        {formats.map((f) => (
                          <td key={f}>
                            {d.exports[f] ? (
                              <a href={api.downloadUrl(job, d.embodiment, f)}>.zip</a>
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
