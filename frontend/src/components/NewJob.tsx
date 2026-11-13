import { DragEvent, useRef, useState } from "react";
import { api, Job, JobConfig, Options } from "../api";
import Recorder from "./Recorder";

interface Props {
  options: Options;
  onCreated: (job: Job) => void;
}

function CheckGroup(props: {
  title: string;
  hint: string;
  items: Record<string, string>;
  value: string[];
  onChange: (next: string[]) => void;
}) {
  const { title, hint, items, value, onChange } = props;
  const keys = Object.keys(items);
  const all = value.length === keys.length;
  return (
    <fieldset className="group">
      <legend>
        {title}
        <button type="button" className="link" onClick={() => onChange(all ? [] : keys)}>
          {all ? "clear" : "select all"}
        </button>
      </legend>
      <p className="muted small">{hint}</p>
      <div className="checks">
        {keys.map((key) => {
          const [name, ...rest] = items[key].split(" — ");
          return (
            <label key={key} className={`check ${value.includes(key) ? "on" : ""}`}>
              <input
                type="checkbox"
                checked={value.includes(key)}
                onChange={(e) => onChange(e.target.checked ? [...value, key] : value.filter((v) => v !== key))}
              />
              <span>
                <strong>{name}</strong>
                {rest.length > 0 && <small>{rest.join(" — ")}</small>}
              </span>
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}

export default function NewJob({ options, onCreated }: Props) {
  const [files, setFiles] = useState<File[]>([]);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [source, setSource] = useState<"upload" | "record">("upload");
  const [config, setConfig] = useState<JobConfig>({
    task: "",
    embodiments: options.generic_embodiments,
    formats: Object.keys(options.formats).filter((f) => f !== "lerobot_v21"),
    target_fps: 30,
    action_mode: "absolute",
    rotation: "euler",
    side: "auto",
    include_images: true,
    person_height_m: null,
    arm_length_m: null,
    marker_size_m: null,
    split: "none",
    track_objects: false,
  });
  const pick = (keys: string[]) =>
    Object.fromEntries(
      keys.map((k) => {
        const m = options.embodiments[k].match(/^(.*?) \((.*)\)$/);
        return [k, m ? `${m[1]} — ${m[2]}` : options.embodiments[k]];
      }),
    );
  const numberOrNull = (v: string) => (v.trim() === "" || !isFinite(Number(v)) ? null : Number(v));
  const input = useRef<HTMLInputElement>(null);
  const set = <K extends keyof JobConfig>(key: K, value: JobConfig[K]) => setConfig((c) => ({ ...c, [key]: value }));

  const addFiles = (list: FileList | null) => {
    if (!list) return;
    const ok = Array.from(list).filter((f) =>
      options.video_extensions.some((ext) => f.name.toLowerCase().endsWith(ext)),
    );
    if (ok.length < list.length) setError(`Only video files are accepted (${options.video_extensions.join(", ")}).`);
    else setError(null);
    setFiles((prev) => [...prev, ...ok]);
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    addFiles(e.dataTransfer.files);
  };

  const submit = async (demo: boolean) => {
    setBusy(true);
    setError(null);
    try {
      const task = config.task.trim() || "do the demonstrated task";
      onCreated(await api.create(demo ? [] : files, { ...config, task }, demo));
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  };

  const totalMb = files.reduce((n, f) => n + f.size, 0) / 1e6;
  const ready = files.length > 0 && config.embodiments.length > 0 && config.formats.length > 0;

  return (
    <form
      className="newjob"
      onSubmit={(e) => {
        e.preventDefault();
        if (ready && !busy) void submit(false);
      }}
    >
      <h1>New conversion</h1>
      <p className="muted">
        Upload footage of a person performing a task. Each clip becomes one demonstration episode: the person&apos;s body
        and hands are tracked in 3D, retargeted to robot states and actions, and exported as ready-to-train datasets.
      </p>

      <div className="tabs source" role="tablist">
        <button type="button" role="tab" aria-selected={source === "upload"} className={source === "upload" ? "on" : ""} onClick={() => setSource("upload")}>
          Upload videos
        </button>
        <button type="button" role="tab" aria-selected={source === "record"} className={source === "record" ? "on" : ""} onClick={() => setSource("record")}>
          Record with camera
        </button>
      </div>

      {source === "record" ? (
        <Recorder takes={files.length} onTake={(file) => setFiles((prev) => [...prev, file])} />
      ) : (
      <div
        className={`dropzone ${dragging ? "dragging" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => input.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
      >
        <input
          ref={input}
          type="file"
          accept="video/*,.mkv"
          multiple
          hidden
          onChange={(e) => {
            addFiles(e.target.files);
            e.target.value = "";
          }}
        />
        <strong>Drop videos here or click to browse</strong>
        <span className="muted small">
          {options.video_extensions.join(" ")} · up to {options.max_upload_mb} MB total · one demonstration per clip
        </span>
      </div>
      )}

      {files.length > 0 && (
        <ul className="files">
          {files.map((f, i) => (
            <li key={`${f.name}-${i}`}>
              <span className="fname">{f.name}</span>
              <span className="muted small">{(f.size / 1e6).toFixed(1)} MB</span>
              <button
                type="button"
                className="link"
                aria-label={`Remove ${f.name}`}
                onClick={() => setFiles(files.filter((_, k) => k !== i))}
              >
                remove
              </button>
            </li>
          ))}
          <li className="muted small">
            {files.length} clip{files.length === 1 ? "" : "s"}, {totalMb.toFixed(1)} MB
          </li>
        </ul>
      )}

      <label className="field">
        <span>Task instruction</span>
        <input
          type="text"
          value={config.task}
          maxLength={500}
          placeholder="e.g. pick up the red cup and place it on the shelf"
          onChange={(e) => set("task", e.target.value)}
        />
        <small className="muted">Stored as the language instruction on every episode.</small>
      </label>

      <CheckGroup
        title="Generic robot types"
        hint="Robot-independent states (hand poses, generic joint angles). One dataset is produced per type."
        items={pick(options.generic_embodiments)}
        value={config.embodiments.filter((e) => options.generic_embodiments.includes(e))}
        onChange={(v) => set("embodiments", [...v, ...config.embodiments.filter((e) => options.robots.includes(e))])}
      />
      <CheckGroup
        title="Specific robots"
        hint="Real joint commands for a particular robot, solved on its model and kept within its joint limits."
        items={pick(options.robots)}
        value={config.embodiments.filter((e) => options.robots.includes(e))}
        onChange={(v) => set("embodiments", [...config.embodiments.filter((e) => !options.robots.includes(e)), ...v])}
      />
      <CheckGroup
        title="Export formats"
        hint="Every selected embodiment is written in every selected format."
        items={options.formats}
        value={config.formats}
        onChange={(v) => set("formats", v)}
      />

      <fieldset className="group">
        <legend>Episodes and objects</legend>
        <label className="field inline">
          <input type="checkbox" checked={config.split === "auto"} onChange={(e) => set("split", e.target.checked ? "auto" : "none")} />
          <span>
            Split long clips into separate demonstrations
            <small className="muted block">Cuts where the hands pause with nothing held. You can adjust the cuts afterwards.</small>
          </span>
        </label>
        <label className="field inline">
          <input type="checkbox" checked={config.track_objects} onChange={(e) => set("track_objects", e.target.checked)} />
          <span>
            Track the object being handled
            <small className="muted block">Follows the object in the image and adds its position to the dataset. Slower.</small>
          </span>
        </label>
      </fieldset>

      <fieldset className="group">
        <legend>Real-world scale (optional)</legend>
        <p className="muted small">
          A single camera can only guess distances. Give one body measurement to get true metres, and optionally print
          the <a href="/api/marker.png" target="_blank" rel="noreferrer">workspace marker</a> and lay it on the table
          to fix where the workspace is.
        </p>
        <div className="grid">
          <label className="field">
            <span>Person&apos;s height (m)</span>
            <input type="number" inputMode="decimal" step="0.01" min="0.8" max="2.5" placeholder="e.g. 1.75"
              value={config.person_height_m ?? ""} onChange={(e) => set("person_height_m", numberOrNull(e.target.value))} />
          </label>
          <label className="field">
            <span>Shoulder-to-wrist length (m)</span>
            <input type="number" inputMode="decimal" step="0.01" min="0.25" max="1" placeholder="more accurate, e.g. 0.55"
              value={config.arm_length_m ?? ""} onChange={(e) => set("arm_length_m", numberOrNull(e.target.value))} />
          </label>
          <label className="field">
            <span>Marker edge length (m)</span>
            <input type="number" inputMode="decimal" step="0.005" min="0.02" max="1" placeholder="only if the marker is in view"
              value={config.marker_size_m ?? ""} onChange={(e) => set("marker_size_m", numberOrNull(e.target.value))} />
          </label>
        </div>
      </fieldset>

      <details className="advanced">
        <summary>Advanced settings</summary>
        <div className="grid">
          <label className="field">
            <span>Frame rate (fps)</span>
            <select
              value={config.target_fps ?? 0}
              onChange={(e) => set("target_fps", Number(e.target.value) || null)}
            >
              <option value={0}>Keep source</option>
              {[10, 15, 20, 30, 50, 60].map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Actions</span>
            <select value={config.action_mode} onChange={(e) => set("action_mode", e.target.value)}>
              {Object.entries(options.action_modes).map(([k, v]) => (
                <option key={k} value={k} title={v}>
                  {k}
                </option>
              ))}
            </select>
            <small className="muted">{options.action_modes[config.action_mode]}</small>
          </label>
          <label className="field">
            <span>Rotation encoding</span>
            <select value={config.rotation} onChange={(e) => set("rotation", e.target.value)}>
              {options.rotations.map((r) => (
                <option key={r}>{r}</option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Hand (single-hand robots)</span>
            <select value={config.side} onChange={(e) => set("side", e.target.value)}>
              <option value="auto">Auto (most active)</option>
              <option value="right">Right</option>
              <option value="left">Left</option>
            </select>
          </label>
          <label className="field inline">
            <input
              type="checkbox"
              checked={config.include_images}
              onChange={(e) => set("include_images", e.target.checked)}
            />
            <span>Include camera images in datasets</span>
          </label>
        </div>
      </details>

      {error && (
        <div className="banner error" role="alert">
          {error}
        </div>
      )}

      <div className="actions">
        <button type="submit" className="primary" disabled={!ready || busy}>
          {busy ? "Uploading…" : "Convert footage"}
        </button>
        <button
          type="button"
          className="secondary"
          disabled={busy || !config.embodiments.length || !config.formats.length}
          onClick={() => void submit(true)}
          title="Runs the pipeline on generated motion so you can explore the output without footage"
        >
          Try with a synthetic demo
        </button>
      </div>
    </form>
  );
}
