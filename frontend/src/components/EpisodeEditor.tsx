interface Props {
  segments: [number, number][];
  total: number;
  frame: number;
  fps: number;
  onChange: (segments: [number, number][]) => void;
  onSeek: (frame: number) => void;
}

const MIN = 5; // frames

export default function EpisodeEditor({ segments, total, frame, fps, onChange, onSeek }: Props) {
  const index = segments.findIndex(([a, b]) => frame >= a && frame < b);
  const cur = index >= 0 ? segments[index] : null;
  const replace = (i: number, ...next: [number, number][]) =>
    onChange([...segments.slice(0, i), ...next, ...segments.slice(i + 1)]);
  const pct = (f: number) => `${(f / total) * 100}%`;
  const t = (f: number) => `${(f / fps).toFixed(1)}s`;

  const addHere = () => {
    const nextStart = segments.find(([a]) => a > frame)?.[0] ?? total;
    const prevEnd = [...segments].reverse().find(([, b]) => b <= frame)?.[1] ?? 0;
    const a = Math.max(prevEnd, Math.min(frame, nextStart - MIN));
    if (nextStart - a < MIN) return;
    onChange([...segments, [a, nextStart] as [number, number]].sort((p, q) => p[0] - q[0]));
  };

  return (
    <div className="editor">
      <div
        className="timeline"
        role="slider"
        aria-label="Episodes timeline"
        aria-valuemin={0}
        aria-valuemax={total - 1}
        aria-valuenow={frame}
        tabIndex={0}
        onClick={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          onSeek(Math.max(0, Math.min(total - 1, Math.round(((e.clientX - r.left) / r.width) * total))));
        }}
        onKeyDown={(e) => {
          if (e.key === "ArrowLeft") onSeek(Math.max(0, frame - 1));
          if (e.key === "ArrowRight") onSeek(Math.min(total - 1, frame + 1));
        }}
      >
        {segments.map(([a, b], i) => (
          <span key={`${a}-${b}`} className={`seg ${i === index ? "current" : ""}`} style={{ left: pct(a), width: pct(b - a) }}>
            {i + 1}
          </span>
        ))}
        <span className="playhead" style={{ left: pct(frame) }} />
      </div>
      <div className="editor-actions">
        <button
          type="button"
          disabled={!cur || frame - cur[0] < MIN || cur[1] - frame < MIN}
          onClick={() => cur && replace(index, [cur[0], frame], [frame, cur[1]])}
        >
          Split here
        </button>
        <button type="button" disabled={!cur || cur[1] - frame < MIN || frame === cur[0]} onClick={() => cur && replace(index, [frame, cur[1]])}>
          Start here
        </button>
        <button type="button" disabled={!cur || frame + 1 - cur[0] < MIN || frame + 1 === cur[1]} onClick={() => cur && replace(index, [cur[0], frame + 1])}>
          End here
        </button>
        <button type="button" className="danger" disabled={!cur || segments.length < 2} onClick={() => replace(index)}>
          Remove episode
        </button>
        <button type="button" disabled={Boolean(cur)} onClick={addHere}>
          Add episode here
        </button>
        <button type="button" disabled={segments.length === 1 && segments[0][0] === 0 && segments[0][1] === total} onClick={() => onChange([[0, total]])}>
          Whole clip
        </button>
      </div>
      <p className="muted small">
        {segments.length} episode{segments.length === 1 ? "" : "s"}
        {cur ? ` · playhead in episode ${index + 1} (${t(cur[0])} – ${t(cur[1])})` : " · playhead is between episodes"}
      </p>
    </div>
  );
}
