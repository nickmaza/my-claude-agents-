import { useMemo, useState } from "react";

interface Props {
  names: string[];
  state: number[][];
  offset: number; // clip frame of the episode's first step
  clipFrames: number;
  frame: number;
  fps: number;
  onSeek: (frame: number) => void;
}

const PALETTE = ["#4cc9f0", "#ff9f1c", "#2ec4b6", "#e71d36", "#b388ff", "#ffd166", "#06d6a0", "#f78fb3", "#9aa5b1"];
const W = 800;
const H = 220;
const PAD = { l: 44, r: 8, t: 8, b: 20 };

export default function SignalChart({ names, state, offset, clipFrames, frame, fps, onSeek }: Props) {
  const [selected, setSelected] = useState<number[]>(() => names.slice(0, Math.min(7, names.length)).map((_, i) => i));
  const [filter, setFilter] = useState("");

  const { paths, lo, hi } = useMemo(() => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const row of state)
      for (const d of selected) {
        if (row[d] < lo) lo = row[d];
        if (row[d] > hi) hi = row[d];
      }
    if (!isFinite(lo)) [lo, hi] = [0, 1];
    if (hi - lo < 1e-6) [lo, hi] = [lo - 0.5, hi + 0.5];
    const x = (k: number) => PAD.l + ((offset + k) / Math.max(1, clipFrames - 1)) * (W - PAD.l - PAD.r);
    const y = (v: number) => PAD.t + (1 - (v - lo) / (hi - lo)) * (H - PAD.t - PAD.b);
    const paths = selected.map((d) => state.map((row, k) => `${x(k).toFixed(1)},${y(row[d]).toFixed(1)}`).join(" "));
    return { paths, lo, hi };
  }, [state, selected, offset, clipFrames]);

  const cursorX = PAD.l + (frame / Math.max(1, clipFrames - 1)) * (W - PAD.l - PAD.r);
  const step = frame - offset;
  const row = step >= 0 && step < state.length ? state[step] : null;
  const shown = names.map((n, i) => [n, i] as const).filter(([n]) => n.toLowerCase().includes(filter.toLowerCase()));

  return (
    <div className="chart">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="Robot state over time"
        onClick={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const px = ((e.clientX - rect.left) / rect.width) * W;
          const f = ((px - PAD.l) / (W - PAD.l - PAD.r)) * (clipFrames - 1);
          onSeek(Math.max(0, Math.min(clipFrames - 1, Math.round(f))));
        }}
      >
        <rect x={PAD.l} y={PAD.t} width={W - PAD.l - PAD.r} height={H - PAD.t - PAD.b} className="plot-bg" />
        {[0, 0.5, 1].map((f) => (
          <g key={f}>
            <line
              x1={PAD.l}
              x2={W - PAD.r}
              y1={PAD.t + f * (H - PAD.t - PAD.b)}
              y2={PAD.t + f * (H - PAD.t - PAD.b)}
              className="grid-line"
            />
            <text x={PAD.l - 6} y={PAD.t + f * (H - PAD.t - PAD.b) + 4} textAnchor="end" className="axis">
              {(hi - f * (hi - lo)).toFixed(2)}
            </text>
          </g>
        ))}
        <text x={PAD.l} y={H - 4} className="axis">
          0s
        </text>
        <text x={W - PAD.r} y={H - 4} textAnchor="end" className="axis">
          {(clipFrames / fps).toFixed(1)}s
        </text>
        {paths.map((pts, i) => (
          <polyline key={selected[i]} points={pts} fill="none" stroke={PALETTE[i % PALETTE.length]} strokeWidth={1.8} />
        ))}
        <line x1={cursorX} x2={cursorX} y1={PAD.t} y2={H - PAD.b} className="cursor" />
      </svg>

      {names.length > 12 && (
        <input
          className="chip-filter"
          type="search"
          placeholder={`Filter ${names.length} signals…`}
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
      )}
      <div className="chips">
        {shown.map(([name, i]) => {
          const pos = selected.indexOf(i);
          return (
            <button
              type="button"
              key={name}
              className={`chip ${pos >= 0 ? "on" : ""}`}
              style={pos >= 0 ? { borderColor: PALETTE[pos % PALETTE.length] } : undefined}
              onClick={() => setSelected(pos >= 0 ? selected.filter((d) => d !== i) : [...selected, i])}
            >
              {pos >= 0 && <i style={{ background: PALETTE[pos % PALETTE.length] }} />}
              {name}
              {pos >= 0 && row && <b>{row[i].toFixed(2)}</b>}
            </button>
          );
        })}
      </div>
    </div>
  );
}
