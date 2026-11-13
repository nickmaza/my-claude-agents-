import { useCallback, useEffect, useRef, useState } from "react";
import { Overlay } from "../api";

interface Props {
  overlay: Overlay;
  videoUrl: string | null;
  frame: number;
  onFrame: (frame: number) => void;
}

const COLORS = { body: "#4cc9f0", left: "#2ec4b6", right: "#ff9f1c", joint: "#ffffff" };

function draw(canvas: HTMLCanvasElement, overlay: Overlay, frame: number, hasVideo: boolean) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const { width: w, height: h } = canvas;
  ctx.clearRect(0, 0, w, h);
  if (!hasVideo) {
    ctx.fillStyle = "#11161d";
    ctx.fillRect(0, 0, w, h);
  }
  const scale = w / 640;
  const line = (pts: ([number, number] | [null, null])[], edges: [number, number][], color: string, width: number) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = width * scale;
    ctx.lineCap = "round";
    ctx.beginPath();
    for (const [a, b] of edges) {
      const p = pts[a];
      const q = pts[b];
      if (!p || !q || p[0] == null || q[0] == null) continue;
      ctx.moveTo(p[0] * w, p[1]! * h);
      ctx.lineTo(q[0] * w, q[1]! * h);
    }
    ctx.stroke();
  };
  const dots = (pts: ([number, number] | [null, null])[], from: number, radius: number) => {
    ctx.fillStyle = COLORS.joint;
    for (let i = from; i < pts.length; i++) {
      const p = pts[i];
      if (!p || p[0] == null) continue;
      ctx.beginPath();
      ctx.arc(p[0] * w, p[1]! * h, radius * scale, 0, Math.PI * 2);
      ctx.fill();
    }
  };
  const body = overlay.body[frame];
  if (body) {
    line(body, overlay.body_edges, COLORS.body, 3.5);
    dots(body, 11, 3);
  }
  const hands = overlay.hands[frame];
  if (hands) {
    hands.forEach((hand, side) => {
      line(hand, overlay.hand_edges, side === 0 ? COLORS.left : COLORS.right, 2.5);
      dots(hand, 0, 1.8);
    });
  }
}

export default function Player({ overlay, videoUrl, frame, onFrame }: Props) {
  const total = overlay.body.length;
  const duration = total / overlay.fps;
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const video = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);
  const [videoOk, setVideoOk] = useState(Boolean(videoUrl));
  const time = useRef(frame / overlay.fps);
  const hasVideo = Boolean(videoUrl) && videoOk;

  // keep the canvas backing store matched to its displayed size
  useEffect(() => {
    const el = wrap.current;
    const cv = canvas.current;
    if (!el || !cv) return;
    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      cv.width = Math.round(el.clientWidth * dpr);
      cv.height = Math.round(el.clientHeight * dpr);
      draw(cv, overlay, Math.min(total - 1, Math.round(time.current * overlay.fps)), hasVideo);
    };
    resize();
    const obs = new ResizeObserver(resize);
    obs.observe(el);
    return () => obs.disconnect();
  }, [overlay, total, hasVideo]);

  useEffect(() => {
    if (canvas.current) draw(canvas.current, overlay, frame, hasVideo);
  }, [overlay, frame, hasVideo]);

  const seek = useCallback(
    (t: number) => {
      const clamped = Math.max(0, Math.min(duration - 1 / overlay.fps, t));
      time.current = clamped;
      if (hasVideo && video.current) video.current.currentTime = overlay.start_s + clamped;
      onFrame(Math.round(clamped * overlay.fps));
    },
    [duration, hasVideo, onFrame, overlay.fps, overlay.start_s],
  );

  // external seeks (e.g. clicking the chart)
  useEffect(() => {
    if (Math.abs(frame / overlay.fps - time.current) > 1.5 / overlay.fps) seek(frame / overlay.fps);
  }, [frame, overlay.fps, seek]);

  useEffect(() => {
    if (!playing) {
      video.current?.pause();
      return;
    }
    if (time.current >= duration - 2 / overlay.fps) seek(0);
    if (hasVideo) void video.current?.play().catch(() => setVideoOk(false));
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      if (hasVideo && video.current) time.current = video.current.currentTime - overlay.start_s;
      else time.current += (now - last) / 1000;
      last = now;
      if (time.current >= duration - 1 / overlay.fps) {
        time.current = duration - 1 / overlay.fps;
        onFrame(total - 1);
        setPlaying(false);
        return;
      }
      onFrame(Math.max(0, Math.round(time.current * overlay.fps)));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, hasVideo, duration, overlay.fps, overlay.start_s, onFrame, seek, total]);

  return (
    <div className="player">
      <div className="stage" ref={wrap} style={{
          aspectRatio: `${overlay.width} / ${overlay.height}`,
          maxWidth: `calc(62vh * ${(overlay.width / overlay.height).toFixed(4)})`,
        }}>
        {videoUrl && videoOk && (
          <video ref={video} src={videoUrl} muted playsInline preload="auto" onError={() => setVideoOk(false)} />
        )}
        <canvas ref={canvas} />
        {videoUrl && !videoOk && (
          <span className="stage-note">This browser can&apos;t play the clip&apos;s codec — showing tracking only.</span>
        )}
      </div>
      <div className="controls">
        <button type="button" className="play" onClick={() => setPlaying((p) => !p)} aria-label={playing ? "Pause" : "Play"}>
          {playing ? "❚❚" : "▶"}
        </button>
        <input
          type="range"
          min={0}
          max={total - 1}
          value={frame}
          aria-label="Seek"
          onChange={(e) => {
            setPlaying(false);
            seek(Number(e.target.value) / overlay.fps);
          }}
        />
        <span className="clock">
          {(frame / overlay.fps).toFixed(2)}s / {duration.toFixed(2)}s
        </span>
      </div>
      <div className="legend">
        <span>
          <i style={{ background: COLORS.body }} /> body
        </span>
        <span>
          <i style={{ background: COLORS.left }} /> left hand
        </span>
        <span>
          <i style={{ background: COLORS.right }} /> right hand
        </span>
      </div>
    </div>
  );
}
