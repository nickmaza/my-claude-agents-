import { useCallback, useEffect, useRef, useState } from "react";

interface Props {
  onTake: (file: File) => void;
  takes: number;
}

const MIME_CANDIDATES = ["video/mp4;codecs=avc1", "video/mp4", "video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];

function pickMime(): string | undefined {
  if (typeof MediaRecorder === "undefined") return undefined;
  return MIME_CANDIDATES.find((m) => MediaRecorder.isTypeSupported(m));
}

function cameraError(e: unknown): string {
  const name = (e as DOMException)?.name;
  if (name === "NotAllowedError") return "Camera access was blocked. Allow the camera for this site in your browser settings, then try again.";
  if (name === "NotFoundError" || name === "OverconstrainedError") return "No camera was found on this device.";
  if (name === "NotReadableError") return "The camera is in use by another app. Close it and try again.";
  return `Could not open the camera: ${(e as Error)?.message ?? "unknown error"}`;
}

export default function Recorder({ onTake, takes }: Props) {
  const video = useRef<HTMLVideoElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const recorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const startedAt = useRef(0);
  const [facing, setFacing] = useState<"environment" | "user">("environment");
  const [state, setState] = useState<"idle" | "starting" | "ready" | "countdown" | "recording">("idle");
  const [count, setCount] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const supported =
    typeof navigator !== "undefined" && Boolean(navigator.mediaDevices?.getUserMedia) && typeof MediaRecorder !== "undefined";
  const secure = typeof window !== "undefined" && window.isSecureContext;

  const stopStream = useCallback(() => {
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
  }, []);

  const open = useCallback(
    async (mode: "environment" | "user") => {
      setError(null);
      setState("starting");
      stopStream();
      try {
        const s = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: { facingMode: { ideal: mode }, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
        });
        stream.current = s;
        if (video.current) {
          video.current.srcObject = s;
          await video.current.play().catch(() => undefined);
        }
        setState("ready");
      } catch (e) {
        setError(cameraError(e));
        setState("idle");
      }
    },
    [stopStream],
  );

  useEffect(() => stopStream, [stopStream]); // release the camera when leaving the screen

  useEffect(() => {
    if (state !== "recording") return;
    const timer = window.setInterval(() => setElapsed((performance.now() - startedAt.current) / 1000), 200);
    return () => window.clearInterval(timer);
  }, [state]);

  const beginRecording = useCallback(() => {
    if (!stream.current) return;
    const mime = pickMime();
    const rec = new MediaRecorder(stream.current, mime ? { mimeType: mime, videoBitsPerSecond: 6_000_000 } : undefined);
    chunks.current = [];
    rec.ondataavailable = (e) => e.data.size > 0 && chunks.current.push(e.data);
    rec.onstop = () => {
      const type = rec.mimeType || mime || "video/webm";
      const blob = new Blob(chunks.current, { type });
      chunks.current = [];
      if (blob.size > 0) {
        const ext = type.includes("mp4") ? "mp4" : "webm";
        onTake(new File([blob], `take_${String(takes + 1).padStart(2, "0")}.${ext}`, { type }));
      }
    };
    recorder.current = rec;
    rec.start(1000);
    startedAt.current = performance.now();
    setElapsed(0);
    setState("recording");
  }, [onTake, takes]);

  // 3-2-1 so the person can get their hands into position
  useEffect(() => {
    if (state !== "countdown") return;
    if (count === 0) {
      beginRecording();
      return;
    }
    const timer = window.setTimeout(() => setCount((c) => c - 1), 1000);
    return () => window.clearTimeout(timer);
  }, [state, count, beginRecording]);

  const stopRecording = () => {
    recorder.current?.stop();
    recorder.current = null;
    setState("ready");
  };

  if (!supported || !secure) {
    return (
      <div className="banner warn">
        <strong>Recording isn&apos;t available here.</strong>{" "}
        {!secure
          ? "Browsers only allow camera access on secure (https) pages. Open this app through its https address — see the README for how to serve it that way — or upload a video instead."
          : "This browser can't record video. Upload a video file instead."}
      </div>
    );
  }

  return (
    <div className="recorder">
      <div className={`viewfinder ${state === "recording" ? "live" : ""}`}>
        <video ref={video} muted playsInline autoPlay className={facing === "user" ? "mirror" : ""} />
        {state === "idle" && (
          <button type="button" className="primary" onClick={() => void open(facing)}>
            Turn on camera
          </button>
        )}
        {state === "starting" && <span className="hint">Opening camera…</span>}
        {state === "countdown" && <span className="count">{count}</span>}
        {state === "recording" && (
          <span className="rec">
            <i /> {Math.floor(elapsed / 60)}:{String(Math.floor(elapsed % 60)).padStart(2, "0")}
          </span>
        )}
      </div>

      {error && (
        <div className="banner error" role="alert">
          {error}
        </div>
      )}

      {state !== "idle" && state !== "starting" && (
        <div className="rec-controls">
          {state === "recording" ? (
            <button type="button" className="shutter stop" onClick={stopRecording} aria-label="Stop recording">
              <i />
            </button>
          ) : (
            <button
              type="button"
              className="shutter"
              disabled={state === "countdown"}
              onClick={() => {
                setCount(3);
                setState("countdown");
              }}
              aria-label="Start recording a demonstration"
            >
              <i />
            </button>
          )}
          <button
            type="button"
            className="secondary"
            disabled={state !== "ready"}
            onClick={() => {
              const next = facing === "environment" ? "user" : "environment";
              setFacing(next);
              void open(next);
            }}
          >
            Flip camera
          </button>
        </div>
      )}
      <p className="muted small">
        Prop the phone so the whole upper body and both hands stay in view. Press record once per demonstration — each
        take becomes one episode. {takes > 0 && <strong>{takes} take{takes === 1 ? "" : "s"} recorded.</strong>}
      </p>
    </div>
  );
}
