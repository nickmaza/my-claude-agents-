export interface Options {
  version: string;
  embodiments: Record<string, string>;
  generic_embodiments: string[];
  robots: string[];
  formats: Record<string, string>;
  action_modes: Record<string, string>;
  rotations: string[];
  video_extensions: string[];
  max_upload_mb: number;
  max_clips: number;
}

export interface ClientConfig {
  version: string;
  auth_required: boolean;
  firebase: Record<string, string> | null;
}

export interface JobConfig {
  task: string;
  embodiments: string[];
  formats: string[];
  target_fps: number | null;
  action_mode: string;
  rotation: string;
  side: string;
  include_images: boolean;
  person_height_m: number | null;
  arm_length_m: number | null;
  marker_size_m: number | null;
  split: "none" | "auto";
  track_objects: boolean;
}

export interface ClipSummary {
  name: string;
  frames: number;
  fps: number;
  duration_s: number;
  coverage: { body: number; left_hand: number; right_hand: number };
  calibration: {
    method?: "height" | "arm_length";
    scale: number;
    tracked_height_m: number;
    marker: boolean;
    marker_distance_m?: number;
  };
  segments: [number, number][];
  object_prompt: [number, number, number] | null;
  object_tracked?: boolean;
  has_video: boolean;
}

export interface EpisodeSummary {
  length: number;
  clip: number;
  segment: number;
  hand?: string;
  tracked_fraction: number;
  defaulted_dims: string[];
  ik_error_mean_m?: number;
  ik_error_max_m?: number;
  ik_reachable_fraction?: number;
  limit_clipped_fraction?: number;
  mapping?: string;
  notes?: string;
}

export interface DatasetSummary {
  embodiment: string;
  description: string;
  exports: Record<string, string>;
  errors: Record<string, string>;
  num_episodes?: number;
  total_frames?: number;
  state_dim?: number;
  state_names?: string[];
  has_images?: boolean;
  env_state_names?: string[];
  episodes?: EpisodeSummary[];
}

export interface Job {
  id: string;
  status: "queued" | "running" | "done" | "failed";
  stage: "convert" | "rebuild";
  progress: number;
  message: string;
  created_at: number;
  finished_at: number | null;
  error: string | null;
  task: string;
  clips: string[];
  embodiments: string[];
  formats: string[];
  access_key?: string;
  summary?: { clips: ClipSummary[]; datasets: DatasetSummary[]; warnings: string[]; elapsed_s: number } | null;
}

export type Point = [number, number] | [null, null];

export interface Overlay {
  fps: number;
  width: number;
  height: number;
  start_s: number;
  body_edges: [number, number][];
  hand_edges: [number, number][];
  body: Point[][];
  body_conf: (number | null)[][];
  hands: Point[][][];
  segments: [number, number][];
  object_prompt: [number, number, number] | null;
  object: number[][] | null; // [cx, cy, w, h, visible] per frame
}

export interface Preview {
  state_names: string[];
  fps: number;
  episodes: { clip: number; offset: number; state: number[][] }[];
}

export interface ClipEdit {
  segments?: [number, number][];
  object_prompt?: [number, number, number] | null;
}

let tokenProvider: (() => Promise<string | null>) | null = null;
export function setTokenProvider(fn: (() => Promise<string | null>) | null) {
  tokenProvider = fn;
}

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

async function request<T>(url: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const token = tokenProvider ? await tokenProvider() : null;
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const res = await fetch(url, { ...init, headers });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(detail || `Request failed (${res.status})`, res.status);
  }
  return res.json() as Promise<T>;
}

const withKey = (url: string, key?: string) => (key ? `${url}${url.includes("?") ? "&" : "?"}key=${encodeURIComponent(key)}` : url);

export const api = {
  config: () => request<ClientConfig>("/api/config"),
  options: () => request<Options>("/api/options"),
  jobs: () => request<Job[]>("/api/jobs"),
  job: (id: string) => request<Job>(`/api/jobs/${id}`),
  remove: (id: string) => request<unknown>(`/api/jobs/${id}`, { method: "DELETE" }),
  overlay: (id: string, clip: number) => request<Overlay>(`/api/jobs/${id}/clips/${clip}/overlay`),
  preview: (id: string, embodiment: string) => request<Preview>(`/api/jobs/${id}/datasets/${embodiment}/preview`),
  edit: (id: string, clips: Record<string, ClipEdit>, task?: string) =>
    request<Job>(`/api/jobs/${id}/edits`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clips, task }),
    }),
  videoUrl: (job: Job, clip: number) => withKey(`/api/jobs/${job.id}/clips/${clip}/video`, job.access_key),
  downloadUrl: (job: Job, embodiment?: string, format?: string) => {
    const q = new URLSearchParams();
    if (embodiment) q.set("embodiment", embodiment);
    if (format) q.set("format", format);
    const s = q.toString();
    return withKey(`/api/jobs/${job.id}/download${s ? `?${s}` : ""}`, job.access_key);
  },
  create: (files: File[], config: JobConfig, demo = false) => {
    const form = new FormData();
    form.set("config", JSON.stringify(config));
    files.forEach((f) => form.append("files", f));
    return request<Job>(demo ? "/api/jobs/demo" : "/api/jobs", { method: "POST", body: form });
  },
};
