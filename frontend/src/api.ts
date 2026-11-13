export interface Options {
  version: string;
  embodiments: Record<string, string>;
  formats: Record<string, string>;
  action_modes: Record<string, string>;
  rotations: string[];
  video_extensions: string[];
  max_upload_mb: number;
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
}

export interface ClipSummary {
  name: string;
  frames: number;
  fps: number;
  duration_s: number;
  coverage: { body: number; left_hand: number; right_hand: number };
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
  episodes?: { length: number; clip: number; hand?: string; tracked_fraction: number; defaulted_dims: string[] }[];
}

export interface Job {
  id: string;
  status: "queued" | "running" | "done" | "failed";
  progress: number;
  message: string;
  created_at: number;
  finished_at: number | null;
  error: string | null;
  task: string;
  clips: string[];
  embodiments: string[];
  formats: string[];
  summary?: { clips: ClipSummary[]; datasets: DatasetSummary[]; warnings: string[]; elapsed_s: number } | null;
}

export interface Overlay {
  fps: number;
  width: number;
  height: number;
  start_s: number;
  body_edges: [number, number][];
  hand_edges: [number, number][];
  body: ([number, number] | [null, null])[][];
  body_conf: (number | null)[][];
  hands: ([number, number] | [null, null])[][][];
}

export interface Preview {
  state_names: string[];
  fps: number;
  episodes: { clip: number; offset: number; state: number[][] }[];
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  options: () => request<Options>("/api/options"),
  jobs: () => request<Job[]>("/api/jobs"),
  job: (id: string) => request<Job>(`/api/jobs/${id}`),
  remove: (id: string) => request<unknown>(`/api/jobs/${id}`, { method: "DELETE" }),
  overlay: (id: string, clip: number) => request<Overlay>(`/api/jobs/${id}/clips/${clip}/overlay`),
  preview: (id: string, embodiment: string) => request<Preview>(`/api/jobs/${id}/datasets/${embodiment}/preview`),
  videoUrl: (id: string, clip: number) => `/api/jobs/${id}/clips/${clip}/video`,
  downloadUrl: (id: string, embodiment?: string, format?: string) => {
    const q = new URLSearchParams();
    if (embodiment) q.set("embodiment", embodiment);
    if (format) q.set("format", format);
    const s = q.toString();
    return `/api/jobs/${id}/download${s ? `?${s}` : ""}`;
  },
  create: (files: File[], config: JobConfig, demo = false) => {
    const form = new FormData();
    form.set("config", JSON.stringify(config));
    files.forEach((f) => form.append("files", f));
    return request<Job>(demo ? "/api/jobs/demo" : "/api/jobs", { method: "POST", body: form });
  },
};
