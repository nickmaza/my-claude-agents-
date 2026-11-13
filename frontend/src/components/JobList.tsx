import { Job } from "../api";

export function StatusBadge({ status }: { status: Job["status"] }) {
  return <span className={`badge ${status}`}>{status}</span>;
}

export function timeAgo(ts: number): string {
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return new Date(ts * 1000).toLocaleDateString();
}

interface Props {
  jobs: Job[];
  selected: string | null;
  onSelect: (id: string) => void;
}

export default function JobList({ jobs, selected, onSelect }: Props) {
  if (!jobs.length) return <p className="muted small pad">No conversions yet.</p>;
  return (
    <ul className="joblist">
      {jobs.map((job) => (
        <li key={job.id}>
          <button className={`jobitem ${job.id === selected ? "current" : ""}`} onClick={() => onSelect(job.id)}>
            <span className="jobtitle">{job.task}</span>
            <span className="jobmeta">
              <StatusBadge status={job.status} />
              <span>
                {job.clips.length} clip{job.clips.length === 1 ? "" : "s"} · {timeAgo(job.created_at)}
              </span>
            </span>
            {(job.status === "running" || job.status === "queued") && (
              <span className="bar">
                <span style={{ width: `${Math.round(job.progress * 100)}%` }} />
              </span>
            )}
          </button>
        </li>
      ))}
    </ul>
  );
}
