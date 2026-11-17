import { Icon, Spinner } from "../../shared/components/UI";
import { extractionLabel } from "./labels";

export default function ExtractionProgress({ job, onRetry, t }) {
  if (!job) return null;
  const pending = ["checking", "queued", "extracting", "importing"].includes(job.status);
  // One bar for the whole run; a stage still loading its model has no item count to show yet.
  const fraction = typeof job.overall === "number" ? job.overall : job.progress;
  const progress = typeof fraction === "number" ? Math.round(fraction * 100) : null;
  const loading = job.status === "extracting" && job.phase === "loading";
  return (
    <div className="import-progress extraction-progress" role="status" aria-live="polite">
      <div className="import-progress-heading">
        {pending ? <Spinner /> : <Icon name={job.status === "completed" ? "checkCircle" : "warning"} size={18} />}
        <strong>{extractionLabel(job, t)}</strong>
        {pending && progress !== null && <span>{progress}%</span>}
      </div>
      {pending && <progress max="100" value={progress ?? undefined} aria-label={extractionLabel(job, t)} />}
      {job.error && <p role="alert">{job.error}</p>}
      {pending && loading && !job.connectionLost && <p>{t.extractLoading}</p>}
      {pending && <p>{job.connectionLost ? t.importReconnectHint : t.extractHint}</p>}
      {["failed", "not_started"].includes(job.status) && onRetry && (
        <button className="button button-secondary" onClick={onRetry}>{t.extractRetry}</button>
      )}
    </div>
  );
}
