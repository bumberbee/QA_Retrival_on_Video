import { Icon, Spinner } from "../../shared/components/UI";
import { extractionLabel } from "./labels";

export default function ExtractionProgress({ job, onRetry, t }) {
  if (!job) return null;
  const pending = ["checking", "queued", "extracting", "importing"].includes(job.status);
  const progress = typeof job.progress === "number" ? Math.round(job.progress * 100) : null;
  return (
    <div className="import-progress extraction-progress" role="status" aria-live="polite">
      <div className="import-progress-heading">
        {pending ? <Spinner /> : <Icon name={job.status === "completed" ? "checkCircle" : "warning"} size={18} />}
        <strong>{extractionLabel(job, t)}</strong>
        {pending && progress !== null && <span>{progress}%</span>}
      </div>
      {pending && <progress max="100" value={progress ?? undefined} aria-label={extractionLabel(job, t)} />}
      {job.error && <p role="alert">{job.error}</p>}
      {pending && <p>{job.connectionLost ? t.importReconnectHint : t.extractHint}</p>}
      {["failed", "not_started"].includes(job.status) && onRetry && (
        <button className="button button-secondary" onClick={onRetry}>{t.extractRetry}</button>
      )}
    </div>
  );
}
