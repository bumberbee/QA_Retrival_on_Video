import { useId, useRef, useState } from "react";
import { Dialog, Icon, Spinner, UploadPanel } from "../../shared/components/UI";
import { videoTitle } from "../../shared/lib/api";
import "./import-dialog.css";

const PLATFORMS = [
  { name: "YouTube", domains: ["youtube.com", "youtu.be"] },
  { name: "Facebook", domains: ["facebook.com", "fb.watch"] },
  { name: "TikTok", domains: ["tiktok.com"] },
];

function identifyPlatform(value) {
  try {
    const url = new URL(value.trim());
    if (!["https:", "http:"].includes(url.protocol) || url.username || url.password || url.port || /[\s\\]/.test(value.trim()))
      return null;
    return PLATFORMS.find((platform) => platform.domains.some((domain) =>
      url.hostname === domain || url.hostname.endsWith(`.${domain}`),
    ));
  } catch {
    return null;
  }
}

export default function ImportDialog({
  open, onClose, mode, onModeChange, onUpload, uploading, filename, error,
  videoImport, onOpenVideo, t,
}) {
  const { sourceUrl: url, setSourceUrl: setUrl } = videoImport;
  const [validationError, setValidationError] = useState("");
  const inputRef = useRef(null);
  const tabRefs = useRef([]);
  const fieldId = useId();
  const { job, pending, submitting, connectionLost, startImport, clearImport } = videoImport;
  const busy = uploading || pending || submitting;
  const platform = identifyPlatform(url);
  const completed = job?.status === "completed";
  const failed = job?.status === "failed";
  const linkError = validationError || videoImport.error || (failed ? job.error || t.importFailed : "");
  const progress = typeof job?.progress === "number" ? Math.round(Math.max(0, Math.min(100, job.progress))) : null;
  const selectedMode = pending || submitting || completed || failed ? "link" : mode;

  function changeMode(nextMode) {
    if (busy) return;
    clearImport();
    setValidationError("");
    onModeChange(nextMode);
  }

  function submit(event) {
    event.preventDefault();
    if (busy) return;
    if (!platform) {
      setValidationError(t.invalidLink);
      inputRef.current?.focus();
      return;
    }
    setValidationError("");
    startImport(url.trim());
  }

  return (
    <Dialog open={open} onClose={onClose} title={t.uploadTitle} closeLabel={t.close} className="upload-dialog import-dialog">
      <span className="section-icon"><Icon name="video" size={25} /></span>
      <h2>{t.uploadTitle}</h2>
      <p className="dialog-description">{t.uploadDescription}</p>
      <div className="import-tabs" role="tablist" aria-label={t.importMethod}>
        {[["link", "link", t.importLink], ["file", "upload", t.uploadFile]].map(([key, icon, label], index) => (
          <button
            key={key}
            ref={(element) => { tabRefs.current[index] = element; }}
            id={`${fieldId}-${key}-tab`}
            role="tab"
            aria-selected={selectedMode === key}
            aria-controls={`${fieldId}-${key}-panel`}
            tabIndex={selectedMode === key ? 0 : -1}
            disabled={busy}
            onClick={() => changeMode(key)}
            onKeyDown={(event) => {
              if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
              event.preventDefault();
              const nextIndex = event.key === "Home" ? 0 : event.key === "End" ? 1 : 1 - index;
              changeMode(nextIndex === 0 ? "link" : "file");
              tabRefs.current[nextIndex]?.focus();
            }}
          >
            <Icon name={icon} size={17} />{label}
          </button>
        ))}
      </div>
      <div id={`${fieldId}-link-panel`} role="tabpanel" aria-labelledby={`${fieldId}-link-tab`} hidden={selectedMode !== "link"}>
        {completed && job.video ? (
          <div className="import-success" role="status">
            <span className="import-success-icon"><Icon name="checkCircle" size={32} /></span>
            <h3>{t.importComplete}</h3>
            <p className="import-video-title">{videoTitle(job.video.filename)}</p>
            <button className="button button-primary" onClick={() => onOpenVideo(job.video)}>
              <Icon name="play" size={17} />{t.openVideo}
            </button>
            <button className="button button-text" onClick={() => { clearImport(); setUrl(""); }}>
              {t.importAnother}
            </button>
          </div>
        ) : (
          <form onSubmit={submit} noValidate>
            <label className="import-url-label" htmlFor={fieldId}>{t.videoLink}</label>
            <div className={`import-url-field ${linkError ? "has-error" : ""}`}>
              <Icon name="link" size={19} />
              <input
                ref={inputRef}
                id={fieldId}
                type="url"
                inputMode="url"
                autoComplete="off"
                spellCheck={false}
                maxLength={2048}
                placeholder="https://www.youtube.com/watch?v=…"
                value={url}
                disabled={pending || submitting}
                aria-invalid={Boolean(linkError)}
                aria-describedby={`${fieldId}-hint${linkError ? ` ${fieldId}-error` : ""}`}
                onChange={(event) => { setUrl(event.target.value); setValidationError(""); clearImport(); }}
              />
            </div>
            <div className="import-platforms" aria-label={t.supportedPlatforms}>
              {PLATFORMS.map((item) => (
                <span key={item.name} className={platform?.name === item.name ? "detected" : ""}>
                  <span className={`platform-dot platform-${item.name.toLowerCase()}`} />
                  {item.name}
                  {platform?.name === item.name && <Icon name="check" size={12} />}
                </span>
              ))}
            </div>
            <p className="import-hint" id={`${fieldId}-hint`}>{t.linkHint}</p>
            {linkError && <p className="inline-error" role="alert" id={`${fieldId}-error`}><Icon name="warning" size={18} />{linkError}</p>}
            {pending || submitting ? (
              <div className="import-progress" role="status" aria-live="polite">
                <div className="import-progress-heading">
                  <Spinner />
                  <strong>{connectionLost ? t.importReconnecting : job?.status === "downloading" ? t.importing : t.importQueued}</strong>
                  {progress !== null && !connectionLost && <span>{progress}%</span>}
                </div>
                <progress max="100" value={progress ?? undefined} aria-label={t.importing} />
                <p>{connectionLost ? t.importReconnectHint : t.importInBackground}</p>
              </div>
            ) : (
              <button className="button button-primary import-submit" type="submit" disabled={!url.trim()}>
                <Icon name="plus" size={17} />{failed ? t.retryImport : t.importAction}
              </button>
            )}
            <p className="upload-note"><Icon name="sparkles" size={17} />{t.uploadNote}</p>
          </form>
        )}
      </div>
      <div id={`${fieldId}-file-panel`} role="tabpanel" aria-labelledby={`${fieldId}-file-tab`} hidden={selectedMode !== "file"}>
        <UploadPanel onUpload={onUpload} uploading={uploading} filename={filename} error={error} t={t} />
      </div>
    </Dialog>
  );
}
