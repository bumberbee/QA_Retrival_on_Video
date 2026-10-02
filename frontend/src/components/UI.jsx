import { useEffect, useId, useRef, useState } from "react";

const paths = {
  play: "m9 5 11 7-11 7V5Z",
  library: "M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6z",
  sparkles:
    "m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3ZM20 2v4m-2-2h4",
  plus: "M12 5v14M5 12h14",
  link: "M10 13a5 5 0 0 0 7 0l3-3a5 5 0 0 0-7-7l-2 2M14 11a5 5 0 0 0-7 0l-3 3a5 5 0 0 0 7 7l2-2",
  close: "m6 6 12 12M6 18 18 6",
  search: "m21 21-4.6-4.6M19 11a8 8 0 1 1-16 0 8 8 0 0 1 16 0",
  arrow: "M5 12h14m-6-6 6 6-6 6",
  back: "M19 12H5m6 6-6-6 6-6",
  chevron: "m9 5 7 7-7 7",
  upload: "M12 16V3m-5 5 5-5 5 5M4 16v4a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-4",
  clock: "M12 7v5l3 2M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0",
  check: "m5 12 4 4L19 6",
  checkCircle: "m8 12 3 3 5-6M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0",
  video:
    "M4 5h12a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2Zm14 5 4-3v10l-4-3",
  list: "M9 6h12M9 12h12M9 18h12M3 6h.01M3 12h.01M3 18h.01",
  sun: "M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5M16 12a4 4 0 1 1-8 0 4 4 0 0 1 8 0",
  moon: "M21 12.8A9 9 0 0 1 11.2 3 9 9 0 1 0 21 12.8Z",
  globe:
    "M2 12h20M12 2c5 5 5 15 0 20-5-5-5-15 0-20Zm10 10a10 10 0 1 1-20 0 10 10 0 0 1 20 0",
  menu: "M4 6h16M4 12h16M4 18h16",
  book: "M12 5c-3-2-6-2-10-1v15c4-1 7-1 10 1 3-2 6-2 10-1V4c-4-1-7-1-10 1Zm0 0v15",
  help: "M9.1 9a3 3 0 0 1 5.8 1c0 2-3 2-3 4m.1 3h.01M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0",
  transcript: "M8 3h8l4 4v14H4V3h4Zm7 0v5h5M8 12h8m-8 4h5",
  slides: "M3 3h18v14H3zM8 21l4-4 4 4M8 8h8m-8 4h5",
  warning: "m12 3 10 18H2L12 3Zm0 6v5m0 3h.01",
  send: "M12 19V5m-6 6 6-6 6 6",
  refresh:
    "M20 7v5h-5M4 17v-5h5M5.2 7a8 8 0 0 1 13.2-2L20 7M4 17l1.6 2a8 8 0 0 0 13.2-2",
  expand: "M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5",
  message: "M21 11.5a8.5 8.5 0 0 1-8.5 8.5H3l1.8-4.2A8.5 8.5 0 1 1 21 11.5Z",
  external: "M14 3h7v7m0-7L10 14M10 3H3v18h18v-7",
  leaf: "M20 3C8 1 2 9 6 15s15 3 14-12ZM4 21 15 10",
};

export function Icon({ name, size = 20, className = "", ...props }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className={className}
      {...props}
    >
      <path d={paths[name] || paths.play} />
    </svg>
  );
}

export function Logo({ onClick }) {
  return (
    <button className="brand" onClick={onClick} aria-label="Video Q&A">
      <span className="brand-mark">
        <Icon name="play" size={21} />
        <span className="brand-spark">✦</span>
      </span>
      <span>
        Video<span className="brand-qa"> Q&A</span>
        <small>WATCH. ASK. UNDERSTAND.</small>
      </span>
    </button>
  );
}

export function Spinner({ label }) {
  return (
    <span className="spinner" role={label ? "status" : undefined}>
      {label && <span className="sr-only">{label}</span>}
    </span>
  );
}

// Native dialogs provide focus containment, Escape handling, and background inertness.
export function Dialog({
  open,
  onClose,
  title,
  children,
  className = "",
  showClose = true,
  closeLabel = "Close",
  onKeyDown,
}) {
  const ref = useRef(null);
  const titleId = useId();
  useEffect(() => {
    const dialog = ref.current;
    if (!open) return;
    const previous = document.activeElement;
    dialog.showModal();
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      dialog.close();
      document.body.style.overflow = previousOverflow;
      if (previous instanceof HTMLElement && previous.isConnected)
        previous.focus({ preventScroll: true });
    };
  }, [open]);

  return (
    <dialog
      ref={ref}
      className={`dialog ${className}`}
      aria-labelledby={titleId}
      onKeyDown={(event) => {
        if (event.key === "Tab") {
          const focusable = [
            ...event.currentTarget.querySelectorAll(
              'button:not(:disabled), a[href], input:not(:disabled):not([type="hidden"]), select:not(:disabled), textarea:not(:disabled), [tabindex]:not([tabindex="-1"])',
            ),
          ].filter((element) => element.getClientRects().length > 0);
          const first = focusable[0];
          const last = focusable.at(-1);
          const active = document.activeElement;
          if (!first) event.preventDefault();
          else if (
            event.shiftKey &&
            (active === first || !event.currentTarget.contains(active))
          ) {
            event.preventDefault();
            last.focus();
          } else if (
            !event.shiftKey &&
            (active === last || !event.currentTarget.contains(active))
          ) {
            event.preventDefault();
            first.focus();
          }
        }
        onKeyDown?.(event);
      }}
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
      onClick={(event) => {
        if (event.target === event.currentTarget) {
          const rect = event.currentTarget.getBoundingClientRect();
          if (
            event.clientX < rect.left ||
            event.clientX > rect.right ||
            event.clientY < rect.top ||
            event.clientY > rect.bottom
          )
            onClose();
        }
      }}
    >
      <h2 id={titleId} className="sr-only">
        {title}
      </h2>
      {showClose && (
        <button
          className="icon-button dialog-close"
          onClick={onClose}
          aria-label={closeLabel}
        >
          <Icon name="close" />
        </button>
      )}
      {open && children}
    </dialog>
  );
}

export function UploadPanel({
  onUpload,
  uploading,
  filename,
  error,
  t,
}) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const dragDepth = useRef(0);
  return (
    <>
      <div
        className={`upload-dropzone ${dragging ? "is-dragging" : ""}`}
        aria-busy={uploading}
        onDragEnter={(event) => {
          event.preventDefault();
          dragDepth.current += 1;
          setDragging(true);
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={(event) => {
          event.preventDefault();
          dragDepth.current -= 1;
          if (dragDepth.current <= 0) setDragging(false);
        }}
        onDrop={(event) => {
          event.preventDefault();
          dragDepth.current = 0;
          setDragging(false);
          if (!uploading) onUpload(event.dataTransfer.files[0]);
        }}
      >
        <span className="upload-cloud">
          {uploading ? <Spinner /> : <Icon name="upload" size={32} />}
        </span>
        <h3>{uploading ? t.uploading : t.dropTitle}</h3>
        <p className="upload-filename">
          {uploading ? filename : t.dropDescription}
        </p>
        {!uploading && (
          <button
            className="button button-primary"
            onClick={() => inputRef.current?.click()}
          >
            <Icon name="plus" size={17} />
            {t.chooseFile}
          </button>
        )}
        <input
          ref={inputRef}
          type="file"
          accept=".mp4,.webm,.mov,.mkv"
          hidden
          disabled={uploading}
          onChange={(event) => {
            onUpload(event.target.files[0]);
            event.target.value = "";
          }}
        />
        <small>{t.formats}</small>
      </div>
      {error && (
        <p className="inline-error" role="alert">
          <Icon name="warning" size={18} />
          {error}
        </p>
      )}
      <p className="upload-note" role={uploading ? "status" : undefined}>
        <Icon name={uploading ? "clock" : "sparkles"} size={17} />
        {uploading ? t.uploadInProgress : t.uploadNote}
      </p>
    </>
  );
}

export function HelpDialog({ open, onClose, t }) {
  return (
    <Dialog
      open={open}
      onClose={onClose}
      title={t.helpTitle}
      closeLabel={t.close}
      className="help-dialog"
    >
      <span className="section-icon">
        <Icon name="book" size={25} />
      </span>
      <h2>{t.helpTitle}</h2>
      <p className="dialog-description">{t.helpDescription}</p>
      <ol className="help-steps">
        {t.helpSteps.map(([title, description], index) => (
          <li key={index}>
            <span className="step-number">0{index + 1}</span>
            <div>
              <h3>{title}</h3>
              <p>{description}</p>
            </div>
          </li>
        ))}
      </ol>
      <button className="button button-primary" onClick={onClose}>
        {t.gotIt}
        <Icon name="arrow" size={17} />
      </button>
    </Dialog>
  );
}

export function EmptyState({
  icon = "video",
  title,
  description,
  children,
  compact = false,
}) {
  return (
    <div className={`empty-state ${compact ? "empty-state-compact" : ""}`}>
      <span className="empty-icon">
        <Icon name={icon} size={27} />
      </span>
      <h3>{title}</h3>
      {description && <p>{description}</p>}
      {children}
    </div>
  );
}
