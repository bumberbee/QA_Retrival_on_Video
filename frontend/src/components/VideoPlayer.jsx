import { useCallback, useEffect, useId, useRef, useState } from "react";
import { API_BASE } from "../lib/api";
import { usePreference } from "../lib/hooks";
import { Icon } from "./UI";

const SPEEDS = ["0.25", "0.5", "0.75", "1", "1.25", "1.5", "1.75", "2"];

function SkipIcon({ forward, size = 23 }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <g
        transform={forward ? "translate(24 0) scale(-1 1)" : undefined}
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M4 8a9 9 0 1 1-1 7M4 3v5h5" />
      </g>
      <text
        x="12"
        y="16"
        textAnchor="middle"
        fill="currentColor"
        fontFamily="Arial, sans-serif"
        fontSize="8"
        fontWeight="700"
      >
        10
      </text>
    </svg>
  );
}

export default function VideoPlayer({
  video,
  videoRef,
  poster,
  onTimeChange,
  onDurationChange,
  t,
}) {
  const [error, setError] = useState(false);
  const [ready, setReady] = useState(false);
  const [version, setVersion] = useState(0);
  const [speed, setSpeed] = usePreference("playback-speed", "1", SPEEDS);
  const [feedback, setFeedback] = useState(null);
  const [announcement, setAnnouncement] = useState("");
  const feedbackTimer = useRef(null);
  const tap = useRef(null);
  const feedbackSequence = useRef(0);
  const speedId = useId();
  const source = `${API_BASE}/videos/${encodeURIComponent(video.video_id)}/file`;

  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = Number(speed);
  }, [speed, version, videoRef]);

  useEffect(() => {
    const player = videoRef.current;
    if (!player) return;
    function syncMetadata() {
      if (player.readyState >= 1 && Number.isFinite(player.duration)) {
        onDurationChange(player.duration);
        setReady(true);
      }
    }
    // A cached video can load metadata before a lazy view commits. Read its
    // current state as well as listening for future media events.
    player.addEventListener("loadedmetadata", syncMetadata);
    player.addEventListener("durationchange", syncMetadata);
    player.addEventListener("canplay", syncMetadata);
    syncMetadata();
    return () => {
      player.removeEventListener("loadedmetadata", syncMetadata);
      player.removeEventListener("durationchange", syncMetadata);
      player.removeEventListener("canplay", syncMetadata);
    };
  }, [version, videoRef, onDurationChange]);

  const skipBy = useCallback(
    (seconds) => {
      const player = videoRef.current;
      if (
        !player ||
        player.readyState < 1 ||
        !Number.isFinite(player.duration) ||
        player.duration <= 0
      )
        return false;
      const before = player.currentTime;
      const target = Math.max(0, Math.min(player.duration, before + seconds));
      // Assigning currentTime preserves the current playing / paused state.
      player.currentTime = target;
      onTimeChange(Math.floor(target));
      const amount = Math.round(Math.abs(target - before));
      if (amount > 0) {
        const direction = seconds > 0 ? "forward" : "backward";
        setFeedback({
          direction,
          amount,
          sequence: ++feedbackSequence.current,
        });
        setAnnouncement(
          (seconds > 0 ? t.skippedForward : t.skippedBackward).replace(
            "{seconds}",
            String(amount),
          ),
        );
        clearTimeout(feedbackTimer.current);
        feedbackTimer.current = setTimeout(() => setFeedback(null), 700);
      }
      return true;
    },
    [videoRef, onTimeChange, t.skippedForward, t.skippedBackward],
  );

  useEffect(() => {
    function onKeyDown(event) {
      if (
        event.defaultPrevented ||
        event.altKey ||
        event.ctrlKey ||
        event.metaKey ||
        event.shiftKey ||
        document.querySelector("dialog[open]")
      )
        return;
      const target = event.target;
      if (
        target instanceof HTMLElement &&
        (target.isContentEditable ||
          target.closest("input, textarea, select, [role='tablist']"))
      )
        return;
      const seconds = {
        ArrowLeft: -5,
        ArrowRight: 5,
        j: -10,
        J: -10,
        l: 10,
        L: 10,
      }[event.key];
      if (seconds && skipBy(seconds)) event.preventDefault();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [skipBy]);

  useEffect(() => () => clearTimeout(feedbackTimer.current), []);

  function handleTouchTap(event) {
    if (event.pointerType !== "touch" || !ready || error) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const x = (event.clientX - bounds.left) / bounds.width;
    // Leave the native timeline and controls at the bottom to the browser.
    if (event.clientY >= bounds.bottom - 58 || (x >= 0.4 && x <= 0.6)) {
      tap.current = null;
      return;
    }
    const direction = x < 0.4 ? -1 : 1;
    const now = performance.now();
    if (tap.current?.direction === direction && now - tap.current.time < 330) {
      event.preventDefault();
      skipBy(direction * 10);
      tap.current = null;
    } else {
      tap.current = { direction, time: now };
    }
  }

  return (
    <>
      <div className="video-player" onPointerUpCapture={handleTouchTap}>
        <video
          key={version}
          ref={videoRef}
          src={source}
          controls
          playsInline
          preload="metadata"
          poster={poster}
          aria-label={video.filename}
          onTimeUpdate={(event) =>
            onTimeChange(Math.floor(event.currentTarget.currentTime))
          }
          onLoadedMetadata={(event) => {
            const player = event.currentTarget;
            if (Number.isFinite(player.duration))
              onDurationChange(player.duration);
            player.playbackRate = Number(speed);
            setReady(true);
          }}
          onRateChange={(event) => {
            const value = String(event.currentTarget.playbackRate);
            if (SPEEDS.includes(value)) setSpeed(value);
          }}
          onError={() => {
            setError(true);
            setReady(false);
          }}
        />
        {feedback && (
          <div
            key={feedback.sequence}
            className={`seek-feedback seek-${feedback.direction}`}
            aria-hidden="true"
          >
            <Icon
              name={feedback.direction === "forward" ? "chevron" : "back"}
              size={26}
            />
            <span>
              {feedback.amount} {t.secondsShort}
            </span>
          </div>
        )}
        {error && (
          <div className="playback-error" role="alert">
            <Icon name="warning" size={28} />
            <p>{t.playbackError}</p>
            <div>
              <button
                className="button button-secondary"
                onClick={() => {
                  setError(false);
                  setReady(false);
                  setVersion((value) => value + 1);
                }}
              >
                <Icon name="refresh" size={15} />
                {t.reloadVideo}
              </button>
              <a
                className="button button-secondary"
                href={source}
                target="_blank"
                rel="noreferrer"
              >
                <Icon name="external" size={15} />
                {t.openOriginal}
              </a>
            </div>
          </div>
        )}
      </div>
      <div
        className="playback-toolbar"
        role="group"
        aria-label={t.playbackControls}
      >
        <div className="seek-buttons">
          <button
            className="icon-button"
            disabled={!ready || error}
            onClick={() => skipBy(-10)}
            aria-label={t.rewind10}
            aria-keyshortcuts="j"
            title={`${t.rewind10} (J)`}
          >
            <SkipIcon />
          </button>
          <button
            className="icon-button"
            disabled={!ready || error}
            onClick={() => skipBy(10)}
            aria-label={t.forward10}
            aria-keyshortcuts="l"
            title={`${t.forward10} (L)`}
          >
            <SkipIcon forward />
          </button>
          <span className="playback-shortcuts" title={t.seekShortcuts}>
            <kbd>←</kbd>
            <kbd>→</kbd>
            <span>5{t.secondsShort}</span>
            <i /> <kbd>J</kbd>
            <kbd>L</kbd>
            <span>10{t.secondsShort}</span>
          </span>
        </div>
        <div className="playback-speed">
          <label htmlFor={speedId}>{t.speed}</label>
          <select
            id={speedId}
            value={speed}
            onChange={(event) => setSpeed(event.target.value)}
            aria-label={t.playbackSpeed}
          >
            {SPEEDS.map((value) => (
              <option value={value} key={value}>
                {value === "1" ? `1× · ${t.normalSpeed}` : `${value}×`}
              </option>
            ))}
          </select>
        </div>
      </div>
      <span className="sr-only" aria-live="polite" aria-atomic="true">
        {announcement}
      </span>
    </>
  );
}
