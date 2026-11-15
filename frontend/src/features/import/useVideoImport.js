import { useEffect, useRef, useState } from "react";
import { request } from "../../shared/lib/api";

const IMPORT_KEY = "video-import-job";
const IMPORT_URL_KEY = "video-import-url";

function rememberImport(importId, sourceUrl) {
  try {
    if (importId) {
      sessionStorage.setItem(IMPORT_KEY, importId);
      sessionStorage.setItem(IMPORT_URL_KEY, sourceUrl);
    } else {
      sessionStorage.removeItem(IMPORT_KEY);
      sessionStorage.removeItem(IMPORT_URL_KEY);
    }
  } catch {}
}

function restoreSourceUrl() {
  try {
    return sessionStorage.getItem(IMPORT_URL_KEY) || "";
  } catch {
    return "";
  }
}

function restoreImport() {
  try {
    const importId = sessionStorage.getItem(IMPORT_KEY);
    return importId ? { import_id: importId, status: "queued", progress: null } : null;
  } catch {
    return null;
  }
}

export function useVideoImport({ onComplete, t }) {
  const [job, setJob] = useState(restoreImport);
  const [sourceUrl, setSourceUrl] = useState(restoreSourceUrl);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [connectionLost, setConnectionLost] = useState(false);
  const startRequest = useRef(null);
  const completeRef = useRef(onComplete);
  const stringsRef = useRef(t);
  const pending = job?.status === "queued" || job?.status === "downloading";
  const importId = job?.import_id;

  useEffect(() => {
    completeRef.current = onComplete;
    stringsRef.current = t;
  }, [onComplete, t]);

  useEffect(() => () => startRequest.current?.abort(), []);

  useEffect(() => {
    if (!importId || !pending) return;
    const controller = new AbortController();
    let timer;

    async function poll() {
      try {
        const next = await request(`/videos/import/${encodeURIComponent(importId)}`, {
          signal: controller.signal,
        });
        if (controller.signal.aborted) return;
        setJob(next);
        setConnectionLost(false);
        if (next.status === "completed" || next.status === "failed") {
          rememberImport(null);
          if (next.status === "completed" && next.video) completeRef.current(next.video);
          return;
        }
        timer = setTimeout(poll, 1500);
      } catch (failure) {
        if (controller.signal.aborted) return;
        if (failure.status === 404) {
          rememberImport(null);
          setJob({ import_id: importId, status: "failed", error: stringsRef.current.importExpired });
          setConnectionLost(false);
          return;
        }
        setConnectionLost(true);
        timer = setTimeout(poll, 5000);
      }
    }

    poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [importId, pending]);

  async function startImport(url) {
    if (startRequest.current || pending) return;
    const controller = new AbortController();
    startRequest.current = controller;
    setSubmitting(true);
    setError("");
    setJob(null);
    setSourceUrl(url);
    setConnectionLost(false);
    try {
      const next = await request("/videos/import", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
        signal: controller.signal,
      });
      if (controller.signal.aborted) return;
      setJob(next);
      const finished = next.status === "completed" || next.status === "failed";
      rememberImport(finished ? null : next.import_id, url);
      if (next.status === "completed" && next.video) completeRef.current(next.video);
    } catch (failure) {
      if (!controller.signal.aborted)
        setError(failure instanceof TypeError ? stringsRef.current.importFailed : failure.message);
    } finally {
      startRequest.current = null;
      if (!controller.signal.aborted) setSubmitting(false);
    }
  }

  function clearImport() {
    if (pending || startRequest.current) return;
    setJob(null);
    setError("");
    setConnectionLost(false);
    rememberImport(null);
  }

  return { job, sourceUrl, setSourceUrl, submitting, pending, error, connectionLost, startImport, clearImport };
}
