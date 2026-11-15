import { useEffect, useRef, useState } from "react";
import { invalidateSlides, request } from "../../shared/lib/api";

export function useExtraction(videos, onComplete) {
  const [enabled, setEnabled] = useState(null);
  const [statuses, setStatuses] = useState({});
  const [revision, setRevision] = useState(0);
  const current = useRef({ videos, onComplete });
  useEffect(() => { current.current = { videos, onComplete }; }, [videos, onComplete]);
  const previous = useRef({});
  const ids = (videos || []).map((video) => video.video_id).sort().join(",");

  useEffect(() => {
    const controller = new AbortController();
    let timer;
    async function check() {
      try {
        const health = await request("/health", { signal: controller.signal });
        if (!controller.signal.aborted) setEnabled(health.extraction_enabled === true);
      } catch {
        if (!controller.signal.aborted) timer = setTimeout(check, 5000);
      }
    }
    check();
    return () => { controller.abort(); clearTimeout(timer); };
  }, []);

  useEffect(() => {
    if (!enabled || !ids) return;
    const controller = new AbortController();
    let timer;
    let remaining = ids.split(",");
    async function poll() {
      const results = await Promise.allSettled(remaining.map((id) =>
        request(`/videos/${encodeURIComponent(id)}/extract`, { signal: controller.signal }),
      ));
      if (controller.signal.aborted) return;
      const updates = {};
      const next = [];
      results.forEach((result, index) => {
        const id = remaining[index];
        if (result.status === "fulfilled") {
          const job = result.value;
          updates[id] = job;
          if (job.status === "completed") {
            const video = current.current.videos?.find((item) => item.video_id === id);
            if (previous.current[id] !== "completed" && (previous.current[id] || !video?.has_transcript)) {
              invalidateSlides(id);
              current.current.onComplete(id);
            }
          } else if (job.status !== "failed") next.push(id);
          previous.current[id] = job.status;
        } else if (result.reason.status === 503) {
          setEnabled(false);
        } else if (result.reason.status === 404) {
          updates[id] = { status: "not_started" };
          next.push(id);
        } else {
          updates[id] = { connectionLost: true };
          next.push(id);
        }
      });
      setStatuses((old) => ({ ...old, ...Object.fromEntries(Object.entries(updates).map(([id, job]) =>
        [id, job.connectionLost ? { ...old[id], ...job } : job],
      )) }));
      remaining = next;
      if (remaining.length) timer = setTimeout(poll, 2000);
    }
    poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [enabled, ids, revision]);

  async function start(videoId) {
    try {
      const job = await request(`/videos/${encodeURIComponent(videoId)}/extract`, { method: "POST" });
      previous.current[videoId] = job.status;
      setStatuses((old) => ({ ...old, [videoId]: job }));
      setRevision((value) => value + 1);
    } catch (error) {
      setStatuses((old) => ({ ...old, [videoId]: { status: "failed", error: error.message } }));
    }
  }

  function status(videoId) {
    return enabled === false ? null : statuses[videoId] || { status: "checking" };
  }
  return { status, start };
}
