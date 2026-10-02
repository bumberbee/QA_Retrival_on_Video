export const API_BASE = (
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8001/api"
).replace(/\/$/, "");

export async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail = body?.detail;
    const error = new Error(
      typeof detail === "string"
        ? detail
        : `${response.status} ${response.statusText}`,
    );
    error.status = response.status;
    throw error;
  }
  return response.json();
}

const slideCache = new Map();

// Library previews and the player share one metadata request per video.
export function getSlides(videoId) {
  if (!slideCache.has(videoId)) {
    const result = request(`/videos/${encodeURIComponent(videoId)}/slides`)
      .then((items) => {
        if (!Array.isArray(items)) throw new Error("Invalid slides response");
        return items;
      })
      .catch((error) => {
        slideCache.delete(videoId);
        throw error;
      });
    slideCache.set(videoId, result);
  }
  return slideCache.get(videoId);
}

export function slideUrl(videoId, name) {
  // Names may include a folder (states/state_00012.jpg); keep the slash.
  const path = name.split("/").map(encodeURIComponent).join("/");
  return `${API_BASE}/videos/${encodeURIComponent(videoId)}/slides/${path}`;
}

// Alternative slide segmentations for evaluation; null when a video has none.
export function getSlideVariants(videoId, options) {
  return request(
    `/videos/${encodeURIComponent(videoId)}/slide-variants`,
    options,
  ).catch((error) => {
    if (error.status === 404) return null;
    throw error;
  });
}

export function getSlideReviews(videoId, options) {
  return request(`/videos/${encodeURIComponent(videoId)}/slide-reviews`, options);
}

export function saveSlideReview(videoId, variant, slideKey, review) {
  return request(
    `/videos/${encodeURIComponent(videoId)}/slide-reviews/${encodeURIComponent(variant)}/${encodeURIComponent(slideKey)}`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(review),
    },
  );
}

export function formatDuration(seconds) {
  if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds < 0)
    return null;
  const total = Math.floor(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const rest = String(total % 60).padStart(2, "0");
  return hours
    ? `${hours}:${String(minutes).padStart(2, "0")}:${rest}`
    : `${minutes}:${rest}`;
}

export function formatDate(value, lang) {
  const date = new Date(value);
  if (!value || Number.isNaN(date.getTime())) return "";
  return date.toLocaleDateString(lang === "vi" ? "vi-VN" : "en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

export function videoTitle(filename) {
  return filename.replace(/\.(mp4|webm|mov|mkv)$/i, "").replace(/[_]+/g, " ");
}

