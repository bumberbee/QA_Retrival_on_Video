import { memo, useDeferredValue, useEffect, useMemo, useState } from "react";
import {
  formatDate,
  formatDuration,
  getSlides,
  slideUrl,
  videoTitle,
} from "../lib/api";
import { usePreference, useVisible } from "../lib/hooks";
import { EmptyState, Icon, Spinner } from "./UI";

const VideoCard = memo(function VideoCard({ video, index, onSelect, lang, t }) {
  const [ref, visible] = useVisible();
  const [thumbnail, setThumbnail] = useState(null);
  const [imageFailed, setImageFailed] = useState(false);
  useEffect(() => {
    if (!visible || !video.has_slides) return;
    let active = true;
    getSlides(video.video_id)
      .then((slides) => {
        if (active && slides.length)
          setThumbnail(slideUrl(video.video_id, slides[0]));
      })
      .catch(() => {});
    return () => {
      active = false;
    };
  }, [video.video_id, video.has_slides, visible]);

  const duration = formatDuration(video.duration_seconds);
  const sourceLabel = {
    youtube: "YouTube",
    facebook: "Facebook",
    tiktok: "TikTok",
  }[video.source_platform?.toLowerCase()];
  const fileType = video.filename.match(/\.([a-z0-9]{2,5})$/i)?.[1];
  return (
    <button
      ref={ref}
      className={`video-card card-tone-${index % 4}`}
      onClick={() => onSelect(video)}
      aria-label={`${t.openVideo}: ${video.filename}`}
    >
      <span className="video-thumbnail">
        <span className="thumbnail-pattern" />
        <span className="thumbnail-video-icon">
          <Icon name="video" size={40} />
        </span>
        {thumbnail && !imageFailed && (
          <img
            src={thumbnail}
            alt=""
            loading="lazy"
            decoding="async"
            onError={() => setImageFailed(true)}
          />
        )}
        <span className="thumbnail-play">
          <Icon name="play" size={24} />
        </span>
        <span className="thumbnail-type">
          {sourceLabel || fileType?.toUpperCase() || t.video}
        </span>
        {duration && <span className="duration-label">{duration}</span>}
      </span>
      <span className="video-card-body">
        <span className="video-card-title" title={video.filename}>
          {videoTitle(video.filename)}
        </span>
        <span className="video-card-date">
          {formatDate(video.uploaded_at, lang) || t.video}
        </span>
        <span className="video-card-bottom">
          <span
            className={`status-badge ${video.has_transcript ? "status-ready" : "status-waiting"}`}
          >
            <Icon
              name={video.has_transcript ? "sparkles" : "clock"}
              size={13}
            />
            {video.has_transcript ? t.ready : t.transcriptMissing}
          </span>
          <Icon name="arrow" className="video-card-arrow" size={18} />
        </span>
      </span>
    </button>
  );
});

function normalizeSearch(value) {
  return value
    .toLocaleLowerCase()
    .normalize("NFD")
    .replace(/\p{Diacritic}/gu, "")
    .replace(/đ/g, "d")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ");
}

export default function Library({
  videoList,
  loading,
  error,
  search,
  filter,
  onFilter,
  onReset,
  onSelect,
  onUpload,
  onHelp,
  onRefresh,
  lang,
  t,
}) {
  const [view, setView] = usePreference("library-view", "grid", [
    "grid",
    "list",
  ]);
  const [sort, setSort] = useState("newest");
  const [pagination, setPagination] = useState({ key: "", count: 12 });
  const deferredSearch = useDeferredValue(search);
  const filtered = useMemo(() => {
    const query = normalizeSearch(deferredSearch.trim());
    return (videoList || [])
      .filter((item) => {
        const matchesFilter =
          filter === "all" ||
          (filter === "ready" ? item.has_transcript : !item.has_transcript);
        return matchesFilter && normalizeSearch(item.filename).includes(query);
      })
      .sort((a, b) => {
        if (sort === "name")
          return a.filename.localeCompare(b.filename, lang, { numeric: true });
        const difference =
          (Date.parse(b.uploaded_at) || 0) - (Date.parse(a.uploaded_at) || 0);
        return sort === "oldest" ? -difference : difference;
      });
  }, [videoList, deferredSearch, filter, sort, lang]);
  const pageKey = `${deferredSearch}|${filter}|${sort}`;
  const visibleCount = pagination.key === pageKey ? pagination.count : 12;
  return (
    <div className="library-page page-enter">
      <section className="welcome-banner" aria-labelledby="welcome-title">
        <div className="welcome-copy">
          <h1 id="welcome-title">
            {t.heroTitle} <span>{t.heroAccent}</span>
          </h1>
          <p className="welcome-description">{t.heroDescription}</p>
        </div>
        <div className="welcome-actions">
          <button
            className="button button-primary"
            onClick={() => onUpload("link")}
          >
            <Icon name="link" size={18} />
            {t.importLink}
          </button>
          <button
            className="button button-secondary"
            onClick={() => onUpload("file")}
          >
            <Icon name="upload" size={18} />
            {t.uploadFile}
          </button>
        </div>
      </section>

      <section className="library-section" aria-labelledby="library-title">
        <div className="section-heading">
          <div>
            <h2 id="library-title">
              {t.libraryTitle}
              <span className="count-pill">{videoList?.length ?? "—"}</span>
            </h2>
          </div>
          <button
            className="icon-button refresh-button"
            onClick={onRefresh}
            disabled={loading}
            aria-label={t.refresh}
            title={t.refresh}
          >
            {loading ? <Spinner /> : <Icon name="refresh" size={18} />}
          </button>
        </div>
        <div className="library-toolbar">
          <div className="filter-tabs" role="group" aria-label={t.library}>
            {[
              ["all", t.allVideos],
              ["ready", t.aiReady],
              ["pending", t.needsTranscript],
            ].map(([key, label]) => (
              <button
                key={key}
                className={filter === key ? "active" : ""}
                aria-pressed={filter === key}
                onClick={() => onFilter(key)}
              >
                {label}
              </button>
            ))}
          </div>
          <div className="library-view-controls">
            <select
              value={sort}
              onChange={(event) => setSort(event.target.value)}
              aria-label={t.sort}
            >
              <option value="newest">{t.newest}</option>
              <option value="oldest">{t.oldest}</option>
              <option value="name">{t.alphabetical}</option>
            </select>
            <div
              className="view-toggle"
              role="group"
              aria-label={`${t.grid} / ${t.list}`}
            >
              <button
                aria-label={t.grid}
                title={t.grid}
                aria-pressed={view === "grid"}
                className={view === "grid" ? "active" : ""}
                onClick={() => setView("grid")}
              >
                <Icon name="library" size={16} />
              </button>
              <button
                aria-label={t.list}
                title={t.list}
                aria-pressed={view === "list"}
                className={view === "list" ? "active" : ""}
                onClick={() => setView("list")}
              >
                <Icon name="list" size={18} />
              </button>
            </div>
          </div>
        </div>

        {error && (
          <div className="library-error" role="alert">
            <EmptyState
              icon="warning"
              title={t.loadError}
              description={t.loadErrorDescription}
              compact
            >
              <button className="button button-secondary" onClick={onRefresh}>
                <Icon name="refresh" size={16} />
                {t.retry}
              </button>
            </EmptyState>
          </div>
        )}
        {loading && !videoList ? (
          <div className="video-grid" role="status" aria-label={t.loading}>
            {[0, 1, 2].map((item) => (
              <div className="video-skeleton" key={item}>
                <div />
                <span />
                <span />
              </div>
            ))}
          </div>
        ) : !error && videoList?.length === 0 ? (
          <div className="library-empty">
            <span className="empty-icon">
              <Icon name="video" size={26} />
            </span>
            <h3>{t.noVideos}</h3>
            <p>{t.noVideosDescription}</p>
            <ol className="library-steps" aria-label={t.howItWorks}>
              {t.helpSteps.map(([title], index) => (
                <li key={title}>
                  <span className="step-number">{index + 1}</span>
                  <span>{title}</span>
                </li>
              ))}
            </ol>
            <button className="button button-text" onClick={onHelp}>
              {t.howItWorks}
              <Icon name="arrow" size={16} />
            </button>
          </div>
        ) : !error && filtered.length === 0 ? (
          <EmptyState
            icon="search"
            title={t.noResults}
            description={t.noResultsDescription}
          >
            <button className="button button-secondary" onClick={onReset}>
              {t.resetFilters}
            </button>
          </EmptyState>
        ) : (
          !error && (
            <>
              {search.trim() && (
                <p className="search-results" role="status">
                  {filtered.length} {filtered.length === 1 ? t.video : t.videos}{" "}
                  · “{search.trim()}”
                </p>
              )}
              <div
                className={view === "grid" ? "video-grid" : "video-list"}
                aria-busy={search !== deferredSearch}
              >
                {filtered.slice(0, visibleCount).map((video, index) => (
                  <VideoCard
                    key={video.video_id}
                    video={video}
                    index={index}
                    onSelect={onSelect}
                    lang={lang}
                    t={t}
                  />
                ))}
              </div>
              {filtered.length > visibleCount && (
                <div className="load-more">
                  <button
                    className="button button-secondary"
                    onClick={() =>
                      setPagination({ key: pageKey, count: visibleCount + 12 })
                    }
                  >
                    {t.showMore}
                    <span>{filtered.length - visibleCount}</span>
                  </button>
                </div>
              )}
            </>
          )
        )}
      </section>
      <footer className="page-footer">
        <Icon name="leaf" size={16} />
        <span>{t.footer}</span>
        <span className="footer-wordmark">Video Q&A</span>
      </footer>
    </div>
  );
}
