import { memo, useDeferredValue, useEffect, useMemo, useState } from "react";
import { formatDuration, getSlides, slideUrl, videoTitle } from "../lib/api";
import { useVisible } from "../lib/hooks";
import { Icon, Spinner } from "./UI";

function normalize(value) {
  return value
    .toLocaleLowerCase()
    .normalize("NFD")
    .replace(/\p{Diacritic}/gu, "")
    .replace(/đ/g, "d")
    .replace(/[_-]+/g, " ");
}

const RailCard = memo(function RailCard({ video, active, checked, onSelect, onToggle, t }) {
  const [ref, visible] = useVisible();
  const [thumbnail, setThumbnail] = useState(null);
  useEffect(() => {
    if (!visible || !video.has_slides) return;
    let alive = true;
    getSlides(video.video_id)
      .then((slides) => alive && slides.length && setThumbnail(slideUrl(video.video_id, slides[0])))
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [video.video_id, video.has_slides, visible]);
  const ready = video.has_transcript;
  const duration = formatDuration(video.duration_seconds);
  return (
    <div
      ref={ref}
      className={`rail-card ${active ? "active" : ""} ${checked ? "checked" : ""} ${ready ? "is-ready" : "is-waiting"}`}
    >
      <button
        className="rail-card-main"
        onClick={() => onSelect(video)}
        aria-current={active ? "true" : undefined}
        title={video.filename}
      >
      <span className="rail-thumb">
        {thumbnail ? <img src={thumbnail} alt="" loading="lazy" decoding="async" /> : <Icon name="video" size={26} />}
        {duration && <span className="rail-duration">{duration}</span>}
      </span>
      <span className="rail-card-body">
        <span className="rail-card-title">{videoTitle(video.filename)}</span>
        <span className="rail-card-status">{ready ? t.ready : t.transcriptMissing}</span>
      </span>
      </button>
      <label className="rail-check" title={ready ? t.askAcrossHint : t.transcriptMissing}>
        <input
          type="checkbox"
          checked={checked}
          disabled={!ready}
          onChange={() => onToggle(video.video_id)}
          aria-label={`${t.askAcrossHint}: ${videoTitle(video.filename)}`}
        />
      </label>
    </div>
  );
});

export default function VideoRail({
  videos,
  loading,
  error,
  selectedId,
  onSelect,
  checkedIds = [],
  onToggle,
  onClearChecked,
  onUpload,
  onRefresh,
  addingVideo,
  lang,
  onLang,
  theme,
  onTheme,
  onHelp,
  onClose,
  t,
}) {
  const [newestFirst, setNewestFirst] = useState(true);
  const [showAll, setShowAll] = useState(true);
  const [search, setSearch] = useState("");
  const query = useDeferredValue(search);
  const list = useMemo(() => {
    const needle = normalize(query.trim());
    return (videos || [])
      .filter((video) => (showAll || video.has_transcript) && normalize(video.filename).includes(needle))
      .sort((a, b) => {
        const difference = (Date.parse(b.uploaded_at) || 0) - (Date.parse(a.uploaded_at) || 0);
        return newestFirst ? difference : -difference;
      });
  }, [videos, query, showAll, newestFirst]);
  const readyCount = (videos || []).filter((video) => video.has_transcript).length;

  return (
    <div className="rail">
      <div className="rail-brand">
        <span className="rail-logo">
          <Icon name="play" size={16} />
        </span>
        <span className="rail-name">
          <b>Multimodal Lecture RAG</b>
          <small>{t.railSubtitle}</small>
        </span>
        {onClose && (
          <button className="icon-button" onClick={onClose} aria-label={t.close}>
            <Icon name="close" />
          </button>
        )}
      </div>
      <button className="button button-primary rail-upload" onClick={onUpload}>
        {addingVideo ? <Spinner /> : <Icon name="plus" size={17} />}
        <span>{addingVideo ? t.viewImport : t.upload}</span>
      </button>
      <div className="rail-search">
        <Icon name="search" size={16} />
        <input
          type="search"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          placeholder={t.search}
          aria-label={t.search}
        />
      </div>
      <div className="rail-tools">
        <button
          className="rail-sort"
          onClick={() => setNewestFirst((value) => !value)}
          title={newestFirst ? t.sortNewest : t.sortOldest}
          aria-label={newestFirst ? t.sortNewest : t.sortOldest}
        >
          <span className={newestFirst ? "on" : ""}>↓</span>
          <span className={newestFirst ? "" : "on"}>↑</span>
          <small>{newestFirst ? t.sortNewest : t.sortOldest}</small>
        </button>
        <label className="rail-all">
          <input type="checkbox" checked={showAll} onChange={(event) => setShowAll(event.target.checked)} />
          <span>{showAll ? `${t.all} (${videos?.length ?? 0})` : `${t.ready} (${readyCount})`}</span>
        </label>
      </div>
      {checkedIds.length > 0 && (
        <div className="rail-scope">
          <span>
            {t.askingAcross} {checkedIds.length} {t.videosWord}
          </span>
          <button onClick={onClearChecked}>{t.clearChoice}</button>
        </div>
      )}
      <div className="rail-list" role="list">
        {loading && !videos ? (
          <div className="rail-empty">
            <Spinner label={t.loading} />
          </div>
        ) : error ? (
          <div className="rail-empty">
            <p>{t.libraryError}</p>
            <button className="button button-secondary" onClick={onRefresh}>
              {t.retry}
            </button>
          </div>
        ) : list.length === 0 ? (
          <div className="rail-empty">
            <p>{t.noResults}</p>
          </div>
        ) : (
          list.map((video) => (
            <RailCard
              key={video.video_id}
              video={video}
              active={video.video_id === selectedId}
              checked={checkedIds.length ? checkedIds.includes(video.video_id) : video.video_id === selectedId}
              onSelect={onSelect}
              onToggle={onToggle}
              t={t}
            />
          ))
        )}
      </div>
      <div className="rail-footer">
        <button className="language-button" onClick={onLang} title={lang === "vi" ? "Switch to English" : "Chuyển sang Tiếng Việt"}>
          <Icon name="globe" size={16} />
          <span>{lang.toUpperCase()}</span>
        </button>
        <button className="icon-button" onClick={onTheme} aria-label={t.switchTheme} title={t.switchTheme}>
          <Icon name={theme === "dark" ? "sun" : "moon"} size={18} />
        </button>
        <button className="icon-button" onClick={onHelp} aria-label={t.howItWorks} title={t.howItWorks}>
          <Icon name="help" size={18} />
        </button>
      </div>
    </div>
  );
}
