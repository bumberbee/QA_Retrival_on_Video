import { useDeferredValue, useEffect, useMemo, useState } from "react";
import { formatDuration, getSlides, slideUrl, videoTitle } from "../lib/api";
import { useVisible } from "../lib/hooks";
import { EmptyState, Icon, Spinner } from "./UI";

function ConversationCard({ video, onSelect, t }) {
  const [ref, visible] = useVisible();
  const [cover, setCover] = useState(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    if (!visible || !video.has_slides) return;
    let active = true;
    getSlides(video.video_id)
      .then((slides) => {
        if (active && slides[0]) setCover(slideUrl(video.video_id, slides[0]));
      })
      .catch(() => {});
    return () => {
      active = false;
    };
  }, [visible, video.video_id, video.has_slides]);

  return (
    <article ref={ref} className="conversation-card">
      <div className="conversation-card-heading">
        <span className="conversation-cover">
          <Icon name="video" size={26} />
          {cover && !failed && (
            <img
              src={cover}
              alt=""
              loading="lazy"
              decoding="async"
              onError={() => setFailed(true)}
            />
          )}
        </span>
        <div className="conversation-video-details">
          <h3 title={video.filename}>{videoTitle(video.filename)}</h3>
          <p>
            <span className="conversation-ready-dot" />
            {t.transcriptAvailable}
            {video.duration_seconds > 0 && (
              <>
                <span className="meta-dot">·</span>
                <Icon name="clock" size={12} />
                {formatDuration(video.duration_seconds)}
              </>
            )}
          </p>
        </div>
        <button
          className="button button-primary conversation-start"
          onClick={() => onSelect(video, { focusChat: true })}
        >
          <Icon name="message" size={16} />
          {t.startConversation}
          <Icon name="arrow" size={15} />
        </button>
      </div>
      <div className="conversation-prompts">
        <p>{t.startWithQuestion}</p>
        <div>
          {t.aiPromptLabels.map((label, index) => (
            <button
              key={label}
              onClick={() =>
                onSelect(video, {
                  focusChat: true,
                  prompt: t.suggestionQuestions[index],
                })
              }
            >
              <Icon
                name={["transcript", "sparkles", "clock"][index]}
                size={15}
              />
              <span>{label}</span>
              <Icon name="plus" size={13} />
            </button>
          ))}
        </div>
      </div>
    </article>
  );
}

export default function ReadyForAI({
  videoList,
  loading,
  error,
  search,
  onClearSearch,
  onSelect,
  onNavigate,
  onRefresh,
  t,
}) {
  const deferredSearch = useDeferredValue(search);
  const readyVideos = useMemo(
    () => (videoList || []).filter((video) => video.has_transcript),
    [videoList],
  );
  const filtered = useMemo(() => {
    const normalize = (value) =>
      value
        .toLocaleLowerCase()
        .normalize("NFD")
        .replace(/\p{Diacritic}/gu, "")
        .replace(/đ/g, "d")
        .replace(/[_-]+/g, " ")
        .replace(/\s+/g, " ");
    return readyVideos.filter((video) =>
      normalize(video.filename).includes(normalize(deferredSearch.trim())),
    );
  }, [readyVideos, deferredSearch]);
  const pendingCount = (videoList?.length || 0) - readyVideos.length;
  const [pagination, setPagination] = useState({ search: "", count: 10 });
  const visibleCount =
    pagination.search === deferredSearch ? pagination.count : 10;

  return (
    <div className="ai-page page-enter">
      <header className="ai-intro">
        <span className="ai-page-icon">
          <Icon name="sparkles" size={28} />
        </span>
        <div>
          <p className="eyebrow">{t.aiEyebrow}</p>
          <h1>{t.aiPageTitle}</h1>
          <p className="ai-page-description">{t.aiPageDescription}</p>
        </div>
      </header>
      <div className="ai-status-strip">
        <span>
          <Icon name="checkCircle" size={17} />
          <strong>{videoList ? readyVideos.length : "—"}</strong>
          {t.videosReadyToDiscuss}
        </span>
        <span className="ai-status-detail">
          <Icon name="transcript" size={15} />
          {t.answersFromTranscripts}
        </span>
      </div>
      <div className="ai-page-layout">
        <section
          className="conversation-list"
          aria-labelledby="conversation-list-title"
        >
          <div className="section-heading">
            <div>
              <h2 id="conversation-list-title">{t.chooseConversation}</h2>
              <p>{t.chooseConversationHint}</p>
            </div>
            <button
              className="icon-button"
              onClick={onRefresh}
              disabled={loading}
              aria-label={t.refresh}
              title={t.refresh}
            >
              {loading ? <Spinner /> : <Icon name="refresh" size={18} />}
            </button>
          </div>
          {error ? (
            <EmptyState
              icon="warning"
              title={t.loadError}
              description={t.loadErrorDescription}
            >
              <button className="button button-secondary" onClick={onRefresh}>
                {t.retry}
              </button>
            </EmptyState>
          ) : loading && !videoList ? (
            <div className="ai-loading" role="status">
              <Spinner />
              <span>{t.loading}</span>
            </div>
          ) : readyVideos.length === 0 ? (
            <EmptyState
              icon="message"
              title={t.aiEmptyTitle}
              description={t.aiEmptyDescription}
            >
              <button
                className="button button-primary"
                onClick={() => onNavigate("all")}
              >
                {t.browseLibrary}
                <Icon name="arrow" size={16} />
              </button>
            </EmptyState>
          ) : filtered.length === 0 ? (
            <EmptyState
              icon="search"
              title={t.noResults}
              description={t.aiNoResults}
            >
              <button
                className="button button-secondary"
                onClick={onClearSearch}
              >
                {t.clearSearch}
              </button>
            </EmptyState>
          ) : (
            <div
              className="conversation-cards"
              aria-busy={search !== deferredSearch}
            >
              {filtered.slice(0, visibleCount).map((video) => (
                <ConversationCard
                  key={video.video_id}
                  video={video}
                  onSelect={onSelect}
                  t={t}
                />
              ))}
              {filtered.length > visibleCount && (
                <button
                  className="button button-secondary"
                  onClick={() =>
                    setPagination({
                      search: deferredSearch,
                      count: visibleCount + 10,
                    })
                  }
                >
                  {t.showMore}
                </button>
              )}
            </div>
          )}
          {pendingCount > 0 && (
            <div className="ai-waiting-note">
              <span className="ai-waiting-icon">
                <Icon name="transcript" size={20} />
              </span>
              <div>
                <h3>
                  {t.waitingForTranscript.replace(
                    "{count}",
                    String(pendingCount),
                  )}
                </h3>
                <p>{t.waitingForTranscriptHint}</p>
              </div>
              <button
                className="button button-text"
                onClick={() => onNavigate("pending")}
              >
                {t.viewVideos}
                <Icon name="arrow" size={15} />
              </button>
            </div>
          )}
        </section>
        <aside className="ai-guide">
          <span className="ai-guide-icon">
            <Icon name="book" size={22} />
          </span>
          <h2>{t.aiGuideTitle}</h2>
          <p>{t.aiGuideDescription}</p>
          <ol>
            {t.aiGuideSteps.map(([title, description], index) => (
              <li key={title}>
                <span>0{index + 1}</span>
                <div>
                  <h3>{title}</h3>
                  <p>{description}</p>
                </div>
              </li>
            ))}
          </ol>
          <div className="ai-guide-note">
            <Icon name="clock" size={16} />
            <p>{t.aiTimestampTip}</p>
          </div>
        </aside>
      </div>
      <footer className="page-footer">
        <Icon name="sparkles" size={15} />
        <span>{t.chatFootnote}</span>
        <span className="footer-wordmark">Video Q&A</span>
      </footer>
    </div>
  );
}
