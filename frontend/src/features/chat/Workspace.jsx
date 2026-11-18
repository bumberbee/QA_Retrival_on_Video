import {
  memo,
  useCallback,
  useDeferredValue,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  formatDate,
  formatDuration,
  getSlideReviews,
  getSlideVariants,
  getSlides,
  request,
  saveSlideReview,
  slideUrl,
  streamAnswer,
  videoTitle,
} from "../../shared/lib/api";
import Splitter, { useStoredSize } from "../../shared/components/Splitter";
import { useMediaQuery } from "../../shared/lib/hooks";
import { Dialog, EmptyState, Icon, Spinner } from "../../shared/components/UI";
import VideoPlayer from "../player/VideoPlayer";
import ExtractionProgress from "../extract/ExtractionProgress";
import { extractionLabel } from "../extract/labels";

const TIME = String.raw`\d{1,3}:[0-5]\d(?::[0-5]\d)?`;
// "[02:10–02:38]", "[Slide seg_00005 · 02:10–02:38]", "[10:22 - 15:27]": one link to the start.
const RANGE = new RegExp(
  String.raw`^\[(?:[^\]\[]*?·\s*)?(${TIME})\s*(?:[–—-]\s*(${TIME}))?\]$`,
);
const INLINE = new RegExp(
  String.raw`(\*\*[^*]+\*\*|\[(?:[^\]\[]*?·\s*)?${TIME}\s*(?:[–—-]\s*${TIME})?\]|${TIME})`,
  "g",
);

function toSeconds(timestamp) {
  return timestamp
    .split(":")
    .reduce((total, value) => total * 60 + Number(value), 0);
}

function clock(seconds) {
  const total = Math.floor(seconds);
  return `${String(Math.floor(total / 60)).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
}

function renderInline(text, onSeek, t, prefix) {
  return text
    .split(INLINE)
    .map((part, index) => {
      const key = `${prefix}-${index}`;
      if (part.startsWith("**") && part.endsWith("**"))
        return (
          <strong key={key}>
            {renderInline(part.slice(2, -2), onSeek, t, key)}
          </strong>
        );
      const range = part.match(RANGE);
      if (range || new RegExp(`^${TIME}$`).test(part)) {
        const start = range ? range[1] : part;
        const timestamp = range?.[2] ? `${start}–${range[2]}` : start;
        const seconds = toSeconds(start);
        return (
          <button
            key={key}
            className="timestamp-link"
            onClick={() => onSeek(seconds)}
            aria-label={`${t.seekTo} ${timestamp}`}
          >
            <Icon name="play" size={11} />
            {timestamp}
          </button>
        );
      }
      return part;
    });
}

function FormattedText({ text, onSeek, t }) {
  const blocks = [];
  let list = [];
  function flushList(key) {
    if (list.length) {
      blocks.push(
        <ul key={`list-${key}`}>
          {list.map((line, index) => (
            <li key={index}>
              {renderInline(line, onSeek, t, `list-${key}-${index}`)}
            </li>
          ))}
        </ul>,
      );
      list = [];
    }
  }
  text.split("\n").forEach((line, index) => {
    const bullet = line.match(/^\s*[-*]\s+(.*)/);
    if (bullet) {
      list.push(bullet[1]);
      return;
    }
    flushList(index);
    if (line.trim())
      blocks.push(
        <p
          key={index}
          className={/^#{1,3}\s/.test(line) ? "chat-heading" : undefined}
        >
          {renderInline(line.replace(/^#{1,3}\s/, ""), onSeek, t, `p-${index}`)}
        </p>,
      );
  });
  flushList("end");
  return <div className="formatted-answer">{blocks}</div>;
}

const ChatMessage = memo(function ChatMessage({ message, onSeek, onRetry, onSource, currentVideoId, t }) {
  const user = message.role === "user";
  return (
    <div
      className={`chat-message ${user ? "message-user" : "message-assistant"} ${message.error ? "message-error" : ""}`}
    >
      {!user && (
        <span className="message-avatar">
          <Icon name={message.error ? "warning" : "sparkles"} size={16} />
        </span>
      )}
      <div className="message-content">
        <span className="message-author">{user ? t.you : t.ai}</span>
        <div className="message-bubble">
          {user ? (
            message.text
          ) : (
            <FormattedText text={message.text} onSeek={onSeek} t={t} />
          )}
        </div>
        {!user && message.rewritten && (
          <p className="answer-rewritten" title={t.rewrittenHint}>
            {t.understoodAs}: {message.rewritten}
          </p>
        )}
        {!user && message.verification && (
          <p className={`answer-verification ${message.verification.label === "verified" ? "is-verified" : "is-unverified"}`}>
            <Icon name={message.verification.label === "verified" ? "checkCircle" : "warning"} size={12} />
            {message.verification.label === "verified" ? t.verified : t.unverified}
            {message.verification.reasons?.length > 0 &&
              `: ${message.verification.reasons.map((reason) => t.verificationReasons[reason] || reason).join(", ")}`}
          </p>
        )}
        {!user && message.sources?.length > 0 && (
          <div className="answer-sources">
            <span>{t.sources}</span>
            {[...message.sources]
              .sort((a, b) => (a.video_title || "").localeCompare(b.video_title || "") || a.start - b.start)
              .map((source) => {
                const elsewhere = source.video_id && source.video_id !== currentVideoId;
                return (
                  <button
                    key={`${source.video_id}-${source.segment_id}`}
                    className={`timestamp-link ${elsewhere ? "other-video" : ""}`}
                    onClick={() => onSource(source)}
                    title={[source.video_title, source.title].filter(Boolean).join(" · ") || undefined}
                    aria-label={`${t.seekTo} ${elsewhere ? `${source.video_title} ` : ""}${clock(source.start)}`}
                  >
                    <Icon name="play" size={11} />
                    {elsewhere && <span className="source-video">{source.video_title}</span>}
                    {clock(source.start)}
                  </button>
                );
              })}
          </div>
        )}
        {message.error && (
          <button className="retry-question" onClick={() => onRetry(message)}>
            <Icon name="refresh" size={13} />
            {t.retryQuestion}
          </button>
        )}
      </div>
    </div>
  );
});

function ChatHistory({ currentId, onOpen, t }) {
  const [items, setItems] = useState(null);
  const [failed, setFailed] = useState(false);
  const [refresh, setRefresh] = useState(0);
  useEffect(() => {
    let alive = true;
    request("/conversations")
      .then((data) => alive && setItems(Array.isArray(data) ? data : []))
      .catch(() => alive && setFailed(true));
    return () => {
      alive = false;
    };
  }, [refresh]);
  async function remove(id) {
    try {
      await request(`/conversations/${id}`, { method: "DELETE" });
    } finally {
      setRefresh((value) => value + 1);
    }
  }
  return (
    <div className="chat-history" role="dialog" aria-label={t.chatHistory}>
      <p className="chat-history-title">{t.chatHistory}</p>
      {failed ? (
        <p className="chat-history-empty">{t.historyError}</p>
      ) : items === null ? (
        <Spinner label={t.loading} />
      ) : items.length === 0 ? (
        <p className="chat-history-empty">{t.historyEmpty}</p>
      ) : (
        <ul>
          {items.map((item) => (
            <li key={item.id} className={item.id === currentId ? "active" : ""}>
              <button className="chat-history-open" onClick={() => onOpen(item.id)}>
                <span>{item.title || t.untitledChat}</span>
                <small>
                  {new Date(item.updated_at).toLocaleString()} · {item.turns} {t.questionsWord}
                  {item.video_ids?.length > 1 ? ` · ${item.video_ids.length} ${t.videosWord}` : ""}
                </small>
              </button>
              <button className="icon-button" onClick={() => remove(item.id)} aria-label={t.deleteChat} title={t.deleteChat}>
                <Icon name="close" size={14} />
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function ChatPanel({
  messages,
  asking,
  question,
  setQuestion,
  onAsk,
  onSeek,
  onRetry,
  onSource,
  currentVideoId,
  ready,
  onClose,
  history,
  lang,
  t,
}) {
  const textareaRef = useRef(null);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [isLiveMode, setIsLiveMode] = useState(false);
  const [liveInput, setLiveInput] = useState("");
  const [micActive, setMicActive] = useState(false);
  const recognitionRef = useRef(null);
  const messagesRef = useRef(null);
  const nearBottom = useRef(true);

  useEffect(() => {
    const element = textareaRef.current;
    if (element) {
      element.style.height = "auto";
      element.style.height = `${Math.min(element.scrollHeight, 120)}px`;
    }
  }, [question]);

  useEffect(() => {
    const element = messagesRef.current;
    if (element && (nearBottom.current || messages.at(-1)?.role === "user")) {
      element.scrollTo({
        top: element.scrollHeight,
        behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
          ? "instant"
          : "smooth",
      });
    }
  }, [messages, asking]);

  function handleLiveSubmit(event) {
    event?.preventDefault();
    if (!liveInput.trim() || asking || !ready) return;
    onAsk(liveInput.trim());
    setLiveInput("");
  }

  function toggleMic() {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert(lang === "vi" ? "Trình duyệt không hỗ trợ nhận diện giọng nói." : "Speech recognition not supported in this browser.");
      return;
    }
    if (micActive) {
      recognitionRef.current?.stop();
      setMicActive(false);
      return;
    }
    try {
      const recognition = new SpeechRecognition();
      recognition.lang = lang === "vi" ? "vi-VN" : "en-US";
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.onstart = () => setMicActive(true);
      recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        if (transcript) {
          setLiveInput(transcript);
          onAsk(transcript);
        }
      };
      recognition.onerror = () => setMicActive(false);
      recognition.onend = () => setMicActive(false);
      recognitionRef.current = recognition;
      recognition.start();
    } catch {
      setMicActive(false);
    }
  }

  // Nếu đang ở chế độ Hỏi Trung (Live Mode) nằm gọn trong frame này:
  if (isLiveMode) {
    const userMessages = messages.filter((m) => m.role === "user");
    const aiMessages = messages.filter((m) => m.role === "ai");
    const latestUser = userMessages.length > 0 ? userMessages[userMessages.length - 1] : null;
    const latestAi = aiMessages.length > 0 ? aiMessages[aiMessages.length - 1] : null;

    return (
      <section className="chat-panel chat-panel--live" aria-label="Hỏi Trung">
        <div className="live-frame-body">
          {/* Tin nhắn user ở góc trên bên phải */}
          <div className="live-frame-user-slot">
            {latestUser && (
              <div className="live-user-pill">
                {latestUser.text}
              </div>
            )}
          </div>

          {/* Lời chào hoặc câu trả lời của AI ở góc trên bên trái */}
          <div className="live-frame-ai-slot">
            <p className="live-ai-text">
              {asking
                ? (lang === "vi" ? "Đang suy nghĩ..." : "Thinking...")
                : latestAi
                  ? latestAi.text
                  : (lang === "vi"
                      ? "Chào bạn, hôm nay bạn muốn mình trò chuyện về chủ đề gì nhỉ?"
                      : "Hi! What would you like to chat about today?")}
            </p>
          </div>

          {/* Quả cầu phát sáng Gemini Live Orb ở giữa phía dưới */}
          <div className="live-frame-orb-wrapper">
            <div className={`live-orb ${asking ? "is-asking" : micActive ? "is-listening" : ""}`}>
              <div className="live-orb-inner" />
            </div>
          </div>

          {/* Thanh input ở đáy */}
          <form className="live-frame-bottom-bar" onSubmit={handleLiveSubmit}>
            <button type="button" className="live-add-btn" title="Thêm">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round">
                <path d="M12 5v14M5 12h14" />
              </svg>
            </button>

            <input
              type="text"
              className="live-input-field"
              value={liveInput}
              onChange={(e) => setLiveInput(e.target.value)}
              placeholder={lang === "vi" ? "Nhập" : "Type..."}
              autoFocus
            />

            <button
              type="button"
              className={`live-mic-btn ${micActive ? "active" : ""}`}
              onClick={toggleMic}
              title={micActive ? "Tắt mic" : "Bật mic"}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
                <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                <line x1="12" y1="19" x2="12" y2="22" />
                <line x1="2" y1="2" x2="22" y2="22" />
              </svg>
            </button>

            <button
              type="button"
              className="live-close-btn"
              onClick={() => setIsLiveMode(false)}
              title="Quay lại giao diện chat"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </form>
        </div>
      </section>
    );
  }

  return (
    <section className="chat-panel" aria-label={t.assistant}>
      <div className="chat-panel-header">
        <h2 className="chat-panel-title">
          {lang === "vi" ? "Hỏi về video này" : "Ask about this video"}
        </h2>
        <div className="chat-panel-actions">
          {history && (
            <div className="chat-session-actions">
              <button
                className="icon-button"
                onClick={() => setHistoryOpen((value) => !value)}
                aria-expanded={historyOpen}
                aria-label={t.chatHistory}
                title={t.chatHistory}
              >
                <Icon name="clock" size={17} />
              </button>
              <button
                className="icon-button"
                onClick={() => {
                  setHistoryOpen(false);
                  history.onNew();
                }}
                aria-label={t.newChat}
                title={t.newChat}
                disabled={!messages.length}
              >
                <Icon name="plus" size={17} />
              </button>
            </div>
          )}
          {onClose && (
            <button
              className="icon-button"
              onClick={onClose}
              aria-label={t.close}
              title={t.close}
            >
              <Icon name="close" size={18} />
            </button>
          )}
        </div>
      </div>
      {history && historyOpen && (
        <ChatHistory
          currentId={history.currentId}
          onOpen={(id) => {
            setHistoryOpen(false);
            history.onOpen(id);
          }}
          t={t}
        />
      )}
      <div
        ref={messagesRef}
        className="chat-messages"
        role="log"
        aria-label={t.assistant}
        aria-live="polite"
        aria-relevant="additions text"
        onScroll={(event) => {
          const element = event.currentTarget;
          nearBottom.current =
            element.scrollHeight - element.scrollTop - element.clientHeight <
            100;
        }}
      >
        {!ready ? (
          <EmptyState
            icon="transcript"
            title={t.noTranscriptTitle}
            description={t.noTranscriptDescription}
            compact
          />
        ) : messages.length === 0 ? (
          <div className="chat-welcome">
            <div className="chat-welcome-star">✦</div>
            <p className="chat-welcome-greeting">
              {lang === "vi"
                ? "Xin chào! Bạn có thắc mắc về nội dung đang xem? Tôi sẵn sàng giúp bạn."
                : "Hi! Have questions about what you're watching? I'm here to help."}
            </p>
            <p className="chat-welcome-prompt">
              {lang === "vi"
                ? "Bạn chưa biết nên hỏi gì? Hãy thử hỏi về:"
                : "Not sure what to ask? Try asking about:"}
            </p>
            <div className="question-suggestions">
              {(lang === "vi"
                ? [
                  { label: "Tóm tắt video này", q: "Tóm tắt video này." },
                  { label: "Những ý chính là gì?", q: "Những ý chính cần ghi nhớ trong video này là gì?" },
                  { label: "Giải thích các khái niệm quan trọng", q: "Giải thích các khái niệm quan trọng được nhắc đến trong bài giảng." },
                ]
                : [
                  { label: "Summarize this video", q: "Summarize this video." },
                  { label: "Suggest related topics", q: "Suggest related topics to this lecture." },
                  { label: "What are the key points?", q: "What are the main key points in this video?" },
                  { label: "Explain key concepts", q: "Explain key concepts mentioned in this lecture." },
                ]
              ).map((item, index) => (
                <button
                  key={index}
                  className="suggestion-pill"
                  disabled={asking}
                  onClick={() => onAsk(item.q)}
                >
                  {item.label}
                </button>
              ))}
            </div>
          </div>
        ) : null}
        {messages.map((message) => (
          <ChatMessage
            key={message.id}
            message={message}
            onSeek={onSeek}
            onRetry={onRetry}
            onSource={onSource}
            currentVideoId={currentVideoId}
            t={t}
          />
        ))}
        {asking && messages[messages.length - 1]?.role !== "ai" && (
          <div className="typing-indicator" role="status">
            <span className="message-avatar">
              <Icon name="sparkles" size={16} />
            </span>
            <div>
              <span />
              <span />
              <span />
            </div>
            <span className="sr-only">{t.thinking}</span>
          </div>
        )}
      </div>
      <div className="chat-composer">
        <p className="composer-disclaimer">
          {lang === "vi"
            ? "AI có thể mắc sai sót, vì vậy hãy xác minh thông tin. Tìm hiểu thêm"
            : t.chatFootnote}
        </p>
        <form
          className="composer-pill-form"
          onSubmit={(event) => {
            event.preventDefault();
            onAsk(question);
          }}
        >
          <textarea
            ref={textareaRef}
            rows={1}
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            aria-label={t.askPlaceholder}
            placeholder={lang === "vi" ? "Đặt câu hỏi…" : t.askPlaceholder}
            disabled={!ready || asking}
            onKeyDown={(event) => {
              if (
                event.key === "Enter" &&
                !event.shiftKey &&
                !event.nativeEvent.isComposing &&
                event.keyCode !== 229
              ) {
                event.preventDefault();
                onAsk(question);
              }
            }}
          />
          <button
            className="send-button-pill"
            type="submit"
            disabled={asking || !ready || !question.trim()}
            aria-label={t.send}
            title={t.enterHint}
          >
            {asking ? <Spinner /> : <Icon name="send" size={17} />}
          </button>
        </form>
        <button
          type="button"
          className="composer-gemini-badge"
          onClick={() => setIsLiveMode(true)}
          title={lang === "vi" ? "Chuyển sang chế độ Hỏi Trung" : "Switch to Ask Trung mode"}
        >
          <span>{lang === "vi" ? "Hỏi" : "Ask"}</span>
          <span className="gemini-star">✦</span>
          <span>Trung</span>
        </button>
      </div>
    </section>
  );
}

const SlideThumbnail = memo(function SlideThumbnail({
  videoId,
  name,
  index,
  active,
  onOpen,
  meta,
  t,
}) {
  const [failed, setFailed] = useState(false);
  return (
    <div className="slide-card">
      <button
        className={`slide-thumbnail ${active ? "active" : ""}`}
        onClick={() => onOpen(index)}
        aria-label={`${t.slide} ${index + 1}`}
        aria-current={active ? "true" : undefined}
      >
        <span className="slide-image">
          {failed ? (
            <Icon name="slides" size={25} />
          ) : (
            <img
              src={slideUrl(videoId, name)}
              alt=""
              loading="lazy"
              decoding="async"
              onError={() => setFailed(true)}
            />
          )}
          <span className="slide-expand">
            <Icon name="expand" size={17} />
          </span>
        </span>
        <span className="slide-caption">
          <span>
            {t.slide} {String(index + 1).padStart(2, "0")}
          </span>
          {active && <span className="active-slide-dot" />}
        </span>
        {meta && (
          <span className="slide-meta">
            <span>{meta}</span>
          </span>
        )}
      </button>
    </div>
  );
});

const VERDICT_LABELS = {
  good: "reviewGood",
  wrong_merge: "reviewWrongMerge",
  extra_split: "reviewExtraSplit",
  bad_image: "reviewBadImage",
  not_slide: "reviewNotSlide",
  other: "reviewOther",
};
const VERDICTS = Object.entries(VERDICT_LABELS);

function SlideReviewPanel({ videoId, item, review, onSave, onSeek, t }) {
  const [note, setNote] = useState(review?.note || "");
  const verdict = review?.verdict || null;
  return (
    <div className="slide-review">
      {item.states.length > 1 && (
        <div className="reveal-states">
          <h3>
            {t.revealStatesTitle} ({item.states.length})
          </h3>
          <div className="reveal-strip">
            {item.states.map((state) => (
              <button
                key={state.image}
                className="reveal-state"
                onClick={() => onSeek(state.start_sec)}
                title={state.decision || ""}
              >
                <img
                  src={slideUrl(videoId, state.image)}
                  alt=""
                  loading="lazy"
                />
                <span>{formatDuration(state.start_sec)}</span>
              </button>
            ))}
          </div>
        </div>
      )}
      <div className="review-controls">
        <h3>{t.reviewTitle}</h3>
        <div className="review-buttons" role="group" aria-label={t.reviewTitle}>
          {VERDICTS.map(([value, label]) => (
            <button
              key={value}
              className={`button button-secondary review-${value} ${verdict === value ? "selected" : ""}`}
              aria-pressed={verdict === value}
              onClick={() =>
                onSave({ verdict: verdict === value ? null : value, note })
              }
            >
              {t[label]}
            </button>
          ))}
        </div>
        <textarea
          id={`review-note-${item.key}`}
          className="review-note"
          placeholder={t.reviewNote}
          value={note}
          maxLength={2000}
          rows={2}
          onChange={(event) => setNote(event.target.value)}
          onBlur={() => {
            if (note !== (review?.note || "")) onSave({ verdict, note });
          }}
        />
      </div>
    </div>
  );
}

function SlideImage({ src, label, t }) {
  const [failed, setFailed] = useState(false);
  return failed ? (
    <EmptyState icon="slides" title={t.slidesError} compact />
  ) : (
    <img
      className="lightbox-image"
      src={src}
      alt={label}
      onError={() => setFailed(true)}
    />
  );
}

function Transcript({ segments, currentTime, onSeek, t }) {
  const [query, setQuery] = useState("");
  const [pagination, setPagination] = useState({ query: "", count: 80 });
  const deferredQuery = useDeferredValue(query);
  const filtered = useMemo(
    () =>
      segments
        .map((segment, index) => ({ ...segment, index }))
        .filter((segment) =>
          segment.text
            .toLocaleLowerCase()
            .includes(deferredQuery.trim().toLocaleLowerCase()),
        ),
    [segments, deferredQuery],
  );
  const count = pagination.query === deferredQuery ? pagination.count : 80;
  return (
    <div className="transcript-content">
      <div className="transcript-search">
        <Icon name="search" size={17} />
        <input
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder={t.transcriptSearch}
          aria-label={t.transcriptSearch}
        />
        <span>
          {filtered.length} {t.segments}
        </span>
      </div>
      <div className="transcript-segments" aria-busy={query !== deferredQuery}>
        {filtered.length === 0 && (
          <p className="transcript-empty">
            {segments.length ? t.noTranscriptResults : t.noTranscriptLines}
          </p>
        )}
        {filtered.slice(0, count).map((segment) => (
          <button
            key={segment.index}
            className={`transcript-segment ${currentTime >= segment.start && currentTime < segment.end ? "active" : ""}`}
            onClick={() => onSeek(segment.start)}
          >
            <span className="segment-time">
              {formatDuration(segment.start)}
            </span>
            <span>{segment.text}</span>
            <Icon name="play" size={13} />
          </button>
        ))}
        {filtered.length > count && (
          <button
            className="button button-secondary transcript-more"
            onClick={() =>
              setPagination({ query: deferredQuery, count: count + 80 })
            }
          >
            {t.showMore} ({filtered.length - count})
          </button>
        )}
      </div>
    </div>
  );
}

export default function Workspace({
  video,
  ready = video.has_transcript,
  onExtract,
  initialQuestion = "",
  initialChatOpen = false,
  initialTime = 0,
  askVideoIds = [],
  chat,
  chatOpen: propChatOpen,
  setChatOpen: propSetChatOpen,
  onOpenSource,
  lang,
  t,
}) {
  const videoRef = useRef(null);
  const qaRequest = useRef(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [mediaDuration, setMediaDuration] = useState(null);
  const [slidesResult, setSlidesResult] = useState({
    items: null,
    error: false,
  });
  const [slidesRetry, setSlidesRetry] = useState(0);
  const [tab] = useState(video.has_slides ? "slides" : "transcript");
  const [contentMode, setContentMode] = useState("video"); // "video" | "slides" | "transcript"
  const contentTabsId = useId();
  const [transcript, setTranscript] = useState(null);
  const [transcriptError, setTranscriptError] = useState(false);
  const [transcriptRetry, setTranscriptRetry] = useState(0);
  const [outline, setOutline] = useState(null);
  const [lightbox, setLightbox] = useState(null);
  const [localChatOpen, setLocalChatOpen] = useState(initialChatOpen);
  const isChatOpen = propChatOpen !== undefined ? propChatOpen : localChatOpen;
  const toggleChat = propSetChatOpen || setLocalChatOpen;
  // App keeps the conversation, so it survives switching to a video a source points at.
  const [ownMessages, setOwnMessages] = useState([]);
  const messages = chat ? chat.messages : ownMessages;
  const setMessages = chat ? chat.setMessages : setOwnMessages;
  const chatHistory = useRef(messages);
  chatHistory.current = messages;
  const [question, setQuestion] = useState(initialQuestion);
  const [asking, setAsking] = useState(false);
  const desktopChat = useMediaQuery("(min-width: 900px)");
  const [chatWidth, setChatWidth, resetChatWidth] = useStoredSize("chat", 380, 280, 720);
  const [playerHeight] = useStoredSize("player", 400, 160, 900);
  const [variants, setVariants] = useState(null);
  const [variantName, setVariantName] = useState(null);
  const [reviews, setReviews] = useState({});
  const [reviewStatus, setReviewStatus] = useState("");
  const duration = mediaDuration || video.duration_seconds || 0;
  // Imported segmentations carry real slide times; otherwise positions are estimated.
  const timedSlides =
    variants && variantName ? variants.variants[variantName].slides : null;
  const slides = timedSlides
    ? timedSlides.map((item) => item.image)
    : video.has_slides
      ? slidesResult.items
      : [];
  const activeSlide = timedSlides
    ? Math.max(
      0,
      timedSlides.findLastIndex((item) => item.start_sec <= currentTime),
    )
    : slides?.length && duration
      ? Math.min(
        slides.length - 1,
        Math.floor((currentTime / duration) * slides.length),
      )
      : null;
  const variantReviews = (variantName && reviews[variantName]) || {};

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([
      getSlideVariants(video.video_id, { signal: controller.signal }),
      getSlideReviews(video.video_id, { signal: controller.signal }),
    ])
      .then(([data, saved]) => {
        if (controller.signal.aborted || !data) return;
        setVariants(data);
        setVariantName(data.default);
        setReviews(saved || {});
      })
      .catch(() => { });
    return () => controller.abort();
  }, [video.video_id, video.extraction?.status]);

  const saveReview = useCallback(
    (slideKey, review) => {
      setReviewStatus("");
      saveSlideReview(video.video_id, variantName, slideKey, review)
        .then((saved) => {
          setReviews(saved);
          setReviewStatus(t.reviewSaved);
        })
        .catch(() => setReviewStatus(t.reviewError));
    },
    [video.video_id, variantName, t.reviewSaved, t.reviewError],
  );

  useEffect(() => {
    if (!video.has_slides) return;
    let active = true;
    getSlides(video.video_id)
      .then((items) => {
        if (active) setSlidesResult({ items, error: false });
      })
      .catch(() => {
        if (active) setSlidesResult({ items: [], error: true });
      });
    return () => {
      active = false;
    };
  }, [video.video_id, video.has_slides, slidesRetry, video.extraction?.status]);

  useEffect(() => {
    if (video.extraction?.status === "completed") {
      setOutline(null);
      setTranscript(null);
    }
  }, [video.extraction?.status]);

  useEffect(() => {
    if (tab !== "structure" || outline) return;
    const controller = new AbortController();
    request(`/videos/${encodeURIComponent(video.video_id)}/outline`, { signal: controller.signal })
      .then((data) => !controller.signal.aborted && setOutline(Array.isArray(data) ? data : []))
      .catch(() => !controller.signal.aborted && setOutline([]));
    return () => controller.abort();
  }, [tab, video.video_id, outline]);

  useEffect(() => {
    if (tab !== "transcript" || !video.has_transcript || transcript) return;
    const controller = new AbortController();
    request(`/videos/${encodeURIComponent(video.video_id)}/transcript`, {
      signal: controller.signal,
    })
      .then((data) => {
        if (!Array.isArray(data.segments))
          throw new Error("Invalid transcript");
        if (!controller.signal.aborted) setTranscript(data);
      })
      .catch(() => {
        if (!controller.signal.aborted) setTranscriptError(true);
      });
    return () => controller.abort();
  }, [tab, video.video_id, video.has_transcript, transcript, transcriptRetry]);

  useEffect(() => () => qaRequest.current?.abort(), []);

  useEffect(() => {
    const element = videoRef.current;
    if (!element || !initialTime) return;
    const go = () => {
      element.currentTime = initialTime;
    };
    if (element.readyState >= 1) go();
    else element.addEventListener("loadedmetadata", go, { once: true });
    return () => element.removeEventListener("loadedmetadata", go);
  }, [initialTime]);

  const seekTo = useCallback((seconds) => {
    const element = videoRef.current;
    if (!element || !Number.isFinite(seconds)) return;
    const time = Math.max(
      0,
      Math.min(
        seconds,
        Number.isFinite(element.duration) ? element.duration : seconds,
      ),
    );
    element.currentTime = time;
    setCurrentTime(Math.floor(time));
    element.play().catch(() => { });
  }, []);

  const sendQuestion = useCallback(
    async (text, contextTime, retryId) => {
      const content = typeof text === "string" ? text.trim() : "";
      if (!content || !ready || qaRequest.current) return;
      const controller = new AbortController();
      qaRequest.current = controller;
      const time = contextTime ?? videoRef.current?.currentTime ?? 0;
      setMessages((items) =>
        retryId
          ? items.filter((item) => item.id !== retryId)
          : [
            ...items,
            { id: crypto.randomUUID(), role: "user", text: content },
          ],
      );
      if (!retryId) setQuestion("");
      setAsking(true);
      // The question before this one, so a follow-up ("còn cái kia?") is searched with it.
      const previousQuestion = [...chatHistory.current]
        .reverse()
        .find((item) => item.role === "user" && item.text !== content)?.text;
      const body = JSON.stringify({
        video_id: video.video_id,
        question: content,
        current_time: time,
        previous_question: previousQuestion ?? null,
        video_ids: askVideoIds.length ? askVideoIds : null,
        conversation_id: chat?.conversationId ?? null,
      });
      const answerId = crypto.randomUUID();
      try {
        // Indexed videos stream the answer as it is written; others answer in one piece.
        let sources = [];
        let rewritten = null;
        const streamed = await streamAnswer(
          body,
          controller.signal,
          (text) =>
            setMessages((items) =>
              items.some((item) => item.id === answerId)
                ? items.map((item) =>
                  item.id === answerId ? { ...item, text: item.text + text } : item,
                )
                : [...items, { id: answerId, role: "ai", text, sources, rewritten }],
            ),
          (found) => {
            sources = found;
          },
          (meta) => {
            rewritten = meta.rewritten || null;
            if (meta.conversation_id) chat?.onConversation?.(meta.conversation_id);
          },
          (done) => {
            const verification = done.agent?.verification;
            if (typeof done.answer === "string" || verification)
              setMessages((items) =>
                items.map((item) =>
                  item.id === answerId
                    ? { ...item, text: typeof done.answer === "string" ? done.answer : item.text, verification }
                    : item,
                ),
              );
          },
        );
        if (!streamed) {
          const data = await request("/qa", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body,
            signal: controller.signal,
          });
          if (typeof data.answer !== "string") throw new Error(t.answerError);
          if (!controller.signal.aborted)
            setMessages((items) => [
              ...items,
              {
                id: answerId,
                role: "ai",
                text: data.answer,
                sources: data.sources,
                rewritten: data.rewritten,
                verification: data.agent?.verification,
              },
            ]);
          if (data.conversation_id) chat?.onConversation?.(data.conversation_id);
        }
      } catch (error) {
        if (!controller.signal.aborted)
          setMessages((items) => [
            ...items,
            {
              id: crypto.randomUUID(),
              role: "ai",
              text: error instanceof TypeError ? t.answerError : error.message,
              error: true,
              retryText: content,
              contextTime: time,
            },
          ]);
      } finally {
        if (!controller.signal.aborted) {
          qaRequest.current = null;
          setAsking(false);
        }
      }
    },
    [video.video_id, ready, t.answerError, askVideoIds, setMessages],
  );

  const retryQuestion = useCallback(
    (message) =>
      sendQuestion(message.retryText, message.contextTime, message.id),
    [sendQuestion],
  );

  function askAboutSlide() {
    const index = lightbox;
    if (index === null || !slides?.length) return;
    const time = timedSlides
      ? timedSlides[index].start_sec
      : duration
        ? (index / slides.length) * duration
        : (videoRef.current?.currentTime ?? 0);
    setLightbox(null);
    toggleChat(true);
    sendQuestion(
      `${t.slideQuestion} (${t.slide} ${index + 1}, ${t.approximateTime}: ${formatDuration(time)})`,
      time,
    );
  }

  const tabKeys = ["video", "slides", "transcript"];

  function handleTabKeys(event) {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const index = tabKeys.indexOf(contentMode);
    const nextTab =
      event.key === "Home"
        ? tabKeys[0]
        : event.key === "End"
          ? tabKeys[tabKeys.length - 1]
          : tabKeys[(index + (event.key === "ArrowRight" ? 1 : tabKeys.length - 1)) % tabKeys.length];
    setContentMode(nextTab);
    document.getElementById(`${contentTabsId}-${nextTab}-tab`)?.focus();
  }

  const chatProps = {
    messages,
    asking,
    question,
    setQuestion,
    onAsk: sendQuestion,
    onSeek: seekTo,
    onRetry: retryQuestion,
    onSource: (source) =>
      source.video_id && source.video_id !== video.video_id
        ? onOpenSource?.(source.video_id, source.start)
        : seekTo(source.start),
    currentVideoId: video.video_id,
    history: chat?.history,
    ready,
    lang,
    t,
  };

  return (
    <div className="workspace-page page-enter">
      <div
        className="study-layout"
        style={desktopChat && isChatOpen ? { gridTemplateColumns: `minmax(0, 1fr) 5px ${chatWidth}px` } : { gridTemplateColumns: "minmax(0, 1fr)" }}
      >
        <div className="study-main" style={{ "--player-h": `${playerHeight}px` }}>
          <section className="player-section player-section--full" aria-label={video.filename}>
            <VideoPlayer
              video={video}
              videoRef={videoRef}
              poster={slides?.[0] ? slideUrl(video.video_id, slides[0]) : undefined}
              onTimeChange={setCurrentTime}
              onDurationChange={setMediaDuration}
              contentMode={contentMode}
              contentTabsId={contentTabsId}
              toolbarCenter={
                <div
                  className="mode-btn-group"
                  role="tablist"
                  aria-label={t.videoContent}
                  onKeyDown={handleTabKeys}
                >
                  {[
                    { key: "video", label: "Video", icon: "play" },
                    { key: "slides", label: "Slide", icon: "slides" },
                    { key: "transcript", label: lang === "vi" ? "Lời giảng" : "Transcript", icon: "transcript" },
                  ].map(({ key, label, icon }) => (
                    <button
                      key={key}
                      id={`${contentTabsId}-${key}-tab`}
                      className={`mode-btn ${contentMode === key ? "active" : ""}`}
                      role="tab"
                      aria-selected={contentMode === key}
                      aria-controls={`${contentTabsId}-${key}-panel`}
                      tabIndex={contentMode === key ? 0 : -1}
                      onClick={() => setContentMode(key)}
                    >
                      <Icon name={icon} size={14} />
                      {label}
                    </button>
                  ))}
                </div>
              }
              slidesContent={
                slidesResult.error ? (
                  <EmptyState icon="warning" title={t.slidesError} compact>
                    <button
                      className="button button-secondary"
                      onClick={() => {
                        setSlidesResult({ items: null, error: false });
                        setSlidesRetry((v) => v + 1);
                      }}
                    >
                      {t.retry}
                    </button>
                  </EmptyState>
                ) : slides === null ? (
                  <div className="slides-loading" role="status" aria-label={t.loadingSlides}>
                    {[0, 1, 2, 3, 4, 5].map((item) => (
                      <div key={item} />
                    ))}
                  </div>
                ) : slides.length === 0 ? (
                  <EmptyState icon="slides" title={t.noSlides} description={t.noSlidesDescription} compact />
                ) : (
                  <div className="slides-panel-inner">
                    {variants && (
                      <div className="variant-picker">
                        <label htmlFor="slide-variant">{t.slideVariant}</label>
                        <select
                          id="slide-variant"
                          value={variantName}
                          onChange={(event) => {
                            setVariantName(event.target.value);
                            setLightbox(null);
                            setReviewStatus("");
                          }}
                        >
                          {Object.entries(variants.variants).map(([name, value]) => (
                            <option key={name} value={name}>
                              {value.label} ({value.slides.length})
                            </option>
                          ))}
                        </select>
                        <span className="variant-progress">
                          {Object.keys(variantReviews).length}/{slides.length} {t.reviewedCount}
                        </span>
                      </div>
                    )}
                    <div className="slides-grid">
                      {slides.map((name, index) => (
                        <SlideThumbnail
                          key={`${variantName}-${index}-${name}`}
                          videoId={video.video_id}
                          name={name}
                          index={index}
                          active={index === activeSlide}
                          onOpen={setLightbox}
                          meta={
                            timedSlides &&
                            `${formatDuration(timedSlides[index].start_sec)}–${formatDuration(timedSlides[index].end_sec)} · ${timedSlides[index].states.length} ${t.revealStates}`
                          }
                          t={t}
                        />
                      ))}
                    </div>
                  </div>
                )
              }
              transcriptContent={
                !video.has_transcript ? (
                  <EmptyState icon="transcript" title={t.noTranscriptTitle} description={t.noTranscriptDescription} compact />
                ) : transcriptError ? (
                  <EmptyState icon="warning" title={t.transcriptError} compact>
                    <button
                      className="button button-secondary"
                      onClick={() => {
                        setTranscriptError(false);
                        setTranscriptRetry((v) => v + 1);
                      }}
                    >
                      {t.retry}
                    </button>
                  </EmptyState>
                ) : !transcript ? (
                  <div className="resource-loading">
                    <Spinner label={t.loadingTranscript} />
                    <span>{t.loadingTranscript}</span>
                  </div>
                ) : (
                  <div className="transcript-panel-inner">
                    <Transcript segments={transcript.segments} currentTime={currentTime} onSeek={seekTo} t={t} />
                  </div>
                )
              }
              t={t}
            />
            {video.extraction && video.extraction.status !== "completed" && (
              <ExtractionProgress job={video.extraction} onRetry={onExtract} t={t} />
            )}
          </section>
        </div>
        {desktopChat && isChatOpen && (
          <Splitter size={chatWidth} onResize={setChatWidth} onReset={resetChatWidth} invert label={t.resizePanel} />
        )}
        {desktopChat && isChatOpen && (
          <div className="desktop-chat">
            <ChatPanel {...chatProps} onClose={() => toggleChat(false)} />
          </div>
        )}
      </div>
      {!desktopChat && (
        <button
          className="mobile-chat-button"
          onClick={() => toggleChat(true)}
          aria-haspopup="dialog"
        >
          <Icon name="sparkles" size={19} />
          {t.ask}
          {asking && <Spinner />}
        </button>
      )}
      <Dialog
        open={!desktopChat && isChatOpen}
        onClose={() => toggleChat(false)}
        title={t.assistant}
        showClose={false}
        className="chat-dialog"
      >
        <ChatPanel {...chatProps} onClose={() => toggleChat(false)} />
      </Dialog>
      <Dialog
        open={lightbox !== null}
        onClose={() => setLightbox(null)}
        title={`${t.slide} ${(lightbox ?? 0) + 1}`}
        closeLabel={t.close}
        className="slide-dialog"
        onKeyDown={(event) => {
          if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
            event.preventDefault();
            setLightbox((index) =>
              Math.max(
                0,
                Math.min(
                  (slides?.length || 1) - 1,
                  index + (event.key === "ArrowLeft" ? -1 : 1),
                ),
              ),
            );
          }
        }}
      >
        {lightbox !== null && slides?.[lightbox] && (
          <div className="slide-viewer">
            <div className="slide-viewer-heading">
              <Icon name="slides" size={21} />
              <h2>
                {t.slide} {lightbox + 1}
                <span> / {slides.length}</span>
                {timedSlides && (
                  <span>
                    {" · "}
                    {formatDuration(timedSlides[lightbox].start_sec)}–
                    {formatDuration(timedSlides[lightbox].end_sec)}
                  </span>
                )}
              </h2>
            </div>
            <SlideImage
              key={slides[lightbox]}
              src={slideUrl(video.video_id, slides[lightbox])}
              label={`${t.slide} ${lightbox + 1}`}
              t={t}
            />
            <div className="slide-viewer-actions">
              <div className="slide-pagination">
                <button
                  className="icon-button"
                  disabled={lightbox === 0}
                  onClick={() => setLightbox((index) => index - 1)}
                  aria-label={t.previous}
                >
                  <Icon name="back" size={19} />
                </button>
                <span>
                  {lightbox + 1} / {slides.length}
                </span>
                <button
                  className="icon-button"
                  disabled={lightbox === slides.length - 1}
                  onClick={() => setLightbox((index) => index + 1)}
                  aria-label={t.next}
                >
                  <Icon name="arrow" size={19} />
                </button>
              </div>
              {timedSlides && (
                <button
                  className="button button-secondary"
                  onClick={() => {
                    const start = timedSlides[lightbox].start_sec;
                    setLightbox(null);
                    seekTo(start);
                  }}
                >
                  <Icon name="clock" size={17} />
                  {t.playFromSlide}
                </button>
              )}
              <button
                className="button button-primary"
                disabled={!ready || asking}
                onClick={askAboutSlide}
                title={!ready ? t.transcriptMissing : undefined}
              >
                <Icon name="sparkles" size={17} />
                {t.askSlide}
              </button>
            </div>
            {timedSlides ? (
              <>
                <SlideReviewPanel
                  key={`${variantName}-${timedSlides[lightbox].key}`}
                  videoId={video.video_id}
                  item={timedSlides[lightbox]}
                  review={variantReviews[timedSlides[lightbox].key]}
                  onSave={(review) =>
                    saveReview(timedSlides[lightbox].key, review)
                  }
                  onSeek={(time) => {
                    setLightbox(null);
                    seekTo(time);
                  }}
                  t={t}
                />
                {reviewStatus && (
                  <p className="resource-note" role="status">
                    {reviewStatus}
                  </p>
                )}
              </>
            ) : (
              <p className="resource-note">{t.estimatedSlides}</p>
            )}
          </div>
        )}
      </Dialog>
    </div>
  );
}
