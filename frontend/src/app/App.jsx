import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import VideoRail from "../features/library/VideoRail";
import Splitter, { useStoredSize } from "../shared/components/Splitter";
import { Dialog, HelpDialog, Icon, Spinner } from "../shared/components/UI";
import ImportDialog from "../features/import/ImportDialog";
import { request, videoReady, videoTitle } from "../shared/lib/api";
import { useMediaQuery, usePreference } from "../shared/lib/hooks";
import { STRINGS } from "../shared/lib/strings";
import { useVideoImport } from "../features/import/useVideoImport";
import { useExtraction } from "../features/extract/useExtraction";

const Workspace = lazy(() => import("../features/chat/Workspace"));

export default function App() {
  const [lang, setLang] = usePreference("lang", "vi", ["vi", "en"]);
  const [theme, setTheme] = usePreference("theme", "dark", ["light", "dark"]);
  const [videoList, setVideoList] = useState(null);
  const [loading, setLoading] = useState(true);
  const [libraryError, setLibraryError] = useState(false);
  const [refreshIndex, setRefreshIndex] = useState(0);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [workspaceStart, setWorkspaceStart] = useState({
    prompt: "",
    focusChat: false,
  });
  const [mobileNav, setMobileNav] = useState(false);
  const [checkedIds, setCheckedIds] = useState([]);   // videos to ask across; [] = the one being watched
  const [messages, setMessages] = useState([]);
  const [conversationId, setConversationId] = useState(null);
  const [startTime, setStartTime] = useState(0);
  const [railWidth, setRailWidth, resetRailWidth] = useStoredSize("rail", 290, 220, 520);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [importMode, setImportMode] = useState("link");
  const [helpOpen, setHelpOpen] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [uploadFilename, setUploadFilename] = useState("");
  const [uploadedVideo, setUploadedVideo] = useState(null);
  const uploadRequest = useRef(null);
  const desktop = useMediaQuery("(min-width: 1100px)");
  const t = STRINGS[lang];
  const { status: extractionStatus, start: startExtraction } = useExtraction(videoList, () => refreshLibrary());
  const videoImport = useVideoImport({
    onComplete: (video) => {
      setVideoList((videos) => [video, ...(videos || []).filter((item) => item.video_id !== video.video_id)]);
      refreshLibrary();
    },
    t,
  });
  const addingVideo = uploading || videoImport.pending || videoImport.submitting;

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.documentElement.lang = lang;
    document
      .querySelector('meta[name="theme-color"]')
      ?.setAttribute("content", theme === "dark" ? "#0b1220" : "#f6f8fb");
  }, [theme, lang]);

  useEffect(() => {
    document.title = `${selectedVideo ? `${videoTitle(selectedVideo.filename)} · ` : ""}Video Q&A`;
  }, [selectedVideo]);

  useEffect(() => {
    const controller = new AbortController();
    request("/videos", { signal: controller.signal })
      .then((videos) => {
        if (!controller.signal.aborted) {
          if (!Array.isArray(videos)) throw new Error("Invalid video list");
          setVideoList(videos);
          setLibraryError(false);
        }
      })
      .catch(() => {
        if (!controller.signal.aborted) setLibraryError(true);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [refreshIndex]);

  useEffect(() => () => uploadRequest.current?.abort(), []);

  const sortedVideos = useMemo(
    () =>
      videoList
        ? videoList.map((video) => ({ ...video, extraction: extractionStatus(video.video_id) })).sort(
            (a, b) =>
              (Date.parse(b.uploaded_at) || 0) -
              (Date.parse(a.uploaded_at) || 0),
          )
        : null,
    [videoList, extractionStatus],
  );
  useEffect(() => {
    if (!selectedVideo && sortedVideos?.length)
      setSelectedVideo(sortedVideos.find(videoReady) || sortedVideos[0]);
  }, [selectedVideo, sortedVideos]);
  const currentVideo = selectedVideo
    ? sortedVideos?.find((item) => item.video_id === selectedVideo.video_id) ||
      { ...selectedVideo, extraction: extractionStatus(selectedVideo.video_id) }
    : null;

  function refreshLibrary() {
    setLoading(true);
    setLibraryError(false);
    setRefreshIndex((index) => index + 1);
  }

  function selectVideo(video, options = {}) {
    setSelectedVideo(video);
    setStartTime(options.time || 0);
    setWorkspaceStart({
      prompt: options.prompt || "",
      focusChat: Boolean(options.focusChat),
    });
    setMobileNav(false);
    window.scrollTo({ top: 0, behavior: "instant" });
  }

  // Reopen a saved chat: its turns become messages again, with the videos it was asked across.
  async function openConversation(id) {
    try {
      const conversation = await request(`/conversations/${id}`);
      setMessages(
        conversation.turns.flatMap((turn, index) => [
          { id: `${id}-q${index}`, role: "user", text: turn.question },
          { id: `${id}-a${index}`, role: "ai", text: turn.answer, sources: turn.sources, rewritten: turn.rewritten },
        ]),
      );
      setConversationId(conversation.id);
      const ids = conversation.video_ids || [];
      setCheckedIds(ids.length > 1 ? ids : []);
      const target = videoList?.find((item) => item.video_id === conversation.video_id);
      if (target && target.video_id !== currentVideo?.video_id) selectVideo(target);
    } catch {
      // the list shows the error when it reloads
    }
  }

  function openUpload(mode = "link") {
    setUploadError("");
    if (!addingVideo) {
      setUploadedVideo(null);
      setImportMode(mode === "file" ? "file" : "link");
      if (mode === "file") videoImport.clearImport();
    }
    setUploadOpen(true);
    setMobileNav(false);
  }

  function openHelp() {
    setMobileNav(false);
    setHelpOpen(true);
  }

  async function uploadFile(file) {
    if (!file || uploadRequest.current || videoImport.pending || videoImport.submitting) return;
    if (!/\.(mp4|webm|mov|mkv)$/i.test(file.name)) {
      setUploadError(t.invalidFile);
      return;
    }
    const controller = new AbortController();
    uploadRequest.current = controller;
    setUploading(true);
    setUploadError("");
    setUploadFilename(file.name);
    setUploadedVideo(null);
    const form = new FormData();
    form.append("file", file);
    try {
      const video = await request("/videos", {
        method: "POST",
        body: form,
        signal: controller.signal,
      });
      setVideoList((videos) => [
        video,
        ...(videos || []).filter((item) => item.video_id !== video.video_id),
      ]);
      selectVideo(video);
      setUploadedVideo(video);
      refreshLibrary();
    } catch (error) {
      if (!controller.signal.aborted) {
        setUploadError(
          error instanceof TypeError ? t.uploadFailed : error.message,
        );
        setUploadOpen(true);
      }
    } finally {
      uploadRequest.current = null;
      setUploading(false);
    }
  }

  const railProps = {
    videos: sortedVideos,
    loading,
    error: libraryError,
    selectedId: currentVideo?.video_id,
    onSelect: selectVideo,
    checkedIds,
    onToggle: (id) =>
      setCheckedIds((ids) => {
        const current = currentVideo?.video_id;
        const scope = ids.length ? ids : [current];
        const next = scope.includes(id) ? scope.filter((item) => item !== id) : [...scope, id];
        if (!next.length) return ids;   // keep at least one video to ask
        return next.length === 1 && next[0] === current ? [] : next;
      }),
    onClearChecked: () => setCheckedIds([]),
    onExtract: startExtraction,
    onUpload: () => openUpload(),
    onRefresh: refreshLibrary,
    addingVideo,
    lang,
    onLang: () => setLang(lang === "vi" ? "en" : "vi"),
    theme,
    onTheme: () => setTheme(theme === "dark" ? "light" : "dark"),
    onHelp: openHelp,
    t,
  };

  return (
    <div
      className="app-shell one-page"
      style={desktop ? { gridTemplateColumns: `${railWidth}px 5px minmax(0, 1fr)` } : undefined}
    >
      <a className="skip-link" href="#main-content">
        {t.skip}
      </a>
      {desktop ? (
        <aside className="rail-column">
          <VideoRail {...railProps} />
        </aside>
      ) : null}
      {desktop ? (
        <Splitter size={railWidth} onResize={setRailWidth} onReset={resetRailWidth} label={t.resizePanel} />
      ) : (
        <>
          <header className="topbar mobile-topbar">
            <button
              className="icon-button"
              aria-label={t.menu}
              aria-expanded={mobileNav}
              aria-haspopup="dialog"
              onClick={() => setMobileNav(true)}
            >
              <Icon name="menu" />
            </button>
            <b className="mobile-title">{currentVideo ? videoTitle(currentVideo.filename) : "Multimodal Lecture RAG"}</b>
          </header>
          <Dialog
            open={mobileNav}
            onClose={() => setMobileNav(false)}
            title={t.navigation}
            showClose={false}
            className="navigation-dialog"
          >
            <VideoRail {...railProps} onClose={() => setMobileNav(false)} />
          </Dialog>
        </>
      )}
      <main id="main-content" tabIndex={-1} className="one-page-main">
        {currentVideo ? (
          <Suspense
            fallback={
              <div className="workspace-loading">
                <Spinner label={t.loading} />
              </div>
            }
          >
            <Workspace
              key={currentVideo.video_id}
              video={currentVideo}
              ready={videoReady(currentVideo) && checkedIds.every((id) => videoReady(sortedVideos?.find((video) => video.video_id === id)))}
              onExtract={() => startExtraction(currentVideo.video_id)}
              initialQuestion={workspaceStart.prompt}
              initialChatOpen={workspaceStart.focusChat}
              initialTime={startTime}
              askVideoIds={checkedIds}
              chat={{
                messages,
                setMessages,
                conversationId,
                onConversation: setConversationId,
                history: {
                  currentId: conversationId,
                  onNew: () => {
                    setMessages([]);
                    setConversationId(null);
                  },
                  onOpen: openConversation,
                },
              }}
              onOpenSource={(videoId, time) => {
                const target = videoList?.find((item) => item.video_id === videoId);
                if (target) selectVideo(target, { time });
              }}
              lang={lang}
              t={t}
            />
          </Suspense>
        ) : (
          <div className="workspace-loading">
            {loading ? <Spinner label={t.loading} /> : <p>{libraryError ? t.libraryError : t.noResults}</p>}
          </div>
        )}
      </main>
      {!uploadOpen && videoImport.job && (
        <div className="import-status-toast" role="status">
          <button onClick={() => { setImportMode("link"); setUploadOpen(true); }}>
            {videoImport.pending ? <Spinner /> : <Icon name={videoImport.job.status === "completed" ? "checkCircle" : "warning"} size={20} />}
            <span>{videoImport.pending ? t.importing : videoImport.job.status === "completed" ? t.importComplete : t.importErrorTitle}</span>
            <Icon name="chevron" size={16} />
          </button>
          {!videoImport.pending && (
            <button className="icon-button" aria-label={t.close} onClick={videoImport.clearImport}>
              <Icon name="close" size={17} />
            </button>
          )}
        </div>
      )}
      <ImportDialog
        open={uploadOpen}
        onClose={() => setUploadOpen(false)}
        mode={importMode}
        onModeChange={(mode) => { setImportMode(mode); setUploadError(""); setUploadedVideo(null); }}
        onUpload={uploadFile}
        uploading={uploading}
        filename={uploadFilename}
        error={uploadError}
        videoImport={videoImport}
        fileVideo={uploadedVideo}
        extractionStatus={extractionStatus}
        onExtract={startExtraction}
        onOpenVideo={(video) => { selectVideo(video); setUploadOpen(false); videoImport.clearImport(); }}
        t={t}
      />
      <HelpDialog open={helpOpen} onClose={() => setHelpOpen(false)} t={t} />
    </div>
  );
}
