import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import Library from "./components/Library";
import {
  Dialog,
  HelpDialog,
  Icon,
  Logo,
  Spinner,
} from "./components/UI";
import ImportDialog from "./components/ImportDialog";
import { request, videoTitle } from "./lib/api";
import { useMediaQuery, usePreference } from "./lib/hooks";
import { STRINGS } from "./lib/strings";
import { useVideoImport } from "./lib/useVideoImport";

const Workspace = lazy(() => import("./components/Workspace"));
const ReadyForAI = lazy(() => import("./components/ReadyForAI"));

function Sidebar({
  videos,
  selectedVideo,
  page,
  onNavigate,
  onSelect,
  onHelp,
  onClose,
  t,
}) {
  return (
    <div className="sidebar-content">
      <div className="sidebar-brand">
        <Logo onClick={() => onNavigate("all")} />
        {onClose && (
          <button
            className="icon-button"
            onClick={onClose}
            aria-label={t.close}
          >
            <Icon name="close" />
          </button>
        )}
      </div>
      <nav className="main-navigation" aria-label={t.navigation}>
        <button
          className={
            !selectedVideo && page === "library"
              ? "nav-item active"
              : "nav-item"
          }
          onClick={() => onNavigate("all")}
          aria-current={
            !selectedVideo && page === "library" ? "page" : undefined
          }
        >
          <Icon name="library" size={19} />
          <span>{t.library}</span>
          <span className="nav-count">{videos?.length ?? "—"}</span>
        </button>
        <button
          className={
            !selectedVideo && page === "ai" ? "nav-item active" : "nav-item"
          }
          onClick={() => onNavigate("ready")}
          aria-current={!selectedVideo && page === "ai" ? "page" : undefined}
        >
          <Icon name="sparkles" size={19} />
          <span>{t.aiReady}</span>
        </button>
      </nav>
      {Boolean(videos?.length) && <div className="sidebar-recents">
        <p className="sidebar-label">{t.recent}</p>
        {videos?.slice(0, 3).map((video, index) => (
          <button
            key={video.video_id}
            className={`recent-video ${selectedVideo?.video_id === video.video_id ? "active" : ""}`}
            onClick={() => onSelect(video)}
            aria-current={
              selectedVideo?.video_id === video.video_id ? "page" : undefined
            }
            title={video.filename}
          >
            <span className={`recent-video-icon card-tone-${index % 4}`}>
              <Icon name="play" size={15} />
            </span>
            <span>{videoTitle(video.filename)}</span>
            <Icon name="chevron" size={13} />
          </button>
        ))}
      </div>}
      <div className="sidebar-bottom">
        <button className="nav-item" onClick={onHelp}>
          <Icon name="help" size={19} />
          <span>{t.howItWorks}</span>
        </button>
      </div>
    </div>
  );
}

export default function App() {
  const [lang, setLang] = usePreference("lang", "vi", ["vi", "en"]);
  const [theme, setTheme] = usePreference("theme", "dark", ["light", "dark"]);
  const [videoList, setVideoList] = useState(null);
  const [loading, setLoading] = useState(true);
  const [libraryError, setLibraryError] = useState(false);
  const [refreshIndex, setRefreshIndex] = useState(0);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [page, setPage] = useState("library");
  const [workspaceStart, setWorkspaceStart] = useState({
    prompt: "",
    focusChat: false,
  });
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("all");
  const [mobileNav, setMobileNav] = useState(false);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [importMode, setImportMode] = useState("link");
  const [helpOpen, setHelpOpen] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [uploadFilename, setUploadFilename] = useState("");
  const uploadRequest = useRef(null);
  const searchRef = useRef(null);
  const desktop = useMediaQuery("(min-width: 1100px)");
  const t = STRINGS[lang];
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
      ?.setAttribute("content", theme === "dark" ? "#0c0b0a" : "#faf6ef");
  }, [theme, lang]);

  useEffect(() => {
    document.title = `${selectedVideo ? `${videoTitle(selectedVideo.filename)} · ` : page === "ai" ? `${t.aiReady} · ` : ""}Video Q&A`;
  }, [selectedVideo, page, t.aiReady]);

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

  useEffect(() => {
    function onKeyDown(event) {
      if (
        (event.ctrlKey || event.metaKey) &&
        event.key.toLowerCase() === "k" &&
        !document.querySelector("dialog[open]")
      ) {
        event.preventDefault();
        setSelectedVideo(null);
        setPage("library");
        requestAnimationFrame(() => searchRef.current?.focus());
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  useEffect(() => () => uploadRequest.current?.abort(), []);

  const sortedVideos = useMemo(
    () =>
      videoList
        ? [...videoList].sort(
            (a, b) =>
              (Date.parse(b.uploaded_at) || 0) -
              (Date.parse(a.uploaded_at) || 0),
          )
        : null,
    [videoList],
  );
  const currentVideo = selectedVideo
    ? videoList?.find((item) => item.video_id === selectedVideo.video_id) ||
      selectedVideo
    : null;

  function refreshLibrary() {
    setLoading(true);
    setLibraryError(false);
    setRefreshIndex((index) => index + 1);
  }

  function navigate(nextFilter = "all") {
    setSelectedVideo(null);
    setPage(nextFilter === "ready" ? "ai" : "library");
    setFilter(nextFilter === "ready" ? "all" : nextFilter);
    setSearch("");
    setMobileNav(false);
    window.scrollTo({ top: 0, behavior: "instant" });
  }

  function selectVideo(video, options = {}) {
    setSelectedVideo(video);
    setWorkspaceStart({
      prompt: options.prompt || "",
      focusChat: Boolean(options.focusChat),
    });
    setMobileNav(false);
    window.scrollTo({ top: 0, behavior: "instant" });
  }

  function openUpload(mode = "link") {
    setUploadError("");
    if (!addingVideo) {
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
      setUploadOpen(false);
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

  const sidebarProps = {
    videos: sortedVideos,
    selectedVideo,
    page,
    onNavigate: navigate,
    onSelect: selectVideo,
    onHelp: openHelp,
    t,
  };
  const searchLabel = page === "ai" ? t.searchReady : t.search;

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">
        {t.skip}
      </a>
      {desktop && (
        <aside className="sidebar">
          <Sidebar {...sidebarProps} />
        </aside>
      )}
      <Dialog
        open={!desktop && mobileNav}
        onClose={() => setMobileNav(false)}
        title={t.navigation}
        showClose={false}
        className="navigation-dialog"
      >
        <Sidebar {...sidebarProps} onClose={() => setMobileNav(false)} />
      </Dialog>
      <div className="app-content">
        <header className="topbar">
          <div className="topbar-leading">
            {!desktop && (
              <button
                className="icon-button"
                aria-label={t.menu}
                aria-expanded={mobileNav}
                aria-haspopup="dialog"
                onClick={() => setMobileNav(true)}
              >
                <Icon name="menu" />
              </button>
            )}
            <div className="breadcrumb">
              <Icon
                name={
                  selectedVideo
                    ? "play"
                    : page === "ai"
                      ? "sparkles"
                      : "library"
                }
                size={17}
              />
              <span>
                {selectedVideo
                  ? t.workspace
                  : page === "ai"
                    ? t.aiReady
                    : t.library}
              </span>
            </div>
          </div>
          {!selectedVideo && (
            <div className="global-search">
              <Icon name="search" size={18} />
              <input
                ref={searchRef}
                type="search"
                aria-label={searchLabel}
                placeholder={searchLabel}
                value={search}
                onChange={(event) => setSearch(event.target.value)}
              />
              {search ? (
                <button
                  className="search-clear"
                  onClick={() => {
                    setSearch("");
                    searchRef.current?.focus();
                  }}
                  aria-label={t.clearSearch}
                >
                  <Icon name="close" size={14} />
                </button>
              ) : (
                <kbd>Ctrl K</kbd>
              )}
            </div>
          )}
          <div className="topbar-actions">
            <button
              className="language-button"
              onClick={() => setLang(lang === "vi" ? "en" : "vi")}
              aria-label={
                lang === "vi" ? "Switch to English" : "Chuyển sang Tiếng Việt"
              }
              title={
                lang === "vi" ? "Switch to English" : "Chuyển sang Tiếng Việt"
              }
            >
              <Icon name="globe" size={17} />
              <span>{lang.toUpperCase()}</span>
            </button>
            <button
              className="icon-button theme-button"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              aria-label={t.switchTheme}
              title={t.switchTheme}
            >
              <Icon name={theme === "dark" ? "sun" : "moon"} size={19} />
            </button>
            <span className="topbar-divider" />
            <button
              className="button button-primary topbar-upload"
              onClick={openUpload}
              aria-label={addingVideo ? t.viewImport : t.upload}
            >
              {addingVideo ? <Spinner /> : <Icon name="plus" size={17} />}
              <span>{addingVideo ? t.viewImport : t.upload}</span>
            </button>
          </div>
        </header>
        <main
          id="main-content"
          tabIndex={-1}
          className={
            selectedVideo ? "main-content main-workspace" : "main-content"
          }
        >
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
                initialQuestion={workspaceStart.prompt}
                initialChatOpen={workspaceStart.focusChat}
                backLabel={page === "ai" ? t.aiReady : undefined}
                onBack={() => {
                  setSelectedVideo(null);
                  window.scrollTo({ top: 0, behavior: "instant" });
                }}
                lang={lang}
                t={t}
              />
            </Suspense>
          ) : page === "ai" ? (
            <Suspense
              fallback={
                <div className="workspace-loading">
                  <Spinner label={t.loading} />
                </div>
              }
            >
              <ReadyForAI
                videoList={sortedVideos}
                loading={loading}
                error={libraryError}
                search={search}
                onClearSearch={() => setSearch("")}
                onSelect={selectVideo}
                onNavigate={navigate}
                onRefresh={refreshLibrary}
                t={t}
              />
            </Suspense>
          ) : (
            <Library
              videoList={sortedVideos}
              loading={loading}
              error={libraryError}
              search={search}
              filter={filter}
              onFilter={setFilter}
              onReset={() => {
                setSearch("");
                setFilter("all");
              }}
              onSelect={selectVideo}
              onUpload={openUpload}
              onHelp={openHelp}
              onRefresh={refreshLibrary}
              lang={lang}
              t={t}
            />
          )}
        </main>
      </div>
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
        onModeChange={(mode) => { setImportMode(mode); setUploadError(""); }}
        onUpload={uploadFile}
        uploading={uploading}
        filename={uploadFilename}
        error={uploadError}
        videoImport={videoImport}
        onOpenVideo={(video) => { selectVideo(video); setUploadOpen(false); videoImport.clearImport(); }}
        t={t}
      />
      <HelpDialog open={helpOpen} onClose={() => setHelpOpen(false)} t={t} />
    </div>
  );
}
