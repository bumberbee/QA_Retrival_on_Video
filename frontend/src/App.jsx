import { useEffect, useRef, useState } from "react";
import "./App.css";

const API_BASE = "http://localhost:8001/api";

function App() {
  const [video, setVideo] = useState(null); // {video_id, filename, has_transcript}
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [askOpen, setAskOpen] = useState(false);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]); // {role, text}
  const [asking, setAsking] = useState(false);
  const videoRef = useRef(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, asking]);

  async function uploadFile(file) {
    if (!file || !file.type.startsWith("video/")) return;
    setUploading(true);
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch(`${API_BASE}/videos`, { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setVideo(data);
      setMessages([]);
      setAskOpen(false);
    } catch (err) {
      alert("Upload lỗi: " + err.message);
    } finally {
      setUploading(false);
    }
  }

  function handleUpload(e) {
    uploadFile(e.target.files[0]);
    e.target.value = "";
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragOver(false);
    uploadFile(e.dataTransfer.files[0]);
  }

  async function handleAsk() {
    if (!question.trim() || !video) return;
    const q = question.trim();
    setMessages((m) => [...m, { role: "user", text: q }]);
    setQuestion("");
    setAsking(true);
    try {
      const currentTime = videoRef.current?.currentTime ?? null;
      const res = await fetch(`${API_BASE}/qa`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video_id: video.video_id, question: q, current_time: currentTime }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || "Lỗi không xác định");
      }
      const data = await res.json();
      setMessages((m) => [...m, { role: "ai", text: data.answer }]);
    } catch (err) {
      setMessages((m) => [...m, { role: "ai", text: "⚠️ " + err.message }]);
    } finally {
      setAsking(false);
    }
  }

  function resetVideo() {
    setVideo(null);
    setMessages([]);
    setAskOpen(false);
  }

  return (
    <div className="page">
      <header className="topbar">
        <div className="brand">
          <span className="brand-icon">▶</span>
          <span className="brand-name">Video Q&A</span>
        </div>
        {video && (
          <button className="new-video-btn" onClick={resetVideo}>
            ＋ Video khác
          </button>
        )}
      </header>

      <main className="content">
        {!video && (
          <div
            className={`upload-box ${dragOver ? "drag-over" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
          >
            <div className="upload-icon">⤒</div>
            <p className="upload-title">Kéo thả video vào đây</p>
            <p className="upload-sub">hoặc</p>
            <label className="upload-label">
              {uploading ? "Đang tải lên..." : "Chọn video từ máy tính"}
              <input type="file" accept="video/*" onChange={handleUpload} disabled={uploading} hidden />
            </label>
          </div>
        )}

        {video && (
          <div className="video-area">
            <div className="player-wrap">
              <video
                ref={videoRef}
                className="player"
                src={`${API_BASE}/videos/${video.video_id}/file`}
                controls
              />
            </div>

            <div className="action-bar">
              <span className="filename" title={video.filename}>
                {video.filename}
              </span>
              <button
                className={`ask-btn ${askOpen ? "active" : ""}`}
                onClick={() => setAskOpen((v) => !v)}
              >
                <span className="sparkle">✦</span> Đặt câu hỏi
              </button>
              {!video.has_transcript && (
                <span className="badge-warn">⚠ Chưa có transcript</span>
              )}
            </div>

            {askOpen && (
              <div className="ask-panel">
                <div className="ask-panel-header">
                  <span className="sparkle">✦</span> Hỏi về video này
                </div>
                <div className="messages">
                  {messages.length === 0 && (
                    <div className="hint">
                      Hỏi bất cứ điều gì về nội dung video, ví dụ: "tóm tắt đoạn đầu",
                      "phút thứ 2 nói gì?"...
                    </div>
                  )}
                  {messages.map((m, i) => (
                    <div key={i} className={`msg-row ${m.role}`}>
                      <div className={`avatar ${m.role}`}>{m.role === "user" ? "🙂" : "✦"}</div>
                      <div className={`msg ${m.role}`}>{m.text}</div>
                    </div>
                  ))}
                  {asking && (
                    <div className="msg-row ai">
                      <div className="avatar ai">✦</div>
                      <div className="msg ai typing">
                        <span></span>
                        <span></span>
                        <span></span>
                      </div>
                    </div>
                  )}
                  <div ref={messagesEndRef} />
                </div>
                <div className="ask-input-row">
                  <input
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleAsk()}
                    placeholder="Đặt câu hỏi về video..."
                    disabled={asking}
                  />
                  <button onClick={handleAsk} disabled={asking || !question.trim()}>
                    ➤
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
