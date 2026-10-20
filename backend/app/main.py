import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from .routers import conversations, qa, videos  # noqa: E402
from .services import llm  # noqa: E402

app = FastAPI(title="Video Q&A API")

app.add_middleware(
    CORSMiddleware,
    # Any local address/port: localhost, 127.0.0.1 or [::1] (the browser may use any of them),
    # plus the deployed frontend(s) listed in CORS_ORIGINS (comma separated, e.g. the Vercel URL).
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?",
    allow_origins=[origin.strip().rstrip("/") for origin in os.getenv("CORS_ORIGINS", "").split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(videos.router)
app.include_router(qa.router)
app.include_router(conversations.router)


@app.get("/api/health")
async def health():
    return {"status": "ok", **llm.describe()}
