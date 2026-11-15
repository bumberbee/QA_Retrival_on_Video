import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from . import storage  # noqa: E402
from .routers import conversations, qa, videos  # noqa: E402
from .services import llm  # noqa: E402

app = FastAPI(title="Video Q&A API")

app.add_middleware(videos.UploadLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    # Any local address/port: localhost, 127.0.0.1 or [::1] (the browser may use any of them),
    # plus the deployed frontend(s) listed in CORS_ORIGINS (comma separated, e.g. the Vercel URL).
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?",
    allow_origins=[origin.strip().rstrip("/") for origin in os.getenv("CORS_ORIGINS", "").split(",")
                   if origin.strip() and origin.strip().rstrip("/") != "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(videos.router)
app.include_router(qa.router)
app.include_router(conversations.router)


@app.exception_handler(storage.InvalidVideoId)
@app.exception_handler(storage.UnsafeStoragePath)
async def invalid_storage_path(request: Request, error: ValueError):
    return JSONResponse({"detail": str(error)}, status_code=400)


@app.exception_handler(storage.CloudWriteUnsupported)
async def cloud_write_unsupported(request: Request, error: storage.CloudWriteUnsupported):
    return JSONResponse({"detail": str(error)}, status_code=403)


@app.get("/api/health")
async def health():
    return {"status": "ok", **llm.describe()}
