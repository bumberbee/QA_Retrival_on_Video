from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from .routers import qa, videos  # noqa: E402
from .services import llm  # noqa: E402

app = FastAPI(title="Video Q&A API")

app.add_middleware(
    CORSMiddleware,
    # Any local address/port: localhost, 127.0.0.1 or [::1] (the browser may use any of them).
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(videos.router)
app.include_router(qa.router)


@app.get("/api/health")
async def health():
    return {"status": "ok", **llm.describe()}
