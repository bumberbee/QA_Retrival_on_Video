"""Supabase Postgres (with pgvector) for videos, slides, transcripts and the search index.

Enabled when SUPABASE_DB_URL is set (Supabase: Project Settings -> Database -> connection string,
"Session pooler" works from IPv4 machines such as Vast). Without it the app keeps the file storage.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager

SCHEMA_TABLES = ("videos", "segments", "transcript_segments", "chunks", "slide_reviews", "jobs", "app_config")
_pool = None
_lock = threading.Lock()


def enabled() -> bool:
    return bool(os.getenv("SUPABASE_DB_URL"))


def pool():
    """One small connection pool per process (the backend is a single uvicorn worker)."""
    global _pool
    with _lock:
        if _pool is None:
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool

            _pool = ConnectionPool(
                os.environ["SUPABASE_DB_URL"], min_size=1, max_size=int(os.getenv("DB_POOL_SIZE", "5")),
                kwargs={"row_factory": dict_row, "autocommit": True, "prepare_threshold": None},
                open=True,
            )
        return _pool


@contextmanager
def connection():
    with pool().connection() as conn:
        yield conn


def vector_literal(values) -> str:
    """pgvector text form '[0.1,0.2,...]' (no numpy adapter needed)."""
    return "[" + ",".join(f"{float(v):.6f}" for v in values) + "]"


def check_schema() -> list[str]:
    """Tables of deploy/supabase/schema.sql that are missing (empty list = ready)."""
    with connection() as conn:
        rows = conn.execute(
            "select table_name from information_schema.tables where table_schema = 'public'").fetchall()
    present = {row["table_name"] for row in rows}
    return [name for name in SCHEMA_TABLES if name not in present]


def set_config(key: str, value: str) -> None:
    with connection() as conn:
        conn.execute(
            "insert into app_config (key, value, updated_at) values (%s, %s, now()) "
            "on conflict (key) do update set value = excluded.value, updated_at = now()", (key, value))


def get_config(key: str) -> str | None:
    with connection() as conn:
        row = conn.execute("select value from app_config where key = %s", (key,)).fetchone()
    return row["value"] if row else None
