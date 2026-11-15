"""Byte-range responses for seeking with the project's pinned Starlette version."""

import re
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse

CHUNK_SIZE = 1024 * 1024


def video_response(path: Path, request: Request) -> Response:
    stat = path.stat()
    size = stat.st_size
    full_response = FileResponse(
        path, stat_result=stat, headers={"Accept-Ranges": "bytes"}
    )
    range_header = request.headers.get("range")

    # HTTP ranges apply to GET. FileResponse already handles HEAD without a body.
    if request.method != "GET" or not range_header:
        return full_response

    if_range = request.headers.get("if-range")
    if if_range and if_range not in {
        full_response.headers["etag"],
        full_response.headers["last-modified"],
    }:
        return full_response

    # Browsers request a single byte range. Other units and multipart ranges can
    # safely fall back to the complete representation, as allowed by HTTP.
    if not range_header.lower().startswith("bytes=") or "," in range_header:
        return full_response

    invalid_range = HTTPException(
        status_code=416,
        detail="Requested video range is not satisfiable",
        headers={"Content-Range": f"bytes */{size}", "Accept-Ranges": "bytes"},
    )
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip(), re.IGNORECASE)
    if not match or size == 0:
        raise invalid_range

    first, last = match.groups()
    try:
        if first:
            start = int(first)
            end = min(int(last), size - 1) if last else size - 1
        elif last and int(last) > 0:
            start = max(0, size - int(last))
            end = size - 1
        else:
            raise invalid_range
    except ValueError:
        raise invalid_range from None

    if start >= size or start > end:
        raise invalid_range

    length = end - start + 1
    headers = dict(full_response.headers)
    headers["content-range"] = f"bytes {start}-{end}/{size}"
    headers["content-length"] = str(length)

    def chunks():
        # StreamingResponse runs this iterator in the thread pool. Memory usage
        # is bounded even when the requested range covers a large lecture.
        with path.open("rb") as stream:
            stream.seek(start)
            remaining = length
            while remaining:
                chunk = stream.read(min(CHUNK_SIZE, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(chunks(), status_code=206, headers=headers)
