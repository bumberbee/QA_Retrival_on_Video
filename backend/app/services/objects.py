"""Cloudflare R2 (S3 API) for video and keyframe files.

Enabled when R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY and R2_BUCKET are set; the
browser plays files from short-lived signed URLs, so they never pass through the backend.
Keys: videos/<video_id>/<file name>, keyframes/<video_id>/<image name>.
"""

from __future__ import annotations

import mimetypes
import os
from functools import lru_cache
from pathlib import Path

SIGNED_URL_SECONDS = 6 * 3600


def enabled() -> bool:
    return all(os.getenv(name) for name in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"))


def bucket() -> str:
    return os.environ["R2_BUCKET"]


@lru_cache(maxsize=1)
def client():
    import boto3
    from botocore.config import Config

    # accept the bare account id or the whole S3 endpoint the dashboard shows
    account = os.environ["R2_ACCOUNT_ID"].strip().removeprefix("https://").split(".")[0].strip("/")
    endpoint = os.getenv("R2_ENDPOINT") or f"https://{account}.r2.cloudflarestorage.com"
    return boto3.client(
        "s3", endpoint_url=endpoint, region_name=os.getenv("R2_REGION", "auto"),
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4", retries={"max_attempts": 3}),
    )


def video_key(video_id: str, filename: str) -> str:
    return f"videos/{video_id}/{Path(filename).name}"


def keyframe_key(video_id: str, name: str) -> str:
    return f"keyframes/{video_id}/{Path(name).name}"


def upload_file(path: Path, key: str, content_type: str | None = None) -> str:
    content_type = content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    client().upload_file(str(path), bucket(), key, ExtraArgs={"ContentType": content_type})
    return key


def exists(key: str) -> bool:
    from botocore.exceptions import ClientError

    try:
        client().head_object(Bucket=bucket(), Key=key)
        return True
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def signed_url(key: str, seconds: int = SIGNED_URL_SECONDS) -> str:
    return client().generate_presigned_url(
        "get_object", Params={"Bucket": bucket(), "Key": key}, ExpiresIn=seconds)


def delete_prefix(prefix: str) -> int:
    """Delete every object under prefix (a video's files); returns how many were removed."""
    removed = 0
    paginator = client().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket(), Prefix=prefix):
        keys = [{"Key": item["Key"]} for item in page.get("Contents", [])]
        if keys:
            client().delete_objects(Bucket=bucket(), Delete={"Objects": keys})
            removed += len(keys)
    return removed
