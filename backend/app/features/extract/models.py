from typing import Literal

from pydantic import BaseModel, Field


class ExtractionStatus(BaseModel):
    video_id: str
    status: Literal["queued", "extracting", "importing", "completed", "failed"]
    stage: str | None = None
    progress: float | None = Field(default=None, ge=0, le=1)   # within the current stage
    overall: float | None = Field(default=None, ge=0, le=1)    # across all stages
    phase: Literal["loading", "working"] | None = None         # loading = the stage's model is not ready yet
    error: str | None = None
    updated_at: str
