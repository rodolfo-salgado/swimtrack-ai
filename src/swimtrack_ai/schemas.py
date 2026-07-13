from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CreateSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fps: float = Field(default=60.0, gt=0, le=240)


class SessionCreated(BaseModel):
    session_id: str
    next_sequence: int
    expires_in_seconds: int


class FrameMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_index: int = Field(ge=0)
    time_ms: float = Field(ge=0)
    original_width: int = Field(gt=0, le=16384)
    original_height: int = Field(gt=0, le=16384)


class BatchMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
    sequence: int = Field(ge=0)
    frames: list[FrameMetadata] = Field(min_length=1)

    @model_validator(mode="after")
    def frames_are_strictly_ordered(self) -> BatchMetadata:
        indices = [frame.frame_index for frame in self.frames]
        times = [frame.time_ms for frame in self.frames]
        if any(current <= previous for previous, current in zip(indices, indices[1:])):
            raise ValueError("frame_index values must be strictly increasing")
        if any(current < previous for previous, current in zip(times, times[1:])):
            raise ValueError("time_ms values must be non-decreasing")
        return self


class BoundingBox(BaseModel):
    id: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    class_id: int = 0


class FrameResult(BaseModel):
    frame_index: int
    time_ms: float
    width: int
    height: int
    boxes: list[BoundingBox]


class BatchResult(BaseModel):
    session_id: str
    batch_id: str
    sequence: int
    next_sequence: int
    frames: list[FrameResult]


class HealthResponse(BaseModel):
    status: str
    backend: str
    detail: str | None = None
