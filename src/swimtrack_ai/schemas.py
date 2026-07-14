from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DiagnosticsLevel = Literal["none", "counts", "boxes"]


class CreateSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fps: float = Field(default=60.0, gt=0, le=240)
    lap_calibration_id: Literal["fixed-camera-v1"] | None = None
    diagnostics: DiagnosticsLevel = "none"


class TrackingConfiguration(BaseModel):
    diagnostic_score_floor: float
    score_threshold: float
    min_box_area: float
    track_threshold: float
    track_buffer: int
    match_threshold: float
    mot20: bool
    lane_roi_enabled: bool
    lane_ids: list[str]
    effective_lost_buffer_frames: int
    effective_lost_buffer_seconds: float


class SessionCreated(BaseModel):
    session_id: str
    next_sequence: int
    expires_in_seconds: int
    tracking_configuration: TrackingConfiguration | None = None


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
    lane_id: str | None = None
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    class_id: int = 0


class DiagnosticBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = Field(ge=0, le=1)


class DiagnosticStage(BaseModel):
    count: int = Field(ge=0)
    boxes: list[DiagnosticBox] | None = None


class LaneTrackingDiagnostics(BaseModel):
    lane_id: str
    after_roi: DiagnosticStage
    active_track_ids: list[int]
    retained_lost_track_count: int = Field(ge=0)


class FrameTrackingDiagnostics(BaseModel):
    diagnostic_floor: float = Field(ge=0, le=1)
    person_candidates: DiagnosticStage
    detector_accepted: DiagnosticStage
    lanes: list[LaneTrackingDiagnostics]


class LapEvidence(BaseModel):
    wall: float = Field(ge=0, le=1)
    approach: float = Field(ge=0, le=1)
    reversal: float = Field(ge=0, le=1)
    departure: float = Field(ge=0, le=1)
    track_quality: float = Field(ge=0, le=1)


class LaneLapScore(BaseModel):
    lane_id: str
    track_id: int | None = None
    lap_score: float = Field(ge=0, le=1)
    no_lap_score: float | None = Field(default=None, ge=0, le=1)
    observation_quality: float = Field(ge=0, le=1)
    evaluable: bool
    longitudinal_position: float | None = Field(default=None, ge=0, le=1)
    endpoint: Literal["far", "near"] | None = None
    candidate_time_ms: float | None = Field(default=None, ge=0)
    candidate_episode_id: int | None = Field(default=None, ge=1)
    window_start_ms: float = Field(ge=0)
    window_end_ms: float = Field(ge=0)
    score_version: str
    evidence: LapEvidence


class FrameResult(BaseModel):
    frame_index: int
    time_ms: float
    width: int
    height: int
    boxes: list[BoundingBox]
    lap_scores: list[LaneLapScore] | None = None
    tracking_diagnostics: FrameTrackingDiagnostics | None = None


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
