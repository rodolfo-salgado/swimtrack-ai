from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.getenv(f"SWIMTRACK_{name}", default)


def _integer(name: str, default: int) -> int:
    value = int(_env(name, str(default)))
    if value <= 0:
        raise ValueError(f"SWIMTRACK_{name} must be greater than zero")
    return value


def _floating(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _boolean(name: str, default: bool) -> bool:
    return _env(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    backend: str = "tensorrt"
    auth_token: str = ""
    model_source_dir: Path = Path("/model-source")
    model_cache_dir: Path = Path("/model-cache")
    onnx_filename: str = "rtdetrv2_s.onnx"
    engine_filename: str = "rtdetrv2_s_fp16.engine"
    bytetrack_root: Path = Path("/app/vendor/ByteTrack")
    device: str = "cuda:0"
    input_width: int = 640
    input_height: int = 640
    diagnostic_score_floor: float = 0.05
    score_threshold: float = 0.15
    person_label: int = 0
    max_detections: int = 20
    min_box_area: float = 250.0
    track_threshold: float = 0.45
    track_buffer: int = 60
    match_threshold: float = 0.80
    mot20: bool = False
    lane_roi_enabled: bool = True
    weak_reactivation_enabled: bool = True
    weak_reactivation_score_threshold: float = 0.10
    weak_reactivation_min_box_area: float = 64.0
    weak_reactivation_max_gap_seconds: float = 1.0
    weak_reactivation_max_center_distance: float = 0.10
    far_crop_enabled: bool = False
    far_crop_left: float = 0.2962962963
    far_crop_top: float = 0.1111111111
    far_crop_right: float = 0.7037037037
    far_crop_bottom: float = 0.5185185185
    far_crop_nms_threshold: float = 0.50
    identity_confirmation_observations: int = 3
    identity_confirmation_seconds: float = 0.20
    identity_confirmation_confidence: float = 0.18
    identity_tentative_max_gap_seconds: float = 0.75
    identity_max_reassociation_gap_seconds: float = 12.0
    identity_max_speed_per_second: float = 0.07
    identity_position_slack: float = 0.08
    identity_max_lane_x_delta: float = 0.22
    identity_duplicate_iou: float = 0.45
    identity_duplicate_position_delta: float = 0.08
    identity_duplicate_lane_x_delta: float = 0.15
    identity_additional_confirmation_observations: int = 8
    identity_additional_confirmation_seconds: float = 0.50
    identity_additional_confirmation_confidence: float = 0.30
    identity_additional_min_position_span: float = 0.15
    identity_additional_cooccurrence_max_gap_seconds: float = 0.25
    identity_max_per_lane: int = 2
    max_batch_frames: int = 8
    max_frame_bytes: int = 2_000_000
    max_batch_bytes: int = 12_000_000
    max_request_bytes: int = 16_000_000
    max_decoded_pixels: int = 4_194_304
    max_metadata_bytes: int = 65_536
    max_video_bytes: int = 1_073_741_824
    video_decode_batch_frames: int = 4
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    video_probe_timeout_seconds: int = 30
    max_sessions: int = 128
    session_ttl_seconds: int = 900
    idempotency_cache_size: int = 32
    cleanup_interval_seconds: int = 30
    decode_workers: int = 4
    preprocess_workers: int = 4
    trt_workspace_gb: float = 4.0
    trt_fp16: bool = True
    trt_opt_batch_size: int = 4
    trt_max_batch_size: int = 8

    def __post_init__(self) -> None:
        if not 0.0 <= self.diagnostic_score_floor <= self.score_threshold <= 1.0:
            raise ValueError("diagnostic_score_floor and score_threshold must satisfy 0 <= floor <= threshold <= 1")
        if self.min_box_area < 0:
            raise ValueError("min_box_area must not be negative")
        if not 0.0 <= self.track_threshold <= 1.0:
            raise ValueError("track_threshold must be between zero and one")
        if not 0.0 <= self.match_threshold <= 1.0:
            raise ValueError("match_threshold must be between zero and one")
        weak_reactivation_score_ceiling = min(self.score_threshold, self.track_threshold)
        if not self.diagnostic_score_floor <= self.weak_reactivation_score_threshold < weak_reactivation_score_ceiling:
            raise ValueError(
                "weak_reactivation_score_threshold must be at least diagnostic_score_floor "
                "and below ordinary thresholds"
            )
        if self.weak_reactivation_min_box_area < 0:
            raise ValueError("weak_reactivation_min_box_area must not be negative")
        if self.weak_reactivation_max_gap_seconds <= 0:
            raise ValueError("weak_reactivation_max_gap_seconds must be greater than zero")
        if not 0.0 < self.weak_reactivation_max_center_distance <= 1.0:
            raise ValueError("weak_reactivation_max_center_distance must be in (0, 1]")
        if not (
            0.0 <= self.far_crop_left < self.far_crop_right <= 1.0
            and 0.0 <= self.far_crop_top < self.far_crop_bottom <= 1.0
        ):
            raise ValueError("far crop coordinates must define a non-empty normalized rectangle")
        if not 0.0 <= self.far_crop_nms_threshold <= 1.0:
            raise ValueError("far_crop_nms_threshold must be between zero and one")
        if self.identity_confirmation_observations < 1:
            raise ValueError("identity_confirmation_observations must be at least one")
        if self.identity_confirmation_seconds < 0:
            raise ValueError("identity_confirmation_seconds must not be negative")
        if not 0.0 <= self.identity_confirmation_confidence <= 1.0:
            raise ValueError("identity_confirmation_confidence must be between zero and one")
        if self.identity_tentative_max_gap_seconds <= 0:
            raise ValueError("identity_tentative_max_gap_seconds must be greater than zero")
        if self.identity_max_reassociation_gap_seconds <= 0:
            raise ValueError("identity_max_reassociation_gap_seconds must be greater than zero")
        if self.identity_max_speed_per_second <= 0:
            raise ValueError("identity_max_speed_per_second must be greater than zero")
        if not 0.0 < self.identity_position_slack <= 1.0:
            raise ValueError("identity_position_slack must be in (0, 1]")
        if not 0.0 < self.identity_max_lane_x_delta <= 1.0:
            raise ValueError("identity_max_lane_x_delta must be in (0, 1]")
        if not 0.0 <= self.identity_duplicate_iou <= 1.0:
            raise ValueError("identity_duplicate_iou must be between zero and one")
        if not 0.0 < self.identity_duplicate_position_delta <= 1.0:
            raise ValueError("identity_duplicate_position_delta must be in (0, 1]")
        if not 0.0 < self.identity_duplicate_lane_x_delta <= 1.0:
            raise ValueError("identity_duplicate_lane_x_delta must be in (0, 1]")
        if self.identity_additional_confirmation_observations < 1:
            raise ValueError("identity_additional_confirmation_observations must be at least one")
        if self.identity_additional_confirmation_seconds < 0:
            raise ValueError("identity_additional_confirmation_seconds must not be negative")
        if not 0.0 <= self.identity_additional_confirmation_confidence <= 1.0:
            raise ValueError("identity_additional_confirmation_confidence must be between zero and one")
        if not 0.0 < self.identity_additional_min_position_span <= 1.0:
            raise ValueError("identity_additional_min_position_span must be in (0, 1]")
        if self.identity_additional_cooccurrence_max_gap_seconds <= 0:
            raise ValueError("identity_additional_cooccurrence_max_gap_seconds must be greater than zero")
        if self.identity_max_per_lane < 1:
            raise ValueError("identity_max_per_lane must be at least one")
        if self.trt_opt_batch_size > self.trt_max_batch_size:
            raise ValueError("trt_opt_batch_size must not exceed trt_max_batch_size")
        if self.video_decode_batch_frames > self.trt_max_batch_size:
            raise ValueError("video_decode_batch_frames must not exceed trt_max_batch_size")
        if not self.ffmpeg_path.strip() or not self.ffprobe_path.strip():
            raise ValueError("ffmpeg_path and ffprobe_path must not be empty")

    @classmethod
    def from_env(cls) -> Settings:
        backend = _env("BACKEND", "tensorrt").strip().lower()
        if backend not in {"tensorrt", "fake"}:
            raise ValueError("SWIMTRACK_BACKEND must be 'tensorrt' or 'fake'")
        return cls(
            backend=backend,
            auth_token=_env("AUTH_TOKEN", ""),
            model_source_dir=Path(_env("MODEL_SOURCE_DIR", "/model-source")),
            model_cache_dir=Path(_env("MODEL_CACHE_DIR", "/model-cache")),
            onnx_filename=_env("ONNX_FILENAME", "rtdetrv2_s.onnx"),
            engine_filename=_env("ENGINE_FILENAME", "rtdetrv2_s_fp16.engine"),
            bytetrack_root=Path(_env("BYTETRACK_ROOT", "/app/vendor/ByteTrack")),
            device=_env("DEVICE", "cuda:0"),
            input_width=_integer("INPUT_WIDTH", 640),
            input_height=_integer("INPUT_HEIGHT", 640),
            diagnostic_score_floor=_floating("DIAGNOSTIC_SCORE_FLOOR", 0.05),
            score_threshold=_floating("SCORE_THRESHOLD", 0.15),
            person_label=int(_env("PERSON_LABEL", "0")),
            max_detections=_integer("MAX_DETECTIONS", 20),
            min_box_area=_floating("MIN_BOX_AREA", 250.0),
            track_threshold=_floating("TRACK_THRESHOLD", 0.45),
            track_buffer=_integer("TRACK_BUFFER", 60),
            match_threshold=_floating("MATCH_THRESHOLD", 0.80),
            mot20=_boolean("MOT20", False),
            lane_roi_enabled=_boolean("LANE_ROI_ENABLED", True),
            weak_reactivation_enabled=_boolean("WEAK_REACTIVATION_ENABLED", True),
            weak_reactivation_score_threshold=_floating("WEAK_REACTIVATION_SCORE_THRESHOLD", 0.10),
            weak_reactivation_min_box_area=_floating("WEAK_REACTIVATION_MIN_BOX_AREA", 64.0),
            weak_reactivation_max_gap_seconds=_floating("WEAK_REACTIVATION_MAX_GAP_SECONDS", 1.0),
            weak_reactivation_max_center_distance=_floating("WEAK_REACTIVATION_MAX_CENTER_DISTANCE", 0.10),
            far_crop_enabled=_boolean("FAR_CROP_ENABLED", False),
            far_crop_left=_floating("FAR_CROP_LEFT", 0.2962962963),
            far_crop_top=_floating("FAR_CROP_TOP", 0.1111111111),
            far_crop_right=_floating("FAR_CROP_RIGHT", 0.7037037037),
            far_crop_bottom=_floating("FAR_CROP_BOTTOM", 0.5185185185),
            far_crop_nms_threshold=_floating("FAR_CROP_NMS_THRESHOLD", 0.50),
            identity_confirmation_observations=_integer("IDENTITY_CONFIRMATION_OBSERVATIONS", 3),
            identity_confirmation_seconds=_floating("IDENTITY_CONFIRMATION_SECONDS", 0.20),
            identity_confirmation_confidence=_floating("IDENTITY_CONFIRMATION_CONFIDENCE", 0.18),
            identity_tentative_max_gap_seconds=_floating("IDENTITY_TENTATIVE_MAX_GAP_SECONDS", 0.75),
            identity_max_reassociation_gap_seconds=_floating("IDENTITY_MAX_REASSOCIATION_GAP_SECONDS", 12.0),
            identity_max_speed_per_second=_floating("IDENTITY_MAX_SPEED_PER_SECOND", 0.07),
            identity_position_slack=_floating("IDENTITY_POSITION_SLACK", 0.08),
            identity_max_lane_x_delta=_floating("IDENTITY_MAX_LANE_X_DELTA", 0.22),
            identity_duplicate_iou=_floating("IDENTITY_DUPLICATE_IOU", 0.45),
            identity_duplicate_position_delta=_floating("IDENTITY_DUPLICATE_POSITION_DELTA", 0.08),
            identity_duplicate_lane_x_delta=_floating("IDENTITY_DUPLICATE_LANE_X_DELTA", 0.15),
            identity_additional_confirmation_observations=_integer("IDENTITY_ADDITIONAL_CONFIRMATION_OBSERVATIONS", 8),
            identity_additional_confirmation_seconds=_floating("IDENTITY_ADDITIONAL_CONFIRMATION_SECONDS", 0.50),
            identity_additional_confirmation_confidence=_floating("IDENTITY_ADDITIONAL_CONFIRMATION_CONFIDENCE", 0.30),
            identity_additional_min_position_span=_floating("IDENTITY_ADDITIONAL_MIN_POSITION_SPAN", 0.15),
            identity_additional_cooccurrence_max_gap_seconds=_floating(
                "IDENTITY_ADDITIONAL_COOCCURRENCE_MAX_GAP_SECONDS", 0.25
            ),
            identity_max_per_lane=_integer("IDENTITY_MAX_PER_LANE", 2),
            max_batch_frames=_integer("MAX_BATCH_FRAMES", 8),
            max_frame_bytes=_integer("MAX_FRAME_BYTES", 2_000_000),
            max_batch_bytes=_integer("MAX_BATCH_BYTES", 12_000_000),
            max_request_bytes=_integer("MAX_REQUEST_BYTES", 16_000_000),
            max_decoded_pixels=_integer("MAX_DECODED_PIXELS", 4_194_304),
            max_metadata_bytes=_integer("MAX_METADATA_BYTES", 65_536),
            max_video_bytes=_integer("MAX_VIDEO_BYTES", 1_073_741_824),
            video_decode_batch_frames=_integer("VIDEO_DECODE_BATCH_FRAMES", 4),
            ffmpeg_path=_env("FFMPEG_PATH", "ffmpeg"),
            ffprobe_path=_env("FFPROBE_PATH", "ffprobe"),
            video_probe_timeout_seconds=_integer("VIDEO_PROBE_TIMEOUT_SECONDS", 30),
            max_sessions=_integer("MAX_SESSIONS", 128),
            session_ttl_seconds=_integer("SESSION_TTL_SECONDS", 900),
            idempotency_cache_size=_integer("IDEMPOTENCY_CACHE_SIZE", 32),
            cleanup_interval_seconds=_integer("CLEANUP_INTERVAL_SECONDS", 30),
            decode_workers=_integer("DECODE_WORKERS", 4),
            preprocess_workers=_integer("PREPROCESS_WORKERS", 4),
            trt_workspace_gb=_floating("TRT_WORKSPACE_GB", 4.0),
            trt_fp16=_boolean("TRT_FP16", True),
            trt_opt_batch_size=_integer("TRT_OPT_BATCH_SIZE", 4),
            trt_max_batch_size=_integer("TRT_MAX_BATCH_SIZE", 8),
        )

    @property
    def onnx_path(self) -> Path:
        return self.model_source_dir / self.onnx_filename

    @property
    def engine_path(self) -> Path:
        return self.model_cache_dir / self.engine_filename

    @property
    def engine_manifest_path(self) -> Path:
        return self.engine_path.with_suffix(self.engine_path.suffix + ".json")

    @property
    def far_crop_box(self) -> tuple[float, float, float, float]:
        return (
            self.far_crop_left,
            self.far_crop_top,
            self.far_crop_right,
            self.far_crop_bottom,
        )
