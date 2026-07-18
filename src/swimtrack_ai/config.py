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
    far_crop_enabled: bool = False
    far_crop_left: float = 0.2962962963
    far_crop_top: float = 0.1111111111
    far_crop_right: float = 0.7037037037
    far_crop_bottom: float = 0.5185185185
    far_crop_nms_threshold: float = 0.50
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
        if not (
            0.0 <= self.far_crop_left < self.far_crop_right <= 1.0
            and 0.0 <= self.far_crop_top < self.far_crop_bottom <= 1.0
        ):
            raise ValueError("far crop coordinates must define a non-empty normalized rectangle")
        if not 0.0 <= self.far_crop_nms_threshold <= 1.0:
            raise ValueError("far_crop_nms_threshold must be between zero and one")
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
            far_crop_enabled=_boolean("FAR_CROP_ENABLED", False),
            far_crop_left=_floating("FAR_CROP_LEFT", 0.2962962963),
            far_crop_top=_floating("FAR_CROP_TOP", 0.1111111111),
            far_crop_right=_floating("FAR_CROP_RIGHT", 0.7037037037),
            far_crop_bottom=_floating("FAR_CROP_BOTTOM", 0.5185185185),
            far_crop_nms_threshold=_floating("FAR_CROP_NMS_THRESHOLD", 0.50),
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
