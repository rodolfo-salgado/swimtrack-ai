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
    score_threshold: float = 0.35
    person_label: int = 0
    max_detections: int = 20
    min_box_area: float = 500.0
    track_threshold: float = 0.45
    track_buffer: int = 60
    match_threshold: float = 0.80
    mot20: bool = False
    lane_roi_enabled: bool = True
    max_batch_frames: int = 8
    max_frame_bytes: int = 2_000_000
    max_batch_bytes: int = 12_000_000
    max_request_bytes: int = 16_000_000
    max_decoded_pixels: int = 4_194_304
    max_metadata_bytes: int = 65_536
    max_sessions: int = 128
    session_ttl_seconds: int = 900
    idempotency_cache_size: int = 32
    cleanup_interval_seconds: int = 30
    trt_workspace_gb: float = 4.0
    trt_fp16: bool = True

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
            score_threshold=_floating("SCORE_THRESHOLD", 0.35),
            person_label=int(_env("PERSON_LABEL", "0")),
            max_detections=_integer("MAX_DETECTIONS", 20),
            min_box_area=_floating("MIN_BOX_AREA", 500.0),
            track_threshold=_floating("TRACK_THRESHOLD", 0.45),
            track_buffer=_integer("TRACK_BUFFER", 60),
            match_threshold=_floating("MATCH_THRESHOLD", 0.80),
            mot20=_boolean("MOT20", False),
            lane_roi_enabled=_boolean("LANE_ROI_ENABLED", True),
            max_batch_frames=_integer("MAX_BATCH_FRAMES", 8),
            max_frame_bytes=_integer("MAX_FRAME_BYTES", 2_000_000),
            max_batch_bytes=_integer("MAX_BATCH_BYTES", 12_000_000),
            max_request_bytes=_integer("MAX_REQUEST_BYTES", 16_000_000),
            max_decoded_pixels=_integer("MAX_DECODED_PIXELS", 4_194_304),
            max_metadata_bytes=_integer("MAX_METADATA_BYTES", 65_536),
            max_sessions=_integer("MAX_SESSIONS", 128),
            session_ttl_seconds=_integer("SESSION_TTL_SECONDS", 900),
            idempotency_cache_size=_integer("IDEMPOTENCY_CACHE_SIZE", 32),
            cleanup_interval_seconds=_integer("CLEANUP_INTERVAL_SECONDS", 30),
            trt_workspace_gb=_floating("TRT_WORKSPACE_GB", 4.0),
            trt_fp16=_boolean("TRT_FP16", True),
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
