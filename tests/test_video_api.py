from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from swimtrack_ai.api import create_app
from swimtrack_ai.config import Settings
from swimtrack_ai.detectors import DetectorResult
from swimtrack_ai.errors import NvdecDecodeError
from swimtrack_ai.tracker import TrackerUpdate
from swimtrack_ai.video_decoder import DecodedVideoFrame


class EmptyDetector:
    def infer_batch(
        self,
        frames: list[np.ndarray],
        target_sizes: list[tuple[int, int]],
    ) -> list[DetectorResult]:
        assert len(frames) == len(target_sizes)
        empty = np.empty((0, 5), dtype=np.float32)
        return [DetectorResult(person_candidates=empty, accepted=empty.copy()) for _ in frames]

    def close(self) -> None:
        return None


class EmptyTracker:
    def update(self, _detections: np.ndarray, _image_size: tuple[int, int]) -> TrackerUpdate:
        return TrackerUpdate(active_tracks=[])


class EmptyTrackerFactory:
    def __init__(self, _settings: Settings) -> None:
        return None

    def __call__(self, _fps: float) -> EmptyTracker:
        return EmptyTracker()


class RecordingVideoDecoder:
    def __init__(self, frames: list[DecodedVideoFrame]) -> None:
        self.frames = frames
        self.cursor = 0
        self.upload_path: Path | None = None
        self.closed = False
        self.batch_sizes: list[int] = []

    def open(self, video_path: Path) -> None:
        assert video_path.is_file()
        self.upload_path = video_path

    def read_batch(self, max_frames: int) -> list[DecodedVideoFrame]:
        self.batch_sizes.append(max_frames)
        result = self.frames[self.cursor : self.cursor + max_frames]
        self.cursor += len(result)
        return result

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        backend="fake",
        auth_token="test-secret",
        model_source_dir=tmp_path,
        model_cache_dir=tmp_path,
        bytetrack_root=tmp_path,
        max_batch_frames=2,
        video_decode_batch_frames=2,
        max_video_bytes=1_024,
    )


def video_frame(index: int, time_ms: float) -> DecodedVideoFrame:
    return DecodedVideoFrame(
        frame_index=index,
        time_ms=time_ms,
        image=np.full((2, 4, 3), index, dtype=np.uint8),
        width=4,
        height=2,
    )


def create_tracking_session(client: TestClient) -> str:
    response = client.post(
        "/v1/tracking-sessions",
        headers={"X-Swimtrack-Auth": "test-secret"},
        json={"fps": 30},
    )
    assert response.status_code == 201
    return response.json()["session_id"]


def test_video_endpoint_streams_ordered_frame_results_and_cleans_up_upload(settings: Settings) -> None:
    decoder = RecordingVideoDecoder(
        [video_frame(0, 0.0), video_frame(1, 33.367), video_frame(2, 100.1)]
    )
    app = create_app(
        settings=settings,
        detector_factory=lambda _settings: EmptyDetector(),
        tracker_factory_builder=EmptyTrackerFactory,
        video_decoder_factory=lambda _settings, _sample_fps: decoder,
    )

    with TestClient(app) as client:
        session_id = create_tracking_session(client)
        response = client.post(
            f"/v1/tracking-sessions/{session_id}/video",
            headers={"X-Swimtrack-Auth": "test-secret"},
            data={"sample_fps": "30"},
            files={"video": ("sample.mp4", b"compressed-video", "video/mp4")},
        )

    assert response.status_code == 200, response.text
    assert response.headers["x-swimtrack-decode-path"] == "nvdec"
    assert response.headers["x-swimtrack-decode-backend"] == "ffmpeg"
    assert response.headers["content-type"].startswith("application/x-ndjson")
    frames = [json.loads(line) for line in response.text.splitlines()]
    assert [frame["frame_index"] for frame in frames] == [0, 1, 2]
    assert [frame["time_ms"] for frame in frames] == pytest.approx([0.0, 33.367, 100.1])
    assert [(frame["width"], frame["height"]) for frame in frames] == [(4, 2), (4, 2), (4, 2)]
    assert decoder.batch_sizes == [2, 2, 2]
    assert decoder.closed is True
    assert decoder.upload_path is not None and not decoder.upload_path.exists()


def test_video_endpoint_returns_a_structured_error_before_streaming(settings: Settings) -> None:
    class FailingDecoder:
        def open(self, _video_path: Path) -> None:
            raise NvdecDecodeError("CUDA device 0 is unavailable")

        def read_batch(self, _max_frames: int) -> list[DecodedVideoFrame]:
            raise AssertionError("read_batch should not be called after an open failure")

        def close(self) -> None:
            return None

    app = create_app(
        settings=settings,
        detector_factory=lambda _settings: EmptyDetector(),
        tracker_factory_builder=EmptyTrackerFactory,
        video_decoder_factory=lambda _settings, _sample_fps: FailingDecoder(),
    )

    with TestClient(app) as client:
        session_id = create_tracking_session(client)
        response = client.post(
            f"/v1/tracking-sessions/{session_id}/video",
            headers={"X-Swimtrack-Auth": "test-secret"},
            data={"sample_fps": "30"},
            files={"video": ("sample.mp4", b"compressed-video", "video/mp4")},
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "nvdec_decode_failed"
    assert "CUDA device 0" in response.json()["error"]["detail"]


def test_video_upload_size_is_enforced_while_storing(settings: Settings) -> None:
    app = create_app(
        settings=settings,
        detector_factory=lambda _settings: EmptyDetector(),
        tracker_factory_builder=EmptyTrackerFactory,
        video_decoder_factory=lambda _settings, _sample_fps: (_ for _ in ()).throw(AssertionError("not reached")),
    )

    with TestClient(app) as client:
        session_id = create_tracking_session(client)
        response = client.post(
            f"/v1/tracking-sessions/{session_id}/video",
            headers={"X-Swimtrack-Auth": "test-secret"},
            data={"sample_fps": "30"},
            files={"video": ("sample.mp4", b"x" * (settings.max_video_bytes + 1), "video/mp4")},
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"
