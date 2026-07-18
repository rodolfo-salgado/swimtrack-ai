from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from swimtrack_ai.config import Settings
from swimtrack_ai.detectors import DetectorResult
from swimtrack_ai.schemas import BatchMetadata
from swimtrack_ai.service import TrackingService
from swimtrack_ai.tracker import TrackerUpdate


class _RecordingTracker:
    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def update(self, detections: np.ndarray, _image_size: tuple[int, int]) -> TrackerUpdate:
        self.calls.append(detections.copy())
        return TrackerUpdate(active_tracks=[])


class _BatchDetector:
    def __init__(self) -> None:
        self.calls: list[list[tuple[tuple[int, ...], tuple[int, int]]]] = []

    def infer(self, _frame: np.ndarray, _target_size: tuple[int, int]) -> DetectorResult:
        raise AssertionError("TrackingService should use infer_batch when it is available")

    def infer_batch(
        self,
        frames: list[np.ndarray],
        target_sizes: list[tuple[int, int]],
    ) -> list[DetectorResult]:
        self.calls.append([(frame.shape, target_size) for frame, target_size in zip(frames, target_sizes)])
        empty = np.empty((0, 5), dtype=np.float32)
        return [DetectorResult(person_candidates=empty, accepted=empty.copy()) for _ in frames]

    def close(self) -> None:
        return None


def _settings(tmp_path: Path, *, far_crop_enabled: bool = False) -> Settings:
    return Settings(
        backend="fake",
        auth_token="test-secret",
        model_source_dir=tmp_path,
        model_cache_dir=tmp_path,
        bytetrack_root=tmp_path,
        far_crop_enabled=far_crop_enabled,
    )


def _metadata(batch_id: str, frame_count: int) -> BatchMetadata:
    return BatchMetadata.model_validate(
        {
            "batch_id": batch_id,
            "sequence": 0,
            "frames": [
                {
                    "frame_index": index,
                    "time_ms": index * 16.667,
                    "original_width": 128,
                    "original_height": 96,
                }
                for index in range(frame_count)
            ],
        }
    )


def test_detector_batch_preserves_result_and_tracker_order(tmp_path: Path) -> None:
    detector = _BatchDetector()
    trackers: list[_RecordingTracker] = []

    def tracker_factory(_fps: float) -> _RecordingTracker:
        tracker = _RecordingTracker()
        trackers.append(tracker)
        return tracker

    service = TrackingService(_settings(tmp_path), detector, tracker_factory)
    session_id = service.create_session(fps=60).session_id
    metadata = _metadata("batch", 3)
    result = service.process_batch(
        session_id,
        metadata,
        [np.zeros((64, 64, 3), dtype=np.uint8) for _ in metadata.frames],
        fingerprint="test",
    )

    assert [frame.frame_index for frame in result.frames] == [0, 1, 2]
    assert detector.calls == [[((64, 64, 3), (128, 96))] * 3]
    assert len(trackers) == 1
    assert len(trackers[0].calls) == 3


def test_far_crop_full_and_crop_views_share_one_detector_batch(tmp_path: Path) -> None:
    detector = _BatchDetector()
    settings = replace(_settings(tmp_path), far_crop_enabled=True)
    service = TrackingService(settings, detector, lambda _fps: _RecordingTracker())
    session_id = service.create_session(fps=60, lap_calibration_id="fixed-camera-v1").session_id
    metadata = _metadata("far-crop-batch", 2)
    service.process_batch(
        session_id,
        metadata,
        [np.zeros((64, 64, 3), dtype=np.uint8) for _ in metadata.frames],
        fingerprint="test",
    )

    assert detector.calls == [
        [
            ((64, 64, 3), (128, 96)),
            ((64, 64, 3), (128, 96)),
            ((27, 28, 3), (56, 41)),
            ((27, 28, 3), (56, 41)),
        ]
    ]
