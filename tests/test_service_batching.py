from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from swimtrack_ai.config import Settings
from swimtrack_ai.detectors import DetectorResult
from swimtrack_ai.detectors.tensorrt import postprocess_detections
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


def test_far_crop_detection_in_calibrated_extension_reaches_lane_tracker(tmp_path: Path) -> None:
    class _FarEndDetector:
        def infer(self, _frame: np.ndarray, _target_size: tuple[int, int]) -> DetectorResult:
            raise AssertionError("TrackingService should use infer_batch when it is available")

        def infer_batch(
            self,
            frames: list[np.ndarray],
            target_sizes: list[tuple[int, int]],
        ) -> list[DetectorResult]:
            assert [target_size for target_size in target_sizes] == [(1080, 1080), (440, 440)]
            empty = np.empty((0, 5), dtype=np.float32)
            far_end_detection = np.asarray([[225.0, 5.0, 245.0, 25.0, 0.90]], dtype=np.float32)
            return [
                DetectorResult(person_candidates=empty, accepted=empty.copy()),
                DetectorResult(person_candidates=far_end_detection.copy(), accepted=far_end_detection),
            ]

        def close(self) -> None:
            return None

    tracker = _RecordingTracker()
    service = TrackingService(
        replace(_settings(tmp_path), far_crop_enabled=True),
        _FarEndDetector(),
        lambda _fps: tracker,
    )
    session_id = service.create_session(fps=60, lap_calibration_id="fixed-camera-v1").session_id
    metadata = BatchMetadata.model_validate(
        {
            "batch_id": "far-end-extension",
            "sequence": 0,
            "frames": [
                {
                    "frame_index": 0,
                    "time_ms": 0.0,
                    "original_width": 1080,
                    "original_height": 1080,
                }
            ],
        }
    )

    service.process_batch(
        session_id,
        metadata,
        [np.zeros((108, 108, 3), dtype=np.uint8)],
        fingerprint="far-end-extension",
    )

    assert len(tracker.calls) == 1
    np.testing.assert_allclose(tracker.calls[0], [[545.0, 125.0, 565.0, 145.0, 0.90]])


def test_far_crop_view_keeps_exact_fixed_camera_boundaries(tmp_path: Path) -> None:
    service = TrackingService(
        replace(_settings(tmp_path), far_crop_enabled=True),
        detector=_BatchDetector(),
        tracker_factory=lambda _fps: _RecordingTracker(),
    )

    crop, target_size, offset = service._far_crop_view(
        np.zeros((108, 108, 3), dtype=np.uint8),
        (1080, 1080),
    )

    assert crop.shape == (44, 44, 3)
    assert target_size == (440, 440)
    assert offset == (320.0, 120.0)


def test_lane_roi_applies_max_detections_after_off_lane_false_positives(tmp_path: Path) -> None:
    class _Detector:
        def infer(self, _frame: np.ndarray, target_size: tuple[int, int]) -> DetectorResult:
            return postprocess_detections(
                np.asarray([0, 0], dtype=np.int64),
                np.asarray([[0.0, 0.0, 100.0, 100.0], [545.0, 190.0, 565.0, 220.0]], dtype=np.float32),
                np.asarray([0.99, 0.20], dtype=np.float32),
                settings,
                target_size,
            )

        def close(self) -> None:
            return None

    settings = replace(_settings(tmp_path), max_detections=1)
    tracker = _RecordingTracker()
    service = TrackingService(settings, _Detector(), lambda _fps: tracker)
    session_id = service.create_session(fps=60, lap_calibration_id="fixed-camera-v1").session_id
    metadata = BatchMetadata.model_validate(
        {
            "batch_id": "route-before-cap",
            "sequence": 0,
            "frames": [
                {
                    "frame_index": 0,
                    "time_ms": 0.0,
                    "original_width": 1080,
                    "original_height": 1080,
                }
            ],
        }
    )

    service.process_batch(
        session_id,
        metadata,
        [np.zeros((1080, 1080, 3), dtype=np.uint8)],
        fingerprint="route-before-cap",
    )

    np.testing.assert_allclose(tracker.calls[0], [[545.0, 190.0, 565.0, 220.0, 0.20]])
