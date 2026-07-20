from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from swimtrack_ai.config import Settings
from swimtrack_ai.detectors import DetectorResult
from swimtrack_ai.schemas import BatchMetadata
from swimtrack_ai.service import TrackingService
from swimtrack_ai.tracker import ByteTrackFactory

ROOT = Path(__file__).resolve().parents[1]


class _SequenceDetector:
    def __init__(self, results: list[DetectorResult]) -> None:
        self._results = deque(results)

    def infer(self, _frame: np.ndarray, _target_size: tuple[int, int]) -> DetectorResult:
        return self._results.popleft()

    def close(self) -> None:
        return None


def _metadata(sequence: int) -> BatchMetadata:
    return BatchMetadata.model_validate(
        {
            "batch_id": f"weak-{sequence}",
            "sequence": sequence,
            "frames": [
                {
                    "frame_index": sequence,
                    "time_ms": sequence * 1000.0 / 60.0,
                    "original_width": 1080,
                    "original_height": 1080,
                }
            ],
        }
    )


def test_service_reactivates_a_lane_routed_candidate_below_detector_acceptance(tmp_path: Path) -> None:
    high = np.asarray([[545.0, 190.0, 565.0, 220.0, 0.90]], dtype=np.float32)
    weak = np.asarray([[545.0, 190.0, 565.0, 220.0, 0.12]], dtype=np.float32)
    empty = np.empty((0, 5), dtype=np.float32)
    detector = _SequenceDetector(
        [
            DetectorResult(person_candidates=high.copy(), accepted=high.copy()),
            DetectorResult(person_candidates=empty, accepted=empty.copy()),
            DetectorResult(person_candidates=weak.copy(), accepted=empty.copy()),
        ]
    )
    settings = Settings(
        backend="fake",
        auth_token="test-secret",
        model_source_dir=tmp_path,
        model_cache_dir=tmp_path,
        bytetrack_root=ROOT / "vendor" / "ByteTrack",
        weak_reactivation_enabled=True,
        weak_reactivation_score_threshold=0.10,
        weak_reactivation_min_box_area=64.0,
        weak_reactivation_max_gap_seconds=1.0,
        weak_reactivation_max_center_distance=0.10,
    )
    service = TrackingService(settings, detector, ByteTrackFactory(settings))
    session_id = service.create_session(
        fps=60,
        lap_calibration_id="fixed-camera-v1",
        diagnostics="boxes",
    ).session_id
    frame = np.zeros((1080, 1080, 3), dtype=np.uint8)

    first = service.process_batch(session_id, _metadata(0), [frame], fingerprint="weak-0")
    missing = service.process_batch(session_id, _metadata(1), [frame], fingerprint="weak-1")
    reactivated = service.process_batch(session_id, _metadata(2), [frame], fingerprint="weak-2")

    first_track_ids = [box.id for box in first.frames[0].boxes]
    assert len(first_track_ids) == 1
    track_id = first_track_ids[0]
    assert missing.frames[0].boxes == []
    assert [box.id for box in reactivated.frames[0].boxes] == [track_id]
    diagnostics = reactivated.frames[0].tracking_diagnostics
    assert diagnostics is not None
    assert diagnostics.detector_accepted.count == 0
    assert diagnostics.weak_candidates.count == 1
    lane = diagnostics.lanes[0]
    assert lane.lane_id == "center"
    assert lane.weak_candidates_after_roi.count == 1
    assert lane.weak_reactivated_track_ids == [track_id]


def test_service_does_not_reactivate_a_lost_track_with_a_detection_used_by_bytetrack(tmp_path: Path) -> None:
    first_detections = np.asarray(
        [
            [450.0, 450.0, 630.0, 650.0, 0.90],
            [540.0, 450.0, 720.0, 650.0, 0.90],
        ],
        dtype=np.float32,
    )
    continuing_detection = first_detections[:1].copy()
    shared_low_detection = np.asarray([[495.0, 450.0, 675.0, 650.0, 0.20]], dtype=np.float32)
    detector = _SequenceDetector(
        [
            DetectorResult(person_candidates=first_detections.copy(), accepted=first_detections.copy()),
            DetectorResult(person_candidates=continuing_detection.copy(), accepted=continuing_detection.copy()),
            DetectorResult(person_candidates=shared_low_detection.copy(), accepted=shared_low_detection.copy()),
        ]
    )
    settings = Settings(
        backend="fake",
        auth_token="test-secret",
        model_source_dir=tmp_path,
        model_cache_dir=tmp_path,
        bytetrack_root=ROOT / "vendor" / "ByteTrack",
        weak_reactivation_enabled=True,
        weak_reactivation_score_threshold=0.10,
        weak_reactivation_min_box_area=64.0,
        weak_reactivation_max_gap_seconds=1.0,
        weak_reactivation_max_center_distance=0.10,
    )
    service = TrackingService(settings, detector, ByteTrackFactory(settings))
    session_id = service.create_session(
        fps=60,
        lap_calibration_id="fixed-camera-v1",
        diagnostics="counts",
    ).session_id
    frame = np.zeros((1080, 1080, 3), dtype=np.uint8)

    first = service.process_batch(session_id, _metadata(0), [frame], fingerprint="shared-0")
    second = service.process_batch(session_id, _metadata(1), [frame], fingerprint="shared-1")
    third = service.process_batch(session_id, _metadata(2), [frame], fingerprint="shared-2")

    first_diagnostics = first.frames[0].tracking_diagnostics
    assert first_diagnostics is not None
    first_ids = set(first_diagnostics.lanes[0].active_track_ids)
    assert len(first_ids) == 2
    second_diagnostics = second.frames[0].tracking_diagnostics
    assert second_diagnostics is not None
    continuing_ids = set(second_diagnostics.lanes[0].active_track_ids)
    assert len(continuing_ids) == 1
    lost_id = next(iter(first_ids - continuing_ids))
    third_diagnostics = third.frames[0].tracking_diagnostics
    assert third_diagnostics is not None
    assert set(third_diagnostics.lanes[0].active_track_ids) == continuing_ids
    assert lost_id not in third_diagnostics.lanes[0].active_track_ids
    assert third_diagnostics.weak_candidates.count == 0
    assert third_diagnostics.lanes[0].weak_candidates_after_roi.count == 0
    assert third_diagnostics.lanes[0].weak_reactivated_track_ids == []
