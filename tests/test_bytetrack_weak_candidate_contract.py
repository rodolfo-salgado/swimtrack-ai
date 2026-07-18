from __future__ import annotations

from pathlib import Path

import numpy as np

from swimtrack_ai.calibration import FIXED_CAMERA_CALIBRATION_ID, LaneRouter
from swimtrack_ai.config import Settings
from swimtrack_ai.tracker import ByteTrackFactory


def test_lane_router_keeps_a_small_low_confidence_far_candidate_in_its_lane() -> None:
    """Geometry routing itself does not discard a weak candidate by score or area."""

    weak_far_candidate = np.asarray(
        [
            [520.0, 190.0, 530.0, 200.0, 0.12],
            [0.0, 0.0, 10.0, 10.0, 0.12],
        ],
        dtype=np.float32,
    )
    router = LaneRouter(FIXED_CAMERA_CALIBRATION_ID, enabled=True)

    routed = router.route(weak_far_candidate, (1080, 1080))

    assert routed["center"].shape == (1, 5)
    np.testing.assert_allclose(routed["center"], weak_far_candidate[:1])


def test_bytetrack_uses_a_low_score_detection_to_extend_an_active_track(tmp_path: Path) -> None:
    """Vendored ByteTrack's second association accepts scores in (0.1, track_threshold)."""

    settings = Settings(
        backend="fake",
        auth_token="test-secret",
        model_source_dir=tmp_path / "model-source",
        model_cache_dir=tmp_path / "model-cache",
        bytetrack_root=Path(__file__).resolve().parents[1] / "vendor" / "ByteTrack",
    )
    tracker = ByteTrackFactory(settings)(fps=60)
    high_score = np.asarray([[450.0, 450.0, 630.0, 650.0, 0.90]], dtype=np.float32)
    low_score = np.asarray([[450.0, 450.0, 630.0, 650.0, 0.12]], dtype=np.float32)

    first_update = tracker.update(high_score, (1080, 1080))
    weak_update = tracker.update(low_score, (1080, 1080))

    assert len(first_update.active_tracks) == 1
    assert [track.track_id for track in weak_update.active_tracks] == [first_update.active_tracks[0].track_id]
    assert weak_update.active_tracks[0].score == np.float32(0.12)


def test_bytetrack_requires_a_high_score_detection_to_reacquire_a_lost_track(tmp_path: Path) -> None:
    """A weak detection cannot reactivate a lost upstream ByteTrack track, unlike a high-score one."""

    settings = Settings(
        backend="fake",
        auth_token="test-secret",
        model_source_dir=tmp_path / "model-source",
        model_cache_dir=tmp_path / "model-cache",
        bytetrack_root=Path(__file__).resolve().parents[1] / "vendor" / "ByteTrack",
    )
    tracker = ByteTrackFactory(settings)(fps=60)
    high_score = np.asarray([[450.0, 450.0, 630.0, 650.0, 0.90]], dtype=np.float32)
    weak_score = np.asarray([[450.0, 450.0, 630.0, 650.0, 0.30]], dtype=np.float32)
    empty = np.empty((0, 5), dtype=np.float32)

    first_update = tracker.update(high_score, (1080, 1080))
    missing_update = tracker.update(empty, (1080, 1080))
    weak_reacquisition = tracker.update(weak_score, (1080, 1080))
    high_reacquisition = tracker.update(high_score, (1080, 1080))

    assert len(first_update.active_tracks) == 1
    assert missing_update.active_tracks == []
    assert missing_update.retained_lost_track_count == 1
    assert weak_reacquisition.active_tracks == []
    assert weak_reacquisition.retained_lost_track_count == 1
    assert [track.track_id for track in high_reacquisition.active_tracks] == [first_update.active_tracks[0].track_id]


def test_bytetrack_adapter_weak_candidate_reacquires_a_recent_lost_track_with_its_original_id(tmp_path: Path) -> None:
    settings = Settings(
        backend="fake",
        auth_token="test-secret",
        model_source_dir=tmp_path / "model-source",
        model_cache_dir=tmp_path / "model-cache",
        bytetrack_root=Path(__file__).resolve().parents[1] / "vendor" / "ByteTrack",
    )
    tracker = ByteTrackFactory(settings)(fps=60)
    high_score = np.asarray([[450.0, 450.0, 630.0, 650.0, 0.90]], dtype=np.float32)
    weak_score = np.asarray([[450.0, 450.0, 630.0, 650.0, 0.30]], dtype=np.float32)
    empty = np.empty((0, 5), dtype=np.float32)

    first_update = tracker.update(high_score, (1080, 1080))
    weak_reacquisition = tracker.update_with_weak_candidates(
        empty,
        weak_score,
        (1080, 1080),
        max_gap_frames=1,
        max_center_distance=0.05,
    )

    track_id = first_update.active_tracks[0].track_id
    assert [track.track_id for track in weak_reacquisition.active_tracks] == [track_id]
    assert weak_reacquisition.weak_reactivated_track_ids == [track_id]
    assert weak_reacquisition.retained_lost_track_count == 0


def test_bytetrack_adapter_weak_candidate_outside_center_gate_does_not_create_or_reacquire(tmp_path: Path) -> None:
    settings = Settings(
        backend="fake",
        auth_token="test-secret",
        model_source_dir=tmp_path / "model-source",
        model_cache_dir=tmp_path / "model-cache",
        bytetrack_root=Path(__file__).resolve().parents[1] / "vendor" / "ByteTrack",
    )
    tracker = ByteTrackFactory(settings)(fps=60)
    high_score = np.asarray([[450.0, 450.0, 630.0, 650.0, 0.90]], dtype=np.float32)
    distant_weak_score = np.asarray([[900.0, 900.0, 1000.0, 1000.0, 0.30]], dtype=np.float32)
    empty = np.empty((0, 5), dtype=np.float32)

    tracker.update(high_score, (1080, 1080))
    weak_reacquisition = tracker.update_with_weak_candidates(
        empty,
        distant_weak_score,
        (1080, 1080),
        max_gap_frames=1,
        max_center_distance=0.05,
    )

    assert weak_reacquisition.active_tracks == []
    assert weak_reacquisition.weak_reactivated_track_ids == []
    assert weak_reacquisition.retained_lost_track_count == 1


def test_bytetrack_adapter_weak_candidate_after_maximum_gap_does_not_reacquire(tmp_path: Path) -> None:
    settings = Settings(
        backend="fake",
        auth_token="test-secret",
        model_source_dir=tmp_path / "model-source",
        model_cache_dir=tmp_path / "model-cache",
        bytetrack_root=Path(__file__).resolve().parents[1] / "vendor" / "ByteTrack",
    )
    tracker = ByteTrackFactory(settings)(fps=60)
    high_score = np.asarray([[450.0, 450.0, 630.0, 650.0, 0.90]], dtype=np.float32)
    weak_score = np.asarray([[450.0, 450.0, 630.0, 650.0, 0.30]], dtype=np.float32)
    empty = np.empty((0, 5), dtype=np.float32)

    tracker.update(high_score, (1080, 1080))
    tracker.update(empty, (1080, 1080))
    tracker.update(empty, (1080, 1080))
    weak_reacquisition = tracker.update_with_weak_candidates(
        empty,
        weak_score,
        (1080, 1080),
        max_gap_frames=1,
        max_center_distance=0.05,
    )

    assert weak_reacquisition.active_tracks == []
    assert weak_reacquisition.weak_reactivated_track_ids == []
    assert weak_reacquisition.retained_lost_track_count == 1
