from __future__ import annotations

import pytest

from swimtrack_ai.config import Settings

TRACKING_ENV_NAMES = (
    "SCORE_THRESHOLD",
    "MIN_BOX_AREA",
    "TRACK_THRESHOLD",
    "TRACK_BUFFER",
    "MATCH_THRESHOLD",
    "LANE_ROI_ENABLED",
    "WEAK_REACTIVATION_ENABLED",
    "WEAK_REACTIVATION_SCORE_THRESHOLD",
    "WEAK_REACTIVATION_MIN_BOX_AREA",
    "WEAK_REACTIVATION_MAX_GAP_SECONDS",
    "WEAK_REACTIVATION_MAX_CENTER_DISTANCE",
    "FAR_CROP_ENABLED",
)

IDENTITY_ENV_NAMES = (
    "IDENTITY_CONFIRMATION_OBSERVATIONS",
    "IDENTITY_CONFIRMATION_SECONDS",
    "IDENTITY_CONFIRMATION_CONFIDENCE",
    "IDENTITY_TENTATIVE_MAX_GAP_SECONDS",
    "IDENTITY_MAX_REASSOCIATION_GAP_SECONDS",
    "IDENTITY_MAX_SPEED_PER_SECOND",
    "IDENTITY_POSITION_SLACK",
    "IDENTITY_MAX_LANE_X_DELTA",
    "IDENTITY_DUPLICATE_IOU",
    "IDENTITY_DUPLICATE_POSITION_DELTA",
    "IDENTITY_DUPLICATE_LANE_X_DELTA",
    "IDENTITY_ADDITIONAL_MIN_POSITION_SPAN",
    "IDENTITY_ADDITIONAL_COOCCURRENCE_MAX_GAP_SECONDS",
    "IDENTITY_MAX_PER_LANE",
)


def test_selected_tracking_baseline_is_the_default(monkeypatch) -> None:
    for name in TRACKING_ENV_NAMES:
        monkeypatch.delenv(f"SWIMTRACK_{name}", raising=False)

    for settings in (Settings(), Settings.from_env()):
        assert settings.score_threshold == 0.15
        assert settings.min_box_area == 250.0
        assert settings.track_threshold == 0.45
        assert settings.track_buffer == 60
        assert settings.match_threshold == 0.80
        assert settings.lane_roi_enabled is True
        assert settings.weak_reactivation_enabled is True
        assert settings.weak_reactivation_score_threshold == 0.10
        assert settings.weak_reactivation_min_box_area == 64.0
        assert settings.weak_reactivation_max_gap_seconds == 1.0
        assert settings.weak_reactivation_max_center_distance == 0.10
        assert settings.far_crop_enabled is False


def test_far_crop_configuration_is_validated() -> None:
    for values in (
        {"far_crop_left": 0.8, "far_crop_right": 0.2},
        {"far_crop_top": -0.1},
        {"far_crop_nms_threshold": 1.1},
    ):
        with pytest.raises(ValueError):
            Settings(**values)


def test_identity_configuration_has_conservative_defaults_and_is_validated(monkeypatch) -> None:
    for name in IDENTITY_ENV_NAMES:
        monkeypatch.delenv(f"SWIMTRACK_{name}", raising=False)

    for settings in (Settings(), Settings.from_env()):
        assert settings.identity_confirmation_observations == 3
        assert settings.identity_confirmation_seconds == 0.20
        assert settings.identity_confirmation_confidence == 0.18
        assert settings.identity_tentative_max_gap_seconds == 0.75
        assert settings.identity_max_reassociation_gap_seconds == 12.0
        assert settings.identity_max_per_lane == 2

    for values in (
        {"identity_confirmation_observations": 0},
        {"identity_confirmation_confidence": 1.1},
        {"identity_tentative_max_gap_seconds": 0.0},
        {"identity_max_reassociation_gap_seconds": 0.0},
        {"identity_max_speed_per_second": 0.0},
        {"identity_duplicate_iou": 1.1},
        {"identity_additional_min_position_span": 0.0},
        {"identity_max_per_lane": 0},
    ):
        with pytest.raises(ValueError):
            Settings(**values)

    monkeypatch.setenv("SWIMTRACK_IDENTITY_CONFIRMATION_OBSERVATIONS", "4")
    monkeypatch.setenv("SWIMTRACK_IDENTITY_CONFIRMATION_SECONDS", "0.3")
    monkeypatch.setenv("SWIMTRACK_IDENTITY_MAX_PER_LANE", "3")

    settings = Settings.from_env()

    assert settings.identity_confirmation_observations == 4
    assert settings.identity_confirmation_seconds == 0.3
    assert settings.identity_max_per_lane == 3


def test_weak_reactivation_configuration_is_validated_and_loaded_from_environment(monkeypatch) -> None:
    for values in (
        {"weak_reactivation_score_threshold": 0.01},
        {"weak_reactivation_score_threshold": 0.15},
        {"weak_reactivation_score_threshold": 0.46},
        {"weak_reactivation_min_box_area": -1.0},
        {"weak_reactivation_max_gap_seconds": 0.0},
        {"weak_reactivation_max_center_distance": 0.0},
    ):
        with pytest.raises(ValueError):
            Settings(**values)

    monkeypatch.setenv("SWIMTRACK_WEAK_REACTIVATION_ENABLED", "false")
    monkeypatch.setenv("SWIMTRACK_WEAK_REACTIVATION_SCORE_THRESHOLD", "0.12")
    monkeypatch.setenv("SWIMTRACK_WEAK_REACTIVATION_MIN_BOX_AREA", "80")
    monkeypatch.setenv("SWIMTRACK_WEAK_REACTIVATION_MAX_GAP_SECONDS", "1.5")
    monkeypatch.setenv("SWIMTRACK_WEAK_REACTIVATION_MAX_CENTER_DISTANCE", "0.08")

    settings = Settings.from_env()

    assert settings.weak_reactivation_enabled is False
    assert settings.weak_reactivation_score_threshold == 0.12
    assert settings.weak_reactivation_min_box_area == 80.0
    assert settings.weak_reactivation_max_gap_seconds == 1.5
    assert settings.weak_reactivation_max_center_distance == 0.08


def test_tensorrt_batch_configuration_is_validated() -> None:
    with pytest.raises(ValueError, match="trt_opt_batch_size"):
        Settings(trt_opt_batch_size=8, trt_max_batch_size=4)


def test_video_decode_configuration_is_validated_and_loaded_from_environment(monkeypatch) -> None:
    with pytest.raises(ValueError, match="video_decode_batch_frames"):
        Settings(video_decode_batch_frames=9, trt_max_batch_size=8)

    monkeypatch.setenv("SWIMTRACK_MAX_VIDEO_BYTES", "123456")
    monkeypatch.setenv("SWIMTRACK_VIDEO_DECODE_BATCH_FRAMES", "8")
    monkeypatch.setenv("SWIMTRACK_FFMPEG_PATH", "/opt/ffmpeg")
    monkeypatch.setenv("SWIMTRACK_FFPROBE_PATH", "/opt/ffprobe")
    monkeypatch.setenv("SWIMTRACK_VIDEO_PROBE_TIMEOUT_SECONDS", "45")

    settings = Settings.from_env()

    assert settings.max_video_bytes == 123456
    assert settings.video_decode_batch_frames == 8
    assert settings.ffmpeg_path == "/opt/ffmpeg"
    assert settings.ffprobe_path == "/opt/ffprobe"
    assert settings.video_probe_timeout_seconds == 45
