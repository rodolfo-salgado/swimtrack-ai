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
    "FAR_CROP_ENABLED",
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
        assert settings.far_crop_enabled is False


def test_far_crop_configuration_is_validated() -> None:
    for values in (
        {"far_crop_left": 0.8, "far_crop_right": 0.2},
        {"far_crop_top": -0.1},
        {"far_crop_nms_threshold": 1.1},
    ):
        with pytest.raises(ValueError):
            Settings(**values)
