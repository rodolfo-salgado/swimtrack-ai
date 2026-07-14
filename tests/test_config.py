from __future__ import annotations

from swimtrack_ai.config import Settings

TRACKING_ENV_NAMES = (
    "SCORE_THRESHOLD",
    "MIN_BOX_AREA",
    "TRACK_THRESHOLD",
    "TRACK_BUFFER",
    "MATCH_THRESHOLD",
    "LANE_ROI_ENABLED",
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
