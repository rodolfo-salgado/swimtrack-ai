from __future__ import annotations

import cv2
import numpy as np
import pytest

from swimtrack_ai.lap_analysis import (
    FIXED_CAMERA_CALIBRATION_ID,
    FIXED_CAMERA_CENTER_LANE,
    LapAnalyzer,
    fixed_camera_visible_polygon,
)
from swimtrack_ai.schemas import BoundingBox


def _box_at(position: float, track_id: int = 7, confidence: float = 0.95) -> BoundingBox:
    source = np.asarray(FIXED_CAMERA_CENTER_LANE.source_quad, dtype=np.float32)
    canonical = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float32)
    lane_to_image = cv2.getPerspectiveTransform(canonical, source)
    point = np.asarray([[[0.5, position]]], dtype=np.float32)
    x_normalized, y_normalized = cv2.perspectiveTransform(point, lane_to_image)[0, 0]
    x = float(x_normalized * 1080)
    y = float(y_normalized * 1080)
    return BoundingBox(
        id=track_id,
        x1=x - 10,
        y1=y - 10,
        x2=x + 10,
        y2=y + 10,
        conf=confidence,
    )


def _run_positions(positions: list[float | None], fps: float = 10.0):
    analyzer = LapAnalyzer(fps=fps, calibration_id=FIXED_CAMERA_CALIBRATION_ID)
    result = None
    for index, position in enumerate(positions):
        boxes = [_box_at(position)] if position is not None else []
        result = analyzer.observe(
            time_ms=index * 1000.0 / fps,
            width=1080,
            height=1080,
            boxes=boxes,
        )[0]
    assert result is not None
    return result


def test_fixed_camera_polygon_matches_reference_image() -> None:
    assert fixed_camera_visible_polygon() == (
        (0.4463, 0.1583),
        (0.5815, 0.1583),
        (1.0, 0.663),
        (1.0, 0.9769),
        (0.0, 0.9769),
        (0.0, 0.6824),
    )


def test_confirmed_near_wall_reversal_gets_high_lap_score() -> None:
    approach = np.linspace(0.55, 0.98, 16).tolist()
    departure = np.linspace(0.98, 0.62, 16)[1:].tolist()
    positions = approach + departure + [0.62] * 15

    result = _run_positions(positions)

    assert result.evaluable
    assert result.endpoint == "near"
    assert result.lap_score > 0.70
    assert result.no_lap_score == pytest.approx(1.0 - result.lap_score)
    assert result.evidence.wall > 0.80
    assert result.evidence.reversal > 0.80


def test_monotonic_middle_trajectory_is_no_lap() -> None:
    result = _run_positions(np.linspace(0.20, 0.80, 46).tolist())

    assert result.evaluable
    assert result.lap_score == 0.0
    assert result.no_lap_score == 1.0
    assert result.endpoint is None


def test_missing_trajectory_is_not_evaluable_as_no_lap() -> None:
    result = _run_positions([None] * 30)

    assert not result.evaluable
    assert result.lap_score == 0.0
    assert result.no_lap_score is None
    assert result.observation_quality == 0.0


def test_projection_uses_lane_as_identity_after_track_id_change() -> None:
    analyzer = LapAnalyzer(fps=10.0, calibration_id=FIXED_CAMERA_CALIBRATION_ID)
    last = None
    positions = np.linspace(0.25, 0.75, 30)
    for index, position in enumerate(positions):
        track_id = 3 if index < 15 else 19
        last = analyzer.observe(
            time_ms=index * 100.0,
            width=1080,
            height=1080,
            boxes=[_box_at(float(position), track_id=track_id)],
        )[0]

    assert last is not None
    assert last.evaluable
    assert last.track_id == 19
    assert last.observation_quality > 0.85
