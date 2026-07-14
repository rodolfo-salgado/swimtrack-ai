from __future__ import annotations

import cv2
import numpy as np
import pytest

from swimtrack_ai.lap_analysis import (
    FIXED_CAMERA_CALIBRATION_ID,
    FIXED_CAMERA_CENTER_LANE,
    LAP_SCORE_VERSION,
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


def _run_timeline(
    positions: list[float | None],
    *,
    fps: float = 10.0,
    track_ids: list[int] | None = None,
):
    analyzer = LapAnalyzer(fps=fps, calibration_id=FIXED_CAMERA_CALIBRATION_ID)
    results = []
    for index, position in enumerate(positions):
        track_id = track_ids[index] if track_ids is not None else 7
        boxes = [_box_at(position, track_id=track_id)] if position is not None else []
        results.append(
            analyzer.observe(
                time_ms=index * 1000.0 / fps,
                width=1080,
                height=1080,
                boxes=boxes,
            )[0]
        )
    return results


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
    assert result.candidate_episode_id == 1
    assert result.score_version == LAP_SCORE_VERSION


def test_near_wall_start_is_not_eligible_without_prior_interior_observation() -> None:
    waiting = [0.97, 0.975, 0.972, 0.978, 0.974, 0.976, 0.973, 0.977, 0.974, 0.976, 0.973, 0.975]
    departure = np.linspace(0.97, 0.55, 22)[1:].tolist()

    results = _run_timeline(waiting + departure + [0.55] * 15)

    assert max(result.lap_score for result in results) == 0.0
    assert all(result.candidate_time_ms is None for result in results)
    assert all(result.candidate_episode_id is None for result in results)


def test_finish_at_wall_without_departure_is_not_a_lap() -> None:
    approach = np.linspace(0.45, 0.98, 22).tolist()

    results = _run_timeline(approach + [0.98] * 25)

    assert max(result.lap_score for result in results) == 0.0


def test_fragmented_turn_with_missing_observations_keeps_positive_score() -> None:
    positions: list[float | None] = np.linspace(0.45, 0.98, 18).tolist()
    positions += np.linspace(0.98, 0.55, 18)[1:].tolist()
    positions += [0.55] * 15
    positions = [None if index % 5 == 4 else position for index, position in enumerate(positions)]
    track_ids = [3 if index < 18 else 19 for index in range(len(positions))]

    results = _run_timeline(positions, track_ids=track_ids)
    best = max(results, key=lambda result: result.lap_score)

    assert best.lap_score > 0.45
    assert best.endpoint == "near"
    assert best.track_id in {3, 19}


def test_confirmed_far_wall_reversal_gets_high_lap_score() -> None:
    approach = np.linspace(0.45, 0.02, 16).tolist()
    departure = np.linspace(0.02, 0.38, 16)[1:].tolist()

    result = _run_positions(approach + departure + [0.38] * 15)

    assert result.evaluable
    assert result.endpoint == "far"
    assert result.lap_score > 0.70


def test_each_wall_visit_has_one_episode_id_and_rearms_in_the_interior() -> None:
    first_turn = np.linspace(0.45, 0.98, 16).tolist() + np.linspace(0.98, 0.45, 16)[1:].tolist()
    second_turn = np.linspace(0.45, 0.97, 16)[1:].tolist() + np.linspace(0.97, 0.50, 16)[1:].tolist()

    results = _run_timeline(first_turn + [0.45] * 12 + second_turn + [0.50] * 15)
    positive_episode_ids = {
        result.candidate_episode_id
        for result in results
        if result.lap_score > 0.0 and result.candidate_episode_id is not None
    }

    assert positive_episode_ids == {1, 2}


def test_far_wall_reversal_survives_a_long_underwater_gap() -> None:
    approach = np.linspace(0.42, 0.03, 18).tolist()
    underwater = [None] * 45
    departure = np.linspace(0.04, 0.42, 18).tolist()

    results = _run_timeline(approach + underwater + departure + [0.42] * 15)
    best = max(results, key=lambda result: result.lap_score)

    assert best.endpoint == "far"
    assert best.candidate_episode_id == 1
    assert best.lap_score > 0.20
    assert best.candidate_time_ms is not None
    assert best.candidate_time_ms < 2_000


def test_gap_longer_than_supported_occlusion_is_not_joined() -> None:
    approach = np.linspace(0.42, 0.03, 18).tolist()
    underwater = [None] * 65
    departure = np.linspace(0.04, 0.42, 18).tolist()

    results = _run_timeline(approach + underwater + departure + [0.42] * 15)

    assert max(result.lap_score for result in results) == 0.0


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
