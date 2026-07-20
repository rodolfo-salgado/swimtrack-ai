from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from swimtrack_ai.config import Settings
from swimtrack_ai.identity import (
    IdentityCandidate,
    IdentityResolution,
    IdentityResolver,
    ResolvedIdentity,
)
from swimtrack_ai.schemas import BoundingBox
from swimtrack_ai.service import TrackingService
from swimtrack_ai.tracker import TrackerUpdate


def _resolver(**overrides) -> IdentityResolver:
    parameters = {
        "calibration_id": None,
        "confirmation_observations": 2,
        "confirmation_seconds": 0.05,
        "confirmation_confidence": 0.10,
        "tentative_max_gap_seconds": 2.0,
        "max_reassociation_gap_seconds": 12.0,
        "max_speed_per_second": 0.20,
        "position_slack": 0.08,
        "max_lane_x_delta": 0.30,
        "duplicate_iou": 0.45,
        "duplicate_position_delta": 0.08,
        "duplicate_lane_x_delta": 0.15,
        "additional_confirmation_observations": 2,
        "additional_confirmation_seconds": 0.05,
        "additional_confirmation_confidence": 0.10,
        "additional_min_position_span": 0.10,
        "additional_cooccurrence_max_gap_seconds": 0.25,
        "max_per_lane": 2,
    }
    parameters.update(overrides)
    return IdentityResolver(**parameters)


def _candidate(
    track_id: int | None,
    *,
    lane_x: float,
    position: float,
    width: float = 12.0,
    height: float = 12.0,
    confidence: float = 0.8,
    source: str = "detection",
) -> IdentityCandidate:
    x1 = lane_x * 100.0 - width / 2.0
    y1 = position * 100.0 - height / 2.0
    box = BoundingBox(
        id=track_id if track_id is not None else -1,
        track_id=track_id,
        lane_id="center",
        x1=x1,
        y1=y1,
        x2=x1 + width,
        y2=y1 + height,
        conf=confidence,
    )
    return IdentityCandidate(
        lane_id="center",
        box=box,
        track_id=track_id,
        source=source,
    )


def _resolve(resolver: IdentityResolver, time_ms: float, candidates: list[IdentityCandidate]):
    return resolver.resolve(time_ms=time_ms, width=100, height=100, candidates=candidates)


def test_track_id_churn_keeps_one_canonical_identity() -> None:
    resolver = _resolver()

    first = _resolve(resolver, 0.0, [_candidate(7, lane_x=0.5, position=0.20)])
    second = _resolve(resolver, 100.0, [_candidate(19, lane_x=0.5, position=0.22)])

    assert [assignment.identity_id for assignment in first.assignments] == [1]
    assert [assignment.identity_id for assignment in second.assignments] == [1]
    assert second.confirmed_count == 1
    assert second.active_count == 1


def test_duplicate_tracklets_never_create_two_people() -> None:
    resolver = _resolver()
    duplicate_pair = [
        _candidate(7, lane_x=0.50, position=0.30, width=18.0, height=18.0),
        _candidate(8, lane_x=0.51, position=0.31, width=18.0, height=18.0),
    ]

    first = _resolve(resolver, 0.0, duplicate_pair)
    second = _resolve(resolver, 100.0, duplicate_pair)

    assert len(first.assignments) == 1
    assert len(second.assignments) == 1
    assert second.confirmed_count == 1


def test_two_swimmers_in_one_lane_are_kept_distinct() -> None:
    resolver = _resolver()
    first = _resolve(
        resolver,
        0.0,
        [
            _candidate(10, lane_x=0.25, position=0.20),
            _candidate(20, lane_x=0.75, position=0.80),
        ],
    )
    second = _resolve(
        resolver,
        100.0,
        [
            _candidate(11, lane_x=0.25, position=0.22),
            _candidate(21, lane_x=0.75, position=0.78),
        ],
    )

    assert {assignment.identity_id for assignment in first.assignments} == {1, 2}
    assert {assignment.identity_id for assignment in second.assignments} == {1, 2}
    assert {assignment.swimmer_id for assignment in second.assignments} == {1, 2}
    assert second.confirmed_count == 2
    assert second.active_count == 2


def test_second_swimmer_is_confirmed_after_persistent_cooccurring_motion() -> None:
    resolver = _resolver()
    _resolve(resolver, 0.0, [_candidate(10, lane_x=0.25, position=0.20)])
    first_confirmed = _resolve(resolver, 100.0, [_candidate(11, lane_x=0.25, position=0.22)])
    initial_pair = _resolve(
        resolver,
        200.0,
        [
            _candidate(12, lane_x=0.25, position=0.24),
            _candidate(None, lane_x=0.75, position=0.80),
        ],
    )
    _resolve(
        resolver,
        400.0,
        [
            _candidate(13, lane_x=0.25, position=0.28),
            _candidate(None, lane_x=0.75, position=0.75),
        ],
    )
    _resolve(
        resolver,
        600.0,
        [
            _candidate(14, lane_x=0.25, position=0.32),
            _candidate(None, lane_x=0.75, position=0.70),
        ],
    )
    stable_pair = _resolve(
        resolver,
        800.0,
        [
            _candidate(15, lane_x=0.25, position=0.36),
            _candidate(None, lane_x=0.75, position=0.66),
        ],
    )

    assert first_confirmed.confirmed_count == 1
    assert initial_pair.confirmed_count == 1
    assert stable_pair.confirmed_count == 2
    assert {assignment.identity_id for assignment in stable_pair.assignments} == {1, 2}


def test_second_swimmer_uses_current_cooccurrence_confidence_not_its_lifetime_average() -> None:
    resolver = _resolver(
        confirmation_confidence=0.5,
        additional_confirmation_confidence=0.5,
    )
    _resolve(resolver, 0.0, [_candidate(10, lane_x=0.25, position=0.20, confidence=0.9)])
    _resolve(resolver, 100.0, [_candidate(11, lane_x=0.25, position=0.22, confidence=0.9)])
    _resolve(
        resolver,
        200.0,
        [
            _candidate(12, lane_x=0.25, position=0.24, confidence=0.9),
            _candidate(None, lane_x=0.75, position=0.80, confidence=0.9),
        ],
    )
    _resolve(
        resolver,
        250.0,
        [
            _candidate(13, lane_x=0.25, position=0.26, confidence=0.9),
            _candidate(None, lane_x=0.75, position=0.80, confidence=0.9),
        ],
    )
    _resolve(
        resolver,
        600.0,
        [
            _candidate(14, lane_x=0.25, position=0.28, confidence=0.9),
            _candidate(None, lane_x=0.75, position=0.80, confidence=0.2),
        ],
    )
    latest = _resolve(
        resolver,
        800.0,
        [
            _candidate(15, lane_x=0.25, position=0.30, confidence=0.9),
            _candidate(None, lane_x=0.75, position=0.69, confidence=0.2),
        ],
    )

    assert latest.confirmed_count == 1


def test_tracker_prediction_cannot_confirm_a_second_swimmer() -> None:
    resolver = _resolver()
    _resolve(resolver, 0.0, [_candidate(10, lane_x=0.25, position=0.20)])
    _resolve(resolver, 100.0, [_candidate(11, lane_x=0.25, position=0.22)])
    _resolve(
        resolver,
        200.0,
        [
            _candidate(12, lane_x=0.25, position=0.24),
            _candidate(None, lane_x=0.75, position=0.80),
        ],
    )
    latest = _resolve(
        resolver,
        400.0,
        [
            _candidate(13, lane_x=0.25, position=0.28),
            _candidate(31, lane_x=0.75, position=0.69, source="track"),
        ],
    )

    assert latest.confirmed_count == 1


def test_second_swimmer_requires_stricter_recent_detector_evidence() -> None:
    resolver = _resolver(
        additional_confirmation_observations=8,
        additional_confirmation_seconds=0.5,
        additional_confirmation_confidence=0.3,
        additional_min_position_span=0.15,
    )
    _resolve(resolver, 0.0, [_candidate(10, lane_x=0.25, position=0.20, confidence=0.9)])
    _resolve(resolver, 100.0, [_candidate(11, lane_x=0.25, position=0.22, confidence=0.9)])

    seventh = None
    eighth = None
    for index in range(8):
        result = _resolve(
            resolver,
            200.0 + index * 100.0,
            [
                _candidate(12 + index, lane_x=0.25, position=0.24 + index * 0.01, confidence=0.9),
                _candidate(None, lane_x=0.75, position=0.80 - index * 0.025, confidence=0.8),
            ],
        )
        if index == 6:
            seventh = result
        if index == 7:
            eighth = result

    assert seventh is not None and seventh.confirmed_count == 1
    assert eighth is not None and eighth.confirmed_count == 2


def test_detector_only_identities_get_distinct_legacy_fallback_keys() -> None:
    first = _candidate(None, lane_x=0.25, position=0.20)
    second = _candidate(None, lane_x=0.75, position=0.80)
    resolution = IdentityResolution(
        assignments=[
            ResolvedIdentity(candidate=first, identity_id=1, confirmed=True),
            ResolvedIdentity(candidate=second, identity_id=2, confirmed=True),
        ],
        confirmed_count=2,
        active_count=2,
    )

    boxes = TrackingService._resolved_identity_boxes(resolution)

    assert [(box.id, box.track_id, box.identity_id) for box in boxes] == [
        (-1, None, 1),
        (-2, None, 2),
    ]


def test_duplicate_raw_track_ids_get_unique_legacy_fallback_keys() -> None:
    first = _candidate(44, lane_x=0.25, position=0.20)
    second = _candidate(44, lane_x=0.75, position=0.80)
    resolution = IdentityResolution(
        assignments=[
            ResolvedIdentity(candidate=first, identity_id=1, confirmed=True),
            ResolvedIdentity(candidate=second, identity_id=2, confirmed=True),
        ],
        confirmed_count=2,
        active_count=2,
    )

    boxes = TrackingService._resolved_identity_boxes(resolution)

    assert [(box.id, box.track_id, box.identity_id) for box in boxes] == [
        (44, 44, 1),
        (-2, 44, 2),
    ]


def test_active_track_is_associated_to_at_most_one_detector_box() -> None:
    service = TrackingService(
        Settings(),
        detector=SimpleNamespace(),
        tracker_factory=lambda _fps: SimpleNamespace(),
    )
    tracker_updates = {
        "center": TrackerUpdate(
            active_tracks=[
                SimpleNamespace(
                    track_id=44,
                    tlbr=np.asarray([612.0, 227.0, 650.0, 270.0]),
                    score=0.9,
                )
            ]
        )
    }
    routed_detections = {
        "center": np.asarray(
            [
                [612.0, 227.0, 650.0, 270.0, 0.28],
                [528.0, 228.0, 650.0, 271.0, 0.22],
            ],
            dtype=np.float32,
        )
    }

    candidates = service._identity_candidates(tracker_updates, routed_detections, (1080, 1080))

    assert [candidate.track_id for candidate in candidates] == [44, None]


def test_crossing_with_raw_id_churn_preserves_the_two_trajectories() -> None:
    resolver = _resolver()
    _resolve(
        resolver,
        0.0,
        [
            _candidate(10, lane_x=0.25, position=0.20),
            _candidate(20, lane_x=0.75, position=0.80),
        ],
    )
    _resolve(
        resolver,
        1_000.0,
        [
            _candidate(10, lane_x=0.25, position=0.30),
            _candidate(20, lane_x=0.75, position=0.70),
        ],
    )
    crossing = _resolve(
        resolver,
        2_000.0,
        [
            _candidate(99, lane_x=0.25, position=0.50),
            _candidate(88, lane_x=0.75, position=0.50),
        ],
    )
    after = _resolve(
        resolver,
        3_000.0,
        [
            _candidate(99, lane_x=0.25, position=0.70),
            _candidate(88, lane_x=0.75, position=0.30),
        ],
    )

    assert {assignment.candidate.track_id: assignment.identity_id for assignment in crossing.assignments} == {
        99: 1,
        88: 2,
    }
    assert {assignment.candidate.track_id: assignment.identity_id for assignment in after.assignments} == {
        99: 1,
        88: 2,
    }
    assert after.confirmed_count == 2


def test_raw_track_id_swap_cannot_override_two_swimmer_trajectories() -> None:
    resolver = _resolver()
    _resolve(
        resolver,
        0.0,
        [
            _candidate(10, lane_x=0.25, position=0.20),
            _candidate(20, lane_x=0.75, position=0.80),
        ],
    )
    _resolve(
        resolver,
        1_000.0,
        [
            _candidate(10, lane_x=0.25, position=0.30),
            _candidate(20, lane_x=0.75, position=0.70),
        ],
    )

    after_swap = _resolve(
        resolver,
        2_000.0,
        [
            _candidate(20, lane_x=0.25, position=0.40),
            _candidate(10, lane_x=0.75, position=0.60),
        ],
    )

    assert {assignment.candidate.track_id: assignment.identity_id for assignment in after_swap.assignments} == {
        20: 1,
        10: 2,
    }


def test_single_observation_during_a_two_swimmer_crossing_does_not_exchange_ids() -> None:
    resolver = _resolver()
    _resolve(
        resolver,
        0.0,
        [
            _candidate(10, lane_x=0.50, position=0.30),
            _candidate(20, lane_x=0.50, position=0.70),
        ],
    )
    _resolve(
        resolver,
        1_000.0,
        [
            _candidate(10, lane_x=0.50, position=0.40),
            _candidate(20, lane_x=0.50, position=0.60),
        ],
    )

    occluded = _resolve(resolver, 2_000.0, [_candidate(99, lane_x=0.50, position=0.50)])
    separated = _resolve(
        resolver,
        3_000.0,
        [
            _candidate(88, lane_x=0.50, position=0.70),
            _candidate(99, lane_x=0.50, position=0.30),
        ],
    )

    assert occluded.assignments == []
    assert {assignment.candidate.track_id: assignment.identity_id for assignment in separated.assignments} == {
        88: 1,
        99: 2,
    }


def test_unique_unmatched_detector_observation_recovers_the_other_confirmed_swimmer() -> None:
    resolver = _resolver()
    _resolve(
        resolver,
        0.0,
        [
            _candidate(10, lane_x=0.25, position=0.20),
            _candidate(20, lane_x=0.75, position=0.80),
        ],
    )
    _resolve(
        resolver,
        1_000.0,
        [
            _candidate(10, lane_x=0.25, position=0.30),
            _candidate(20, lane_x=0.75, position=0.70),
        ],
    )

    recovered = _resolve(
        resolver,
        2_000.0,
        [
            _candidate(11, lane_x=0.25, position=0.40),
            _candidate(None, lane_x=0.75, position=0.98),
        ],
    )

    assert {assignment.identity_id for assignment in recovered.assignments} == {1, 2}
    assert {assignment.candidate.track_id: assignment.identity_id for assignment in recovered.assignments} == {
        11: 1,
        None: 2,
    }


def test_long_detector_gap_reacquires_the_only_swimmer_without_new_identity() -> None:
    resolver = _resolver()
    _resolve(resolver, 0.0, [_candidate(7, lane_x=0.45, position=0.20)])
    confirmed = _resolve(resolver, 100.0, [_candidate(7, lane_x=0.45, position=0.22)])
    reacquired = _resolve(resolver, 5_200.0, [_candidate(31, lane_x=0.55, position=0.62)])

    assert confirmed.confirmed_count == 1
    assert [assignment.identity_id for assignment in reacquired.assignments] == [1]
    assert reacquired.confirmed_count == 1


def test_short_false_positive_never_becomes_confirmed() -> None:
    resolver = _resolver()
    first = _resolve(resolver, 0.0, [_candidate(None, lane_x=0.2, position=0.2)])
    after_gap = _resolve(resolver, 3_000.0, [_candidate(8, lane_x=0.6, position=0.6)])

    assert first.confirmed_count == 0
    assert after_gap.confirmed_count == 0
