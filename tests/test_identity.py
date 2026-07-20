from __future__ import annotations

from swimtrack_ai.identity import (
    IdentityCandidate,
    IdentityResolution,
    IdentityResolver,
    ResolvedIdentity,
)
from swimtrack_ai.schemas import BoundingBox
from swimtrack_ai.service import TrackingService


def _resolver() -> IdentityResolver:
    return IdentityResolver(
        calibration_id=None,
        confirmation_observations=2,
        confirmation_seconds=0.05,
        confirmation_confidence=0.10,
        tentative_max_gap_seconds=2.0,
        max_reassociation_gap_seconds=12.0,
        max_speed_per_second=0.20,
        position_slack=0.08,
        max_lane_x_delta=0.30,
        duplicate_iou=0.45,
        duplicate_position_delta=0.08,
        duplicate_lane_x_delta=0.15,
        additional_min_position_span=0.10,
        additional_cooccurrence_max_gap_seconds=0.25,
        max_per_lane=2,
    )


def _candidate(
    track_id: int | None,
    *,
    lane_x: float,
    position: float,
    width: float = 12.0,
    height: float = 12.0,
    confidence: float = 0.8,
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
        source="detection",
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
