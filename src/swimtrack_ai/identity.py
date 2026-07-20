"""Canonical swimmer identities built from detector observations.

ByteTrack identifiers describe short-lived tracklets, not people.  This module
keeps a small, session-local association layer above those tracklets so callers
can count physical swimmers without treating every ByteTrack restart as a new
person.  It deliberately uses detector observations as well as active tracks:
low-confidence swimmers can be visible while ByteTrack has no active track.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from swimtrack_ai.calibration import lanes_for_calibration, perspective_matrix
from swimtrack_ai.schemas import BoundingBox


@dataclass(frozen=True, slots=True)
class IdentityCandidate:
    """One detector or tracker observation routed to a calibrated lane."""

    lane_id: str
    box: BoundingBox
    track_id: int | None
    source: str


@dataclass(frozen=True, slots=True)
class ResolvedIdentity:
    """One current-frame observation assigned to a canonical identity."""

    candidate: IdentityCandidate
    identity_id: int
    confirmed: bool
    swimmer_id: int | None = None


@dataclass(frozen=True, slots=True)
class IdentityResolution:
    """Resolved observations plus the monotonic and current identity counts."""

    assignments: list[ResolvedIdentity]
    confirmed_count: int
    active_count: int


@dataclass(frozen=True, slots=True)
class _ProjectedCandidate:
    candidate: IdentityCandidate
    lane_x: float
    position: float


@dataclass(slots=True)
class _Identity:
    identity_id: int
    lane_id: str
    first_seen_ms: float
    last_seen_ms: float
    lane_x: float
    position: float
    box: BoundingBox
    minimum_position: float = 0.0
    maximum_position: float = 0.0
    observations: int = 1
    confidence_sum: float = 0.0
    velocity_lane_x: float = 0.0
    velocity_position: float = 0.0
    confirmed: bool = False
    requires_cooccurrence: bool = False
    cooccurrence_observations: int = 0
    cooccurrence_confidence_sum: float = 0.0
    first_cooccurrence_ms: float | None = None
    last_cooccurrence_ms: float | None = None
    cooccurrence_minimum_position: float | None = None
    cooccurrence_maximum_position: float | None = None
    swimmer_id: int | None = None

    def __post_init__(self) -> None:
        self.confidence_sum = float(self.box.conf)
        self.minimum_position = self.position
        self.maximum_position = self.position


def _box_iou(left: BoundingBox, right: BoundingBox) -> float:
    intersection_width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    intersection_height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left.x2 - left.x1) * max(0.0, left.y2 - left.y1)
    right_area = max(0.0, right.x2 - right.x1) * max(0.0, right.y2 - right.y1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


class IdentityResolver:
    """Associate detector observations to at most two swimmers per lane.

    The fixed camera currently observes a single lane and the product scenarios
    contain one or two swimmers.  The limit is configurable, but it is also a
    useful safety gate: ambiguous duplicate detections must not manufacture an
    unlimited number of people.
    """

    _OCCLUSION_POSITION_DELTA = 0.18
    _OCCLUSION_HOLD_SECONDS = 1.5

    def __init__(
        self,
        *,
        calibration_id: str | None,
        confirmation_observations: int,
        confirmation_seconds: float,
        confirmation_confidence: float,
        tentative_max_gap_seconds: float,
        max_reassociation_gap_seconds: float,
        max_speed_per_second: float,
        position_slack: float,
        max_lane_x_delta: float,
        duplicate_iou: float,
        duplicate_position_delta: float,
        duplicate_lane_x_delta: float,
        additional_confirmation_observations: int,
        additional_confirmation_seconds: float,
        additional_confirmation_confidence: float,
        additional_min_position_span: float,
        additional_cooccurrence_max_gap_seconds: float,
        max_per_lane: int,
    ) -> None:
        self.confirmation_observations = confirmation_observations
        self.confirmation_seconds = confirmation_seconds
        self.confirmation_confidence = confirmation_confidence
        self.tentative_max_gap_seconds = tentative_max_gap_seconds
        self.max_reassociation_gap_seconds = max_reassociation_gap_seconds
        self.max_speed_per_second = max_speed_per_second
        self.position_slack = position_slack
        self.max_lane_x_delta = max_lane_x_delta
        self.duplicate_iou = duplicate_iou
        self.duplicate_position_delta = duplicate_position_delta
        self.duplicate_lane_x_delta = duplicate_lane_x_delta
        self.additional_confirmation_observations = additional_confirmation_observations
        self.additional_confirmation_seconds = additional_confirmation_seconds
        self.additional_confirmation_confidence = additional_confirmation_confidence
        self.additional_min_position_span = additional_min_position_span
        self.additional_cooccurrence_max_gap_seconds = additional_cooccurrence_max_gap_seconds
        self.max_per_lane = max_per_lane
        self._matrices = (
            {
                calibration.lane_id: perspective_matrix(calibration)
                for calibration in lanes_for_calibration(calibration_id)
            }
            if calibration_id is not None
            else {}
        )
        self._identities: dict[str, list[_Identity]] = {}
        self._occlusion_until_ms: dict[str, float] = {}
        self._next_identity_id = 1
        self._next_swimmer_id = 1

    def resolve(
        self,
        *,
        time_ms: float,
        width: int,
        height: int,
        candidates: list[IdentityCandidate],
    ) -> IdentityResolution:
        """Return one-to-one assignments for the supplied current-frame boxes."""

        if width <= 0 or height <= 0:
            raise ValueError("identity resolution requires positive image dimensions")
        projected_by_lane: dict[str, list[_ProjectedCandidate]] = {}
        for candidate in candidates:
            projected = self._project(candidate, width, height)
            projected_by_lane.setdefault(candidate.lane_id, []).append(projected)

        assignments: list[ResolvedIdentity] = []
        for lane_id, projected_candidates in projected_by_lane.items():
            lane_assignments = self._resolve_lane(
                lane_id,
                time_ms,
                self._deduplicate(projected_candidates),
            )
            assignments.extend(lane_assignments)

        confirmed_count = sum(identity.confirmed for identities in self._identities.values() for identity in identities)
        active_count = sum(assignment.confirmed for assignment in assignments)
        return IdentityResolution(
            assignments=assignments,
            confirmed_count=confirmed_count,
            active_count=active_count,
        )

    def _project(self, candidate: IdentityCandidate, width: int, height: int) -> _ProjectedCandidate:
        center_x = (candidate.box.x1 + candidate.box.x2) / (2.0 * width)
        center_y = (candidate.box.y1 + candidate.box.y2) / (2.0 * height)
        matrix = self._matrices.get(candidate.lane_id)
        if matrix is None:
            return _ProjectedCandidate(candidate, lane_x=center_x, position=center_y)
        center = np.asarray([[[center_x, center_y]]], dtype=np.float32)
        lane_x, position = cv2.perspectiveTransform(center, matrix)[0, 0]
        return _ProjectedCandidate(candidate, lane_x=float(lane_x), position=float(position))

    def _deduplicate(self, candidates: list[_ProjectedCandidate]) -> list[_ProjectedCandidate]:
        """Keep one observation for a physical-looking box before association."""

        def priority(item: _ProjectedCandidate) -> tuple[int, int, float]:
            # A detector box paired with a raw track is the most useful current
            # observation; a tracker-only prediction comes next, then confidence.
            return (
                int(item.candidate.track_id is not None),
                int(item.candidate.source == "detection"),
                float(item.candidate.box.conf),
            )

        kept: list[_ProjectedCandidate] = []
        for candidate in sorted(candidates, key=priority, reverse=True):
            if any(self._looks_duplicate(candidate, previous) for previous in kept):
                continue
            kept.append(candidate)
        return kept

    def _looks_duplicate(self, left: _ProjectedCandidate, right: _ProjectedCandidate) -> bool:
        if _box_iou(left.candidate.box, right.candidate.box) >= self.duplicate_iou:
            return True
        return (
            abs(left.position - right.position) < self.duplicate_position_delta
            and abs(left.lane_x - right.lane_x) < self.duplicate_lane_x_delta
        )

    def _resolve_lane(
        self,
        lane_id: str,
        time_ms: float,
        candidates: list[_ProjectedCandidate],
    ) -> list[ResolvedIdentity]:
        identities = self._identities.setdefault(lane_id, [])
        self._expire_tentative(identities, time_ms)
        if not candidates:
            return []

        # When two confirmed swimmers overlap, a detector can legitimately
        # collapse them into one physical-looking observation.  Updating just
        # one canonical trajectory in that interval is what later lets the
        # two identities exchange.  Hold both until they separate instead.
        if self._is_two_swimmer_occlusion(lane_id, identities, candidates, time_ms):
            return []

        matched_candidates: set[int] = set()
        matched_identities: set[int] = set()
        assigned: dict[int, _Identity] = {}
        if identities:
            costs = np.full((len(identities), len(candidates)), 10.0, dtype=np.float64)
            for identity_index, identity in enumerate(identities):
                for candidate_index, candidate in enumerate(candidates):
                    cost = self._matching_cost(identity, candidate, time_ms)
                    if cost is not None:
                        costs[identity_index, candidate_index] = cost
            rows, columns = linear_sum_assignment(costs)
            for identity_index, candidate_index in zip(rows, columns, strict=True):
                if costs[identity_index, candidate_index] >= 1.0:
                    continue
                identity = identities[int(identity_index)]
                candidate = candidates[int(candidate_index)]
                self._update_identity(identity, candidate, time_ms)
                matched_candidates.add(int(candidate_index))
                matched_identities.add(int(identity_index))
                assigned[int(candidate_index)] = identity

        unmatched = [index for index in range(len(candidates)) if index not in matched_candidates]
        # A lone dormant trajectory is much stronger evidence than a new person
        # after a detector gap.  This is what stitches the long test08 gaps while
        # still allowing a second swimmer to be born when both are visible.
        if len(identities) == 1 and not matched_identities and unmatched:
            candidate_index = unmatched.pop(0)
            self._update_identity(identities[0], candidates[candidate_index], time_ms)
            matched_candidates.add(candidate_index)
            matched_identities.add(0)
            assigned[candidate_index] = identities[0]

        # At the two-swimmer limit, one clearly matched swimmer plus exactly
        # one remaining detector observation identifies the missing swimmer by
        # exclusion.  This is intentionally narrower than loosening the normal
        # motion gate: it recovers from an occlusion without admitting a third
        # detection as a new person.
        unmatched_identity_indices = [
            index for index in range(len(identities)) if index not in matched_identities
        ]
        unmatched_detector_indices = [
            index
            for index in unmatched
            if candidates[index].candidate.source == "detection"
            and candidates[index].candidate.box.conf >= self.confirmation_confidence
        ]
        if (
            len(identities) == self.max_per_lane
            and all(identity.confirmed for identity in identities)
            and matched_identities
            and len(unmatched_identity_indices) == 1
            and len(unmatched_detector_indices) == 1
        ):
            identity_index = unmatched_identity_indices[0]
            candidate_index = unmatched_detector_indices[0]
            identity = identities[identity_index]
            self._update_identity(identity, candidates[candidate_index], time_ms)
            matched_candidates.add(candidate_index)
            matched_identities.add(identity_index)
            assigned[candidate_index] = identity
            unmatched.remove(candidate_index)

        for candidate_index in unmatched:
            if len(identities) >= self.max_per_lane:
                # Do not turn an unresolved duplicate into a third swimmer.  The
                # next frames may provide enough motion evidence to match it.
                continue
            identity = self._new_identity(
                lane_id,
                candidates[candidate_index],
                time_ms,
                requires_cooccurrence=any(item.confirmed for item in identities),
            )
            identities.append(identity)
            matched_candidates.add(candidate_index)
            matched_identities.add(len(identities) - 1)
            assigned[candidate_index] = identity

        active_confirmed = any(identity.confirmed for identity in assigned.values())
        for candidate_index, identity in assigned.items():
            candidate = candidates[candidate_index]
            if identity.requires_cooccurrence and not identity.confirmed and active_confirmed:
                self._record_cooccurrence(identity, candidate, time_ms)
            # A tracker prediction can maintain an established identity, but it
            # is not sufficient evidence to create a new physical swimmer.
            if not identity.requires_cooccurrence or identity.confirmed or candidate.candidate.source == "detection":
                self._promote_if_ready(identity, time_ms)

        result: list[ResolvedIdentity] = []
        for candidate_index in sorted(matched_candidates):
            candidate = candidates[candidate_index]
            identity = assigned[candidate_index]
            result.append(
                ResolvedIdentity(
                    candidate=candidate.candidate,
                    identity_id=identity.identity_id,
                    confirmed=identity.confirmed,
                    swimmer_id=identity.swimmer_id,
                )
            )
        return result

    def _is_two_swimmer_occlusion(
        self,
        lane_id: str,
        identities: list[_Identity],
        candidates: list[_ProjectedCandidate],
        time_ms: float,
    ) -> bool:
        """Hold an ambiguous one-observation crossing until separation."""

        occlusion_until_ms = self._occlusion_until_ms.get(lane_id, 0.0)
        if occlusion_until_ms > time_ms:
            if len(candidates) < 2:
                return True
            self._occlusion_until_ms.pop(lane_id, None)
            return False
        self._occlusion_until_ms.pop(lane_id, None)
        if len(identities) != 2 or len(candidates) >= 2 or not all(identity.confirmed for identity in identities):
            return False

        first, second = identities
        first_elapsed = max(0.0, time_ms - first.last_seen_ms) / 1000.0
        second_elapsed = max(0.0, time_ms - second.last_seen_ms) / 1000.0
        first_predicted = first.position + first.velocity_position * first_elapsed
        second_predicted = second.position + second.velocity_position * second_elapsed
        are_approaching = first.velocity_position * second.velocity_position < 0.0
        if not are_approaching or abs(first_predicted - second_predicted) > self._OCCLUSION_POSITION_DELTA:
            return False

        self._occlusion_until_ms[lane_id] = time_ms + self._OCCLUSION_HOLD_SECONDS * 1000.0
        return True

    def _record_cooccurrence(
        self,
        identity: _Identity,
        candidate: _ProjectedCandidate,
        time_ms: float,
    ) -> None:
        if (
            identity.last_cooccurrence_ms is None
            or (time_ms - identity.last_cooccurrence_ms) / 1000.0 > self.additional_cooccurrence_max_gap_seconds
        ):
            identity.first_cooccurrence_ms = None
            identity.last_cooccurrence_ms = None
            identity.cooccurrence_observations = 0
            identity.cooccurrence_confidence_sum = 0.0
            identity.cooccurrence_minimum_position = None
            identity.cooccurrence_maximum_position = None
        if candidate.candidate.source != "detection":
            return
        if identity.first_cooccurrence_ms is None:
            identity.first_cooccurrence_ms = time_ms
        identity.cooccurrence_observations += 1
        identity.cooccurrence_confidence_sum += float(candidate.candidate.box.conf)
        identity.last_cooccurrence_ms = time_ms
        identity.cooccurrence_minimum_position = min(
            identity.cooccurrence_minimum_position
            if identity.cooccurrence_minimum_position is not None
            else identity.position,
            identity.position,
        )
        identity.cooccurrence_maximum_position = max(
            identity.cooccurrence_maximum_position
            if identity.cooccurrence_maximum_position is not None
            else identity.position,
            identity.position,
        )

    def _matching_cost(
        self,
        identity: _Identity,
        candidate: _ProjectedCandidate,
        time_ms: float,
    ) -> float | None:
        elapsed_seconds = max(0.0, time_ms - identity.last_seen_ms) / 1000.0
        if elapsed_seconds > self.max_reassociation_gap_seconds:
            return None
        predicted_position = identity.position + identity.velocity_position * elapsed_seconds
        predicted_lane_x = identity.lane_x + identity.velocity_lane_x * elapsed_seconds
        position_error = abs(candidate.position - predicted_position)
        lane_x_error = abs(candidate.lane_x - predicted_lane_x)
        allowed_position_error = min(1.25, self.position_slack + self.max_speed_per_second * elapsed_seconds)
        allowed_lane_x_error = min(0.5, self.max_lane_x_delta + 0.02 * elapsed_seconds)
        if position_error > allowed_position_error or lane_x_error > allowed_lane_x_error:
            return None
        iou_cost = 1.0 - _box_iou(identity.box, candidate.candidate.box)
        identity_area = max(1.0, (identity.box.x2 - identity.box.x1) * (identity.box.y2 - identity.box.y1))
        candidate_area = max(
            1.0,
            (candidate.candidate.box.x2 - candidate.candidate.box.x1)
            * (candidate.candidate.box.y2 - candidate.candidate.box.y1),
        )
        scale_cost = min(1.0, abs(np.log(candidate_area / identity_area)))
        return max(
            0.0,
            0.62 * position_error / max(allowed_position_error, 1e-6)
            + 0.18 * lane_x_error / max(allowed_lane_x_error, 1e-6)
            + 0.10 * iou_cost
            + 0.05 * scale_cost
        )

    def _new_identity(
        self,
        lane_id: str,
        candidate: _ProjectedCandidate,
        time_ms: float,
        *,
        requires_cooccurrence: bool,
    ) -> _Identity:
        identity = _Identity(
            identity_id=self._next_identity_id,
            lane_id=lane_id,
            first_seen_ms=time_ms,
            last_seen_ms=time_ms,
            lane_x=candidate.lane_x,
            position=candidate.position,
            box=candidate.candidate.box,
            requires_cooccurrence=requires_cooccurrence,
        )
        self._next_identity_id += 1
        return identity

    def _update_identity(
        self,
        identity: _Identity,
        candidate: _ProjectedCandidate,
        time_ms: float,
    ) -> None:
        elapsed_seconds = max(0.0, time_ms - identity.last_seen_ms) / 1000.0
        if elapsed_seconds > 0:
            observed_position_velocity = (candidate.position - identity.position) / elapsed_seconds
            observed_lane_x_velocity = (candidate.lane_x - identity.lane_x) / elapsed_seconds
            limit = self.max_speed_per_second * 2.0
            observed_position_velocity = float(np.clip(observed_position_velocity, -limit, limit))
            observed_lane_x_velocity = float(np.clip(observed_lane_x_velocity, -limit, limit))
            # Use the latest observation as the dominant term.  A heavily
            # smoothed velocity lags behind a swimmer after a short occlusion,
            # which makes the nearest-position assignment prefer the swimmer
            # travelling in the opposite direction after they cross.
            identity.velocity_position = 0.3 * identity.velocity_position + 0.7 * observed_position_velocity
            identity.velocity_lane_x = 0.3 * identity.velocity_lane_x + 0.7 * observed_lane_x_velocity
        identity.last_seen_ms = time_ms
        identity.position = candidate.position
        identity.minimum_position = min(identity.minimum_position, candidate.position)
        identity.maximum_position = max(identity.maximum_position, candidate.position)
        identity.lane_x = candidate.lane_x
        identity.box = candidate.candidate.box
        identity.observations += 1
        identity.confidence_sum += float(candidate.candidate.box.conf)

    def _promote_if_ready(self, identity: _Identity, time_ms: float) -> None:
        elapsed_seconds = (time_ms - identity.first_seen_ms) / 1000.0
        mean_confidence = identity.confidence_sum / identity.observations
        cooccurrence_elapsed_seconds = (
            (time_ms - identity.first_cooccurrence_ms) / 1000.0 if identity.first_cooccurrence_ms is not None else 0.0
        )
        cooccurrence_mean_confidence = (
            identity.cooccurrence_confidence_sum / identity.cooccurrence_observations
            if identity.cooccurrence_observations
            else 0.0
        )
        if (
            identity.observations >= self.confirmation_observations
            and elapsed_seconds >= self.confirmation_seconds
            and mean_confidence >= self.confirmation_confidence
            and (
                not identity.requires_cooccurrence
                or (
                    identity.cooccurrence_observations >= self.additional_confirmation_observations
                    and cooccurrence_elapsed_seconds >= self.additional_confirmation_seconds
                    and cooccurrence_mean_confidence >= self.additional_confirmation_confidence
                    and identity.cooccurrence_minimum_position is not None
                    and identity.cooccurrence_maximum_position is not None
                    and identity.cooccurrence_maximum_position - identity.cooccurrence_minimum_position
                    >= self.additional_min_position_span
                )
            )
        ) and not identity.confirmed:
            identity.confirmed = True
            identity.swimmer_id = self._next_swimmer_id
            self._next_swimmer_id += 1

    def _expire_tentative(self, identities: list[_Identity], time_ms: float) -> None:
        identities[:] = [
            identity
            for identity in identities
            if identity.confirmed or (time_ms - identity.last_seen_ms) / 1000.0 <= self.tentative_max_gap_seconds
        ]
