from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import prod, sqrt

import cv2
import numpy as np

from swimtrack_ai.calibration import (
    FIXED_CAMERA_CALIBRATION_ID as FIXED_CAMERA_CALIBRATION_ID,
)
from swimtrack_ai.calibration import (
    FIXED_CAMERA_CENTER_LANE,
    LaneCalibration,
    lanes_for_calibration,
    perspective_matrix,
)
from swimtrack_ai.schemas import BoundingBox, LaneLapScore, LapEvidence

LAP_SCORE_VERSION = "trajectory-v4"


@dataclass(frozen=True, slots=True)
class _Observation:
    time_ms: float
    position: float | None
    confidence: float | None
    track_id: int | None


@dataclass(slots=True)
class _LaneRuntime:
    calibration: LaneCalibration
    image_to_lane: np.ndarray
    history: deque[_Observation] = field(default_factory=deque)
    armed_since_ms: float | None = None
    episodes: deque[_WallEpisode] = field(default_factory=deque)
    active_episode: _WallEpisode | None = None
    next_episode_id: int = 1


@dataclass(slots=True)
class _WallEpisode:
    episode_id: int
    endpoint: str
    start_ms: float
    end_ms: float | None = None


@dataclass(frozen=True, slots=True)
class _CandidateScore:
    score: float
    endpoint: str
    candidate_time_ms: float
    episode_id: int
    track_id: int | None
    evidence: LapEvidence


def fixed_camera_visible_polygon() -> tuple[tuple[float, float], ...]:
    """Return the normalized polygon of the only lane visible end-to-end."""

    return FIXED_CAMERA_CENTER_LANE.visible_polygon


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _slope(observations: list[_Observation]) -> float | None:
    valid = [item for item in observations if item.position is not None]
    if len(valid) < 3 or valid[-1].time_ms - valid[0].time_ms < 300:
        return None
    times = np.asarray([(item.time_ms - valid[0].time_ms) / 1000.0 for item in valid], dtype=float)
    positions = np.asarray([item.position for item in valid], dtype=float)
    centered_times = times - times.mean()
    denominator = float(np.dot(centered_times, centered_times))
    if denominator <= 1e-9:
        return None
    return float(np.dot(centered_times, positions - positions.mean()) / denominator)


def _observation_quality(observations: list[_Observation]) -> float:
    valid = [item for item in observations if item.position is not None]
    if not observations or not valid:
        return 0.0
    coverage = len(valid) / len(observations)
    mean_confidence = float(np.mean([item.confidence for item in valid]))
    return _clip01(coverage * sqrt(max(0.0, mean_confidence)))


class LapAnalyzer:
    """Produce a heuristic lap/no-lap score from tracked swimmer trajectories.

    The score is intentionally not a calibrated probability. It records the
    geometric and temporal evidence needed to fit weights and a decision
    threshold after ground-truth annotations exist.
    """

    history_ms = 10_000.0
    context_before_ms = 1_000.0
    context_after_ms = 1_000.0
    max_side_gap_ms = 6_000.0
    candidate_guard_ms = 100.0
    endpoint_zone = 0.15
    reference_speed = 0.15
    reference_departure = 0.08
    lane_margin = 0.05
    interior_zone = (0.20, 0.80)

    def __init__(self, fps: float, calibration_id: str) -> None:
        self.fps = fps
        self._lanes = {
            calibration.lane_id: _LaneRuntime(
                calibration=calibration,
                image_to_lane=perspective_matrix(calibration),
            )
            for calibration in lanes_for_calibration(calibration_id)
        }

    def observe(
        self,
        *,
        time_ms: float,
        width: int,
        height: int,
        boxes: list[BoundingBox],
    ) -> list[LaneLapScore]:
        results: list[LaneLapScore] = []
        for runtime in self._lanes.values():
            if runtime.history and time_ms < runtime.history[-1].time_ms:
                raise ValueError("Lap observations must not move backwards in time")
            observation = self._select_observation(runtime, time_ms, width, height, boxes)
            runtime.history.append(observation)
            self._update_wall_episode(runtime, observation)
            cutoff = time_ms - self.history_ms
            while runtime.history and runtime.history[0].time_ms < cutoff:
                runtime.history.popleft()
            while runtime.episodes and runtime.episodes[0].end_ms is not None and runtime.episodes[0].end_ms < cutoff:
                runtime.episodes.popleft()
            results.append(self._score(runtime, observation))
        return results

    def _update_wall_episode(self, runtime: _LaneRuntime, observation: _Observation) -> None:
        if observation.position is None:
            return
        position = observation.position
        if self.interior_zone[0] <= position <= self.interior_zone[1]:
            if runtime.armed_since_ms is None:
                runtime.armed_since_ms = observation.time_ms
            if runtime.active_episode is not None:
                runtime.active_episode.end_ms = observation.time_ms
                runtime.active_episode = None
            return
        endpoint = "far" if position <= self.endpoint_zone else "near" if position >= 1.0 - self.endpoint_zone else None
        if endpoint is None or runtime.armed_since_ms is None or observation.time_ms <= runtime.armed_since_ms:
            return
        if runtime.active_episode is not None and runtime.active_episode.endpoint == endpoint:
            return
        if runtime.active_episode is not None:
            runtime.active_episode.end_ms = observation.time_ms
        episode = _WallEpisode(
            episode_id=runtime.next_episode_id,
            endpoint=endpoint,
            start_ms=observation.time_ms,
        )
        runtime.next_episode_id += 1
        runtime.episodes.append(episode)
        runtime.active_episode = episode

    def _select_observation(
        self,
        runtime: _LaneRuntime,
        time_ms: float,
        width: int,
        height: int,
        boxes: list[BoundingBox],
    ) -> _Observation:
        candidates: list[tuple[float, float, BoundingBox]] = []
        recent_position = next(
            (
                item.position
                for item in reversed(runtime.history)
                if item.position is not None and time_ms - item.time_ms <= self.max_side_gap_ms
            ),
            None,
        )
        for box in boxes:
            center = np.asarray(
                [[[(box.x1 + box.x2) / (2.0 * width), (box.y1 + box.y2) / (2.0 * height)]]],
                dtype=np.float32,
            )
            lane_x, position = cv2.perspectiveTransform(center, runtime.image_to_lane)[0, 0]
            if (
                -self.lane_margin <= lane_x <= 1.0 + self.lane_margin
                and -self.lane_margin <= position <= 1.0 + self.lane_margin
            ):
                center_penalty = abs(float(lane_x) - 0.5) * 0.05
                continuity_penalty = (
                    abs(float(position) - recent_position) * 0.25 if recent_position is not None else 0.0
                )
                candidates.append((float(box.conf) - center_penalty - continuity_penalty, float(position), box))

        if not candidates:
            return _Observation(time_ms=time_ms, position=None, confidence=None, track_id=None)

        _rank, position, box = max(candidates, key=lambda item: item[0])
        return _Observation(
            time_ms=time_ms,
            position=_clip01(position),
            confidence=_clip01(box.conf),
            track_id=box.id if box.id > 0 else None,
        )

    def _score(self, runtime: _LaneRuntime, current: _Observation) -> LaneLapScore:
        history = list(runtime.history)
        valid = [item for item in history if item.position is not None]
        quality = _observation_quality(history)
        span_ms = history[-1].time_ms - history[0].time_ms if len(history) > 1 else 0.0
        min_samples = max(5, round(self.fps * 0.5))
        evaluable = span_ms >= self.context_before_ms + self.context_after_ms and len(valid) >= min_samples

        best = self._best_candidate(
            history,
            history[0].time_ms,
            history[-1].time_ms,
            list(runtime.episodes),
        )
        lap_score = best.score if best is not None and evaluable else 0.0
        no_lap_score = 1.0 - lap_score if evaluable else None
        score_quality = best.evidence.track_quality if best is not None else quality
        empty_evidence = LapEvidence(
            wall=0.0,
            approach=0.0,
            reversal=0.0,
            departure=0.0,
            track_quality=quality,
        )
        return LaneLapScore(
            lane_id=runtime.calibration.lane_id,
            track_id=best.track_id if best is not None else current.track_id,
            lap_score=_clip01(lap_score),
            no_lap_score=_clip01(no_lap_score) if no_lap_score is not None else None,
            observation_quality=score_quality,
            evaluable=evaluable,
            longitudinal_position=current.position,
            endpoint=best.endpoint if best is not None else None,
            candidate_time_ms=best.candidate_time_ms if best is not None else None,
            candidate_episode_id=best.episode_id if best is not None else None,
            window_start_ms=history[0].time_ms,
            window_end_ms=history[-1].time_ms,
            score_version=LAP_SCORE_VERSION,
            evidence=best.evidence if best is not None else empty_evidence,
        )

    def _best_candidate(
        self,
        history: list[_Observation],
        window_start_ms: float,
        window_end_ms: float,
        episodes: list[_WallEpisode],
    ) -> _CandidateScore | None:
        if not episodes:
            return None
        latest_episode = episodes[-1]
        valid = [item for item in history if item.position is not None]
        best: _CandidateScore | None = None
        for candidate in valid:
            if (
                candidate.time_ms < window_start_ms + self.context_before_ms
                or candidate.time_ms > window_end_ms - self.context_after_ms
            ):
                continue
            assert candidate.position is not None
            endpoint_distances = {"far": candidate.position, "near": 1.0 - candidate.position}
            for endpoint, wall_distance in endpoint_distances.items():
                if wall_distance > self.endpoint_zone:
                    continue
                episode = self._episode_for_candidate([latest_episode], endpoint, candidate.time_ms)
                if episode is None:
                    continue
                scored = self._score_candidate(history, candidate, endpoint, wall_distance, episode.episode_id)
                if scored is not None and (best is None or scored.score > best.score):
                    best = scored
        return best

    @staticmethod
    def _episode_for_candidate(
        episodes: list[_WallEpisode],
        endpoint: str,
        candidate_time_ms: float,
    ) -> _WallEpisode | None:
        for episode in reversed(episodes):
            if episode.endpoint != endpoint or candidate_time_ms < episode.start_ms:
                continue
            if episode.end_ms is None or candidate_time_ms <= episode.end_ms:
                return episode
        return None

    def _score_candidate(
        self,
        history: list[_Observation],
        candidate: _Observation,
        endpoint: str,
        wall_distance: float,
        episode_id: int,
    ) -> _CandidateScore | None:
        before = self._side_window(history, candidate.time_ms, before=True)
        after = self._side_window(history, candidate.time_ms, before=False)
        approach_slope = _slope(before)
        departure_slope = _slope(after)
        if approach_slope is None or departure_slope is None:
            return None

        direction = -1.0 if endpoint == "far" else 1.0
        inbound_speed = direction * approach_slope
        outbound_speed = -direction * departure_slope
        approach = _clip01(inbound_speed / self.reference_speed)
        outbound = _clip01(outbound_speed / self.reference_speed)
        reversal = sqrt(approach * outbound)

        assert candidate.position is not None
        valid_after = [item for item in after if item.position is not None]
        tail_count = max(1, len(valid_after) // 4)
        departure_position = float(np.median([item.position for item in valid_after[-tail_count:]]))
        departure_distance = (
            departure_position - candidate.position if endpoint == "far" else candidate.position - departure_position
        )
        departure = _clip01(departure_distance / self.reference_departure)
        wall = _clip01((self.endpoint_zone - wall_distance) / self.endpoint_zone)
        valid_before = [item for item in before if item.position is not None]
        before_gap_ms = candidate.time_ms - valid_before[-1].time_ms
        after_gap_ms = valid_after[0].time_ms - candidate.time_ms
        gap_quality = sqrt(
            min(1.0, self.context_before_ms / max(self.context_before_ms, before_gap_ms))
            * min(1.0, self.context_after_ms / max(self.context_after_ms, after_gap_ms))
        )
        quality = _observation_quality([*before, candidate, *after]) * gap_quality
        components = (wall, approach, reversal, departure)
        evidence_score = prod(components) ** (1.0 / len(components)) if all(value > 0 for value in components) else 0.0
        evidence = LapEvidence(
            wall=wall,
            approach=approach,
            reversal=reversal,
            departure=departure,
            track_quality=quality,
        )
        return _CandidateScore(
            score=_clip01(evidence_score * quality),
            endpoint=endpoint,
            candidate_time_ms=candidate.time_ms,
            episode_id=episode_id,
            track_id=candidate.track_id,
            evidence=evidence,
        )

    def _side_window(
        self,
        history: list[_Observation],
        candidate_time_ms: float,
        *,
        before: bool,
    ) -> list[_Observation]:
        if before:
            valid = [
                item
                for item in history
                if item.position is not None
                and candidate_time_ms - self.max_side_gap_ms
                <= item.time_ms
                <= candidate_time_ms - self.candidate_guard_ms
            ]
            if not valid:
                return []
            window_end_ms = valid[-1].time_ms
            window_start_ms = window_end_ms - self.context_before_ms
        else:
            valid = [
                item
                for item in history
                if item.position is not None
                and candidate_time_ms + self.candidate_guard_ms
                <= item.time_ms
                <= candidate_time_ms + self.max_side_gap_ms
            ]
            if not valid:
                return []
            window_start_ms = valid[0].time_ms
            window_end_ms = window_start_ms + self.context_after_ms
        return [item for item in history if window_start_ms <= item.time_ms <= window_end_ms]
