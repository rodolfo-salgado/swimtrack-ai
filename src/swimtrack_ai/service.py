from __future__ import annotations

import hashlib
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from swimtrack_ai.calibration import LaneRouter
from swimtrack_ai.config import Settings
from swimtrack_ai.detectors import Detector, DetectorResult
from swimtrack_ai.errors import ConflictError, SessionCapacityError, SessionNotFoundError
from swimtrack_ai.identity import IdentityCandidate, IdentityResolver
from swimtrack_ai.lap_analysis import LapAnalyzer
from swimtrack_ai.schemas import (
    BatchMetadata,
    BatchResult,
    BoundingBox,
    DiagnosticBox,
    DiagnosticsLevel,
    DiagnosticStage,
    FrameResult,
    FrameTrackingDiagnostics,
    IdentitySummary,
    LaneTrackingDiagnostics,
    SessionCreated,
    TrackingConfiguration,
)
from swimtrack_ai.tracker import Tracker, TrackerUpdate


@dataclass(slots=True)
class CachedBatch:
    fingerprint: str
    result: BatchResult


@dataclass(slots=True)
class SessionState:
    session_id: str
    trackers: dict[str, Tracker]
    lane_router: LaneRouter
    lap_analyzer: LapAnalyzer | None
    identity_resolver: IdentityResolver
    diagnostics_level: DiagnosticsLevel
    weak_reactivation_enabled: bool
    weak_reactivation_max_gap_frames: int
    expires_at: float
    next_sequence: int = 0
    last_frame_index: int | None = None
    last_time_ms: float | None = None
    poisoned_reason: str | None = None
    cache: OrderedDict[str, CachedBatch] = field(default_factory=OrderedDict)
    lock: threading.Lock = field(default_factory=threading.Lock)


class TrackingService:
    def __init__(
        self,
        settings: Settings,
        detector: Detector,
        tracker_factory: Callable[[float], Tracker],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self.detector = detector
        self.tracker_factory = tracker_factory
        self.clock = clock
        self._sessions: dict[str, SessionState] = {}
        self._sessions_lock = threading.RLock()

    def create_session(
        self,
        fps: float,
        lap_calibration_id: str | None = None,
        diagnostics: DiagnosticsLevel = "none",
    ) -> SessionCreated:
        self.expire_sessions()
        tracker_frame_rate = max(1, round(fps))
        effective_buffer_frames = int(tracker_frame_rate / 30.0 * self.settings.track_buffer)
        weak_reactivation_max_gap_frames = max(
            1,
            round(tracker_frame_rate * self.settings.weak_reactivation_max_gap_seconds),
        )
        with self._sessions_lock:
            if len(self._sessions) >= self.settings.max_sessions:
                raise SessionCapacityError("Maximum number of active tracking sessions reached")
            session_id = str(uuid.uuid4())
            far_crop_box = (
                self.settings.far_crop_box
                if self.settings.far_crop_enabled and lap_calibration_id is not None
                else None
            )
            lane_router = LaneRouter(
                lap_calibration_id,
                enabled=self.settings.lane_roi_enabled,
                far_crop_box=far_crop_box,
            )
            weak_reactivation_enabled = self.settings.weak_reactivation_enabled and lane_router.enabled
            self._sessions[session_id] = SessionState(
                session_id=session_id,
                trackers={lane_id: self.tracker_factory(fps) for lane_id in lane_router.lane_ids},
                lane_router=lane_router,
                lap_analyzer=LapAnalyzer(fps, lap_calibration_id) if lap_calibration_id is not None else None,
                identity_resolver=IdentityResolver(
                    calibration_id=lap_calibration_id,
                    confirmation_observations=self.settings.identity_confirmation_observations,
                    confirmation_seconds=self.settings.identity_confirmation_seconds,
                    confirmation_confidence=self.settings.identity_confirmation_confidence,
                    tentative_max_gap_seconds=self.settings.identity_tentative_max_gap_seconds,
                    max_reassociation_gap_seconds=self.settings.identity_max_reassociation_gap_seconds,
                    max_speed_per_second=self.settings.identity_max_speed_per_second,
                    position_slack=self.settings.identity_position_slack,
                    max_lane_x_delta=self.settings.identity_max_lane_x_delta,
                    duplicate_iou=self.settings.identity_duplicate_iou,
                    duplicate_position_delta=self.settings.identity_duplicate_position_delta,
                    duplicate_lane_x_delta=self.settings.identity_duplicate_lane_x_delta,
                    additional_confirmation_observations=(self.settings.identity_additional_confirmation_observations),
                    additional_confirmation_seconds=self.settings.identity_additional_confirmation_seconds,
                    additional_confirmation_confidence=(self.settings.identity_additional_confirmation_confidence),
                    additional_min_position_span=self.settings.identity_additional_min_position_span,
                    additional_cooccurrence_max_gap_seconds=(
                        self.settings.identity_additional_cooccurrence_max_gap_seconds
                    ),
                    max_per_lane=self.settings.identity_max_per_lane,
                ),
                diagnostics_level=diagnostics,
                weak_reactivation_enabled=weak_reactivation_enabled,
                weak_reactivation_max_gap_frames=weak_reactivation_max_gap_frames,
                expires_at=self.clock() + self.settings.session_ttl_seconds,
            )
        return SessionCreated(
            session_id=session_id,
            next_sequence=0,
            expires_in_seconds=self.settings.session_ttl_seconds,
            tracking_configuration=(
                TrackingConfiguration(
                    diagnostic_score_floor=self.settings.diagnostic_score_floor,
                    score_threshold=self.settings.score_threshold,
                    min_box_area=self.settings.min_box_area,
                    track_threshold=self.settings.track_threshold,
                    track_buffer=self.settings.track_buffer,
                    match_threshold=self.settings.match_threshold,
                    mot20=self.settings.mot20,
                    lane_roi_enabled=lane_router.enabled,
                    lane_ids=list(lane_router.lane_ids),
                    far_crop_enabled=far_crop_box is not None,
                    far_crop_box=list(far_crop_box) if far_crop_box is not None else None,
                    far_crop_nms_threshold=self.settings.far_crop_nms_threshold,
                    effective_lost_buffer_frames=effective_buffer_frames,
                    effective_lost_buffer_seconds=effective_buffer_frames / tracker_frame_rate,
                    weak_reactivation_enabled=weak_reactivation_enabled,
                    weak_reactivation_score_threshold=self.settings.weak_reactivation_score_threshold,
                    weak_reactivation_min_box_area=self.settings.weak_reactivation_min_box_area,
                    weak_reactivation_max_gap_frames=weak_reactivation_max_gap_frames,
                    weak_reactivation_max_gap_seconds=self.settings.weak_reactivation_max_gap_seconds,
                    weak_reactivation_max_center_distance=self.settings.weak_reactivation_max_center_distance,
                )
                if diagnostics != "none"
                else None
            ),
        )

    @staticmethod
    def _diagnostic_stage(detections: np.ndarray, level: DiagnosticsLevel) -> DiagnosticStage:
        boxes = None
        if level == "boxes":
            boxes = [
                DiagnosticBox(
                    x1=float(detection[0]),
                    y1=float(detection[1]),
                    x2=float(detection[2]),
                    y2=float(detection[3]),
                    conf=float(detection[4]),
                )
                for detection in detections
            ]
        return DiagnosticStage(count=len(detections), boxes=boxes)

    @staticmethod
    def _offset_detections(detections: np.ndarray, offset: tuple[float, float]) -> np.ndarray:
        if not len(detections):
            return detections.copy()
        remapped = detections.copy()
        remapped[:, [0, 2]] += offset[0]
        remapped[:, [1, 3]] += offset[1]
        return remapped

    @staticmethod
    def _nms(detections: np.ndarray, iou_threshold: float) -> np.ndarray:
        if not len(detections):
            return np.empty((0, 5), dtype=np.float32)
        order = np.argsort(detections[:, 4], kind="stable")[::-1]
        keep: list[int] = []
        while len(order):
            current = int(order[0])
            keep.append(current)
            remaining = order[1:]
            if not len(remaining):
                break
            x1 = np.maximum(detections[current, 0], detections[remaining, 0])
            y1 = np.maximum(detections[current, 1], detections[remaining, 1])
            x2 = np.minimum(detections[current, 2], detections[remaining, 2])
            y2 = np.minimum(detections[current, 3], detections[remaining, 3])
            intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
            current_area = max(0.0, float(detections[current, 2] - detections[current, 0])) * max(
                0.0,
                float(detections[current, 3] - detections[current, 1]),
            )
            remaining_area = np.maximum(0.0, detections[remaining, 2] - detections[remaining, 0]) * np.maximum(
                0.0,
                detections[remaining, 3] - detections[remaining, 1],
            )
            union = current_area + remaining_area - intersection
            iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
            order = remaining[iou <= iou_threshold]
        return detections[np.asarray(keep, dtype=np.int64)].astype(np.float32, copy=False)

    def _merge_detector_results(
        self,
        primary: DetectorResult,
        supplemental: DetectorResult,
        offset: tuple[float, float],
    ) -> DetectorResult:
        crop_candidates = self._offset_detections(supplemental.person_candidates, offset)
        crop_accepted = self._offset_detections(supplemental.accepted, offset)
        candidates = self._nms(
            np.concatenate((primary.person_candidates, crop_candidates), axis=0),
            self.settings.far_crop_nms_threshold,
        )
        accepted = self._nms(
            np.concatenate((primary.accepted, crop_accepted), axis=0),
            self.settings.far_crop_nms_threshold,
        )
        return DetectorResult(person_candidates=candidates, accepted=accepted)

    def _limit_routed_detections(self, routed_detections: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Apply the detection cap only after calibrated lane routing."""

        return {
            lane_id: detections[: self.settings.max_detections] for lane_id, detections in routed_detections.items()
        }

    def _infer_detector_batch(
        self,
        frames: list[np.ndarray],
        target_sizes: list[tuple[int, int]],
    ) -> list[DetectorResult]:
        """Use the detector's batch path without requiring it from legacy test doubles."""

        if len(frames) != len(target_sizes):
            raise ValueError("frames and target_sizes must have the same length")
        infer_batch = getattr(self.detector, "infer_batch", None)
        if callable(infer_batch):
            results = list(infer_batch(frames, target_sizes))
        else:
            results = [self.detector.infer(frame, target_size) for frame, target_size in zip(frames, target_sizes)]
        if len(results) != len(frames):
            raise RuntimeError(f"Detector returned {len(results)} results for {len(frames)} input views")
        return results

    def _far_crop_view(
        self,
        frame: np.ndarray,
        target_size: tuple[int, int],
    ) -> tuple[np.ndarray, tuple[int, int], tuple[float, float]]:
        """Create one far-camera crop and its mapping back to the original image."""

        source_height, source_width = frame.shape[:2]
        left, top, right, bottom = self.settings.far_crop_box
        # The fixed-camera crop coordinates intentionally fall on whole source pixels
        # at 1080p. Guard against their decimal representation landing infinitesimally
        # below or above that pixel; otherwise the crop and its calibrated ROI disagree
        # by one source pixel at the far end.
        coordinate_epsilon = 1e-6
        source_left = int(np.floor(left * source_width + coordinate_epsilon))
        source_top = int(np.floor(top * source_height + coordinate_epsilon))
        source_right = int(np.ceil(right * source_width - coordinate_epsilon))
        source_bottom = int(np.ceil(bottom * source_height - coordinate_epsilon))
        cropped = frame[source_top:source_bottom, source_left:source_right]

        target_width, target_height = target_size
        target_left = round(source_left * target_width / source_width)
        target_top = round(source_top * target_height / source_height)
        target_right = round(source_right * target_width / source_width)
        target_bottom = round(source_bottom * target_height / source_height)
        return (
            cropped,
            (max(1, target_right - target_left), max(1, target_bottom - target_top)),
            (target_left, target_top),
        )

    def _infer_frames(
        self,
        frames: list[np.ndarray],
        metadata: BatchMetadata,
        *,
        use_far_crop: bool,
    ) -> list[DetectorResult]:
        """Batch detector views while preserving frame order for later ByteTrack updates."""

        if len(frames) != len(metadata.frames):
            raise ValueError("frames and metadata.frames must have the same length")
        target_sizes = [(item.original_width, item.original_height) for item in metadata.frames]
        if not self.settings.far_crop_enabled or not use_far_crop:
            return self._infer_detector_batch(frames, target_sizes)

        cropped_views = [self._far_crop_view(frame, target_size) for frame, target_size in zip(frames, target_sizes)]
        detector_results = self._infer_detector_batch(
            [*frames, *(crop[0] for crop in cropped_views)],
            [*target_sizes, *(crop[1] for crop in cropped_views)],
        )
        primary = detector_results[: len(frames)]
        supplemental = detector_results[len(frames) :]
        return [
            self._merge_detector_results(primary_result, supplemental_result, crop[2])
            for primary_result, supplemental_result, crop in zip(primary, supplemental, cropped_views)
        ]

    def _tracking_diagnostics(
        self,
        detector_result: DetectorResult,
        weak_candidates: np.ndarray,
        routed_detections: dict[str, np.ndarray],
        routed_weak_candidates: dict[str, np.ndarray],
        tracker_updates: dict[str, TrackerUpdate],
        level: DiagnosticsLevel,
    ) -> FrameTrackingDiagnostics | None:
        if level == "none":
            return None
        return FrameTrackingDiagnostics(
            diagnostic_floor=self.settings.diagnostic_score_floor,
            person_candidates=self._diagnostic_stage(detector_result.person_candidates, level),
            detector_accepted=self._diagnostic_stage(detector_result.accepted, level),
            weak_candidates=self._diagnostic_stage(weak_candidates, level),
            lanes=[
                LaneTrackingDiagnostics(
                    lane_id=lane_id,
                    after_roi=self._diagnostic_stage(routed_detections[lane_id], level),
                    weak_candidates_after_roi=self._diagnostic_stage(routed_weak_candidates[lane_id], level),
                    active_track_ids=[int(track.track_id) for track in tracker_updates[lane_id].active_tracks],
                    retained_lost_track_count=tracker_updates[lane_id].retained_lost_track_count,
                    weak_reactivated_track_ids=tracker_updates[lane_id].weak_reactivated_track_ids,
                )
                for lane_id in routed_detections
            ],
        )

    def _weak_candidates(self, detector_result: DetectorResult) -> np.ndarray:
        candidates = detector_result.person_candidates
        if not len(candidates):
            return np.empty((0, 5), dtype=np.float32)
        widths = np.maximum(0.0, candidates[:, 2] - candidates[:, 0])
        heights = np.maximum(0.0, candidates[:, 3] - candidates[:, 1])
        areas = widths * heights
        # A candidate accepted by the ordinary detector path may already be
        # associated to an active ByteTrack track in this same update. Keep the
        # manual recovery path strictly below that acceptance threshold so one
        # detection cannot be consumed by both association paths.
        weak_score_ceiling = min(self.settings.score_threshold, self.settings.track_threshold)
        mask = (
            (candidates[:, 4] >= self.settings.weak_reactivation_score_threshold)
            & (candidates[:, 4] < weak_score_ceiling)
            & (areas >= self.settings.weak_reactivation_min_box_area)
        )
        return candidates[mask].astype(np.float32, copy=False)

    def _update_tracker(
        self,
        tracker: Tracker,
        detections: np.ndarray,
        weak_detections: np.ndarray,
        image_size: tuple[int, int],
        state: SessionState,
    ) -> TrackerUpdate:
        update_with_weak_candidates = getattr(tracker, "update_with_weak_candidates", None)
        if state.weak_reactivation_enabled and callable(update_with_weak_candidates):
            return update_with_weak_candidates(
                detections,
                weak_detections,
                image_size,
                max_gap_frames=state.weak_reactivation_max_gap_frames,
                max_center_distance=self.settings.weak_reactivation_max_center_distance,
            )
        return tracker.update(detections, image_size)

    @staticmethod
    def _clamped_box(
        *,
        track_id: int | None,
        lane_id: str,
        coordinates: np.ndarray,
        width: int,
        height: int,
        confidence: float,
    ) -> BoundingBox:
        x1, y1, x2, y2 = np.asarray(coordinates, dtype=float)
        return BoundingBox(
            # ``-1`` is the established fallback sentinel for a detection that
            # has not been assigned a ByteTrack tracklet yet.
            id=track_id if track_id is not None else -1,
            track_id=track_id,
            lane_id=None if lane_id == "global" else lane_id,
            x1=max(0.0, min(float(x1), width - 1)),
            y1=max(0.0, min(float(y1), height - 1)),
            x2=max(0.0, min(float(x2), width - 1)),
            y2=max(0.0, min(float(y2), height - 1)),
            conf=float(confidence),
        )

    @staticmethod
    def _box_iou(left: BoundingBox, right: BoundingBox) -> float:
        intersection_width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
        intersection_height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
        intersection = intersection_width * intersection_height
        left_area = max(0.0, left.x2 - left.x1) * max(0.0, left.y2 - left.y1)
        right_area = max(0.0, right.x2 - right.x1) * max(0.0, right.y2 - right.y1)
        union = left_area + right_area - intersection
        return intersection / union if union > 0 else 0.0

    def _identity_candidates(
        self,
        tracker_updates: dict[str, TrackerUpdate],
        routed_detections: dict[str, np.ndarray],
        image_size: tuple[int, int],
    ) -> list[IdentityCandidate]:
        """Combine post-ROI detections with active ByteTrack observations.

        A detector observation is retained even if it has no active ByteTrack
        track.  This is necessary for two swimmers in one lane: a distant swimmer
        can remain visible below ByteTrack's track-init threshold.
        """

        width, height = image_size
        result: list[IdentityCandidate] = []
        for lane_id, detections in routed_detections.items():
            active_boxes = [
                self._clamped_box(
                    track_id=int(track.track_id),
                    lane_id=lane_id,
                    coordinates=np.asarray(track.tlbr, dtype=float),
                    width=width,
                    height=height,
                    confidence=float(track.score),
                )
                for track in tracker_updates[lane_id].active_tracks
            ]
            detector_boxes = [
                self._clamped_box(
                    track_id=None,
                    lane_id=lane_id,
                    coordinates=np.asarray(detection[:4], dtype=float),
                    width=width,
                    height=height,
                    confidence=float(detection[4]),
                )
                for detection in detections
            ]
            # A ByteTrack state may overlap more than one detector box, notably
            # when the far crop contributes a partial supplemental box.  Claim
            # each raw tracklet at most once so it cannot manufacture two
            # canonical observations in the same frame.
            detector_track_matches: dict[int, BoundingBox] = {}
            matched_track_ids: set[int] = set()
            overlaps = sorted(
                (
                    (self._box_iou(detector_box, active_box), detector_index, active_box)
                    for detector_index, detector_box in enumerate(detector_boxes)
                    for active_box in active_boxes
                ),
                key=lambda item: (-item[0], item[1], item[2].track_id or -1),
            )
            for overlap, detector_index, active_box in overlaps:
                track_id = active_box.track_id
                if overlap < 0.30 or detector_index in detector_track_matches or track_id in matched_track_ids:
                    continue
                detector_track_matches[detector_index] = active_box
                matched_track_ids.add(track_id)
            for detector_index, detector_box in enumerate(detector_boxes):
                active_box = detector_track_matches.get(detector_index)
                track_id = active_box.track_id if active_box is not None else None
                if track_id is not None:
                    detector_box = detector_box.model_copy(update={"id": track_id, "track_id": track_id})
                result.append(
                    IdentityCandidate(
                        lane_id=lane_id,
                        box=detector_box,
                        track_id=track_id,
                        source="detection",
                    )
                )
            for active_box in active_boxes:
                if active_box.track_id in matched_track_ids:
                    continue
                result.append(
                    IdentityCandidate(
                        lane_id=lane_id,
                        box=active_box,
                        track_id=active_box.track_id,
                        source="track",
                    )
                )
        return result

    @staticmethod
    def _resolved_identity_boxes(resolution) -> list[BoundingBox]:
        boxes = []
        used_legacy_ids: set[int] = set()
        for assignment in resolution.assignments:
            box = assignment.candidate.box
            # A detector-only observation used to share the ``-1`` sentinel
            # with every other untracked box.  That makes two real swimmers
            # indistinguishable to legacy consumers which key their overlays by
            # ``id``.  Keep the raw ByteTrack ID whenever it exists and use a
            # stable, negative session-local fallback otherwise.  ``track_id``
            # remains the unambiguous indication of whether ByteTrack produced
            # a tracklet for this observation.
            legacy_id = (
                box.track_id
                if box.track_id is not None and box.track_id not in used_legacy_ids
                else -assignment.identity_id
            )
            used_legacy_ids.add(legacy_id)
            boxes.append(
                box.model_copy(
                    update={
                        "id": legacy_id,
                        "identity_id": assignment.identity_id,
                        "swimmer_id": assignment.swimmer_id,
                    }
                )
            )
        return sorted(boxes, key=lambda box: (box.identity_id or 0, box.id))

    @staticmethod
    def _lap_analysis_boxes(
        tracked_boxes: list[BoundingBox],
        routed_detections: dict[str, np.ndarray],
        image_size: tuple[int, int],
    ) -> list[BoundingBox]:
        result = tracked_boxes.copy()
        tracked_lane_ids = {box.lane_id for box in tracked_boxes}
        width, height = image_size
        for lane_id, detections in routed_detections.items():
            if lane_id in tracked_lane_ids:
                continue
            for detection in detections:
                x1, y1, x2, y2, confidence = np.asarray(detection, dtype=float)
                result.append(
                    BoundingBox(
                        id=-1,
                        lane_id=None if lane_id == "global" else lane_id,
                        x1=max(0.0, min(float(x1), width - 1)),
                        y1=max(0.0, min(float(y1), height - 1)),
                        x2=max(0.0, min(float(x2), width - 1)),
                        y2=max(0.0, min(float(y2), height - 1)),
                        conf=float(confidence),
                    )
                )
        return result

    def delete_session(self, session_id: str) -> None:
        with self._sessions_lock:
            state = self._sessions.get(session_id)
        if state is None:
            raise SessionNotFoundError(f"Tracking session {session_id!r} does not exist")
        with state.lock:
            with self._sessions_lock:
                if self._sessions.pop(session_id, None) is None:
                    raise SessionNotFoundError(f"Tracking session {session_id!r} does not exist")

    def next_sequence(self, session_id: str) -> int:
        """Return the next accepted sequence without exposing mutable session state."""

        with self._sessions_lock:
            state = self._sessions.get(session_id)
        if state is None:
            raise SessionNotFoundError(f"Tracking session {session_id!r} does not exist")
        with state.lock:
            with self._sessions_lock:
                if self._sessions.get(session_id) is not state:
                    raise SessionNotFoundError(f"Tracking session {session_id!r} does not exist")
            if state.expires_at <= self.clock():
                with self._sessions_lock:
                    self._sessions.pop(session_id, None)
                raise SessionNotFoundError(f"Tracking session {session_id!r} expired")
            return state.next_sequence

    def expire_sessions(self) -> int:
        now = self.clock()
        expired = []
        with self._sessions_lock:
            for session_id, state in self._sessions.items():
                if state.expires_at <= now and state.lock.acquire(blocking=False):
                    expired.append((session_id, state))
            for session_id, state in expired:
                if self._sessions.get(session_id) is state:
                    del self._sessions[session_id]
        for _, state in expired:
            state.lock.release()
        return len(expired)

    @staticmethod
    def fingerprint(metadata_json: str, encoded_frames: list[bytes]) -> str:
        digest = hashlib.sha256(metadata_json.encode("utf-8"))
        for payload in encoded_frames:
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        return digest.hexdigest()

    def process_batch(
        self,
        session_id: str,
        metadata: BatchMetadata,
        frames: list[np.ndarray],
        fingerprint: str,
    ) -> BatchResult:
        with self._sessions_lock:
            state = self._sessions.get(session_id)
        if state is None:
            raise SessionNotFoundError(f"Tracking session {session_id!r} does not exist")

        with state.lock:
            with self._sessions_lock:
                if self._sessions.get(session_id) is not state:
                    raise SessionNotFoundError(f"Tracking session {session_id!r} does not exist")
            if state.expires_at <= self.clock():
                with self._sessions_lock:
                    self._sessions.pop(session_id, None)
                raise SessionNotFoundError(f"Tracking session {session_id!r} expired")
            cached = state.cache.get(metadata.batch_id)
            if cached is not None:
                if cached.fingerprint != fingerprint:
                    raise ConflictError("batch_id was already used with a different payload")
                state.cache.move_to_end(metadata.batch_id)
                state.expires_at = self.clock() + self.settings.session_ttl_seconds
                return cached.result
            if state.poisoned_reason:
                raise ConflictError(f"Session cannot continue after a tracking failure: {state.poisoned_reason}")
            if metadata.sequence != state.next_sequence:
                raise ConflictError(f"Expected sequence {state.next_sequence}, received {metadata.sequence}")
            first = metadata.frames[0]
            if state.last_frame_index is not None and first.frame_index <= state.last_frame_index:
                raise ConflictError("frame_index must increase across batches")
            if state.last_time_ms is not None and first.time_ms < state.last_time_ms:
                raise ConflictError("time_ms must not move backwards across batches")

            # Infer every frame before mutating ByteTrack. Detector failures are safe to retry.
            detector_results = self._infer_frames(
                frames,
                metadata,
                use_far_crop=state.lap_analyzer is not None,
            )
            frame_results: list[FrameResult] = []
            try:
                for detector_result, item in zip(detector_results, metadata.frames):
                    image_size = (item.original_width, item.original_height)
                    routed_detections = self._limit_routed_detections(
                        state.lane_router.route(detector_result.accepted, image_size)
                    )
                    weak_candidates = (
                        self._weak_candidates(detector_result)
                        if state.weak_reactivation_enabled
                        else np.empty((0, 5), dtype=np.float32)
                    )
                    routed_weak_candidates = state.lane_router.route(weak_candidates, image_size)
                    tracker_updates = {
                        lane_id: self._update_tracker(
                            state.trackers[lane_id],
                            detections,
                            routed_weak_candidates[lane_id],
                            image_size,
                            state,
                        )
                        for lane_id, detections in routed_detections.items()
                    }
                    identity_resolution = state.identity_resolver.resolve(
                        time_ms=item.time_ms,
                        width=item.original_width,
                        height=item.original_height,
                        candidates=self._identity_candidates(
                            tracker_updates,
                            routed_detections,
                            image_size,
                        ),
                    )
                    boxes = self._resolved_identity_boxes(identity_resolution)
                    frame_results.append(
                        FrameResult(
                            frame_index=item.frame_index,
                            time_ms=item.time_ms,
                            width=item.original_width,
                            height=item.original_height,
                            boxes=boxes,
                            identity_summary=IdentitySummary(
                                confirmed_count=identity_resolution.confirmed_count,
                                active_count=identity_resolution.active_count,
                            ),
                            lap_scores=(
                                state.lap_analyzer.observe(
                                    time_ms=item.time_ms,
                                    width=item.original_width,
                                    height=item.original_height,
                                    boxes=boxes,
                                )
                                if state.lap_analyzer is not None
                                else None
                            ),
                            tracking_diagnostics=self._tracking_diagnostics(
                                detector_result,
                                weak_candidates,
                                routed_detections,
                                routed_weak_candidates,
                                tracker_updates,
                                state.diagnostics_level,
                            ),
                        )
                    )
            except Exception as exc:
                state.poisoned_reason = str(exc)
                raise

            result = BatchResult(
                session_id=session_id,
                batch_id=metadata.batch_id,
                sequence=metadata.sequence,
                next_sequence=metadata.sequence + 1,
                frames=frame_results,
            )
            state.next_sequence += 1
            state.last_frame_index = metadata.frames[-1].frame_index
            state.last_time_ms = metadata.frames[-1].time_ms
            state.expires_at = self.clock() + self.settings.session_ttl_seconds
            state.cache[metadata.batch_id] = CachedBatch(fingerprint=fingerprint, result=result)
            while len(state.cache) > self.settings.idempotency_cache_size:
                state.cache.popitem(last=False)
            return result

    def close(self) -> None:
        with self._sessions_lock:
            self._sessions.clear()
        self.detector.close()
