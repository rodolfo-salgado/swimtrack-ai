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
    diagnostics_level: DiagnosticsLevel
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
        with self._sessions_lock:
            if len(self._sessions) >= self.settings.max_sessions:
                raise SessionCapacityError("Maximum number of active tracking sessions reached")
            session_id = str(uuid.uuid4())
            lane_router = LaneRouter(
                lap_calibration_id,
                enabled=self.settings.lane_roi_enabled,
            )
            self._sessions[session_id] = SessionState(
                session_id=session_id,
                trackers={lane_id: self.tracker_factory(fps) for lane_id in lane_router.lane_ids},
                lane_router=lane_router,
                lap_analyzer=LapAnalyzer(fps, lap_calibration_id) if lap_calibration_id is not None else None,
                diagnostics_level=diagnostics,
                expires_at=self.clock() + self.settings.session_ttl_seconds,
            )
        tracker_frame_rate = max(1, round(fps))
        effective_buffer_frames = int(tracker_frame_rate / 30.0 * self.settings.track_buffer)
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
                    effective_lost_buffer_frames=effective_buffer_frames,
                    effective_lost_buffer_seconds=effective_buffer_frames / tracker_frame_rate,
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

    def _tracking_diagnostics(
        self,
        detector_result: DetectorResult,
        routed_detections: dict[str, np.ndarray],
        tracker_updates: dict[str, TrackerUpdate],
        level: DiagnosticsLevel,
    ) -> FrameTrackingDiagnostics | None:
        if level == "none":
            return None
        return FrameTrackingDiagnostics(
            diagnostic_floor=self.settings.diagnostic_score_floor,
            person_candidates=self._diagnostic_stage(detector_result.person_candidates, level),
            detector_accepted=self._diagnostic_stage(detector_result.accepted, level),
            lanes=[
                LaneTrackingDiagnostics(
                    lane_id=lane_id,
                    after_roi=self._diagnostic_stage(routed_detections[lane_id], level),
                    active_track_ids=[int(track.track_id) for track in tracker_updates[lane_id].active_tracks],
                    retained_lost_track_count=tracker_updates[lane_id].retained_lost_track_count,
                )
                for lane_id in routed_detections
            ],
        )

    def delete_session(self, session_id: str) -> None:
        with self._sessions_lock:
            state = self._sessions.get(session_id)
        if state is None:
            raise SessionNotFoundError(f"Tracking session {session_id!r} does not exist")
        with state.lock:
            with self._sessions_lock:
                if self._sessions.pop(session_id, None) is None:
                    raise SessionNotFoundError(f"Tracking session {session_id!r} does not exist")

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
            detector_results = [
                self.detector.infer(frame, (item.original_width, item.original_height))
                for frame, item in zip(frames, metadata.frames)
            ]
            frame_results: list[FrameResult] = []
            try:
                for detector_result, item in zip(detector_results, metadata.frames):
                    image_size = (item.original_width, item.original_height)
                    routed_detections = state.lane_router.route(detector_result.accepted, image_size)
                    tracker_updates = {
                        lane_id: state.trackers[lane_id].update(detections, image_size)
                        for lane_id, detections in routed_detections.items()
                    }
                    boxes = []
                    for lane_id, tracker_update in tracker_updates.items():
                        for track in tracker_update.active_tracks:
                            x1, y1, x2, y2 = np.asarray(track.tlbr, dtype=float)
                            boxes.append(
                                BoundingBox(
                                    id=int(track.track_id),
                                    lane_id=None if lane_id == "global" else lane_id,
                                    x1=max(0.0, min(float(x1), item.original_width - 1)),
                                    y1=max(0.0, min(float(y1), item.original_height - 1)),
                                    x2=max(0.0, min(float(x2), item.original_width - 1)),
                                    y2=max(0.0, min(float(y2), item.original_height - 1)),
                                    conf=float(track.score),
                                )
                            )
                    frame_results.append(
                        FrameResult(
                            frame_index=item.frame_index,
                            time_ms=item.time_ms,
                            width=item.original_width,
                            height=item.original_height,
                            boxes=boxes,
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
                                routed_detections,
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
