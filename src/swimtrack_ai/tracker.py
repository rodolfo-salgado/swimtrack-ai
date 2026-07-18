from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol

import numpy as np
from scipy.optimize import linear_sum_assignment

from swimtrack_ai.config import Settings


@dataclass(frozen=True, slots=True)
class TrackerUpdate:
    active_tracks: list
    retained_lost_track_count: int = 0
    weak_reactivated_track_ids: list[int] = field(default_factory=list)


class Tracker(Protocol):
    def update(self, detections: np.ndarray, image_size: tuple[int, int]) -> TrackerUpdate: ...


def _install_bytetrack_compatibility(root: Path) -> None:
    # The pinned ByteTrack upstream still references aliases removed in NumPy 1.24.
    for name, value in (("float", float), ("int", int), ("bool", bool)):
        if name not in np.__dict__:
            setattr(np, name, value)

    try:
        import cython_bbox  # noqa: F401
    except ImportError:
        cython_bbox = types.ModuleType("cython_bbox")

        def bbox_overlaps(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
            boxes_a = np.asarray(boxes_a, dtype=np.float32)
            boxes_b = np.asarray(boxes_b, dtype=np.float32)
            if boxes_a.size == 0 or boxes_b.size == 0:
                return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
            top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
            bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
            dimensions = np.clip(bottom_right - top_left, 0, None)
            intersection = dimensions[..., 0] * dimensions[..., 1]
            area_a = np.clip(boxes_a[:, 2] - boxes_a[:, 0], 0, None) * np.clip(boxes_a[:, 3] - boxes_a[:, 1], 0, None)
            area_b = np.clip(boxes_b[:, 2] - boxes_b[:, 0], 0, None) * np.clip(boxes_b[:, 3] - boxes_b[:, 1], 0, None)
            return intersection / np.clip(area_a[:, None] + area_b[None, :] - intersection, 1e-6, None)

        cython_bbox.bbox_overlaps = bbox_overlaps  # type: ignore[attr-defined]
        sys.modules["cython_bbox"] = cython_bbox

    try:
        import lap  # noqa: F401
    except ImportError:
        lap = types.ModuleType("lap")

        def lapjv(cost_matrix, extend_cost=True, cost_limit=np.inf):
            del extend_cost
            rows, columns = linear_sum_assignment(cost_matrix)
            row_assignment = np.full(cost_matrix.shape[0], -1, dtype=int)
            column_assignment = np.full(cost_matrix.shape[1], -1, dtype=int)
            total = 0.0
            for row, column in zip(rows, columns):
                cost = float(cost_matrix[row, column])
                if cost <= cost_limit:
                    row_assignment[row] = column
                    column_assignment[column] = row
                    total += cost
            return total, row_assignment, column_assignment

        lap.lapjv = lapjv  # type: ignore[attr-defined]
        sys.modules["lap"] = lap

    # Tracking the Nx5 NumPy outputs does not use torch, despite the upstream import.
    try:
        import torch  # noqa: F401
    except ImportError:
        torch = types.ModuleType("torch")
        nn = types.ModuleType("torch.nn")
        functional = types.ModuleType("torch.nn.functional")
        torch.nn = nn  # type: ignore[attr-defined]
        nn.functional = functional  # type: ignore[attr-defined]
        sys.modules.update({"torch": torch, "torch.nn": nn, "torch.nn.functional": functional})

    yolox_root = root / "yolox"
    if not (yolox_root / "tracker" / "byte_tracker.py").is_file():
        raise FileNotFoundError(f"ByteTrack submodule is missing or incomplete: {root}")
    # Avoid running yolox/__init__.py, which imports training-only torch utilities.
    if "yolox" not in sys.modules:
        yolox = types.ModuleType("yolox")
        yolox.__path__ = [str(yolox_root)]  # type: ignore[attr-defined]
        sys.modules["yolox"] = yolox
    if "yolox.tracker" not in sys.modules:
        tracker_package = types.ModuleType("yolox.tracker")
        tracker_package.__path__ = [str(yolox_root / "tracker")]  # type: ignore[attr-defined]
        sys.modules["yolox.tracker"] = tracker_package


class ByteTrackAdapter:
    def __init__(self, tracker) -> None:
        self._tracker = tracker

    def update(self, detections: np.ndarray, image_size: tuple[int, int]) -> TrackerUpdate:
        width, height = image_size
        active_tracks = self._tracker.update(detections, [height, width], [height, width])
        return TrackerUpdate(
            active_tracks=active_tracks,
            retained_lost_track_count=len(self._tracker.lost_stracks),
        )

    @staticmethod
    def _as_detections(detections: np.ndarray) -> np.ndarray:
        array = np.asarray(detections, dtype=np.float32)
        if array.size == 0:
            return np.empty((0, 5), dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 5:
            raise ValueError("weak detections must be an Nx5 array")
        return array

    @staticmethod
    def _center_distance(
        track,
        detection: np.ndarray,
        image_size: tuple[int, int],
    ) -> float:
        width, height = image_size
        if width <= 0 or height <= 0:
            raise ValueError("image_size must be positive")
        track_box = np.asarray(track.tlbr, dtype=np.float32)
        track_center = (track_box[:2] + track_box[2:]) / 2.0
        detection_center = (detection[:2] + detection[2:4]) / 2.0
        offset = (track_center - detection_center) / np.asarray((width, height), dtype=np.float32)
        return float(np.linalg.norm(offset))

    def update_with_weak_candidates(
        self,
        detections: np.ndarray,
        weak_detections: np.ndarray,
        image_size: tuple[int, int],
        *,
        max_gap_frames: int,
        max_center_distance: float,
    ) -> TrackerUpdate:
        """Re-activate recently lost tracks with lane-gated weak detector evidence.

        The caller is responsible for routing candidates to one calibrated lane and
        filtering their score and area. Weak detections never enter ByteTrack's
        ordinary update path, so they cannot initialize a new track. They can only
        re-activate a retained lost track after an explicit temporal and spatial
        gate succeeds.
        """

        if max_gap_frames < 1:
            raise ValueError("max_gap_frames must be at least one")
        if not 0.0 < max_center_distance <= 1.0:
            raise ValueError("max_center_distance must be in (0, 1]")

        width, height = image_size
        active_tracks = list(self._tracker.update(detections, [height, width], [height, width]))
        candidates = self._as_detections(weak_detections)
        if not len(candidates) or not self._tracker.lost_stracks:
            return TrackerUpdate(
                active_tracks=active_tracks,
                retained_lost_track_count=len(self._tracker.lost_stracks),
            )

        next_frame_id = int(self._tracker.frame_id)
        eligible_tracks = [
            track
            for track in self._tracker.lost_stracks
            if 0 < next_frame_id - int(track.end_frame) <= max_gap_frames
        ]
        if not eligible_tracks:
            return TrackerUpdate(
                active_tracks=active_tracks,
                retained_lost_track_count=len(self._tracker.lost_stracks),
            )

        unmatched_tracks = list(eligible_tracks)
        matches: list[tuple[object, np.ndarray]] = []
        for candidate in candidates[np.argsort(candidates[:, 4], kind="stable")[::-1]]:
            nearby = [
                (self._center_distance(track, candidate, image_size), index, track)
                for index, track in enumerate(unmatched_tracks)
            ]
            nearby = [item for item in nearby if item[0] <= max_center_distance]
            if not nearby:
                continue
            _distance, index, track = min(nearby, key=lambda item: item[0])
            matches.append((track, candidate))
            del unmatched_tracks[index]

        if not matches:
            return TrackerUpdate(
                active_tracks=active_tracks,
                retained_lost_track_count=len(self._tracker.lost_stracks),
            )

        reactivated_ids: list[int] = []
        for track, candidate in matches:
            detection_type = type(track)
            detection = detection_type(
                detection_type.tlbr_to_tlwh(np.asarray(candidate[:4], dtype=np.float32)),
                float(candidate[4]),
            )
            track.re_activate(detection, next_frame_id, new_id=False)
            reactivated_ids.append(int(track.track_id))

        reactivated_set = set(reactivated_ids)
        existing_ids = {int(track.track_id) for track in self._tracker.tracked_stracks}
        for track, _candidate in matches:
            if int(track.track_id) not in existing_ids:
                self._tracker.tracked_stracks.append(track)
                existing_ids.add(int(track.track_id))
            if all(int(active.track_id) != int(track.track_id) for active in active_tracks):
                active_tracks.append(track)
        self._tracker.lost_stracks = [
            track for track in self._tracker.lost_stracks if int(track.track_id) not in reactivated_set
        ]

        return TrackerUpdate(
            active_tracks=active_tracks,
            retained_lost_track_count=len(self._tracker.lost_stracks),
            weak_reactivated_track_ids=reactivated_ids,
        )


class ByteTrackFactory:
    def __init__(self, settings: Settings) -> None:
        _install_bytetrack_compatibility(settings.bytetrack_root)
        module = importlib.import_module("yolox.tracker.byte_tracker")
        self.tracker_type = module.BYTETracker
        self.args = SimpleNamespace(
            track_thresh=settings.track_threshold,
            track_buffer=settings.track_buffer,
            match_thresh=settings.match_threshold,
            mot20=settings.mot20,
        )

    def __call__(self, fps: float) -> ByteTrackAdapter:
        return ByteTrackAdapter(self.tracker_type(self.args, frame_rate=max(1, round(fps))))
