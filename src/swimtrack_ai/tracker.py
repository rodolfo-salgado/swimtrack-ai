from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol

import numpy as np
from scipy.optimize import linear_sum_assignment

from swimtrack_ai.config import Settings


class Tracker(Protocol):
    def update(self, detections: np.ndarray, image_size: tuple[int, int]) -> list: ...


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

    def update(self, detections: np.ndarray, image_size: tuple[int, int]) -> list:
        width, height = image_size
        return self._tracker.update(detections, [height, width], [height, width])


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
