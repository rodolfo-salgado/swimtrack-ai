from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

FIXED_CAMERA_CALIBRATION_ID = "fixed-camera-v1"


@dataclass(frozen=True, slots=True)
class LaneCalibration:
    """Perspective calibration expressed relative to image width and height."""

    lane_id: str
    source_quad: tuple[tuple[float, float], ...]
    visible_polygon: tuple[tuple[float, float], ...]


FIXED_CAMERA_CENTER_LANE = LaneCalibration(
    lane_id="center",
    source_quad=(
        (0.4463, 0.1583),
        (0.5815, 0.1583),
        (1.2603, 0.9769),
        (-0.2507, 0.9769),
    ),
    visible_polygon=(
        (0.4463, 0.1583),
        (0.5815, 0.1583),
        (1.0000, 0.6630),
        (1.0000, 0.9769),
        (0.0000, 0.9769),
        (0.0000, 0.6824),
    ),
)


def lanes_for_calibration(calibration_id: str) -> tuple[LaneCalibration, ...]:
    if calibration_id != FIXED_CAMERA_CALIBRATION_ID:
        raise ValueError(f"Unsupported lap calibration: {calibration_id}")
    return (FIXED_CAMERA_CENTER_LANE,)


def perspective_matrix(calibration: LaneCalibration) -> np.ndarray:
    source = np.asarray(calibration.source_quad, dtype=np.float32)
    target = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float32)
    return cv2.getPerspectiveTransform(source, target)


class LaneRouter:
    """Assign accepted detections to at most one calibrated lane."""

    def __init__(self, calibration_id: str | None, *, enabled: bool, margin: float = 0.05) -> None:
        self.enabled = enabled and calibration_id is not None
        self.margin = margin
        self._lanes = lanes_for_calibration(calibration_id) if self.enabled and calibration_id is not None else ()
        self._matrices = {lane.lane_id: perspective_matrix(lane) for lane in self._lanes}

    @property
    def lane_ids(self) -> tuple[str, ...]:
        return tuple(lane.lane_id for lane in self._lanes) if self.enabled else ("global",)

    def route(self, detections: np.ndarray, image_size: tuple[int, int]) -> dict[str, np.ndarray]:
        if not self.enabled:
            return {"global": detections}
        routed: dict[str, list[np.ndarray]] = {lane_id: [] for lane_id in self.lane_ids}
        width, height = image_size
        for detection in detections:
            center = np.asarray(
                [[[(detection[0] + detection[2]) / (2.0 * width), (detection[1] + detection[3]) / (2.0 * height)]]],
                dtype=np.float32,
            )
            candidates: list[tuple[float, str]] = []
            for lane_id, matrix in self._matrices.items():
                lane_x, position = cv2.perspectiveTransform(center, matrix)[0, 0]
                if (
                    -self.margin <= lane_x <= 1.0 + self.margin
                    and -self.margin <= position <= 1.0 + self.margin
                ):
                    candidates.append((abs(float(lane_x) - 0.5), lane_id))
            if candidates:
                _distance, lane_id = min(candidates)
                routed[lane_id].append(detection)
        return {
            lane_id: np.asarray(items, dtype=np.float32).reshape(-1, 5)
            for lane_id, items in routed.items()
        }
