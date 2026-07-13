from __future__ import annotations

import cv2
import numpy as np


class FakeDetector:
    """Deterministic CPU detector for API tests and local integration."""

    def infer(self, frame: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 32, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        source_height, source_width = frame.shape[:2]
        target_width, target_height = target_size
        scale_x = target_width / source_width
        scale_y = target_height / source_height
        detections: list[list[float]] = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            if width * height < 4:
                continue
            detections.append(
                [
                    x * scale_x,
                    y * scale_y,
                    (x + width) * scale_x,
                    (y + height) * scale_y,
                    0.99,
                ]
            )
        detections.sort(key=lambda item: item[0])
        return np.asarray(detections, dtype=np.float32).reshape(-1, 5)

    def close(self) -> None:
        return None
