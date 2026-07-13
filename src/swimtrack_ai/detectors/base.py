from __future__ import annotations

from typing import Protocol

import numpy as np


class Detector(Protocol):
    def infer(self, frame: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
        """Return detections as float32 [x1, y1, x2, y2, confidence]."""

    def close(self) -> None: ...
