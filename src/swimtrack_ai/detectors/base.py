from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class DetectorResult:
    """Detector output before and after the runtime acceptance filters."""

    person_candidates: np.ndarray
    accepted: np.ndarray


class Detector(Protocol):
    def infer(self, frame: np.ndarray, target_size: tuple[int, int]) -> DetectorResult:
        """Return person candidates and accepted float32 Nx5 detections."""

    def infer_batch(
        self,
        frames: Sequence[np.ndarray],
        target_sizes: Sequence[tuple[int, int]],
    ) -> list[DetectorResult]:
        """Return results in the same order as frames, internally chunking when necessary."""

    def close(self) -> None: ...
