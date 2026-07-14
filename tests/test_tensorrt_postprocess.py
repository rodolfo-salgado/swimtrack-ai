from __future__ import annotations

from pathlib import Path

import numpy as np

from swimtrack_ai.config import Settings
from swimtrack_ai.detectors.tensorrt import postprocess_detections


def test_postprocess_exposes_low_score_person_candidates_before_runtime_filters(tmp_path: Path) -> None:
    settings = Settings(
        backend="fake",
        auth_token="test-secret",
        model_source_dir=tmp_path,
        model_cache_dir=tmp_path,
        diagnostic_score_floor=0.05,
        score_threshold=0.35,
        min_box_area=500.0,
    )
    labels = np.asarray([0, 0, 0, 1], dtype=np.int64)
    boxes = np.asarray(
        [
            [5, 5, 15, 15],
            [10, 10, 40, 30],
            [-5, -5, 35, 25],
            [0, 0, 50, 50],
        ],
        dtype=np.float32,
    )
    scores = np.asarray([0.08, 0.20, 0.80, 0.99], dtype=np.float32)

    result = postprocess_detections(labels, boxes, scores, settings, (32, 24))

    assert result.person_candidates.shape == (3, 5)
    assert result.person_candidates[:, 4].tolist() == [0.800000011920929, 0.20000000298023224, 0.07999999821186066]
    assert result.accepted.shape == (1, 5)
    assert result.accepted[0].tolist() == [0.0, 0.0, 31.0, 23.0, 0.800000011920929]
