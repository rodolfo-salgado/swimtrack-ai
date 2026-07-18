from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from swimtrack_ai.config import Settings
from swimtrack_ai.detectors.tensorrt import (
    TensorRTDetector,
    _network_supports_dynamic_batch,
    _profile_batch_size,
    postprocess_detections,
)


def test_tracking_threshold_configuration_is_validated(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="floor <= threshold"):
        Settings(
            model_source_dir=tmp_path,
            model_cache_dir=tmp_path,
            diagnostic_score_floor=0.4,
            score_threshold=0.2,
        )

    with pytest.raises(ValueError, match="match_threshold"):
        Settings(
            model_source_dir=tmp_path,
            model_cache_dir=tmp_path,
            match_threshold=1.1,
        )


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


class _Tensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class _Network:
    def __init__(self, input_shapes: list[tuple[int, ...]], output_shapes: list[tuple[int, ...]]) -> None:
        self._inputs = [_Tensor(shape) for shape in input_shapes]
        self._outputs = [_Tensor(shape) for shape in output_shapes]
        self.num_inputs = len(self._inputs)
        self.num_outputs = len(self._outputs)

    def get_input(self, index: int) -> _Tensor:
        return self._inputs[index]

    def get_output(self, index: int) -> _Tensor:
        return self._outputs[index]


def test_dynamic_batch_requires_dynamic_onnx_outputs(tmp_path: Path) -> None:
    settings = Settings(
        model_source_dir=tmp_path,
        model_cache_dir=tmp_path,
        trt_opt_batch_size=4,
        trt_max_batch_size=8,
    )
    dynamic_network = _Network(
        [(-1, 3, 640, 640), (-1, 2)],
        [(-1, 300), (-1, 300, 4), (-1, 300)],
    )
    fixed_output_network = _Network(
        [(-1, 3, 640, 640), (-1, 2)],
        [(1, 300), (1, 300, 4), (1, 300)],
    )

    assert _network_supports_dynamic_batch(dynamic_network)
    assert _profile_batch_size(settings, dynamic_batch=True) == (1, 4, 8)
    assert not _network_supports_dynamic_batch(fixed_output_network)
    assert _profile_batch_size(settings, dynamic_batch=False) == (1, 1, 1)


class _Engine:
    def __init__(self, tensor_shapes: dict[str, tuple[int, ...]], maximum_batch_size: int = 8) -> None:
        self.tensor_shapes = tensor_shapes
        self.maximum_batch_size = maximum_batch_size

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        return self.tensor_shapes[name]

    def get_tensor_profile_shape(self, _name: str, _profile_index: int):
        return ((1, 3, 640, 640), (4, 3, 640, 640), (self.maximum_batch_size, 3, 640, 640))


def test_serialized_engine_with_fixed_output_is_limited_to_b1(tmp_path: Path) -> None:
    detector = object.__new__(TensorRTDetector)
    detector.settings = Settings(
        model_source_dir=tmp_path,
        model_cache_dir=tmp_path,
        trt_max_batch_size=8,
    )
    detector.inputs = ["images", "orig_target_sizes"]
    detector.outputs = ["labels", "boxes", "scores"]
    detector.engine = _Engine(
        {
            "images": (-1, 3, 640, 640),
            "orig_target_sizes": (-1, 2),
            "labels": (1, 300),
            "boxes": (1, 300, 4),
            "scores": (1, 300),
        }
    )

    assert detector._engine_batch_capacity() == 1


def test_serialized_dynamic_engine_uses_configured_batch_ceiling(tmp_path: Path) -> None:
    detector = object.__new__(TensorRTDetector)
    detector.settings = Settings(
        model_source_dir=tmp_path,
        model_cache_dir=tmp_path,
        trt_max_batch_size=8,
    )
    detector.inputs = ["images", "orig_target_sizes"]
    detector.outputs = ["labels", "boxes", "scores"]
    detector.engine = _Engine(
        {
            "images": (-1, 3, 640, 640),
            "orig_target_sizes": (-1, 2),
            "labels": (-1, 300),
            "boxes": (-1, 300, 4),
            "scores": (-1, 300),
        },
        maximum_batch_size=16,
    )

    assert detector._engine_batch_capacity() == 8
