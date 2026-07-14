from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from swimtrack_ai.config import Settings
from swimtrack_ai.detectors.base import DetectorResult


class EngineLoadError(RuntimeError):
    pass


def _as_detections(boxes: np.ndarray, scores: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    if not len(boxes):
        return np.empty((0, 5), dtype=np.float32)
    target_width, target_height = target_size
    selected_boxes = boxes.astype(np.float32, copy=True)
    selected_boxes[:, [0, 2]] = np.clip(selected_boxes[:, [0, 2]], 0, target_width - 1)
    selected_boxes[:, [1, 3]] = np.clip(selected_boxes[:, [1, 3]], 0, target_height - 1)
    detections = np.column_stack((selected_boxes, scores)).astype(np.float32)
    return detections[np.argsort(detections[:, 4])[::-1]]


def postprocess_detections(
    labels: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    settings: Settings,
    target_size: tuple[int, int],
) -> DetectorResult:
    """Expose low-score person candidates while preserving runtime filters."""

    person_mask = (labels == settings.person_label) & (scores >= settings.diagnostic_score_floor)
    person_candidates = _as_detections(boxes[person_mask], scores[person_mask], target_size)
    areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    accepted_mask = (
        (labels == settings.person_label)
        & (scores >= settings.score_threshold)
        & (areas >= settings.min_box_area)
    )
    accepted = _as_detections(boxes[accepted_mask], scores[accepted_mask], target_size)[: settings.max_detections]
    return DetectorResult(person_candidates=person_candidates, accepted=accepted)


def _model_files(settings: Settings) -> list[Path]:
    files = [settings.onnx_path]
    external_data = settings.onnx_path.with_suffix(settings.onnx_path.suffix + ".data")
    if external_data.is_file():
        files.append(external_data)
    return files


def engine_signature(settings: Settings) -> dict[str, Any]:
    import tensorrt as trt

    digest = hashlib.sha256()
    model_files = _model_files(settings)
    for path in model_files:
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return {
        "schema": 1,
        "model_sha256": digest.hexdigest(),
        "model_files": [path.name for path in model_files],
        "tensorrt_version": trt.__version__,
        "device": settings.device,
        "input_width": settings.input_width,
        "input_height": settings.input_height,
        "fp16": settings.trt_fp16,
        "workspace_gb": settings.trt_workspace_gb,
    }


def engine_cache_is_current(settings: Settings) -> bool:
    if not settings.engine_path.is_file() or not settings.engine_manifest_path.is_file():
        return False
    try:
        cached = json.loads(settings.engine_manifest_path.read_text(encoding="utf-8"))
        return cached == engine_signature(settings)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def invalidate_engine_cache(settings: Settings) -> None:
    settings.engine_path.unlink(missing_ok=True)
    settings.engine_manifest_path.unlink(missing_ok=True)


def build_engine(settings: Settings) -> None:
    import tensorrt as trt

    if not settings.onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {settings.onnx_path}")
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    parsed = parser.parse_from_file(str(settings.onnx_path))
    if not parsed:
        errors = [parser.get_error(index).desc() for index in range(parser.num_errors)]
        raise RuntimeError(f"TensorRT could not parse {settings.onnx_path}: {errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(settings.trt_workspace_gb * 1024**3))
    if settings.trt_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    image_shape = (1, 3, settings.input_height, settings.input_width)
    profile.set_shape("images", image_shape, image_shape, image_shape)
    target_shape = (1, 2)
    profile.set_shape("orig_target_sizes", target_shape, target_shape, target_shape)
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the engine")
    temporary_path = settings.engine_path.with_suffix(settings.engine_path.suffix + ".tmp")
    temporary_path.write_bytes(serialized)
    temporary_path.replace(settings.engine_path)
    manifest = json.dumps(engine_signature(settings), sort_keys=True, separators=(",", ":"))
    temporary_manifest = settings.engine_manifest_path.with_suffix(settings.engine_manifest_path.suffix + ".tmp")
    temporary_manifest.write_text(manifest, encoding="utf-8")
    temporary_manifest.replace(settings.engine_manifest_path)


class TensorRTDetector:
    """RT-DETRv2 TensorRT runner with persistent device buffers and batch size one."""

    def __init__(self, settings: Settings) -> None:
        import tensorrt as trt
        from cuda.bindings import runtime as cudart

        self.settings = settings
        self.trt = trt
        self.cudart = cudart
        self._lock = threading.Lock()
        self._buffers: dict[str, tuple[int, int]] = {}
        self._closed = False
        self.cuda_device_index = self._parse_device(settings.device)
        self._cuda(cudart.cudaSetDevice(self.cuda_device_index), "select CUDA device")

        logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(logger, "")
        with trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(settings.engine_path.read_bytes())
        if self.engine is None:
            raise EngineLoadError(f"Could not deserialize TensorRT engine {settings.engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise EngineLoadError("Could not create TensorRT execution context")
        self.stream = self._cuda(cudart.cudaStreamCreate(), "create CUDA stream")
        self.names = [self.engine.get_tensor_name(index) for index in range(self.engine.num_io_tensors)]
        self.inputs = [name for name in self.names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT]
        self.outputs = [name for name in self.names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT]

    @staticmethod
    def _parse_device(device: str) -> int:
        if not device.startswith("cuda"):
            raise ValueError(f"TensorRT requires a CUDA device, got {device!r}")
        return int(device.split(":", 1)[1]) if ":" in device else 0

    def _cuda(self, result: tuple, action: str):
        error, *values = result
        if error != self.cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA failed while trying to {action}: {error}")
        if len(values) == 1:
            return values[0]
        return tuple(values) if values else None

    def _device_pointer(self, name: str, size: int) -> int:
        existing = self._buffers.get(name)
        if existing is not None and existing[1] >= size:
            return existing[0]
        if existing is not None:
            self._cuda(self.cudart.cudaFree(existing[0]), f"resize buffer {name}")
        pointer = self._cuda(self.cudart.cudaMalloc(size), f"allocate buffer {name}")
        self._buffers[name] = (pointer, size)
        return pointer

    def _preprocess(self, frame: np.ndarray, target_size: tuple[int, int]) -> dict[str, np.ndarray]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb,
            (self.settings.input_width, self.settings.input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        image = np.transpose(resized, (2, 0, 1)).astype(np.float32, copy=False) / np.float32(255.0)
        target_width, target_height = target_size
        return {
            "images": np.ascontiguousarray(image[None, ...]),
            "orig_target_sizes": np.asarray([[target_width, target_height]], dtype=np.int64),
        }

    def _output_arrays(self) -> dict[str, np.ndarray]:
        arrays = {}
        for name in self.outputs:
            shape = tuple(int(value) for value in self.context.get_tensor_shape(name))
            if any(value < 0 for value in shape):
                raise RuntimeError(f"Unresolved TensorRT output shape for {name}: {shape}")
            arrays[name] = np.empty(shape, dtype=np.dtype(self.trt.nptype(self.engine.get_tensor_dtype(name))))
        return arrays

    def _execute(self, inputs: dict[str, np.ndarray], outputs: dict[str, np.ndarray]) -> None:
        arrays = {**inputs, **outputs}
        pointers = {name: self._device_pointer(name, array.nbytes) for name, array in arrays.items()}
        for name, pointer in pointers.items():
            if not self.context.set_tensor_address(name, pointer):
                raise RuntimeError(f"Could not bind TensorRT tensor {name}")
        for name, array in inputs.items():
            self._cuda(
                self.cudart.cudaMemcpyAsync(
                    pointers[name],
                    array.ctypes.data,
                    array.nbytes,
                    self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    self.stream,
                ),
                f"copy {name} to GPU",
            )
        if not self.context.execute_async_v3(stream_handle=int(self.stream)):
            raise RuntimeError("TensorRT inference failed")
        for name, array in outputs.items():
            self._cuda(
                self.cudart.cudaMemcpyAsync(
                    array.ctypes.data,
                    pointers[name],
                    array.nbytes,
                    self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    self.stream,
                ),
                f"copy {name} to CPU",
            )
        self._cuda(self.cudart.cudaStreamSynchronize(self.stream), "synchronize CUDA stream")

    def infer(self, frame: np.ndarray, target_size: tuple[int, int]) -> DetectorResult:
        with self._lock:
            inputs = self._preprocess(frame, target_size)
            for name, array in inputs.items():
                if not self.context.set_input_shape(name, tuple(array.shape)):
                    raise RuntimeError(f"TensorRT rejected input shape {array.shape} for {name}")
            outputs = self._output_arrays()
            self._execute(inputs, outputs)

        labels = outputs["labels"][0].astype(np.int64, copy=False)
        boxes = outputs["boxes"][0].astype(np.float32, copy=False)
        scores = outputs["scores"][0].astype(np.float32, copy=False)
        return postprocess_detections(labels, boxes, scores, self.settings, target_size)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for pointer, _ in self._buffers.values():
            self._cuda(self.cudart.cudaFree(pointer), "free inference buffer")
        self._buffers.clear()
        self._cuda(self.cudart.cudaStreamDestroy(self.stream), "destroy CUDA stream")
