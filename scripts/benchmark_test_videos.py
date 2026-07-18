#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "opencv-python-headless>=4.11.0.86,<5.0.0",
# ]
# ///
"""Benchmark SwimTrack AI on the eight single-swimmer reference videos.

The runner deliberately keeps the detector configuration outside the request
contract.  It records the operator-declared model/crop metadata and the
authoritative tracking configuration returned by the service, so two runs can
be compared without pretending that a client-side flag changed TensorRT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mimetypes
import os
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Iterator, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2

PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent
DEFAULT_INPUT_DIR = WORKSPACE_DIR / "input_vids"
DEFAULT_RESULTS_ROOT = WORKSPACE_DIR / "results" / "benchmarks"
PRIMARY_VIDEO_IDS = tuple(f"test{number:02d}" for number in range(1, 9))
DIAGNOSTIC_LEVELS = ("none", "counts", "boxes")
TRANSPORTS = ("frames", "video")


class BenchmarkError(ValueError):
    """Raised when a benchmark cannot produce a trustworthy artifact."""


@dataclass(frozen=True)
class VideoSpec:
    """Immutable local-media information used to make a run reproducible."""

    video_id: str
    path: Path
    sha256: str
    bytes: int
    width: int
    height: int
    source_fps: float
    source_frames: int
    duration_ms: float
    sample_stride: int
    tracking_fps: float
    expected_sampled_frames: int


@dataclass(frozen=True)
class EncodedFrame:
    """One JPEG payload equivalent to the Front's historical frame transport."""

    frame_index: int
    time_ms: float
    original_width: int
    original_height: int
    jpeg: bytes


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _finite_number(value: Any, name: str) -> float:
    if not _is_number(value):
        raise BenchmarkError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise BenchmarkError(f"{name} must be finite")
    return result


def _integer(value: Any, name: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise BenchmarkError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise BenchmarkError(f"{name} must be at least {minimum}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dt%H%M%Sz")


def _video_path(input_dir: Path, video_id: str) -> Path:
    candidates = sorted(input_dir.glob(f"*_{video_id}.mp4"))
    if not candidates:
        raise BenchmarkError(f"could not find {video_id} in {input_dir}")
    if len(candidates) != 1:
        raise BenchmarkError(f"expected exactly one file for {video_id} in {input_dir}, found {candidates}")
    return candidates[0]


def select_video_ids(selected: Sequence[str] | None) -> list[str]:
    """Return only the eight primary videos and reject test09 explicitly."""

    if not selected:
        return list(PRIMARY_VIDEO_IDS)
    duplicates = sorted({video_id for video_id in selected if selected.count(video_id) > 1})
    if duplicates:
        raise BenchmarkError(f"duplicate --video selection(s): {duplicates}")
    unsupported = [video_id for video_id in selected if video_id not in PRIMARY_VIDEO_IDS]
    if unsupported:
        raise BenchmarkError(
            f"unsupported video selection(s): {unsupported}; this benchmark intentionally excludes test09"
        )
    return list(selected)


def inspect_video(path: Path, video_id: str, max_fps: float, *, hash_source: bool = True) -> VideoSpec:
    """Read local media metadata without requesting a detector or a GPU."""

    if not path.is_file():
        raise BenchmarkError(f"video does not exist: {path}")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise BenchmarkError(f"could not open video: {path}")
        width = int(round(float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))))
        height = int(round(float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        source_frames = int(round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if width < 1 or height < 1:
        raise BenchmarkError(f"{path} reported invalid dimensions {width}x{height}")
    if source_frames < 1:
        raise BenchmarkError(f"{path} reported no decodable frames")
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise BenchmarkError(f"{path} reported invalid FPS {source_fps!r}")
    if not math.isfinite(max_fps) or max_fps <= 0:
        raise BenchmarkError("max_fps must be finite and greater than zero")
    stride = max(1, math.ceil(source_fps / max_fps))
    expected_sampled_frames = (source_frames - 1) // stride + 1
    return VideoSpec(
        video_id=video_id,
        path=path.resolve(),
        sha256=_sha256(path) if hash_source else "not-requested",
        bytes=path.stat().st_size,
        width=width,
        height=height,
        source_fps=source_fps,
        source_frames=source_frames,
        duration_ms=source_frames * 1_000.0 / source_fps,
        sample_stride=stride,
        tracking_fps=source_fps / stride,
        expected_sampled_frames=expected_sampled_frames,
    )


def inspect_selected_videos(
    input_dir: Path,
    video_ids: Sequence[str],
    max_fps: float,
    *,
    hash_source: bool,
) -> list[VideoSpec]:
    return [
        inspect_video(_video_path(input_dir, video_id), video_id, max_fps, hash_source=hash_source)
        for video_id in video_ids
    ]


def video_metadata(spec: VideoSpec) -> dict[str, Any]:
    return {
        "id": spec.video_id,
        "path": str(spec.path),
        "sha256": spec.sha256,
        "bytes": spec.bytes,
        "media": {
            "width": spec.width,
            "height": spec.height,
            "source_fps": spec.source_fps,
            "source_frames": spec.source_frames,
            "duration_ms": spec.duration_ms,
        },
        "sampling": {
            "sample_stride": spec.sample_stride,
            "tracking_fps": spec.tracking_fps,
            "expected_sampled_frames": spec.expected_sampled_frames,
        },
    }


def _parse_metadata(values: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in values:
        key, separator, raw_value = item.partition("=")
        if not separator or not key:
            raise BenchmarkError("--metadata must use KEY=VALUE syntax")
        if key in result:
            raise BenchmarkError(f"duplicate metadata key {key!r}")
        try:
            result[key] = json.loads(raw_value)
        except json.JSONDecodeError:
            result[key] = raw_value
    return result


def benchmark_configuration(args: argparse.Namespace) -> dict[str, Any]:
    """Return requested settings without claiming they reconfigure the server."""

    frame_transport: dict[str, Any] | None
    if args.transport == "frames":
        frame_transport = {
            "emulation": "OpenCV decode + INTER_LINEAR resize + JPEG upload",
            "inference_size": args.inference_size,
            "jpeg_quality": args.jpeg_quality,
            "batch_size": args.batch_size,
        }
    else:
        frame_transport = None
    return {
        "transport": args.transport,
        "requested_max_fps": args.max_fps,
        "diagnostics": args.diagnostics,
        "lap_calibration_id": None if args.calibration_id == "none" else args.calibration_id,
        "frame_transport": frame_transport,
        "model": {
            "source": "operator_declared",
            "label": args.model_label,
            "artifact": args.model_artifact,
        },
        "crop": {
            "source": "operator_declared_and_session_response",
            "label": args.crop_label,
        },
        "metadata": _parse_metadata(args.metadata),
    }


def _response_text(error: HTTPError) -> str:
    try:
        payload = error.read().decode("utf-8", errors="replace").strip()
    except OSError:
        payload = ""
    return f": {payload[:1_000]}" if payload else ""


def _request(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    payload: bytes | None,
    timeout_seconds: float,
):
    request = Request(url, data=payload, headers=headers, method=method)
    try:
        return urlopen(request, timeout=timeout_seconds)
    except HTTPError as exc:
        raise BenchmarkError(f"{method} {url} returned HTTP {exc.code}{_response_text(exc)}") from exc
    except URLError as exc:
        raise BenchmarkError(f"could not connect to {url}: {exc.reason}") from exc


def _request_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    payload: bytes | None,
    timeout_seconds: float,
    expected_status: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    response = _request(method, url, headers=headers, payload=payload, timeout_seconds=timeout_seconds)
    try:
        status = int(getattr(response, "status", response.getcode()))
        body = response.read()
        response_headers = {key.lower(): value for key, value in response.headers.items()}
    finally:
        response.close()
    if status != expected_status:
        raise BenchmarkError(f"{method} {url} returned HTTP {status}, expected {expected_status}")
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"{method} {url} returned invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise BenchmarkError(f"{method} {url} returned a JSON value that is not an object")
    return decoded, response_headers


def _auth_headers(token: str) -> dict[str, str]:
    return {"X-Swimtrack-Auth": token}


def preflight_service(base_url: str, timeout_seconds: float) -> dict[str, Any]:
    """Capture readiness before mutating a tracking session."""

    url = f"{base_url.rstrip('/')}/readyz"
    response = _request("GET", url, headers={}, payload=None, timeout_seconds=timeout_seconds)
    try:
        status = int(getattr(response, "status", response.getcode()))
        body = response.read()
        headers = {key.lower(): value for key, value in response.headers.items()}
    finally:
        response.close()
    if status != 200:
        raise BenchmarkError(f"{url} returned HTTP {status}, expected 200")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"{url} returned invalid JSON") from exc
    if not isinstance(payload, dict) or payload.get("status") != "ready":
        raise BenchmarkError(f"{url} did not report a ready service")
    return {"url": url, "response": payload, "headers": headers}


def create_session(base_url: str, token: str, spec: VideoSpec, args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {"fps": spec.tracking_fps, "diagnostics": args.diagnostics}
    if args.calibration_id != "none":
        payload["lap_calibration_id"] = args.calibration_id
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    response, _ = _request_json(
        "POST",
        f"{base_url.rstrip('/')}/v1/tracking-sessions",
        headers={**_auth_headers(token), "Content-Type": "application/json", "Accept": "application/json"},
        payload=body,
        timeout_seconds=args.timeout_seconds,
        expected_status=201,
    )
    session_id = response.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise BenchmarkError("session creation response has no valid session_id")
    if response.get("next_sequence") != 0:
        raise BenchmarkError("session creation response did not start at sequence zero")
    return response


def close_session(base_url: str, token: str, session_id: str, timeout_seconds: float) -> str | None:
    try:
        response = _request(
            "DELETE",
            f"{base_url.rstrip('/')}/v1/tracking-sessions/{session_id}",
            headers=_auth_headers(token),
            payload=None,
            timeout_seconds=timeout_seconds,
        )
        try:
            status = int(getattr(response, "status", response.getcode()))
        finally:
            response.close()
        if status != 204:
            return f"DELETE returned HTTP {status}, expected 204"
    except BenchmarkError as exc:
        return str(exc)
    return None


def _multipart(
    fields: Sequence[tuple[str, str]],
    files: Sequence[tuple[str, str, bytes, str]],
    *,
    boundary: str | None = None,
) -> tuple[str, bytes]:
    """Build the small benchmark multipart bodies without an extra HTTP dependency."""

    boundary = boundary or f"----swimtrack-benchmark-{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for name, value in fields:
        parts.extend(
            (
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode("utf-8"),
                b"\r\n",
            )
        )
    for name, filename, content, content_type in files:
        parts.extend(
            (
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode(),
                f"Content-Type: {content_type}\r\n\r\n".encode(),
                content,
                b"\r\n",
            )
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return f"multipart/form-data; boundary={boundary}", b"".join(parts)


def _frame_contract(payload: Any, previous_index: int | None, previous_time: float | None) -> tuple[int, float]:
    if not isinstance(payload, dict):
        raise BenchmarkError("video endpoint emitted a JSON value that is not an object")
    frame_index = _integer(payload.get("frame_index"), "frame_index", minimum=0)
    time_ms = _finite_number(payload.get("time_ms"), "time_ms")
    if time_ms < 0:
        raise BenchmarkError("time_ms cannot be negative")
    _integer(payload.get("width"), "width", minimum=1)
    _integer(payload.get("height"), "height", minimum=1)
    if not isinstance(payload.get("boxes"), list):
        raise BenchmarkError(f"frame {frame_index}: boxes must be a list")
    if previous_index is not None and frame_index <= previous_index:
        raise BenchmarkError(f"frame indexes are not strictly increasing ({previous_index}, {frame_index})")
    if previous_time is not None and time_ms < previous_time:
        raise BenchmarkError(f"frame timestamps move backwards ({previous_time}, {time_ms})")
    return frame_index, time_ms


def _write_frame(destination, frame: dict[str, Any]) -> None:
    destination.write(json.dumps(frame, separators=(",", ":"), sort_keys=True) + "\n")


def _numeric_header(headers: dict[str, str], name: str) -> float | None:
    raw = headers.get(name.lower())
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) and value >= 0 else None


def _summarize_values(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "mean": mean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def _consume_ndjson(response, destination: Path, started_at: float) -> tuple[int, float | None]:
    frame_count = 0
    first_frame_ms: float | None = None
    previous_index: int | None = None
    previous_time: float | None = None
    with destination.open("w", encoding="utf-8") as output:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="strict").strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BenchmarkError("video endpoint emitted invalid NDJSON") from exc
            frame_index, frame_time = _frame_contract(payload, previous_index, previous_time)
            previous_index = frame_index
            previous_time = frame_time
            _write_frame(output, payload)
            frame_count += 1
            if first_frame_ms is None:
                first_frame_ms = (time.perf_counter() - started_at) * 1_000.0
    if frame_count == 0:
        raise BenchmarkError("video endpoint did not emit any frames")
    return frame_count, first_frame_ms


def _run_video_transport(
    base_url: str,
    token: str,
    spec: VideoSpec,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Upload one original MP4 and persist the NDJSON response incrementally."""

    session = create_session(base_url, token, spec, args)
    session_id = str(session["session_id"])
    started_at = time.perf_counter()
    close_error: str | None = None
    try:
        payload_started = time.perf_counter()
        filename = f"upload{spec.path.suffix.lower() or '.mp4'}"
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        multipart_type, body = _multipart(
            [("sample_fps", f"{spec.tracking_fps:.12g}")],
            [("video", filename, spec.path.read_bytes(), content_type)],
        )
        payload_build_ms = (time.perf_counter() - payload_started) * 1_000.0
        request_started = time.perf_counter()
        response = _request(
            "POST",
            f"{base_url.rstrip('/')}/v1/tracking-sessions/{session_id}/video",
            headers={
                **_auth_headers(token),
                "Accept": "application/x-ndjson",
                "Content-Type": multipart_type,
            },
            payload=body,
            timeout_seconds=args.timeout_seconds,
        )
        try:
            status = int(getattr(response, "status", response.getcode()))
            headers = {key.lower(): value for key, value in response.headers.items()}
            if status != 200:
                raise BenchmarkError(f"video endpoint returned HTTP {status}, expected 200")
            media_type = headers.get("content-type", "").split(";", maxsplit=1)[0].strip().lower()
            if media_type != "application/x-ndjson":
                raise BenchmarkError(
                    f"video endpoint returned Content-Type {media_type or 'missing'}, expected application/x-ndjson"
                )
            frames, first_frame_ms = _consume_ndjson(response, output_dir / "frames.ndjson", request_started)
        finally:
            response.close()
        return session, {
            "transport": "video",
            "payload_build_ms": payload_build_ms,
            "uploaded_bytes": len(body),
            "response_frames": frames,
            "expected_response_frames": spec.expected_sampled_frames,
            "response_frame_count_matches_expected": frames == spec.expected_sampled_frames,
            "first_result_ms": first_frame_ms,
            "first_result_end_to_end_ms": (
                first_frame_ms + (request_started - started_at) * 1_000.0 if first_frame_ms is not None else None
            ),
            "request_wall_ms": (time.perf_counter() - request_started) * 1_000.0,
            "client_wall_ms": (time.perf_counter() - started_at) * 1_000.0,
            "response_headers": headers,
            "server_decode_path": headers.get("x-swimtrack-decode-path"),
            "server_decode_backend": headers.get("x-swimtrack-decode-backend"),
        }
    finally:
        close_error = close_session(base_url, token, session_id, args.timeout_seconds)
        if close_error is not None:
            (output_dir / "session-close-warning.txt").write_text(close_error + "\n", encoding="utf-8")


def _encoded_frames(spec: VideoSpec, args: argparse.Namespace) -> Iterator[tuple[EncodedFrame, dict[str, float]]]:
    """Yield the Front-compatible JPEG sequence, measuring local preparation."""

    capture = cv2.VideoCapture(str(spec.path))
    try:
        if not capture.isOpened():
            raise BenchmarkError(f"could not open video: {spec.path}")
        source_index = 0
        while True:
            decode_started = time.perf_counter()
            ok, frame = capture.read()
            decode_ms = (time.perf_counter() - decode_started) * 1_000.0
            if not ok:
                break
            if source_index % spec.sample_stride:
                source_index += 1
                continue
            height, width = frame.shape[:2]
            resize_started = time.perf_counter()
            resized = cv2.resize(
                frame,
                (args.inference_size, args.inference_size),
                interpolation=cv2.INTER_LINEAR,
            )
            resize_ms = (time.perf_counter() - resize_started) * 1_000.0
            jpeg_started = time.perf_counter()
            encoded, jpeg = cv2.imencode(
                ".jpg",
                resized,
                [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
            )
            jpeg_ms = (time.perf_counter() - jpeg_started) * 1_000.0
            if not encoded:
                raise BenchmarkError(f"could not encode source frame {source_index}")
            yield (
                EncodedFrame(
                    frame_index=source_index,
                    time_ms=source_index * 1_000.0 / spec.source_fps,
                    original_width=int(width),
                    original_height=int(height),
                    jpeg=jpeg.tobytes(),
                ),
                {"decode_ms": decode_ms, "resize_ms": resize_ms, "jpeg_encode_ms": jpeg_ms},
            )
            source_index += 1
    finally:
        capture.release()


def _batched(
    items: Iterable[tuple[EncodedFrame, dict[str, float]]],
    size: int,
) -> Iterator[list[tuple[EncodedFrame, dict[str, float]]]]:
    batch: list[tuple[EncodedFrame, dict[str, float]]] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _send_frame_batch(
    base_url: str,
    token: str,
    session_id: str,
    sequence: int,
    frames: Sequence[EncodedFrame],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, str], float]:
    metadata = {
        "batch_id": f"benchmark-{sequence:06d}",
        "sequence": sequence,
        "frames": [
            {
                "frame_index": frame.frame_index,
                "time_ms": frame.time_ms,
                "original_width": frame.original_width,
                "original_height": frame.original_height,
            }
            for frame in frames
        ],
    }
    multipart_type, body = _multipart(
        [("metadata", json.dumps(metadata, separators=(",", ":")))],
        [
            ("frames", f"frame-{frame.frame_index:08d}.jpg", frame.jpeg, "image/jpeg")
            for frame in frames
        ],
    )
    started_at = time.perf_counter()
    response, headers = _request_json(
        "POST",
        f"{base_url.rstrip('/')}/v1/tracking-sessions/{session_id}/batches",
        headers={**_auth_headers(token), "Accept": "application/json", "Content-Type": multipart_type},
        payload=body,
        timeout_seconds=args.timeout_seconds,
        expected_status=200,
    )
    elapsed_ms = (time.perf_counter() - started_at) * 1_000.0
    payload_frames = response.get("frames")
    if not isinstance(payload_frames, list) or len(payload_frames) != len(frames):
        raise BenchmarkError(f"batch {sequence} response does not contain one result per submitted frame")
    if response.get("sequence") != sequence or response.get("next_sequence") != sequence + 1:
        raise BenchmarkError(f"batch {sequence} response did not preserve the session sequence")
    result_frames: list[dict[str, Any]] = []
    previous_index: int | None = None
    previous_time: float | None = None
    for expected, payload_frame in zip(frames, payload_frames):
        frame_index, frame_time = _frame_contract(payload_frame, previous_index, previous_time)
        if frame_index != expected.frame_index or not math.isclose(frame_time, expected.time_ms, abs_tol=1e-3):
            raise BenchmarkError(f"batch {sequence} response does not preserve submitted frame metadata")
        previous_index = frame_index
        previous_time = frame_time
        result_frames.append(payload_frame)
    return result_frames, headers, elapsed_ms


def _run_frames_transport(
    base_url: str,
    token: str,
    spec: VideoSpec,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the original JPEG transport, preserving its source-frame selection."""

    session = create_session(base_url, token, spec, args)
    session_id = str(session["session_id"])
    started_at = time.perf_counter()
    close_error: str | None = None
    telemetry: dict[str, list[float]] = defaultdict(list)
    preparation: dict[str, float | int] = {
        "source_frames_read": 0,
        "selected_frames": 0,
        "encoded_jpeg_bytes": 0,
        "decode_ms": 0.0,
        "resize_ms": 0.0,
        "jpeg_encode_ms": 0.0,
    }
    response_frames = 0
    batches = 0
    first_result_ms: float | None = None
    try:
        with (output_dir / "frames.ndjson").open("w", encoding="utf-8") as output:
            for batch_with_timings in _batched(_encoded_frames(spec, args), args.batch_size):
                frames = [item[0] for item in batch_with_timings]
                for frame, timing in batch_with_timings:
                    preparation["selected_frames"] = int(preparation["selected_frames"]) + 1
                    preparation["encoded_jpeg_bytes"] = int(preparation["encoded_jpeg_bytes"]) + len(frame.jpeg)
                    for key in ("decode_ms", "resize_ms", "jpeg_encode_ms"):
                        preparation[key] = float(preparation[key]) + timing[key]
                results, headers, request_ms = _send_frame_batch(
                    base_url,
                    token,
                    session_id,
                    batches,
                    frames,
                    args,
                )
                telemetry["client_batch_ms"].append(request_ms)
                for header_name, telemetry_name in (
                    ("x-swimtrack-decode-ms", "server_decode_ms"),
                    ("x-swimtrack-process-ms", "server_process_ms"),
                    ("x-swimtrack-total-ms", "server_total_ms"),
                ):
                    value = _numeric_header(headers, header_name)
                    if value is not None:
                        telemetry[telemetry_name].append(value)
                for frame in results:
                    _write_frame(output, frame)
                    response_frames += 1
                    if first_result_ms is None:
                        first_result_ms = (time.perf_counter() - started_at) * 1_000.0
                batches += 1
        preparation["source_frames_read"] = spec.source_frames
        if response_frames == 0:
            raise BenchmarkError("frame transport did not emit any frames")
        return session, {
            "transport": "frames",
            "response_frames": response_frames,
            "expected_response_frames": spec.expected_sampled_frames,
            "response_frame_count_matches_expected": response_frames == spec.expected_sampled_frames,
            "first_result_ms": first_result_ms,
            "client_wall_ms": (time.perf_counter() - started_at) * 1_000.0,
            "batches": batches,
            "preparation": preparation,
            "batch_telemetry_ms": {name: _summarize_values(values) for name, values in sorted(telemetry.items())},
        }
    finally:
        close_error = close_session(base_url, token, session_id, args.timeout_seconds)
        if close_error is not None:
            (output_dir / "session-close-warning.txt").write_text(close_error + "\n", encoding="utf-8")


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    if not 0 <= quantile <= 1:
        raise BenchmarkError("percentile quantile must be in [0, 1]")
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def _stage_summary(counts: Sequence[int], total_frames: int) -> dict[str, Any]:
    observations = sum(counts)
    nonempty = sum(count > 0 for count in counts)
    return {
        "observations": observations,
        "frames_nonempty": nonempty,
        "frame_coverage": nonempty / total_frames if total_frames else 0.0,
        "mean_per_frame": observations / len(counts) if counts else None,
        "p50_per_frame": percentile([float(count) for count in counts], 0.50),
        "p95_per_frame": percentile([float(count) for count in counts], 0.95),
        "max_per_frame": max(counts, default=0),
    }


def _stage_count(value: Any, name: str) -> int:
    if not isinstance(value, dict):
        raise BenchmarkError(f"{name} must be an object")
    return _integer(value.get("count"), f"{name}.count", minimum=0)


def _gaps(presence: Sequence[bool], sample_fps: float | None) -> dict[str, Any]:
    if not presence or not any(presence):
        values: list[int] = []
    else:
        first = presence.index(True)
        last = len(presence) - 1 - list(reversed(presence)).index(True)
        values = []
        current = 0
        for visible in presence[first : last + 1]:
            if visible:
                if current:
                    values.append(current)
                    current = 0
            else:
                current += 1
        if current:
            values.append(current)
    seconds = [value / sample_fps for value in values] if sample_fps else []
    return {
        "count": len(values),
        "frames": {
            "p50": percentile([float(value) for value in values], 0.50),
            "p95": percentile([float(value) for value in values], 0.95),
            "max": max(values, default=0),
        },
        "seconds": {
            "p50": percentile(seconds, 0.50),
            "p95": percentile(seconds, 0.95),
            "max": max(seconds, default=0.0),
        },
    }


def _inferred_fps(times: Sequence[float]) -> float | None:
    deltas = [later - earlier for earlier, later in zip(times, times[1:]) if later > earlier]
    if not deltas:
        return None
    value = median(deltas)
    return 1_000.0 / value if value > 0 else None


def summarize_frames(frames: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate public boxes and each optional detector/tracker diagnostic stage."""

    total_frames = 0
    times: list[float] = []
    active_counts: list[int] = []
    active_presence: list[bool] = []
    candidate_counts: list[int] = []
    accepted_counts: list[int] = []
    weak_candidate_counts: list[int] = []
    roi_counts: list[int] = []
    weak_roi_counts: list[int] = []
    diagnostic_active_counts: list[int] = []
    weak_reactivated_counts: list[int] = []
    lost_counts: list[int] = []
    diagnostics_frames = 0
    diagnostic_floors: set[float] = set()
    track_occurrences: dict[tuple[str | None, int], list[int]] = defaultdict(list)
    lane_track_ids: dict[str | None, set[int]] = defaultdict(set)
    weak_reactivated_track_ids: dict[str, set[int]] = defaultdict(set)
    lap_scores: list[float] = []
    evaluable_lap_frames = 0
    previous_index: int | None = None
    previous_time: float | None = None
    accepted_no_track_frames = 0
    detector_accepted_no_track_frames = 0

    for ordinal, frame in enumerate(frames):
        frame_index, frame_time = _frame_contract(frame, previous_index, previous_time)
        previous_index = frame_index
        previous_time = frame_time
        total_frames += 1
        times.append(frame_time)
        boxes = frame["boxes"]
        active_ids: set[tuple[str | None, int]] = set()
        for box_index, box in enumerate(boxes):
            if not isinstance(box, dict):
                raise BenchmarkError(f"frame {frame_index}, box {box_index} must be an object")
            track_id = _integer(box.get("id"), f"frame {frame_index}, box {box_index}.id")
            lane_id = box.get("lane_id")
            if lane_id is not None and not isinstance(lane_id, str):
                raise BenchmarkError(f"frame {frame_index}, box {box_index}.lane_id must be a string or null")
            key = (lane_id, track_id)
            if key in active_ids:
                raise BenchmarkError(f"frame {frame_index} contains duplicate active track {key}")
            active_ids.add(key)
            track_occurrences[key].append(ordinal)
            lane_track_ids[lane_id].add(track_id)
        active_counts.append(len(active_ids))
        active_presence.append(bool(active_ids))

        scores = frame.get("lap_scores")
        if scores is not None:
            if not isinstance(scores, list):
                raise BenchmarkError(f"frame {frame_index}.lap_scores must be a list")
            for score in scores:
                if not isinstance(score, dict):
                    raise BenchmarkError(f"frame {frame_index}.lap_scores contains a non-object")
                lap_scores.append(_finite_number(score.get("lap_score"), f"frame {frame_index}.lap_score"))
                if score.get("evaluable") is True:
                    evaluable_lap_frames += 1

        diagnostics = frame.get("tracking_diagnostics")
        if diagnostics is None:
            continue
        if not isinstance(diagnostics, dict):
            raise BenchmarkError(f"frame {frame_index}.tracking_diagnostics must be an object")
        diagnostics_frames += 1
        diagnostic_floors.add(_finite_number(diagnostics.get("diagnostic_floor"), "diagnostic_floor"))
        candidates = _stage_count(diagnostics.get("person_candidates"), "person_candidates")
        accepted = _stage_count(diagnostics.get("detector_accepted"), "detector_accepted")
        weak_candidates = _stage_count(diagnostics.get("weak_candidates"), "weak_candidates")
        lanes = diagnostics.get("lanes")
        if not isinstance(lanes, list):
            raise BenchmarkError(f"frame {frame_index}.tracking_diagnostics.lanes must be a list")
        after_roi = 0
        weak_after_roi = 0
        retained_lost = 0
        diagnostic_active_ids: set[tuple[str | None, int]] = set()
        frame_weak_reactivated_ids: set[tuple[str, int]] = set()
        for lane_index, lane in enumerate(lanes):
            if not isinstance(lane, dict):
                raise BenchmarkError(f"frame {frame_index}, lane {lane_index} must be an object")
            lane_id = lane.get("lane_id")
            if not isinstance(lane_id, str) or not lane_id:
                raise BenchmarkError(f"frame {frame_index}, lane {lane_index}.lane_id must be non-empty")
            after_roi += _stage_count(lane.get("after_roi"), f"lane {lane_id}.after_roi")
            weak_after_roi += _stage_count(
                lane.get("weak_candidates_after_roi"),
                f"lane {lane_id}.weak_candidates_after_roi",
            )
            retained_lost += _integer(
                lane.get("retained_lost_track_count"),
                f"lane {lane_id}.retained_lost_track_count",
                minimum=0,
            )
            lane_active = lane.get("active_track_ids")
            if not isinstance(lane_active, list):
                raise BenchmarkError(f"lane {lane_id}.active_track_ids must be a list")
            for track_id in lane_active:
                key = (None if lane_id == "global" else lane_id, _integer(track_id, "active_track_id"))
                if key in diagnostic_active_ids:
                    raise BenchmarkError(f"frame {frame_index}: duplicate diagnostic active track {key}")
                diagnostic_active_ids.add(key)
            reactivated = lane.get("weak_reactivated_track_ids")
            if not isinstance(reactivated, list):
                raise BenchmarkError(f"lane {lane_id}.weak_reactivated_track_ids must be a list")
            for track_id in reactivated:
                key = (lane_id, _integer(track_id, "weak_reactivated_track_id"))
                if key in frame_weak_reactivated_ids:
                    raise BenchmarkError(f"frame {frame_index}: duplicate weak reactivation for track {key}")
                frame_weak_reactivated_ids.add(key)
                weak_reactivated_track_ids[lane_id].add(key[1])
        if after_roi > accepted:
            raise BenchmarkError(f"frame {frame_index}: after_roi cannot exceed detector_accepted")
        if weak_candidates > candidates:
            raise BenchmarkError(f"frame {frame_index}: weak_candidates cannot exceed person_candidates")
        if weak_after_roi > weak_candidates:
            raise BenchmarkError(f"frame {frame_index}: weak_candidates_after_roi cannot exceed weak_candidates")
        candidate_counts.append(candidates)
        accepted_counts.append(accepted)
        weak_candidate_counts.append(weak_candidates)
        roi_counts.append(after_roi)
        weak_roi_counts.append(weak_after_roi)
        diagnostic_active_counts.append(len(diagnostic_active_ids))
        weak_reactivated_counts.append(len(frame_weak_reactivated_ids))
        lost_counts.append(retained_lost)
        if after_roi > 0 and not active_ids:
            accepted_no_track_frames += 1
        if accepted > 0 and not active_ids:
            detector_accepted_no_track_frames += 1

    if total_frames == 0:
        raise BenchmarkError("cannot summarize an empty frame stream")
    sample_fps = _inferred_fps(times)
    longest_track: dict[str, Any] = {
        "lane_id": None,
        "track_id": None,
        "frames": 0,
        "seconds": 0.0,
    }
    same_id_reacquisitions = 0
    max_same_id_gap = 0
    for (lane_id, track_id), occurrences in track_occurrences.items():
        current = 1
        longest = 1
        current_start = occurrences[0]
        longest_start = occurrences[0]
        for previous, current_index in zip(occurrences, occurrences[1:]):
            if current_index == previous + 1:
                current += 1
            else:
                same_id_reacquisitions += 1
                max_same_id_gap = max(max_same_id_gap, current_index - previous - 1)
                if current > longest:
                    longest = current
                    longest_start = current_start
                current = 1
                current_start = current_index
        if current > longest:
            longest = current
            longest_start = current_start
        seconds = longest / sample_fps if sample_fps else None
        if longest > longest_track["frames"]:
            longest_track = {
                "lane_id": lane_id,
                "track_id": track_id,
                "frames": longest,
                "seconds": seconds,
                "start_time_ms": times[longest_start],
                "end_time_ms": times[longest_start + longest - 1],
            }
    fragmentations = sum(max(0, len(track_ids) - 1) for track_ids in lane_track_ids.values())
    diagnostics_available = diagnostics_frames > 0
    diagnostics: dict[str, Any] = {
        "available": diagnostics_available,
        "frames": diagnostics_frames,
        "frame_coverage": diagnostics_frames / total_frames,
        "diagnostic_floors": sorted(diagnostic_floors),
        "stages": None,
        "funnel": None,
        "accepted_no_track_frames": None,
        "detector_accepted_no_track_frames": None,
        "retained_lost": None,
    }
    if diagnostics_available:
        stages = {
            "person_candidates": _stage_summary(candidate_counts, total_frames),
            "detector_accepted": _stage_summary(accepted_counts, total_frames),
            "weak_candidates": _stage_summary(weak_candidate_counts, total_frames),
            "after_roi": _stage_summary(roi_counts, total_frames),
            "weak_candidates_after_roi": _stage_summary(weak_roi_counts, total_frames),
            "active_tracks": _stage_summary(diagnostic_active_counts, total_frames),
        }
        candidates = stages["person_candidates"]["observations"]
        accepted = stages["detector_accepted"]["observations"]
        weak_candidates = stages["weak_candidates"]["observations"]
        after_roi = stages["after_roi"]["observations"]
        weak_after_roi = stages["weak_candidates_after_roi"]["observations"]
        diagnostics.update(
            {
                "stages": stages,
                "funnel": {
                    "candidate_to_accepted": accepted / candidates if candidates else None,
                    "accepted_to_roi": after_roi / accepted if accepted else None,
                    "candidate_to_weak": weak_candidates / candidates if candidates else None,
                    "weak_to_roi": weak_after_roi / weak_candidates if weak_candidates else None,
                },
                "accepted_no_track_frames": accepted_no_track_frames,
                "detector_accepted_no_track_frames": detector_accepted_no_track_frames,
                "retained_lost": {
                    "frames_nonempty": sum(value > 0 for value in lost_counts),
                    "mean_per_diagnostic_frame": mean(lost_counts) if lost_counts else None,
                    "peak": max(lost_counts, default=0),
                },
            }
        )
    return {
        "schema_version": 1,
        "frame_count": total_frames,
        "first_time_ms": times[0],
        "last_time_ms": times[-1],
        "timeline_span_ms": times[-1] - times[0],
        "inferred_fps": sample_fps,
        "diagnostics": diagnostics,
        "tracking": {
            "active_tracks": _stage_summary(active_counts, total_frames),
            "unique_track_ids": len(track_occurrences),
            "fragmentations": fragmentations,
            "track_ids_by_lane": {
                "global" if lane_id is None else lane_id: sorted(track_ids)
                for lane_id, track_ids in sorted(lane_track_ids.items(), key=lambda item: str(item[0]))
            },
            "longest_consecutive_run": longest_track,
            "internal_active_gaps": _gaps(active_presence, sample_fps),
            "same_id_reacquisitions": same_id_reacquisitions,
            "longest_same_id_reacquisition_gap_frames": max_same_id_gap,
            "weak_reactivations": {
                "events": sum(weak_reactivated_counts),
                "frames_nonempty": sum(count > 0 for count in weak_reactivated_counts),
                "unique_track_ids": sum(len(track_ids) for track_ids in weak_reactivated_track_ids.values()),
                "track_ids_by_lane": {
                    lane_id: sorted(track_ids) for lane_id, track_ids in sorted(weak_reactivated_track_ids.items())
                },
            },
        },
        "lap_scores": {
            "observations": len(lap_scores),
            "maximum": max(lap_scores, default=None),
            "mean": mean(lap_scores) if lap_scores else None,
            "frames_evaluable": evaluable_lap_frames,
        },
    }


def _iter_ndjson(path: Path) -> Iterator[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, raw_line in enumerate(source, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise BenchmarkError(f"{path}:{line_number} contains invalid JSON") from exc
                if not isinstance(payload, dict):
                    raise BenchmarkError(f"{path}:{line_number} must contain a JSON object")
                yield payload
    except OSError as exc:
        raise BenchmarkError(f"could not read {path}: {exc}") from exc


def summarize_ndjson(path: Path) -> dict[str, Any]:
    return summarize_frames(_iter_ndjson(path))


def aggregate_videos(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pool count-based metrics while retaining per-video results as the authority."""

    total_frames = sum(int(result["analysis"]["frame_count"]) for result in results)
    total_wall_ms = sum(float(result["execution"].get("client_wall_ms", 0.0)) for result in results)
    total_response_frames = sum(int(result["execution"].get("response_frames", 0)) for result in results)
    aggregate: dict[str, Any] = {
        "video_ids": [result["video"]["id"] for result in results],
        "video_count": len(results),
        "frame_count": total_frames,
        "response_frames": total_response_frames,
        "client_wall_ms": total_wall_ms,
        "response_fps": total_response_frames / (total_wall_ms / 1_000.0) if total_wall_ms else None,
        "diagnostics": {
            "available_for_all_videos": all(
                result["analysis"]["diagnostics"]["available"] for result in results
            )
        },
        "tracking": {
            "unique_track_ids_sum": sum(result["analysis"]["tracking"]["unique_track_ids"] for result in results),
            "fragmentations_sum": sum(result["analysis"]["tracking"]["fragmentations"] for result in results),
            "max_internal_active_gap_frames": max(
                (result["analysis"]["tracking"]["internal_active_gaps"]["frames"]["max"] for result in results),
                default=0,
            ),
            "weak_reactivation_events_sum": sum(
                result["analysis"]["tracking"]["weak_reactivations"]["events"] for result in results
            ),
            "weak_reactivation_frames_sum": sum(
                result["analysis"]["tracking"]["weak_reactivations"]["frames_nonempty"] for result in results
            ),
        },
    }
    if aggregate["diagnostics"]["available_for_all_videos"]:
        stages: dict[str, dict[str, Any]] = {}
        for stage_name in (
            "person_candidates",
            "detector_accepted",
            "weak_candidates",
            "after_roi",
            "weak_candidates_after_roi",
            "active_tracks",
        ):
            observations = sum(
                result["analysis"]["diagnostics"]["stages"][stage_name]["observations"] for result in results
            )
            frames_nonempty = sum(
                result["analysis"]["diagnostics"]["stages"][stage_name]["frames_nonempty"] for result in results
            )
            stages[stage_name] = {
                "observations": observations,
                "frames_nonempty": frames_nonempty,
                "frame_coverage": frames_nonempty / total_frames if total_frames else 0.0,
                "mean_per_frame": observations / total_frames if total_frames else None,
            }
        candidates = stages["person_candidates"]["observations"]
        accepted = stages["detector_accepted"]["observations"]
        weak_candidates = stages["weak_candidates"]["observations"]
        after_roi = stages["after_roi"]["observations"]
        weak_after_roi = stages["weak_candidates_after_roi"]["observations"]
        aggregate["diagnostics"].update(
            {
                "stages": stages,
                "funnel": {
                    "candidate_to_accepted": accepted / candidates if candidates else None,
                    "accepted_to_roi": after_roi / accepted if accepted else None,
                    "candidate_to_weak": weak_candidates / candidates if candidates else None,
                    "weak_to_roi": weak_after_roi / weak_candidates if weak_candidates else None,
                },
                "accepted_no_track_frames": sum(
                    result["analysis"]["diagnostics"]["accepted_no_track_frames"] for result in results
                ),
                "detector_accepted_no_track_frames": sum(
                    result["analysis"]["diagnostics"]["detector_accepted_no_track_frames"] for result in results
                ),
            }
        )
    return aggregate


def _validate_args(args: argparse.Namespace) -> None:
    if args.transport not in TRANSPORTS:
        raise BenchmarkError(f"transport must be one of {TRANSPORTS}")
    if args.diagnostics not in DIAGNOSTIC_LEVELS:
        raise BenchmarkError(f"diagnostics must be one of {DIAGNOSTIC_LEVELS}")
    if not math.isfinite(args.max_fps) or args.max_fps <= 0:
        raise BenchmarkError("--max-fps must be finite and greater than zero")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        raise BenchmarkError("--timeout-seconds must be finite and greater than zero")
    if args.inference_size < 1:
        raise BenchmarkError("--inference-size must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        raise BenchmarkError("--jpeg-quality must be between 1 and 100")
    if args.batch_size < 1:
        raise BenchmarkError("--batch-size must be positive")
    if args.mode == "remote" and not args.base_url:
        raise BenchmarkError("--base-url is required in remote mode")
    if args.mode == "remote" and not args.auth_token:
        raise BenchmarkError("--auth-token or SWIMTRACK_BENCHMARK_AUTH_TOKEN is required in remote mode")


def _result_directory(args: argparse.Namespace) -> Path:
    run_id = args.run_id or _run_id()
    if not run_id.replace("_", "").replace("-", "").isalnum():
        raise BenchmarkError("--run-id may contain only alphanumeric characters, underscores, and hyphens")
    root = args.results_root.resolve()
    result_dir = root / run_id
    if result_dir.exists():
        raise BenchmarkError(f"result directory already exists: {result_dir}; choose another --run-id")
    return result_dir


def run_diagnostics(args: argparse.Namespace, specs: Sequence[VideoSpec], result_dir: Path) -> dict[str, Any]:
    result = {
        "schema_version": 1,
        "status": "diagnostic_only",
        "created_at": _timestamp(),
        "reason": "No HTTP request, TensorRT inference, CUDA, or NVDEC is performed in diagnostic mode.",
        "benchmark_configuration": benchmark_configuration(args),
        "videos": [video_metadata(spec) for spec in specs],
    }
    _atomic_json(result_dir / "run.json", result)
    return result


def run_remote(args: argparse.Namespace, specs: Sequence[VideoSpec], result_dir: Path) -> dict[str, Any]:
    """Execute one session per video and leave useful artifacts after a failure."""

    base_url = args.base_url.rstrip("/")
    run = {
        "schema_version": 1,
        "status": "running",
        "started_at": _timestamp(),
        "finished_at": None,
        "benchmark_configuration": benchmark_configuration(args),
        "service_preflight": None,
        "videos": [],
    }
    _atomic_json(result_dir / "run.json", run)
    completed: list[dict[str, Any]] = []
    try:
        run["service_preflight"] = preflight_service(base_url, args.timeout_seconds)
        _atomic_json(result_dir / "run.json", run)
        for spec in specs:
            video_dir = result_dir / spec.video_id
            video_dir.mkdir(parents=True, exist_ok=False)
            _atomic_json(video_dir / "input.json", video_metadata(spec))
            if args.transport == "video":
                session, execution = _run_video_transport(base_url, args.auth_token, spec, args, video_dir)
            else:
                session, execution = _run_frames_transport(base_url, args.auth_token, spec, args, video_dir)
            analysis = summarize_ndjson(video_dir / "frames.ndjson")
            result = {
                "schema_version": 1,
                "status": "completed",
                "video": video_metadata(spec),
                "session": {
                    "tracking_configuration": session.get("tracking_configuration"),
                    "expires_in_seconds": session.get("expires_in_seconds"),
                },
                "execution": execution,
                "analysis": analysis,
            }
            _atomic_json(video_dir / "result.json", result)
            completed.append(result)
            run["videos"].append({"id": spec.video_id, "status": "completed", "path": str(video_dir / "result.json")})
            _atomic_json(result_dir / "run.json", run)
        aggregate = aggregate_videos(completed)
        _atomic_json(result_dir / "aggregate.json", aggregate)
        run["status"] = "completed"
        run["finished_at"] = _timestamp()
        run["aggregate_path"] = str(result_dir / "aggregate.json")
        _atomic_json(result_dir / "run.json", run)
        return {"run": run, "aggregate": aggregate}
    except BaseException:
        run["status"] = "failed"
        run["finished_at"] = _timestamp()
        _atomic_json(result_dir / "run.json", run)
        if completed:
            _atomic_json(result_dir / "aggregate.partial.json", aggregate_videos(completed))
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("diagnose", "remote"), default="diagnose")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-id", help="Stable result-directory name; defaults to a UTC timestamp.")
    parser.add_argument("--video", action="append", help="Only test01 through test08; repeat to select several.")
    parser.add_argument("--transport", choices=TRANSPORTS, default="video")
    parser.add_argument("--max-fps", type=float, default=30.0)
    parser.add_argument("--diagnostics", choices=DIAGNOSTIC_LEVELS, default="boxes")
    parser.add_argument("--calibration-id", default="fixed-camera-v1", help="Use 'none' to omit calibration.")
    parser.add_argument("--base-url", help="Private SwimTrack AI URL, for example http://10.0.218.101:7001.")
    parser.add_argument("--auth-token", default=os.environ.get("SWIMTRACK_BENCHMARK_AUTH_TOKEN", ""))
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--inference-size", type=int, default=640, help="Only for --transport frames.")
    parser.add_argument("--jpeg-quality", type=int, default=85, help="Only for --transport frames.")
    parser.add_argument("--batch-size", type=int, default=4, help="Only for --transport frames.")
    parser.add_argument(
        "--model-label",
        default="undeclared",
        help="Recorded metadata; it does not change the server model.",
    )
    parser.add_argument(
        "--model-artifact",
        default=None,
        help="Recorded metadata; it does not change the server model.",
    )
    parser.add_argument(
        "--crop-label",
        default="undeclared",
        help="Recorded metadata; session config is authoritative.",
    )
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra JSON-or-text metadata.",
    )
    parser.add_argument(
        "--no-source-hash",
        action="store_true",
        help="Skip SHA-256 only when a faster local diagnostic is needed.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        _validate_args(args)
        video_ids = select_video_ids(args.video)
        specs = inspect_selected_videos(
            args.input_dir.resolve(),
            video_ids,
            args.max_fps,
            hash_source=not args.no_source_hash,
        )
        result_dir = _result_directory(args)
        result_dir.mkdir(parents=True, exist_ok=False)
        if args.mode == "diagnose":
            result = run_diagnostics(args, specs, result_dir)
            print(json.dumps({"status": result["status"], "result_dir": str(result_dir)}, sort_keys=True))
        else:
            result = run_remote(args, specs, result_dir)
            print(json.dumps({"status": result["run"]["status"], "result_dir": str(result_dir)}, sort_keys=True))
        return 0
    except (BenchmarkError, OSError, cv2.error) as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
