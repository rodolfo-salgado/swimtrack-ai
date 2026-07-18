from __future__ import annotations

import json
import math
import queue
import re
import subprocess
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from swimtrack_ai.config import Settings
from swimtrack_ai.errors import InvalidVideoError, NvdecDecodeError

_PTS_TIME_PATTERN = re.compile(
    r"\bpts_time:(?P<pts>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\b"
)
_SIZE_PATTERN = re.compile(r"\bs:(?P<width>\d+)x(?P<height>\d+)\b")


@dataclass(frozen=True, slots=True)
class VideoStreamInfo:
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class DecodedVideoFrame:
    """A sampled source frame and its presentation timestamp."""

    frame_index: int
    time_ms: float
    image: np.ndarray
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class _FrameTiming:
    time_seconds: float
    width: int
    height: int


class NvdecVideoDecoder:
    """Decode sampled video frames through FFmpeg's CUDA/NVDEC path only.

    ``hwaccel_output_format=cuda`` and ``hwdownload`` intentionally make a
    software decoder unusable. That is important here: silently decoding on the
    CPU would hide a GPU deployment fault and reintroduce the bottleneck this
    endpoint is meant to remove.
    """

    def __init__(self, settings: Settings, sample_fps: float) -> None:
        if not math.isfinite(sample_fps) or sample_fps <= 0:
            raise InvalidVideoError("sample_fps must be greater than zero")
        self.settings = settings
        self.sample_fps = sample_fps
        self._stream_info: VideoStreamInfo | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._timings: queue.Queue[_FrameTiming] = queue.Queue(
            maxsize=max(2, settings.video_decode_batch_frames * 2)
        )
        self._stderr_tail: deque[str] = deque(maxlen=30)
        self._stderr_done = threading.Event()
        self._stop_stderr = threading.Event()
        self._stderr_thread: threading.Thread | None = None
        self._next_frame_index = 0

    def _probe_command(self, video_path: Path) -> list[str]:
        return [
            self.settings.ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_type,width,height",
            "-of",
            "json",
            str(video_path),
        ]

    def _ffmpeg_command(self, video_path: Path) -> list[str]:
        sample_interval = 1.0 / self.sample_fps
        # Source timestamps are commonly represented as fractions such as 1/10.
        # A small epsilon prevents FFmpeg's floating-point expression evaluator
        # from intermittently dropping a frame exactly on the sampling boundary.
        minimum_interval = max(0.0, sample_interval - 0.000001)
        # ``select`` runs before ``hwdownload``. It therefore bases selection on
        # decoder presentation timestamps while only transferring sampled frames
        # from GPU memory to host memory. CUDA surfaces can only be downloaded
        # into a native software format such as NV12; a second format filter then
        # converts that CPU frame to OpenCV's BGR representation. The escaped
        # comma belongs to FFmpeg's expression parser, not the filter-chain
        # separator.
        filter_graph = (
            f"select=isnan(prev_selected_t)+gte(t-prev_selected_t\\,{minimum_interval:.12f}),"
            "hwdownload,format=nv12,format=bgr24,showinfo"
        )
        return [
            self.settings.ffmpeg_path,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "info",
            "-hwaccel",
            "cuda",
            "-hwaccel_device",
            "0",
            "-hwaccel_output_format",
            "cuda",
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-vf",
            filter_graph,
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]

    @staticmethod
    def _stderr_text(error: subprocess.CompletedProcess[str]) -> str:
        detail = (error.stderr or "").strip()
        return detail[-2_000:] if detail else "no diagnostic output"

    def _probe(self, video_path: Path) -> VideoStreamInfo:
        try:
            completed = subprocess.run(
                self._probe_command(video_path),
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.settings.video_probe_timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise NvdecDecodeError(f"FFprobe executable is unavailable: {self.settings.ffprobe_path!r}") from exc
        except subprocess.TimeoutExpired as exc:
            raise InvalidVideoError(
                f"Video inspection exceeded {self.settings.video_probe_timeout_seconds} seconds"
            ) from exc
        if completed.returncode != 0:
            raise InvalidVideoError(f"FFprobe could not inspect the uploaded video: {self._stderr_text(completed)}")
        try:
            payload = json.loads(completed.stdout)
            streams = payload["streams"]
            stream = next(item for item in streams if item.get("codec_type") == "video")
            width = int(stream["width"])
            height = int(stream["height"])
        except (KeyError, StopIteration, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise InvalidVideoError("Uploaded file does not contain a valid video stream") from exc
        if width <= 0 or height <= 0:
            raise InvalidVideoError("Uploaded video has invalid dimensions")
        if width * height > self.settings.max_decoded_pixels:
            raise InvalidVideoError(
                f"Decoded video frames are limited to {self.settings.max_decoded_pixels} pixels"
            )
        return VideoStreamInfo(width=width, height=height)

    def _collect_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            self._stderr_done.set()
            return
        try:
            for raw_line in iter(process.stderr.readline, b""):
                line = raw_line.decode("utf-8", errors="replace").strip()
                if line:
                    self._stderr_tail.append(line)
                if "showinfo" not in line:
                    continue
                pts_match = _PTS_TIME_PATTERN.search(line)
                size_match = _SIZE_PATTERN.search(line)
                if pts_match is None or size_match is None:
                    continue
                timing = _FrameTiming(
                    time_seconds=float(pts_match.group("pts")),
                    width=int(size_match.group("width")),
                    height=int(size_match.group("height")),
                )
                while not self._stop_stderr.is_set():
                    try:
                        self._timings.put(timing, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        finally:
            self._stderr_done.set()

    def open(self, video_path: Path) -> None:
        if self._process is not None:
            raise RuntimeError("NVDEC decoder is already open")
        self._stream_info = self._probe(video_path)
        try:
            self._process = subprocess.Popen(
                self._ffmpeg_command(video_path),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise NvdecDecodeError(f"FFmpeg executable is unavailable: {self.settings.ffmpeg_path!r}") from exc
        except OSError as exc:
            raise NvdecDecodeError(f"Could not start FFmpeg NVDEC decoder: {exc}") from exc
        self._stderr_thread = threading.Thread(
            target=self._collect_stderr,
            name="swimtrack-nvdec-stderr",
            daemon=True,
        )
        self._stderr_thread.start()

    def _failure_detail(self, detail: str) -> NvdecDecodeError:
        process = self._process
        return_code = process.poll() if process is not None else None
        diagnostics = "\n".join(self._stderr_tail)
        suffix = f"; ffmpeg exit code {return_code}" if return_code is not None else ""
        if diagnostics:
            suffix += f": {diagnostics[-2_000:]}"
        return NvdecDecodeError(f"NVDEC decode failed: {detail}{suffix}")

    def _read_exact_frame(self, byte_count: int) -> bytes | None:
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("NVDEC decoder is not open")
        chunks: list[bytes] = []
        remaining = byte_count
        while remaining:
            chunk = process.stdout.read(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        if not chunks:
            self._finish_stream()
            return None
        raw_frame = b"".join(chunks)
        if len(raw_frame) != byte_count:
            raise self._failure_detail("FFmpeg emitted a truncated raw frame")
        return raw_frame

    def _finish_stream(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            return_code = process.wait(timeout=self.settings.video_probe_timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            raise self._failure_detail("FFmpeg did not exit after the video stream ended") from exc
        self._stderr_done.wait(timeout=1)
        if return_code != 0:
            raise self._failure_detail("FFmpeg could not decode the uploaded video with CUDA/NVDEC")

    def _next_timing(self) -> _FrameTiming:
        while True:
            try:
                return self._timings.get(timeout=0.1)
            except queue.Empty:
                if self._stderr_done.is_set():
                    raise self._failure_detail("FFmpeg did not emit presentation timestamp metadata")

    def read_batch(self, max_frames: int) -> list[DecodedVideoFrame]:
        if max_frames <= 0:
            raise ValueError("max_frames must be greater than zero")
        stream_info = self._stream_info
        if stream_info is None:
            raise RuntimeError("NVDEC decoder is not open")
        frame_bytes = stream_info.width * stream_info.height * 3
        decoded: list[DecodedVideoFrame] = []
        while len(decoded) < max_frames:
            raw_frame = self._read_exact_frame(frame_bytes)
            if raw_frame is None:
                break
            timing = self._next_timing()
            if timing.width != stream_info.width or timing.height != stream_info.height:
                raise InvalidVideoError("Videos that change resolution mid-stream are not supported")
            if not math.isfinite(timing.time_seconds) or timing.time_seconds < 0:
                raise InvalidVideoError("Video presentation timestamps must be finite and non-negative")
            image = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                (stream_info.height, stream_info.width, 3)
            ).copy()
            decoded.append(
                DecodedVideoFrame(
                    frame_index=self._next_frame_index,
                    time_ms=timing.time_seconds * 1_000.0,
                    image=image,
                    width=stream_info.width,
                    height=stream_info.height,
                )
            )
            self._next_frame_index += 1
        return decoded

    def close(self) -> None:
        self._stop_stderr.set()
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1)
        self._process = None
