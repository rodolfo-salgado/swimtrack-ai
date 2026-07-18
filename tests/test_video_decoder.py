from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

import swimtrack_ai.video_decoder as video_decoder_module
from swimtrack_ai.config import Settings
from swimtrack_ai.errors import NvdecDecodeError
from swimtrack_ai.video_decoder import NvdecVideoDecoder


class FakeProcess:
    def __init__(self, stdout: bytes, stderr: bytes, returncode: int = 0) -> None:
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self.returncode = returncode
        self.terminated = False
        self.killed = False

    def poll(self) -> int:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


def settings(tmp_path: Path) -> Settings:
    return Settings(
        backend="fake",
        auth_token="test-secret",
        model_source_dir=tmp_path,
        model_cache_dir=tmp_path,
        bytetrack_root=tmp_path,
        max_decoded_pixels=64,
        video_decode_batch_frames=2,
    )


def install_probe(monkeypatch: pytest.MonkeyPatch, width: int = 4, height: int = 2) -> None:
    def fake_run(*_args, **_kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args="ffprobe",
            returncode=0,
            stdout=json.dumps({"streams": [{"codec_type": "video", "width": width, "height": height}]}),
            stderr="",
        )

    monkeypatch.setattr(video_decoder_module.subprocess, "run", fake_run)


def test_nvdec_decoder_uses_cuda_and_preserves_presentation_timestamps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_probe(monkeypatch)
    first = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    second = np.full((2, 4, 3), 17, dtype=np.uint8)
    process = FakeProcess(
        first.tobytes() + second.tobytes(),
        (
            b"[Parsed_showinfo_3] n:   0 pts:      5 pts_time:0.041667 pos:0 fmt:bgr24 s:4x2\n"
            b"[Parsed_showinfo_3] n:   1 pts:     17 pts_time:0.141667 pos:24 fmt:bgr24 s:4x2\n"
        ),
    )
    commands: list[list[str]] = []

    def fake_popen(command: list[str], **_kwargs) -> FakeProcess:
        commands.append(command)
        return process

    monkeypatch.setattr(video_decoder_module.subprocess, "Popen", fake_popen)
    decoder = NvdecVideoDecoder(settings(tmp_path), sample_fps=30.0)

    decoder.open(tmp_path / "clip.mp4")
    frames = decoder.read_batch(8)
    decoder.close()

    assert [frame.frame_index for frame in frames] == [0, 1]
    assert [frame.time_ms for frame in frames] == pytest.approx([41.667, 141.667])
    assert [frame.width for frame in frames] == [4, 4]
    assert [frame.height for frame in frames] == [2, 2]
    assert np.array_equal(frames[0].image, first)
    assert np.array_equal(frames[1].image, second)
    assert commands[0][commands[0].index("-hwaccel") + 1] == "cuda"
    assert commands[0][commands[0].index("-hwaccel_device") + 1] == "0"
    assert commands[0][commands[0].index("-hwaccel_output_format") + 1] == "cuda"
    filter_graph = commands[0][commands[0].index("-vf") + 1]
    assert "select=isnan(prev_selected_t)+gte(t-prev_selected_t\\,0.033332333333)" in filter_graph
    assert filter_graph.endswith("hwdownload,format=nv12,format=bgr24,showinfo")


def test_nvdec_decoder_reports_a_clear_error_when_ffmpeg_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_probe(monkeypatch)
    process = FakeProcess(b"", b"Device setup failed: CUDA is unavailable\n", returncode=1)
    monkeypatch.setattr(video_decoder_module.subprocess, "Popen", lambda *_args, **_kwargs: process)
    decoder = NvdecVideoDecoder(settings(tmp_path), sample_fps=30.0)

    decoder.open(tmp_path / "clip.mp4")
    with pytest.raises(NvdecDecodeError, match="NVDEC decode failed") as error:
        decoder.read_batch(1)
    decoder.close()

    assert "CUDA is unavailable" in error.value.detail
