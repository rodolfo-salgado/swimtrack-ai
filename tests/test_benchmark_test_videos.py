from __future__ import annotations

import json
import socket
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from benchmark_test_videos import (  # noqa: E402
    BenchmarkError,
    VideoSpec,
    _multipart,
    aggregate_videos,
    inspect_video,
    parse_args,
    run_remote,
    select_video_ids,
    summarize_frames,
)


def _frame(
    frame_index: int,
    time_ms: float,
    *,
    boxes: list[dict] | None = None,
    candidates: int = 2,
    accepted: int = 1,
    weak_candidates: int = 0,
    after_roi: int = 1,
    weak_after_roi: int = 0,
    active_ids: list[int] | None = None,
    weak_reactivated_ids: list[int] | None = None,
    lost: int = 0,
) -> dict:
    return {
        "frame_index": frame_index,
        "time_ms": time_ms,
        "width": 1080,
        "height": 1080,
        "boxes": boxes or [],
        "lap_scores": [{"lap_score": 0.2 + frame_index * 0.1, "evaluable": frame_index != 0}],
        "tracking_diagnostics": {
            "diagnostic_floor": 0.05,
            "person_candidates": {"count": candidates},
            "detector_accepted": {"count": accepted},
            "weak_candidates": {"count": weak_candidates},
            "lanes": [
                {
                    "lane_id": "center",
                    "after_roi": {"count": after_roi},
                    "weak_candidates_after_roi": {"count": weak_after_roi},
                    "active_track_ids": active_ids or [],
                    "retained_lost_track_count": lost,
                    "weak_reactivated_track_ids": weak_reactivated_ids or [],
                }
            ],
        },
    }


def test_primary_selection_defaults_to_test01_through_test08() -> None:
    assert select_video_ids(None) == [f"test{number:02d}" for number in range(1, 9)]
    with pytest.raises(BenchmarkError, match="excludes test09"):
        select_video_ids(["test09"])


def test_inspect_video_uses_front_sampling_stride_and_needs_no_gpu(tmp_path: Path) -> None:
    path = tmp_path / "fixture.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (12, 8))
    assert writer.isOpened()
    for index in range(5):
        writer.write(np.full((8, 12, 3), index * 20, dtype=np.uint8))
    writer.release()

    spec = inspect_video(path, "test01", 3.0)

    assert spec.width == 12
    assert spec.height == 8
    assert spec.source_frames == 5
    assert spec.sample_stride == 4
    assert spec.expected_sampled_frames == 2
    assert spec.tracking_fps == pytest.approx(2.5)
    assert len(spec.sha256) == 64


def test_summary_preserves_the_detector_to_tracker_funnel_and_internal_gap() -> None:
    first_box = {
        "id": 7,
        "lane_id": "center",
        "x1": 100.0,
        "y1": 100.0,
        "x2": 150.0,
        "y2": 180.0,
        "conf": 0.9,
    }
    last_box = {**first_box, "x1": 120.0, "x2": 170.0}
    summary = summarize_frames(
        [
            _frame(0, 0.0, boxes=[first_box], active_ids=[7]),
            _frame(1, 100.0, boxes=[], active_ids=[], lost=1),
            _frame(
                2,
                200.0,
                boxes=[last_box],
                active_ids=[7],
                weak_candidates=1,
                weak_after_roi=1,
                weak_reactivated_ids=[7],
            ),
        ]
    )

    stages = summary["diagnostics"]["stages"]
    assert stages["person_candidates"]["observations"] == 6
    assert stages["detector_accepted"]["observations"] == 3
    assert stages["weak_candidates"]["observations"] == 1
    assert stages["after_roi"]["observations"] == 3
    assert stages["weak_candidates_after_roi"]["observations"] == 1
    assert summary["diagnostics"]["funnel"] == {
        "candidate_to_accepted": 0.5,
        "accepted_to_roi": 1.0,
        "candidate_to_weak": pytest.approx(1 / 6),
        "weak_to_roi": 1.0,
    }
    assert summary["diagnostics"]["accepted_no_track_frames"] == 1
    assert summary["tracking"]["unique_track_ids"] == 1
    assert summary["tracking"]["same_id_reacquisitions"] == 1
    assert summary["tracking"]["weak_reactivations"] == {
        "events": 1,
        "frames_nonempty": 1,
        "unique_track_ids": 1,
        "track_ids_by_lane": {"center": [7]},
    }
    assert summary["tracking"]["internal_active_gaps"]["frames"]["max"] == 1
    assert summary["lap_scores"]["maximum"] == pytest.approx(0.4)


def test_summary_allows_no_diagnostics_without_misrepresenting_stage_metrics() -> None:
    summary = summarize_frames(
        [
            {
                "frame_index": 0,
                "time_ms": 0.0,
                "width": 16,
                "height": 16,
                "boxes": [],
            }
        ]
    )

    assert summary["diagnostics"] == {
        "available": False,
        "frames": 0,
        "frame_coverage": 0.0,
        "diagnostic_floors": [],
        "stages": None,
        "funnel": None,
        "accepted_no_track_frames": None,
        "detector_accepted_no_track_frames": None,
        "retained_lost": None,
    }


def test_multipart_keeps_repeated_frame_fields_and_deterministic_boundary() -> None:
    content_type, body = _multipart(
        [("metadata", "{\"sequence\":0}")],
        [("frames", "frame.jpg", b"jpeg", "image/jpeg")],
        boundary="benchmark-boundary",
    )

    assert content_type == "multipart/form-data; boundary=benchmark-boundary"
    assert body.count(b'name="frames"') == 1
    assert b'filename="frame.jpg"' in body
    assert body.endswith(b"--benchmark-boundary--\r\n")


def test_aggregate_pools_stage_counts_only_when_every_video_has_diagnostics() -> None:
    analysis = summarize_frames([_frame(0, 0.0, boxes=[], active_ids=[])])
    result = {
        "video": {"id": "test01"},
        "execution": {"client_wall_ms": 100.0, "response_frames": 1},
        "analysis": analysis,
    }
    aggregate = aggregate_videos([result])

    assert aggregate["response_fps"] == pytest.approx(10.0)
    assert aggregate["diagnostics"]["stages"]["person_candidates"]["observations"] == 2
    assert aggregate["diagnostics"]["stages"]["weak_candidates"]["observations"] == 0
    assert aggregate["tracking"]["fragmentations_sum"] == 0
    assert aggregate["tracking"]["weak_reactivation_events_sum"] == 0


def test_aggregate_refuses_to_fabricate_pooled_diagnostics_when_a_video_lacks_them() -> None:
    diagnostic = {
        "video": {"id": "test01"},
        "execution": {"client_wall_ms": 100.0, "response_frames": 1},
        "analysis": summarize_frames([_frame(0, 0.0, boxes=[], active_ids=[])]),
    }
    no_diagnostics = {
        "video": {"id": "test02"},
        "execution": {"client_wall_ms": 100.0, "response_frames": 1},
        "analysis": summarize_frames(
            [{"frame_index": 0, "time_ms": 0.0, "width": 16, "height": 16, "boxes": []}]
        ),
    }

    aggregate = aggregate_videos([diagnostic, no_diagnostics])

    assert aggregate["diagnostics"] == {"available_for_all_videos": False}


def test_video_spec_is_plain_data_for_diagnostic_mode() -> None:
    spec = VideoSpec(
        video_id="test01",
        path=Path("/tmp/test01.mp4"),
        sha256="a" * 64,
        bytes=1,
        width=1,
        height=1,
        source_fps=60.0,
        source_frames=1,
        duration_ms=1.0,
        sample_stride=2,
        tracking_fps=30.0,
        expected_sampled_frames=1,
    )

    assert spec.video_id == "test01"


def _service_frame(frame_index: int, time_ms: float) -> dict[str, Any]:
    """Return the complete public diagnostics contract required by benchmark aggregation."""

    return {
        "frame_index": frame_index,
        "time_ms": time_ms,
        "width": 16,
        "height": 12,
        "boxes": [
            {
                "id": 41,
                "lane_id": "center",
                "x1": 2.0,
                "y1": 3.0,
                "x2": 10.0,
                "y2": 11.0,
                "conf": 0.9,
            }
        ],
        "lap_scores": [{"lap_score": 0.3, "evaluable": True}],
        "tracking_diagnostics": {
            "diagnostic_floor": 0.05,
            "person_candidates": {"count": 2},
            "detector_accepted": {"count": 1},
            "weak_candidates": {"count": 1},
            "lanes": [
                {
                    "lane_id": "center",
                    "after_roi": {"count": 1},
                    "weak_candidates_after_roi": {"count": 1},
                    "active_track_ids": [41],
                    "retained_lost_track_count": 0,
                    "weak_reactivated_track_ids": [41],
                }
            ],
        },
    }


@contextmanager
def _local_benchmark_service() -> Iterator[tuple[str, dict[str, Any]]]:
    """Serve the benchmark's real HTTP contract over a loopback TCP socket."""

    state: dict[str, Any] = {
        "ready_requests": [],
        "created_sessions": [],
        "batch_requests": [],
        "video_requests": [],
        "deleted_session_ids": [],
    }
    app = FastAPI()

    @app.get("/readyz")
    async def ready(request: Request) -> JSONResponse:
        state["ready_requests"].append(dict(request.headers))
        return JSONResponse({"status": "ready", "device": "cpu-test"})

    @app.post("/v1/tracking-sessions")
    async def create_tracking_session(request: Request) -> JSONResponse:
        payload = await request.json()
        state["created_sessions"].append({"headers": dict(request.headers), "payload": payload})
        session_id = f"local-{len(state['created_sessions'])}"
        return JSONResponse(
            {
                "session_id": session_id,
                "next_sequence": 0,
                "expires_in_seconds": 60,
                "tracking_configuration": {
                    "track_threshold": 0.45,
                    "weak_reactivation_enabled": True,
                    "weak_reactivation_score_threshold": 0.1,
                    "weak_reactivation_min_box_area": 64.0,
                    "weak_reactivation_max_gap_frames": 30,
                    "weak_reactivation_max_gap_seconds": 1.0,
                    "weak_reactivation_max_center_distance": 0.1,
                },
            },
            status_code=201,
        )

    @app.delete("/v1/tracking-sessions/{session_id}")
    async def delete_tracking_session(session_id: str, request: Request) -> Response:
        state["deleted_session_ids"].append({"session_id": session_id, "headers": dict(request.headers)})
        return Response(status_code=204)

    @app.post("/v1/tracking-sessions/{session_id}/batches")
    async def submit_batch(session_id: str, request: Request) -> JSONResponse:
        form = await request.form()
        metadata_value = form.get("metadata")
        assert isinstance(metadata_value, str)
        metadata = json.loads(metadata_value)
        uploads = form.getlist("frames")
        uploaded_frames = []
        for upload in uploads:
            uploaded_frames.append(
                {
                    "filename": upload.filename,
                    "content_type": upload.content_type,
                    "payload": await upload.read(),
                }
            )
        state["batch_requests"].append(
            {
                "session_id": session_id,
                "headers": dict(request.headers),
                "metadata": metadata,
                "frames": uploaded_frames,
            }
        )
        return JSONResponse(
            {
                "sequence": metadata["sequence"],
                "next_sequence": metadata["sequence"] + 1,
                "frames": [
                    _service_frame(frame["frame_index"], frame["time_ms"])
                    for frame in metadata["frames"]
                ],
            }
        )

    @app.post("/v1/tracking-sessions/{session_id}/video")
    async def submit_video(session_id: str, request: Request) -> StreamingResponse:
        form = await request.form()
        upload = form.get("video")
        assert upload is not None
        state["video_requests"].append(
            {
                "session_id": session_id,
                "headers": dict(request.headers),
                "sample_fps": form.get("sample_fps"),
                "filename": upload.filename,
                "content_type": upload.content_type,
                "payload": await upload.read(),
            }
        )
        frames = (_service_frame(index, index * 100.0) for index in range(3))
        return StreamingResponse(
            (json.dumps(frame, separators=(",", ":")) + "\n" for frame in frames),
            media_type="application/x-ndjson",
            headers={"X-Swimtrack-Decode-Path": "nvdec", "X-Swimtrack-Decode-Backend": "test"},
        )

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    host, port = listener.getsockname()
    server = uvicorn.Server(
        uvicorn.Config(app, host=host, port=port, lifespan="off", log_level="critical", access_log=False)
    )
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5.0
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=5.0)
        listener.close()
        raise RuntimeError("local FastAPI test server did not start")
    try:
        yield f"http://{host}:{port}", state
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)
        listener.close()
        assert not thread.is_alive(), "local FastAPI test server did not stop"


def _benchmark_video_spec(tmp_path: Path) -> VideoSpec:
    path = tmp_path / "benchmark-source.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (16, 12))
    assert writer.isOpened()
    for index in range(3):
        writer.write(np.full((12, 16, 3), 25 + index * 25, dtype=np.uint8))
    writer.release()
    return inspect_video(path, "test01", 10.0)


def _remote_args(base_url: str, results_root: Path, transport: str) -> object:
    return parse_args(
        [
            "--mode",
            "remote",
            "--transport",
            transport,
            "--base-url",
            base_url,
            "--auth-token",
            "benchmark-token",
            "--results-root",
            str(results_root),
            "--max-fps",
            "10",
            "--diagnostics",
            "boxes",
            "--batch-size",
            "2",
            "--inference-size",
            "16",
            "--jpeg-quality",
            "90",
        ]
    )


def _assert_run_has_new_diagnostics(result_dir: Path) -> None:
    run = json.loads((result_dir / "run.json").read_text(encoding="utf-8"))
    result = json.loads((result_dir / "test01" / "result.json").read_text(encoding="utf-8"))

    assert run["status"] == "completed"
    assert result["session"]["tracking_configuration"]["weak_reactivation_enabled"] is True
    assert result["execution"]["response_frame_count_matches_expected"] is True
    assert result["analysis"]["diagnostics"]["stages"]["weak_candidates"]["observations"] == 3
    assert result["analysis"]["diagnostics"]["stages"]["weak_candidates_after_roi"]["observations"] == 3
    assert result["analysis"]["tracking"]["weak_reactivations"]["events"] == 3
    assert not (result_dir / "test01" / "session-close-warning.txt").exists()


def test_remote_frames_transport_uses_manual_multipart_and_closes_the_session(tmp_path: Path) -> None:
    spec = _benchmark_video_spec(tmp_path)
    result_dir = tmp_path / "frame-results"
    result_dir.mkdir()

    with _local_benchmark_service() as (base_url, state):
        run_remote(_remote_args(base_url, tmp_path, "frames"), [spec], result_dir)

    assert len(state["ready_requests"]) == 1
    assert state["created_sessions"][0]["headers"]["x-swimtrack-auth"] == "benchmark-token"
    assert state["created_sessions"][0]["headers"]["content-type"] == "application/json"
    assert state["created_sessions"][0]["payload"] == {
        "fps": pytest.approx(10.0),
        "diagnostics": "boxes",
        "lap_calibration_id": "fixed-camera-v1",
    }
    assert [request["metadata"]["sequence"] for request in state["batch_requests"]] == [0, 1]
    assert [len(request["frames"]) for request in state["batch_requests"]] == [2, 1]
    assert [
        [frame["filename"] for frame in request["frames"]]
        for request in state["batch_requests"]
    ] == [["frame-00000000.jpg", "frame-00000001.jpg"], ["frame-00000002.jpg"]]
    assert all(request["headers"]["x-swimtrack-auth"] == "benchmark-token" for request in state["batch_requests"])
    assert all(request["headers"]["accept"] == "application/json" for request in state["batch_requests"])
    assert all(
        request["headers"]["content-type"].startswith("multipart/form-data; boundary=")
        for request in state["batch_requests"]
    )
    assert all(
        frame["content_type"] == "image/jpeg" and frame["payload"]
        for request in state["batch_requests"]
        for frame in request["frames"]
    )
    assert [entry["session_id"] for entry in state["deleted_session_ids"]] == ["local-1"]
    assert state["deleted_session_ids"][0]["headers"]["x-swimtrack-auth"] == "benchmark-token"
    _assert_run_has_new_diagnostics(result_dir)


def test_remote_video_transport_streams_ndjson_and_closes_the_session(tmp_path: Path) -> None:
    spec = _benchmark_video_spec(tmp_path)
    result_dir = tmp_path / "video-results"
    result_dir.mkdir()

    with _local_benchmark_service() as (base_url, state):
        run_remote(_remote_args(base_url, tmp_path, "video"), [spec], result_dir)

    assert len(state["ready_requests"]) == 1
    assert state["created_sessions"][0]["headers"]["x-swimtrack-auth"] == "benchmark-token"
    assert len(state["video_requests"]) == 1
    upload = state["video_requests"][0]
    assert upload["session_id"] == "local-1"
    assert upload["headers"]["x-swimtrack-auth"] == "benchmark-token"
    assert upload["headers"]["accept"] == "application/x-ndjson"
    assert upload["headers"]["content-type"].startswith("multipart/form-data; boundary=")
    assert upload["sample_fps"] == "10"
    assert upload["filename"] == "upload.avi"
    assert upload["content_type"].startswith("video/")
    assert upload["payload"] == spec.path.read_bytes()
    assert state["deleted_session_ids"][0]["session_id"] == "local-1"
    assert state["deleted_session_ids"][0]["headers"]["x-swimtrack-auth"] == "benchmark-token"
    _assert_run_has_new_diagnostics(result_dir)
