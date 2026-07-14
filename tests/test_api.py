from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from swimtrack_ai.api import create_app
from swimtrack_ai.config import Settings
from swimtrack_ai.detectors import DetectorResult
from swimtrack_ai.tracker import TrackerUpdate


class StubTracker:
    def __init__(self) -> None:
        self.calls = 0
        self.detections: list[np.ndarray] = []

    def update(self, detections: np.ndarray, image_size: tuple[int, int]) -> TrackerUpdate:
        del image_size
        self.calls += 1
        self.detections.append(detections.copy())
        return TrackerUpdate(
            active_tracks=[
                SimpleNamespace(track_id=index + 1, tlbr=detection[:4], score=float(detection[4]))
                for index, detection in enumerate(detections)
            ]
        )


class StubTrackerFactory:
    def __init__(self, _settings: Settings) -> None:
        self.trackers: list[StubTracker] = []

    def __call__(self, fps: float) -> StubTracker:
        assert fps > 0
        tracker = StubTracker()
        self.trackers.append(tracker)
        return tracker


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        backend="fake",
        auth_token="test-secret",
        model_source_dir=tmp_path,
        model_cache_dir=tmp_path,
        bytetrack_root=tmp_path,
        max_batch_frames=3,
        session_ttl_seconds=60,
    )


@pytest.fixture
def client(settings: Settings):
    app = create_app(settings=settings, tracker_factory_builder=StubTrackerFactory)
    with TestClient(app) as test_client:
        yield test_client


def encoded_frame(offset: int = 0) -> bytes:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.rectangle(frame, (8 + offset, 10), (28 + offset, 40), (255, 255, 255), -1)
    success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert success
    return encoded.tobytes()


def metadata(batch_id: str, sequence: int, frame_index: int = 0) -> str:
    return json.dumps(
        {
            "batch_id": batch_id,
            "sequence": sequence,
            "frames": [
                {
                    "frame_index": frame_index,
                    "time_ms": frame_index * 16.667,
                    "original_width": 128,
                    "original_height": 96,
                }
            ],
        }
    )


def create_session(client: TestClient, diagnostics: str = "none") -> str:
    payload = {"fps": 60}
    if diagnostics != "none":
        payload["diagnostics"] = diagnostics
    response = client.post(
        "/v1/tracking-sessions",
        headers={"X-Swimtrack-Auth": "test-secret"},
        json=payload,
    )
    assert response.status_code == 201
    return response.json()["session_id"]


def submit(client: TestClient, session_id: str, metadata_json: str, image: bytes | None = None):
    return client.post(
        f"/v1/tracking-sessions/{session_id}/batches",
        headers={"X-Swimtrack-Auth": "test-secret"},
        data={"metadata": metadata_json},
        files=[("frames", ("frame.jpg", image or encoded_frame(), "image/jpeg"))],
    )


def test_health_readiness_and_authentication(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok", "backend": "fake", "detail": None}
    assert client.get("/readyz").status_code == 200
    assert client.post("/v1/tracking-sessions", json={"fps": 60}).status_code == 401


def test_batch_request_content_length_is_rejected_before_parsing(settings: Settings) -> None:
    app = create_app(settings=settings, tracker_factory_builder=StubTrackerFactory)
    with TestClient(app) as client:
        response = client.post(
            "/v1/tracking-sessions/not-used/batches",
            headers={
                "X-Swimtrack-Auth": "test-secret",
                "Content-Length": str(settings.max_request_bytes + 1),
            },
            content=b"not-a-multipart-body",
        )
    assert response.status_code == 413


def test_session_batch_coordinates_and_delete(client: TestClient) -> None:
    session_id = create_session(client)
    response = submit(client, session_id, metadata("batch-0", 0))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["next_sequence"] == 1
    assert body["frames"][0]["width"] == 128
    assert body["frames"][0]["height"] == 96
    assert body["frames"][0]["boxes"][0]["id"] == 1
    assert body["frames"][0]["boxes"][0]["x1"] >= 14
    assert "lap_scores" not in body["frames"][0]
    deleted = client.delete(
        f"/v1/tracking-sessions/{session_id}",
        headers={"X-Swimtrack-Auth": "test-secret"},
    )
    assert deleted.status_code == 204
    assert submit(client, session_id, metadata("batch-1", 1, 1)).status_code == 404


def test_fixed_camera_lap_scoring_is_opt_in(client: TestClient) -> None:
    created = client.post(
        "/v1/tracking-sessions",
        headers={"X-Swimtrack-Auth": "test-secret"},
        json={"fps": 60, "lap_calibration_id": "fixed-camera-v1"},
    )
    assert created.status_code == 201

    response = submit(client, created.json()["session_id"], metadata("lap-batch", 0))

    assert response.status_code == 200, response.text
    score = response.json()["frames"][0]["lap_scores"][0]
    assert score["lane_id"] == "center"
    assert score["lap_score"] == 0.0
    assert score["evaluable"] is False
    assert score["score_version"] == "trajectory-v1"


def test_tracking_diagnostics_are_opt_in(client: TestClient) -> None:
    default_session = create_session(client)
    default_response = submit(client, default_session, metadata("default-diagnostics", 0))
    assert "tracking_diagnostics" not in default_response.json()["frames"][0]

    counts_session = create_session(client, diagnostics="counts")
    counts_response = submit(client, counts_session, metadata("count-diagnostics", 0))
    frame = counts_response.json()["frames"][0]
    diagnostics = frame["tracking_diagnostics"]
    assert diagnostics["person_candidates"] == {"count": 1}
    assert diagnostics["detector_accepted"] == {"count": 1}
    assert diagnostics["lanes"] == [
        {
            "lane_id": "global",
            "after_roi": {"count": 1},
            "active_track_ids": [1],
            "retained_lost_track_count": 0,
        }
    ]


def test_box_diagnostics_include_candidates_and_effective_configuration(client: TestClient) -> None:
    created = client.post(
        "/v1/tracking-sessions",
        headers={"X-Swimtrack-Auth": "test-secret"},
        json={"fps": 60, "diagnostics": "boxes"},
    )
    assert created.status_code == 201
    configuration = created.json()["tracking_configuration"]
    assert configuration["diagnostic_score_floor"] == 0.05
    assert configuration["effective_lost_buffer_frames"] == 120
    assert configuration["effective_lost_buffer_seconds"] == 2.0

    response = submit(client, created.json()["session_id"], metadata("box-diagnostics", 0))
    diagnostics = response.json()["frames"][0]["tracking_diagnostics"]
    assert diagnostics["person_candidates"]["count"] == 1
    assert diagnostics["person_candidates"]["boxes"][0]["conf"] == pytest.approx(0.99)
    assert diagnostics["lanes"][0]["after_roi"]["boxes"][0]["x1"] >= 14


def test_unknown_tracking_diagnostics_level_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/tracking-sessions",
        headers={"X-Swimtrack-Auth": "test-secret"},
        json={"fps": 60, "diagnostics": "verbose"},
    )

    assert response.status_code == 422


def test_unknown_lap_calibration_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/tracking-sessions",
        headers={"X-Swimtrack-Auth": "test-secret"},
        json={"fps": 60, "lap_calibration_id": "unknown-camera"},
    )

    assert response.status_code == 422


def test_fixed_camera_roi_routes_detections_before_its_lane_tracker(settings: Settings) -> None:
    detections = np.asarray(
        [
            [54.0, 48.0, 74.0, 72.0, 0.90],
            [0.0, 0.0, 10.0, 10.0, 0.95],
        ],
        dtype=np.float32,
    )

    class StagedDetector:
        def infer(self, _frame: np.ndarray, _target_size: tuple[int, int]) -> DetectorResult:
            return DetectorResult(person_candidates=detections.copy(), accepted=detections.copy())

        def close(self) -> None:
            return None

    tracker_factory = StubTrackerFactory(settings)
    app = create_app(
        settings=settings,
        detector_factory=lambda _settings: StagedDetector(),
        tracker_factory_builder=lambda _settings: tracker_factory,
    )
    with TestClient(app) as test_client:
        created = test_client.post(
            "/v1/tracking-sessions",
            headers={"X-Swimtrack-Auth": "test-secret"},
            json={
                "fps": 60,
                "lap_calibration_id": "fixed-camera-v1",
                "diagnostics": "counts",
            },
        )
        response = submit(test_client, created.json()["session_id"], metadata("roi-batch", 0))

    assert created.status_code == 201
    assert created.json()["tracking_configuration"]["lane_ids"] == ["center"]
    assert len(tracker_factory.trackers) == 1
    assert tracker_factory.trackers[0].detections[0].tolist() == [detections[0].tolist()]
    frame = response.json()["frames"][0]
    assert frame["boxes"][0]["lane_id"] == "center"
    assert frame["tracking_diagnostics"]["detector_accepted"]["count"] == 2
    assert frame["tracking_diagnostics"]["lanes"][0]["after_roi"]["count"] == 1


def test_identical_retry_is_idempotent(client: TestClient) -> None:
    session_id = create_session(client)
    metadata_json = metadata("same-batch", 0)
    first = submit(client, session_id, metadata_json)
    retry = submit(client, session_id, metadata_json)
    assert retry.status_code == 200
    assert retry.json() == first.json()
    next_batch = submit(client, session_id, metadata("next-batch", 1, 1), encoded_frame(1))
    assert next_batch.status_code == 200


def test_multiple_frames_are_returned_in_order(client: TestClient) -> None:
    session_id = create_session(client)
    metadata_json = json.dumps(
        {
            "batch_id": "multi-frame",
            "sequence": 0,
            "frames": [
                {
                    "frame_index": index,
                    "time_ms": index * 16.667,
                    "original_width": 128,
                    "original_height": 96,
                }
                for index in range(3)
            ],
        }
    )
    response = client.post(
        f"/v1/tracking-sessions/{session_id}/batches",
        headers={"X-Swimtrack-Auth": "test-secret"},
        data={"metadata": metadata_json},
        files=[("frames", (f"frame-{index}.jpg", encoded_frame(index), "image/jpeg")) for index in range(3)],
    )
    assert response.status_code == 200, response.text
    assert [frame["frame_index"] for frame in response.json()["frames"]] == [0, 1, 2]


def test_reused_batch_id_and_wrong_sequence_conflict(client: TestClient) -> None:
    session_id = create_session(client)
    assert submit(client, session_id, metadata("batch-0", 0)).status_code == 200
    reused = submit(client, session_id, metadata("batch-0", 0), encoded_frame(2))
    assert reused.status_code == 409
    wrong_sequence = submit(client, session_id, metadata("batch-2", 3, 3))
    assert wrong_sequence.status_code == 409


def test_metadata_count_and_image_validation(client: TestClient) -> None:
    session_id = create_session(client)
    missing_metadata_item = client.post(
        f"/v1/tracking-sessions/{session_id}/batches",
        headers={"X-Swimtrack-Auth": "test-secret"},
        data={"metadata": metadata("batch-0", 0)},
        files=[
            ("frames", ("one.jpg", encoded_frame(), "image/jpeg")),
            ("frames", ("two.jpg", encoded_frame(), "image/jpeg")),
        ],
    )
    assert missing_metadata_item.status_code == 422
    invalid_image = submit(client, session_id, metadata("bad-image", 0), b"not-an-image")
    assert invalid_image.status_code == 422


def test_startup_fails_fast_when_detector_cannot_load(settings: Settings) -> None:
    def broken_detector(_settings: Settings):
        raise RuntimeError("model unavailable")

    app = create_app(
        settings=settings,
        detector_factory=broken_detector,
        tracker_factory_builder=StubTrackerFactory,
    )
    with pytest.raises(RuntimeError, match="model unavailable"), TestClient(app):
        pass
