from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated

import anyio
import cv2
import numpy as np
from fastapi import Body, Depends, FastAPI, File, Form, Request, Response, Security, UploadFile
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import ValidationError

from swimtrack_ai.config import Settings
from swimtrack_ai.detectors import Detector, create_detector
from swimtrack_ai.errors import InvalidFrameError, NotReadyError, PayloadTooLargeError, ServiceError
from swimtrack_ai.schemas import (
    BatchMetadata,
    BatchResult,
    CreateSessionRequest,
    HealthResponse,
    SessionCreated,
)
from swimtrack_ai.service import TrackingService
from swimtrack_ai.tracker import ByteTrackFactory, Tracker

logger = logging.getLogger(__name__)
auth_header = APIKeyHeader(name="X-Swimtrack-Auth", auto_error=False)
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "application/octet-stream"}

DetectorFactory = Callable[[Settings], Detector]
TrackerFactoryBuilder = Callable[[Settings], Callable[[float], Tracker]]


def _decode_image(payload: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)


def create_app(
    settings: Settings | None = None,
    detector_factory: DetectorFactory = create_detector,
    tracker_factory_builder: TrackerFactoryBuilder = ByteTrackFactory,
) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = settings
        app.state.service = None
        cleanup_task = None
        detector = None
        try:
            if not settings.auth_token:
                raise ValueError("SWIMTRACK_AUTH_TOKEN must not be empty")
            detector = await anyio.to_thread.run_sync(detector_factory, settings)
            tracker_factory = await anyio.to_thread.run_sync(tracker_factory_builder, settings)
            app.state.service = TrackingService(settings, detector, tracker_factory)

            async def clean_expired_sessions() -> None:
                while True:
                    await asyncio.sleep(settings.cleanup_interval_seconds)
                    await anyio.to_thread.run_sync(app.state.service.expire_sessions)

            cleanup_task = asyncio.create_task(clean_expired_sessions())
        except Exception:
            logger.exception("Inference service startup failed")
            if detector is not None:
                await anyio.to_thread.run_sync(detector.close)
            raise
        yield
        if cleanup_task is not None:
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass
        if app.state.service is not None:
            await anyio.to_thread.run_sync(app.state.service.close)

    app = FastAPI(
        title="SwimTrack AI",
        version="0.1.0",
        description="Stateful RT-DETRv2 and ByteTrack inference service",
        lifespan=lifespan,
    )

    @app.exception_handler(ServiceError)
    async def service_error_handler(_request: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "detail": exc.detail}},
        )

    @app.middleware("http")
    async def reject_unauthorized_or_oversized_requests(request: Request, call_next):
        if request.url.path.startswith("/v1/"):
            supplied_token = request.headers.get("X-Swimtrack-Auth")
            if supplied_token is None or not hmac.compare_digest(supplied_token, settings.auth_token):
                return JSONResponse(status_code=401, content={"detail": "Invalid or missing X-Swimtrack-Auth header"})
            if request.method == "POST" and request.url.path.endswith("/batches"):
                content_length = request.headers.get("content-length")
                if content_length is not None:
                    try:
                        request_bytes = int(content_length)
                    except ValueError:
                        return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header"})
                    if request_bytes > settings.max_request_bytes:
                        return JSONResponse(
                            status_code=413,
                            content={
                                "error": {
                                    "code": "payload_too_large",
                                    "detail": f"Request is limited to {settings.max_request_bytes} bytes",
                                }
                            },
                        )
        return await call_next(request)

    def authenticated(
        request: Request,
        supplied_token: Annotated[str | None, Security(auth_header)],
    ) -> None:
        expected_token = request.app.state.settings.auth_token
        if not expected_token or supplied_token is None or not hmac.compare_digest(supplied_token, expected_token):
            from fastapi import HTTPException

            raise HTTPException(status_code=401, detail="Invalid or missing X-Swimtrack-Auth header")

    def ready_service(request: Request) -> TrackingService:
        service = request.app.state.service
        if service is None:
            raise NotReadyError("Detector is not initialized")
        return service

    @app.get("/healthz", response_model=HealthResponse, tags=["operations"])
    async def health(request: Request) -> HealthResponse:
        return HealthResponse(status="ok", backend=request.app.state.settings.backend)

    @app.get(
        "/readyz",
        response_model=HealthResponse,
        responses={503: {"description": "Model or tracker failed to initialize"}},
        tags=["operations"],
    )
    async def readiness(request: Request) -> HealthResponse | JSONResponse:
        if request.app.state.service is None:
            return JSONResponse(
                status_code=503,
                content=HealthResponse(
                    status="not_ready",
                    backend=request.app.state.settings.backend,
                    detail="Detector is not initialized",
                ).model_dump(),
            )
        return HealthResponse(status="ready", backend=request.app.state.settings.backend)

    @app.post(
        "/v1/tracking-sessions",
        response_model=SessionCreated,
        response_model_exclude_none=True,
        status_code=201,
        dependencies=[Depends(authenticated)],
        tags=["tracking"],
    )
    async def create_session(
        request: Request,
        payload: Annotated[CreateSessionRequest, Body()] = CreateSessionRequest(),
    ) -> SessionCreated:
        service = ready_service(request)
        return await anyio.to_thread.run_sync(
            service.create_session,
            payload.fps,
            payload.lap_calibration_id,
            payload.diagnostics,
        )

    @app.delete(
        "/v1/tracking-sessions/{session_id}",
        status_code=204,
        dependencies=[Depends(authenticated)],
        tags=["tracking"],
    )
    async def delete_session(request: Request, session_id: str) -> Response:
        service = ready_service(request)
        await anyio.to_thread.run_sync(service.delete_session, session_id)
        return Response(status_code=204)

    @app.post(
        "/v1/tracking-sessions/{session_id}/batches",
        response_model=BatchResult,
        response_model_exclude_none=True,
        dependencies=[Depends(authenticated)],
        tags=["tracking"],
    )
    async def process_batch(
        request: Request,
        response: Response,
        session_id: str,
        frames: Annotated[list[UploadFile], File(description="Ordered JPEG, PNG, or WebP frames")],
        metadata: Annotated[str, Form(description="JSON BatchMetadata in the same order as frames")],
    ) -> BatchResult:
        request_started = time.perf_counter()
        service = ready_service(request)
        if len(metadata.encode("utf-8")) > settings.max_metadata_bytes:
            raise PayloadTooLargeError(f"Batch metadata is limited to {settings.max_metadata_bytes} bytes")
        try:
            parsed_metadata = BatchMetadata.model_validate_json(metadata)
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            raise InvalidFrameError(f"Invalid batch metadata: {exc}") from exc
        if len(frames) != len(parsed_metadata.frames):
            raise InvalidFrameError("metadata.frames and uploaded frames must have the same length")
        if len(frames) > settings.max_batch_frames:
            raise PayloadTooLargeError(f"A batch can contain at most {settings.max_batch_frames} frames")

        encoded_frames: list[bytes] = []
        decoded_frames: list[np.ndarray] = []
        total_bytes = 0
        for upload in frames:
            try:
                if upload.content_type not in ALLOWED_IMAGE_TYPES:
                    raise InvalidFrameError(f"Unsupported image content type: {upload.content_type}")
                payload = await upload.read(settings.max_frame_bytes + 1)
            finally:
                await upload.close()
            if not payload:
                raise InvalidFrameError("Uploaded frames cannot be empty")
            if len(payload) > settings.max_frame_bytes:
                raise PayloadTooLargeError(f"Each encoded frame is limited to {settings.max_frame_bytes} bytes")
            total_bytes += len(payload)
            if total_bytes > settings.max_batch_bytes:
                raise PayloadTooLargeError(f"Encoded batch is limited to {settings.max_batch_bytes} bytes")
            decoded = await anyio.to_thread.run_sync(_decode_image, payload)
            if decoded is None or decoded.ndim != 3 or decoded.shape[2] != 3:
                raise InvalidFrameError(f"Could not decode image {upload.filename!r}")
            if decoded.shape[0] * decoded.shape[1] > settings.max_decoded_pixels:
                raise PayloadTooLargeError(f"Decoded frames are limited to {settings.max_decoded_pixels} pixels")
            encoded_frames.append(payload)
            decoded_frames.append(decoded)

        canonical_metadata = parsed_metadata.model_dump_json()
        fingerprint = service.fingerprint(canonical_metadata, encoded_frames)
        processing_started = time.perf_counter()
        result = await anyio.to_thread.run_sync(
            service.process_batch,
            session_id,
            parsed_metadata,
            decoded_frames,
            fingerprint,
        )
        completed = time.perf_counter()
        response.headers["X-Swimtrack-Decode-Ms"] = f"{(processing_started - request_started) * 1000.0:.3f}"
        response.headers["X-Swimtrack-Process-Ms"] = f"{(completed - processing_started) * 1000.0:.3f}"
        response.headers["X-Swimtrack-Total-Ms"] = f"{(completed - request_started) * 1000.0:.3f}"
        return result

    return app
