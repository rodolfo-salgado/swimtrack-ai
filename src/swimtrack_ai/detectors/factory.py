from __future__ import annotations

import logging

from swimtrack_ai.config import Settings

from .base import Detector

logger = logging.getLogger(__name__)


def create_detector(settings: Settings) -> Detector:
    if settings.backend == "fake":
        from .fake import FakeDetector

        return FakeDetector()

    from .tensorrt import (
        EngineLoadError,
        TensorRTDetector,
        build_engine,
        engine_cache_is_current,
        invalidate_engine_cache,
    )

    if not engine_cache_is_current(settings):
        invalidate_engine_cache(settings)
        logger.info("Building TensorRT engine at %s", settings.engine_path)
        build_engine(settings)
    try:
        return TensorRTDetector(settings)
    except EngineLoadError:
        logger.exception("Cached TensorRT engine is incompatible; rebuilding it once")
        invalidate_engine_cache(settings)
        build_engine(settings)
        return TensorRTDetector(settings)
