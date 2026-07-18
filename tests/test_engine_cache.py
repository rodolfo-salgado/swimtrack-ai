from __future__ import annotations

import json
import sys
from dataclasses import replace
from types import SimpleNamespace

from swimtrack_ai.config import Settings
from swimtrack_ai.detectors.tensorrt import engine_cache_is_current, engine_signature


def test_engine_manifest_invalidates_when_external_model_data_changes(tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "tensorrt", SimpleNamespace(__version__="10.13.3.9"))
    onnx = tmp_path / "model.onnx"
    external_data = tmp_path / "model.onnx.data"
    engine = tmp_path / "model.engine"
    onnx.write_bytes(b"graph")
    external_data.write_bytes(b"weights-v1")
    engine.write_bytes(b"engine")
    settings = Settings(
        model_source_dir=tmp_path,
        model_cache_dir=tmp_path,
        onnx_filename=onnx.name,
        engine_filename=engine.name,
    )
    settings.engine_manifest_path.write_text(json.dumps(engine_signature(settings)), encoding="utf-8")

    assert engine_cache_is_current(settings)
    assert not engine_cache_is_current(replace(settings, trt_max_batch_size=4))
    external_data.write_bytes(b"weights-v2")
    assert not engine_cache_is_current(settings)
