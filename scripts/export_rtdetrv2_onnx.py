#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "onnx==1.16.1",
#   "onnxscript>=0.1.0",
#   "pyyaml>=6.0",
#   "torch==2.13.0",
#   "torchvision==0.28.0",
# ]
# ///
"""Export the SwimTrack RT-DETRv2 checkpoint to a dynamically-batched ONNX artifact."""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RTDETR_ROOT = ROOT / "vendor" / "RT-DETRv2" / "rtdetrv2_pytorch"
DEFAULT_CHECKPOINT_URL = "https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
DEFAULT_CHECKPOINT_SHA256 = "2ace52184b620204004509b72752ac7bfe64aadaf7fc1d076b18df8ab5a5c77e"
DEFAULT_CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
DEFAULT_CONFIG = RTDETR_ROOT / "configs" / "rtdetrv2" / "rtdetrv2_r18vd_120e_coco.yml"
DEFAULT_OUTPUT = ROOT / "artifacts" / "models" / "rtdetrv2_s.onnx"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_checkpoint(path: Path, url: str, expected_sha256: str) -> None:
    if path.exists():
        actual_sha256 = sha256(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Checkpoint checksum mismatch for {path}: expected {expected_sha256}, got {actual_sha256}"
            )
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as response, temporary_path.open("wb") as destination:  # noqa: S310
            while chunk := response.read(1024 * 1024):
                destination.write(chunk)
        actual_sha256 = sha256(temporary_path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Downloaded checkpoint checksum mismatch: expected {expected_sha256}, got {actual_sha256}"
            )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def normalise_batched_output_shapes(path: Path) -> None:
    """Declare fixed Top-K dimensions while leaving the batch axis dynamic."""
    import onnx

    expected_shapes = {
        "labels": ("N", 300),
        "boxes": ("N", 300, 4),
        "scores": ("N", 300),
    }
    model = onnx.load(path, load_external_data=False)
    for output in model.graph.output:
        expected_shape = expected_shapes.get(output.name)
        if expected_shape is None:
            continue
        dimensions = output.type.tensor_type.shape.dim
        if len(dimensions) != len(expected_shape):
            raise RuntimeError(f"Unexpected ONNX output rank for {output.name}: {len(dimensions)}")
        for dimension, value in zip(dimensions, expected_shape, strict=True):
            dimension.ClearField("dim_value")
            dimension.ClearField("dim_param")
            if isinstance(value, str):
                dimension.dim_param = value
            else:
                dimension.dim_value = value
    onnx.checker.check_model(model)
    onnx.save_model(model, path)


def validate_dynamic_batch_artifact(path: Path, *, input_size: int) -> None:
    import onnx

    expected_shapes = {
        "images": ("N", 3, input_size, input_size),
        "orig_target_sizes": ("N", 2),
        "labels": ("N", 300),
        "boxes": ("N", 300, 4),
        "scores": ("N", 300),
    }
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    values = {value.name: value for value in (*model.graph.input, *model.graph.output)}
    for name, expected_shape in expected_shapes.items():
        value = values.get(name)
        if value is None:
            raise RuntimeError(f"ONNX artifact is missing {name!r}")
        dimensions = value.type.tensor_type.shape.dim
        observed_shape = tuple(
            dimension.dim_param if dimension.HasField("dim_param") else dimension.dim_value for dimension in dimensions
        )
        if observed_shape != expected_shape:
            raise RuntimeError(f"Unexpected shape for {name}: expected {expected_shape}, got {observed_shape}")


def export_onnx(*, config_path: Path, checkpoint_path: Path, output_path: Path, input_size: int) -> None:
    if not RTDETR_ROOT.is_dir():
        raise FileNotFoundError(f"RT-DETRv2 vendor submodule is missing: {RTDETR_ROOT}")
    if not config_path.is_file():
        raise FileNotFoundError(f"RT-DETRv2 config is missing: {config_path}")
    if str(RTDETR_ROOT) not in sys.path:
        sys.path.insert(0, str(RTDETR_ROOT))

    import torch
    import torch.nn as nn
    from src.core import YAMLConfig, yaml_utils

    config = YAMLConfig(str(config_path), **yaml_utils.parse_cli(["PResNet.pretrained=False"]))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("ema", {}).get("module") or checkpoint.get("model")
    if state is None:
        raise RuntimeError("Checkpoint does not contain an EMA module or model state")
    config.model.load_state_dict(state)

    class ExportModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = config.model.deploy()
            self.postprocessor = config.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            return self.postprocessor(self.model(images), orig_target_sizes)

    model = ExportModel().eval()
    images = torch.rand(1, 3, input_size, input_size)
    original_sizes = torch.tensor([[input_size, input_size]])
    with torch.inference_mode():
        _ = model(images, original_sizes)
        torch.onnx.export(
            model,
            (images, original_sizes),
            str(output_path),
            input_names=["images", "orig_target_sizes"],
            output_names=["labels", "boxes", "scores"],
            dynamic_axes={
                "images": {0: "N"},
                "orig_target_sizes": {0: "N"},
                "labels": {0: "N"},
                "boxes": {0: "N"},
                "scores": {0: "N"},
            },
            opset_version=16,
            do_constant_folding=True,
            dynamo=False,
        )
    normalise_batched_output_shapes(output_path)
    validate_dynamic_batch_artifact(output_path, input_size=input_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-url", default=DEFAULT_CHECKPOINT_URL)
    parser.add_argument("--checkpoint-sha256", default=DEFAULT_CHECKPOINT_SHA256)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--force", action="store_true", help="Regenerate an existing ONNX artifact")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_size <= 0:
        raise ValueError("--input-size must be greater than zero")
    checkpoint_path = args.checkpoint.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists() or args.force:
        ensure_checkpoint(checkpoint_path, args.checkpoint_url, args.checkpoint_sha256)
        temporary_output = output_path.with_suffix(output_path.suffix + ".part")
        temporary_output.unlink(missing_ok=True)
        try:
            export_onnx(
                config_path=args.config.resolve(),
                checkpoint_path=checkpoint_path,
                output_path=temporary_output,
                input_size=args.input_size,
            )
            temporary_output.replace(output_path)
        finally:
            temporary_output.unlink(missing_ok=True)
    validate_dynamic_batch_artifact(output_path, input_size=args.input_size)
    checksum_path = output_path.with_suffix(output_path.suffix + ".sha256")
    checksum_path.write_text(f"{sha256(output_path)}  {output_path.name}\n", encoding="utf-8")
    print(f"Validated dynamic-batch ONNX artifact: {output_path}")
    print(f"SHA256 manifest: {checksum_path}")


if __name__ == "__main__":
    main()
