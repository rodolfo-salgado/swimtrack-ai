# SwimTrack AI model artifact

`rtdetrv2_s.onnx` is generated and intentionally not committed. Generate it from this repository alone with `uv run --script scripts/export_rtdetrv2_onnx.py`.

The exporter downloads the pinned RT-DETRv2-S checkpoint, verifies SHA-256 `2ace52184b620204004509b72752ac7bfe64aadaf7fc1d076b18df8ab5a5c77e`, uses the vendored RT-DETRv2 source, and writes an embedded dynamic-batch ONNX file. The current released artifact has SHA-256 `17eb1ce5c325a685474c75bfa658245117ddbf764301ed9fdce2175bedb20fba`; `rtdetrv2_s.onnx.sha256` is generated beside it for deployment verification.

The artifact must expose `images [N,3,640,640]`, `orig_target_sizes [N,2]`, `labels [N,300]`, `boxes [N,300,4]`, and `scores [N,300]`. This lets TensorRT build the B1/B4/B8 optimization profile used by the service.
