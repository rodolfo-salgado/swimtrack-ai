from __future__ import annotations

import numpy as np

from swimtrack_ai.calibration import FIXED_CAMERA_CALIBRATION_ID, LaneRouter


def test_lane_router_assigns_each_detection_to_at_most_one_calibrated_lane() -> None:
    detections = np.asarray(
        [
            [450.0, 450.0, 630.0, 650.0, 0.90],
            [0.0, 0.0, 100.0, 100.0, 0.95],
        ],
        dtype=np.float32,
    )
    router = LaneRouter(FIXED_CAMERA_CALIBRATION_ID, enabled=True)

    routed = router.route(detections, (1080, 1080))

    assert router.lane_ids == ("center",)
    assert routed["center"].tolist() == [detections[0].tolist()]


def test_disabled_lane_router_preserves_legacy_global_input() -> None:
    detections = np.asarray([[0.0, 0.0, 100.0, 100.0, 0.90]], dtype=np.float32)
    router = LaneRouter(FIXED_CAMERA_CALIBRATION_ID, enabled=False)

    routed = router.route(detections, (1080, 1080))

    assert router.lane_ids == ("global",)
    assert np.array_equal(routed["global"], detections)
