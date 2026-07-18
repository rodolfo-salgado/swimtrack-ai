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


def test_far_crop_extends_only_the_calibrated_far_end_of_the_lane() -> None:
    far_crop_box = (320.0 / 1080.0, 120.0 / 1080.0, 760.0 / 1080.0, 560.0 / 1080.0)
    far_end_detection = np.asarray([[545.0, 125.0, 565.0, 145.0, 0.90]], dtype=np.float32)
    above_crop_detection = np.asarray([[545.0, 85.0, 565.0, 105.0, 0.90]], dtype=np.float32)
    outside_lane_detection = np.asarray([[320.0, 125.0, 340.0, 145.0, 0.90]], dtype=np.float32)

    default_router = LaneRouter(FIXED_CAMERA_CALIBRATION_ID, enabled=True)
    crop_router = LaneRouter(
        FIXED_CAMERA_CALIBRATION_ID,
        enabled=True,
        far_crop_box=far_crop_box,
    )

    assert default_router.route(far_end_detection, (1080, 1080))["center"].shape == (0, 5)
    np.testing.assert_allclose(crop_router.route(far_end_detection, (1080, 1080))["center"], far_end_detection)
    assert crop_router.route(above_crop_detection, (1080, 1080))["center"].shape == (0, 5)
    assert crop_router.route(outside_lane_detection, (1080, 1080))["center"].shape == (0, 5)
