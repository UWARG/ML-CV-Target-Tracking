"""
Unit tests for the target tracking pipeline nodes.
Mocks depthai and blobconverter so no OAK-D hardware is required.
"""

import sys
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Mock depthai + blobconverter BEFORE importing any project modules that
# import them at module level.  This must happen at collection time.
# ---------------------------------------------------------------------------
_dai = MagicMock()

# Give TrackerType / TrackerIdAssignmentPolicy real-ish sentinel values so
# the modules can assign them to constants without blowing up.
_dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM = "ZERO_TERM_COLOR_HISTOGRAM"
_dai.TrackerIdAssignmentPolicy.SMALLEST_ID = "SMALLEST_ID"
_dai.MonoCameraProperties.SensorResolution.THE_400_P = "400P"
_dai.ColorCameraProperties.ColorOrder.BGR = "BGR"
_dai.CameraBoardSocket.LEFT = "LEFT"
_dai.CameraBoardSocket.RIGHT = "RIGHT"
_dai.CameraBoardSocket.RGB = "RGB"

sys.modules["depthai"] = _dai
sys.modules["blobconverter"] = MagicMock()

# Now safe to import project modules
from modules.target_tracking.stereo_node import create_stereo_depth  # noqa: E402
from modules.target_tracking.spatial_detection_node import (  # noqa: E402
    create_spatial_detection_network,
)
from modules.target_tracking.object_tracker_node import create_object_tracker  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline():
    """Return a fresh mock pipeline whose create() returns distinct mocks."""
    pipeline = MagicMock()
    pipeline.create.side_effect = lambda node_type: MagicMock(name=str(node_type))
    return pipeline


# ---------------------------------------------------------------------------
# stereo_node tests
# ---------------------------------------------------------------------------


class TestCreateStereoDepth:
    def test_returns_stereo_node(self):
        pipeline = _make_pipeline()
        stereo = create_stereo_depth(pipeline)
        assert stereo is not None

    def test_creates_three_nodes(self):
        """Expects: mono_left, mono_right, stereo = 3 pipeline.create() calls."""
        pipeline = _make_pipeline()
        create_stereo_depth(pipeline)
        assert pipeline.create.call_count == 3

    def test_depth_aligned_to_rgb(self):
        pipeline = _make_pipeline()
        stereo = create_stereo_depth(pipeline)
        stereo.setDepthAlign.assert_called_once_with(_dai.CameraBoardSocket.RGB)

    def test_quality_flags_set(self):
        pipeline = _make_pipeline()
        stereo = create_stereo_depth(pipeline)
        stereo.setSubpixel.assert_called_once_with(True)
        stereo.setLeftRightCheck.assert_called_once_with(True)
        stereo.setExtendedDisparity.assert_called_once_with(False)

    def test_mono_cameras_linked_to_stereo(self):
        """mono_left.out.link and mono_right.out.link must each be called once."""
        nodes = []
        pipeline = MagicMock()
        pipeline.create.side_effect = lambda _: (nodes.append(MagicMock()) or nodes[-1])

        create_stereo_depth(pipeline)

        # nodes[0]=mono_left, nodes[1]=mono_right, nodes[2]=stereo
        mono_left, mono_right, _ = nodes
        mono_left.out.link.assert_called_once()
        mono_right.out.link.assert_called_once()


# ---------------------------------------------------------------------------
# spatial_detection_node tests
# ---------------------------------------------------------------------------


class TestCreateSpatialDetectionNetwork:
    def test_returns_network_and_camera(self):
        pipeline = _make_pipeline()
        stereo = MagicMock()
        network, cam = create_spatial_detection_network(pipeline, stereo)
        assert network is not None
        assert cam is not None

    def test_creates_two_nodes(self):
        """Expects: color_cam + YoloSpatialDetectionNetwork = 2 calls."""
        pipeline = _make_pipeline()
        stereo = MagicMock()
        create_spatial_detection_network(pipeline, stereo)
        assert pipeline.create.call_count == 2

    def test_color_camera_preview_size(self):
        pipeline = _make_pipeline()
        stereo = MagicMock()
        _, cam = create_spatial_detection_network(pipeline, stereo)
        cam.setPreviewSize.assert_called_once_with(416, 416)

    def test_confidence_threshold_set(self):
        pipeline = _make_pipeline()
        stereo = MagicMock()
        network, _ = create_spatial_detection_network(pipeline, stereo)
        network.setConfidenceThreshold.assert_called_once_with(0.5)

    def test_depth_linked_to_network(self):
        """stereo.depth.link(spatial_detection.inputDepth) must be called."""
        pipeline = _make_pipeline()
        stereo = MagicMock()
        network, _ = create_spatial_detection_network(pipeline, stereo)
        stereo.depth.link.assert_called_once_with(network.inputDepth)

    def test_color_preview_linked_to_network(self):
        pipeline = _make_pipeline()
        stereo = MagicMock()
        network, cam = create_spatial_detection_network(pipeline, stereo)
        cam.preview.link.assert_called_once_with(network.input)

    def test_custom_model_path_used(self):
        pipeline = _make_pipeline()
        stereo = MagicMock()
        network, _ = create_spatial_detection_network(
            pipeline, stereo, model_path="/fake/model.blob"
        )
        network.setBlobPath.assert_called_once_with("/fake/model.blob")

    def test_default_model_downloaded_when_no_path(self):
        import blobconverter

        blobconverter.from_zoo.reset_mock()
        blobconverter.from_zoo.return_value = "/mocked/zoo_model.blob"
        pipeline = _make_pipeline()
        stereo = MagicMock()
        network, _ = create_spatial_detection_network(pipeline, stereo, model_path=None)
        blobconverter.from_zoo.assert_called_once()
        network.setBlobPath.assert_called_once_with("/mocked/zoo_model.blob")


# ---------------------------------------------------------------------------
# object_tracker_node tests
# ---------------------------------------------------------------------------


class TestCreateObjectTracker:
    def test_returns_tracker(self):
        pipeline = _make_pipeline()
        spatial_detection = MagicMock()
        tracker = create_object_tracker(pipeline, spatial_detection)
        assert tracker is not None

    def test_creates_one_node(self):
        pipeline = _make_pipeline()
        spatial_detection = MagicMock()
        create_object_tracker(pipeline, spatial_detection)
        assert pipeline.create.call_count == 1

    def test_tracks_only_person_class(self):
        pipeline = _make_pipeline()
        spatial_detection = MagicMock()
        tracker = create_object_tracker(pipeline, spatial_detection)
        tracker.setDetectionLabelsToTrack.assert_called_once_with([0])

    def test_tracker_type_set(self):
        pipeline = _make_pipeline()
        spatial_detection = MagicMock()
        tracker = create_object_tracker(pipeline, spatial_detection)
        tracker.setTrackerType.assert_called_once_with(
            _dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM
        )

    def test_detections_linked_to_tracker(self):
        """spatial_detection.out.link(tracker.inputDetections) must be called."""
        pipeline = _make_pipeline()
        spatial_detection = MagicMock()
        tracker = create_object_tracker(pipeline, spatial_detection)
        spatial_detection.out.link.assert_called_once_with(tracker.inputDetections)

    def test_passthrough_linked_twice(self):
        """passthrough links to both inputTrackerFrame and inputDetectionFrame."""
        pipeline = _make_pipeline()
        spatial_detection = MagicMock()
        tracker = create_object_tracker(pipeline, spatial_detection)
        assert spatial_detection.passthrough.link.call_count == 2
        link_targets = {c.args[0] for c in spatial_detection.passthrough.link.call_args_list}
        assert tracker.inputTrackerFrame in link_targets
        assert tracker.inputDetectionFrame in link_targets
