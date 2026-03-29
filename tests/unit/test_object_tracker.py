import pytest
from unittest.mock import MagicMock
import depthai as dai

from modules.object_tracker.object_tracker import parse_tracklets
from modules.object_tracker.tracked_object import TrackingStatus

def test_parse_tracklets_valid_data():
    """Verify that raw hardware tracklets are correctly parsed into TrackedObjects."""
    
    # 1. Create a fake hardware tracklet using MagicMock
    mock_tracklet = MagicMock()
    mock_tracklet.id = 42
    mock_tracklet.status = dai.Tracklet.TrackingStatus.TRACKED
    mock_tracklet.label = 1 # Index for "car" in our map
    mock_tracklet.srcImgDetection.confidence = 0.85
    
    # Mock spatial coordinates (millimeters from the hardware)
    mock_tracklet.spatialCoordinates.x = 1000.0 # 1 meter
    mock_tracklet.spatialCoordinates.y = -500.0 # -0.5 meters
    mock_tracklet.spatialCoordinates.z = 5000.0 # 5 meters
    
    # Mock ROI (Region of Interest) denormalization
    mock_roi = MagicMock()
    mock_roi.topLeft.return_value.x = 100
    mock_roi.topLeft.return_value.y = 150
    mock_roi.bottomRight.return_value.x = 300
    mock_roi.bottomRight.return_value.y = 350
    mock_tracklet.roi.denormalize.return_value = mock_roi

    # 2. Bundle it into a fake tracklets data payload
    mock_tracklets_data = MagicMock()
    mock_tracklets_data.tracklets = [mock_tracklet]

    # 3. Define our camera parameters
    label_map = ["person", "car", "drone"]
    frame_width = 1920
    frame_height = 1080

    # 4. Run the function
    results = parse_tracklets(mock_tracklets_data, label_map, frame_width, frame_height)

    # 5. Assert the conversion logic worked (mm to m, denormalization, etc.)
    assert len(results) == 1
    obj = results[0]
    
    assert obj.object_id == 42
    assert obj.status == TrackingStatus.TRACKED
    assert obj.label == "car"
    assert obj.confidence == 0.85
    
    # Check spatial math (mm -> m)
    assert obj.x == 1.0
    assert obj.y == -0.5
    assert obj.z == 5.0
    
    # Check bounding box math
    assert obj.bbox_x == 100
    assert obj.bbox_y == 150
    assert obj.bbox_width == 200   # 300 - 100
    assert obj.bbox_height == 200  # 350 - 150

def test_parse_tracklets_ignores_removed():
    """Verify that tracklets flagged as REMOVED by the hardware are skipped."""
    mock_tracklet = MagicMock()
    mock_tracklet.status = dai.Tracklet.TrackingStatus.REMOVED
    
    mock_tracklets_data = MagicMock()
    mock_tracklets_data.tracklets = [mock_tracklet]
    
    results = parse_tracklets(mock_tracklets_data, ["person"], 1920, 1080)
    
    # The REMOVED tracklet should be filtered out entirely
    assert len(results) == 0

def test_parse_tracklets_unknown_label_fallback():
    """If the hardware returns a label index outside our map, it should fallback to the stringified index."""
    mock_tracklet = MagicMock()
    mock_tracklet.status = dai.Tracklet.TrackingStatus.NEW
    mock_tracklet.label = 99  # Out of bounds!
    
    mock_tracklet.roi.denormalize.return_value.topLeft.return_value.x = 0
    mock_tracklet.roi.denormalize.return_value.topLeft.return_value.y = 0
    mock_tracklet.roi.denormalize.return_value.bottomRight.return_value.x = 10
    mock_tracklet.roi.denormalize.return_value.bottomRight.return_value.y = 10
    
    mock_tracklets_data = MagicMock()
    mock_tracklets_data.tracklets = [mock_tracklet]
    
    results = parse_tracklets(mock_tracklets_data, ["person", "car"], 1920, 1080)
    
    assert len(results) == 1
    assert results[0].label == "99"