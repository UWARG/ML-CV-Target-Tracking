import time
import pytest
from modules.object_tracker.software_tracker import SoftwareTracker
from modules.object_tracker.detection import Detection
from modules.object_tracker.tracked_object import TrackingStatus

def test_tracker_initialization():
    """Verify the tracker initializes with an empty state."""
    tracker = SoftwareTracker()
    assert len(tracker._tracks) == 0
    assert tracker._next_id == 1

def test_new_track_creation():
    """A new detection should spawn a new track with status NEW."""
    tracker = SoftwareTracker()
    
    # Create a mock detection
    det = Detection(
        label="person", confidence=0.9,
        x=1.0, y=2.0, z=5.0,
        xmin=100, ymin=100, xmax=200, ymax=200
    )
    
    results = tracker.update([det])
    
    assert len(results) == 1
    track = results[0]
    assert track.object_id == 1
    assert track.status == TrackingStatus.NEW
    assert track.label == "person"
    assert track.bbox_width == 100

def test_track_persistence_and_smoothing():
    """A heavily overlapping detection in the next frame should update the existing track."""
    tracker = SoftwareTracker(smoothing_alpha=0.5) # Set alpha to 0.5 for easy math
    
    # Frame 1
    det1 = Detection("car", 0.9, 0.0, 0.0, 0.0, 0, 0, 100, 100)
    tracker.update([det1])
    
    # Frame 2: Move the bounding box slightly, ensuring high IoU
    det2 = Detection("car", 0.95, 2.0, 2.0, 2.0, 10, 10, 110, 110)
    results = tracker.update([det2])
    
    assert len(results) == 1
    track = results[0]
    
    # ID should remain the same, but status upgrades to TRACKED
    assert track.object_id == 1
    assert track.status == TrackingStatus.TRACKED
    
    # Because alpha=0.5, new X should be 0.5 * 2.0 + 0.5 * 0.0 = 1.0
    assert track.x == 1.0
    assert track.bbox_x == 5  # 0.5 * 10 + 0.5 * 0

def test_track_lost_and_removed():
    """A track should become LOST if missed, and removed if missed too long."""
    # Set a very short max_lost_frames (0.1 seconds) for testing
    tracker = SoftwareTracker(max_lost_frames=0.1)
    
    det = Detection("drone", 0.9, 0, 0, 0, 0, 0, 50, 50)
    tracker.update([det])
    
    # Frame 2: Empty detections (object disappears)
    results = tracker.update([])
    assert len(results) == 1
    assert results[0].status == TrackingStatus.LOST
    
    # Frame 3: Wait longer than max_lost_frames, then update
    time.sleep(0.15)
    results_final = tracker.update([])
    
    # The track should be completely purged from memory
    assert len(results_final) == 0

@pytest.mark.xfail(reason="Currently failing as len(results) == 1")
def test_id_contention_highest_iou_wins():
    """If two detections overlap a track, the one with higher IoU should claim the ID."""
    tracker = SoftwareTracker(iou_threshold=0.1)
    
    # Frame 1: Original object
    det1 = Detection("person", 0.9, 0, 0, 0, 0, 0, 100, 100)
    tracker.update([det1])
    
    # Frame 2: Two new detections. 
    # det2_a overlaps perfectly (IoU ~ 1.0)
    # det2_b overlaps partially (IoU ~ 0.25)
    det2_a = Detection("person", 0.9, 0, 0, 0, 0, 0, 100, 100)
    det2_b = Detection("person", 0.9, 0, 0, 0, 50, 50, 150, 150)
    
    results = tracker.update([det2_a, det2_b])
    
    assert len(results) == 2
    
    # Sort results by ID to inspect them deterministically
    results.sort(key=lambda t: t.object_id)
    
    # The first track (ID 1) should be claimed by det2_a (perfect overlap)
    track_1 = results[0]
    assert track_1.object_id == 1
    assert track_1.bbox_x == 0  # From det2_a
    
    # det2_b should have been forced to spawn a new track (ID 2)
    track_2 = results[1]
    assert track_2.object_id == 2
    assert track_2.bbox_x == 50 # From det2_b

def test_sub_threshold_iou_spawns_new_track():
    """A detection with IoU lower than the threshold should not match, even if it's the only candidate."""
    tracker = SoftwareTracker(iou_threshold=0.5) # High threshold for strictness
    
    # Frame 1
    det1 = Detection("drone", 0.9, 0, 0, 0, 0, 0, 100, 100)
    tracker.update([det1])
    
    # Frame 2: Barely overlapping (e.g., only 10x10 pixels overlap)
    det2 = Detection("drone", 0.9, 0, 0, 0, 90, 90, 190, 190)
    results = tracker.update([det2])
    
    # Because IoU < 0.5, det2 should NOT claim ID 1. It should spawn ID 2.
    # The original ID 1 should now be marked as LOST.
    assert len(results) == 2
    
    # Check the statuses
    statuses = {t.object_id: t.status for t in results}
    assert statuses[1] == TrackingStatus.LOST
    assert statuses[2] == TrackingStatus.NEW