"""
ObjectTracker module for persistent tracking of detected objects across frames.

Two modes available:
1. On-device (DepthAI): Use configure_tracker_node() + parse_tracklets()
2. Software (host): Use SoftwareTracker.update(detections)
"""

from .tracked_object import TrackedObject, TrackingStatus
from .detection import Detection

# On-device DepthAI tracker
from .object_tracker import configure_tracker_node, parse_tracklets

# Software tracker (accepts Detection objects)
from .software_tracker import SoftwareTracker

# Workers
from .object_tracker_worker import object_tracker_run, object_tracker_read_loop
