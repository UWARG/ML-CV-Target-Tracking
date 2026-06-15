"""
Module for initializing the Object Tracker node.
Wraps DepthAI's ObjectTracker to track detections across frames with persistent IDs.
"""

import depthai as dai


# Track only class 0 (person in COCO dataset)
PERSON_CLASS_ID = 0

# ZERO_TERM_COLOR_HISTOGRAM: lightweight, works without re-identification network
TRACKER_TYPE = dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM

# SMALLEST_ID: reuse lowest available ID, keeps IDs stable across frames
ASSIGNMENT_POLICY = dai.TrackerIdAssignmentPolicy.SMALLEST_ID


def create_object_tracker(
    pipeline: dai.Pipeline,
    spatial_detection: dai.node.SpatialDetectionNetwork,
) -> dai.node.ObjectTracker:
    """
    Creates the ObjectTracker node and links it to the spatial detection network.

    Args:
        pipeline: The DepthAI pipeline object.
        spatial_detection: The configured SpatialDetectionNetwork node.

    Returns:
        Configured ObjectTracker node.
    """
    tracker = pipeline.create(dai.node.ObjectTracker)

    # Only track humans (COCO class 0)
    tracker.setDetectionLabelsToTrack([PERSON_CLASS_ID])
    tracker.setTrackerType(TRACKER_TYPE)
    tracker.setTrackerIdAssignmentPolicy(ASSIGNMENT_POLICY)
    tracker.setRunOnHost(True)

    # passthrough provides the preview frame used to extract appearance features
    spatial_detection.passthrough.link(tracker.inputTrackerFrame)
    spatial_detection.passthrough.link(tracker.inputDetectionFrame)
    spatial_detection.out.link(tracker.inputDetections)

    return tracker
