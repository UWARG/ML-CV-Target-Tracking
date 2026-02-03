"""
ObjectTracker module using DepthAI's built-in ObjectTracker node.

Configures the ObjectTracker node within a DepthAI pipeline and parses
tracklet output into TrackedObject data classes.

The ObjectTracker node is part of the on-device pipeline:
    SpatialDetectionNetwork.out ──► ObjectTracker ──► XLinkOut("tracklets")

This module provides:
- configure_tracker_node(): sets up the node in a shared pipeline
- parse_tracklets(): converts raw DepthAI tracklets into TrackedObject list

Reference: https://docs.luxonis.com/software/depthai/depthai-components/nodes/objecttracker/
"""

from typing import List

import depthai as dai

from .tracked_object import TrackedObject, TrackingStatus


# Map DepthAI tracklet status to our TrackingStatus enum
_STATUS_MAP = {
    dai.Tracklet.TrackingStatus.NEW: TrackingStatus.NEW,
    dai.Tracklet.TrackingStatus.TRACKED: TrackingStatus.TRACKED,
    dai.Tracklet.TrackingStatus.LOST: TrackingStatus.LOST,
    dai.Tracklet.TrackingStatus.REMOVED: TrackingStatus.LOST,
}

# Available tracker algorithms
TRACKER_TYPES = {
    "ZERO_TERM_COLOR_HISTOGRAM": dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM,
    "ZERO_TERM_IMAGELESS": dai.TrackerType.ZERO_TERM_IMAGELESS,
    "SHORT_TERM_IMAGELESS": dai.TrackerType.SHORT_TERM_IMAGELESS,
    "SHORT_TERM_KCF": dai.TrackerType.SHORT_TERM_KCF,
}


def configure_tracker_node(
    pipeline: dai.Pipeline,
    spatial_detection_network: dai.node.SpatialDetectionNetwork,
    tracker_type: str = "SHORT_TERM_IMAGELESS",
    labels_to_track: List[int] = None,
) -> dai.node.ObjectTracker:
    """
    Create and configure an ObjectTracker node in the DepthAI pipeline.

    This wires the tracker to the SpatialDetectionNetwork outputs.
    Teammates provide the pipeline and spatial_detection_network node;
    this function adds the tracker on top.

    Args:
        pipeline: The shared DepthAI pipeline (created by teammates).
        spatial_detection_network: The detection network node whose
            outputs we consume.
        tracker_type: Algorithm name. One of:
            ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS,
            SHORT_TERM_IMAGELESS, SHORT_TERM_KCF.
        labels_to_track: List of class label indices to track.
            If None, tracks all detected labels.

    Returns:
        The configured ObjectTracker node (already linked to inputs
        and to an XLinkOut named "tracklets").
    """
    if tracker_type not in TRACKER_TYPES:
        raise ValueError(
            f"Unknown tracker_type '{tracker_type}'. "
            f"Options: {list(TRACKER_TYPES.keys())}"
        )

    # --- create tracker node ---
    tracker = pipeline.create(dai.node.ObjectTracker)
    tracker.setTrackerType(TRACKER_TYPES[tracker_type])
    tracker.setTrackerIdAssignmentPolicy(
        dai.TrackerIdAssignmentPolicy.UNIQUE_ID,
    )

    if labels_to_track is not None:
        tracker.setDetectionLabelsToTrack(labels_to_track)

    # --- link detection network outputs into tracker inputs ---
    # passthrough frame (RGB preview used for detection)
    spatial_detection_network.passthrough.link(tracker.inputTrackerFrame)
    # detection frame (same frame, used for re-identification)
    spatial_detection_network.passthrough.link(tracker.inputDetectionFrame)
    # detection results (bounding boxes + spatial coords)
    spatial_detection_network.out.link(tracker.inputDetections)

    # --- create XLinkOut so host can read tracklets ---
    tracker_out = pipeline.create(dai.node.XLinkOut)
    tracker_out.setStreamName("tracklets")
    tracker.out.link(tracker_out.input)

    return tracker


def parse_tracklets(
    tracklets_data: dai.Tracklets,
    label_map: List[str],
    frame_width: int,
    frame_height: int,
) -> List[TrackedObject]:
    """
    Convert raw DepthAI Tracklets output into a list of TrackedObject.

    Called each frame after reading from the device output queue.

    Args:
        tracklets_data: Raw tracklets from device.getOutputQueue("tracklets").get()
        label_map: Ordered list of class names matching model label indices
            (e.g. ["person", "car", "landing_pad"]).
        frame_width: Original frame width in pixels (for denormalizing bbox).
        frame_height: Original frame height in pixels.

    Returns:
        List of TrackedObject with persistent IDs, status, and smoothed
        spatial coordinates.
    """
    tracked_objects: List[TrackedObject] = []

    for tracklet in tracklets_data.tracklets:
        # --- status ---
        status = _STATUS_MAP.get(tracklet.status, TrackingStatus.LOST)

        # skip objects that have been fully removed
        if tracklet.status == dai.Tracklet.TrackingStatus.REMOVED:
            continue

        # --- label ---
        label_index = tracklet.label
        label = (
            label_map[label_index]
            if label_index < len(label_map)
            else str(label_index)
        )

        # --- confidence ---
        confidence = tracklet.srcImgDetection.confidence

        # --- smoothed spatial coordinates (meters) ---
        spatial = tracklet.spatialCoordinates
        x = spatial.x / 1000.0  # mm -> m
        y = spatial.y / 1000.0
        z = spatial.z / 1000.0

        # --- bounding box (denormalize from 0-1 to pixels) ---
        roi = tracklet.roi.denormalize(frame_width, frame_height)
        bbox_x = int(roi.topLeft().x)
        bbox_y = int(roi.topLeft().y)
        bbox_width = int(roi.bottomRight().x - roi.topLeft().x)
        bbox_height = int(roi.bottomRight().y - roi.topLeft().y)

        tracked_objects.append(
            TrackedObject(
                object_id=tracklet.id,
                status=status,
                label=label,
                confidence=confidence,
                x=x,
                y=y,
                z=z,
                bbox_x=bbox_x,
                bbox_y=bbox_y,
                bbox_width=bbox_width,
                bbox_height=bbox_height,
            )
        )

    return tracked_objects
