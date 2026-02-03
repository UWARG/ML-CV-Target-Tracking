"""
Worker process for ObjectTracker.

Reads tracklet output from the OAK-D device queue, converts it into
TrackedObject data classes, and pushes them to the next pipeline stage.

Follows the existing worker pattern (producer-consumer via queues).
"""

import logging
from typing import List

import depthai as dai

from .object_tracker import configure_tracker_node, parse_tracklets
from .tracked_object import TrackedObject

logger = logging.getLogger(__name__)


def object_tracker_run(
    pipeline: dai.Pipeline,
    spatial_detection_network: dai.node.SpatialDetectionNetwork,
    label_map: List[str],
    frame_width: int,
    frame_height: int,
    output_queue,  # multiprocessing.Queue[List[TrackedObject]]
    tracker_type: str = "SHORT_TERM_IMAGELESS",
    labels_to_track: List[int] = None,
) -> None:
    """
    Main worker entry point for the ObjectTracker.

    Configures the tracker node inside the given pipeline, then
    continuously reads tracklet output and pushes TrackedObject lists
    to output_queue.

    In the full system the pipeline is started externally (because
    StereoDepth and SpatialDetectionNetwork share the same device
    pipeline). This function is called *before* pipeline start so it
    can wire the tracker node, and then enters the read loop *after*
    the caller starts the device.

    Args:
        pipeline: The shared DepthAI pipeline.
        spatial_detection_network: Detection node to wire into.
        label_map: Ordered class names matching model label indices.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.
        output_queue: Queue for downstream consumers.
        tracker_type: Tracker algorithm name.
        labels_to_track: Label indices to track (None = all).
    """
    configure_tracker_node(
        pipeline=pipeline,
        spatial_detection_network=spatial_detection_network,
        tracker_type=tracker_type,
        labels_to_track=labels_to_track,
    )

    logger.info(
        "ObjectTracker node configured (type=%s). "
        "Waiting for pipeline to start on device.",
        tracker_type,
    )


def object_tracker_read_loop(
    device: dai.Device,
    label_map: List[str],
    frame_width: int,
    frame_height: int,
    output_queue,  # multiprocessing.Queue[List[TrackedObject]]
) -> None:
    """
    Blocking loop that reads tracklets from the device and pushes
    TrackedObject lists to output_queue.

    Call this after the device has been started with the pipeline.

    Args:
        device: Running OAK-D device.
        label_map: Ordered class names.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.
        output_queue: Queue for downstream consumers.
    """
    tracklet_queue = device.getOutputQueue(
        name="tracklets",
        maxSize=4,
        blocking=False,
    )

    logger.info("ObjectTracker read loop started.")

    while True:
        tracklets_data = tracklet_queue.get()  # blocks until next frame

        tracked_objects = parse_tracklets(
            tracklets_data=tracklets_data,
            label_map=label_map,
            frame_width=frame_width,
            frame_height=frame_height,
        )

        if tracked_objects:
            logger.debug(
                "Frame produced %d tracked objects: %s",
                len(tracked_objects),
                [
                    f"id={t.object_id} status={t.status.value}"
                    for t in tracked_objects
                ],
            )

        output_queue.put(tracked_objects)
