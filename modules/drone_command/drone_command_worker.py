"""
Worker that reads TrackedObject lists from the ObjectTracker output queue
and commands the drone to move toward / face the selected target.

Target selection policy (simple):
    - Pick the TRACKED object closest to the camera (smallest z).
    - Ignore NEW objects until they become TRACKED (avoids false positives).
    - If all objects are LOST, hold position and do nothing.
"""

import logging
from typing import List, Optional

from modules.object_tracker.tracked_object import TrackedObject, TrackingStatus
from .drone_command import DroneCommander

logger = logging.getLogger(__name__)


def _select_target(
    tracked_objects: List[TrackedObject],
) -> Optional[TrackedObject]:
    """
    Choose which tracked object to pursue.

    Policy: closest TRACKED object by depth (z).

    Args:
        tracked_objects: All objects from the current frame.

    Returns:
        The selected TrackedObject, or None if nothing to pursue.
    """
    candidates = [
        obj
        for obj in tracked_objects
        if obj.status == TrackingStatus.TRACKED
    ]

    if not candidates:
        return None

    # smallest z = closest to camera
    return min(candidates, key=lambda obj: obj.z)


def drone_command_worker_run(
    input_queue,  # multiprocessing.Queue[List[TrackedObject]]
    connection_string: str,
    baud_rate: int = 57600,
    move: bool = True,
    face: bool = True,
) -> None:
    """
    Main loop: read tracked objects, pick a target, command the drone.

    Args:
        input_queue: Queue of List[TrackedObject] from ObjectTracker.
        connection_string: MAVLink address (e.g. "tcp:localhost:5762").
        baud_rate: Serial baud rate.
        move: Send position commands.
        face: Send yaw commands.
    """
    commander = DroneCommander(
        connection_string=connection_string,
        baud_rate=baud_rate,
    )
    commander.connect()

    logger.info("Drone command worker started. Waiting for tracked objects …")

    while True:
        tracked_objects: List[TrackedObject] = input_queue.get()

        target = _select_target(tracked_objects)

        if target is None:
            logger.debug("No TRACKED target this frame — holding position.")
            continue

        logger.info(
            "Pursuing target id=%d label=%s at (%.2f, %.2f, %.2f)",
            target.object_id,
            target.label,
            target.x,
            target.y,
            target.z,
        )

        # cam coords → body → MAVLink
        # yaw_rad=0.0 for now; in production read from flight controller
        commander.track_target(
            cam_x=target.x,
            cam_y=target.y,
            cam_z=target.z,
            yaw_rad=0.0,
            move=move,
            face=face,
        )
