"""
Data class for tracked object output from the ObjectTracker node.
"""

from enum import Enum
from dataclasses import dataclass


class TrackingStatus(Enum):
    """Status of a tracked object."""
    NEW = "NEW"           # Object just appeared
    TRACKED = "TRACKED"   # Object being actively tracked
    LOST = "LOST"         # Object lost (not detected in recent frames)


@dataclass
class TrackedObject:
    """
    Represents a tracked object with persistent ID and smoothed coordinates.

    Attributes:
        object_id: Persistent unique identifier for this object across frames
        status: Current tracking status (NEW, TRACKED, LOST)
        label: Object class label from detection model
        confidence: Detection confidence score (0.0 - 1.0)
        x: Smoothed X coordinate (meters, relative to camera)
        y: Smoothed Y coordinate (meters, relative to camera)
        z: Smoothed Z coordinate (depth in meters, relative to camera)
        bbox_x: Bounding box top-left X (pixels)
        bbox_y: Bounding box top-left Y (pixels)
        bbox_width: Bounding box width (pixels)
        bbox_height: Bounding box height (pixels)
    """
    object_id: int
    status: TrackingStatus
    label: str
    confidence: float
    x: float
    y: float
    z: float
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int

    def __repr__(self) -> str:
        return (
            f"TrackedObject(id={self.object_id}, status={self.status.value}, "
            f"label='{self.label}', pos=({self.x:.2f}, {self.y:.2f}, {self.z:.2f}))"
        )
