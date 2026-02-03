"""
Detection input class - interface contract with detection team.

This matches the Detection class from the SpatialDetectionNetwork team.
"""

from dataclasses import dataclass


@dataclass
class Detection:
    """
    Standardized detection result from SpatialDetectionNetwork.

    This is the input format we receive from the detection team.
    """

    label: str
    confidence: float
    x: float  # spatial X (meters, camera frame)
    y: float  # spatial Y (meters, camera frame)
    z: float  # spatial Z / depth (meters, camera frame)
    xmin: float  # bbox left (pixels or normalized)
    ymin: float  # bbox top (pixels or normalized)
    xmax: float  # bbox right (pixels or normalized)
    ymax: float  # bbox bottom (pixels or normalized)
