"""
Module for initializing the Spatial Detection Network.
Integrates the stereo depth node with YOLO for 3D object localization.
"""

import depthai as dai


# Detection thresholds
CONFIDENCE_THRESHOLD = 0.5

# Depth ROI scale (fraction of bounding box used to sample depth)
BOUNDING_BOX_SCALE_FACTOR = 0.5

# Valid depth range in mm
DEPTH_LOWER_THRESHOLD_MM = 100
DEPTH_UPPER_THRESHOLD_MM = 10000


def create_spatial_detection_network(
    pipeline: dai.Pipeline,
    stereo: dai.node.StereoDepth,
    model_name: str,
) -> dai.node.SpatialDetectionNetwork:
    """
    Creates the SpatialDetectionNetwork and links it with stereo depth + color camera.

    Args:
        pipeline: The DepthAI pipeline object.
        stereo: Configured StereoDepth node with depth aligned to CAM_A.
        model_name: Luxonis model zoo name (e.g. "yolov6-nano").

    Returns:
        Configured SpatialDetectionNetwork node.
    """
    # CAM_A is the RGB sensor — must match stereo's setDepthAlign(CAM_A)
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

    spatial_detection = pipeline.create(dai.node.SpatialDetectionNetwork).build(
        cam, stereo, model_name
    )
    spatial_detection.setConfidenceThreshold(CONFIDENCE_THRESHOLD)
    spatial_detection.setBoundingBoxScaleFactor(BOUNDING_BOX_SCALE_FACTOR)
    spatial_detection.setDepthLowerThreshold(DEPTH_LOWER_THRESHOLD_MM)
    spatial_detection.setDepthUpperThreshold(DEPTH_UPPER_THRESHOLD_MM)
    spatial_detection.input.setBlocking(False)

    return spatial_detection
