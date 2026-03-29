"""
Module for initializing the Spatial Detection Network.
Integrates the stereo depth node with YOLO for 3D object localization.
"""

import depthai as dai


# YOLOv4-tiny input resolution (must match model training size)
YOLO_INPUT_SIZE = 416

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

# COCO dataset: 80 classes, YOLOv4-tiny anchor config
NUM_CLASSES = 80
COORDINATE_SIZE = 4
ANCHORS = [10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319]
ANCHOR_MASKS = {"side26": [1, 2, 3], "side13": [3, 4, 5]}

# Depth ROI scale (fraction of bounding box used to sample depth)
BOUNDING_BOX_SCALE_FACTOR = 0.5

# Valid depth range in mm
DEPTH_LOWER_THRESHOLD_MM = 100
DEPTH_UPPER_THRESHOLD_MM = 10000


def create_spatial_detection_network(
    pipeline: dai.Pipeline,
    stereo: dai.node.StereoDepth,
    model_path: str,
) -> "tuple[dai.node.YoloSpatialDetectionNetwork, dai.node.ColorCamera]":
    """
    Creates the YoloSpatialDetectionNetwork and links it with stereo depth + color camera.

    The color camera provides RGB frames for YOLO detection.
    The stereo depth node provides per-pixel depth aligned to the RGB camera.
    The network fuses both to produce detections with (x, y, z) spatial coordinates.

    Args:
        pipeline: The DepthAI pipeline object.
        stereo: Configured StereoDepth node (from stereo_node.py, depth aligned to RGB).
        model_path: Path to the .blob model file. Set via spatial_detection.model_path in config.yaml.

    Returns:
        Tuple of (spatial_detection_network, color_camera) nodes.
    """
    # --- 1. Color Camera ---
    color_cam = pipeline.create(dai.node.ColorCamera)
    color_cam.setPreviewSize(YOLO_INPUT_SIZE, YOLO_INPUT_SIZE)
    color_cam.setInterleaved(False)
    color_cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    color_cam.setFps(30)

    # --- 2. Spatial Detection Network ---
    spatial_detection = pipeline.create(dai.node.YoloSpatialDetectionNetwork)

    spatial_detection.setBlobPath(model_path)

    # Detection settings
    spatial_detection.setConfidenceThreshold(CONFIDENCE_THRESHOLD)
    spatial_detection.setIouThreshold(IOU_THRESHOLD)

    # YOLO architecture settings
    spatial_detection.setNumClasses(NUM_CLASSES)
    spatial_detection.setCoordinateSize(COORDINATE_SIZE)
    spatial_detection.setAnchors(ANCHORS)
    spatial_detection.setAnchorMasks(ANCHOR_MASKS)

    # Depth fusion settings
    spatial_detection.setBoundingBoxScaleFactor(BOUNDING_BOX_SCALE_FACTOR)
    spatial_detection.setDepthLowerThreshold(DEPTH_LOWER_THRESHOLD_MM)
    spatial_detection.setDepthUpperThreshold(DEPTH_UPPER_THRESHOLD_MM)

    # --- 3. Linking ---
    color_cam.preview.link(spatial_detection.input)
    stereo.depth.link(spatial_detection.inputDepth)

    return spatial_detection, color_cam
