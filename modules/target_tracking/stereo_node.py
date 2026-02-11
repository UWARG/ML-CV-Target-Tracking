"""
Module for initializing and configuring the StereoDepth node.
This setup aligns the depth map to the RGB camera for spatial logic.
Updated to depthai v3 API.
"""
import depthai as dai


def create_stereo_depth(pipeline: dai.Pipeline) -> dai.node.StereoDepth:
    """
    Creates the StereoDepth node and links it to the stereo cameras.

    The pipeline must be created with a device:
        device = dai.Device()
        with dai.Pipeline(device) as pipeline:
            stereo = create_stereo_depth(pipeline)

    Args:
        pipeline (dai.Pipeline): The DepthAI pipeline object (created with device).

    Returns:
        dai.node.StereoDepth: The configured stereo node.
    """
    # --- 1. Define Sources ) ---
    cam_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    cam_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    # --- 2. Define the Processor ---
    stereo = pipeline.create(dai.node.StereoDepth)

    # --- 3. Configuration ---
    # Enable rectification 
    stereo.setRectification(True)

    # Align depth to RGB camera (CAM_A) for spatial logic
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    # Improve quality
    stereo.setSubpixel(True)
    stereo.setLeftRightCheck(True)  # Removes ghost pixels at edges
    # Change to True if <50cm tracking needed
    stereo.setExtendedDisparity(False)

    left_out = cam_left.requestOutput((640, 400), dai.ImgFrame.Type.GRAY8)
    right_out = cam_right.requestOutput((640, 400), dai.ImgFrame.Type.GRAY8)

    left_out.link(stereo.left)
    right_out.link(stereo.right)

    return stereo
