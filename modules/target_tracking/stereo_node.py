"""
Module for initializing and configuring the StereoDepth node.
This setup aligns the depth map to the RGB camera for spatial logic.
"""

import depthai as dai


def create_stereo_depth(pipeline: dai.Pipeline) -> dai.node.StereoDepth:
    """
    Creates the StereoDepth node and links it to the Mono cameras.

    Args:
        pipeline (dai.Pipeline): The DepthAI pipeline object.

    Returns:
        dai.node.StereoDepth: The configured stereo node.
    """
    # --- 1. Define Sources  ---
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)

    # Configure the hardware sockets (Left vs Right)
    mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    # Set Resolution (400p is standard)
    # Breaking line to satisfy flake8 line length limit
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    # --- 2. Define the Processor  ---
    stereo = pipeline.create(dai.node.StereoDepth)

    # --- 3. Configuration ---
    # CRITICAL: Align depth to RGB (bc mono cams are 20 pixels off)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

    # Improve quality
    stereo.setSubpixel(True)
    stereo.setLeftRightCheck(True)  # Removes ghost pixels at edges
    # Change to True if <50cm need tracking needed
    stereo.setExtendedDisparity(False)

    # --- 4. Linking ---
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    return stereo
