"""
Module for initializing and configuring the StereoDepth node.
Depth aligned to CAM_A (RGB) for spatial detection.
"""

import depthai as dai


def create_stereo_depth(pipeline: dai.Pipeline) -> dai.node.StereoDepth:
    """
    Creates the StereoDepth node and links it to the mono cameras.

    Args:
        pipeline: The DepthAI pipeline object.

    Returns:
        Configured StereoDepth node with depth aligned to CAM_A.
    """
    mono_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    mono_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setSubpixel(True)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)

    mono_left.requestOutput((640, 400)).link(stereo.left)
    mono_right.requestOutput((640, 400)).link(stereo.right)

    return stereo
