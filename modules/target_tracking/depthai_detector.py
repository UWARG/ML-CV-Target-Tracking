"""
DepthAI detector implementation using Pipeline 2.0 API.
"""
import depthai as dai
import numpy as np
import cv2
from .stereo_node import create_stereo_depth

class DepthAIDetector:
    """
    DepthAI detector class using SpatialDetectionNetwork.
    """
    def __init__(self, pipeline: "dai.Pipeline"):
        self.pipeline = pipeline
        self.latest_detections = []
        self.latest_frames = {}

    class DataCollector(dai.node.HostNode):
        """
        Host node to collect data from the pipeline and make it available to the detector class.
        """
        def __init__(self, detector):
            dai.node.HostNode.__init__(self)
            self.detector = detector
            self.sendProcessingToPipeline(True)

        def build(self, depth: dai.Node.Output, detections: dai.Node.Output, rgb: dai.Node.Output):
            self.link_args(depth, detections, rgb)

        def process(self, depthPreview, detections, rgbPreview):
            # Store data in the parent detector instance
            self.detector.latest_frames["depth"] = depthPreview.getCvFrame()
            self.detector.latest_frames["rgb"] = rgbPreview.getCvFrame()
            self.detector.latest_detections = detections.detections

    @classmethod
    def create(cls, config: dict) -> "tuple[bool, DepthAIDetector | None]":
        try:
            model_name = config.get("model_path", "yolov6-nano")
            size = config.get("input_size", (640, 400))
            depth_source_type = config.get("depth_source", "stereo")
            fps = config.get("fps", 30)

            p = dai.Pipeline()
            platform = p.getDefaultDevice().getPlatform()

            # Define sources
            cam_rgb = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A, sensorFps=fps)

            # Define depth source
            if depth_source_type == "stereo":
                depth_source = create_stereo_depth(p)
                if platform == dai.Platform.RVC2:
                    depth_source.setOutputSize(*size)
            else:
                print(f"Invalid depth source: {depth_source_type}")
                return False, None

            # Spatial Detection Network
            model_description = dai.NNModelDescription(model_name)
            spatial_nn = p.create(dai.node.SpatialDetectionNetwork).build(
                cam_rgb, depth_source, model_description
            )
            
            # Settings
            spatial_nn.input.setBlocking(False)
            spatial_nn.setBoundingBoxScaleFactor(0.5)
            spatial_nn.setDepthLowerThreshold(100)
            spatial_nn.setDepthUpperThreshold(5000)

            # Create detector instance
            detector = cls(p)
            
            # Use HostNode to collect data
            collector = p.create(cls.DataCollector, detector)
            collector.build(
                spatial_nn.passthroughDepth,
                spatial_nn.out,
                spatial_nn.passthrough,
            )

            # Start the pipeline in the background
            p.start()
            
            return True, detector
        except Exception as e:
            print(f"Error creating DepthAIDetector: {e}")
            return False, None

    def run(self) -> "tuple[bool, list, dict]":
        """
        Returns the latest detections and frames collected by the HostNode.
        """
        # Since the pipeline is running in the background, we just return the latest captured data.
        return True, self.latest_detections, self.latest_frames

    def __del__(self):
        if hasattr(self, 'pipeline'):
            self.pipeline.stop()