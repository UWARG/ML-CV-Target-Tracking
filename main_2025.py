"""
for target tracking
"""

import pathlib

import depthai as dai
import yaml

from modules.target_tracking.stereo_node import create_stereo_depth
from modules.target_tracking.spatial_detection_node import create_spatial_detection_network
from modules.target_tracking.object_tracker_node import create_object_tracker


CONFIG_FILE_PATH = pathlib.Path("config.yaml")

# Queue settings
OUTPUT_QUEUE_SIZE = 4


def main() -> int:
    """Main function for target tracking pipeline."""
    # Load config
    with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    spatial_config = config.get("spatial_detection", {})
    model_path = spatial_config.get("model_path")

    # Build DepthAI pipeline
    pipeline = dai.Pipeline()

    # 1. Stereo Depth Node (mono cams → depth aligned to RGB)
    stereo = create_stereo_depth(pipeline)

    # 2. Spatial Detection Network (RGB cam + stereo depth → 3D detections)
    spatial_detection, color_cam = create_spatial_detection_network(
        pipeline, stereo, model_path=model_path
    )

    # 3. Object Tracker (detections → tracked targets with persistent IDs)
    tracker = create_object_tracker(pipeline, spatial_detection)

    # --- Output XLinks ---
    xout_tracker = pipeline.create(dai.node.XLinkOut)
    xout_tracker.setStreamName("tracklets")
    tracker.out.link(xout_tracker.input)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    color_cam.video.link(xout_rgb.input)

    # --- Run pipeline ---
    with dai.Device(pipeline) as device:
        tracklet_queue = device.getOutputQueue("tracklets", maxSize=OUTPUT_QUEUE_SIZE, blocking=False)

        print("Pipeline started. Tracking humans (COCO class 0)...")

        while True:
            tracklets_msg = tracklet_queue.get()

            for tracklet in tracklets_msg.tracklets:
                if tracklet.status != dai.Tracklet.TrackingStatus.TRACKED:
                    continue

                roi = tracklet.roi.denormalize(
                    tracklets_msg.getWidth(), tracklets_msg.getHeight()
                )
                x_mm = tracklet.spatialCoordinates.x
                y_mm = tracklet.spatialCoordinates.y
                z_mm = tracklet.spatialCoordinates.z

                print(
                    f"Target ID {tracklet.id}: "
                    f"xyz=({x_mm:.0f}mm, {y_mm:.0f}mm, {z_mm:.0f}mm)  "
                    f"bbox=({int(roi.topLeft().x)}, {int(roi.topLeft().y)}, "
                    f"{int(roi.bottomRight().x)}, {int(roi.bottomRight().y)})"
                )

    return 0


if __name__ == "__main__":
    result_main = main()
    if result_main < 0:
        print(f"ERROR: Status code: {result_main}")

    print("Done!")
