"""
for target tracking
"""

import pathlib

import cv2
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
    with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    model_path: str = config["spatial_detection"]["model_path"]

    pipeline = dai.Pipeline()

    # 1. Stereo Depth Node (mono cams → depth aligned to RGB)
    stereo = create_stereo_depth(pipeline)

    # 2. Spatial Detection Network (RGB cam + stereo depth → 3D detections)
    spatial_detection, color_cam = create_spatial_detection_network(pipeline, stereo, model_path)

    # 3. Object Tracker (detections → tracked targets with persistent IDs)
    tracker = create_object_tracker(pipeline, spatial_detection)

    # --- Output XLinks ---
    xout_tracker = pipeline.create(dai.node.XLinkOut)
    xout_tracker.setStreamName("tracklets")
    tracker.out.link(xout_tracker.input)

    # Link preview (same 416x416 frame the NN ran on) so bbox coords align
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    color_cam.preview.link(xout_rgb.input)

    with dai.Device(pipeline) as device:
        tracklet_queue = device.getOutputQueue("tracklets", maxSize=OUTPUT_QUEUE_SIZE, blocking=False)
        rgb_queue = device.getOutputQueue("rgb", maxSize=OUTPUT_QUEUE_SIZE, blocking=False)

        print("Pipeline started. Tracking humans (COCO class 0)...")

        while True:
            tracklets_msg = tracklet_queue.get()
            frame_msg = rgb_queue.tryGet()
            frame = frame_msg.getCvFrame() if frame_msg is not None else None

            for tracklet in tracklets_msg.tracklets:
                if tracklet.status != dai.Tracklet.TrackingStatus.TRACKED:
                    continue

                roi = tracklet.roi.denormalize(tracklets_msg.getWidth(), tracklets_msg.getHeight())
                x_mm = tracklet.spatialCoordinates.x
                y_mm = tracklet.spatialCoordinates.y
                z_mm = tracklet.spatialCoordinates.z

                print(
                    f"Target ID {tracklet.id}: "
                    f"xyz=({x_mm:.0f}mm, {y_mm:.0f}mm, {z_mm:.0f}mm)  "
                    f"bbox=({int(roi.topLeft().x)}, {int(roi.topLeft().y)}, "
                    f"{int(roi.bottomRight().x)}, {int(roi.bottomRight().y)})"
                )

                if frame is not None:
                    cv2.rectangle(
                        frame,
                        (int(roi.topLeft().x), int(roi.topLeft().y)),
                        (int(roi.bottomRight().x), int(roi.bottomRight().y)),
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"ID {tracklet.id} | {z_mm:.0f}mm",
                        (int(roi.topLeft().x), int(roi.topLeft().y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

            if frame is not None:
                cv2.imshow("Target Tracking", frame)
                if cv2.waitKey(1) == ord("q"):
                    break

    return 0


if __name__ == "__main__":
    result_main = main()
    if result_main < 0:
        print(f"ERROR: Status code: {result_main}")
    print("Done!")
