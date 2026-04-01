"""
Target tracking pipeline for OAK-D.
Runs YOLOv4-tiny spatial detection + object tracking on-device.
"""

import pathlib

import cv2
import depthai as dai
import yaml

from modules.target_tracking.stereo_node import create_stereo_depth
from modules.target_tracking.spatial_detection_node import create_spatial_detection_network
from modules.target_tracking.object_tracker_node import create_object_tracker


CONFIG_FILE_PATH = pathlib.Path("config.yaml")
OUTPUT_QUEUE_SIZE = 4


def main() -> int:
    with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    model_name: str = config["spatial_detection"]["model_name"]

    with dai.Pipeline() as pipeline:
        stereo = create_stereo_depth(pipeline)
        spatial_detection = create_spatial_detection_network(pipeline, stereo, model_name)
        tracker = create_object_tracker(pipeline, spatial_detection)

        tracklet_queue = tracker.out.createOutputQueue(maxSize=OUTPUT_QUEUE_SIZE, blocking=False)
        preview_queue = tracker.passthroughTrackerFrame.createOutputQueue(
            maxSize=OUTPUT_QUEUE_SIZE, blocking=False
        )

        pipeline.start()
        while pipeline.isRunning():
            tracklets_msg = tracklet_queue.get()
            frame_msg = preview_queue.get()
            frame = frame_msg.getCvFrame()

            for tracklet in tracklets_msg.tracklets:
                if tracklet.status != dai.Tracklet.TrackingStatus.TRACKED:
                    continue

                roi = tracklet.roi.denormalize(frame.shape[1], frame.shape[0])
                x_mm = tracklet.spatialCoordinates.x
                y_mm = tracklet.spatialCoordinates.y
                z_mm = tracklet.spatialCoordinates.z

                print(
                    f"Target ID {tracklet.id}: "
                    f"xyz=({x_mm:.0f}mm, {y_mm:.0f}mm, {z_mm:.0f}mm)  "
                    f"bbox=({int(roi.topLeft().x)}, {int(roi.topLeft().y)}, "
                    f"{int(roi.bottomRight().x)}, {int(roi.bottomRight().y)})"
                )

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

            cv2.imshow("Target Tracking", frame)
            if cv2.waitKey(1) == ord("q"):
                break

    return 0


if __name__ == "__main__":
    result_main = main()
    if result_main < 0:
        print(f"ERROR: Status code: {result_main}")
    print("Done!")
