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

# Z-bias calibration anchors: (raw_z_mm, offset_mm_to_subtract).
# Measured at 0.5/1.0/1.5/2.0m; final (2200, 0) tapers smoothly to factory calibration.
# Beyond the last anchor the raw camera value is trusted as-is.
# See documentation/accuracy/calibration.png for the fit visualization.
Z_CALIBRATION_ANCHORS = (
    (527.0, 27.5),  # 0.5m
    (1075.0, 75.1),  # 1.0m
    (1573.0, 73.2),  # 1.5m
    (1951.0, -48.7),  # 2.0m
    (2200.0, 0.0),  # taper end — trust factory beyond this
)


def calibrate_z(raw_z: float) -> float:
    """Apply piecewise-linear bias correction to a raw stereo-depth z value (mm)."""
    if raw_z <= Z_CALIBRATION_ANCHORS[0][0]:
        return raw_z - Z_CALIBRATION_ANCHORS[0][1]
    if raw_z >= Z_CALIBRATION_ANCHORS[-1][0]:
        return raw_z
    for (z0, o0), (z1, o1) in zip(Z_CALIBRATION_ANCHORS, Z_CALIBRATION_ANCHORS[1:]):
        if z0 <= raw_z <= z1:
            t = (raw_z - z0) / (z1 - z0)
            return raw_z - (o0 + t * (o1 - o0))
    return raw_z


def main() -> int:
    """Run the OAK-D target tracking pipeline."""
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
                z_mm = calibrate_z(tracklet.spatialCoordinates.z)

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
