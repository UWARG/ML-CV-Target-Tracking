"""
Accuracy recording script for OAK-D depth calibration.

Controls:
  Press 1 → 5s countdown then record 10s at 0.5m
  Press 2 → 5s countdown then record 10s at 1.0m
  Press 3 → 5s countdown then record 10s at 1.5m
  Press 4 → 5s countdown then record 10s at 2.0m
  Press 5 → 5s countdown then record 10s at 3.0m
  Press s → stop recording early
  Press q → quit
"""

import csv
import pathlib
import time

import cv2
import depthai as dai
import yaml

from modules.target_tracking.stereo_node import create_stereo_depth
from modules.target_tracking.spatial_detection_node import create_spatial_detection_network
from modules.target_tracking.object_tracker_node import create_object_tracker

REPO_DIR = pathlib.Path(__file__).parent
CONFIG_FILE_PATH = REPO_DIR / "config.yaml"
LOG_FILE = REPO_DIR / "accuracy_log.csv"
OUTPUT_QUEUE_SIZE = 4
COUNTDOWN_SECS = 5
RECORD_SECS = 10

DISTANCE_KEYS = {
    ord("1"): "0.5m",
    ord("2"): "1.0m",
    ord("3"): "1.5m",
    ord("4"): "2.0m",
    ord("5"): "3.0m",
}


def main() -> None:
    """Run the recording pipeline."""
    with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_name: str = config["spatial_detection"]["model_name"]

    write_header = not LOG_FILE.exists() or LOG_FILE.stat().st_size == 0
    log_file = open(LOG_FILE, "a", newline="", encoding="utf-8")  # noqa: SIM115
    writer = csv.writer(log_file)
    if write_header:
        writer.writerow(["timestamp", "test_distance", "target_id", "x_mm", "y_mm", "z_mm"])

    active_distance: str = ""
    countdown_until: float = 0.0
    countdown_target: str = ""
    logging_until: float = 0.0

    print("Controls: [1]=0.5m  [2]=1.0m  [3]=1.5m  [4]=2.0m  [5]=3.0m  [s]=stop  [q]=quit")

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
            now = time.time()

            if countdown_until and now >= countdown_until:
                active_distance = countdown_target
                logging_until = now + RECORD_SECS
                countdown_until = 0.0
                countdown_target = ""
                print(f"\n--- Recording {active_distance} for {RECORD_SECS}s ---")

            if active_distance and now >= logging_until:
                print(f"\n--- Done recording {active_distance} ---")
                active_distance = ""
                logging_until = 0.0

            for tracklet in tracklets_msg.tracklets:
                if tracklet.status != dai.Tracklet.TrackingStatus.TRACKED:
                    continue

                roi = tracklet.roi.denormalize(frame.shape[1], frame.shape[0])
                x_mm = tracklet.spatialCoordinates.x
                y_mm = tracklet.spatialCoordinates.y
                z_mm = tracklet.spatialCoordinates.z

                if active_distance:
                    writer.writerow(
                        [
                            round(now, 3),
                            active_distance,
                            tracklet.id,
                            round(x_mm),
                            round(y_mm),
                            round(z_mm),
                        ]
                    )
                    log_file.flush()

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

            if countdown_until:
                secs_left = max(0, int(countdown_until - now) + 1)
                label = f"Get ready ({countdown_target})... {secs_left}s"
                color = (0, 140, 255)
            elif active_distance:
                secs_left = max(0, int(logging_until - now) + 1)
                label = f"Recording {active_distance}... {secs_left}s"
                color = (0, 200, 100)
            else:
                label = "Idle  [1]=0.5m [2]=1.0m [3]=1.5m [4]=2.0m [5]=3.0m"
                color = (160, 160, 160)
            cv2.putText(frame, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("Record Accuracy", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key in DISTANCE_KEYS:
                countdown_until = time.time() + COUNTDOWN_SECS
                countdown_target = DISTANCE_KEYS[key]
                active_distance = ""
                print(f"\n--- {COUNTDOWN_SECS}s countdown — stand at {DISTANCE_KEYS[key]} ---")
            elif key == ord("s"):
                print(f"\n--- Stopped ({active_distance or countdown_target}) ---")
                active_distance = ""
                countdown_until = 0.0
                countdown_target = ""

    log_file.close()
    print(f"Saved to {LOG_FILE}")


if __name__ == "__main__":
    main()
