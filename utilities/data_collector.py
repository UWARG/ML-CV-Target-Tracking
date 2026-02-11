"""
Data Collection Utility for OAK-D
Captures synchronized Left, Right, and RGB images for stereo depth calibration and testing.

Usage:
    python utilities/data_collector.py --interval 2 --out dataset_01
"""

import cv2
import depthai as dai
import time
import os
import argparse
import shutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--interval", type=float, default=3.0, help="Time in seconds between captures"
    )
    parser.add_argument("-o", "--out", type=str, default="dataset", help="Output directory path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Create unique session folder to prevent overwriting
    timestamp_start = int(time.time())
    session_dir = f"{args.out}_{timestamp_start}"

    for stream in ["left", "right", "rgb"]:
        path = os.path.join(session_dir, stream)
        os.makedirs(path, exist_ok=True)

    print(f"Starting Data Collector... Saving to '{session_dir}'")

    # In depthai v3.x, device is created first
    device = dai.Device()

    with dai.Pipeline(device) as pipeline:
        # Define Camera Sources using the new Camera node
        camLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)  # Left mono
        camRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)  # Right mono
        camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)  # RGB center

        # Request outputs and create output queues
        qLeft = camLeft.requestOutput((640, 400), dai.ImgFrame.Type.GRAY8).createOutputQueue()
        qRight = camRight.requestOutput((640, 400), dai.ImgFrame.Type.GRAY8).createOutputQueue()
        qRgb = camRgb.requestOutput((640, 400), dai.ImgFrame.Type.BGR888p).createOutputQueue()

        pipeline.start()

        count = 0
        last_capture_time = time.time()
        last_capture_display_time = 0  # Track when to show "CAPTURED!" message

        # Store the latest frames from each camera
        latestLeft = None
        latestRight = None
        latestRgb = None

        while pipeline.isRunning():
            # Non-blocking calls to get frames - update latest if available
            inLeft = qLeft.tryGet()
            inRight = qRight.tryGet()
            inRgb = qRgb.tryGet()

            # Update latest frames when new ones arrive
            if inLeft:
                latestLeft = inLeft
            if inRight:
                latestRight = inRight
            if inRgb:
                latestRgb = inRgb

            if latestRgb:
                frameRgb = latestRgb.getCvFrame().copy()  # Copy for drawing

                # --- VISUAL COUNTDOWN LOGIC ---
                time_since_last = time.time() - last_capture_time
                time_remaining = args.interval - time_since_last
                time_since_capture_display = time.time() - last_capture_display_time

                if time_since_capture_display < 0.5:
                    # Show "CAPTURED!" for 0.5 seconds after capture
                    text = "CAPTURED!"
                    color = (0, 255, 0)  # Green
                elif time_remaining > 0:
                    # Draw countdown on screen
                    text = f"Capture in: {int(time_remaining) + 1}"
                    color = (0, 255, 255)  # Yellow
                else:
                    text = "CAPTURING..."
                    color = (0, 0, 255)  # Red

                cv2.putText(frameRgb, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.imshow("Data Collector (Preview)", frameRgb)

            # Check if we have frames from all cameras AND it's time to capture
            if (
                latestLeft
                and latestRight
                and latestRgb
                and (time.time() - last_capture_time > args.interval)
            ):
                timestamp = int(time.time() * 1000)

                # Save frames
                cv2.imwrite(f"{session_dir}/left/{timestamp}.png", latestLeft.getCvFrame())
                cv2.imwrite(f"{session_dir}/right/{timestamp}.png", latestRight.getCvFrame())
                cv2.imwrite(f"{session_dir}/rgb/{timestamp}.png", latestRgb.getCvFrame())

                print(f"[{count}] Saved set at {timestamp}ms")
                count += 1
                last_capture_time = time.time()  # Reset timer
                last_capture_display_time = time.time()  # Trigger "CAPTURED!" display

            key = cv2.waitKey(10)  # Increased to 10ms for better GUI responsiveness
            if key == ord("q"):
                break

    print("Data collection finished.")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
