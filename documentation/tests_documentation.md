# Tests Documentation

This document provides a comprehensive overview of all test files and individual test cases within the `tests/` and `modules/common/tests/` directories, describing their purpose and functionality.

---

## 1. Object Tracking & Control Unit Tests (`tests/unit/`)

### [test_object_tracker.py](../../tests/unit/test_object_tracker.py)
Tests the parsing of raw hardware tracklets into `TrackedObject` formats.
- **`test_parse_tracklets_valid_data`**: Verifies that raw hardware tracklets are correctly parsed into `TrackedObject`s.
- **`test_parse_tracklets_ignores_removed`**: Verifies that tracklets flagged as `REMOVED` by the hardware are safely skipped.
- **`test_parse_tracklets_unknown_label_fallback`**: Ensures that, if the hardware returns a label index outside the predefined map, it gracefully falls back to the stringified index representation.

### [test_object_tracker_worker.py](../../tests/unit/test_object_tracker_worker.py)
- **`test_worker_read_loop_data_flow`**: Verifies that the worker loop correctly accesses and moves data from the hardware input queue downstream to the output queue.

### [test_sitl_connection.py](../../tests/unit/test_sitl_connection.py)
Tests Software-In-The-Loop (SITL) operational connectivity.
- **`test_connection`**: Basic connection verification to ensure the automated system can communicate correctly to the virtual SITL environment.

### [test_software_tracker.py](../../tests/unit/test_software_tracker.py)
Evaluates logic to bridge, sustain, or expire target tracks.
- **`test_tracker_initialization`**: Verifies the software tracker initializes properly with an empty state.
- **`test_new_track_creation`**: Validates a new valid detection spawns a new tracking instance with status `NEW`.
- **`test_track_persistence_and_smoothing`**: Checks that a heavily overlapping detection in a sequential frame adequately updates the pre-existing track ID.
- **`test_track_lost_and_removed`**: Ensures a track appropriately becomes `LOST` if missed, and eventually is removed if lost for too long.
- **`test_id_contention_highest_iou_wins`**: Given two overlapping detections against a known track, enforces that the detection providing the highest Intersection over Union (IoU) rightfully claims the ID.
- **`test_sub_threshold_iou_spawns_new_track`**: Validates a detection with an IoU falling lower than the baseline threshold does not falsely match, effectively defaulting to spawning a new distinct track.

### [test_velocity_control.py](../../tests/unit/test_velocity_control.py)
Script to evaluate velocity command formulation and output against targets.
- **Integration Function**: Includes primary methodologies (`send_velocity_command` and `calculate_velocity_commands`) utilizing simple proportional P-controllers. Converts target errors to velocities and leverages `pymavlink` via local NED frames to actuate tracking directives safely.

---

## 2. Common Modules Unit Tests (`modules/common/tests/unit/`)

### Data Encoding & Decoding
* **[test_message_encoding_decoding.py](../../modules/common/tests/unit/data_encoding/test_message_encoding_decoding.py)**
  - **`test_encoding_decoding`**: Function to verify encode and decode workflows for system messaging.
* **[test_metadata_encoding_decoding.py](../../modules/common/tests/unit/data_encoding/test_metadata_encoding_decoding.py)**
  - **`test_encoding_metadata`**: Function to check if image/payload metadata operates symmetrically via encode/decode workflows.
* **[test_image_encode_decode.py](../../modules/common/tests/unit/test_image_encode_decode.py)**
  - **`test_image_encode_decode`**: Primary testing sequence compressing and decompressing images. Accommodates the understanding that JPEG encoding holds lossy properties.

### GPS, KML, and Conversion Systems
* **[test_kml_conversion.py](../../modules/common/tests/unit/test_kml_conversion.py)**
  Tests processes saving locational metadata to KML configurations.
  - **`test_with_save_path`**: Asserts files save properly assuming correct validation path.
  - **`test_nonexistent_save_path`**: Simulates and analyzes saving files under uncreated directories.
  - **`test_no_locations`**: Validates script execution on Empty geographical lists.
  - **`test_named_locations` / `test_locations`**: Sub-checks for named lists vs pure coordinate entries.
* **[test_local_global_conversion.py](../../modules/common/tests/unit/test_local_global_conversion.py)**
  Tests internal invocation methods for `DroneOdometry`, location mappings, and coordinates.
  - Test suites covering `test_position_global_from_position_local`, `test_drone_odometry_local_from_global`, `test_position_local_from_position_global`, etc. assure coordinate conversions are properly dispatched.

### Telemetry Network Endpoints (`network/`)
* **[test_send_image.py](../../modules/common/tests/unit/network/test_send_image.py)**: Integration test combining image encoding sequences, sending over internal network topologies (TCP/UDP).
* **[test_tcp.py](../../modules/common/tests/unit/network/test_tcp.py)** / **[test_udp.py](../../modules/common/tests/unit/network/test_udp.py)**: Verify message payloads sending to dummy echo servers accurately over expected sockets.

### Hardware & Operations Output
* **[test_logger.py](../../modules/common/tests/unit/test_logger.py)**
  Standardizes debugging processes across the software system.
  - Asserts logging functions work universally across varying output targets (`test_log_to_file`, `test_debug_log_info_to_stdout`, etc.).
  - Evaluates capturing video/image frames inside logs (`test_log_with_frame_info`).
* **[test_qr.py](../../modules/common/tests/unit/test_qr.py)**
  Tests fundamental QR pattern scanner implementations matching historical metrics (e.g. `test_2023_task1_route`, `test_2023_task2_routes`).
* **[test_read_yaml.py](../../modules/common/tests/unit/test_read_yaml.py)**
  - Tests whether the global `read_yaml` configurator accesses configuration items or safely flags `test_open_config_file_not_found`.
* **[test_hitl_inject_position.py](../../modules/common/tests/unit/test_hitl_inject_position.py)**
  - Tests injecting random coordinate streams effectively into the Hardware-In-The-Loop (HITL) setup loop printed functionally to console mechanisms.

---

## 3. Common Modules Integration Tests (`modules/common/tests/integration/`)
These files generally serve as standalone test scripts verifying hardware endpoints or robust integration paths.

### Cameras & Vision Testing
* **[test_camera_arducamir.py](../../modules/common/tests/integration/test_camera_arducamir.py)**: Physically tests an ArducamIR configuration.
* **[test_camera_opencv.py](../../modules/common/tests/integration/test_camera_opencv.py)**: Verifies the default generic OpenCV optical camera outputs.
* **[test_camera_picamera2.py](../../modules/common/tests/integration/test_camera_picamera2.py)**: Checks native Raspberry Pi `picamera2` environment pipelines.
* **[test_camera_qr_example.py](../../modules/common/tests/integration/test_camera_qr_example.py)**: Evaluates a dynamic end-to-end QR code reading using localized vision setups.

### Emulator & Hardware in The Loop (HITL)
* **[test_camera_emulator.py](../../modules/common/tests/integration/camera_emulator/test_camera_emulator.py)** & **[test_camera_read.py](../../modules/common/tests/integration/camera_emulator/test_camera_read.py)**: Streamlines tests evaluating virtual Windows OBS stream capabilities injecting feed streams acting as mock image sensors.
* **[test_hitl_json_parser.py](../../modules/common/tests/integration/hitl/test_hitl_json_parser.py)**: Parses pre-recorded coordinate maps continuously demonstrating the position emulator capabilities against real targets.
* **[test_threading.py](../../modules/common/tests/integration/hitl/test_threading.py)**: Assesses threading concurrency overlaps validating safe runtime threading across mock GPS updates and local virtual cameras.

### Flight Controllers
* **[test_flight_controller.py](../../modules/common/tests/integration/test_flight_controller.py)**: Standard connectivity test fetching device outputs straight to diagnostic outputs.
* **[test_flight_controller_mission_ended.py](../../modules/common/tests/integration/test_flight_controller_mission_ended.py)**: Evaluates autopilot waypointing metrics testing proper flight termination signals when a drone achieves its end destination.
* **[test_send_messages.py](../../modules/common/tests/integration/test_send_messages.py)**: Checks end-to-end capabilities transmitting arbitrary command messages actively to the established drone endpoint via the FlightController class.
