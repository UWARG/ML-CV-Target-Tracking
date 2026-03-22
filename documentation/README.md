# Unit Tests Documentation

This directory contains the unit tests for the ML-CV-Target-Tracking project, focusing on isolated testing of specific modules and components. 

The tests are written using the `pytest` framework and utilize `unittest.mock` for isolating components from hardware interfaces and external dependencies.

## Structure

### 1. Object Tracker Worker (`test_object_tracker_worker.py`)
Tests the core data flow of the hardware tracking thread.
- **`test_worker_read_loop_data_flow`**: Validates the `object_tracker_read_loop` function. Ensures that data correctly shifts from the hardware queue, goes through the tracklets parsing logic, and is successfully placed into the generic software output queue. It uses mock devices and exceptions (`StopLoopException`) to test iteration control safely.

### 2. Object Tracker (`test_object_tracker.py`)
Tests the parsing and translation of raw hardware tracking data into the system's generalized `TrackedObject` format.
- **`test_parse_tracklets_valid_data`**: Verifies that successful hardware tracklets correctly translate millimeter coordinates into meters and denormalize bounding box Regions of Interest (ROI) into correct pixel dimensions.
- **`test_parse_tracklets_ignores_removed`**: Ensures that detections flagged with a `REMOVED` status by the hardware are filtered out of the active tracklets payload.
- **`test_parse_tracklets_unknown_label_fallback`**: Determines that bounding labels falling outside the defined label map boundaries default reliably to the stringified integer index.

### 3. Software Tracker (`test_software_tracker.py`)
Validates the fallback custom software-based tracking algorithm, which stitches independent frames of raw detections into continuous, ID-persistent tracked objects.
- **`test_tracker_initialization`**: Verifies the tracker starts correctly with a clean slate and resets ID counters.
- **`test_new_track_creation`**: Evaluates that a brand-new, unseen detection triggers a brand-new track with a `NEW` tracking status.
- **`test_track_persistence_and_smoothing`**: Tests exponential smoothing logic (`alpha` parameter), verifying that bounding boxes and coordinates calculate partial adjustments seamlessly between contiguous frames.
- **`test_track_lost_and_removed`**: Simulates temporal tracking loss, checking that missed detections transition tracks to `LOST` status, and eventually drop out of memory entirely upon exceeding the `max_lost_frames` duration tolerance.
- **`test_id_contention_highest_iou_wins`**: In scenarios where multiple detections overlap an ongoing track, this guarantees that the highest Intersection over Union (IoU) claims the existing ID.
- **`test_sub_threshold_iou_spawns_new_track`**: Validates the `iou_threshold` strictness parameters; weak overlaps must not mistakenly hijack existing identities, but must spawn entirely new tracks instead.

## Running the Unit Tests

You can execute the unit tests from the workspace root by using `pytest`:

```bash
# Run all unit tests
pytest tests/unit/

# Run tests with verbose output
pytest -vv tests/unit/
```
