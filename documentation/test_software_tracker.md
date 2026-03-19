# SoftwareTracker Unit Tests & Bug Report

## Overview
This document details the unit testing suite created for the `SoftwareTracker` (`modules/object_tracker/software_tracker.py`) and outlines a critical logic bug discovered during edge-case testing.

## Test Suite Implementation (`test_software_tracker.py`)
A new unit test file was created to verify the state transitions of the software tracker. The tests validate the following expected behaviors:
* **Initialization:** Verifies the tracker starts with an empty state and an ID counter at 1.
* **Track Creation:** Ensures a new, unmatched detection correctly spawns a `TrackedObject` with `TrackingStatus.NEW`.
* **Track Persistence & Smoothing:** Confirms that a matching detection in a subsequent frame upgrades the status to `TrackingStatus.TRACKED` and correctly applies the Exponential Moving Average (EMA) to smooth the spatial coordinates and bounding box.
* **Track Expiration:** Validates that tracks without matching detections transition to `TrackingStatus.LOST` and are completely purged from memory after `max_lost_frames` is exceeded.

## The Bug: ID Contention (The "Crossing Paths" Problem)
While the tracker handles isolated, sequential detections perfectly, it fails when processing ambiguous detections (e.g., when two objects cross paths). 

A test case (`test_id_contention_highest_iou_wins`) was written to simulate two distinct detections in the current frame overlapping with a single tracked object from the previous frame. 

**Expected Behavior:** The detection with the highest Intersection over Union (IoU) should claim the existing track ID. The second detection (the "loser") should be forced to spawn a new track ID.
**Actual Behavior:** The tracker assigned *both* detections to the same ID sequentially. The test resulted in `len(results) == 1` instead of `2`.

## Root Cause Analysis
The bug stems from how detections are matched to existing tracks in `software_tracker.py`. 

Currently, the `update()` method iterates through each detection and calls `_find_best_match()` to find the highest IoU track. However, there is no mechanism to "lock" a track once it has been claimed. 
1. The tracker processes `Detection A`, finds it overlaps with `Track 1`, and updates `Track 1`'s state.
2. The tracker immediately processes `Detection B`, sees it *also* overlaps with `Track 1`, and updates `Track 1` again, entirely overwriting the data from `Detection A`.

Because the loop evaluates detections independently rather than globally, multiple detections can hijack the same track ID in a single frame, causing the tracker to permanently lose tracking data for the overwritten objects.

## Proposed Fix
The `_find_best_match` logic needs to be replaced with a **Global Greedy Matching** algorithm. 
1. Compute the IoU for all possible detection-track pairs.
2. Sort these pairs by highest IoU.
3. Assign matches greedily, ensuring that once a track or a detection is claimed in a frame, it is removed from the pool of available options.