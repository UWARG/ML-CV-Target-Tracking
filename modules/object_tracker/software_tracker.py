"""
Software-based object tracker that accepts Detection objects.

Use this when:
- Detection team outputs Python Detection objects (not on-device pipeline)
- You want to run tracking on the host rather than OAK-D device

This implements simple IoU-based tracking with ID persistence and
coordinate smoothing via exponential moving average.
"""

import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from .detection import Detection
from .tracked_object import TrackedObject, TrackingStatus


@dataclass
class _TrackedState:
    """Internal state for a tracked object."""

    object_id: int
    label: str
    confidence: float
    x: float
    y: float
    z: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    last_seen: float
    frames_tracked: int = 1
    status: TrackingStatus = TrackingStatus.NEW


class SoftwareTracker:
    """
    Host-side object tracker with ID assignment and coordinate smoothing.

    Accepts Detection objects from the detection team and outputs
    TrackedObject with persistent IDs and smoothed coordinates.

    Usage::

        tracker = SoftwareTracker()

        # Each frame:
        detections = [Detection(...), Detection(...)]
        tracked = tracker.update(detections)
        # tracked is List[TrackedObject]
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_lost_frames: float = 0.5,  # seconds
        smoothing_alpha: float = 0.4,
    ) -> None:
        """
        Args:
            iou_threshold: Min IoU to match detection to existing track.
            max_lost_frames: Seconds before a lost object is removed.
            smoothing_alpha: EMA alpha for coordinate smoothing (0-1).
                Higher = more weight on new detections.
        """
        self._iou_threshold = iou_threshold
        self._max_lost_time = max_lost_frames
        self._alpha = smoothing_alpha
        self._next_id = 1
        self._tracks: Dict[int, _TrackedState] = {}

    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        """
        Process a new frame of detections.

        Args:
            detections: List of Detection objects from this frame.

        Returns:
            List of TrackedObject with persistent IDs and smoothed coords.
        """
        now = time.time()
        matched_track_ids = set()
        results: List[TrackedObject] = []

        # --- match detections to existing tracks ---
        for det in detections:
            best_id = self._find_best_match(det)

            if best_id is not None:
                # update existing track
                track = self._tracks[best_id]
                self._update_track(track, det, now)
                matched_track_ids.add(best_id)
            else:
                # create new track
                new_id = self._next_id
                self._next_id += 1
                self._tracks[new_id] = _TrackedState(
                    object_id=new_id,
                    label=det.label,
                    confidence=det.confidence,
                    x=det.x,
                    y=det.y,
                    z=det.z,
                    xmin=det.xmin,
                    ymin=det.ymin,
                    xmax=det.xmax,
                    ymax=det.ymax,
                    last_seen=now,
                    status=TrackingStatus.NEW,
                )
                matched_track_ids.add(new_id)

        # --- mark unmatched tracks as LOST, remove old ones ---
        to_remove = []
        for track_id, track in self._tracks.items():
            if track_id not in matched_track_ids:
                track.status = TrackingStatus.LOST
                if now - track.last_seen > self._max_lost_time:
                    to_remove.append(track_id)

        for track_id in to_remove:
            del self._tracks[track_id]

        # --- convert to TrackedObject output ---
        for track in self._tracks.values():
            results.append(
                TrackedObject(
                    object_id=track.object_id,
                    status=track.status,
                    label=track.label,
                    confidence=track.confidence,
                    x=track.x,
                    y=track.y,
                    z=track.z,
                    bbox_x=int(track.xmin),
                    bbox_y=int(track.ymin),
                    bbox_width=int(track.xmax - track.xmin),
                    bbox_height=int(track.ymax - track.ymin),
                )
            )

        return results

    def _find_best_match(self, det: Detection) -> Optional[int]:
        """Find the track with highest IoU above threshold."""
        best_id = None
        best_iou = self._iou_threshold

        for track_id, track in self._tracks.items():
            iou = self._compute_iou(
                det.xmin, det.ymin, det.xmax, det.ymax,
                track.xmin, track.ymin, track.xmax, track.ymax,
            )
            if iou > best_iou:
                best_iou = iou
                best_id = track_id

        return best_id

    def _update_track(
        self,
        track: _TrackedState,
        det: Detection,
        now: float,
    ) -> None:
        """Update track with new detection, applying EMA smoothing."""
        a = self._alpha

        # smooth spatial coords
        track.x = a * det.x + (1 - a) * track.x
        track.y = a * det.y + (1 - a) * track.y
        track.z = a * det.z + (1 - a) * track.z

        # smooth bbox
        track.xmin = a * det.xmin + (1 - a) * track.xmin
        track.ymin = a * det.ymin + (1 - a) * track.ymin
        track.xmax = a * det.xmax + (1 - a) * track.xmax
        track.ymax = a * det.ymax + (1 - a) * track.ymax

        # update metadata
        track.label = det.label
        track.confidence = det.confidence
        track.last_seen = now
        track.frames_tracked += 1
        track.status = TrackingStatus.TRACKED

    @staticmethod
    def _compute_iou(
        x1min: float, y1min: float, x1max: float, y1max: float,
        x2min: float, y2min: float, x2max: float, y2max: float,
    ) -> float:
        """Compute Intersection over Union of two bounding boxes."""
        xi_min = max(x1min, x2min)
        yi_min = max(y1min, y2min)
        xi_max = min(x1max, x2max)
        yi_max = min(y1max, y2max)

        if xi_max <= xi_min or yi_max <= yi_min:
            return 0.0

        inter_area = (xi_max - xi_min) * (yi_max - yi_min)
        box1_area = (x1max - x1min) * (y1max - y1min)
        box2_area = (x2max - x2min) * (y2max - y2min)
        union_area = box1_area + box2_area - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area
