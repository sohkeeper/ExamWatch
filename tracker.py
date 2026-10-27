"""Small motion tracker: predicted centres + IoU + linear-sum assignment.

It is intentionally not facial recognition or long-term person re-identification.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from .types import Detection, Person


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


@dataclass
class Track:
    detection: Detection
    time: float
    vx: float = 0.0
    vy: float = 0.0


class MotionTracker:
    def __init__(self, max_missing=2.0):
        self.tracks = {}
        self.next_id = 1
        self.max_missing = max_missing

    def update(self, detections, timestamp):
        self.tracks = {k: t for k, t in self.tracks.items()
                       if timestamp - t.time <= self.max_missing}
        ids = list(self.tracks)
        costs = np.full((len(ids), len(detections)), 1e6)
        for r, tid in enumerate(ids):
            track = self.tracks[tid]
            dt = max(0, timestamp - track.time)
            cx, cy = track.detection.center
            predicted = (cx + track.vx * dt, cy + track.vy * dt)
            box = track.detection.box
            scale = max(1, math.hypot(box[2] - box[0], box[3] - box[1]))
            for c, det in enumerate(detections):
                distance = math.dist(predicted, det.center) / scale
                overlap = iou(box, det.box)
                if distance < 1.2 or overlap > 0.05:
                    costs[r, c] = 0.65 * distance + 0.35 * (1 - overlap)
        matched = {}
        if costs.size:
            rows, cols = linear_sum_assignment(costs)
            matched = {int(c): ids[int(r)] for r, c in zip(rows, cols) if costs[r, c] < 1.2}
        people = []
        for index, det in enumerate(detections):
            tid = matched.get(index)
            speed = 0.0
            if tid is None:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = Track(det, timestamp)
            else:
                old = self.tracks[tid]
                dt = max(1e-6, timestamp - old.time)
                dx, dy = det.center[0] - old.detection.center[0], det.center[1] - old.detection.center[1]
                diagonal = max(1, math.hypot(det.box[2] - det.box[0], det.box[3] - det.box[1]))
                speed = math.hypot(dx, dy) / dt / diagonal
                self.tracks[tid] = Track(det, timestamp, 0.6 * old.vx + 0.4 * dx / dt,
                                         0.6 * old.vy + 0.4 * dy / dt)
            people.append(Person(tid, det, speed))
        return people
