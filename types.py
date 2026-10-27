from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class Detection:
    box: tuple[float, float, float, float]
    confidence: float
    head_offset: Optional[float] = None
    head_confidence: float = 0.0
    phone_confidence: float = 0.0

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass
class Person:
    track_id: int
    detection: Detection
    speed: float = 0.0  # displacement / person-box diagonal / media second


@dataclass
class Seat:
    seat_id: str
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self):
        values = (self.x, self.y, self.width, self.height)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Seat coordinates must be finite.")
        if not (0 <= self.x <= 1 and 0 <= self.y <= 1 and 0 < self.width <= 1 and 0 < self.height <= 1):
            raise ValueError("Seat centres must be 0..1 and dimensions must be >0..1.")
        if not self.seat_id.strip():
            raise ValueError("Each seat needs an ID.")

    def contains(self, point, margin=0.0):
        return (abs(point[0] - self.x) <= self.width * (0.5 + margin)
                and abs(point[1] - self.y) <= self.height * (0.5 + margin))


@dataclass
class Config:
    sample_fps: float = 5.0
    min_person_conf: float = 0.35
    min_phone_conf: float = 0.30
    head_delta: float = 0.28
    head_count: int = 3
    window: float = 20.0
    leave_seconds: float = 3.0
    phone_seconds: float = 2.0
    group_seconds: float = 3.0
    movement_seconds: float = 2.0
    speed_threshold: float = 0.70
    group_radius: float = 1.30  # in average person-box diagonals
    group_min_people: int = 3
    recovery_half_life: float = 8.0
    log_interval: float = 5.0
    max_gap: float = 0.8
    calibration_seconds: float = 2.0

    def __post_init__(self):
        if self.sample_fps <= 0 or self.window <= 0 or self.head_count < 1:
            raise ValueError("Sampling rate/window/count must be positive.")
        if min(self.leave_seconds, self.phone_seconds, self.group_seconds,
               self.movement_seconds, self.recovery_half_life) <= 0:
            raise ValueError("Duration thresholds and half-life must be positive.")


@dataclass
class Event:
    timestamp: float
    track_id: int
    seat_id: str
    event_type: str
    confidence: float
    duration: float
    frequency: int
    risk_score: float
    risk_level: str
    agent_state: str
    explanation: str
    evidence_frame: str = ""
    source_mode: str = ""


def risk_level(score):
    return "HIGH" if score >= 65 else "MEDIUM" if score >= 35 else "LOW"
