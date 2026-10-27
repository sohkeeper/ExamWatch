"""Stateful rules using VIDEO time, episodes, confidence, memory and decay."""
from collections import deque
from dataclasses import dataclass, field
import math
import statistics
from .types import Config, Event, risk_level

LABELS = {
    "SEAT_LEAVE": "Outside assigned seat region",
    "HEAD_TURN": "Repeated side-turn proxy",
    "PHONE": "Phone-like object associated with track",
    "GATHERING": "Nearby group of at least three people",
    "MOVEMENT": "Sustained rapid displacement",
}
WEIGHTS = {"SEAT_LEAVE": 0.95, "HEAD_TURN": 0.90, "PHONE": 1.0,
           "GATHERING": 0.85, "MOVEMENT": 0.75}
STATES = ["NORMAL", "OBSERVE", "SUSPICIOUS", "ALERT"]


@dataclass
class SignalMemory:
    samples: deque = field(default_factory=deque)
    starts: deque = field(default_factory=deque)
    active: bool = False
    start: float = 0.0
    last_positive: float = -1e9
    last_logged: float = -1e9
    last_level: str = ""

    def update(self, active, confidence, timestamp, dt, window):
        # A new occurrence requires a neutral interval; frames do not count as turns.
        if active:
            if not self.active:
                self.start = timestamp
                self.starts.append(timestamp)
            self.samples.append((timestamp, dt if self.active else 0.0, confidence))
            self.active = True
            self.last_positive = timestamp
        elif timestamp - self.last_positive >= 0.6:
            self.active = False
        while self.samples and self.samples[0][0] < timestamp - window:
            self.samples.popleft()
        while self.starts and self.starts[0] < timestamp - window:
            self.starts.popleft()

    @property
    def confidence(self):
        return statistics.mean(s[2] for s in self.samples) if self.samples else 0.0


@dataclass
class StudentMemory:
    track_id: int
    first_seen: float
    last_seen: float
    last_updated: float = 0.0
    seat_id: str = "UNASSIGNED"
    risk: float = 0.0
    state: str = "NORMAL"
    head_samples: list = field(default_factory=list)
    head_baseline: float | None = None
    signals: dict = field(default_factory=lambda: {key: SignalMemory() for key in LABELS})


class ModelBasedAgent:
    def __init__(self, config=None, seats=None, ignored_ids=None):
        self.config = config or Config()
        self.seats = {s.seat_id: s for s in (seats or [])}
        if len(self.seats) != len(seats or []):
            raise ValueError("Seat IDs must be unique.")
        self.students = {}
        self.ignored_ids = set(ignored_ids or [])
        self.last_timestamp = -1.0
        self.pair_baselines = {}

    def _set_state(self, memory):
        target = 3 if memory.risk >= 65 else 2 if memory.risk >= 35 else 1 if memory.risk >= 15 else 0
        previous = STATES.index(memory.state)
        # Walk through the state ladder on escalation; decay can return to normal.
        memory.state = STATES[min(target, previous + 1)] if target > previous else STATES[target]

    def process(self, people, timestamp, frame_size):
        if timestamp <= self.last_timestamp:
            raise ValueError("Media timestamps must increase monotonically.")
        self.last_timestamp = timestamp
        cfg = self.config
        people = [p for p in people if p.track_id not in self.ignored_ids]
        seen = {p.track_id for p in people}
        width, height = frame_size
        if width <= 0 or height <= 0:
            raise ValueError("Video dimensions must be positive.")
        events = []
        # Initial inter-person separation is part of the world model. Normal,
        # closely seated students must not automatically count as a gathering.
        for i, first in enumerate(people):
            for second in people[i + 1:]:
                key = tuple(sorted((first.track_id, second.track_id)))
                if key not in self.pair_baselines:
                    self.pair_baselines[key] = math.dist(first.detection.center, second.detection.center)
        for tid, memory in self.students.items():
            if tid not in seen:
                dt = max(0, timestamp - memory.last_updated)
                memory.risk *= 0.5 ** (dt / cfg.recovery_half_life)
                memory.last_updated = timestamp
                for signal in memory.signals.values():
                    signal.active = False  # Missing video is not positive evidence.
                self._set_state(memory)

        for person in people:
            tid, detection = person.track_id, person.detection
            memory = self.students.get(tid)
            if memory is None:
                memory = StudentMemory(tid, timestamp, timestamp)
                self.students[tid] = memory
            actual_dt = max(0, timestamp - memory.last_seen)
            decay_dt = max(0, timestamp - memory.last_updated)
            dt = actual_dt if actual_dt <= cfg.max_gap else 0.0
            if actual_dt > cfg.max_gap:
                for signal in memory.signals.values():
                    signal.active = False
            memory.last_seen = timestamp
            memory.last_updated = timestamp
            point = (detection.center[0] / width, detection.center[1] / height)
            # Bind only to a configured seat observed during initial calibration.
            if memory.seat_id == "UNASSIGNED" and timestamp - memory.first_seen <= cfg.calibration_seconds:
                occupied = {m.seat_id for k, m in self.students.items()
                            if k != tid and timestamp - m.last_seen < 2.0}
                candidates = [s for s in self.seats.values() if s.contains(point) and s.seat_id not in occupied]
                if candidates:
                    seat = min(candidates, key=lambda s: math.dist(point, (s.x, s.y)))
                    memory.seat_id = seat.seat_id
            if detection.head_offset is not None and detection.head_confidence >= 0.5:
                if timestamp - memory.first_seen <= cfg.calibration_seconds:
                    memory.head_samples.append(detection.head_offset)
                elif memory.head_baseline is None and len(memory.head_samples) >= 3:
                    memory.head_baseline = statistics.median(memory.head_samples)

            neighbours = []
            head_targets = []
            box = detection.box
            own_scale = math.hypot(box[2] - box[0], box[3] - box[1])
            for other in people:
                if other.track_id == tid:
                    continue
                ob = other.detection.box
                scale = (own_scale + math.hypot(ob[2] - ob[0], ob[3] - ob[1])) / 2
                if math.dist(detection.center, other.detection.center) < cfg.group_radius * max(1, scale):
                    neighbours.append(other)
                if (math.dist(detection.center, other.detection.center) < 3.0 * max(1, scale)
                        and abs(detection.center[1] - other.detection.center[1]) < 0.7 * (box[3] - box[1])):
                    head_targets.append(other)
            head_turn = False
            if memory.head_baseline is not None and detection.head_offset is not None and detection.head_confidence >= 0.5:
                delta = detection.head_offset - memory.head_baseline
                # Require another visible person on the side of the movement.
                head_turn = abs(delta) >= cfg.head_delta and any(
                    (p.detection.center[0] - detection.center[0]) * delta > 0 for p in head_targets)
            leave = memory.seat_id in self.seats and not self.seats[memory.seat_id].contains(point, margin=0.15)
            converged = [p for p in neighbours if math.dist(detection.center, p.detection.center)
                         < 0.60 * self.pair_baselines.get(tuple(sorted((tid, p.track_id))), 0)]
            signals = {
                "SEAT_LEAVE": (leave, detection.confidence),
                "HEAD_TURN": (head_turn, min(detection.confidence, detection.head_confidence)),
                "PHONE": (detection.phone_confidence >= cfg.min_phone_conf, detection.phone_confidence),
                "GATHERING": (len(converged) + 1 >= cfg.group_min_people,
                              min([detection.confidence] + [p.detection.confidence for p in neighbours])),
                "MOVEMENT": (person.speed >= cfg.speed_threshold, detection.confidence),
            }
            qualified = []
            observing = False
            thresholds = {"SEAT_LEAVE": cfg.leave_seconds, "PHONE": cfg.phone_seconds,
                          "HEAD_TURN": 2.0, "GATHERING": cfg.group_seconds, "MOVEMENT": cfg.movement_seconds}
            for kind, (active, confidence) in signals.items():
                signal = memory.signals[kind]
                signal.update(active, confidence, timestamp, dt, cfg.window)
                # No decision until >1 second of positive observation, even for one frame.
                current_duration = max(0, timestamp - signal.start) if signal.active else 0.0
                duration = sum(s[1] for s in signal.samples) if kind == "HEAD_TURN" else current_duration
                count = len(signal.starts)
                if active and duration >= 1.0:
                    observing = True
                eligible = active and duration >= thresholds[kind]
                if kind == "HEAD_TURN":
                    eligible = eligible and count >= cfg.head_count
                if eligible:
                    confidence = signal.confidence
                    score = 100 * WEIGHTS[kind] * confidence * min(1.0, duration / (2 * thresholds[kind])) * (0.8 + 0.2 * min(1, (count - 1) / 2))
                    qualified.append((kind, signal, duration, count, score))
            combined = 100 * (1 - math.prod(1 - x[4] / 100 for x in qualified))
            memory.risk = max(combined, memory.risk * 0.5 ** (decay_dt / cfg.recovery_half_life), 15.0 if observing else 0.0)
            memory.risk = min(100.0, memory.risk)
            self._set_state(memory)
            for kind, signal, duration, count, score in qualified:
                level = risk_level(memory.risk)
                if timestamp - signal.last_logged < cfg.log_interval and level == signal.last_level:
                    continue
                signal.last_logged, signal.last_level = timestamp, level
                identity = f"Seat {memory.seat_id} / Track {tid}" if memory.seat_id != "UNASSIGNED" else f"Track {tid} (seat unassigned)"
                explanation = (f"{identity}: {LABELS[kind]} for {duration:.1f}s; "
                               f"{count} occurrence(s) in {cfg.window:g}s; "
                               f"observation confidence {signal.confidence:.0%}; "
                               f"combined rule risk {memory.risk:.0f}/100 ({level}). Review video context.")
                events.append(Event(timestamp, tid, memory.seat_id, kind, round(signal.confidence, 4),
                                    round(duration, 2), count, round(memory.risk, 1), level, memory.state, explanation))
        return events
