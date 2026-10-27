"""Scripted observations and drawings, NOT model predictions or evaluation data."""
import math
import cv2
import numpy as np
from .types import Detection, Person, Seat

WIDTH, HEIGHT, DURATION = 960, 600, 34.0
BASE = {1: (190, 190), 2: (470, 190), 3: (750, 190),
        4: (190, 420), 5: (470, 420), 6: (750, 420)}


def demo_seats():
    return [Seat(str(i), x / WIDTH, y / HEIGHT, 0.17, 0.25) for i, (x, y) in BASE.items()]


def position(track_id, timestamp):
    x, y = BASE[track_id]
    if track_id == 5 and 15 <= timestamp <= 24:
        progress = min(1, (timestamp - 15) / 1.4, (24 - timestamp) / 1.4)
        x += 130 * progress
        y -= 120 * progress
    if track_id in (1, 3) and 23 <= timestamp <= 31:
        progress = min(1, (timestamp - 23) / 1.4, (31 - timestamp) / 1.4)
        x += (BASE[2][0] + (-50 if track_id == 1 else 50) - x) * progress
        y += 5 * progress
    # Sustained large movement for T6, before its later return to its seat.
    if track_id == 6 and 4 <= timestamp <= 8:
        x += 90 * math.sin((timestamp - 4) * 2.8)
    return x, y


def sample(timestamp, step=0.2):
    frame = np.full((HEIGHT, WIDTH, 3), (43, 38, 31), dtype=np.uint8)
    for seat in demo_seats():
        x, y = int(seat.x * WIDTH), int(seat.y * HEIGHT)
        cv2.rectangle(frame, (x - 76, y - 60), (x + 76, y + 72), (74, 68, 58), -1)
        cv2.line(frame, (x - 70, y + 26), (x + 70, y + 26), (120, 110, 92), 3)
    people = []
    turning = any(start <= timestamp < start + 1.1 for start in (5, 7.5, 10, 12.5))
    for tid in BASE:
        x, y = position(tid, timestamp)
        old = position(tid, max(0, timestamp - step))
        phone = tid == 4 and 9 <= timestamp <= 20
        offset = 0.65 if tid == 2 and turning else 0.0
        # For the head-turn demonstration, T3 is a visible neighbour to T2.
        box = (x - 36, y - 56, x + 36, y + 56)
        detection = Detection(box, 0.95, offset, 0.95, 0.94 if phone else 0)
        speed = math.dist((x, y), old) / max(step, 0.001) / math.hypot(72, 112)
        people.append(Person(tid, detection, speed))
        cv2.ellipse(frame, (int(x), int(y + 10)), (25, 37), 0, 0, 360, (160, 140, 105), -1)
        cv2.circle(frame, (int(x + offset * 12), int(y - 32)), 15, (190, 200, 215), -1)
        if phone:
            cv2.rectangle(frame, (int(x + 25), int(y - 4)), (int(x + 37), int(y + 18)), (40, 40, 230), -1)
    cv2.putText(frame, "SCRIPTED SIMULATION - observations are synthetic, not YOLO output", (35, 574),
                cv2.FONT_HERSHEY_SIMPLEX, 0.61, (180, 200, 240), 1, cv2.LINE_AA)
    return frame, people
