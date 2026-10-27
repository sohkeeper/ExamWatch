import cv2

COLORS = {"NORMAL": (130, 170, 140), "OBSERVE": (80, 200, 240),
          "SUSPICIOUS": (30, 135, 255), "ALERT": (80, 70, 240)}


def annotate(frame, people, agent, timestamp, seats, mode):
    image = frame.copy()
    height, width = image.shape[:2]
    for seat in seats:
        x1, y1 = int((seat.x - seat.width / 2) * width), int((seat.y - seat.height / 2) * height)
        x2, y2 = int((seat.x + seat.width / 2) * width), int((seat.y + seat.height / 2) * height)
        cv2.rectangle(image, (x1, y1), (x2, y2), (100, 105, 120), 1)
        cv2.putText(image, f"S{seat.seat_id}", (max(0, x1), max(16, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 185, 200), 1)
    for person in people:
        memory = agent.students.get(person.track_id)
        if memory is None or person.track_id in agent.ignored_ids:
            continue
        x1, y1, x2, y2 = map(int, person.detection.box)
        color = COLORS[memory.state]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"T{person.track_id} | S{memory.seat_id} | {memory.state} {memory.risk:.0f}"
        cv2.putText(image, label, (max(0, x1), max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    cv2.rectangle(image, (0, 0), (width, 35), (24, 28, 42), -1)
    cv2.putText(image, f"EXAMWATCH | {mode} | VIDEO {timestamp:.1f}s", (14, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 245), 1, cv2.LINE_AA)
    return image
