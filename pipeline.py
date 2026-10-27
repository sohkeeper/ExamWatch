from dataclasses import asdict
from pathlib import Path
import json
import uuid
import cv2
from .agent import ModelBasedAgent
from .storage import EventStore
from .tracker import MotionTracker
from .visuals import annotate
from . import demo


def new_run(root="runs"):
    path = Path(root) / uuid.uuid4().hex[:12]
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_analysis(run_dir, config, seats, video_path=None, detector=None, max_seconds=60,
                 ignored_ids=None, callback=None):
    run_dir = Path(run_dir)
    mode = "REAL VIDEO" if video_path else "SCRIPTED SIMULATION"
    store = EventStore(run_dir, mode)
    agent = ModelBasedAgent(config, seats, ignored_ids)
    tracker = MotionTracker()
    capture = None
    samples = []
    processed = 0
    last_time = 0.0
    latest = None
    warnings = []
    if video_path and detector is None:
        raise ValueError("Real video requires a detector.")
    try:
        if video_path:
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise ValueError("Cannot decode this video. Export it as a standard MP4 or AVI.")
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not 0 < fps <= 240:
                raise ValueError("Video FPS metadata is missing/invalid; re-encode at a known frame rate.")
            step = max(1, round(fps / config.sample_fps))
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = min(max_seconds, frames / fps) if frames > 0 else max_seconds
            iterator = range(max(1, int(max_seconds * fps)))
        else:
            duration = min(max_seconds, demo.DURATION)
            iterator = range(int(duration * config.sample_fps) + 1)
            fps = config.sample_fps
            step = 1
        for index in iterator:
            timestamp = index / fps
            if video_path:
                ok, frame = capture.read()
                if not ok:
                    break
                if index % step:
                    continue
                people = tracker.update(detector.detect(frame), timestamp)
            else:
                frame, people = demo.sample(timestamp, 1 / config.sample_fps)
            height, width = frame.shape[:2]
            events = agent.process(people, timestamp, (width, height))
            latest = annotate(frame, people, agent, timestamp, seats, mode)
            for event in events:
                store.save(event, latest)
            if processed % max(1, round(config.sample_fps * 2)) == 0:
                filename = f"frame_{processed:06d}.jpg"
                cv2.imwrite(str(run_dir / filename), latest)
                samples.append({"time": round(timestamp, 2), "path": filename})
            processed += 1
            last_time = timestamp
            if callback and processed % 5 == 0:
                callback(latest, min(1.0, timestamp / max(duration, 0.1)), len(store.read()))
        if processed == 0:
            raise ValueError("No decodable frames were found in the input video.")
    finally:
        if capture is not None:
            capture.release()
    cv2.imwrite(str(run_dir / "latest.jpg"), latest)
    if mode == "REAL VIDEO" and not seats:
        warnings.append("No seat map supplied: seat-leaving is disabled and seat IDs remain UNASSIGNED.")
    summary = {
        "mode": mode, "run_id": run_dir.name, "processed_frames": processed,
        "analysed_video_seconds": round(last_time, 2), "events": len(store.read()),
        "config": asdict(config), "seats": [asdict(s) for s in seats],
        "ignored_ids": sorted(ignored_ids or []), "warnings": warnings,
        "samples": samples,
        "students": [{"Track": m.track_id, "Seat": m.seat_id, "State": m.state,
                      "Risk": round(m.risk, 1), "Last_seen_s": round(m.last_seen, 1),
                      "Visible_at_end": last_time - m.last_seen < 1.0,
                      "Head_baseline_ready": m.head_baseline is not None} for m in agent.students.values()],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    store.export()
    return summary
