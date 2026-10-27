from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import io
import json
import sqlite3
import zipfile


class EventStore:
    def __init__(self, run_dir, source_mode):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "evidence").mkdir(exist_ok=True)
        self.source_mode = source_mode
        self.db_path = self.run_dir / "events.sqlite"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY, logged_at_utc TEXT NOT NULL, timestamp REAL NOT NULL,
                track_id INTEGER, seat_id TEXT, event_type TEXT, confidence REAL,
                duration REAL, frequency INTEGER, risk_score REAL, risk_level TEXT,
                agent_state TEXT, explanation TEXT, evidence_frame TEXT, source_mode TEXT)""")

    def save(self, event, frame):
        import cv2
        event.source_mode = self.source_mode
        name = f"evidence/{event.timestamp:09.2f}_track{event.track_id}_{event.event_type}.jpg"
        if not cv2.imwrite(str(self.run_dir / name), frame):
            raise OSError("Could not save the evidence frame.")
        event.evidence_frame = name
        values = asdict(event)
        values["logged_at_utc"] = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            columns = ",".join(values)
            placeholders = ",".join("?" for _ in values)
            conn.execute(f"INSERT INTO events ({columns}) VALUES ({placeholders})", tuple(values.values()))

    def read(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(row) for row in conn.execute("SELECT * FROM events ORDER BY timestamp, id")]

    def export(self):
        rows = self.read()
        output = io.StringIO()
        fields = ["id", "logged_at_utc", "timestamp", "track_id", "seat_id", "event_type",
                  "confidence", "duration", "frequency", "risk_score", "risk_level",
                  "agent_state", "explanation", "evidence_frame", "source_mode"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        (self.run_dir / "events.csv").write_text(output.getvalue(), encoding="utf-8")
        (self.run_dir / "events.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        return output.getvalue().encode("utf-8")

    def bundle(self):
        self.export()
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(self.run_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv"}:
                    archive.write(path, path.relative_to(self.run_dir))
        return output.getvalue()
