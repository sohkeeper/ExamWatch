# ExamWatch

## Intelligent Examination Hall Monitoring and Anomaly Detection Agent

A local Python project combining pretrained visual detection with a stateful,
rule-based agent. It flags observable events for an invigilator to review.
An alert is not proof of cheating; risk is an explainable heuristic, not a
calibrated probability or a disciplinary decision.

## Start here on Windows

1. Install Python **3.11 or 3.12**, with “Add Python to PATH” enabled.
2. Extract this entire folder.
3. Double-click **run_windows.bat**. Initial dependency installation can take
   several minutes because the AI models use PyTorch.
4. A local dashboard opens at **http://localhost:8501**.
5. Start with **Run demonstration**. This uses drawn people and scripted
   observations and is explicitly labelled as a simulation.
6. For real AI processing, choose **CCTV / examination video**, upload a clip,
   configure seat regions, and select **Analyse video**.

For a quicker dashboard-only trial, run **run_demo_windows.bat**. It skips
PyTorch/Ultralytics; real video mode then requires the full installation.

### Terminal setup (Windows, macOS or Linux)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux instead:
# source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

For a CPU-only PyTorch installation, run this before installing requirements:
`python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.
The Windows launcher already includes this step.

The first AI run downloads `yolo11n-pose.pt` and `yolo11n.pt` into the working
folder. Keep those files for offline checking. Do one real-video run while
online before taking the laptop to the viva. No API key or paid service is used.
This project does not require MySQL, Java, a GPU or training a model from scratch.

## What is implemented

| Assignment item | Implementation |
|---|---|
| Hall video input | Local MP4, AVI, MOV or MKV upload |
| Person detection | YOLO11n-pose person boxes and confidence |
| Student tracking | Predicted centre/IoU cost + linear-sum assignment; temporary IDs |
| Seat departure | Person-box centre outside an explicitly configured assigned region for at least 3s |
| Repeated turning | Confident nose/ear geometry relative to the track's initial baseline; distinct turns toward the image side of another visible person |
| Mobile phone | YOLO11n cell-phone class, associated to one nearby person box, persistent for at least 2s |
| Gathering | At least 3 nearby tracks whose separation has contracted relative to initial observations |
| Unusual movement | Sustained displacement speed in person-box diagonals per video second |
| Model-based state | NORMAL → OBSERVE → SUSPICIOUS → ALERT, memory, missing-data handling and risk decay |
| Explainable reasoning | Rule name, duration, occurrence count, observation confidence and combined risk |
| Persistent events | SQLite + CSV + JSON + annotated evidence JPGs |
| Dashboard | Hall frames, track state table, risk filters, event history and downloadable evidence bundle |

## Seat map and calibration

Seat IDs are supplied by the invigilator; they are not recognized student names.
`seat_map_example.csv` illustrates the format for the synthetic hall only.
Do not use its coordinates unchanged for another camera.

Each row is `seat_id,x,y,width,height`. All coordinates are fractions of image
dimensions. `(0,0)` is the top-left and `(1,1)` is the bottom-right. The centre
of each person's bounding box should lie inside that person's assigned region.
Example: a 100-pixel-wide region in a 1000-pixel-wide frame has width `0.10`.
Edit rows and inspect the first-frame preview before analysis.

During the first two seconds, people should be seated at their assigned seats
and facing their normal direction. The system binds an observed track to a
configured region and calibrates visible head geometry. It cannot discover
administrative seat assignments. Without a seat map, seat-departure detection
is disabled and alerts show `UNASSIGNED` rather than an invented seat.

The detector identifies people, including staff. After a first pass, enter
invigilator track IDs in **Ignore track IDs**, then reanalyse the same clip.
Exclude staff from any evaluation. Re-identification after a long occlusion is
not implemented; review tracking mistakes and recalibrate if needed.

## How the AI agent works

The perception models produce observations, not misconduct labels. The agent
retains an internal world model for every track: assigned region, initial head
baseline, recent event samples, distinct episode starts, last-seen time, risk
and state. It reasons using media timestamps, regardless of how slow inference is.

For event type `e`:

```
event_score = 100 × type_weight × mean_observation_confidence
              × min(1, positive_duration / (2 × minimum_duration))
              × (0.8 + 0.2 × min(1, (recent_occurrences - 1) / 2))

combined_score = 100 × (1 - product(1 - event_score / 100))
risk = max(combined_score, previous_risk × 0.5^(elapsed_seconds / 8))
```

After at least one second of activity, the OBSERVE state has a score floor of
15 while the signal remains present. State boundaries are 15, 35 and 65.
Escalation passes through the state ladder one level per sampled observation.
The risk is a hand-designed ranking heuristic; the combination formula does
not establish independence or create a calibrated probability.

Side turns require **three distinct episodes within 20s** and at least two
seconds of accumulated positive samples. A neutral gap of 0.6s separates
episodes. One frame never counts as multiple turns. Missing detections break
continuous event streaks; they are not treated as suspicious activity.

An event snapshot is logged when its temporal gate is met, when its risk band
changes, or after a five-second logging cooldown. Thus snapshot count is not
the number of unique incidents. The score can rise with duration, frequency
and simultaneous evidence; it decays during inactivity.

## Files and outputs

```
app.py                    Streamlit dashboard and setup controls
monitor/types.py          Observation, person, seat, configuration and event data
monitor/detector.py       Pretrained YOLO person/pose and phone adapter
monitor/tracker.py        Motion/IoU tracker and assignment
monitor/agent.py          Temporal rules, memory, scoring and state transitions
monitor/storage.py        SQLite records, evidence and exports
monitor/pipeline.py       Media sampling and end-to-end orchestration
monitor/demo.py           Explicitly synthetic observations and hall drawings
monitor/visuals.py        Seat overlays and risk-coloured boxes
tests/test_agent.py       Temporal, tracking and end-to-end regression checks
docs/                     Project report and viva guide
runs/<run-id>/            Local outputs created for each analysis
```

Run outputs include `events.sqlite`, `events.csv`, `events.json`, `summary.json`,
sampled annotated frames, and `evidence/*.jpg`. `timestamp` is seconds from the
video start; `logged_at_utc` is the processing wall-clock time. Evidence bundles
exclude the original video. Uploaded source previews are kept in `runs/preview`;
delete that directory when no longer required. All processing is local after
model downloads. Nothing is automatically uploaded to LMS or another service.

## Testing and evaluation

```bash
python -m pip install pytest
python -m pytest -q
```

The regression checks cover single-frame rejection, sustained escalation and
recovery, missing detections, timestamp gaps, distinct-turn counting, seat-map
binding, normal close seating, ignored staff, stable IDs, and all five event
types with evidence in the synthetic end-to-end run. They validate software
behaviour, not real-world precision/recall.

Before submission, stage a short video of the specified actions, include normal
head movements and innocent interactions, and manually label each interval.
Measure event precision/recall, false alerts per video minute and alert delay.
Check ID switches separately. Record actual measured values in the report;
do not describe synthetic demonstration counts as AI accuracy.

## Limits to explain in the viva

- General-purpose COCO models are not trained specifically for examination halls.
- Small/covered phones, crowded scenes and low-resolution heads can be missed.
- Head turning is a 2D pose proxy; it does not establish gaze, intent or cheating.
  If both ears/nose are not sufficiently visible, the proxy is unavailable.
- Movement thresholds are camera-specific. Tracking may switch identities during
  crossings/long occlusions. A fixed camera and correct initial calibration matter.
- Gathering requires observed convergence. A group already gathered at the start
  is part of the baseline and may not trigger this rule.
- No face recognition, named-student identification, disciplinary decision,
  multi-camera tracking, email/SMS or production authentication is implemented.
- The dashboard is bound to localhost. Use the project locally for assessment.

## Before project checking

1. Test the real clip on the actual laptop; keep downloaded model files locally.
2. Verify seat IDs, exclude staff and inspect several event evidence frames.
3. Check the selected analysis duration: the default real-video run covers the
   first 60 seconds, and the sidebar allows up to 600 seconds.
4. Fill group details and genuine individual contributions in the report, add
   actual hall-video results, and print one copy.
5. Keep the source folder, report, demo clip, real clip and run exports together.
6. Upload the required files to LMS manually before the teacher's deadline.
7. Attend checking with every group member and prepare for individual questions.

### Personalize the printed report

The PDF in `docs/Project_Report.pdf` is a technical report with blank group and
evaluation fields. Fill `project_details.json` with the actual names, IDs and
contributions, then run:

```bash
python -m pip install reportlab
python tools/build_report.py
```

Real hall-video measurements still need to be entered in the evaluation section.
`docs/VERIFICATION.json` records the tests actually run for this version.
`docs/VIVA_GUIDE.md` contains the oral explanation, 25 questions and code pointers.

## Sources and licensing

- Ultralytics YOLO11: https://docs.ultralytics.com/models/yolo11/
- Pose/keypoint format: https://docs.ultralytics.com/tasks/pose/
- Streamlit API: https://docs.streamlit.io/develop/api-reference
- Ultralytics licensing: https://www.ultralytics.com/license

Pretrained weights are downloaded by Ultralytics, not bundled here. Observe
the upstream package/weight licences when redistributing or publishing this
project; the college prototype does not claim ownership of pretrained models.
