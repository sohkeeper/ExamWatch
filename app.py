"""Run with: python -m streamlit run app.py"""
from pathlib import Path
import json
import cv2
import pandas as pd
import streamlit as st
from monitor.agent import LABELS
from monitor.types import Config, Seat
from monitor import demo
from monitor.pipeline import new_run, run_analysis
from monitor.storage import EventStore
from monitor.visuals import annotate

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="ExamWatch | Invigilator", page_icon="◉", layout="wide")
st.markdown("""<style>
.stApp {background: #0c1422; color:#e6edf7;}
[data-testid="stSidebar"] {background:#131e30;}
h1,h2,h3 {letter-spacing:-.03em;}
[data-testid="stMetric"] {background:#172338;border:1px solid #2b3c55;border-radius:12px;padding:16px;}
.eyebrow {font-size:12px;font-weight:700;letter-spacing:.18em;color:#60dacb;}
.subtitle {color:#a9b9ce;max-width:850px;}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="eyebrow">EXAMWATCH / MODEL-BASED MONITORING AGENT</div>', unsafe_allow_html=True)
st.title("An observation. A reason. A review.")
st.markdown('<p class="subtitle">Anonymous video tracks, temporal event rules and supporting evidence. '
            'Alerts flag observable activity for invigilator review; they do not establish misconduct.</p>', unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading YOLO person/pose and phone models...")
def load_detector(person_conf, phone_conf):
    from monitor.detector import YoloDetector
    return YoloDetector(person_conf, phone_conf)


def parse_seats(rows):
    seats = []
    for row in rows.to_dict("records"):
        if pd.isna(row.get("seat_id")) or str(row.get("seat_id", "")).strip() == "":
            continue
        seats.append(Seat(str(row["seat_id"]).strip(), *[float(row[k]) for k in ("x", "y", "width", "height")]))
    if len({s.seat_id for s in seats}) != len(seats):
        raise ValueError("Each seat must have a unique ID.")
    return seats


with st.sidebar:
    st.header("Analysis setup")
    source = st.radio("Input source", ["Scripted demonstration", "CCTV / examination video"])
    simulated = source == "Scripted demonstration"
    max_seconds = st.slider("Analyse first N video seconds", 10, 600, 40 if simulated else 60, step=10)
    sample_fps = st.slider("Samples per video second", 2, 10, 5)
    st.caption("Durations use video timestamps, not the computer's processing time.")
    with st.expander("Event thresholds"):
        phone_seconds = st.number_input("Phone persistence (seconds)", 1.0, 15.0, 2.0, 0.5)
        leave_seconds = st.number_input("Outside seat (seconds)", 1.0, 15.0, 3.0, 0.5)
        head_count = st.number_input("Distinct side-turn episodes / 20s", 2, 10, 3)
        head_delta = st.slider("Head movement threshold", 0.10, 0.80, 0.28, 0.02)
        person_conf = st.slider("Minimum person confidence", 0.10, 0.90, 0.35, 0.05)
        phone_conf = st.slider("Minimum phone confidence", 0.10, 0.90, 0.30, 0.05)
    ignored = st.text_input("Ignore track IDs (e.g. invigilators)", placeholder="7, 9")
    st.caption("Run once to inspect track IDs, then exclude staff IDs and reanalyse the same clip.")
    st.divider()
    st.caption("NORMAL → OBSERVE → SUSPICIOUS → ALERT")
    st.caption("Rule scores are heuristic scores out of 100, not probabilities of cheating.")

video_path = None
uploaded = None
seats = demo.demo_seats() if simulated else []
if simulated:
    st.info("SCRIPTED DEMONSTRATION: six drawn people and known, synthetic observations. "
            "This tests the temporal agent, dashboard and evidence export, not AI detection accuracy.")
    with st.expander("Demonstration timeline"):
        st.write("0–4s: normal seating · 4–8s: rapid movement · 5–14s: separate side turns · "
                 "9–20s: phone observation · 15–24s: seat departure · 23–31s: three people converge.")
else:
    uploaded = st.file_uploader("Upload examination-hall video", type=["mp4", "avi", "mov", "mkv"])
    st.caption("Use a fixed camera. For the first two seconds, students should be seated and facing their normal direction.")
    with st.expander("Configure assigned seat regions", expanded=True):
        st.write("Seat rectangles cover the expected centre of each person's detected box, not the desk alone. "
                 "Use fractional coordinates: 0 is left/top, 1 is right/bottom. Leave this empty to disable seat-leaving detection.")
        csv_file = st.file_uploader("Optional: load seat-map CSV", type=["csv"], key="seats_csv")
        initial = pd.DataFrame(columns=["seat_id", "x", "y", "width", "height"])
        if csv_file is not None:
            try:
                initial = pd.read_csv(csv_file, dtype={"seat_id": str})
                if not set(initial.columns) >= {"seat_id", "x", "y", "width", "height"}:
                    raise ValueError("CSV needs seat_id,x,y,width,height columns.")
            except Exception as exc:
                st.error(str(exc))
        seat_rows = st.data_editor(initial, num_rows="dynamic", key="seat_editor", hide_index=True,
            column_config={"seat_id": st.column_config.TextColumn("Seat ID"),
                           **{k: st.column_config.NumberColumn(k, min_value=0.0, max_value=1.0, step=0.01)
                              for k in ("x", "y", "width", "height")}})
        try:
            seats = parse_seats(seat_rows)
        except (ValueError, TypeError) as exc:
            st.error(f"Invalid seat map: {exc}")
            seats = None
        if uploaded is not None:
            preview_path = ROOT / "runs" / "preview"
            preview_path.mkdir(parents=True, exist_ok=True)
            # Treat upload names as untrusted: use a fixed local filename.
            target = preview_path / ("source" + Path(uploaded.name).suffix.lower())
            target.write_bytes(uploaded.getvalue())
            video_path = target
            cap = cv2.VideoCapture(str(target))
            ok, frame = cap.read()
            cap.release()
            if ok:
                from monitor.agent import ModelBasedAgent
                preview = annotate(frame, [], ModelBasedAgent(), 0, seats or [], "SEAT-MAP PREVIEW")
                st.image(preview, channels="BGR", caption="First frame with your seat regions", width="stretch")
                h, w = frame.shape[:2]
                st.caption(f"Frame: {w} × {h}px. Example: pixel (x,y)=({w//2},{h//2}) is fractional (0.5,0.5).")
            else:
                st.error("This video cannot be decoded. Convert it to MP4 (H.264) or AVI and retry.")

start = st.button("Run demonstration" if simulated else "Analyse video", type="primary",
                  disabled=(not simulated and (uploaded is None or seats is None)))
if start:
    try:
        ignored_ids = {int(x.strip()) for x in ignored.split(",") if x.strip()}
        cfg = Config(sample_fps=sample_fps, phone_seconds=phone_seconds, leave_seconds=leave_seconds,
                     head_count=int(head_count), head_delta=head_delta,
                     min_person_conf=person_conf, min_phone_conf=phone_conf)
        detector = None if simulated else load_detector(person_conf, phone_conf)
        run_dir = new_run(ROOT / "runs")
        progress = st.progress(0.0, text="Analysing video observations...")
        live = st.empty()
        def update(image, fraction, count):
            progress.progress(fraction, text=f"Analysing · {count} logged event snapshots")
            live.image(image, channels="BGR", width="stretch")
        summary = run_analysis(run_dir, cfg, seats or [], video_path, detector, max_seconds,
                               ignored_ids, update)
        st.session_state["last_run"] = str(run_dir)
        progress.empty()
        live.empty()
        st.success(f"Analysis complete: {summary['analysed_video_seconds']} video seconds, "
                   f"{summary['processed_frames']} sampled frames. Results saved locally.")
    except Exception as exc:
        st.error(f"Analysis could not finish: {exc}")
        st.caption("For AI mode, install requirements.txt and connect to the internet for the first model download. "
                   "After both .pt model files are downloaded, the app can process local videos offline.")

if "last_run" in st.session_state:
    run_dir = Path(st.session_state["last_run"])
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    store = EventStore(run_dir, summary["mode"])
    events = store.read()
    st.divider()
    st.caption(f"DISPLAYED RESULTS: {summary['mode']} · Run {summary['run_id']} · "
               f"First {summary['analysed_video_seconds']:.1f} video seconds. These belong to the last completed run.")
    if summary["mode"] != "REAL VIDEO":
        st.warning("Simulation results below. No YOLO inference was used in this run.")
    for warning in summary["warnings"]:
        st.warning(warning)
    cards = st.columns(4)
    cards[0].metric("Anonymous tracks seen", len(summary["students"]))
    cards[1].metric("Logged event snapshots", len(events))
    cards[2].metric("High-risk snapshots", sum(e["risk_level"] == "HIGH" for e in events))
    cards[3].metric("Video analysed", f"{summary['analysed_video_seconds']:.1f}s")
    hall, states = st.columns([1.7, 1])
    with hall:
        st.subheader("Examination hall")
        samples = summary.get("samples", [])
        if len(samples) > 1:
            selected_frame = st.slider("Inspect saved frame", 0, len(samples) - 1, len(samples) - 1,
                                       key=f"frames_{summary['run_id']}")
            item = samples[selected_frame]
            st.image(str(run_dir / item["path"]), caption=f"Video timestamp {item['time']:.1f}s", width="stretch")
        else:
            st.image(str(run_dir / "latest.jpg"), width="stretch")
    with states:
        st.subheader("Agent state at run end")
        st.dataframe(pd.DataFrame(summary["students"]), hide_index=True, width="stretch")
        st.caption("Risk colours: normal / observe / suspicious / alert. Track IDs are temporary; "
                   "occlusion or crossings can create a new ID. They are not verified student identities.")
    st.subheader("Alert history & supporting evidence")
    filters = st.columns(2)
    level = filters[0].selectbox("Risk filter", ["ALL", "HIGH", "MEDIUM", "LOW"])
    kind = filters[1].selectbox("Event filter", ["ALL"] + sorted({e["event_type"] for e in events}))
    filtered = [e for e in events if (level == "ALL" or e["risk_level"] == level)
                and (kind == "ALL" or e["event_type"] == kind)]
    if filtered:
        st.dataframe(pd.DataFrame(filtered)[["timestamp", "seat_id", "track_id", "event_type",
                     "duration", "frequency", "confidence", "risk_score", "risk_level", "agent_state"]],
                     hide_index=True, width="stretch")
        index = st.selectbox("Review an event", range(len(filtered)),
            format_func=lambda i: f"{filtered[i]['timestamp']:.1f}s · Seat {filtered[i]['seat_id']} / "
                                 f"Track {filtered[i]['track_id']} · {LABELS[filtered[i]['event_type']]} · {filtered[i]['risk_level']}")
        event = filtered[index]
        evidence, reason = st.columns([1.4, 1])
        evidence.image(str(run_dir / event["evidence_frame"]), width="stretch")
        with reason:
            st.markdown(f"### {event['risk_level']} · {event['risk_score']:.0f}/100")
            st.write(event["explanation"])
            st.caption(f"Recorded: {event['logged_at_utc']} · Source: {event['source_mode']}")
            st.caption("Repeated entries can be updated snapshots of the same continuing event, "
                       "logged on a risk-band change or after the five-second cooldown.")
    else:
        st.info("No event snapshots match these filters. No alerts is a valid result.")
    downloads = st.columns(2)
    downloads[0].download_button("Download event CSV", store.export(), f"events_{run_dir.name}.csv", "text/csv")
    downloads[1].download_button("Download evidence bundle", store.bundle(), f"evidence_{run_dir.name}.zip", "application/zip")
else:
    st.divider()
    st.subheader("From observations to explainable alerts")
    st.write("Video → person/pose + phone detections → motion tracking → per-track memory → "
             "temporal rules → risk and agent state → evidence and dashboard.")
    st.caption("Start with the 34-second scripted demonstration to inspect every agent feature. "
               "Switch to CCTV video to run the pretrained models on actual frames.")
