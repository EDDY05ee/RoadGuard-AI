"""
ROADGUARD AI — Modern Ivory + Forest Green Edition
==================================================
Run:
    .\.venv\Scripts\python.exe -m streamlit run app.py

Uses your existing database.py:
    init_db(), save_report(), get_reports(), update_status()
"""

from datetime import datetime
import uuid

import pandas as pd
import streamlit as st
from PIL import Image

try:
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO
    AI_AVAILABLE = True
except ImportError:
    hf_hub_download = None
    YOLO = None
    AI_AVAILABLE = False

from database import init_db, save_report, get_reports, update_status


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="RoadGuard AI | Road Safety Intelligence",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_REPO = "vinothvikas1987/pothole-detection-yolov8"
MODEL_FILE = "best.pt"

SEVERITY_POINTS = {"Low": 10, "Medium": 30, "High": 60, "Critical": 80}
HAZARD_POINTS = {
    "Pothole": 15,
    "Cracked Road": 8,
    "Damaged Road Surface": 12,
    "Broken Streetlight": 12,
    "Missing Road Sign": 15,
    "Waterlogging": 15,
    "Other": 5,
}
STATUSES = ["Open", "In Progress", "Resolved", "Closed"]


# ---------------------------------------------------------
# THEME: IVORY, FOREST GREEN, LIME
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --rg-bg: #f4f6f0;
        --rg-card: #ffffff;
        --rg-ink: #182b22;
        --rg-muted: #68786e;
        --rg-green: #176b45;
        --rg-lime: #c9ed78;
        --rg-border: #dce5da;
    }

    .stApp {
        background: #f4f6f0;
        color: #182b22;
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(244,246,240,.94);
    }

    [data-testid="stSidebar"] {
        background: #173c2d;
        border-right: 1px solid #285440;
    }

    [data-testid="stSidebar"] * {
        color: #f5f8f2;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #d0dfd3;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.7rem;
        padding-bottom: 2.2rem;
    }

    h1, h2, h3 {
        color: #182b22 !important;
        letter-spacing: -.035em;
        font-family: 'Manrope', sans-serif;
    }

    p, label, span {
        font-family: 'DM Sans', sans-serif;
    }

    .brand {
        font-family: 'Manrope', sans-serif;
        font-size: 1.48rem;
        font-weight: 800;
        letter-spacing: -.04em;
        color: #ffffff;
    }

    .brand span { color: #c9ed78; }

    .side-caption {
        color: #c0d5c6;
        font-size: .73rem;
        letter-spacing: .08em;
    }

    .side-status {
        margin-top: 1.1rem;
        padding: .85rem;
        border: 1px solid #46735a;
        border-radius: 12px;
        background: #204a37;
        color: #e9f5e9;
        font-size: .82rem;
    }

    .hero {
        background: #173c2d;
        color: white;
        border-radius: 22px;
        padding: 2rem 2.1rem;
        margin-bottom: 1.25rem;
        border: 1px solid #24543e;
        position: relative;
        overflow: hidden;
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        right: -75px;
        top: -115px;
        border-radius: 50%;
        background: rgba(201,237,120,.13);
        border: 1px solid rgba(201,237,120,.25);
    }

    .eyebrow {
        color: #c9ed78;
        font-size: .75rem;
        font-weight: 800;
        letter-spacing: .16em;
        text-transform: uppercase;
    }

    .hero-title {
        font-family: 'Manrope', sans-serif;
        color: #ffffff;
        font-size: clamp(1.8rem,3vw,2.7rem);
        line-height: 1.13;
        font-weight: 800;
        margin: .65rem 0;
        max-width: 850px;
    }

    .hero-title span { color: #c9ed78; }

    .hero-copy {
        color: #d5e4d8;
        max-width: 720px;
        font-size: .98rem;
        line-height: 1.65;
    }

    .metric {
        background: #ffffff;
        border: 1px solid #dce5da;
        border-radius: 17px;
        padding: 1.15rem;
        min-height: 128px;
        box-shadow: 0 5px 18px rgba(30,60,42,.035);
    }

    .metric-label {
        color: #68786e;
        font-size: .82rem;
        font-weight: 700;
    }

    .metric-value {
        color: #173c2d;
        font-family: 'Manrope', sans-serif;
        font-size: 2.15rem;
        line-height: 1.25;
        font-weight: 800;
        margin: .25rem 0;
    }

    .metric-note {
        color: #176b45;
        font-size: .76rem;
        font-weight: 600;
    }

    .panel {
        background: #ffffff;
        border: 1px solid #dce5da;
        border-radius: 17px;
        padding: 1.2rem;
        box-shadow: 0 5px 18px rgba(30,60,42,.035);
    }

    .panel-title {
        color: #20392b;
        font-family: 'Manrope', sans-serif;
        font-size: 1.04rem;
        font-weight: 800;
        margin-bottom: .7rem;
    }

    .muted {
        color: #68786e;
        line-height: 1.65;
    }

    .stButton > button {
        background: #176b45;
        color: #ffffff;
        border: 1px solid #176b45;
        border-radius: 10px;
        min-height: 2.65rem;
        font-weight: 800;
        transition: .15s ease;
    }

    .stButton > button:hover {
        background: #24583c;
        border-color: #24583c;
        color: #ffffff;
    }

    .stDownloadButton > button {
        background: #eaf3e4;
        color: #17462f;
        border: 1px solid #cbdcc8;
        border-radius: 10px;
        font-weight: 800;
    }

    .stDownloadButton > button:hover {
        background: #dcebd4;
        color: #17462f;
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    .stTextArea textarea {
        background: #ffffff !important;
        border-color: #ccd9cb !important;
        color: #182b22 !important;
        border-radius: 9px !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #dce5da;
        border-radius: 12px;
        overflow: hidden;
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dce5da;
        padding: .85rem;
        border-radius: 12px;
    }

    [data-testid="stMetricLabel"] { color: #68786e; }
    [data-testid="stMetricValue"] { color: #173c2d; }

    .footer {
        color: #718075;
        font-size: .76rem;
        border-top: 1px solid #dce5da;
        margin-top: 1.8rem;
        padding-top: 1.2rem;
    }

    hr { border-color: #dce5da; }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        border-radius: 9px;
        padding: .3rem .35rem;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(255,255,255,.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# APP HELPERS
# ---------------------------------------------------------
def assess_risk(hazard, severity):
    """Simple transparent prototype estimate, not a certified safety rating."""
    score = min(
        100,
        SEVERITY_POINTS.get(severity, 10)
        + HAZARD_POINTS.get(hazard, 5),
    )

    if score >= 80:
        priority = "URGENT"
        action = "Arrange immediate inspection and secure the affected area if necessary."
    elif score >= 60:
        priority = "HIGH"
        action = "Schedule a prompt field inspection and repair assessment."
    elif score >= 35:
        priority = "MEDIUM"
        action = "Add to the maintenance queue and monitor the location."
    else:
        priority = "LOW"
        action = "Include the issue in routine maintenance checks."

    return score, priority, action


def make_report_code():
    return (
        "RG-"
        + datetime.now().strftime("%Y%m%d-%H%M%S")
        + "-"
        + uuid.uuid4().hex[:4].upper()
    )


def load_reports():
    rows = get_reports()
    df = pd.DataFrame(rows if rows else [])

    defaults = {
        "report_code": "",
        "location": "Not provided",
        "hazard": "Other",
        "severity": "Medium",
        "description": "",
        "risk_score": 0,
        "priority": "LOW",
        "suggested_action": "",
        "status": "Open",
        "created_at": "",
    }

    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default)

    if not df.empty:
        df["risk_score"] = pd.to_numeric(
            df["risk_score"], errors="coerce"
        ).fillna(0).astype(int)

    return df


def save_new_report(hazard, severity, location, description, source="Manual"):
    score, priority, action = assess_risk(hazard, severity)

    if source != "Manual":
        description = f"[{source}] {description}"

    report = {
        "report_code": make_report_code(),
        "location": location.strip(),
        "hazard": hazard,
        "severity": severity,
        "description": description.strip(),
        "risk_score": score,
        "priority": priority,
        "suggested_action": action,
        "status": "Open",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    save_report(report)
    return report


def metric_card(label, value, note, icon):
    st.markdown(
        f"""
        <div class="metric">
            <div class="metric-label">{icon} &nbsp; {label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def csv_data(df):
    return df.to_csv(index=False).encode("utf-8-sig")


@st.cache_resource(show_spinner=False)
def load_ai_model():
    if not AI_AVAILABLE:
        raise RuntimeError(
            "AI dependencies are missing. Install ultralytics and huggingface_hub."
        )
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
    )
    return YOLO(model_path)


# ---------------------------------------------------------
# DATABASE STARTUP
# ---------------------------------------------------------
try:
    init_db()
except Exception as error:
    st.error(f"Database initialization failed: {error}")
    st.stop()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding:.45rem 0 1.25rem">
            <div style="font-size:2.2rem">🛣️</div>
            <div class="brand">RoadGuard <span>AI</span></div>
            <div class="side-caption">ROAD SAFETY INTELLIGENCE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "Report a Hazard",
            "AI Inspection",
            "Report Management",
            "Location Analytics",
            "About RoadGuard",
        ],
    )

    st.markdown(
        """
        <div class="side-status">
            <b>● System ready</b><br>
            <span style="font-size:.76rem">
                Local database connected<br>
                AI model loads when needed
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.caption("Safer roads start with better information.")


try:
    reports = load_reports()
except Exception as error:
    st.error(f"Could not load reports: {error}")
    reports = pd.DataFrame()

total = len(reports)
active = int(reports["status"].isin(["Open", "In Progress"]).sum()) if total else 0
urgent = int((reports["priority"].astype(str).str.upper() == "URGENT").sum()) if total else 0
resolved = int(reports["status"].isin(["Resolved", "Closed"]).sum()) if total else 0
ai_reports = (
    int(
        reports["description"].astype(str).str.contains(
            "[AI Inspection]", regex=False, case=False
        ).sum()
    )
    if total else 0
)


# ---------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------
if page == "Overview":
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">ROAD INTELLIGENCE PLATFORM</div>
            <div class="hero-title">
                Better roads begin with <span>better insight.</span>
            </div>
            <div class="hero-copy">
                Detect visible road distress, collect community reports,
                prioritize maintenance, and follow every issue from report
                to resolution.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(5)
    cards = [
        ("Total reports", total, "All recorded hazards", "▤"),
        ("Open / active", active, "Needs follow-up", "◷"),
        ("Urgent priority", urgent, "Review immediately", "⚠"),
        ("Resolved", resolved, "Resolved or closed", "✓"),
        ("AI findings", ai_reports, "Saved from inspections", "✦"),
    ]

    for col, item in zip(cols, cards):
        with col:
            metric_card(*item)

    st.write("")
    left, right = st.columns(2)

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-title">Hazards reported</div>',
            unsafe_allow_html=True,
        )
        if total:
            st.bar_chart(reports["hazard"].value_counts(), height=260)
        else:
            st.info("Hazard analytics will appear after your first report.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-title">Work status</div>',
            unsafe_allow_html=True,
        )
        if total:
            st.bar_chart(reports["status"].value_counts(), height=260)
        else:
            st.info("Status analytics will appear here.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("### Latest reports")

    if reports.empty:
        st.info("No reports yet. Create a report or inspect a road image.")
    else:
        cols = [
            "report_code", "location", "hazard", "severity",
            "risk_score", "priority", "status", "created_at",
        ]
        recent = reports[cols].head(8).copy()
        recent.columns = [
            "Report ID", "Location", "Hazard", "Severity",
            "Risk Score", "Priority", "Status", "Created At",
        ]
        st.dataframe(recent, use_container_width=True, hide_index=True)

        st.download_button(
            "Download all reports as CSV",
            data=csv_data(reports),
            file_name="roadguard_reports.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------
# REPORT A HAZARD
# ---------------------------------------------------------
elif page == "Report a Hazard":
    st.markdown('<div class="eyebrow">COMMUNITY REPORTING</div>', unsafe_allow_html=True)
    st.title("Create a hazard report")
    st.markdown(
        '<div class="muted">Provide clear details so the responsible team '
        'can review and prioritize the issue.</div>',
        unsafe_allow_html=True,
    )

    with st.form("hazard_form", clear_on_submit=True):
        col1, col2 = st.columns(2)

        with col1:
            location = st.text_input(
                "Location or landmark *",
                placeholder="e.g. Main Road, near the bus stop",
            )
            hazard = st.selectbox("Hazard type", list(HAZARD_POINTS.keys()))

        with col2:
            severity = st.select_slider(
                "Observed severity",
                options=["Low", "Medium", "High", "Critical"],
                value="Medium",
            )
            description = st.text_area(
                "Description and extra details",
                placeholder="Describe the issue and nearby landmarks.",
                height=125,
            )

        score, priority, action = assess_risk(hazard, severity)
        st.markdown(
            f"""
            <div class="panel">
                <div class="panel-title">Preliminary risk estimate</div>
                <h2>{score}/100 &nbsp; • &nbsp; {priority}</h2>
                <div class="muted">{action}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        submitted = st.form_submit_button(
            "Submit hazard report",
            use_container_width=True,
        )

        if submitted:
            if not location.strip():
                st.error("Please enter a location before submitting.")
            else:
                try:
                    report = save_new_report(
                        hazard, severity, location, description
                    )
                    st.success(
                        f"Report saved. Reference: {report['report_code']}"
                    )
                    st.rerun()
                except Exception as error:
                    st.error(f"Could not save report: {error}")

    st.caption(
        "Risk estimates use prototype rules and are not certified "
        "engineering assessments."
    )


# ---------------------------------------------------------
# AI INSPECTION
# ---------------------------------------------------------
elif page == "AI Inspection":
    st.markdown('<div class="eyebrow">COMPUTER VISION LAB</div>', unsafe_allow_html=True)
    st.title("Inspect a road image")
    st.markdown(
        '<div class="muted">Upload an image and use the YOLO model to '
        'look for visible road distress.</div>',
        unsafe_allow_html=True,
    )

    st.warning(
        "AI can miss hazards or make incorrect detections. Verify results "
        "before making road-safety decisions."
    )

    if not AI_AVAILABLE:
        st.error("AI dependencies are not installed in this environment.")
        st.code(
            r".\.venv\Scripts\python.exe -m pip install "
            "ultralytics huggingface_hub"
        )

    uploaded = st.file_uploader(
        "Choose a road image",
        type=["jpg", "jpeg", "png", "webp"],
    )

    confidence = st.slider(
        "Detection confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
    )
    image_size = st.select_slider(
        "Analysis resolution",
        options=[320, 480, 640, 800, 960],
        value=640,
    )

    if uploaded is not None:
        try:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded road image", use_container_width=True)

            if st.button("Run AI inspection", use_container_width=True):
                with st.spinner("Loading model and analyzing image..."):
                    model = load_ai_model()
                    results = model.predict(
                        source=image,
                        conf=confidence,
                        imgsz=image_size,
                        verbose=False,
                    )
                    result = results[0]
                    annotated = result.plot()[:, :, ::-1].copy()
                    found = []
                    names = result.names

                    if result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls[0].item())
                            probability = float(box.conf[0].item())
                            if isinstance(names, dict):
                                label = names.get(class_id, str(class_id))
                            else:
                                label = (
                                    names[class_id]
                                    if class_id < len(names)
                                    else str(class_id)
                                )
                            found.append({
                                "Detected class": str(label),
                                "Confidence": f"{probability:.1%}",
                            })

                    st.session_state["rg_detection"] = {
                        "annotated": annotated,
                        "found": found,
                        "filename": uploaded.name,
                    }

                st.success("Image analysis completed.")

        except Exception as error:
            st.error(f"Inspection failed: {error}")
            st.caption(
                "The first run may need an internet connection to download "
                "the model. Check that the packages are installed."
            )

    detection = st.session_state.get("rg_detection")

    if detection:
        st.markdown("---")
        st.subheader("Inspection result")
        st.image(
            detection["annotated"],
            caption="AI annotated image",
            use_container_width=True,
        )

        found = detection["found"]
        if found:
            st.dataframe(
                pd.DataFrame(found),
                use_container_width=True,
                hide_index=True,
            )
            st.metric("Objects detected", len(found))
        else:
            st.info(
                "No objects passed the selected confidence threshold. "
                "This does not prove that the road is hazard-free."
            )

        with st.expander("Save a detection as a report"):
            if not found:
                st.info("No detections are available to save.")
            else:
                selected = st.selectbox(
                    "Select a detected class",
                    [item["Detected class"] for item in found],
                )
                location = st.text_input(
                    "Location or landmark *",
                    key="rg_ai_location",
                )
                severity = st.selectbox(
                    "Estimated severity",
                    ["Low", "Medium", "High", "Critical"],
                    index=1,
                    key="rg_ai_severity",
                )
                notes = st.text_area(
                    "Additional notes",
                    key="rg_ai_notes",
                )

                if st.button("Save AI finding"):
                    if not location.strip():
                        st.error("Please enter a location.")
                    else:
                        label = selected.lower()
                        if "pothole" in label:
                            hazard = "Pothole"
                        elif "crack" in label:
                            hazard = "Cracked Road"
                        elif any(
                            word in label
                            for word in [
                                "road", "surface", "distress",
                                "alligator", "transverse", "longitudinal",
                            ]
                        ):
                            hazard = "Damaged Road Surface"
                        else:
                            hazard = "Other"

                        description = (
                            f"AI model detected: {selected}. "
                            f"Image: {detection['filename']}. {notes}"
                        )

                        try:
                            report = save_new_report(
                                hazard,
                                severity,
                                location,
                                description,
                                source="AI Inspection",
                            )
                            st.success(
                                f"Saved report {report['report_code']}"
                            )
                            st.rerun()
                        except Exception as error:
                            st.error(f"Could not save report: {error}")


# ---------------------------------------------------------
# REPORT MANAGEMENT
# ---------------------------------------------------------
elif page == "Report Management":
    st.markdown('<div class="eyebrow">MAINTENANCE WORKSPACE</div>', unsafe_allow_html=True)
    st.title("Manage road reports")
    st.markdown(
        '<div class="muted">Search, filter, export, and update reports '
        'as maintenance progresses.</div>',
        unsafe_allow_html=True,
    )

    if reports.empty:
        st.info("No reports have been submitted yet.")
    else:
        c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])

        with c1:
            search = st.text_input(
                "Search reports",
                placeholder="ID, location, hazard, or description",
            )
        with c2:
            status_filter = st.selectbox("Status", ["All"] + STATUSES)
        with c3:
            priority_filter = st.selectbox(
                "Priority", ["All", "URGENT", "HIGH", "MEDIUM", "LOW"]
            )
        with c4:
            hazard_filter = st.selectbox(
                "Hazard",
                ["All"] + sorted(reports["hazard"].astype(str).unique()),
            )

        filtered = reports.copy()

        if search.strip():
            mask = filtered.astype(str).agg(
                " ".join, axis=1
            ).str.contains(search.strip(), case=False, na=False)
            filtered = filtered[mask]

        if status_filter != "All":
            filtered = filtered[filtered["status"] == status_filter]

        if priority_filter != "All":
            filtered = filtered[
                filtered["priority"].astype(str).str.upper()
                == priority_filter
            ]

        if hazard_filter != "All":
            filtered = filtered[filtered["hazard"] == hazard_filter]

        st.caption(f"{len(filtered)} report(s) match your filters.")

        columns = [
            "report_code", "location", "hazard", "severity",
            "risk_score", "priority", "status", "created_at",
        ]
        table = filtered[columns].copy()
        table.columns = [
            "Report ID", "Location", "Hazard", "Severity",
            "Risk Score", "Priority", "Status", "Created At",
        ]
        st.dataframe(table, use_container_width=True, hide_index=True)

        st.download_button(
            "Download filtered CSV",
            data=csv_data(filtered),
            file_name="roadguard_filtered_reports.csv",
            mime="text/csv",
        )

        st.markdown("---")
        st.subheader("Update report status")

        codes = filtered["report_code"].astype(str).tolist()

        if codes:
            selected_code = st.selectbox("Select report", codes)
            row = filtered[
                filtered["report_code"].astype(str) == selected_code
            ].iloc[0]

            current = row["status"]
            index = STATUSES.index(current) if current in STATUSES else 0

            new_status = st.selectbox(
                "New status",
                STATUSES,
                index=index,
            )

            st.markdown(
                f"**{row['hazard']}** at {row['location']}  \n"
                f"Risk score: {row['risk_score']}/100 • "
                f"Priority: {row['priority']}"
            )

            if st.button("Save status update", use_container_width=True):
                try:
                    update_status(selected_code, new_status)
                    st.success(
                        f"{selected_code} updated to {new_status}."
                    )
                    st.rerun()
                except Exception as error:
                    st.error(f"Could not update status: {error}")
        else:
            st.info("No reports match the selected filters.")


# ---------------------------------------------------------
# LOCATION ANALYTICS
# ---------------------------------------------------------
elif page == "Location Analytics":
    st.markdown('<div class="eyebrow">LOCATION INTELLIGENCE</div>', unsafe_allow_html=True)
    st.title("Hazards by location")
    st.markdown(
        '<div class="muted">Review the location names supplied by reporters. '
        'This view does not invent GPS coordinates.</div>',
        unsafe_allow_html=True,
    )

    if reports.empty:
        st.info("Location analytics will appear after reports are submitted.")
    else:
        summary = (
            reports.groupby("location", dropna=False)
            .agg(
                Reports=("report_code", "count"),
                Urgent=(
                    "priority",
                    lambda values: int(
                        (values.astype(str).str.upper() == "URGENT").sum()
                    ),
                ),
                Active=(
                    "status",
                    lambda values: int(
                        values.isin(["Open", "In Progress"]).sum()
                    ),
                ),
            )
            .reset_index()
            .sort_values(["Urgent", "Reports"], ascending=False)
        )

        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.subheader("Location comparison")
        st.bar_chart(
            summary.set_index("location")[["Reports", "Urgent", "Active"]],
            height=330,
        )
        st.caption(
            "To build a true map, add verified latitude and longitude fields "
            "to your database and report form."
        )


# ---------------------------------------------------------
# ABOUT
# ---------------------------------------------------------
elif page == "About RoadGuard":
    st.markdown('<div class="eyebrow">ABOUT THIS PROJECT</div>', unsafe_allow_html=True)
    st.title("Building safer roads through better data")

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">RoadGuard <span>AI</span></div>
            <div class="hero-copy">
                A road-safety prototype that brings community reports,
                AI image inspection, transparent risk rules, and maintenance
                tracking together in one simple workspace.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)

    with left:
        st.markdown("### Main capabilities")
        st.write("✓ Community hazard reporting")
        st.write("✓ YOLO road-image inspection")
        st.write("✓ Transparent risk scoring")
        st.write("✓ SQLite data persistence")
        st.write("✓ Status tracking and CSV exports")
        st.write("✓ Location analytics")

    with right:
        st.markdown("### Responsible use")
        st.write("• AI detections may be incorrect or incomplete.")
        st.write("• Risk scores are not engineering certifications.")
        st.write("• Location names are not verified GPS positions.")
        st.write("• Field teams should verify findings before taking action.")

    st.markdown("---")
    st.markdown("### Future improvements")
    st.write("1. Add GPS coordinates and an interactive map.")
    st.write("2. Save uploaded evidence with each report.")
    st.write("3. Add user accounts and access permissions.")
    st.write("4. Evaluate model performance on local road images.")


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        <b>RoadGuard AI</b> &nbsp; • &nbsp;
        Safer roads. Smarter tomorrow.<br>
        AI detections and risk estimates are decision-support tools
        and require human verification.
    </div>
    """,
    unsafe_allow_html=True,
)