from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# Make repository imports reliable when Streamlit starts this file.
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from Marl.mappo.communication.schema import (
        EventType,
        TargetType,
        ThreatLevel,
        ConfidenceLevel,
        HostStatus,
        Priority,
        StructuredMessage,
        confidence_level_to_value,
    )
    from Marl.mappo.communication.evaluator import MessageEvaluator
    from Marl.mappo.communication.trust import DynamicTrust
    IMPORT_ERROR = None
except Exception as exc:
    IMPORT_ERROR = exc


# =====================================================================
# Page configuration
# =====================================================================
st.set_page_config(
    page_title="CC4 MARL | Cyber Defense Control Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# Styling
# =====================================================================
st.markdown(
    """
    <style>
    :root {
        --bg: #07111f;
        --panel: #0d1a2b;
        --panel-2: #101f32;
        --line: #20334a;
        --text: #e8eef7;
        --muted: #91a2b8;
        --blue: #4da3ff;
        --green: #35d18a;
        --amber: #f0b44f;
        --red: #f06464;
        --purple: #9a7cff;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 80% -10%, rgba(48, 112, 255, 0.15), transparent 36%),
            linear-gradient(180deg, #07111f 0%, #081321 100%);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(7, 17, 31, 0.92);
    }

    [data-testid="stSidebar"] {
        background: #081422;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] * {
        color: var(--text);
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.15rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3, h4 {
        color: var(--text) !important;
        letter-spacing: -0.02em;
    }

    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 20px;
        padding: 6px 0 18px 0;
        border-bottom: 1px solid var(--line);
        margin-bottom: 18px;
    }

    .title {
        font-size: 2.15rem;
        font-weight: 760;
        line-height: 1.1;
        margin: 0;
    }

    .subtitle {
        color: var(--muted);
        font-size: 0.98rem;
        margin-top: 7px;
    }

    .status-box {
        min-width: 220px;
        padding: 12px 15px;
        border: 1px solid #3155a5;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(29, 67, 170, .45), rgba(11, 25, 55, .55));
    }

    .status-title {
        font-weight: 700;
        font-size: 0.95rem;
    }

    .status-sub {
        color: #b8c8de;
        margin-top: 4px;
        font-size: 0.82rem;
    }

    .metric-card {
        background: linear-gradient(180deg, rgba(16, 31, 50, .95), rgba(11, 24, 40, .95));
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 14px 16px;
        min-height: 112px;
    }

    .metric-label {
        color: #a9b8cb;
        font-size: 0.83rem;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 760;
        margin-top: 2px;
    }

    .metric-note {
        color: var(--muted);
        font-size: 0.74rem;
        margin-top: 2px;
    }

    .panel {
        background: linear-gradient(180deg, rgba(13, 26, 43, .97), rgba(10, 22, 37, .97));
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 14px;
    }

    .panel-title {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }

    .panel-title-main {
        font-size: 1.03rem;
        font-weight: 700;
    }

    .panel-title-sub {
        color: var(--muted);
        font-size: 0.73rem;
    }

    .chip {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 999px;
        font-size: 0.68rem;
        font-weight: 700;
        border: 1px solid;
        margin-left: 6px;
    }

    .chip-blue {
        color: #99cbff;
        background: rgba(77, 163, 255, .10);
        border-color: rgba(77, 163, 255, .35);
    }

    .chip-green {
        color: #9ef0c8;
        background: rgba(53, 209, 138, .10);
        border-color: rgba(53, 209, 138, .35);
    }

    .chip-muted {
        color: #a6b1c0;
        background: rgba(145, 162, 184, .08);
        border-color: rgba(145, 162, 184, .20);
    }

    .message-card {
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 10px 12px;
        margin-bottom: 8px;
        background: rgba(17, 34, 54, .55);
    }

    .message-head {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        align-items: center;
    }

    .message-route {
        font-weight: 700;
    }

    .message-main {
        color: #dbe5f2;
        margin-top: 4px;
        font-size: 0.84rem;
    }

    .message-meta {
        color: var(--muted);
        margin-top: 5px;
        font-size: 0.72rem;
    }

    .quality-good {
        color: var(--green);
        font-weight: 750;
    }

    .quality-mid {
        color: var(--amber);
        font-weight: 750;
    }

    .quality-bad {
        color: var(--red);
        font-weight: 750;
    }

    .big-quality {
        font-size: 2.25rem;
        font-weight: 800;
        line-height: 1;
    }

    .detail-row {
        display: flex;
        justify-content: space-between;
        gap: 15px;
        padding: 7px 0;
        border-bottom: 1px solid rgba(32, 51, 74, .8);
        font-size: 0.82rem;
    }

    .detail-row:last-child {
        border-bottom: 0;
    }

    .detail-key {
        color: var(--muted);
    }

    .detail-value {
        text-align: right;
        font-weight: 650;
    }

    .pipeline {
        display: flex;
        flex-wrap: wrap;
        gap: 7px;
        align-items: center;
        font-size: .78rem;
    }

    .pipe-item {
        background: #0c1a2c;
        border: 1px solid var(--line);
        border-radius: 7px;
        padding: 6px 9px;
    }

    .pipe-arrow {
        color: #667990;
    }

    .agent-node {
        border: 1px solid #2d5e92;
        background: #0b1c31;
        border-radius: 10px;
        padding: 10px;
        text-align: center;
        min-height: 86px;
    }

    .agent-node.active {
        border-color: var(--blue);
        box-shadow: 0 0 0 1px rgba(77, 163, 255, .16) inset;
    }

    .agent-id {
        font-size: 1.1rem;
        font-weight: 800;
    }

    .agent-zone {
        color: #9eafc3;
        font-size: .71rem;
        margin-top: 2px;
    }

    .agent-role {
        color: #c6d4e4;
        font-size: .73rem;
        margin-top: 6px;
    }

    .legend {
        display: flex;
        flex-wrap: wrap;
        gap: 14px;
        color: var(--muted);
        font-size: .73rem;
        margin-top: 10px;
    }

    .legend-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 5px;
    }

    .section-note {
        color: var(--muted);
        font-size: .75rem;
        line-height: 1.45;
    }

    .footer-note {
        color: #71859d;
        text-align: center;
        font-size: .72rem;
        padding: 12px 0 4px 0;
    }

    [data-testid="stMetric"] {
        background: transparent;
    }

    [data-testid="stTabs"] button {
        font-weight: 650;
    }

    .coming-box {
        border: 1px dashed #39506b;
        border-radius: 10px;
        padding: 14px;
        background: rgba(13, 25, 40, .6);
        min-height: 125px;
    }

    .coming-title {
        font-weight: 740;
        color: #cbd5e2;
    }

    .coming-text {
        color: #8799ae;
        font-size: .76rem;
        margin-top: 7px;
        line-height: 1.45;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================================
# Import failure page
# =====================================================================
if IMPORT_ERROR is not None:
    st.error("The project communication modules could not be imported.")
    st.code(
        "python3 -c \"from Marl.mappo.communication.schema import StructuredMessage\"",
        language="bash",
    )
    st.exception(IMPORT_ERROR)
    st.stop()

# =====================================================================
# Constants from the real project schema
# =====================================================================
AGENTS = {
    0: "Restricted A",
    1: "Operational A",
    2: "Restricted B",
    3: "Operational B",
    4: "HQ",
}

AGENT_SHORT = {
    0: "A0",
    1: "A1",
    2: "A2",
    3: "A3",
    4: "A4",
}

# =====================================================================
# Session state
# =====================================================================
if "trust_engine" not in st.session_state:
    st.session_state.trust_engine = DynamicTrust(num_agents=5)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "trust_history" not in st.session_state:
    initial = st.session_state.trust_engine.get_trust_matrix().detach().cpu().numpy()
    st.session_state.trust_history = [initial.copy()]

if "last_evaluation" not in st.session_state:
    st.session_state.last_evaluation = None

if "selected_message" not in st.session_state:
    st.session_state.selected_message = None


def trust_matrix() -> pd.DataFrame:
    matrix = (
        st.session_state.trust_engine
        .get_trust_matrix()
        .detach()
        .cpu()
        .numpy()
    )
    labels = [AGENT_SHORT[i] for i in range(5)]
    return pd.DataFrame(matrix, index=labels, columns=labels)


def quality_class(score: float) -> str:
    if score >= 0.75:
        return "quality-good"
    if score >= 0.45:
        return "quality-mid"
    return "quality-bad"


def enum_name(value) -> str:
    return getattr(value, "name", str(value))


def build_ground_truth(
    target_type,
    target_id: int,
    event_type,
    threat_level,
    status,
):
    """
    Small review/demo ground truth.

    This is intentionally manual demo input. It is NOT presented as
    live CybORG ground truth unless a live environment connector is
    later wired into this dashboard.
    """
    base = {
        "event_type": event_type,
        "threat_level": threat_level,
        "status": status,
    }

    return {
        "target_status": {target_id: base}
        if target_type == TargetType.HOST
        else {},
        "subnet_status": {target_id: base}
        if target_type == TargetType.SUBNET
        else {},
    }


def add_message_event(sender, receiver, message, evaluation, trust_before, trust_after):
    record = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "sender": sender,
        "receiver": receiver,
        "event": enum_name(message.event_type),
        "target_type": enum_name(message.target_type),
        "target_id": int(message.target_id),
        "threat": enum_name(message.threat_level),
        "confidence": float(message.confidence),
        "status": enum_name(message.status),
        "priority": enum_name(message.priority),
        "quality": float(evaluation.overall_score),
        "trust_before": float(trust_before),
        "trust_after": float(trust_after),
    }
    st.session_state.messages.insert(0, record)
    st.session_state.selected_message = record


def reset_demo():
    st.session_state.trust_engine = DynamicTrust(num_agents=5)
    matrix = st.session_state.trust_engine.get_trust_matrix().detach().cpu().numpy()
    st.session_state.trust_history = [matrix.copy()]
    st.session_state.messages = []
    st.session_state.last_evaluation = None
    st.session_state.selected_message = None


# =====================================================================
# Sidebar
# =====================================================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding:4px 0 14px 0;">
            <div style="font-size:1.45rem;font-weight:800;">🛡️ MARL</div>
            <div style="color:#91a2b8;font-size:.78rem;margin-top:2px;">
                Cyber Defense Control Center
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Module status")

    st.success("Structured Communication")
    st.caption("Implemented • 7-field schema • 128-D vector")

    st.success("Dynamic Trust")
    st.caption("Implemented • directional trust • forgetting")

    st.warning("Adaptive Action Mask")
    st.caption("Not implemented yet")

    st.warning("Explainable AI")
    st.caption("Not implemented yet")

    st.markdown("---")
    st.markdown("### Review controls")

    if st.button("Reset Demo State", use_container_width=True):
        reset_demo()
        st.rerun()

    st.markdown("---")
    st.caption("Environment")
    st.write("CybORG • CAGE Challenge 4")
    st.caption("Dashboard scope: implemented modules only")

# =====================================================================
# Header
# =====================================================================
st.markdown(
    """
    <div class="topbar">
        <div>
            <div class="title">Multi-Agent Cyber Defense</div>
            <div class="subtitle">
                CC4 coordination monitor — structured communication + dynamic trust
            </div>
        </div>
        <div class="status-box">
            <div class="status-title">PROJECT DEMO</div>
            <div class="status-sub">Implemented modules: <b>2 / 4</b></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =====================================================================
# Metrics
# =====================================================================
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Blue Agents</div>
            <div class="metric-value">5</div>
            <div class="metric-note">CC4 defender agents</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Message Fields</div>
            <div class="metric-value">7</div>
            <div class="metric-note">Structured semantic schema</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m3:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Communication Vector</div>
            <div class="metric-value">128-D</div>
            <div class="metric-note">Encoded receiver input</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m4:
    current = trust_matrix().values
    off_diag = current[~pd.isna(current)]
    non_self = [current[i, j] for i in range(5) for j in range(5) if i != j]
    avg_trust = sum(non_self) / len(non_self)
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Average Directed Trust</div>
            <div class="metric-value">{avg_trust:.2f}</div>
            <div class="metric-note">Range constrained to 0.05 – 0.95</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# =====================================================================
# Main tabs
# =====================================================================
overview_tab, comm_tab, trust_tab, demo_tab = st.tabs(
    ["Overview", "Communication", "Trust", "Interactive Demo"]
)

# =====================================================================
# Overview
# =====================================================================
with overview_tab:
    left, right = st.columns([1.08, 1.25], gap="medium")

    with left:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">
                    <div class="panel-title-main">Defender Network</div>
                    <div class="panel-title-sub">Logical CC4 agent topology</div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        # Schematic topology; no fake host status is implied.
        top = st.columns([1, 1, 1], gap="small")
        with top[0]:
            st.markdown(
                """
                <div class="agent-node active">
                    <div class="agent-id">A0</div>
                    <div class="agent-zone">Restricted A</div>
                    <div class="agent-role">Local Defender</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with top[1]:
            st.markdown(
                """
                <div class="agent-node active">
                    <div class="agent-id">A4</div>
                    <div class="agent-zone">HQ</div>
                    <div class="agent-role">Coordinator</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with top[2]:
            st.markdown(
                """
                <div class="agent-node active">
                    <div class="agent-id">A2</div>
                    <div class="agent-zone">Restricted B</div>
                    <div class="agent-role">Local Defender</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")
        bottom = st.columns([1, 1], gap="small")
        with bottom[0]:
            st.markdown(
                """
                <div class="agent-node active">
                    <div class="agent-id">A1</div>
                    <div class="agent-zone">Operational A</div>
                    <div class="agent-role">Local Defender</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with bottom[1]:
            st.markdown(
                """
                <div class="agent-node active">
                    <div class="agent-id">A3</div>
                    <div class="agent-zone">Operational B</div>
                    <div class="agent-role">Local Defender</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
                <div class="legend">
                    <span><span class="legend-dot" style="background:#4da3ff;"></span>Blue defender</span>
                    <span><span class="legend-dot" style="background:#35d18a;"></span>Implemented communication</span>
                    <span><span class="legend-dot" style="background:#9a7cff;"></span>Trust-weighted reception</span>
                </div>
                </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">
                    <div class="panel-title-main">Recent Communication</div>
                    <div class="panel-title-sub">Latest evaluated messages</div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        if not st.session_state.messages:
            st.markdown(
                """
                <div class="message-card">
                    <div class="message-main">
                        No messages have been evaluated yet.
                    </div>
                    <div class="message-meta">
                        Use <b>Interactive Demo</b> to send a structured message.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            for msg in st.session_state.messages[:5]:
                q = msg["quality"]
                qclass = quality_class(q)
                st.markdown(
                    f"""
                    <div class="message-card">
                        <div class="message-head">
                            <div class="message-route">
                                A{msg['sender']} → A{msg['receiver']}
                            </div>
                            <div class="{qclass}">
                                Quality {q:.2f}
                            </div>
                        </div>
                        <div class="message-main">
                            {msg['event']} • {msg['target_type']} #{msg['target_id']}
                            • {msg['threat']} • {msg['status']}
                        </div>
                        <div class="message-meta">
                            {msg['time']} • {msg['priority']} • trust
                            {msg['trust_before']:.2f} → {msg['trust_after']:.2f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)

    # Lower section
    c1, c2 = st.columns([1, 1], gap="medium")

    with c1:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">
                    <div class="panel-title-main">Trust Matrix</div>
                    <div class="panel-title-sub">sender → receiver</div>
                </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(
            trust_matrix().round(3),
            use_container_width=True,
            height=250,
        )
        st.markdown(
            """
            <div class="section-note">
                Each cell is directional: trust[sender][receiver] means
                how much that receiver currently trusts that sender.
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">
                    <div class="panel-title-main">Trust Trend</div>
                    <div class="panel-title-sub">selected relationship</div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        if len(st.session_state.trust_history) > 1:
            hist_rows = []
            for i, mat in enumerate(st.session_state.trust_history):
                hist_rows.append(
                    {
                        "Evaluation": i,
                        "A0 → A1": mat[0, 1],
                        "A2 → A1": mat[2, 1],
                        "A4 → A1": mat[4, 1],
                    }
                )
            chart_df = pd.DataFrame(hist_rows).set_index("Evaluation")
            st.line_chart(chart_df, height=220)
        else:
            st.caption("Trust history appears here after message evaluations.")

        st.markdown("</div>", unsafe_allow_html=True)

    # Honest scope notice
    s1, s2 = st.columns(2)
    with s1:
        st.markdown(
            """
            <div class="coming-box">
                <div class="coming-title">Adaptive Action Mask</div>
                <div class="coming-text">
                    Coming next. The live dashboard does not claim this module
                    is implemented.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            """
            <div class="coming-box">
                <div class="coming-title">Explainable AI</div>
                <div class="coming-text">
                    Coming next. Natural-language decision explanations are
                    intentionally not shown as implemented functionality.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =====================================================================
# Communication tab
# =====================================================================
with comm_tab:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">
                <div class="panel-title-main">Structured Communication Pipeline</div>
                <div class="panel-title-sub">actual project design</div>
            </div>
            <div class="pipeline">
                <span class="pipe-item">Agent latent</span>
                <span class="pipe-arrow">→</span>
                <span class="pipe-item">Message Decoder</span>
                <span class="pipe-arrow">→</span>
                <span class="pipe-item">7-field StructuredMessage</span>
                <span class="pipe-arrow">→</span>
                <span class="pipe-item">Message Encoder</span>
                <span class="pipe-arrow">→</span>
                <span class="pipe-item">128-D vector</span>
                <span class="pipe-arrow">→</span>
                <span class="pipe-item">Trust-weighted reception</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.05, 1.3], gap="medium")

    with left:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">
                    <div class="panel-title-main">Message Schema</div>
                    <div class="panel-title-sub">7 semantic fields</div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        schema_rows = [
            ("event_type", "What happened?"),
            ("target_type", "HOST / SUBNET / NONE"),
            ("target_id", "Target index in selected vocabulary"),
            ("threat_level", "LOW → CRITICAL"),
            ("confidence", "Continuous value derived from 4 categories"),
            ("status", "UNKNOWN → CONTAINED"),
            ("priority", "LOW → URGENT"),
        ]
        for key, desc in schema_rows:
            st.markdown(
                f"""
                <div class="detail-row">
                    <span class="detail-key">{key}</span>
                    <span class="detail-value">{desc}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">
                    <div class="panel-title-main">Message Log</div>
                    <div class="panel-title-sub">evaluation results</div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.messages:
            log_df = pd.DataFrame(st.session_state.messages)
            st.dataframe(
                log_df[
                    [
                        "time",
                        "sender",
                        "receiver",
                        "event",
                        "target_type",
                        "target_id",
                        "threat",
                        "status",
                        "quality",
                    ]
                ],
                use_container_width=True,
                height=390,
            )
        else:
            st.info("No evaluated messages yet.")

        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================================
# Trust tab
# =====================================================================
with trust_tab:
    left, right = st.columns([1.0, 1.25], gap="medium")

    with left:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">
                    <div class="panel-title-main">Directional Trust</div>
                    <div class="panel-title-sub">Beta-Bernoulli style reputation</div>
                </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(
            trust_matrix().round(3),
            use_container_width=True,
            height=310,
        )
        st.markdown(
            """
            <div class="section-note">
                Correct/useful information contributes positive evidence;
                incorrect/useless information contributes negative evidence.
                Recent evidence has more influence because the mechanism uses
                exponential forgetting.
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">
                    <div class="panel-title-main">Latest Trust Update</div>
                    <div class="panel-title-sub">what changed and why</div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        last = st.session_state.last_evaluation
        msg = st.session_state.selected_message

        if last is None or msg is None:
            st.info("Evaluate a message to see the trust update explanation.")
        else:
            before = msg["trust_before"]
            after = msg["trust_after"]
            delta = after - before

            q = float(last.overall_score)
            st.markdown(
                f"""
                <div style="display:flex;justify-content:space-between;align-items:flex-end;gap:10px;">
                    <div>
                        <div style="color:#91a2b8;font-size:.78rem;">Relationship</div>
                        <div style="font-size:1.35rem;font-weight:800;">
                            A{msg['sender']} → A{msg['receiver']}
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <div style="color:#91a2b8;font-size:.78rem;">Trust</div>
                        <div class="big-quality {'quality-good' if delta > 0 else 'quality-bad' if delta < 0 else 'quality-mid'}">
                            {before:.3f} → {after:.3f}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            st.progress(min(max(q, 0.0), 1.0), text=f"Message quality: {q:.3f}")

            rows = [
                ("Event correctness", last.event_score),
                ("Target correctness", last.target_score),
                ("Threat correctness", last.threat_score),
                ("Status correctness", last.status_score),
                ("Operational usefulness", last.usefulness_score),
            ]

            for label, value in rows:
                shown = "Ungraded" if value is None else f"{float(value):.3f}"
                st.markdown(
                    f"""
                    <div class="detail-row">
                        <span class="detail-key">{label}</span>
                        <span class="detail-value">{shown}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            direction = "increased" if delta > 0 else "decreased" if delta < 0 else "did not change"
            st.markdown(
                f"""
                <div class="section-note" style="margin-top:10px;">
                    The evaluated message caused trust to <b>{direction}</b>
                    because the evaluator's overall message quality was
                    <b>{q:.3f}</b>. The updated trust is then available to
                    weight future communication for that sender/receiver pair.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================================
# Interactive demo tab
# =====================================================================
with demo_tab:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">
                <div class="panel-title-main">Interactive Structured Message Demo</div>
                <div class="panel-title-sub">real evaluator + real trust engine</div>
            </div>
            <div class="section-note">
                Choose a sender, receiver and structured message. Then provide
                the ground truth for the message. The dashboard runs the actual
                MessageEvaluator and updates the actual DynamicTrust object.
                This is a review/demo control, not a live CybORG telemetry feed.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    controls = st.columns(4, gap="small")

    with controls[0]:
        sender = st.selectbox(
            "Sender",
            list(AGENTS.keys()),
            format_func=lambda x: f"A{x} • {AGENTS[x]}",
        )

    with controls[1]:
        receiver = st.selectbox(
            "Receiver",
            list(AGENTS.keys()),
            index=1,
            format_func=lambda x: f"A{x} • {AGENTS[x]}",
        )

    with controls[2]:
        event = st.selectbox(
            "Event",
            list(EventType),
            format_func=lambda x: x.name,
        )

    with controls[3]:
        target_type = st.selectbox(
            "Target type",
            list(TargetType),
            format_func=lambda x: x.name,
        )

    controls2 = st.columns(5, gap="small")

    with controls2[0]:
        target_id = st.number_input(
            "Target ID",
            min_value=0,
            max_value=99,
            value=17,
            step=1,
        )

    with controls2[1]:
        threat = st.selectbox(
            "Threat",
            list(ThreatLevel),
            format_func=lambda x: x.name,
            index=2 if event == EventType.COMPROMISE else 0,
        )

    with controls2[2]:
        confidence_level = st.selectbox(
            "Confidence",
            list(ConfidenceLevel),
            format_func=lambda x: x.name,
            index=2,
        )

    with controls2[3]:
        status = st.selectbox(
            "Status",
            list(HostStatus),
            format_func=lambda x: x.name,
            index=3 if event == EventType.COMPROMISE else 0,
        )

    with controls2[4]:
        priority = st.selectbox(
            "Priority",
            list(Priority),
            format_func=lambda x: x.name,
            index=3 if threat in (ThreatLevel.HIGH, ThreatLevel.CRITICAL) else 1,
        )

    st.write("")

    gt1, gt2, gt3, gt4, gt5 = st.columns(5, gap="small")

    with gt1:
        gt_event = st.selectbox(
            "Ground truth • Event",
            list(EventType),
            format_func=lambda x: x.name,
            key="gt_event",
            index=4 if event == EventType.COMPROMISE else 0,
        )

    with gt2:
        gt_target_type = st.selectbox(
            "Ground truth • Target type",
            list(TargetType),
            format_func=lambda x: x.name,
            key="gt_target_type",
            index=1 if target_type == TargetType.HOST else 0,
        )

    with gt3:
        gt_target_id = st.number_input(
            "Ground truth • Target ID",
            min_value=0,
            max_value=99,
            value=int(target_id),
            step=1,
            key="gt_target_id",
        )

    with gt4:
        gt_threat = st.selectbox(
            "Ground truth • Threat",
            list(ThreatLevel),
            format_func=lambda x: x.name,
            key="gt_threat",
            index=2 if threat == ThreatLevel.HIGH else 0,
        )

    with gt5:
        gt_status = st.selectbox(
            "Ground truth • Status",
            list(HostStatus),
            format_func=lambda x: x.name,
            key="gt_status",
            index=3 if status == HostStatus.COMPROMISED else 0,
        )

    st.write("")

    action_col1, action_col2, action_col3 = st.columns([1, 1, 2])

    with action_col1:
        send_clicked = st.button(
            "Send → Evaluate → Update Trust",
            type="primary",
            use_container_width=True,
        )

    with action_col2:
        demo_clicked = st.button(
            "Run 3-Step Demo Scenario",
            use_container_width=True,
        )

    if send_clicked:
        if sender == receiver:
            st.error("Sender and receiver must be different agents.")
        else:
            message = StructuredMessage(
                event_type=event,
                target_type=target_type,
                target_id=int(target_id),
                threat_level=threat,
                confidence=confidence_level_to_value(confidence_level),
                status=status,
                priority=priority,
            )

            ground_truth = build_ground_truth(
                gt_target_type,
                int(gt_target_id),
                gt_event,
                gt_threat,
                gt_status,
            )

            evaluator = MessageEvaluator()
            evaluation = evaluator.evaluate(
                message=message,
                ground_truth=ground_truth,
                receiver_relevance=1.0,
            )

            before = st.session_state.trust_engine.get_trust(sender, receiver)

            after = st.session_state.trust_engine.update(
                sender=sender,
                receiver=receiver,
                correctness=float(evaluation.overall_score),
                )

            matrix = (
                st.session_state.trust_engine
                .get_trust_matrix()
                .detach()
                .cpu()
                .numpy()
            )
            st.session_state.trust_history.append(matrix.copy())

            st.session_state.last_evaluation = evaluation
            add_message_event(
                sender,
                receiver,
                message,
                evaluation,
                before,
                after,
            )

            st.success(
                f"Message evaluated. Quality = {evaluation.overall_score:.3f}; "
                f"trust A{sender} → A{receiver}: {before:.3f} → {after:.3f}"
            )

    if demo_clicked:
        reset_demo()

        scenario = [
            dict(
                sender=0,
                receiver=1,
                event=EventType.COMPROMISE,
                target_type=TargetType.HOST,
                target_id=17,
                threat=ThreatLevel.HIGH,
                conf=ConfidenceLevel.HIGH,
                status=HostStatus.COMPROMISED,
                priority=Priority.URGENT,
                gt_event=EventType.COMPROMISE,
                gt_type=TargetType.HOST,
                gt_id=17,
                gt_threat=ThreatLevel.HIGH,
                gt_status=HostStatus.COMPROMISED,
            ),
            dict(
                sender=0,
                receiver=1,
                event=EventType.SUSPICIOUS_ACTIVITY,
                target_type=TargetType.HOST,
                target_id=17,
                threat=ThreatLevel.HIGH,
                conf=ConfidenceLevel.HIGH,
                status=HostStatus.SUSPICIOUS,
                priority=Priority.HIGH,
                gt_event=EventType.COMPROMISE,
                gt_type=TargetType.HOST,
                gt_id=17,
                gt_threat=ThreatLevel.HIGH,
                gt_status=HostStatus.COMPROMISED,
            ),
            dict(
                sender=2,
                receiver=1,
                event=EventType.SCAN,
                target_type=TargetType.HOST,
                target_id=22,
                threat=ThreatLevel.MEDIUM,
                conf=ConfidenceLevel.LOW,
                status=HostStatus.NORMAL,
                priority=Priority.MEDIUM,
                gt_event=EventType.NONE,
                gt_type=TargetType.NONE,
                gt_id=0,
                gt_threat=ThreatLevel.LOW,
                gt_status=HostStatus.UNKNOWN,
            ),
        ]

        evaluator = MessageEvaluator()

        for item in scenario:
            message = StructuredMessage(
                event_type=item["event"],
                target_type=item["target_type"],
                target_id=item["target_id"],
                threat_level=item["threat"],
                confidence=confidence_level_to_value(item["conf"]),
                status=item["status"],
                priority=item["priority"],
            )

            truth = build_ground_truth(
                item["gt_type"],
                item["gt_id"],
                item["gt_event"],
                item["gt_threat"],
                item["gt_status"],
            )

            evaluation = evaluator.evaluate(
                message=message,
                ground_truth=truth,
                receiver_relevance=1.0,
            )

            before = st.session_state.trust_engine.get_trust(
                item["sender"], item["receiver"]
            )

            after = st.session_state.trust_engine.update(
                sender=item["sender"],
                receiver=item["receiver"],
                correctness=float(evaluation.overall_score),
            )

            matrix = (
                st.session_state.trust_engine
                .get_trust_matrix()
                .detach()
                .cpu()
                .numpy()
            )
            st.session_state.trust_history.append(matrix.copy())

            st.session_state.last_evaluation = evaluation
            add_message_event(
                item["sender"],
                item["receiver"],
                message,
                evaluation,
                before,
                after,
            )

        st.success("3-step review scenario completed using the real evaluator and trust engine.")

    st.write("")

    result_left, result_right = st.columns([1.0, 1.2], gap="medium")

    with result_left:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">
                    <div class="panel-title-main">Latest Message</div>
                    <div class="panel-title-sub">structured representation</div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        msg = st.session_state.selected_message
        evaluation = st.session_state.last_evaluation

        if msg is None or evaluation is None:
            st.caption("No message evaluated yet.")
        else:
            field_rows = [
                ("Sender", f"A{msg['sender']}"),
                ("Receiver", f"A{msg['receiver']}"),
                ("Event", msg["event"]),
                ("Target", f"{msg['target_type']} #{msg['target_id']}"),
                ("Threat", msg["threat"]),
                ("Confidence", f"{msg['confidence']:.3f}"),
                ("Status", msg["status"]),
                ("Priority", msg["priority"]),
            ]

            for k, v in field_rows:
                st.markdown(
                    f"""
                    <div class="detail-row">
                        <span class="detail-key">{k}</span>
                        <span class="detail-value">{v}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)

    with result_right:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">
                    <div class="panel-title-main">Evaluator + Trust Result</div>
                    <div class="panel-title-sub">decision trace</div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        evaluation = st.session_state.last_evaluation
        msg = st.session_state.selected_message

        if evaluation is None or msg is None:
            st.caption("Results appear after a message is evaluated.")
        else:
            q = float(evaluation.overall_score)
            qclass = quality_class(q)

            st.markdown(
                f"""
                <div class="{qclass}" style="font-size:.78rem;">MESSAGE QUALITY</div>
                <div class="big-quality {qclass}">{q:.3f}</div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")

            st.progress(min(max(q, 0.0), 1.0))

            for label, value in [
                ("Event correctness", evaluation.event_score),
                ("Target correctness", evaluation.target_score),
                ("Threat correctness", evaluation.threat_score),
                ("Status correctness", evaluation.status_score),
                ("Operational usefulness", evaluation.usefulness_score),
            ]:
                shown = "Ungraded" if value is None else f"{float(value):.3f}"
                st.markdown(
                    f"""
                    <div class="detail-row">
                        <span class="detail-key">{label}</span>
                        <span class="detail-value">{shown}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"""
                <div class="detail-row">
                    <span class="detail-key">Trust before</span>
                    <span class="detail-value">{msg['trust_before']:.3f}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-key">Trust after</span>
                    <span class="detail-value">{msg['trust_after']:.3f}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================================
# Footer
# =====================================================================
st.markdown(
    """
    <div class="footer-note">
        CC4 MARL review console • Focused on implemented Structured Communication
        and Dynamic Trust • Action Mask and Explainable AI are intentionally shown
        as pending.
    </div>
    """,
    unsafe_allow_html=True,
)
