"""
app.py
Elite Athlete Monitoring and Injury Risk Dashboard
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Athlete Performance Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — CLEAN, DARK, PROFESSIONAL
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .stApp {
        background-color: #0d0f14;
        color: #e2e8f0;
    }

    section[data-testid="stSidebar"] {
        background-color: #111318;
        border-right: 1px solid #1e2330;
    }

    .metric-card {
        background: #141720;
        border: 1px solid #1e2330;
        border-radius: 6px;
        padding: 18px 20px;
        margin-bottom: 10px;
    }

    .metric-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 6px;
        font-family: 'IBM Plex Mono', monospace;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 600;
        color: #f1f5f9;
        font-family: 'IBM Plex Mono', monospace;
        line-height: 1;
    }

    .metric-sub {
        font-size: 11px;
        color: #475569;
        margin-top: 4px;
        font-family: 'IBM Plex Mono', monospace;
    }

    .section-header {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #475569;
        padding: 24px 0 10px 0;
        border-bottom: 1px solid #1e2330;
        margin-bottom: 16px;
        font-family: 'IBM Plex Mono', monospace;
    }

    .risk-high {
        background: #1a0a0a;
        border: 1px solid #7f1d1d;
        border-left: 3px solid #ef4444;
        color: #fca5a5;
        padding: 14px 18px;
        border-radius: 4px;
        font-size: 13px;
        margin: 10px 0;
    }

    .risk-moderate {
        background: #14100a;
        border: 1px solid #713f12;
        border-left: 3px solid #f59e0b;
        color: #fcd34d;
        padding: 14px 18px;
        border-radius: 4px;
        font-size: 13px;
        margin: 10px 0;
    }

    .risk-low {
        background: #0a1410;
        border: 1px solid #14532d;
        border-left: 3px solid #22c55e;
        color: #86efac;
        padding: 14px 18px;
        border-radius: 4px;
        font-size: 13px;
        margin: 10px 0;
    }

    .athlete-header {
        font-size: 22px;
        font-weight: 600;
        color: #f8fafc;
        letter-spacing: -0.02em;
        margin-bottom: 2px;
    }

    .athlete-sub {
        font-size: 12px;
        color: #64748b;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.05em;
        margin-bottom: 20px;
    }

    .phase-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 3px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-family: 'IBM Plex Mono', monospace;
    }

    .stSelectbox label, .stRadio label {
        font-size: 11px !important;
        font-weight: 600 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: #64748b !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }

    .stPlotlyChart {
        border: 1px solid #1e2330;
        border-radius: 6px;
        overflow: hidden;
    }

    div[data-testid="metric-container"] {
        background: #141720;
        border: 1px solid #1e2330;
        border-radius: 6px;
        padding: 16px;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: #111318;
        border-bottom: 1px solid #1e2330;
        gap: 0;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #475569;
        padding: 10px 20px;
        border-radius: 0;
        border-bottom: 2px solid transparent;
    }

    .stTabs [aria-selected="true"] {
        color: #e2e8f0 !important;
        border-bottom: 2px solid #3b82f6 !important;
        background: transparent !important;
    }

    hr {
        border-color: #1e2330;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CHART THEME — CONSISTENT DARK PLOTLY
# ============================================================
CHART_BG   = "#0d0f14"
CHART_PAPER = "#0d0f14"
FONT_COLOR = "#94a3b8"
GRID_COLOR = "#1e2330"
LINE_COLOR = "#334155"

def apply_theme(fig, title="", height=300):
    fig.update_layout(
        title=dict(
            text=title.upper(),
            font=dict(family="IBM Plex Mono", size=10, color="#475569"),
            x=0,
            xref="paper",
            pad=dict(l=4, t=4)
        ),
        paper_bgcolor=CHART_PAPER,
        plot_bgcolor=CHART_BG,
        font=dict(family="IBM Plex Sans", size=11, color=FONT_COLOR),
        height=height,
        margin=dict(l=12, r=12, t=36, b=12),
        xaxis=dict(
            gridcolor=GRID_COLOR,
            linecolor=LINE_COLOR,
            tickfont=dict(family="IBM Plex Mono", size=10),
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            linecolor=LINE_COLOR,
            tickfont=dict(family="IBM Plex Mono", size=10),
            showgrid=True,
            zeroline=False
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(family="IBM Plex Mono", size=10)
        ),
        hoverlabel=dict(
            bgcolor="#1e2330",
            font=dict(family="IBM Plex Mono", size=11),
            bordercolor="#334155"
        )
    )
    fig.update_layout(height=height)
    return fig

# Color palette
BLUE    = "#3b82f6"
GREEN   = "#22c55e"
AMBER   = "#f59e0b"
RED     = "#ef4444"
PURPLE  = "#a855f7"
CYAN    = "#06b6d4"
ORANGE  = "#f97316"

# ============================================================
# LOAD DATA AND MODELS
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("athlete_bioplausible_dataset_FINAL.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_resource
def load_models():
    models = {}
    required = [
        "model_female.pkl",
        "model_male.pkl",
        "female_feature_cols.pkl",
        "male_feature_cols.pkl",
        "female_importance.csv",
        "male_importance.csv",
    ]
    for f in required:
        if not os.path.exists(f):
            return None
    models["female"]      = joblib.load("model_female.pkl")
    models["male"]        = joblib.load("model_male.pkl")
    models["female_cols"] = joblib.load("female_feature_cols.pkl")
    models["male_cols"]   = joblib.load("male_feature_cols.pkl")
    models["female_imp"]  = pd.read_csv("female_importance.csv")
    models["male_imp"]    = pd.read_csv("male_importance.csv")
    return models

# ---- Check files exist ----
if not os.path.exists("athlete_bioplausible_dataset_FINAL.csv"):
    st.error("Dataset file not found. Make sure athlete_bioplausible_dataset_FINAL.csv is in this folder.")
    st.stop()

df_all = load_data()
models = load_models()

if models is None:
    st.error("Trained model files not found. Please run: python3 train_models.py")
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 20px 0; border-bottom:1px solid #1e2330; margin-bottom:20px;'>
        <div style='font-family:IBM Plex Mono; font-size:13px; font-weight:600; color:#f1f5f9; letter-spacing:0.05em;'>
            ATHLETE PERFORMANCE<br>INTELLIGENCE
        </div>
        <div style='font-family:IBM Plex Mono; font-size:10px; color:#475569; margin-top:4px;'>
            Sports Science Research System
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="metric-label">Athlete Sex</div>', unsafe_allow_html=True)
    sex_choice = st.radio(
        "Athlete Sex",
        options=["Male", "Female"],
        label_visibility="collapsed"
    )
    sex_code = "M" if sex_choice == "Male" else "F"

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

    athlete_list = sorted(df_all[df_all["Sex"] == sex_code]["Athlete_ID"].unique())

    st.markdown('<div class="metric-label">Athlete ID</div>', unsafe_allow_html=True)
    selected_id = st.selectbox(
        "Athlete ID",
        options=athlete_list,
        label_visibility="collapsed"
    )

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

    # Athlete quick profile
    ath_df = df_all[df_all["Athlete_ID"] == selected_id].copy()
    sport  = ath_df["Sport_Type"].iloc[0]
    age    = ath_df["Age"].iloc[0]
    weeks  = ath_df["Week"].nunique()

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Athlete Profile</div>
        <div style='font-family:IBM Plex Mono; font-size:13px; color:#e2e8f0; margin-top:4px;'>
            {selected_id}
        </div>
        <div style='font-size:11px; color:#64748b; margin-top:8px;'>
            Sex &nbsp;&nbsp;&nbsp; {sex_choice}<br>
            Age &nbsp;&nbsp;&nbsp; {age} years<br>
            Sport &nbsp;&nbsp; {sport}<br>
            Data &nbsp;&nbsp;&nbsp; {weeks} weeks tracked
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Week filter
    st.markdown('<div class="metric-label" style="margin-top:12px;">Select Week</div>', unsafe_allow_html=True)
    all_weeks = sorted(ath_df["Week"].unique())
    week_choice = st.selectbox(
        "Week",
        options=["All Weeks"] + [f"Week {w}" for w in all_weeks],
        label_visibility="collapsed"
    )

    st.markdown("""
    <div style='margin-top:auto; padding-top:40px; font-family:IBM Plex Mono;
                font-size:10px; color:#334155; border-top:1px solid #1e2330; margin-top:60px; padding-top:16px;'>
        v2.0 — May 2026<br>
        Bioplausible Dataset
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FILTER DATA BY WEEK
# ============================================================
if week_choice == "All Weeks":
    ath_filtered = ath_df.copy()
else:
    wk = int(week_choice.split(" ")[1])
    ath_filtered = ath_df[ath_df["Week"] == wk].copy()

# Most recent row for ML prediction
latest = ath_df.sort_values("Day_Number").iloc[-1]

# ============================================================
# MAIN CONTENT
# ============================================================

# ---- Athlete Header ----
phase_text = ""
if sex_code == "F":
    phase = latest["Cycle_Phase"]
    phase_colors = {
        "Menstrual":   ("#1e1a2e", "#a855f7"),
        "Follicular":  ("#0a1a2e", "#3b82f6"),
        "Ovulatory":   ("#0a1e14", "#22c55e"),
        "Luteal":      ("#1e130a", "#f97316"),
    }
    bg, fg = phase_colors.get(phase, ("#1e2330", "#94a3b8"))
    phase_text = f"""
    <span class="phase-badge" style="background:{bg}; color:{fg}; border:1px solid {fg}40; margin-left:10px;">
        {phase}
    </span>
    """

acwr_val   = round(latest["ACWR"], 3)
acwr_zone  = latest["ACWR_Zone"]

st.markdown(f"""
<div class="athlete-header">{selected_id} {phase_text}</div>
<div class="athlete-sub">{sport.upper()} &nbsp;/&nbsp; {sex_choice.upper()} &nbsp;/&nbsp; AGE {age} &nbsp;/&nbsp; {week_choice.upper()}</div>
""", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Load & ACWR",
    "Recovery",
    "Female Physiology",
    "Squad View",
])

# ===========================================================
# TAB 1 — OVERVIEW
# ===========================================================
with tab1:

    # ---- AI Risk Prediction ----
    st.markdown('<div class="section-header">Injury Risk Assessment — AI Model</div>', unsafe_allow_html=True)

    if sex_code == "F":
        row = latest.copy()
        cycle_dummies = {f"Phase_{p}": 0 for p in ["Follicular", "Luteal", "Male_NA", "Menstrual", "Ovulatory"]}
        phase_key = f"Phase_{row['Cycle_Phase']}"
        if phase_key in cycle_dummies:
            cycle_dummies[phase_key] = 1

        shared_vals = {col: row[col] for col in models["female_cols"] if col in row.index}
        input_dict = {**shared_vals, **cycle_dummies}
        input_df = pd.DataFrame([input_dict]).reindex(columns=models["female_cols"], fill_value=0)
        risk_prob = models["female"].predict_proba(input_df)[0][1]
    else:
        input_vals = {col: latest[col] for col in models["male_cols"]}
        input_df = pd.DataFrame([input_vals])
        risk_prob = models["male"].predict_proba(input_df)[0][1]

    risk_pct = round(risk_prob * 100, 1)

    col_risk, col_gauge = st.columns([1, 1])

    with col_risk:
        # Risk badge
        if risk_pct > 60:
            badge = "risk-high"
            verdict = f"HIGH RISK — {risk_pct}% injury probability in next training block."
            rec = "Immediate load reduction recommended. Prioritise active recovery, sleep, and clinical review."
        elif risk_pct > 30:
            badge = "risk-moderate"
            verdict = f"MODERATE RISK — {risk_pct}% injury probability. Monitor closely."
            rec = "Reduce acute load. Check HRV trend. Ensure minimum 7h sleep and mood stability."
        else:
            badge = "risk-low"
            verdict = f"LOW RISK — {risk_pct}% injury probability. Athlete is well-prepared."
            rec = "Maintain current training structure. Continue monitoring daily HRV and perceived fatigue."

        st.markdown(f"""
        <div class="{badge}">
            <div style='font-weight:600; font-family:IBM Plex Mono; font-size:13px; margin-bottom:6px;'>{verdict}</div>
            <div style='font-size:12px; opacity:0.8;'>{rec}</div>
        </div>
        """, unsafe_allow_html=True)

        # Key inputs summary
        st.markdown("""<div style='height:10px'></div>""", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ACWR", f"{round(latest['ACWR'], 3)}", help="Acute:Chronic Workload Ratio. Optimal: 0.8–1.3")
        c2.metric("HRV", f"{round(latest['HRV_rMSSD_ms'], 1)} ms", help="rMSSD. Higher is better.")
        c3.metric("Fatigue", f"{round(latest['Perceived_Fatigue'], 1)}/10", help="Perceived fatigue. Lower is better.")
        c4.metric("Sleep", f"{round(latest['Sleep_Quality'], 1)}/10", help="Sleep quality score.")

    with col_gauge:
        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            number={"suffix": "%", "font": {"family": "IBM Plex Mono", "size": 28, "color": "#f1f5f9"}},
            gauge={
                "axis": {"range": [0, 100], "tickfont": {"family": "IBM Plex Mono", "size": 9}, "tickcolor": "#475569"},
                "bar": {"color": RED if risk_pct > 60 else AMBER if risk_pct > 30 else GREEN, "thickness": 0.18},
                "bgcolor": "#141720",
                "bordercolor": "#1e2330",
                "steps": [
                    {"range": [0, 30], "color": "#0a1410"},
                    {"range": [30, 60], "color": "#14100a"},
                    {"range": [60, 100], "color": "#1a0a0a"},
                ],
                "threshold": {
                    "line": {"color": "#334155", "width": 1},
                    "thickness": 0.7,
                    "value": risk_pct,
                },
            },
            title={"text": "INJURY RISK PROBABILITY", "font": {"family": "IBM Plex Mono", "size": 9, "color": "#475569"}},
        ))
        apply_theme(fig_gauge, height=220)
        fig_gauge.update_layout(margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_gauge, width="stretch")

    # ---- Feature Importance ----
    st.markdown('<div class="section-header">Key Risk Drivers</div>', unsafe_allow_html=True)

    imp_df = models["female_imp"] if sex_code == "F" else models["male_imp"]
    imp_df = imp_df.head(8).sort_values("Importance")

    # Clean feature labels
    label_map = {
        "HRV_Deviation_ms": "HRV Deviation",
        "Perceived_Fatigue": "Perceived Fatigue",
        "Sleep_Quality": "Sleep Quality",
        "ACWR": "ACWR",
        "Acute_Load_7d_AU": "Acute Load (7d)",
        "Resting_HR_bpm": "Resting HR",
        "Mood_Score": "Mood Score",
        "Muscle_Soreness": "Muscle Soreness",
        "HRV_rMSSD_ms": "HRV rMSSD",
        "Chronic_Load_28d_AU": "Chronic Load (28d)",
        "Training_Load_AU": "Daily Training Load",
        "Session_RPE": "Session RPE",
        "Phase_Luteal": "Phase: Luteal",
        "Phase_Menstrual": "Phase: Menstrual",
        "Phase_Follicular": "Phase: Follicular",
        "Phase_Ovulatory": "Phase: Ovulatory",
    }
    imp_df["Feature"] = imp_df["Feature"].map(lambda x: label_map.get(x, x))

    colors = [GREEN if v < 0.10 else AMBER if v < 0.18 else RED for v in imp_df["Importance"]]

    fig_imp = go.Figure(go.Bar(
        x=imp_df["Importance"],
        y=imp_df["Feature"],
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v:.3f}" for v in imp_df["Importance"]],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=9, color="#64748b"),
    ))
    apply_theme(fig_imp, title="Feature Importance — Gradient Boosting Model", height=280)
    fig_imp.update_layout(yaxis=dict(tickfont=dict(size=11)))
    st.plotly_chart(fig_imp, width="stretch")

    # ---- Daily metrics table ----
    st.markdown('<div class="section-header">Session Log</div>', unsafe_allow_html=True)
    display_cols = [
        "Date", "Week", "Day_of_Week", "Training_Load_AU", "Session_RPE",
        "ACWR", "ACWR_Zone", "HRV_rMSSD_ms", "Perceived_Fatigue",
        "Sleep_Quality", "Mood_Score", "Muscle_Soreness", "Injury_Risk"
    ]
    if sex_code == "F":
        display_cols += ["Cycle_Phase", "Cycle_Day"]

    tbl = ath_filtered[display_cols].copy()
    tbl["Date"] = tbl["Date"].dt.strftime("%d %b")
    tbl.columns = [c.replace("_", " ") for c in tbl.columns]

    def color_injury(val):
        if val == 1:
            return "background-color: #1a0a0a; color: #ef4444;"
        return ""
    def color_acwr(val):
        if isinstance(val, str):
            if val == "High" or val == "Spike":
                return "color: #f59e0b;"
        return ""

    styled = tbl.style.map(color_injury, subset=["Injury Risk"])
    st.dataframe(styled, width="stretch", height=260)

# ===========================================================
# TAB 2 — LOAD AND ACWR
# ===========================================================
with tab2:
    st.markdown('<div class="section-header">Training Load Analysis</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # ACWR trend line
        fig_acwr = go.Figure()
        fig_acwr.add_hrect(y0=0.8, y1=1.3, fillcolor=GREEN, opacity=0.05, layer="below", line_width=0)
        fig_acwr.add_hline(y=1.3, line_dash="dash", line_color=AMBER, line_width=1,
                           annotation_text="Upper Threshold", annotation_font=dict(size=9, family="IBM Plex Mono"))
        fig_acwr.add_hline(y=0.8, line_dash="dash", line_color=BLUE, line_width=1,
                           annotation_text="Lower Threshold", annotation_font=dict(size=9, family="IBM Plex Mono"))

        acwr_colors = [RED if v > 1.3 else AMBER if v > 1.1 else GREEN if v >= 0.8 else BLUE
                       for v in ath_filtered["ACWR"]]

        fig_acwr.add_trace(go.Scatter(
            x=ath_filtered["Day_Number"],
            y=ath_filtered["ACWR"],
            mode="lines+markers",
            line=dict(color=BLUE, width=2),
            marker=dict(color=acwr_colors, size=7, line=dict(color="#0d0f14", width=1)),
            name="ACWR",
            hovertemplate="Day %{x}<br>ACWR: %{y:.3f}<extra></extra>",
        ))
        apply_theme(fig_acwr, title="ACWR — Acute:Chronic Workload Ratio", height=280)
        st.plotly_chart(fig_acwr, width="stretch")

    with col_b:
        # Acute vs Chronic load
        fig_loads = go.Figure()
        fig_loads.add_trace(go.Scatter(
            x=ath_filtered["Day_Number"],
            y=ath_filtered["Acute_Load_7d_AU"],
            mode="lines",
            line=dict(color=ORANGE, width=2),
            name="Acute Load (7d)",
            fill="tozeroy",
            fillcolor=f"rgba(249,115,22,0.06)",
        ))
        fig_loads.add_trace(go.Scatter(
            x=ath_filtered["Day_Number"],
            y=ath_filtered["Chronic_Load_28d_AU"],
            mode="lines",
            line=dict(color=CYAN, width=2),
            name="Chronic Load (28d)",
        ))
        apply_theme(fig_loads, title="Acute vs Chronic Load (AU)", height=280)
        fig_loads.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_loads, width="stretch")

    # Daily training load bar
    st.markdown('<div class="section-header">Daily Training Load</div>', unsafe_allow_html=True)

    bar_colors = [RED if r == 1 else BLUE for r in ath_filtered["Injury_Risk"]]

    fig_load_bar = go.Figure(go.Bar(
        x=ath_filtered["Day_Number"],
        y=ath_filtered["Training_Load_AU"],
        marker_color=bar_colors,
        marker_line_width=0,
        hovertemplate="Day %{x}<br>Load: %{y:.0f} AU<extra></extra>",
    ))
    fig_load_bar.add_trace(go.Scatter(
        x=ath_filtered["Day_Number"],
        y=ath_filtered["Session_RPE"] * 50,
        mode="lines",
        line=dict(color=AMBER, width=1.5, dash="dot"),
        name="RPE (scaled)",
        yaxis="y2",
    ))
    apply_theme(fig_load_bar, title="Daily Training Load (AU) — Red bars indicate injury risk days", height=300)
    fig_load_bar.update_layout(
        yaxis2=dict(
            overlaying="y",
            side="right",
            showgrid=False,
            tickfont=dict(family="IBM Plex Mono", size=9),
        ),
        showlegend=True
    )
    st.plotly_chart(fig_load_bar, width="stretch")

    # ACWR zone distribution
    col_c, col_d = st.columns(2)
    with col_c:
        zone_counts = ath_filtered["ACWR_Zone"].value_counts().reset_index()
        zone_counts.columns = ["Zone", "Count"]
        zone_color_map = {"Optimal": GREEN, "High": AMBER, "Low": BLUE, "Spike": RED}
        zone_colors = [zone_color_map.get(z, CYAN) for z in zone_counts["Zone"]]

        fig_zone = go.Figure(go.Bar(
            x=zone_counts["Zone"],
            y=zone_counts["Count"],
            marker_color=zone_colors,
            marker_line_width=0,
            text=zone_counts["Count"],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=10),
        ))
        apply_theme(fig_zone, title="ACWR Zone Distribution", height=250)
        st.plotly_chart(fig_zone, width="stretch")

    with col_d:
        # RPE vs Training Load scatter
        fig_rpe = px.scatter(
            ath_filtered,
            x="Session_RPE",
            y="Training_Load_AU",
            color="Injury_Risk",
            color_discrete_map={0: BLUE, 1: RED},
            labels={"Session_RPE": "Session RPE", "Training_Load_AU": "Training Load (AU)"},
        )
        apply_theme(fig_rpe, title="Session RPE vs Training Load", height=250)
        fig_rpe.update_traces(marker=dict(size=6, opacity=0.7, line=dict(width=0)))
        st.plotly_chart(fig_rpe, width="stretch")

# ===========================================================
# TAB 3 — RECOVERY
# ===========================================================
with tab3:
    st.markdown('<div class="section-header">Recovery and Readiness Indicators</div>', unsafe_allow_html=True)

    col_e, col_f = st.columns(2)

    with col_e:
        # HRV trend with baseline
        fig_hrv = go.Figure()
        fig_hrv.add_trace(go.Scatter(
            x=ath_filtered["Day_Number"],
            y=ath_filtered["Baseline_HRV_ms"],
            mode="lines",
            line=dict(color=LINE_COLOR, width=1.5, dash="dot"),
            name="Baseline HRV",
        ))
        fig_hrv.add_trace(go.Scatter(
            x=ath_filtered["Day_Number"],
            y=ath_filtered["HRV_rMSSD_ms"],
            mode="lines+markers",
            line=dict(color=GREEN, width=2),
            marker=dict(size=5),
            name="Daily HRV",
            fill="tonexty",
            fillcolor="rgba(34,197,94,0.05)",
        ))
        apply_theme(fig_hrv, title="HRV rMSSD — Daily vs Baseline (ms)", height=270)
        fig_hrv.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_hrv, width="stretch")

    with col_f:
        # HRV deviation
        dev_colors = [RED if v < -8 else AMBER if v < -3 else GREEN for v in ath_filtered["HRV_Deviation_ms"]]
        fig_dev = go.Figure(go.Bar(
            x=ath_filtered["Day_Number"],
            y=ath_filtered["HRV_Deviation_ms"],
            marker_color=dev_colors,
            marker_line_width=0,
            hovertemplate="Day %{x}<br>Deviation: %{y:.1f} ms<extra></extra>",
        ))
        fig_dev.add_hline(y=0, line_color=LINE_COLOR, line_width=1)
        apply_theme(fig_dev, title="HRV Deviation from Baseline (ms)", height=270)
        st.plotly_chart(fig_dev, width="stretch")

    # Wellness composite
    st.markdown('<div class="section-header">Athlete Wellness Composite</div>', unsafe_allow_html=True)

    fig_well = go.Figure()
    fig_well.add_trace(go.Scatter(
        x=ath_filtered["Day_Number"],
        y=ath_filtered["Sleep_Quality"],
        mode="lines",
        line=dict(color=CYAN, width=2),
        name="Sleep Quality",
    ))
    fig_well.add_trace(go.Scatter(
        x=ath_filtered["Day_Number"],
        y=ath_filtered["Mood_Score"],
        mode="lines",
        line=dict(color=PURPLE, width=2),
        name="Mood Score",
    ))
    fig_well.add_trace(go.Scatter(
        x=ath_filtered["Day_Number"],
        y=ath_filtered["Perceived_Fatigue"],
        mode="lines",
        line=dict(color=AMBER, width=2),
        name="Perceived Fatigue",
    ))
    fig_well.add_trace(go.Scatter(
        x=ath_filtered["Day_Number"],
        y=ath_filtered["Muscle_Soreness"],
        mode="lines",
        line=dict(color=ORANGE, width=1.5, dash="dot"),
        name="Muscle Soreness",
    ))
    apply_theme(fig_well, title="Daily Wellness Metrics (1–10 Scale)", height=300)
    fig_well.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_well, width="stretch")

    # Resting HR trend
    col_g, col_h = st.columns(2)
    with col_g:
        fig_hr = go.Figure(go.Scatter(
            x=ath_filtered["Day_Number"],
            y=ath_filtered["Resting_HR_bpm"],
            mode="lines+markers",
            line=dict(color=RED, width=2),
            marker=dict(size=5),
            fill="tozeroy",
            fillcolor="rgba(239,68,68,0.05)",
        ))
        apply_theme(fig_hr, title="Resting Heart Rate (bpm)", height=240)
        st.plotly_chart(fig_hr, width="stretch")

    with col_h:
        # Fatigue vs Sleep scatter
        fig_fs = px.scatter(
            ath_filtered,
            x="Sleep_Quality",
            y="Perceived_Fatigue",
            color="Injury_Risk",
            color_discrete_map={0: BLUE, 1: RED},
            trendline="ols",
            labels={"Sleep_Quality": "Sleep Quality", "Perceived_Fatigue": "Perceived Fatigue"},
        )
        apply_theme(fig_fs, title="Sleep Quality vs Perceived Fatigue", height=240)
        fig_fs.update_traces(marker=dict(size=6, opacity=0.7, line=dict(width=0)))
        st.plotly_chart(fig_fs, width="stretch")

# ===========================================================
# TAB 4 — FEMALE PHYSIOLOGY
# ===========================================================
with tab4:
    if sex_code == "M":
        st.markdown("""
        <div style='padding:40px; text-align:center; color:#475569; font-family:IBM Plex Mono; font-size:13px;'>
            This module is available for female athletes only.<br>
            <span style='font-size:11px; color:#334155;'>Select a female athlete from the sidebar to view menstrual cycle analysis.</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-header">Menstrual Cycle Performance Analysis</div>', unsafe_allow_html=True)

        f_data = ath_filtered[ath_filtered["Cycle_Phase"] != "Male_NA"].copy()

        phase_colors_map = {
            "Menstrual": PURPLE,
            "Follicular": BLUE,
            "Ovulatory": GREEN,
            "Luteal": ORANGE,
        }

        col_i, col_j = st.columns(2)

        with col_i:
            # HRV by cycle phase
            phase_hrv = f_data.groupby("Cycle_Phase")["HRV_rMSSD_ms"].mean().reset_index()
            fig_ph_hrv = go.Figure(go.Bar(
                x=phase_hrv["Cycle_Phase"],
                y=phase_hrv["HRV_rMSSD_ms"],
                marker_color=[phase_colors_map.get(p, CYAN) for p in phase_hrv["Cycle_Phase"]],
                marker_line_width=0,
                text=[f"{v:.1f}" for v in phase_hrv["HRV_rMSSD_ms"]],
                textposition="outside",
                textfont=dict(family="IBM Plex Mono", size=10),
            ))
            apply_theme(fig_ph_hrv, title="Mean HRV rMSSD by Cycle Phase (ms)", height=270)
            st.plotly_chart(fig_ph_hrv, width="stretch")

        with col_j:
            # Fatigue by cycle phase
            phase_fat = f_data.groupby("Cycle_Phase")["Perceived_Fatigue"].mean().reset_index()
            fig_ph_fat = go.Figure(go.Bar(
                x=phase_fat["Cycle_Phase"],
                y=phase_fat["Perceived_Fatigue"],
                marker_color=[phase_colors_map.get(p, CYAN) for p in phase_fat["Cycle_Phase"]],
                marker_line_width=0,
                text=[f"{v:.2f}" for v in phase_fat["Perceived_Fatigue"]],
                textposition="outside",
                textfont=dict(family="IBM Plex Mono", size=10),
            ))
            apply_theme(fig_ph_fat, title="Mean Perceived Fatigue by Cycle Phase", height=270)
            st.plotly_chart(fig_ph_fat, width="stretch")

        col_k, col_l = st.columns(2)
        with col_k:
            # Sleep by cycle phase
            phase_sl = f_data.groupby("Cycle_Phase")["Sleep_Quality"].mean().reset_index()
            fig_ph_sl = go.Figure(go.Bar(
                x=phase_sl["Cycle_Phase"],
                y=phase_sl["Sleep_Quality"],
                marker_color=[phase_colors_map.get(p, CYAN) for p in phase_sl["Cycle_Phase"]],
                marker_line_width=0,
                text=[f"{v:.2f}" for v in phase_sl["Sleep_Quality"]],
                textposition="outside",
                textfont=dict(family="IBM Plex Mono", size=10),
            ))
            apply_theme(fig_ph_sl, title="Mean Sleep Quality by Cycle Phase", height=270)
            st.plotly_chart(fig_ph_sl, width="stretch")

        with col_l:
            # Injury rate by cycle phase
            phase_inj = f_data.groupby("Cycle_Phase")["Injury_Risk"].mean().reset_index()
            phase_inj["Injury_Rate_Pct"] = (phase_inj["Injury_Risk"] * 100).round(1)
            fig_ph_inj = go.Figure(go.Bar(
                x=phase_inj["Cycle_Phase"],
                y=phase_inj["Injury_Rate_Pct"],
                marker_color=[RED if v > 20 else AMBER if v > 10 else GREEN
                              for v in phase_inj["Injury_Rate_Pct"]],
                marker_line_width=0,
                text=[f"{v:.1f}%" for v in phase_inj["Injury_Rate_Pct"]],
                textposition="outside",
                textfont=dict(family="IBM Plex Mono", size=10),
            ))
            apply_theme(fig_ph_inj, title="Injury Incidence Rate by Cycle Phase (%)", height=270)
            st.plotly_chart(fig_ph_inj, width="stretch")

        # Cycle phase over training days
        st.markdown('<div class="section-header">Cycle Phase Timeline</div>', unsafe_allow_html=True)

        phase_int = {"Menstrual": 1, "Follicular": 2, "Ovulatory": 3, "Luteal": 4}
        f_data["Phase_Num"] = f_data["Cycle_Phase"].map(phase_int)

        fig_timeline = px.scatter(
            f_data,
            x="Day_Number",
            y="Phase_Num",
            color="Cycle_Phase",
            color_discrete_map=phase_colors_map,
            size_max=8,
        )
        fig_timeline.update_traces(marker=dict(size=9, symbol="square"))
        apply_theme(fig_timeline, title="Menstrual Cycle Phase Across Training Period", height=220)
        fig_timeline.update_yaxes(
            tickvals=[1, 2, 3, 4],
            ticktext=["Menstrual", "Follicular", "Ovulatory", "Luteal"],
        )
        fig_timeline.update_layout(showlegend=False)
        st.plotly_chart(fig_timeline, width="stretch")

        # Training Load by phase
        phase_load = f_data.groupby("Cycle_Phase")[["Training_Load_AU", "ACWR", "Muscle_Soreness"]].mean().round(2)
        st.markdown('<div class="section-header">Phase-Level Performance Summary Table</div>', unsafe_allow_html=True)
        st.dataframe(phase_load, width="stretch")

# ===========================================================
# TAB 5 — SQUAD VIEW
# ===========================================================
with tab5:
    st.markdown('<div class="section-header">Full Squad Overview</div>', unsafe_allow_html=True)

    # Filter squad by sex
    squad_sex = st.radio(
        "Squad Sex Filter",
        options=["All Athletes", "Male Only", "Female Only"],
        horizontal=True,
        label_visibility="collapsed"
    )

    if squad_sex == "Male Only":
        squad_df = df_all[df_all["Sex"] == "M"].copy()
    elif squad_sex == "Female Only":
        squad_df = df_all[df_all["Sex"] == "F"].copy()
    else:
        squad_df = df_all.copy()

    # Latest data per athlete
    latest_all = squad_df.sort_values("Day_Number").groupby("Athlete_ID").last().reset_index()

    # Predict risk for all athletes
    def predict_batch(row):
        try:
            if row["Sex"] == "F":
                cycle_dummies = {f"Phase_{p}": 0 for p in ["Follicular", "Luteal", "Male_NA", "Menstrual", "Ovulatory"]}
                phase_key = f"Phase_{row['Cycle_Phase']}"
                if phase_key in cycle_dummies:
                    cycle_dummies[phase_key] = 1
                shared_vals = {col: row[col] for col in models["female_cols"] if col in row.index}
                inp = pd.DataFrame([{**shared_vals, **cycle_dummies}]).reindex(columns=models["female_cols"], fill_value=0)
                return round(models["female"].predict_proba(inp)[0][1] * 100, 1)
            else:
                inp = pd.DataFrame([{col: row[col] for col in models["male_cols"]}])
                return round(models["male"].predict_proba(inp)[0][1] * 100, 1)
        except Exception:
            return 0.0

    latest_all["AI_Risk_Pct"] = latest_all.apply(predict_batch, axis=1)
    latest_all["Risk_Level"] = latest_all["AI_Risk_Pct"].apply(
        lambda x: "High" if x > 60 else "Moderate" if x > 30 else "Low"
    )

    # Squad risk summary
    col_m, col_n, col_o = st.columns(3)
    high_count = (latest_all["Risk_Level"] == "High").sum()
    mod_count  = (latest_all["Risk_Level"] == "Moderate").sum()
    low_count  = (latest_all["Risk_Level"] == "Low").sum()

    col_m.metric("High Risk Athletes", high_count)
    col_n.metric("Moderate Risk Athletes", mod_count)
    col_o.metric("Low Risk Athletes", low_count)

    # Squad risk bar chart
    squad_sorted = latest_all.sort_values("AI_Risk_Pct", ascending=True)
    risk_colors_squad = [RED if r == "High" else AMBER if r == "Moderate" else GREEN
                         for r in squad_sorted["Risk_Level"]]

    fig_squad = go.Figure(go.Bar(
        x=squad_sorted["AI_Risk_Pct"],
        y=squad_sorted["Athlete_ID"],
        orientation="h",
        marker_color=risk_colors_squad,
        marker_line_width=0,
        text=[f"{v:.1f}%" for v in squad_sorted["AI_Risk_Pct"]],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=9, color="#64748b"),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig_squad.add_vline(x=30, line_dash="dash", line_color=AMBER, line_width=1)
    fig_squad.add_vline(x=60, line_dash="dash", line_color=RED, line_width=1)
    apply_theme(fig_squad, title="Squad Injury Risk — AI Prediction (%)", height=max(350, len(squad_sorted) * 22))
    st.plotly_chart(fig_squad, width="stretch")

    # Squad ACWR heatmap-style
    st.markdown('<div class="section-header">Squad ACWR Status</div>', unsafe_allow_html=True)

    acwr_table = latest_all[["Athlete_ID", "Sex", "Sport_Type", "Age", "ACWR", "ACWR_Zone",
                              "HRV_rMSSD_ms", "Perceived_Fatigue", "Sleep_Quality", "AI_Risk_Pct", "Risk_Level"]].copy()
    acwr_table = acwr_table.rename(columns={
        "ACWR": "ACWR",
        "HRV_rMSSD_ms": "HRV (ms)",
        "Perceived_Fatigue": "Fatigue",
        "Sleep_Quality": "Sleep",
        "AI_Risk_Pct": "Risk %",
        "Risk_Level": "Risk Level",
        "Sport_Type": "Sport",
    })
    acwr_table = acwr_table.round(2)

    def highlight_risk(row):
        rl = row.get("Risk Level", "Low")
        if rl == "High":
            return ["background-color: #1a0a0a; color:#fca5a5"] + [""] * (len(row) - 1)
        elif rl == "Moderate":
            return ["background-color: #14100a; color:#fcd34d"] + [""] * (len(row) - 1)
        return [""] * len(row)

    st.dataframe(acwr_table.style.apply(highlight_risk, axis=1), width="stretch", height=350)

    # Sport type distribution
    col_p, col_q = st.columns(2)
    with col_p:
        sport_risk = squad_df.groupby("Sport_Type")["Injury_Risk"].mean().reset_index()
        sport_risk["Injury_Rate"] = (sport_risk["Injury_Risk"] * 100).round(1)
        fig_sport = go.Figure(go.Bar(
            x=sport_risk["Sport_Type"],
            y=sport_risk["Injury_Rate"],
            marker_color=BLUE,
            marker_line_width=0,
            text=[f"{v:.1f}%" for v in sport_risk["Injury_Rate"]],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=10),
        ))
        apply_theme(fig_sport, title="Injury Rate by Sport Type (%)", height=260)
        st.plotly_chart(fig_sport, width="stretch")

    with col_q:
        # HRV by sport
        hrv_sport = squad_df.groupby("Sport_Type")["HRV_rMSSD_ms"].mean().reset_index()
        fig_hrv_sport = go.Figure(go.Bar(
            x=hrv_sport["Sport_Type"],
            y=hrv_sport["HRV_rMSSD_ms"],
            marker_color=GREEN,
            marker_line_width=0,
            text=[f"{v:.1f}" for v in hrv_sport["HRV_rMSSD_ms"]],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=10),
        ))
        apply_theme(fig_hrv_sport, title="Mean HRV rMSSD by Sport Type (ms)", height=260)
        st.plotly_chart(fig_hrv_sport, width="stretch")
