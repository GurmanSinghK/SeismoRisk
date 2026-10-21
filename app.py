from pathlib import Path
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import nbinom, poisson, ttest_ind

from src.usgs_live_client import fetch_live_usgs_feed, query_usgs_fdsnws
from src.aftershock_model import (
    compute_aftershock_expectation,
    calculate_aftershock_probabilities,
    filter_qualifying_events,
    integrate_omori_rate,
    haversine_distance_km,
    OMORI_P_GLOBAL,
    OMORI_C_DAYS,
    SCALING_B_GLOBAL,
    DISPERSION_R_GLOBAL
)

# ---------------------------------------------------------
# Page & Layout Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="SeismoRisk | Probabilistic Earthquake Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REGIONAL_DATA = PROJECT_ROOT / "data" / "processed" / "fiji_earthquake_sequence.csv"
GLOBAL_SUMMARY_FILE = PROJECT_ROOT / "data" / "processed" / "global_sequence_summary.csv"
GLOBAL_EVENTS_FILE = PROJECT_ROOT / "data" / "processed" / "global_sequence_events.csv"
GLOBAL_EVAL_FILE = PROJECT_ROOT / "data" / "processed" / "global_model_evaluation.csv"
GLOBAL_FREQ_FILE = PROJECT_ROOT / "data" / "processed" / "global_count_frequencies.csv"
GLOBAL_RAW_FILE = PROJECT_ROOT / "data" / "raw" / "global_earthquakes_2000_2025.csv"
CLEAN_MULTIYEAR_CATALOG = PROJECT_ROOT / "data" / "processed" / "earthquakes_clean.csv"


# ---------------------------------------------------------
# Custom Modern Scientific Theme (Navy / Slate / Teal / White)
# ---------------------------------------------------------
st.markdown("""
<style>
/* App background & typography */
.stApp {
    background: #090f1d;
    color: #e2e8f0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0d1527;
    border-right: 1px solid #1e293b;
}

/* Metric card container */
[data-testid="stMetric"] {
    background: #111c33;
    border: 1px solid #1e2e4f;
    padding: 16px 20px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
}
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    color: #f8fafc !important;
    font-size: 1.85rem !important;
    font-weight: 700 !important;
}

/* Tab navigation styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 4px;
}
.stTabs [data-baseweb="tab"] {
    height: 44px;
    border-radius: 8px 8px 0px 0px;
    padding: 8px 16px;
    color: #94a3b8;
    background-color: transparent;
    font-weight: 600;
    font-size: 0.92rem;
    border: none;
}
.stTabs [aria-selected="true"] {
    color: #38bdf8 !important;
    background-color: #13213d !important;
    border-bottom: 2px solid #38bdf8 !important;
}

/* Hero banner */
.hero-container {
    background: linear-gradient(135deg, #0e1d3a 0%, #0a1428 100%);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 24px 30px;
    margin-bottom: 24px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
}
.hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #f8fafc;
    margin: 0 0 6px 0;
}
.hero-badge {
    display: inline-block;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #38bdf8;
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.28);
    padding: 3px 10px;
    border-radius: 9999px;
    margin-bottom: 10px;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    line-height: 1.5;
    margin-bottom: 0;
}

/* Info & Callout cards */
.notice-box {
    border-left: 4px solid #38bdf8;
    background: rgba(56, 189, 248, 0.08);
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin-bottom: 20px;
    font-size: 0.92rem;
    color: #cbd5e1;
}
.warning-box {
    border-left: 4px solid #f59e0b;
    background: rgba(245, 158, 11, 0.08);
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin-bottom: 20px;
    font-size: 0.92rem;
    color: #fde68a;
}
.stat-card {
    background: #0f1c34;
    border: 1px solid #1e3357;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 16px;
}
.card-header {
    font-size: 1.05rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 8px;
}
.card-caption {
    font-size: 0.85rem;
    color: #94a3b8;
    line-height: 1.4;
}

/* Streamlit data table styling */
[data-testid="stDataFrame"] {
    border-radius: 8px;
    border: 1px solid #1e293b;
}

/* Headings */
h1, h2, h3, h4 {
    color: #f8fafc;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Plotly Unified Scientific Theme Helper
# ---------------------------------------------------------
def apply_scientific_theme(fig, title="", height=400):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0c1527",
        height=height,
        margin=dict(l=40, r=20, t=50 if title else 25, b=40),
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(size=14, color="#f1f5f9")
        ),
        xaxis=dict(
            gridcolor="#1e293b",
            zerolinecolor="#334155",
            tickfont=dict(color="#94a3b8", size=11),
            title_font=dict(color="#cbd5e1", size=12)
        ),
        yaxis=dict(
            gridcolor="#1e293b",
            zerolinecolor="#334155",
            tickfont=dict(color="#94a3b8", size=11),
            title_font=dict(color="#cbd5e1", size=12)
        ),
        legend=dict(
            bgcolor="rgba(13, 21, 39, 0.8)",
            bordercolor="#1e293b",
            font=dict(color="#cbd5e1", size=11)
        )
    )
    return fig

# ---------------------------------------------------------
# Cached Data Loaders
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv_data(file_bytes: bytes) -> pd.DataFrame:
    raw = pd.read_csv(io.BytesIO(file_bytes))
    required = {"event_id", "time", "magnitude", "longitude", "latitude", "depth_km"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    raw["time"] = pd.to_datetime(raw["time"], format="mixed", utc=True, errors="coerce")
    for col in ["magnitude", "depth_km", "longitude", "latitude"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw["event_id"] = raw["event_id"].astype(str)
    return raw.dropna(subset=["time", "magnitude", "depth_km"]).copy()

@st.cache_data(show_spinner=False)
def load_global_summaries():
    if GLOBAL_SUMMARY_FILE.exists():
        sm = pd.read_csv(GLOBAL_SUMMARY_FILE)
        sm["mainshock_time"] = pd.to_datetime(sm["mainshock_time"], utc=True, errors="coerce")
        # Optional: join place names if available
        if GLOBAL_RAW_FILE.exists():
            try:
                places = pd.read_csv(GLOBAL_RAW_FILE, usecols=["event_id", "place"])
                places["event_id"] = places["event_id"].astype(str)
                sm["mainshock_id"] = sm["mainshock_id"].astype(str)
                sm = sm.merge(places, left_on="mainshock_id", right_on="event_id", how="left")
                sm["place"] = sm["place"].fillna("Unknown Location")
            except Exception:
                sm["place"] = "Global Mainshock"
        else:
            sm["place"] = "Global Mainshock"
        return sm
    return None

@st.cache_data(show_spinner=False)
def load_global_events():
    if GLOBAL_EVENTS_FILE.exists():
        ev = pd.read_csv(GLOBAL_EVENTS_FILE)
        ev["event_time"] = pd.to_datetime(ev["event_time"], utc=True, errors="coerce")
        ev["mainshock_time"] = pd.to_datetime(ev["mainshock_time"], utc=True, errors="coerce")
        ev["days_after_mainshock"] = pd.to_numeric(ev["days_after_mainshock"], errors="coerce")
        ev["distance_km"] = pd.to_numeric(ev["distance_km"], errors="coerce")
        return ev
    return None

# Load Global Data in background
global_summary_df = load_global_summaries()
global_events_df = load_global_events()

# ---------------------------------------------------------
# Sidebar Controls & Scope Selection
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌐 Data & Environment")
    data_mode = st.radio(
        "Active Analysis Scope",
        options=[
            "Global Catalogue (9,805 Sequences)", 
            "Multi-Year Global Catalogue (2015–2025)",
            "Fiji Deep Sequence Benchmark (30 Days)", 
            "Custom CSV Upload"
        ],
        help="Switch between sequence-level clustering, multi-year catalogue exploration (any date from 2015 to 2025), or the Fiji 30-day benchmark sequence."
    )
    
    st.markdown("---")
    
    if data_mode == "Custom CSV Upload":
        uploaded = st.file_uploader("Upload Earthquake CSV", type=["csv"])
        if uploaded is not None:
            try:
                regional_df = load_csv_data(uploaded.getvalue())
                source_label = f"Uploaded: {uploaded.name}"
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
                st.stop()
        else:
            st.info("Upload a CSV file containing `event_id`, `time`, `magnitude`, `depth_km`, `latitude`, and `longitude`.")
            st.stop()
    elif data_mode.startswith("Multi-Year"):
        if CLEAN_MULTIYEAR_CATALOG.exists():
            regional_df = load_csv_data(CLEAN_MULTIYEAR_CATALOG.read_bytes())
            source_label = "Multi-Year Global Catalogue (2015–2025, 313,219 Events)"
        else:
            regional_df = load_csv_data(DEFAULT_REGIONAL_DATA.read_bytes())
            source_label = "Fiji Reference Sequence (us1000gcii, M 8.2)"
    else:
        if DEFAULT_REGIONAL_DATA.exists():
            regional_df = load_csv_data(DEFAULT_REGIONAL_DATA.read_bytes())
            source_label = "Fiji Reference Sequence (us1000gcii, M 8.2)"
        else:
            st.error("Default benchmark sequence not found at `data/processed/fiji_earthquake_sequence.csv`.")
            st.stop()

    # Filters applicable to regional explorer
    st.markdown("### 🎛️ Catalogue Date & Magnitude Filters")
    min_date = regional_df["time"].min().date()
    max_date = (regional_df["time"].max() + pd.Timedelta(days=1)).date()
    
    # Default selection depends on dataset span
    if data_mode.startswith("Multi-Year"):
        default_val = (pd.Timestamp("2020-01-01").date(), pd.Timestamp("2021-01-01").date())
    else:
        default_val = (min_date, max_date)
        
    date_range = st.date_input(
        "Date Range (UTC)",
        value=default_val,
        min_value=min_date,
        max_value=max_date,
        help="UTC date bounds for filtering earthquakes. You can select any range across the dataset."
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_ts = pd.Timestamp(date_range[0], tz="UTC")
        end_ts = pd.Timestamp(date_range[1], tz="UTC")
    elif isinstance(date_range, tuple) and len(date_range) == 1:
        start_ts = pd.Timestamp(date_range[0], tz="UTC")
        end_ts = start_ts + pd.Timedelta(days=30)
    else:
        start_ts = pd.Timestamp(min_date, tz="UTC")
        end_ts = pd.Timestamp(max_date, tz="UTC")


    min_mag = float(np.floor(regional_df["magnitude"].min() * 2) / 2)
    max_mag = float(np.ceil(regional_df["magnitude"].max() * 2) / 2)
    mag_slider = st.slider("Magnitude Range", min_value=min_mag, max_value=max_mag, value=(min_mag, max_mag), step=0.1)

    min_depth = float(max(0, np.floor(regional_df["depth_km"].min())))
    max_depth = float(np.ceil(regional_df["depth_km"].max()))
    depth_slider = st.slider("Depth Range (km)", min_value=min_depth, max_value=max_depth, value=(min_depth, max_depth), step=25.0)

    exclude_mainshock = st.checkbox("Exclude Reference Mainshock (M 8.2)", value=True, help="Prevents the mainshock itself from biasing subsequent daily event counts.")
    
    st.markdown("---")
    st.caption("SeismoRisk v2.0 | University Probability & Statistics Research")

# Filter regional events based on user input
filtered_regional = regional_df[
    (regional_df["time"] >= start_ts) & 
    (regional_df["time"] < end_ts) & 
    (regional_df["magnitude"].between(*mag_slider)) & 
    (regional_df["depth_km"].between(*depth_slider))
].copy()

if exclude_mainshock:
    filtered_regional = filtered_regional[~filtered_regional["event_id"].isin({"us1000gcii", "us2000h9e2"})]

# Daily binning for filtered regional data
day_idx = pd.date_range(start=start_ts.normalize(), end=(end_ts - pd.Timedelta(seconds=1)).normalize(), freq="D", tz="UTC")
filtered_regional["date_bin"] = filtered_regional["time"].dt.floor("D")
daily_series = filtered_regional.groupby("date_bin").size().reindex(day_idx, fill_value=0)
reg_daily_df = pd.DataFrame({"date": daily_series.index, "count": daily_series.values})
reg_counts = reg_daily_df["count"].to_numpy(dtype=int)

n_reg_events = len(filtered_regional)
n_reg_days = len(reg_daily_df)
reg_mean = float(reg_counts.mean()) if n_reg_days else 0.0
reg_var = float(reg_counts.var(ddof=1)) if n_reg_days > 1 else 0.0
reg_vmr = reg_var / reg_mean if reg_mean > 0 else np.nan

# Method of moments for regional
if reg_var > reg_mean and reg_mean > 0:
    reg_nb_r = reg_mean**2 / (reg_var - reg_mean)
    reg_nb_p = reg_nb_r / (reg_nb_r + reg_mean)
else:
    reg_nb_r, reg_nb_p = None, None

# ---------------------------------------------------------
# Hero Header & Academic Scope Notice
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">Academic Research & Statistical Modeling</div>
    <div class="hero-title">SeismoRisk: Probabilistic Earthquake Analytics</div>
    <div class="hero-subtitle">
        Investigating seismic clustering, overdispersion, and stochastic event forecasting using global earthquake catalogues (USGS 2000–2025).
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="notice-box">
    <b>Research & Academic Disclaimer:</b> SeismoRisk evaluates probabilistic count models (Poisson vs. Negative Binomial) over defined space-time windows. 
    It is <b>not an operational earthquake early-warning system</b> and does not predict the exact time, magnitude, or location of future seismic ruptures.
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Top High-Level Scientific KPIs
# ---------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
if data_mode.startswith("Global"):
    total_seqs = len(global_summary_df) if global_summary_df is not None else 0
    total_evs = len(global_events_df) if global_events_df is not None else 0
    global_mean_c = global_summary_df["assigned_event_count"].mean() if global_summary_df is not None else 0
    global_vmr = (global_summary_df["assigned_event_count"].var(ddof=1) / global_mean_c) if (global_summary_df is not None and global_mean_c > 0) else 0
    
    k1.metric("Candidate Mainshocks", f"{total_seqs:,}", help="Identified mainshocks (M ≥ 5.5) isolated by 100 km and 30 days.")
    k2.metric("Assigned Subsequent Events", f"{total_evs:,}", help="Subsequent events (M ≥ 4.5) within 100 km and 30 days.")
    k3.metric("Mean Events / Sequence", f"{global_mean_c:.2f}", help="Average number of triggered events per mainshock.")
    k4.metric("Variance-to-Mean Ratio (VMR)", f"{global_vmr:.1f}", help="VMR >> 1 indicates severe statistical overdispersion.")
else:
    k1.metric("Selected Regional Events", f"{n_reg_events:,}", help="Earthquakes satisfying the active regional filters.")
    k2.metric("Mean Magnitude", f"{filtered_regional['magnitude'].mean():.2f}" if n_reg_events else "—")
    k3.metric("Daily Mean Rate (λ)", f"{reg_mean:.2f} / day" if n_reg_days else "—")
    k4.metric("Regional VMR", f"{reg_vmr:.2f}" if np.isfinite(reg_vmr) else "—", help="Variance-to-mean ratio of daily counts.")

# ---------------------------------------------------------
# Primary Dashboard Tabs (Streamlined Information Architecture)
# ---------------------------------------------------------
tab_overview, tab_explorer, tab_sequences, tab_models, tab_forecast, tab_validation, tab_live = st.tabs([
    "1. Overview & Dataset",
    "2. Earthquake Explorer",
    "3. Sequence Delineation",
    "4. Statistical Models",
    "5. Probabilistic Forecasting",
    "6. Out-of-Sample Validation",
    "7. Live Earthquake & Aftershock Analysis"
])


# =========================================================
# TAB 1: OVERVIEW & DATASET
# =========================================================
with tab_overview:
    st.subheader("Global Seismic Hazard Modeling & Project Scope")
    st.markdown("""
    When a large earthquake occurs, the surrounding crust experiences static and dynamic stress perturbations that trigger 
    clusters of subsequent earthquakes (commonly called aftershocks). A central question in mathematical statistics is whether 
    these events occur as a memoryless Poisson process or exhibit clustered compound dynamics.
    """)
    
    col_ov1, col_ov2 = st.columns([1.4, 1.0])
    with col_ov1:
        st.markdown("#### Primary Objectives & Statistical Concepts")
        st.markdown("""
        - **Poisson Equidispersion Testing:** Evaluating whether empirical counts satisfy $\\mathrm{Var}(N) = E[N]$.
        - **Negative Binomial Mixture Fitting:** Modeling overdispersion via continuous Poisson-Gamma compounding.
        - **Omori-Utsu Temporal Decay:** Characterizing power-law rate decay over elapsed time $t$.
        - **Leakage-Free Validation:** Testing prospective forecast accuracy on strictly quarantined chronological holdout sequences.
        """)
        
        st.markdown("#### Curated Dataset Specifications")
        spec_df = pd.DataFrame([
            {"Parameter": "Catalogue Source", "Value": "USGS FDSNWS Earthquake API (Global)"},
            {"Parameter": "Observation Window", "Value": "2000-01-01 to 2025-12-31 (26 Years)"},
            {"Parameter": "Global Raw Earthquakes", "Value": "181,680 records (Cleaned: 178,522 events)"},
            {"Parameter": "Minimum Global Magnitude", "Value": "M ≥ 4.5 (Catalogue completeness threshold)"},
            {"Parameter": "Mainshock Magnitude Criterion", "Value": "M ≥ 5.5 (Isolated by 100 km, 30 days)"},
            {"Parameter": "Regional Benchmark Sequence", "Value": "Fiji Subduction Zone (2018-08-19, M 8.2 deep event)"}
        ])
        st.dataframe(spec_df, hide_index=True, width="stretch")
        
    with col_ov2:
        st.markdown("#### Key Scientific Findings at a Glance")
        st.markdown(f"""
        <div class="stat-card">
            <div class="card-header">Severe Empirical Overdispersion</div>
            <div class="card-caption">
                Global sequence counts exhibit a Variance-to-Mean Ratio of <b>67.09</b>, strongly violating the 
                Poisson assumption (VMR = 1.0) and necessitating Negative Binomial modeling.
            </div>
        </div>
        <div class="stat-card">
            <div class="card-header">Omori Temporal Decay Rate</div>
            <div class="card-caption">
                Triggered event rates peak on Day 1 (19,260 events) and decay smoothly to ~530 events/day by Day 30, 
                reproducing the physical Omori power-law.
            </div>
        </div>
        <div class="stat-card">
            <div class="card-header">Out-of-Sample Brier Calibration</div>
            <div class="card-caption">
                On the held-out test set (2017–2025), Negative Binomial achieves a Brier score of <b>0.2303</b> vs. 
                Poisson's <b>0.3331</b> (lower is better; Poisson severely over-predicts occurrence).
            </div>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# TAB 2: EARTHQUAKE EXPLORER
# =========================================================
with tab_explorer:
    st.subheader("Interactive Seismic Activity Explorer")
    st.caption("Visualizing filtered temporal, magnitude, and spatial distributions.")
    
    col_ex1, col_ex2 = st.columns([1.5, 1.0])
    with col_ex1:
        st.markdown("#### Daily Event Count Series")
        fig_daily = px.bar(
            reg_daily_df, x="date", y="count",
            labels={"date": "UTC Date", "count": "Recorded Events"},
            color="count",
            color_continuous_scale=["#0284c7", "#38bdf8", "#7dd3fc"]
        )
        apply_scientific_theme(fig_daily, title="Daily Event Activity (UTC Calendar Days)", height=350)
        fig_daily.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_daily, width="stretch")
        st.caption("Daily event counts demonstrate high initial bursting followed by rapid temporal attenuation.")
        
    with col_ex2:
        st.markdown("#### Magnitude Histogram")
        if n_reg_events:
            fig_mag = px.histogram(
                filtered_regional, x="magnitude", nbins=18,
                labels={"magnitude": "Magnitude", "count": "Events"},
                color_discrete_sequence=["#38bdf8"]
            )
            apply_scientific_theme(fig_mag, title="Magnitude Frequency Distribution", height=350)
            st.plotly_chart(fig_mag, width="stretch")
            st.caption(f"Magnitude span: [{filtered_regional['magnitude'].min():.1f}, {filtered_regional['magnitude'].max():.1f}]. Follows Gutenberg-Richter scaling.")
        else:
            st.info("No events match the active filters.")
            
    col_ex3, col_ex4 = st.columns(2)
    with col_ex3:
        st.markdown("#### Magnitude vs. Hypocentral Depth")
        if n_reg_events >= 2:
            fig_depth = px.scatter(
                filtered_regional, x="depth_km", y="magnitude",
                labels={"depth_km": "Depth (km)", "magnitude": "Magnitude"},
                hover_data=["event_id", "time"],
                color_discrete_sequence=["#22d3ee"],
                opacity=0.75
            )
            corr_val = filtered_regional["magnitude"].corr(filtered_regional["depth_km"])
            apply_scientific_theme(fig_depth, title=f"Magnitude vs. Depth (Pearson r = {corr_val:.4f})", height=340)
            st.plotly_chart(fig_depth, width="stretch")
        else:
            st.info("Insufficient observations for correlation.")
            
    with col_ex4:
        st.markdown("#### Descriptive Depth Stratification")
        if n_reg_events:
            depth_bands = pd.cut(
                filtered_regional["depth_km"],
                bins=[-1, 70, 300, 1000],
                labels=["Shallow (≤70 km)", "Intermediate (70–300 km)", "Deep (>300 km)"]
            )
            depth_summary = filtered_regional.groupby(depth_bands, observed=False)["magnitude"].agg(
                ["count", "mean", "median", "std", "min", "max"]
            ).reset_index().rename(columns={"depth_km": "Depth Group"}).round(3)
            st.dataframe(depth_summary, hide_index=True, width="stretch")
            st.markdown(
                '<div class="notice-box">Deep subduction earthquakes (depth > 300 km, e.g., Fiji) show significantly lower aftershock productivity than shallow crustal faults due to high confining temperatures and ductile deformation.</div>',
                unsafe_allow_html=True
            )

# =========================================================
# TAB 3: SEQUENCE DELINEATION
# =========================================================
with tab_sequences:
    st.subheader("Global Sequence Clustering & Inspection")
    st.markdown("""
    Earthquake sequences are identified using a **3D spatial KD-tree chord algorithm** ($R = 100\\text{ km}$, $\\Delta t = 30\\text{ days}$). 
    Events are assigned to the most recent eligible candidate mainshock ($M \\ge 5.5$) that exceeds the event's magnitude.
    """)
    
    if global_summary_df is not None:
        col_seq1, col_seq2 = st.columns([1.0, 1.5])
        with col_seq1:
            st.markdown("#### Select Mainshock to Inspect")
            top_mainshocks = global_summary_df.sort_values("assigned_event_count", ascending=False).head(50)
            
            selected_main_id = st.selectbox(
                "Top 50 Most Productive Global Mainshocks",
                options=top_mainshocks["mainshock_id"].tolist(),
                format_func=lambda x: f"{x} | M {top_mainshocks.loc[top_mainshocks['mainshock_id']==x, 'mainshock_magnitude'].values[0]} | {top_mainshocks.loc[top_mainshocks['mainshock_id']==x, 'assigned_event_count'].values[0]} events | {top_mainshocks.loc[top_mainshocks['mainshock_id']==x, 'place'].values[0]}"
            )
            
            selected_row = global_summary_df[global_summary_df["mainshock_id"] == selected_main_id].iloc[0]
            st.markdown(f"""
            <div class="stat-card">
                <div class="card-header">Mainshock: {selected_main_id}</div>
                <div class="card-caption">
                    <b>Date:</b> {selected_row['mainshock_time']}<br>
                    <b>Magnitude:</b> M {selected_row['mainshock_magnitude']:.1f}<br>
                    <b>Location:</b> {selected_row['place']}<br>
                    <b>Assigned Subsequent Events:</b> {selected_row['assigned_event_count']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_seq2:
            st.markdown("#### Sequence Event Timing (Elapsed Days)")
            if global_events_df is not None:
                seq_events = global_events_df[global_events_df["mainshock_id"] == selected_main_id].sort_values("days_after_mainshock")
                if not seq_events.empty:
                    fig_seq = px.scatter(
                        seq_events, x="days_after_mainshock", y="event_magnitude",
                        size="event_magnitude", color="distance_km",
                        labels={"days_after_mainshock": "Elapsed Time (Days)", "event_magnitude": "Magnitude", "distance_km": "Distance (km)"},
                        color_continuous_scale="Viridis"
                    )
                    apply_scientific_theme(fig_seq, title=f"Assigned Events for {selected_main_id}", height=320)
                    st.plotly_chart(fig_seq, width="stretch")
                    st.dataframe(seq_events[["event_id", "event_time", "event_magnitude", "days_after_mainshock", "distance_km"]].head(8), hide_index=True, width="stretch")
                else:
                    st.info("This sequence has zero assigned subsequent events within 100 km and 30 days.")
                    
        st.markdown("---")
        st.markdown("#### Global Sequence Size Distribution (Histogram of Cluster Sizes)")
        fig_seq_sizes = px.histogram(
            global_summary_df, x="assigned_event_count", nbins=50,
            labels={"assigned_event_count": "Assigned Events per Sequence", "count": "Number of Mainshocks"},
            log_y=True, color_discrete_sequence=["#38bdf8"]
        )
        apply_scientific_theme(fig_seq_sizes, title="Distribution of Cluster Sizes Across 9,805 Global Sequences (Log-Scale)", height=320)
        st.plotly_chart(fig_seq_sizes, width="stretch")
        st.caption("35.5% of mainshocks have zero assigned subsequent events; the tail extends to over 400 events in major plate boundary ruptures.")

# =========================================================
# TAB 4: STATISTICAL MODELS
# =========================================================
with tab_models:
    st.subheader("Poisson vs. Negative Binomial Count Modeling")
    st.markdown("""
    Under the null hypothesis of memoryless occurrences, sequence counts follow a **Poisson($\\lambda$)** distribution. 
    When counts cluster, productivity varies between mainshocks, yielding a **Negative Binomial($r, p$)** compound model.
    """)
    
    col_m1, col_m2 = st.columns([1.2, 1.0])
    with col_m1:
        st.markdown("#### Theoretical Mass Function Comparison")
        if data_mode.startswith("Global") and global_summary_df is not None:
            fit_mean = global_summary_df["assigned_event_count"].mean()
            fit_r, fit_p = 0.3108, 0.0607 # MLE from fit_global_models.py
            ks = np.arange(0, 16)
            pois_pmf = poisson.pmf(ks, fit_mean)
            nb_pmf = nbinom.pmf(ks, fit_r, fit_p)
            label_context = "Global Sequence Model (MLE)"
        else:
            fit_mean = reg_mean
            fit_r, fit_p = reg_nb_r, reg_nb_p
            ks = np.arange(0, max(15, int(reg_counts.max()) if len(reg_counts) else 15) + 1)
            pois_pmf = poisson.pmf(ks, fit_mean)
            nb_pmf = nbinom.pmf(ks, fit_r, fit_p) if fit_r is not None else None
            label_context = "Regional Daily Count Model (MoM)"
            
        fig_pmf = go.Figure()
        fig_pmf.add_trace(go.Bar(x=ks - 0.18, y=pois_pmf, name=f"Poisson (λ = {fit_mean:.2f})", marker_color="#38bdf8", width=0.35))
        if nb_pmf is not None:
            fig_pmf.add_trace(go.Bar(x=ks + 0.18, y=nb_pmf, name=f"Negative Binomial (r = {fit_r:.2f}, p = {fit_p:.3f})", marker_color="#f59e0b", width=0.35))
        apply_scientific_theme(fig_pmf, title=f"Probability Mass Function P(N = k) — {label_context}", height=380)
        fig_pmf.update_layout(xaxis_title="Event Count (k)", yaxis_title="Probability P(N = k)")
        st.plotly_chart(fig_pmf, width="stretch")
        
    with col_m2:
        st.markdown("#### Mathematical Formulation & Parameters")
        st.markdown(r"""
        **Poisson Model:**
        $$P(N = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad E[N] = \lambda, \quad \mathrm{Var}(N) = \lambda$$
        
        **Negative Binomial (Poisson-Gamma Mixture):**
        $$P(N = k) = \frac{\Gamma(k+r)}{k!\Gamma(r)} (1-p)^r p^k$$
        $$\mathrm{Var}(N) = E[N] + \frac{1}{r} E[N]^2 > E[N]$$
        """)
        
        if nb_pmf is not None:
            st.markdown(f"""
            <div class="good-box">
                <b>Parameter Estimates:</b><br>
                • Mean Rate $\\lambda$: <b>{fit_mean:.3f}</b><br>
                • Shape Parameter $r$: <b>{fit_r:.4f}</b><br>
                • Probability $p$: <b>{fit_p:.4f}</b><br>
                • Implied Zero-Probability $P(N = 0)$: <b>{nb_pmf[0]:.1%}</b> (vs. Poisson: {pois_pmf[0]:.1%})
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("---")
    st.markdown("#### Observed vs. Model-Predicted Frequencies")
    if GLOBAL_FREQ_FILE.exists():
        freq_table_df = pd.read_csv(GLOBAL_FREQ_FILE)
        st.dataframe(freq_table_df.head(12), hide_index=True, width="stretch")
        st.caption("The Poisson model severely underestimates both zero counts (underestimating zero-hazard sequences) and large cluster tails.")

# =========================================================
# TAB 5: PROBABILISTIC FORECASTING
# =========================================================
with tab_forecast:
    st.subheader("Responsible Probabilistic Scenario Forecasting")
    st.markdown("""
    Estimate the probability distribution of future triggered activity given an initial mainshock scenario.
    """)
    
    st.markdown("""
    <div class="warning-box">
        <b>Scientific Safety Notice:</b> These values are statistical expectation integrals derived from historical catalogue distributions. 
        They express the probability of experiencing at least one subsequent event within the defined space-time horizon, 
        <b>not deterministic predictions</b> of exact rupture coordinates or timings.
    </div>
    """, unsafe_allow_html=True)
    
    f_col1, f_col2 = st.columns([1.0, 1.4])
    with f_col1:
        st.markdown("#### Scenario Configuration")
        fc_horizon = st.radio(
            "Forecast Prediction Horizon",
            options=[1, 3, 7, 14, 30],
            index=2,
            format_func=lambda x: f"Next {x} Days",
            help="Prediction window beginning at the forecast origin."
        )
        
        fc_mag_threshold = st.selectbox(
            "Target Event Magnitude Threshold",
            options=["M ≥ 4.5 (Catalogue completeness)", "M ≥ 5.0 (Moderate shaking hazard)", "M ≥ 5.5 (Major triggered event)"],
            index=0
        )
        
        fc_mainshock_mag = st.slider("Triggering Mainshock Magnitude (M)", 5.5, 8.5, 7.0, 0.1)
        
    with f_col2:
        st.markdown("#### Calculated Probabilities & Expected Counts")
        # Global validated scaling: Omori integral scaling with horizon
        # Integrated rate over window [0, horizon] / [0, 30]
        base_rate = 4.810
        # Time fraction based on empirical Omori decay (approximate cumulative fraction)
        time_fraction = np.log(1 + fc_horizon) / np.log(1 + 30)
        # Magnitude scaling (Båth's law / Gutenberg-Richter)
        mag_scale = 10**(0.75 * (fc_mainshock_mag - 7.0))
        
        expected_n_pois = max(0.05, base_rate * time_fraction * mag_scale)
        p_at_least_one_pois = float(1.0 - np.exp(-expected_n_pois))
        
        # NegBin dispersion scaling
        nb_r_scaled = 0.3108 * time_fraction
        nb_p_val = 0.0607
        p_at_least_one_nb = float(1.0 - nbinom.pmf(0, nb_r_scaled, nb_p_val))
        expected_n_nb = expected_n_pois
        
        fc_res_df = pd.DataFrame([
            {"Model": "Baseline Poisson", "Expected Count": f"{expected_n_pois:.2f}", "P(≥ 1 Event)": f"{p_at_least_one_pois:.1%}", "Calibration Assessment": "Overconfident on rare clusters"},
            {"Model": "Negative Binomial (MLE)", "Expected Count": f"{expected_n_nb:.2f}", "P(≥ 1 Event)": f"{p_at_least_one_nb:.1%}", "Calibration Assessment": "Validated out-of-sample (Brier = 0.230)"}
        ])
        st.dataframe(fc_res_df, hide_index=True, width="stretch")
        
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Bar(
            x=["Baseline Poisson", "Negative Binomial (MLE)"],
            y=[p_at_least_one_pois, p_at_least_one_nb],
            marker_color=["#38bdf8", "#f59e0b"],
            text=[f"{p_at_least_one_pois:.1%}", f"{p_at_least_one_nb:.1%}"],
            textposition="outside"
        ))
        apply_scientific_theme(fig_fc, title=f"Forecast P(N ≥ 1) Over Next {fc_horizon} Days (M ≥ {fc_mainshock_mag:.1f} Scenario)", height=300)
        fig_fc.update_layout(yaxis=dict(range=[0, 1.15], tickformat=".0%"))
        st.plotly_chart(fig_fc, width="stretch")
        st.caption(f"Scenario assumes a {fc_horizon}-day window within 100 km chord distance of an M {fc_mainshock_mag:.1f} mainshock.")

# =========================================================
# TAB 6: OUT-OF-SAMPLE VALIDATION
# =========================================================
with tab_validation:
    st.subheader("Strict Leakage-Free Model Validation")
    st.markdown("""
    To prevent data leakage, evaluation is conducted on a **chronologically quarantined sequence test set**:
    - **Training Set (70%):** 6,863 sequences occurring between 2000-01-01 and 2017-12-16.
    - **Out-of-Sample Test Set (30%):** 2,942 sequences occurring between 2017-12-16 and 2025-12-30.
    """)
    
    if GLOBAL_EVAL_FILE.exists():
        eval_df = pd.read_csv(GLOBAL_EVAL_FILE)
        st.markdown("#### Prospective Out-of-Sample Performance Comparison")
        st.dataframe(eval_df, hide_index=True, width="stretch")
        
    val_c1, val_c2 = st.columns(2)
    with val_c1:
        st.markdown("#### Test Set Mean Log-Likelihood")
        fig_ll = go.Figure()
        fig_ll.add_trace(go.Bar(
            x=["Baseline Poisson", "Negative Binomial (MLE)"],
            y=[-9.3068, -2.4570],
            marker_color=["#f43f5e", "#10b981"],
            text=["-9.31", "-2.46"],
            textposition="outside"
        ))
        apply_scientific_theme(fig_ll, title="Out-of-Sample Mean Log-Likelihood (Higher is Better)", height=320)
        st.plotly_chart(fig_ll, width="stretch")
        st.caption("Negative Binomial achieves a substantially higher log-likelihood by assigning realistic probability mass to extreme counts.")
        
    with val_c2:
        st.markdown("#### Binary Occurrence Calibration (Brier Score)")
        fig_brier = go.Figure()
        fig_brier.add_trace(go.Bar(
            x=["Empirical Baseline", "Negative Binomial (MLE)", "Baseline Poisson"],
            y=[0.2245, 0.2303, 0.3331],
            marker_color=["#94a3b8", "#10b981", "#f43f5e"],
            text=["0.224", "0.230", "0.333"],
            textposition="outside"
        ))
        apply_scientific_theme(fig_brier, title="Brier Score on Test Sequences (Lower is Better)", height=320)
        st.plotly_chart(fig_brier, width="stretch")
        st.caption("Poisson produces a poor Brier score (0.333) because it aggressively predicts P ≈ 99.2% for all sequences, failing on the 33.8% of test sequences with zero subsequent events.")

# =========================================================
# TAB 7: LIVE EARTHQUAKE & AFTERSHOCK ANALYSIS
# =========================================================
with tab_live:
    st.subheader("Live USGS Earthquake & Aftershock Probability Analysis")
    st.markdown("""
    Explore recently reported seismic events directly from the official **USGS Earthquake Hazards Program**, 
    select a candidate mainshock, and generate time-dependent aftershock rate models and probabilities.
    """)
    
    st.markdown("""
    <div class="notice-box">
        <b>Live Data Feed:</b> Powered by the USGS GeoJSON and FDSNWS endpoints. 
        A recently reported event is designated as a <i>candidate mainshock</i>; subsequent events are filtered by explicit 
        spatial proximity, temporal succession, and magnitude thresholds.
    </div>
    """, unsafe_allow_html=True)
    
    # Mode selector: Real-Time USGS Feed vs Historical Replay Mode
    sub_mode = st.radio(
        "Analysis Mode",
        options=["Real-Time USGS Live Feed", "Historical Replay & Backtesting Mode"],
        horizontal=True,
        help="Switch between live real-time feeds from USGS and backtesting on historical candidate mainshocks without data leakage."
    )
    
    st.markdown("---")
    
    if sub_mode == "Real-Time USGS Live Feed":
        col_feed1, col_feed2 = st.columns([1.2, 2.0])
        with col_feed1:
            st.markdown("#### Live Feed Settings")
            feed_choice = st.selectbox(
                "Select USGS Real-Time Feed",
                options=[
                    ("past_week_m45", "Past 7 Days (M ≥ 4.5) — Recommended"),
                    ("past_day_m45", "Past 24 Hours (M ≥ 4.5)"),
                    ("past_month_m45", "Past 30 Days (M ≥ 4.5)"),
                    ("past_week_significant", "Past 7 Days (Significant Events)")
                ],
                format_func=lambda x: x[1]
            )[0]
            
            btn_refresh = st.button("🔄 Refresh Live Feed from USGS", use_container_width=True)
            
        # Fetch live data with caching or explicit refresh
        @st.cache_data(ttl=120, show_spinner=False)
        def get_cached_live_feed(key: str, force_refresh_token: int = 0):
            return fetch_live_usgs_feed(key, timeout_secs=8)
            
        # Token to force refresh when button clicked
        refresh_token = int(pd.Timestamp.now().timestamp()) if btn_refresh else 0
        with st.spinner("Connecting to official USGS Earthquake Feed..."):
            live_result = get_cached_live_feed(feed_choice, refresh_token)
            
        with col_feed2:
            st.markdown("#### Live Connection Status")
            if live_result["success"]:
                fetched_time_str = live_result["fetched_at"].strftime("%Y-%m-%d %H:%M:%S UTC")
                st.markdown(f"""
                <div class="good-box">
                    <b>Status: Connected to USGS</b><br>
                    • <b>Feed Title:</b> {live_result['title']}<br>
                    • <b>Events Retrieved:</b> {live_result['count']:,} qualifying earthquakes<br>
                    • <b>Last Polled:</b> {fetched_time_str}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="warning-box">
                    <b>Live Feed Unavailable:</b> {live_result['error']}<br>
                    Unable to contact USGS server. Please check your internet connection or use Historical Replay Mode below.
                </div>
                """, unsafe_allow_html=True)
                
        if live_result["success"] and not live_result["data"].empty:
            live_df = live_result["data"]
            
            st.markdown("#### Step 1: Select Candidate Mainshock from Live Feed")
            
            # Interactive earthquake table
            display_cols = ["event_id", "magnitude", "place", "time_utc", "depth_km", "latitude", "longitude"]
            st.dataframe(
                live_df[display_cols].rename(columns={
                    "event_id": "Event ID", "magnitude": "M", "place": "Location",
                    "time_utc": "Occurrence Time (UTC)", "depth_km": "Depth (km)"
                }),
                hide_index=True,
                height=220,
                width="stretch"
            )
            
            # Selection controls
            event_options = live_df["event_id"].tolist()
            selected_live_id = st.selectbox(
                "Choose Reference Mainshock for Aftershock Modeling",
                options=event_options,
                format_func=lambda x: f"M {live_df.loc[live_df['event_id']==x, 'magnitude'].values[0]:.1f} — {live_df.loc[live_df['event_id']==x, 'place'].values[0]} ({pd.Timestamp(live_df.loc[live_df['event_id']==x, 'time_utc'].values[0]).strftime('%Y-%m-%d %H:%M')} UTC)"
            )
            
            selected_event_row = live_df[live_df["event_id"] == selected_live_id].iloc[0]
            
            # Display Mainshock Summary Card
            st.markdown("#### Candidate Mainshock Parameters")
            m_c1, m_c2, m_c3, m_c4 = st.columns(4)
            m_c1.metric("Mainshock Magnitude", f"M {selected_event_row['magnitude']:.1f}")
            m_c2.metric("Hypocentral Depth", f"{selected_event_row['depth_km']:.1f} km")
            m_c3.metric("Latitude, Longitude", f"{selected_event_row['latitude']:.2f}°, {selected_event_row['longitude']:.2f}°")
            m_c4.metric("Occurrence Time", pd.Timestamp(selected_event_row["time_utc"]).strftime("%m-%d %H:%M UTC"))
            
            st.markdown("---")
            st.markdown("#### Step 2: Configure Aftershock Forecast Scenario")
            cfg_c1, cfg_c2, cfg_c3 = st.columns(3)
            with cfg_c1:
                fc_duration_days = st.select_slider(
                    "Forecast Duration (Window)",
                    options=[1.0, 2.0, 3.0, 5.0, 7.0, 14.0, 21.0, 30.0],
                    value=7.0,
                    format_func=lambda x: f"{int(x)} Days" if x >= 1 else f"{x} Days"
                )
            with cfg_c2:
                fc_radius_km = st.slider(
                    "Search Radius (Chord Distance)",
                    min_value=30.0,
                    max_value=250.0,
                    value=100.0,
                    step=10.0,
                    help="Radius centered on the mainshock epicenter to detect triggered activity."
                )
            with cfg_c3:
                fc_target_mag = st.slider(
                    "Subsequent Event Min Magnitude (M_target)",
                    min_value=3.0,
                    max_value=max(4.5, float(selected_event_row["magnitude"]) - 0.5),
                    value=4.5,
                    step=0.1,
                    help="Catalogue completeness limit is M >= 4.5 globally."
                )
                
            # Compute Time-dependent Omori rate & Poisson / Negative Binomial probabilities
            t_start = 0.0
            t_end = float(fc_duration_days)
            expected_count = compute_aftershock_expectation(
                mainshock_mag=float(selected_event_row["magnitude"]),
                target_min_mag=float(fc_target_mag),
                t_start_days=t_start,
                t_end_days=t_end,
                radius_km=float(fc_radius_km)
            )
            prob_dict = calculate_aftershock_probabilities(expected_count)
            
            st.markdown("#### Step 3: Probabilistic Aftershock Forecast Results")
            res_c1, res_c2, res_c3, res_c4 = st.columns(4)
            res_c1.metric("Expected Aftershocks (Λ)", f"{expected_count:.2f}")
            res_c2.metric("P(≥ 1 Event) — NegBin", f"{prob_dict['negative_binomial']['p_at_least_one']:.1%}")
            res_c3.metric("P(≥ 1 Event) — Poisson", f"{prob_dict['poisson']['p_at_least_one']:.1%}")
            res_c4.metric("95% Predictive Interval", f"[{prob_dict['negative_binomial']['interval_95'][0]}, {prob_dict['negative_binomial']['interval_95'][1]}] events")
            
            # Visualizations: Rate Decay Curve + Predictive Count Mass Function
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                # Plot Omori Rate Decay curve
                t_eval = np.linspace(0.01, fc_duration_days, 150)
                delta_m = float(selected_event_row["magnitude"]) - fc_target_mag
                K_mag = 10.0 ** (-1.250 + 0.692 * delta_m) * ((fc_radius_km / 100.0) ** 1.5)
                rate_t = K_mag / ((t_eval + 0.05) ** 0.9029)
                
                fig_omori = go.Figure()
                fig_omori.add_trace(go.Scatter(
                    x=t_eval, y=rate_t,
                    mode="lines", line=dict(color="#38bdf8", width=2.5),
                    name="Omori Rate λ(t)"
                ))
                apply_scientific_theme(fig_omori, title=f"Expected Activity Decay Rate Over Next {int(fc_duration_days)} Days", height=320)
                fig_omori.update_layout(xaxis_title="Days After Mainshock", yaxis_title="Expected Rate (events/day)")
                st.plotly_chart(fig_omori, width="stretch")
                st.caption("Modified Omori-Utsu model: λ(t) = K / (t + c)^p with empirical parameters K(M), c=0.05d, p=0.903.")
                
            with chart_col2:
                # Plot Predictive Mass Function P(N = k)
                k_show = np.arange(0, min(15, max(8, int(expected_count * 2))))
                pmf_nb_show = [float(nbinom.pmf(k, 0.3108, 0.3108 / (0.3108 + expected_count))) if expected_count > 0 else (1.0 if k==0 else 0.0) for k in k_show]
                pmf_pois_show = [float(poisson.pmf(k, expected_count)) for k in k_show]
                
                fig_k = go.Figure()
                fig_k.add_trace(go.Bar(x=k_show - 0.18, y=pmf_pois_show, name="Poisson", marker_color="#38bdf8", width=0.35))
                fig_k.add_trace(go.Bar(x=k_show + 0.18, y=pmf_nb_show, name="Negative Binomial (MLE)", marker_color="#f59e0b", width=0.35))
                apply_scientific_theme(fig_k, title="Predictive Probability Distribution P(N = k)", height=320)
                fig_k.update_layout(xaxis_title="Number of Events (k)", yaxis_title="Probability")
                st.plotly_chart(fig_k, width="stretch")
                st.caption("Negative Binomial assigns realistic probability mass to k = 0 as well as large count tails.")

            # Search qualifying subsequent events reported so far in the live feed
            st.markdown("#### Qualifying Events Observed So Far (Near Real-Time Check)")
            qualifying_evs, meta_qual = filter_qualifying_events(
                candidate_events_df=live_df,
                mainshock_id=selected_live_id,
                mainshock_lat=float(selected_event_row["latitude"]),
                mainshock_lon=float(selected_event_row["longitude"]),
                mainshock_time=pd.Timestamp(selected_event_row["time_utc"]),
                forecast_start_time=pd.Timestamp(selected_event_row["time_utc"]),
                forecast_end_time=pd.Timestamp(selected_event_row["time_utc"]) + pd.Timedelta(days=fc_duration_days),
                target_min_mag=float(fc_target_mag),
                radius_km=float(fc_radius_km)
            )
            if not qualifying_evs.empty:
                st.success(f"Detected {len(qualifying_evs)} qualifying subsequent event(s) in the active feed within {fc_radius_km} km!")
                st.dataframe(
                    qualifying_evs[["event_id", "magnitude", "place", "time_utc", "distance_km", "days_after_mainshock"]].rename(columns={
                        "distance_km": "Distance (km)", "days_after_mainshock": "Elapsed Days"
                    }),
                    hide_index=True,
                    width="stretch"
                )
            else:
                st.info(f"No qualifying subsequent events (M ≥ {fc_target_mag}) detected within {fc_radius_km} km in the current live feed window.")

    else:
        # =====================================================
        # HISTORICAL REPLAY & BACKTESTING MODE
        # =====================================================
        st.markdown("### Historical Replay & Prospective Backtesting")
        st.markdown("""
        Select any historical candidate mainshock from the global catalog, simulate a forecast generated 
        at a specific elapsed time, and evaluate the model's out-of-sample performance without future data leakage.
        """)
        
        if global_summary_df is not None and global_events_df is not None:
            # Let user choose from top historical mainshocks
            replay_candidates = global_summary_df[global_summary_df["mainshock_magnitude"] >= 6.5].sort_values("mainshock_time", ascending=False).head(40)
            
            rep_col1, rep_col2 = st.columns([1.2, 1.0])
            with rep_col1:
                selected_rep_id = st.selectbox(
                    "Select Historical Benchmark Mainshock",
                    options=replay_candidates["mainshock_id"].tolist(),
                    format_func=lambda x: f"{x} | M {replay_candidates.loc[replay_candidates['mainshock_id']==x, 'mainshock_magnitude'].values[0]:.1f} | {pd.Timestamp(replay_candidates.loc[replay_candidates['mainshock_id']==x, 'mainshock_time'].values[0]).strftime('%Y-%m-%d')} | {replay_candidates.loc[replay_candidates['mainshock_id']==x, 'place'].values[0]}"
                )
                rep_main_row = replay_candidates[replay_candidates["mainshock_id"] == selected_rep_id].iloc[0]

                
            with rep_col2:
                rep_duration = st.select_slider("Evaluation Window (Duration)", options=[1, 3, 7, 14, 30], value=7)
                rep_offset = st.number_input("Forecast Origin Delay (Days after mainshock)", min_value=0.0, max_value=20.0, value=0.0, step=0.5)
                
            # Filter historical events strictly within window [rep_offset, rep_offset + rep_duration]
            seq_evs = global_events_df[global_events_df["mainshock_id"] == selected_rep_id].copy()
            observed_subsequent = seq_evs[
                (seq_evs["days_after_mainshock"] > rep_offset) & 
                (seq_evs["days_after_mainshock"] <= (rep_offset + rep_duration))
            ]
            actual_count = len(observed_subsequent)
            
            # Predict using model
            expected_rep_lam = compute_aftershock_expectation(
                mainshock_mag=float(rep_main_row["mainshock_magnitude"]),
                target_min_mag=4.5,
                t_start_days=float(rep_offset),
                t_end_days=float(rep_offset + rep_duration),
                radius_km=100.0
            )
            rep_probs = calculate_aftershock_probabilities(expected_rep_lam)
            
            # Verification metrics
            st.markdown("#### Backtest Outcome: Forecast vs. Ground Truth")
            b_c1, b_c2, b_c3, b_c4 = st.columns(4)
            b_c1.metric("Observed Events (Ground Truth)", f"{actual_count} events")
            b_c2.metric("Model Expected Count (Λ)", f"{expected_rep_lam:.2f}")
            b_c3.metric("Predicted P(≥ 1 Event)", f"{rep_probs['negative_binomial']['p_at_least_one']:.1%}")
            
            # Brier outcome for this single sequence
            actual_binary = 1 if actual_count >= 1 else 0
            brier_single = (rep_probs['negative_binomial']['p_at_least_one'] - actual_binary)**2
            b_c4.metric("Single-Event Brier Loss", f"{brier_single:.4f}", help="(P - Observed_binary)^2; 0.0 is perfect prediction.")
            
            st.markdown(f"""
            <div class="good-box">
                <b>Historical Verification Summary:</b><br>
                • <b>Candidate Mainshock:</b> {selected_rep_id} (M {rep_main_row['mainshock_magnitude']:.1f}) at {rep_main_row['place']}<br>
                • <b>Forecast Horizon:</b> Day {rep_offset} to Day {rep_offset + rep_duration} (Duration: {rep_duration} days)<br>
                • <b>Ground Truth Events Observed:</b> {actual_count} earthquakes (M ≥ 4.5 within 100 km)<br>
                • <b>Zero Leakage Guarantee:</b> Model parameters were fitted exclusively on 2000–2017 training sequences.
            </div>
            """, unsafe_allow_html=True)
            
            if not observed_subsequent.empty:
                st.markdown("#### Observed Events During Evaluation Window")
                st.dataframe(
                    observed_subsequent[["event_id", "event_time", "event_magnitude", "days_after_mainshock", "distance_km"]].rename(columns={
                        "event_magnitude": "M", "days_after_mainshock": "Elapsed (d)", "distance_km": "Dist (km)"
                    }),
                    hide_index=True,
                    width="stretch"
                )
        else:
            st.info("Global historical sequence catalog is required for backtesting.")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
f_left, f_right = st.columns([2, 1])
with f_left:
    st.caption("SeismoRisk v2.0 • USGS Global Catalogue Ingestion (2000–2025)")
with f_right:
    st.caption("Status: Validated Out-of-Sample Models Available")


