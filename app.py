from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import nbinom

# ──────────────────────────────────────────────────────────────────────
# SEISMORISK // INTERACTIVE SEISMIC PLAYGROUND
# ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
GLOBAL_SUMMARY = ROOT / "data" / "processed" / "global_sequence_summary.csv"
GLOBAL_EVENTS = ROOT / "data" / "processed" / "global_sequence_events.csv"
FIJI_EVENTS = ROOT / "data" / "processed" / "fiji_earthquake_sequence.csv"

st.set_page_config(
    page_title="SEISMORISK // Earthquake playground",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS: neon science-fair / arcade terminal ───────────────────────────
st.markdown(
    r"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
    :root { --bg:#080914; --panel:#111326; --line:#292c4b; --cyan:#63f5ff; --pink:#ff5bd6; --lime:#d7ff62; --muted:#9ba2c7; }
    html, body, [class*="css"] { font-family:'Space Grotesk',sans-serif; }
    .stApp {
      background:
       radial-gradient(ellipse at 12% 0%,rgba(94,56,190,.22),transparent 32%),
       radial-gradient(ellipse at 90% 18%,rgba(255,55,178,.10),transparent 30%),
       linear-gradient(135deg,#080914 0%,#0b1020 55%,#090b17 100%);
      color:#f4f5ff;
    }
    [data-testid="stHeader"] {background:rgba(8,9,20,.6);}
    .block-container {max-width:1500px;padding-top:1.25rem;padding-bottom:3rem;}
    h1,h2,h3 {color:#f6f5ff !important;letter-spacing:-.04em;}
    p,li,label {color:#c9cce6;}
    [data-testid="stSidebar"] {background:#0d0f20;border-right:1px solid #292c4b;}
    [data-testid="stMetric"] {background:linear-gradient(145deg,#171a32,#101224);border:1px solid #292c4b;border-radius:16px;padding:15px 18px;}
    [data-testid="stMetricLabel"] {color:#a7aed2 !important;}
    [data-testid="stMetricValue"] {color:#f5f5ff !important;font-weight:700;}
    div[data-baseweb="tab-list"] {gap:7px;border-bottom:1px solid #292c4b;}
    button[data-baseweb="tab"] {border-radius:10px 10px 0 0;background:#121429;color:#9ba2c7;padding:10px 15px;}
    button[data-baseweb="tab"][aria-selected="true"] {background:#22203b;color:#63f5ff;border-bottom:2px solid #ff5bd6;}
    .ticker {font:11px 'Space Mono',monospace;letter-spacing:.14em;color:#63f5ff;text-transform:uppercase;}
    .hero {border:1px solid #38345e;border-radius:23px;padding:25px 28px;background:
      radial-gradient(circle at 80% 20%,rgba(255,91,214,.14),transparent 22%),
      radial-gradient(circle at 68% 90%,rgba(99,245,255,.11),transparent 25%),
      linear-gradient(115deg,rgba(23,23,49,.98),rgba(13,17,36,.95));margin-bottom:15px;}
    .hero-title {font-size:clamp(2.4rem,5vw,4rem);font-weight:700;line-height:1.0;margin:10px 0;color:#f8f6ff;}
    .hero-title span {color:#63f5ff;text-shadow:0 0 26px rgba(99,245,255,.35);}
    .hero-copy {max-width:750px;color:#aeb5d9;line-height:1.65;font-size:1rem;}
    .pill {display:inline-block;border:1px solid #38345e;border-radius:99px;padding:6px 10px;margin:12px 6px 0 0;color:#d7ff62;background:#16182c;font:10px 'Space Mono',monospace;letter-spacing:.06em;}
    .panel {background:linear-gradient(145deg,rgba(20,22,44,.92),rgba(13,15,31,.94));border:1px solid #292c4b;border-radius:18px;padding:18px 20px;}
    .fun-panel {background:linear-gradient(135deg,rgba(42,23,62,.85),rgba(16,25,48,.9));border:1px solid #5b3b75;border-radius:18px;padding:17px 20px;}
    .section-label {font:11px 'Space Mono',monospace;color:#ff5bd6;letter-spacing:.16em;text-transform:uppercase;}
    .smallmono {font:11px 'Space Mono',monospace;color:#9ba2c7;}
    .callout {padding:13px 15px;border-radius:12px;background:rgba(99,245,255,.06);border:1px solid rgba(99,245,255,.18);color:#cceff2;line-height:1.55;}
    .caution {padding:13px 15px;border-radius:12px;background:rgba(255,196,87,.07);border:1px solid rgba(255,196,87,.22);color:#f2dfb5;line-height:1.55;}
    div[data-testid="stDataFrame"] {border:1px solid #292c4b;border-radius:12px;overflow:hidden;}
    .stButton button {border:1px solid #4b3f78;border-radius:11px;background:linear-gradient(100deg,#262347,#1b2540);color:#f6f5ff;font-weight:700;transition:all .18s ease;}
    .stButton button:hover {border-color:#63f5ff;color:#63f5ff;transform:translateY(-1px);box-shadow:0 0 18px rgba(99,245,255,.12);}
    .stDownloadButton button {border-radius:10px;background:#19253c;color:#63f5ff;border:1px solid #344567;}
    hr {border-color:#292c4b;}
    </style>
    """,
    unsafe_allow_html=True,
)

PLOT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk, sans-serif", color="#d6d8f2"),
    margin=dict(l=15, r=15, t=38, b=15),
    xaxis=dict(gridcolor="rgba(120,130,180,.13)", zerolinecolor="rgba(120,130,180,.16)"),
    yaxis=dict(gridcolor="rgba(120,130,180,.13)", zerolinecolor="rgba(120,130,180,.16)"),
)

@st.cache_data(show_spinner=False)
def read_csv(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw))

def parse_events(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"event_id", "time", "magnitude", "longitude", "latitude", "depth_km"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("Event-level CSV missing: " + ", ".join(sorted(missing)))
    out = frame.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    for c in ["magnitude", "longitude", "latitude", "depth_km"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["event_id"] = out["event_id"].astype(str)
    return out.dropna(subset=["time", "magnitude", "longitude", "latitude", "depth_km"]).drop_duplicates("event_id")

def nb_fit(series: pd.Series):
    x = pd.Series(series).dropna().astype(float)
    if len(x) < 2 or x.mean() <= 0:
        return None
    mu, var = float(x.mean()), float(x.var(ddof=1))
    if not np.isfinite(var) or var <= mu:
        return None
    r = mu**2 / (var - mu)
    p = r / (r + mu)
    return {"mean": mu, "variance": var, "r": r, "p": p}

def plot_style(fig, height=360):
    fig.update_layout(**PLOT, height=height)
    return fig

# ── Header ────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
      <div class="ticker">● SYSTEM ONLINE &nbsp; / &nbsp; AFTERSHOCK FORECAST</div>
      <div class="hero-title">SEISMO<span>RISK</span>_</div>
      <div class="hero-copy">Analyze historical earthquake sequences and estimate future event counts.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Load data ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="ticker">DATA INPUT / 001</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Drop an earthquake CSV", type=["csv"])
    st.caption("Auto-load priority: global sequence summary → global event records → Fiji sample.")

source = None
raw = None
if uploaded is not None:
    raw = read_csv(uploaded.getvalue())
    source = f"Upload / {uploaded.name}"
else:
    for path in [GLOBAL_SUMMARY, GLOBAL_EVENTS, FIJI_EVENTS]:
        if path.exists():
            raw = read_csv(path.read_bytes())
            source = str(path.relative_to(ROOT))
            break

if raw is None:
    st.markdown('<div class="panel"><h3>Waiting for data drop</h3><p>Upload <code>global_sequence_summary.csv</code> for sequence modelling or an event-level CSV for the globe.</p></div>', unsafe_allow_html=True)
    st.stop()

is_summary = "assigned_event_count" in raw.columns
events = None
seq = None
if is_summary:
    seq = raw.copy()
    seq["assigned_event_count"] = pd.to_numeric(seq["assigned_event_count"], errors="coerce")
    seq = seq.dropna(subset=["assigned_event_count"])
    seq["assigned_event_count"] = seq["assigned_event_count"].clip(lower=0).astype(int)
    counts = seq["assigned_event_count"]
    unit_label = "events / candidate sequence"
else:
    try:
        events = parse_events(raw)
    except Exception as e:
        st.error(f"Could not parse this CSV: {e}")
        st.stop()
    counts = None
    unit_label = "events / UTC day"

# ── Sidebar filters ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown('<div class="ticker">MODEL KNOBS / 002</div>', unsafe_allow_html=True)
    if is_summary:
        if "mainshock_magnitude" in seq.columns and len(seq):
            lo = float(np.floor(seq["mainshock_magnitude"].min() * 10) / 10)
            hi = float(np.ceil(seq["mainshock_magnitude"].max() * 10) / 10)
            if lo < hi:
                mags = st.slider("Mainshock magnitude", lo, hi, (lo, hi), .1)
                seq = seq[seq["mainshock_magnitude"].between(*mags)]
        counts = seq["assigned_event_count"].astype(int)
    else:
        default_start = events["time"].min().date()
        default_end = (events["time"].max().floor("D") + pd.Timedelta(days=1)).date()
        start = st.date_input("From (UTC)", value=default_start)
        end = st.date_input("To (exclusive)", value=default_end)
        minmag = float(np.floor(events["magnitude"].min() * 2) / 2)
        maxmag = float(np.ceil(events["magnitude"].max() * 2) / 2)
        if minmag >= maxmag:
            maxmag = minmag + .5
        mag_range = st.slider("Event magnitude", minmag, maxmag, (minmag, maxmag), .1)
        events = events[(events["time"] >= pd.Timestamp(start, tz="UTC")) &
                        (events["time"] < pd.Timestamp(end, tz="UTC")) &
                        events["magnitude"].between(*mag_range)].copy()
        events["date"] = events["time"].dt.floor("D")
        day_index = pd.date_range(pd.Timestamp(start, tz="UTC"),
                                  pd.Timestamp(end, tz="UTC") - pd.Timedelta(days=1),
                                  freq="D", tz="UTC")
        daily = events.groupby("date").size().reindex(day_index, fill_value=0).rename("count").reset_index()
        daily.columns = ["date", "count"]
        counts = daily["count"].astype(int)

# ── Model stats ───────────────────────────────────────────────────────
fit = nb_fit(counts)
n = len(counts)
mu = float(counts.mean()) if n else 0.0
var = float(counts.var(ddof=1)) if n > 1 else 0.0
vmr = var / mu if mu > 0 else np.nan

# ── Forecast dashboard ────────────────────────────────────────────────
st.markdown('<div class="section-label">01 / AFTERSHOCK ANALYSIS</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="panel"><h3>Historical sequence analysis</h3>'
    '<p>Explore the observed sequence counts, estimate aftershock-count probabilities, '
    'and inspect temporal and magnitude patterns. The dashboard focuses on statistical '
    'analysis and forecasting; the interactive globe has been removed.</p></div>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Main tabs ─────────────────────────────────────────────────────────
tab_lab, tab_stats, tab_explorer = st.tabs(["Aftershock forecast", "Data analysis", "Sequence explorer"])

with tab_lab:
    st.markdown('<div class="section-label">02 / AFTERSHOCK FORECAST</div>', unsafe_allow_html=True)
    st.header("Historical-data-based count forecast")

    if fit is None:
        st.warning("A Negative Binomial forecast requires at least two observations, a positive mean, and sample variance greater than the mean. Use the global sequence summary for sequence-level forecasts.")
    else:
        # The fitted data units are 30-day candidate sequences for the global summary,
        # or daily counts for event-level data. Scale expected count to the chosen horizon.
        base_days = 30.0 if is_summary else 1.0
        horizon = st.slider("Forecast horizon (days)", min_value=1, max_value=30, value=7)
        threshold = st.number_input("Count threshold k", min_value=1, max_value=10000, value=5, step=1)

        base_mean = float(counts.mean())
        base_var = float(counts.var(ddof=1))
        horizon_mean = base_mean * horizon / base_days

        # Preserve fitted overdispersion under proportional exposure scaling.
        fitted_r = float(fit["r"])
        horizon_r = fitted_r * horizon / base_days
        horizon_p = horizon_r / (horizon_r + horizon_mean)

        p_zero = float(nbinom.pmf(0, horizon_r, horizon_p))
        p_atleast_one = 1.0 - p_zero
        p_ge_k = float(nbinom.sf(int(threshold) - 1, horizon_r, horizon_p))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Expected count", f"{horizon_mean:.2f}")
        m2.metric("P(no events)", f"{p_zero:.2%}")
        m3.metric("P(at least one)", f"{p_atleast_one:.2%}")
        m4.metric(f"P(count ≥ {int(threshold)})", f"{p_ge_k:.2%}")

        max_k = max(int(nbinom.ppf(0.995, horizon_r, horizon_p)), int(horizon_mean + 5*np.sqrt(horizon_mean + horizon_mean**2/horizon_r)), 10)
        max_k = min(max_k, 500)
        k_values = np.arange(max_k + 1)
        probabilities = nbinom.pmf(k_values, horizon_r, horizon_p)

        forecast_fig = go.Figure()
        forecast_fig.add_trace(go.Bar(
            x=k_values, y=probabilities,
            marker_color="#63f5ff",
            name="Forecast probability",
            hovertemplate="Count: %{x}<br>Probability: %{y:.3%}<extra></extra>",
        ))
        forecast_fig.update_layout(**PLOT, height=420,
                                   title=f"Forecast count distribution over {horizon} day(s)",
                                   xaxis_title="Number of events",
                                   yaxis_title="Probability P(X = k)")
        st.plotly_chart(forecast_fig, use_container_width=True)

        st.caption(
            f"Forecast is scaled from the selected historical observations "
            f"(baseline mean {base_mean:.2f} per {base_days:g}-day unit; "
            f"sample variance {base_var:.2f})."
        )
        st.markdown(
            '<div class="caution"><b>Interpretation:</b> this is a statistical forecast of event counts based on the selected catalog and fitted Negative Binomial model. It does not predict an exact earthquake time, location, or magnitude, and is not an official warning.</div>',
            unsafe_allow_html=True,
        )

with tab_stats:
    st.markdown('<div class="section-label">03 / STATISTICAL CHAOS</div>', unsafe_allow_html=True)
    st.header("Statistical summary")
    if len(counts):
        if is_summary:
            hist = px.histogram(seq, x="assigned_event_count", nbins=45,
                                color_discrete_sequence=["#ff5bd6"],
                                labels={"assigned_event_count":"Assigned events per sequence"})
            hist.update_layout(title="Sequence-size distribution")
        else:
            hist = px.histogram(daily, x="count", nbins=35,
                                color_discrete_sequence=["#ff5bd6"],
                                labels={"count":"Events per day"})
            hist.update_layout(title="Daily count distribution")
        # Keep the sequence-size histogram and let it occupy the full row.
        st.plotly_chart(plot_style(hist, 360), use_container_width=True)
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Median", f"{counts.median():.2f}")
        s2.metric("Standard deviation", f"{counts.std(ddof=1):.2f}" if len(counts)>1 else "—")
        s3.metric("Min / max", f"{counts.min():.0f} / {counts.max():.0f}")
        s4.metric("VMR", f"{vmr:.2f}" if np.isfinite(vmr) else "—")
        st.markdown(
            f'<div class="callout">Mean = <b>{mu:.2f}</b> · Sample variance = <b>{var:.2f}</b> · '
            f'Variance-to-mean ratio = <b>{vmr:.2f}</b>. A ratio far above 1 signals overdispersion relative to the Poisson model, '
            'which is one reason we are exploring the Negative Binomial distribution.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("No observations in the current selection.")

with tab_explorer:
    st.markdown('<div class="section-label">04 / DATA EXPLORER</div>', unsafe_allow_html=True)
    st.header("Sequence data")
    if is_summary:
        cols = [c for c in ["mainshock_id", "mainshock_time", "mainshock_magnitude", "assigned_event_count"] if c in seq.columns]
        st.dataframe(seq[cols] if cols else seq, use_container_width=True, hide_index=True)
        st.download_button("Export sequence summary CSV", seq.to_csv(index=False).encode(), "seismorisk_sequence_summary.csv", "text/csv")
    else:
        st.dataframe(events, use_container_width=True, hide_index=True)
        st.download_button("Export filtered events CSV", events.to_csv(index=False).encode(), "seismorisk_filtered_events.csv", "text/csv")
        st.download_button("Export daily counts CSV", daily.to_csv(index=False).encode(), "seismorisk_daily_counts.csv", "text/csv")


st.markdown("---")
st.markdown(
    '<div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;">'
    '<span class="smallmono">SEISMORISK — PROBABILITY & STATISTICS</span>'
    '<span class="smallmono">HISTORICAL DATA AND STATISTICAL COUNT FORECASTS</span></div>',
    unsafe_allow_html=True,
)
