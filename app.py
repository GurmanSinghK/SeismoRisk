from pathlib import Path
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import nbinom, poisson, ttest_ind

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "fiji_earthquake_sequence.csv"
MAINSHOCK_ID = "us1000gcii"
MAINSHOCK_TIME = pd.Timestamp("2018-08-19T00:19:40.670Z")
EXCLUDED_IDS = {MAINSHOCK_ID, "us2000h9e2"}

st.set_page_config(page_title="SeismoRisk | Earthquake Analytics", page_icon="🌐", layout="wide")

st.markdown("""
<style>
.stApp {background:linear-gradient(135deg,#07111f 0%,#0b1728 55%,#0a1c2e 100%);}
[data-testid="stSidebar"] {background:#081321;border-right:1px solid #19334b;}
[data-testid="stMetric"] {background:linear-gradient(145deg,#102238,#0c1b2b);border:1px solid #1c4058;padding:16px 18px;border-radius:14px;}
[data-testid="stMetricLabel"] {color:#9bb5ca !important;}
[data-testid="stMetricValue"] {color:#e8f7ff !important;}
h1,h2,h3 {color:#e9f6ff;}
p,li,label {color:#c6d8e6;}
.hero {background:linear-gradient(110deg,rgba(14,39,61,.98),rgba(9,26,43,.92));border:1px solid #1b5068;border-radius:18px;padding:24px 28px;margin-bottom:18px;}
.eyebrow {color:#43d9f5;font-size:.78rem;letter-spacing:.18em;text-transform:uppercase;font-weight:700;}
.subtle {color:#9bb5ca;}
.warning-box {border-left:4px solid #f0b44c;background:rgba(240,180,76,.09);padding:12px 16px;border-radius:6px;color:#f7dfad;}
.good-box {border-left:4px solid #37d6c0;background:rgba(55,214,192,.08);padding:12px 16px;border-radius:6px;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    raw = pd.read_csv(io.BytesIO(file_bytes))
    required = {"event_id", "time", "magnitude", "longitude", "latitude", "depth_km"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(sorted(missing)))
    raw["time"] = pd.to_datetime(raw["time"], format="mixed", utc=True, errors="coerce")
    for col in ["magnitude", "depth_km", "longitude", "latitude"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw["event_id"] = raw["event_id"].astype(str)
    return raw.dropna(subset=["time", "magnitude", "depth_km"]).copy()

st.sidebar.markdown("## Data")
uploaded = st.sidebar.file_uploader("Upload earthquake CSV", type=["csv"])
if uploaded is not None:
    try:
        df_all = load_csv_bytes(uploaded.getvalue())
        source_label = f"Uploaded: {uploaded.name}"
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()
elif DEFAULT_DATA.exists():
    try:
        df_all = load_csv_bytes(DEFAULT_DATA.read_bytes())
        source_label = "data/processed/fiji_earthquake_sequence.csv"
    except Exception as exc:
        st.error(f"Could not read default CSV: {exc}")
        st.stop()
else:
    st.title("SeismoRisk")
    st.info("Place your CSV at `data/processed/fiji_earthquake_sequence.csv` or upload it using the sidebar.")
    st.stop()

st.sidebar.markdown("## Analysis controls")
st.sidebar.caption("Filters recalculate the descriptive analysis and charts.")
use_reference_window = st.sidebar.checkbox("Use reference 30-day window", value=True)
if use_reference_window:
    default_start = MAINSHOCK_TIME
    default_end = MAINSHOCK_TIME + pd.Timedelta(days=30)
else:
    default_start = df_all["time"].min().floor("D")
    default_end = df_all["time"].max().floor("D") + pd.Timedelta(days=1)

with st.sidebar:
    start_date = st.date_input("Start date (UTC)", value=default_start.date(),
                               min_value=df_all["time"].min().date(),
                               max_value=(df_all["time"].max().date() + pd.Timedelta(days=1)))
    end_date = st.date_input("End date (exclusive, UTC)", value=default_end.date(),
                             min_value=df_all["time"].min().date(),
                             max_value=(df_all["time"].max().date() + pd.Timedelta(days=1)))
    mag_min = float(np.floor(df_all["magnitude"].min() * 2) / 2)
    mag_max = float(np.ceil(df_all["magnitude"].max() * 2) / 2)
    if mag_min == mag_max:
        mag_max = mag_min + 0.5
    mag_range = st.slider("Magnitude range", min_value=mag_min, max_value=mag_max,
                          value=(mag_min, mag_max), step=0.1)
    depth_min = float(max(0, np.floor(df_all["depth_km"].min())))
    depth_max = float(np.ceil(df_all["depth_km"].max()))
    depth_range = st.slider("Depth range (km)", min_value=depth_min, max_value=depth_max,
                            value=(depth_min, depth_max), step=10.0)
    exclude_major = st.checkbox("Exclude reference / later major events", value=True)

start = pd.Timestamp(start_date, tz="UTC")
end = pd.Timestamp(end_date, tz="UTC")
if end <= start:
    st.error("End date must be later than start date.")
    st.stop()

events = df_all[(df_all["time"] >= start) & (df_all["time"] < end)].copy()
if exclude_major:
    events = events[~events["event_id"].isin(EXCLUDED_IDS)]
events = events[events["magnitude"].between(*mag_range) & events["depth_km"].between(*depth_range)].copy()

# UTC daily bins covering the chosen [start, end) interval
day_index = pd.date_range(start=start.normalize(), end=(end - pd.Timedelta(nanoseconds=1)).normalize(), freq="D", tz="UTC")
in_window = events[(events["time"] >= start) & (events["time"] < end)].copy()
in_window["date"] = in_window["time"].dt.floor("D")
daily_counts = in_window.groupby("date").size().reindex(day_index, fill_value=0)
daily = pd.DataFrame({"date": daily_counts.index, "count": daily_counts.values})
counts = daily["count"].astype(int)
n_events, n_days = len(events), len(daily)
mean_count = float(counts.mean()) if n_days else 0.0
variance = float(counts.var(ddof=1)) if n_days > 1 else 0.0
vmr = variance / mean_count if mean_count > 0 else np.nan
if mean_count > 0 and variance > mean_count:
    nb_r = mean_count**2 / (variance - mean_count)
    nb_p = nb_r / (nb_r + mean_count)
else:
    nb_r, nb_p = None, None

st.markdown(f"""
<div class="hero">
<div class="eyebrow">Seismic statistics • Local research dashboard</div>
<h1 style="margin:8px 0 4px 0;">SeismoRisk</h1>
<div style="font-size:1.05rem;color:#b8d6e8;">Earthquake activity analytics & probability model exploration</div>
<div class="subtle" style="margin-top:12px;">Data source: {source_label} &nbsp;•&nbsp; Time zone: UTC</div>
</div>""", unsafe_allow_html=True)
st.markdown('<div class="warning-box"><b>Research caveat:</b> this dashboard analyzes selected regional earthquake events. The selection does not independently confirm that every event is a true aftershock, and this is not an operational earthquake prediction system.</div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Events in selection", f"{n_events:,}")
k2.metric("Mean magnitude", f"{events['magnitude'].mean():.2f}" if n_events else "—")
k3.metric("Mean depth", f"{events['depth_km'].mean():.2f} km" if n_events else "—")
k4.metric("Daily mean count", f"{mean_count:.2f}" if n_days else "—")

overview, models, relationships, tests, data_tab, predictions = st.tabs(["Overview", "Probability models", "Magnitude & depth", "Statistical tests", "Data explorer", "Predictions"])

with overview:
    left, right = st.columns([1.5, 1])
    with left:
        st.subheader("Daily earthquake activity")
        fig = px.bar(daily, x="date", y="count", labels={"date":"Date (UTC)","count":"Events"},
                     color="count", color_continuous_scale=["#164e63","#22d3ee","#67e8f9"])
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          coloraxis_showscale=False, margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Magnitude distribution")
        if n_events:
            fig = px.histogram(events, x="magnitude", nbins=18, labels={"magnitude":"Magnitude","count":"Events"},
                               color_discrete_sequence=["#22d3ee"])
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No events match the current filters.")
    a,b,c = st.columns(3)
    a.metric("Sample variance of daily counts", f"{variance:.2f}")
    b.metric("Variance-to-mean ratio", f"{vmr:.2f}" if np.isfinite(vmr) else "—")
    c.metric("Days with zero events", int((counts == 0).sum()) if n_days else 0)
    if np.isfinite(vmr) and vmr > 1:
        st.markdown(f'<div class="good-box">Variance-to-mean ratio = <b>{vmr:.2f}</b>, greater than 1. The daily counts are overdispersed relative to the Poisson equal-mean-and-variance property.</div>', unsafe_allow_html=True)
    st.caption("Daily bins are UTC calendar days. Filters can change counts and fitted parameters.")

with models:
    st.subheader("Poisson vs. Negative Binomial")
    st.write("These models describe the selected daily counts. Probabilities are fitted estimates, not guaranteed forecasts.")
    if n_days < 2 or mean_count <= 0:
        st.warning("At least two daily observations and a positive mean count are needed.")
    else:
        ks = np.arange(0, max(int(counts.max()), 10) + 1)
        pois = poisson.pmf(ks, mean_count)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ks, y=pois, name="Poisson", marker_color="#22d3ee", opacity=.75))
        if nb_r is not None:
            fig.add_trace(go.Scatter(x=ks, y=nbinom.pmf(ks, nb_r, nb_p), name="Negative Binomial",
                                     mode="lines+markers", line=dict(color="#fbbf24", width=2)))
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          xaxis_title="Daily event count (k)", yaxis_title="Probability P(N = k)")
        st.plotly_chart(fig, width="stretch")
        a,b,c = st.columns(3)
        a.metric("Poisson λ", f"{mean_count:.4f}")
        b.metric("Observed variance", f"{variance:.4f}")
        c.metric("Variance / mean", f"{vmr:.4f}" if np.isfinite(vmr) else "—")
        if nb_r is not None:
            st.write(f"Negative Binomial method-of-moments estimates: **r = {nb_r:.4f}**, **p = {nb_p:.4f}**.")
        else:
            st.info("Negative Binomial method-of-moments parameters are not available when variance is not greater than the mean.")
        kfreq = np.arange(0, int(counts.max()) + 1)
        freq = pd.DataFrame({
            "Daily count": kfreq,
            "Observed days": [(counts == k).sum() for k in kfreq],
            "Poisson expected days": poisson.pmf(kfreq, mean_count) * n_days,
        })
        if nb_r is not None:
            freq["Negative Binomial expected days"] = nbinom.pmf(kfreq, nb_r, nb_p) * n_days
        st.subheader("Observed vs. expected number of days")
        st.dataframe(freq, width="stretch", hide_index=True)
        st.download_button("Download model frequency table (CSV)", freq.to_csv(index=False).encode(), "seismorisk_model_frequencies.csv", "text/csv")
        st.markdown('<div class="warning-box"><b>Interpretation:</b> overdispersion does not prove either model is adequate. Compare observed and expected frequencies, and do not use these curves as actionable earthquake warnings.</div>', unsafe_allow_html=True)

with relationships:
    st.subheader("Magnitude vs. depth")
    if n_events >= 2:
        corr = events["magnitude"].corr(events["depth_km"], method="pearson")
        hover = ["event_id","time"] + (["place"] if "place" in events.columns else [])
        fig = px.scatter(events, x="depth_km", y="magnitude", hover_data=hover,
                         labels={"depth_km":"Depth (km)","magnitude":"Magnitude"},
                         color_discrete_sequence=["#22d3ee"], opacity=.75)
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")
        a,b,c = st.columns(3)
        a.metric("Pearson correlation", f"{corr:.4f}")
        b.metric("Mean magnitude", f"{events['magnitude'].mean():.4f}")
        c.metric("Mean depth", f"{events['depth_km'].mean():.2f} km")
        if abs(corr) < .1:
            st.info("The observed linear association is negligible in this selected sample. Correlation does not establish causation.")
        elif corr > 0:
            st.info("The selected sample has a positive linear association. Correlation does not establish causation.")
        else:
            st.info("The selected sample has a negative linear association. Correlation does not establish causation.")
        events_depth = events.copy()
        events_depth["Depth group"] = pd.cut(events_depth["depth_km"], [-np.inf,70,300,np.inf],
                                             labels=["Shallow (≤70 km)","Intermediate (70–300 km)","Deep (>300 km)"])
        depth_summary = events_depth.groupby("Depth group", observed=False)["magnitude"].agg(
            count="count", mean="mean", median="median", std="std", min="min", max="max").reset_index()
        st.subheader("Magnitude summary by depth group")
        st.dataframe(depth_summary.round(3), width="stretch", hide_index=True)
        st.caption("These are descriptive comparisons. Very small groups make between-group conclusions uncertain.")
    else:
        st.info("At least two events are needed for correlation.")

with tests:
    st.subheader("Exploratory comparison of two periods")
    st.write("Welch's two-sample t-test compares the mean daily counts in an early period and a following period.")
    if n_days >= 5:
        max_early = min(10, n_days - 2)
        early_n = st.slider("Number of initial days in early period", 2, max_early, min(5, max_early))
        later_n = st.slider("Number of following days in later period", 2, n_days - early_n, min(10, n_days - early_n))
        early = counts.iloc[:early_n].to_numpy(dtype=float)
        later = counts.iloc[early_n:early_n + later_n].to_numpy(dtype=float)
        t_stat, p_value = ttest_ind(early, later, equal_var=False)
        a,b,c,d = st.columns(4)
        a.metric("Early mean", f"{early.mean():.3f}")
        b.metric("Later mean", f"{later.mean():.3f}")
        c.metric("t statistic", f"{t_stat:.4f}")
        d.metric("p-value", f"{p_value:.4f}")
        st.markdown("**H₀:** the two periods have equal population mean daily counts.")
        st.markdown("**H₁:** the two periods have different population mean daily counts.")
        if p_value < .05:
            st.write("At the 5% significance level, reject H₀ for this exploratory test.")
        else:
            st.write("At the 5% significance level, fail to reject H₀. This does not prove the means are equal.")
        st.markdown('<div class="warning-box">Daily counts can be skewed and time-dependent. A short sample with extreme counts makes this test uncertain; treat it as exploratory.</div>', unsafe_allow_html=True)
    else:
        st.info("At least five daily observations are needed for this comparison.")

with data_tab:
    st.subheader("Filtered earthquake records")
    cols = [c for c in ["event_id","time","magnitude","depth_km","latitude","longitude","place"] if c in events.columns]
    st.dataframe(events[cols].sort_values("time"), width="stretch", hide_index=True)
    st.download_button("Download filtered earthquake data (CSV)", events[cols].to_csv(index=False).encode(), "seismorisk_filtered_events.csv", "text/csv")
    st.download_button("Download daily counts (CSV)", daily.to_csv(index=False).encode(), "seismorisk_daily_counts.csv", "text/csv")
    summary = {
        "data_source": source_label, "analysis_start_utc": start.isoformat(), "analysis_end_exclusive_utc": end.isoformat(),
        "events_analyzed": n_events, "days_observed": n_days,
        "mean_magnitude": float(events["magnitude"].mean()) if n_events else None,
        "mean_depth_km": float(events["depth_km"].mean()) if n_events else None,
        "mean_daily_count": mean_count, "sample_variance_daily_count": variance,
        "variance_to_mean_ratio": float(vmr) if np.isfinite(vmr) else None,
        "pearson_magnitude_depth": float(events["magnitude"].corr(events["depth_km"])) if n_events > 1 else None,
        "reference_event_id": MAINSHOCK_ID, "excluded_event_ids": sorted(EXCLUDED_IDS) if exclude_major else [],
        "caveat": "Selected regional earthquakes are not independently verified as true aftershocks; not an operational forecast."
    }
    st.download_button("Download analysis summary (JSON)", pd.Series(summary).to_json(indent=2).encode(), "seismorisk_summary.json", "application/json")


with predictions:
    st.subheader("Experimental event-count projections")
    st.markdown(
        '<div class="warning-box"><b>Important:</b> These are model-based projections from the selected historical daily counts, not reliable earthquake forecasts. '
        'The models fit this dataset poorly and assume the selected daily-count pattern can represent the next period. '
        'They do not predict the time, location, depth, or magnitude of a future earthquake.</div>',
        unsafe_allow_html=True
    )
    horizon = st.radio("Projection period", [1, 3, 7], horizontal=True,
                       format_func=lambda x: f"Next {x} day" if x == 1 else f"Next {x} days")
    if n_days < 2 or mean_count <= 0:
        st.warning("Select at least two daily observations with a positive mean count to calculate projections.")
    else:
        poisson_expected = mean_count * horizon
        poisson_probability = float(1 - poisson.pmf(0, poisson_expected))
        st.caption(
            f"Baseline: {n_days} selected daily bins; historical mean = {mean_count:.2f} events/day. "
            "This is a constant-rate extrapolation, not a time-varying aftershock-decay model."
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Poisson expected count", f"{poisson_expected:.1f}")
        c2.metric("Poisson P(at least 1 event)", f"{poisson_probability:.1%}")
        if nb_r is not None:
            nb_expected = mean_count * horizon
            nb_probability = float(1 - nbinom.pmf(0, nb_r * horizon, nb_p))
            c3.metric("Negative Binomial expected count", f"{nb_expected:.1f}")
        else:
            nb_probability = None
            c3.metric("Negative Binomial expected count", "Unavailable")

        comparison_rows = [
            {"Model": "Poisson", "Expected count over period": poisson_expected,
             "P(at least 1 event)": poisson_probability}
        ]
        if nb_r is not None:
            comparison_rows.append({
                "Model": "Negative Binomial",
                "Expected count over period": nb_expected,
                "P(at least 1 event)": nb_probability
            })
        comparison = pd.DataFrame(comparison_rows)
        st.dataframe(comparison, hide_index=True, width="stretch")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=comparison["Model"], y=comparison["Expected count over period"],
            marker_color=["#22d3ee", "#fbbf24"][:len(comparison)],
            text=[f"{v:.1f}" for v in comparison["Expected count over period"]],
            textposition="outside"
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", yaxis_title="Expected number of events",
            xaxis_title="", margin=dict(l=10, r=10, t=25, b=10)
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown("**How to read this:** expected count is a model average, not a promise of how many events will occur. "
                    "The probability of at least one event is calculated from each fitted distribution.")
        st.markdown(
            '<div class="warning-box">Model limitations: the observed counts are highly overdispersed and contain large spikes. '
            'The Negative Binomial calculation assumes its shape parameter scales with the number of days and that daily counts behave independently. '
            'Neither model has been validated for prospective forecasting, so treat these values as a demonstration of the statistical models only.</div>',
            unsafe_allow_html=True
        )

st.markdown("---")
st.caption("SeismoRisk is an academic statistical exploration. It does not predict earthquakes or provide safety warnings.")
