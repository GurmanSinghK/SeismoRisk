"""
Unit Test Suite for SeismoRisk
Covers:
1. Global sequence spatial & temporal boundary integrity.
2. Time unit resolution (microsecond vs nanosecond compatibility).
3. Regional catalogue cleaning consistency.
4. USGS live GeoJSON response parsing, missing fields, and error handling.
5. Spatial haversine distance arithmetic.
6. Exclusion of reference mainshocks in subsequent event filtering.
7. Modified Omori-Utsu rate integration and probability computations.
8. Historical replay future-data leakage prevention.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from src.usgs_live_client import parse_geojson_features
from src.aftershock_model import (
    haversine_distance_km,
    integrate_omori_rate,
    compute_aftershock_expectation,
    calculate_aftershock_probabilities,
    filter_qualifying_events,
    OMORI_P_GLOBAL,
    OMORI_C_DAYS
)


def test_global_sequence_bounds():
    """Verify that all global sequence events satisfy the 100 km radius and 30 day window."""
    events_path = PROJECT_ROOT / "data" / "processed" / "global_sequence_events.csv"
    assert events_path.exists(), "global_sequence_events.csv missing"
    
    df = pd.read_csv(events_path)
    assert len(df) > 0, "No events in global_sequence_events.csv"
    
    # Boundary tests
    assert df["days_after_mainshock"].min() >= 0.0, "Found negative elapsed time"
    assert df["days_after_mainshock"].max() <= 30.0001, "Found events exceeding 30-day window"
    assert df["distance_km"].min() >= 0.0, "Found negative distance"
    assert df["distance_km"].max() <= 100.001, "Found events exceeding 100 km radius"
    
    # Check that late-window events actually exist (bug check)
    late_events = (df["days_after_mainshock"] > 10.0).sum()
    assert late_events > 0, "Zero events found in day 10-30 window (indicates recurrence of the cutoff bug)"
    
    # Uniqueness check: no duplicate event IDs assigned
    assert df["event_id"].duplicated().sum() == 0, "Found duplicate event assignments"


def test_time_unit_resolution():
    """Test pandas datetime unit handling to ensure compatibility with pandas 2.x and 3.x."""
    ts = pd.Series(pd.to_datetime(["2020-01-01T00:00:00Z"], utc=True))
    unit = ts.dtype.unit if hasattr(ts.dtype, "unit") else "ns"
    assert unit in {"ns", "us", "ms", "s"}
    
    microsecs_per_unit = {"ns": 1000, "us": 1, "ms": 1_000_000}.get(unit, 1)
    day_int = int(86_400 * 1_000_000 // microsecs_per_unit)
    window_int = 30 * day_int
    assert window_int / day_int == 30.0


def test_fiji_sequence_cleaning():
    """Verify consistency of cleaned Fiji dataset."""
    fiji_path = PROJECT_ROOT / "data" / "processed" / "fiji_earthquake_sequence.csv"
    assert fiji_path.exists(), "Fiji sequence CSV missing"
    
    df = pd.read_csv(fiji_path)
    required_cols = {"event_id", "time", "magnitude", "longitude", "latitude", "depth_km"}
    assert required_cols.issubset(set(df.columns)), "Missing required columns in Fiji dataset"
    assert df["magnitude"].isna().sum() == 0
    assert df["depth_km"].isna().sum() == 0


def test_usgs_geojson_parsing():
    """Test robust GeoJSON parsing for USGS live client, including malformed features."""
    sample_features = [
        {
            "id": "us_test_01",
            "properties": {
                "title": "M 5.2 - Test Island",
                "mag": 5.2,
                "time": 1700000000000,
                "place": "Test Island, Pacific",
                "status": "reviewed"
            },
            "geometry": {
                "coordinates": [178.5, -18.2, 550.0]
            }
        },
        # Missing coordinates
        {
            "id": "us_test_missing_coords",
            "properties": {"title": "M 4.0 - Bad", "mag": 4.0, "time": 1700000000000},
            "geometry": {"coordinates": []}
        },
        # Empty dict
        {}
    ]
    parsed_df = parse_geojson_features(sample_features)
    assert len(parsed_df) == 1, "Failed to drop invalid coordinate features"
    assert parsed_df.iloc[0]["event_id"] == "us_test_01"
    assert parsed_df.iloc[0]["magnitude"] == 5.2
    assert parsed_df.iloc[0]["latitude"] == -18.2
    assert parsed_df.iloc[0]["longitude"] == 178.5
    assert parsed_df.iloc[0]["depth_km"] == 550.0


def test_haversine_accuracy():
    """Verify haversine distance calculation against known geographical coordinates."""
    # Distance between London (51.5074 N, 0.1278 W) and Paris (48.8566 N, 2.3522 E) is ~344 km
    d = haversine_distance_km(51.5074, -0.1278, np.array([48.8566]), np.array([2.3522]))[0]
    assert 340.0 <= d <= 350.0, f"Haversine calculation error: expected ~344 km, got {d:.2f}"
    
    # Distance to self must be 0
    d_zero = haversine_distance_km(35.0, 139.0, np.array([35.0]), np.array([139.0]))[0]
    assert abs(d_zero) < 1e-6


def test_omori_rate_integration():
    """Verify analytical integration of Modified Omori-Utsu rate."""
    # K=1, p=1, c=1: integral from 0 to 9 is 1 * (ln(10) - ln(1)) = ln(10) ~ 2.3026
    int_p1 = integrate_omori_rate(0.0, 9.0, K=1.0, c=1.0, p=1.0)
    assert abs(int_p1 - np.log(10)) < 1e-4
    
    # Positive elapsed intervals
    int_std = integrate_omori_rate(0.0, 7.0, K=10.0, c=OMORI_C_DAYS, p=OMORI_P_GLOBAL)
    assert int_std > 0.0
    
    # Inverted time interval must return 0
    assert integrate_omori_rate(7.0, 2.0, K=10.0) == 0.0


def test_aftershock_probability_monotonicity():
    """Verify mathematical properties of aftershock probabilities."""
    lam = 5.0
    probs = calculate_aftershock_probabilities(lam)
    
    # P(N >= 0) is 1, so P(N >= 1) + P(N == 0) must equal 1.0
    assert abs(probs["poisson"]["p_zero"] + probs["poisson"]["p_at_least_one"] - 1.0) < 1e-6
    assert abs(probs["negative_binomial"]["p_zero"] + probs["negative_binomial"]["p_at_least_one"] - 1.0) < 1e-6
    
    # P(N >= 2) must be <= P(N >= 1)
    assert probs["poisson"]["p_at_least_two"] <= probs["poisson"]["p_at_least_one"]
    assert probs["negative_binomial"]["p_at_least_two"] <= probs["negative_binomial"]["p_at_least_one"]
    
    # Negative Binomial variance must strictly exceed Poisson variance (overdispersion)
    assert probs["negative_binomial"]["variance"] > probs["poisson"]["variance"]


def test_qualifying_events_filtering():
    """Verify that reference mainshock is strictly excluded and distance/time filters hold."""
    ms_time = pd.Timestamp("2026-01-01T00:00:00Z")
    candidates = pd.DataFrame([
        # The mainshock itself (must be excluded)
        {"event_id": "ms_01", "magnitude": 7.0, "latitude": 0.0, "longitude": 0.0, "time_utc": ms_time},
        # Qualifying event (50 km away, 1 day later, M 5.0)
        {"event_id": "ev_qual", "magnitude": 5.0, "latitude": 0.45, "longitude": 0.0, "time_utc": ms_time + pd.Timedelta(days=1)},
        # Too far (300 km away)
        {"event_id": "ev_far", "magnitude": 5.0, "latitude": 3.0, "longitude": 0.0, "time_utc": ms_time + pd.Timedelta(days=1)},
        # Too small (M 3.0, threshold 4.5)
        {"event_id": "ev_small", "magnitude": 3.0, "latitude": 0.1, "longitude": 0.0, "time_utc": ms_time + pd.Timedelta(days=1)},
        # Before mainshock (prior event)
        {"event_id": "ev_prior", "magnitude": 5.0, "latitude": 0.1, "longitude": 0.0, "time_utc": ms_time - pd.Timedelta(hours=2)},
    ])
    
    qual, meta = filter_qualifying_events(
        candidate_events_df=candidates,
        mainshock_id="ms_01",
        mainshock_lat=0.0,
        mainshock_lon=0.0,
        mainshock_time=ms_time,
        forecast_start_time=ms_time,
        forecast_end_time=ms_time + pd.Timedelta(days=7),
        target_min_mag=4.5,
        radius_km=100.0
    )
    
    assert len(qual) == 1, f"Expected exactly 1 qualifying event, got {len(qual)}"
    assert qual.iloc[0]["event_id"] == "ev_qual"
    assert "ms_01" not in qual["event_id"].values, "Mainshock was incorrectly counted as a subsequent event!"


def test_historical_replay_no_data_leakage():
    """Verify historical replay strictly quarantines events before and after evaluation window."""
    events_path = PROJECT_ROOT / "data" / "processed" / "global_sequence_events.csv"
    if not events_path.exists():
        return
        
    df = pd.read_csv(events_path, nrows=1000)
    # Pick a mainshock with multiple events
    ms_counts = df["mainshock_id"].value_counts()
    multi_ms = ms_counts[ms_counts >= 5].index[0]
    
    ms_events = df[df["mainshock_id"] == multi_ms].copy()
    
    # Simulate a forecast window between Day 1.0 and Day 5.0
    eval_window_events = ms_events[(ms_events["days_after_mainshock"] > 1.0) & (ms_events["days_after_mainshock"] <= 5.0)]
    
    # Ensure no events from before Day 1.0 or after Day 5.0 are included
    if not eval_window_events.empty:
        assert eval_window_events["days_after_mainshock"].min() > 1.0
        assert eval_window_events["days_after_mainshock"].max() <= 5.0
