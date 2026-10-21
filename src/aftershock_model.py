"""
Aftershock Probability and Time-Dependent Rate Modeling Engine for SeismoRisk
Implements Modified Omori-Utsu, Reasenberg-Jones empirical scaling,
Non-Homogeneous Poisson and Negative Binomial mixture probability calculations.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import poisson, nbinom

EARTH_RADIUS_KM = 6371.0

# --------------------------------------------------------------------------
# Empirically fitted global parameters (from 2000-2017 training sequence split)
# Omori rate: lambda(t) = K / (t + c)^p
# Magnitude scaling: log10(K) = a + b * (M_main - M_target)
# --------------------------------------------------------------------------
OMORI_P_GLOBAL = 0.9029   # Empirical temporal decay exponent
OMORI_C_DAYS = 0.0500     # Time offset preventing singularity at t=0
SCALING_B_GLOBAL = 0.6920 # Utsu magnitude productivity slope
SCALING_A_GLOBAL = -1.250 # Normalization constant for M_target = 4.5
DISPERSION_R_GLOBAL = 0.3108 # Negative Binomial shape parameter


def haversine_distance_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Computes great-circle distances in kilometres on a spherical Earth."""
    phi1, lambda1 = np.radians(lat1), np.radians(lon1)
    phi2, lambda2 = np.radians(lat2), np.radians(lon2)
    dphi = phi2 - phi1
    dlambda = lambda2 - lambda1
    
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
    return EARTH_RADIUS_KM * c


def integrate_omori_rate(
    t_start_days: float,
    t_end_days: float,
    K: float,
    c: float = OMORI_C_DAYS,
    p: float = OMORI_P_GLOBAL
) -> float:
    """
    Integrates the modified Omori-Utsu rate lambda(t) = K / (t + c)^p from t_start to t_end.
    Yields expected count Lambda over the time interval.
    """
    if t_end_days <= t_start_days or K <= 0.0:
        return 0.0
    
    t1 = max(0.0, t_start_days)
    t2 = max(t1, t_end_days)
    
    if abs(p - 1.0) < 1e-5:
        integral = K * (np.log(t2 + c) - np.log(t1 + c))
    else:
        integral = (K / (1.0 - p)) * ((t2 + c)**(1.0 - p) - (t1 + c)**(1.0 - p))
        
    return max(0.0, float(integral))


def compute_aftershock_expectation(
    mainshock_mag: float,
    target_min_mag: float,
    t_start_days: float,
    t_end_days: float,
    radius_km: float = 100.0,
    reference_radius_km: float = 100.0
) -> float:
    """
    Computes expected number of triggered events Lambda(t_start, t_end, M >= M_target | M_main)
    combining Reasenberg-Jones magnitude scaling and Omori temporal integration.
    """
    # 1. Base productivity K for M_target
    delta_m = mainshock_mag - target_min_mag
    log10_K = SCALING_A_GLOBAL + SCALING_B_GLOBAL * delta_m
    K_mag = 10.0 ** log10_K
    
    # 2. Spatial area scaling (relative to 100 km standard circle)
    # Area ratio capped to prevent extreme extrapolation
    area_ratio = min(4.0, max(0.04, (radius_km / reference_radius_km) ** 1.5))
    K_eff = K_mag * area_ratio
    
    # 3. Integrate temporal decay
    expected_lambda = integrate_omori_rate(t_start_days, t_end_days, K=K_eff, c=OMORI_C_DAYS, p=OMORI_P_GLOBAL)
    return expected_lambda


def calculate_aftershock_probabilities(
    expected_lambda: float,
    dispersion_r: float = DISPERSION_R_GLOBAL,
    max_k: int = 10
) -> Dict[str, Any]:
    """
    Calculates exact Poisson and Negative Binomial probability distributions for expected count Lambda.
    """
    lam = max(0.0, float(expected_lambda))
    
    # 1. Poisson model (equidispersed baseline)
    p_zero_pois = float(np.exp(-lam))
    p_ge1_pois = float(1.0 - p_zero_pois)
    p_ge2_pois = float(1.0 - poisson.cdf(1, lam))
    
    # 2. Negative Binomial model (overdispersed Poisson-Gamma mixture)
    # For a given mean lam and shape r, success probability p_nb = r / (r + lam)
    if lam > 0.0 and dispersion_r > 0.0:
        p_nb = float(dispersion_r / (dispersion_r + lam))
        p_zero_nb = float(nbinom.pmf(0, dispersion_r, p_nb))
        p_ge1_nb = float(1.0 - p_zero_nb)
        p_ge2_nb = float(1.0 - nbinom.cdf(1, dispersion_r, p_nb))
        variance_nb = float(lam + (lam**2) / dispersion_r)
    else:
        p_nb = 1.0
        p_zero_nb = 1.0
        p_ge1_nb = 0.0
        p_ge2_nb = 0.0
        variance_nb = 0.0
        
    ks = np.arange(0, max_k + 1)
    pmf_pois = [float(poisson.pmf(k, lam)) for k in ks]
    pmf_nb = [float(nbinom.pmf(k, dispersion_r, p_nb)) if lam > 0 else (1.0 if k==0 else 0.0) for k in ks]
    
    # Uncertainty intervals (95% predictive interval for counts)
    pi_low_pois = int(poisson.ppf(0.025, lam)) if lam > 0 else 0
    pi_high_pois = int(poisson.ppf(0.975, lam)) if lam > 0 else 0
    pi_low_nb = int(nbinom.ppf(0.025, dispersion_r, p_nb)) if lam > 0 else 0
    pi_high_nb = int(nbinom.ppf(0.975, dispersion_r, p_nb)) if lam > 0 else 0
    
    return {
        "expected_lambda": lam,
        "poisson": {
            "p_zero": p_zero_pois,
            "p_at_least_one": p_ge1_pois,
            "p_at_least_two": p_ge2_pois,
            "variance": lam,
            "interval_95": (pi_low_pois, pi_high_pois),
            "pmf_k": pmf_pois,
        },
        "negative_binomial": {
            "p_zero": p_zero_nb,
            "p_at_least_one": p_ge1_nb,
            "p_at_least_two": p_ge2_nb,
            "variance": variance_nb,
            "r_param": dispersion_r,
            "p_param": p_nb,
            "interval_95": (pi_low_nb, pi_high_nb),
            "pmf_k": pmf_nb,
        },
        "k_values": ks.tolist()
    }


def filter_qualifying_events(
    candidate_events_df: pd.DataFrame,
    mainshock_id: str,
    mainshock_lat: float,
    mainshock_lon: float,
    mainshock_time: pd.Timestamp,
    forecast_start_time: pd.Timestamp,
    forecast_end_time: pd.Timestamp,
    target_min_mag: float,
    radius_km: float
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Filters events that strictly qualify as subsequent events relative to a reference mainshock.
    Explicitly excludes the mainshock itself, enforces spatial radius, time boundaries, and magnitude.
    """
    if candidate_events_df.empty:
        return pd.DataFrame(), {"total_inspected": 0, "qualifying": 0}
        
    df = candidate_events_df.copy()
    
    # 1. Exclude the reference mainshock itself
    df = df[df["event_id"] != str(mainshock_id)]
    
    # 2. Must meet target minimum magnitude
    df = df[df["magnitude"] >= float(target_min_mag)]
    
    # 3. Time bounds: strictly between forecast_start_time and forecast_end_time
    # and strictly after the mainshock origin time
    df = df[(df["time_utc"] > mainshock_time) & (df["time_utc"] >= forecast_start_time) & (df["time_utc"] <= forecast_end_time)]
    
    if df.empty:
        return pd.DataFrame(), {"total_inspected": len(candidate_events_df), "qualifying": 0}
        
    # 4. Spatial distance constraint
    lats = df["latitude"].to_numpy(dtype=float)
    lons = df["longitude"].to_numpy(dtype=float)
    dists = haversine_distance_km(mainshock_lat, mainshock_lon, lats, lons)
    df["distance_km"] = dists
    
    df = df[df["distance_km"] <= float(radius_km)].copy()
    df["days_after_mainshock"] = (df["time_utc"] - mainshock_time).dt.total_seconds() / 86400.0
    df = df.sort_values("time_utc").reset_index(drop=True)
    
    meta = {
        "total_inspected": len(candidate_events_df),
        "qualifying": len(df),
        "mainshock_id": str(mainshock_id),
        "min_dist_km": float(df["distance_km"].min()) if not df.empty else None,
        "max_dist_km": float(df["distance_km"].max()) if not df.empty else None
    }
    return df, meta
