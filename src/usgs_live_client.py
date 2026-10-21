"""
USGS Live Client Module for SeismoRisk
Handles reliable, cached, timeout-safe interactions with the official USGS earthquake endpoints.
"""

from datetime import datetime, timezone
import json
import urllib.request
import urllib.parse
from typing import Dict, Any, List, Optional
import pandas as pd

FEED_URLS = {
    "past_day_m45": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_day.geojson",
    "past_week_m45": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_week.geojson",
    "past_month_m45": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_month.geojson",
    "past_day_significant": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_day.geojson",
    "past_week_significant": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_week.geojson",
    "past_month_significant": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_month.geojson",
}

FDSNWS_QUERY_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def parse_geojson_features(features: List[Dict[str, Any]]) -> pd.DataFrame:
    """Parses GeoJSON features into a clean pandas DataFrame with strict typing."""
    records = []
    for f in features:
        props = f.get("properties") or {}
        geom = f.get("geometry") or {}
        coords = geom.get("coordinates") or [None, None, None]
        
        event_id = str(f.get("id") or props.get("code") or "")
        mag = props.get("mag")
        time_ms = props.get("time")
        updated_ms = props.get("updated")
        
        # Parse timestamps safely
        time_utc = pd.to_datetime(time_ms, unit="ms", utc=True) if time_ms is not None else pd.NaT
        updated_utc = pd.to_datetime(updated_ms, unit="ms", utc=True) if updated_ms is not None else pd.NaT
        
        lon = float(coords[0]) if len(coords) > 0 and coords[0] is not None else None
        lat = float(coords[1]) if len(coords) > 1 and coords[1] is not None else None
        depth = float(coords[2]) if len(coords) > 2 and coords[2] is not None else None
        
        records.append({
            "event_id": event_id,
            "title": str(props.get("title") or f"M {mag} Earthquake"),
            "place": str(props.get("place") or "Unknown Region"),
            "magnitude": float(mag) if mag is not None else None,
            "time_utc": time_utc,
            "updated_utc": updated_utc,
            "latitude": lat,
            "longitude": lon,
            "depth_km": depth,
            "url": str(props.get("url") or ""),
            "status": str(props.get("status") or "automatic"),
            "significance": int(props.get("sig") or 0),
        })
        
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.dropna(subset=["event_id", "time_utc", "magnitude", "latitude", "longitude"])
        df = df.sort_values("time_utc", ascending=False).reset_index(drop=True)
    return df


def fetch_live_usgs_feed(feed_key: str = "past_week_m45", timeout_secs: int = 8) -> Dict[str, Any]:
    """
    Fetches real-time GeoJSON summaries from the official USGS feed.
    Returns dictionary with status, timestamp, DataFrame, and metadata.
    """
    url = FEED_URLS.get(feed_key, FEED_URLS["past_week_m45"])
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "SeismoRisk-Academic-Analytics/2.0 (University Research Tool)"}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=timeout_secs) as resp:
            if resp.status != 200:
                return {
                    "success": False,
                    "error": f"USGS HTTP Error {resp.status}",
                    "data": pd.DataFrame(),
                    "fetched_at": datetime.now(timezone.utc)
                }
            payload = json.loads(resp.read().decode("utf-8"))
            features = payload.get("features") or []
            df = parse_geojson_features(features)
            
            return {
                "success": True,
                "error": None,
                "data": df,
                "count": len(df),
                "fetched_at": datetime.now(timezone.utc),
                "title": payload.get("metadata", {}).get("title", "USGS Real-Time Feed")
            }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "data": pd.DataFrame(),
            "count": 0,
            "fetched_at": datetime.now(timezone.utc),
            "title": "USGS Feed (Failed)"
        }


def query_usgs_fdsnws(
    starttime: str,
    endtime: str,
    minmagnitude: float = 4.5,
    maxmagnitude: Optional[float] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    maxradiuskm: Optional[float] = None,
    limit: int = 500,
    timeout_secs: int = 10
) -> Dict[str, Any]:
    """
    Queries the official USGS FDSNWS event endpoint with space, time, and magnitude filters.
    """
    params = {
        "format": "geojson",
        "starttime": starttime,
        "endtime": endtime,
        "minmagnitude": str(minmagnitude),
        "orderby": "time-asc",
        "limit": str(limit)
    }
    if maxmagnitude is not None:
        params["maxmagnitude"] = str(maxmagnitude)
    if latitude is not None and longitude is not None and maxradiuskm is not None:
        params["latitude"] = str(latitude)
        params["longitude"] = str(longitude)
        params["maxradiuskm"] = str(maxradiuskm)
        
    query_str = urllib.parse.urlencode(params)
    full_url = f"{FDSNWS_QUERY_URL}?{query_str}"
    
    req = urllib.request.Request(
        full_url,
        headers={"User-Agent": "SeismoRisk-Academic-Analytics/2.0"}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=timeout_secs) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            features = payload.get("features") or []
            df = parse_geojson_features(features)
            return {
                "success": True,
                "error": None,
                "data": df,
                "count": len(df),
                "query_url": full_url,
                "fetched_at": datetime.now(timezone.utc)
            }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "data": pd.DataFrame(),
            "count": 0,
            "query_url": full_url,
            "fetched_at": datetime.now(timezone.utc)
        }
