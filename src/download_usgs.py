import requests
import pandas as pd
from pathlib import Path

URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

start = pd.Timestamp("2015-01-01")
end = pd.Timestamp("2026-01-01")

all_rows = []
current = start

while current < end:
    next_month = current + pd.DateOffset(months=1)

    params = {
        "format": "geojson",
        "starttime": current.strftime("%Y-%m-%d"),
        "endtime": (next_month - pd.Timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%S"),
        "minmagnitude": 2.5,
        "limit": 20000,
        "orderby": "time-asc"
    }

    response = requests.get(URL, params=params, timeout=120)

    if not response.ok:
        print("Request failed:", response.status_code)
        print(response.text[:500])
        response.raise_for_status()

    events = response.json()["features"]

    for event in events:
        p = event["properties"]
        c = event["geometry"]["coordinates"]

        all_rows.append({
            "event_id": event["id"],
            "time": pd.to_datetime(p["time"], unit="ms", utc=True),
            "magnitude": p["mag"],
            "place": p["place"],
            "longitude": c[0],
            "latitude": c[1],
            "depth_km": c[2]
        })

    print(f"{current.strftime('%Y-%m')}: {len(events)} events")
    current = next_month

df = pd.DataFrame(all_rows)

if not df.empty:
    df = df.drop_duplicates(subset="event_id")
    df = df.sort_values("time")

output_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "earthquakes.csv"
df.to_csv(output_file, index=False)

print(f"\nTotal records saved: {len(df)}")
print(f"Output file: {output_file}")
print(df.head())
