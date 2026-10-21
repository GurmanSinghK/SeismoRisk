
from pathlib import Path
import time
import requests
import pandas as pd

BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
START_YEAR = 2000
END_YEAR = 2025
MIN_MAGNITUDE = 4.5
LIMIT = 20000

OUTPUT = Path("data/raw/global_earthquakes_2000_2025.csv")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

records = []

for year in range(START_YEAR, END_YEAR + 1):
    params = {
        "format": "geojson",
        "starttime": f"{year}-01-01",
        "endtime": f"{year + 1}-01-01",
        "minmagnitude": MIN_MAGNITUDE,
        "orderby": "time-asc",
        "limit": LIMIT
    }

    response = requests.get(BASE_URL, params=params, timeout=120)
    response.raise_for_status()
    events = response.json()["features"]

    if len(events) >= LIMIT:
        raise RuntimeError(
            f"{year} reached the API limit. Split this year "
            "into smaller date ranges before continuing."
        )

    for event in events:
        props = event["properties"]
        coords = event["geometry"]["coordinates"]

        if props.get("mag") is None or not coords:
            continue

        records.append({
            "event_id": event["id"],
            "time": pd.to_datetime(
                props["time"], unit="ms", utc=True
            ).isoformat(),
            "magnitude": props["mag"],
            "place": props.get("place"),
            "longitude": coords[0],
            "latitude": coords[1],
            "depth_km": coords[2] if len(coords) > 2 else None
        })

    print(f"{year}: downloaded {len(events)} events")
    time.sleep(0.15)

df = pd.DataFrame(records)
df = df.drop_duplicates(subset="event_id")
df = df.sort_values("time")
df.to_csv(OUTPUT, index=False)

print(f"\nSaved {len(df)} unique events to {OUTPUT}")
print(df.head())
