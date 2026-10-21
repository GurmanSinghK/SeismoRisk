import requests
import pandas as pd
from pathlib import Path

MAINSHOCK_ID = "us1000gcii"
MAINSHOCK_TIME = "2018-08-19T00:19:40"
LATITUDE = -18.113
LONGITUDE = -178.153
RADIUS_DEGREES = 3
DAYS_AFTER = 30
MIN_MAGNITUDE = 2.5

start_time = pd.Timestamp(MAINSHOCK_TIME, tz="UTC")
end_time = start_time + pd.Timedelta(days=DAYS_AFTER)

params = {
    "format": "geojson",
    "starttime": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
    "endtime": end_time.strftime("%Y-%m-%dT%H:%M:%S"),
    "minmagnitude": MIN_MAGNITUDE,
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "maxradius": RADIUS_DEGREES,
    "orderby": "time-asc",
    "limit": 20000
}

url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

response = requests.get(url, params=params, timeout=60)
response.raise_for_status()

features = response.json()["features"]
records = []

for feature in features:
    props = feature["properties"]
    coords = feature["geometry"]["coordinates"]

    records.append({
        "event_id": feature["id"],
        "time": pd.to_datetime(props["time"], unit="ms", utc=True),
        "magnitude": props["mag"],
        "place": props["place"],
        "longitude": coords[0],
        "latitude": coords[1],
        "depth_km": coords[2]
    })

df = pd.DataFrame(records)

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "fiji_earthquake_sequence.csv"
df.to_csv(output_file, index=False)

print("Mainshock ID:", MAINSHOCK_ID)
print("Mainshock time:", start_time)
print("Search period:", start_time, "to", end_time)
print("Events downloaded:", len(df))
print("Mainshock present:", MAINSHOCK_ID in df["event_id"].astype(str).values)
print("Output file:", output_file)

if not df.empty:
    print("\nFirst 10 events:")
    print(df.head(10).to_string(index=False))

    print("\nDepth summary (km):")
    print(df["depth_km"].describe())

    print("\nMagnitude summary:")
    print(df["magnitude"].describe())
