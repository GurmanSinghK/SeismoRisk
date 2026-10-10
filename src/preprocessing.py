import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parents[1]

raw_file = root / "data" / "raw" / "earthquakes.csv"
processed_dir = root / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(raw_file)
df["time"] = pd.to_datetime(df["time"], format="mixed", utc=True)

initial_count = len(df)

df = df.dropna(subset=[
    "event_id", "time", "magnitude",
    "longitude", "latitude", "depth_km"
])

df = df.drop_duplicates(subset="event_id")
df = df[df["magnitude"] >= 2.5]
df = df.sort_values("time")

output_file = processed_dir / "earthquakes_clean.csv"
df.to_csv(output_file, index=False)

print("Initial records:", initial_count)
print("Clean records:", len(df))
print("Removed records:", initial_count - len(df))
print("Missing values remaining:", df.isna().sum().sum())
print("Minimum magnitude:", df["magnitude"].min())
print("Maximum magnitude:", df["magnitude"].max())
print("Saved to:", output_file)