
from pathlib import Path
import pandas as pd

path = Path("data/raw/global_earthquakes_2000_2025.csv")
df = pd.read_csv(path)

df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")
df["depth_km"] = pd.to_numeric(df["depth_km"], errors="coerce")

df = df.dropna(subset=["event_id", "time", "magnitude"])
df = df.drop_duplicates(subset="event_id")
df["year"] = df["time"].dt.year

print("\n--- DATASET SUMMARY ---")
print(f"Valid unique records: {len(df):,}")
print(f"Date range: {df['time'].min()} to {df['time'].max()}")
print(f"Missing depth: {df['depth_km'].isna().sum():,}")
print(f"Duplicate event IDs after cleaning: {df['event_id'].duplicated().sum()}")

print("\n--- EVENTS BY YEAR ---")
print(df.groupby("year").size().to_string())

print("\n--- MAGNITUDE SUMMARY ---")
print(df["magnitude"].describe().round(3))

print("\n--- POTENTIAL MAINSHOCKS ---")
for threshold in [5.5, 6.0, 6.5, 7.0]:
    count = (df["magnitude"] >= threshold).sum()
    print(f"Magnitude >= {threshold}: {count:,}")

print("\n--- EVENTS BY MAGNITUDE BAND ---")
bins = [4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 10]
print(
    pd.cut(df["magnitude"], bins=bins, include_lowest=True)
    .value_counts()
    .sort_index()
    .to_string()
)
