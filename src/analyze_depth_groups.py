
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_FILE = Path("data/processed/fiji_earthquake_sequence.csv")
OUTPUT_DIR = Path("data/processed")

MAINSHOCK_TIME = pd.Timestamp("2018-08-19T00:19:40.670Z")
EXCLUDE_IDS = {"us1000gcii", "us2000h9e2"}
DAYS_AFTER = 30

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE)
df["time"] = pd.to_datetime(df["time"], format="mixed", utc=True)
df["event_id"] = df["event_id"].astype(str)
df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")
df["depth_km"] = pd.to_numeric(df["depth_km"], errors="coerce")

end_time = MAINSHOCK_TIME + pd.Timedelta(days=DAYS_AFTER)

df = df[
    (~df["event_id"].isin(EXCLUDE_IDS))
    & (df["time"] > MAINSHOCK_TIME)
    & (df["time"] < end_time)
].dropna(subset=["magnitude", "depth_km"]).copy()

df["Depth Group"] = pd.cut(
    df["depth_km"],
    bins=[-1, 70, 300, float("inf")],
    labels=["Shallow (<=70 km)", "Intermediate (70-300 km)", "Deep (>300 km)"]
)

summary = df.groupby("Depth Group", observed=False)["magnitude"].agg(
    ["count", "mean", "median", "std", "min", "max"]
).round(3)

print("SEISMORISK: MAGNITUDE BY DEPTH GROUP")
print("=" * 45)
print(summary.to_string())

print("\nNote: These are descriptive comparisons, not proof of causation.")

plt.figure(figsize=(9, 6))
df.boxplot(column="magnitude", by="Depth Group", grid=False)
plt.title("Earthquake Magnitude by Depth Group")
plt.suptitle("")
plt.xlabel("Depth Group")
plt.ylabel("Magnitude")
plt.xticks(rotation=15)
plt.tight_layout()

output_file = OUTPUT_DIR / "magnitude_by_depth_group.png"
plt.savefig(output_file, dpi=300)
plt.close()

print("\nGraph saved to:", output_file)
