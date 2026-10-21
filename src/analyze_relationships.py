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

correlation = df["magnitude"].corr(df["depth_km"], method="pearson")

print("SEISMORISK: MAGNITUDE-DEPTH RELATIONSHIP")
print("=" * 45)
print("Events analyzed:", len(df))
print("Mean magnitude:", round(df["magnitude"].mean(), 4))
print("Mean depth (km):", round(df["depth_km"].mean(), 4))
print("Pearson correlation:", round(correlation, 4))

if correlation > 0:
    print("Interpretation: Positive linear association in this sample.")
elif correlation < 0:
    print("Interpretation: Negative linear association in this sample.")
else:
    print("Interpretation: No linear association in this sample.")

print("Correlation does not establish causation.")

plt.figure(figsize=(8, 6))
plt.scatter(df["depth_km"], df["magnitude"], alpha=0.7)
plt.xlabel("Depth (km)")
plt.ylabel("Magnitude")
plt.title("Earthquake Magnitude vs Depth")
plt.tight_layout()

output_file = OUTPUT_DIR / "magnitude_depth_relationship.png"
plt.savefig(output_file, dpi=300)
plt.close()

print("\nGraph saved to:", output_file)