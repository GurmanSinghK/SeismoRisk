
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

file_path = PROJECT_ROOT / "data" / "processed" / "fiji_earthquake_sequence.csv"

df = pd.read_csv(file_path)

df["time"] = pd.to_datetime(df["time"], utc=True, format="mixed")
df["event_id"] = df["event_id"].astype(str)
for col in ["magnitude", "latitude", "longitude", "depth_km"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["time", "magnitude", "latitude", "longitude", "depth_km"])

MAINSHOCK_ID = "us1000gcii"
EXCLUDE_IDS = {MAINSHOCK_ID, "us2000h9e2"}
mainshock_time = pd.Timestamp("2018-08-19T00:19:40.670Z")
end_time = mainshock_time + pd.Timedelta(days=30)

sequence = df[
    (~df["event_id"].isin(EXCLUDE_IDS)) &
    (df["time"] > mainshock_time) &
    (df["time"] <= end_time)
].copy()

mainshock_rows = df[df["event_id"] == MAINSHOCK_ID]
if not mainshock_rows.empty:
    mainshock = mainshock_rows.iloc[0]
else:
    mainshock = df.loc[df["magnitude"].idxmax()]


lat1 = np.radians(mainshock["latitude"])
lon1 = np.radians(mainshock["longitude"])
lat2 = np.radians(sequence["latitude"].to_numpy())
lon2 = np.radians(sequence["longitude"].to_numpy())

dlat = lat2 - lat1
dlon = lon2 - lon1

a = (
    np.sin(dlat / 2) ** 2
    + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
)

a = np.clip(a, 0, 1)
sequence["distance_km"] = 6371 * 2 * np.arcsin(np.sqrt(a))

print("SEISMORISK: DISTANCE AND DEPTH CHECK")
print("====================================")
print("Events inspected:", len(sequence))
print(
    "Reference event:",
    mainshock["event_id"],
    "| Magnitude:",
    mainshock["magnitude"],
    "| Place:",
    mainshock["place"]
)

print("\nDISTANCE FROM MAINSHOCK (km)")
print("----------------------------")
print(sequence["distance_km"].describe().round(2))

print("\nDEPTH (km)")
print("----------")
print(sequence["depth_km"].describe().round(2))

print("\nMAGNITUDE")
print("---------")
print(sequence["magnitude"].describe().round(2))

print("\nFARTHEST 10 EVENTS")
print("------------------")
print(
    sequence.nlargest(10, "distance_km")[
        ["event_id", "time", "magnitude", "place",
         "distance_km", "depth_km"]
    ].to_string(index=False)
)

print("\nDEEPEST 10 EVENTS")
print("-----------------")
print(
    sequence.nlargest(10, "depth_km")[
        ["event_id", "time", "magnitude", "place",
         "distance_km", "depth_km"]
    ].to_string(index=False)
)

print("\nInspection complete.")
