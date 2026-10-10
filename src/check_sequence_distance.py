
import pandas as pd
import numpy as np
from pathlib import Path

file_path = Path("data/processed/fiji_earthquake_sequence.csv")

df = pd.read_csv(file_path)

df["time"] = pd.to_datetime(df["time"], utc=True, format="mixed")
for col in ["magnitude", "latitude", "longitude", "depth_km"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["time", "magnitude", "latitude", "longitude", "depth_km"])

mainshock_time = pd.Timestamp("2018-09-06T15:49:18Z")
end_time = mainshock_time + pd.Timedelta(days=30)

sequence = df[
    (df["time"] > mainshock_time) &
    (df["time"] <= end_time)
].copy()

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
