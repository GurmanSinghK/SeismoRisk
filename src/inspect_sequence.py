
import pandas as pd
from pathlib import Path

file_path = Path("data/processed/fiji_earthquake_sequence.csv")

df = pd.read_csv(file_path)

df["time"] = pd.to_datetime(df["time"], format="mixed", utc=True)
df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

df = df.dropna(
    subset=["time", "magnitude", "latitude", "longitude"]
)

mainshock_id = "us2000h9e2"
mainshock_time = pd.Timestamp("2018-09-06T15:49:18.710Z")
end_time = mainshock_time + pd.Timedelta(days=30)

sequence = df[
    (df["time"] > mainshock_time) &
    (df["time"] <= end_time) &
    (df["event_id"] != mainshock_id)
].copy()

print("SEISMORISK: SEQUENCE LOCATION INSPECTION")
print("========================================")
print("Events:", len(sequence))

print("\nLATITUDE SUMMARY")
print(sequence["latitude"].describe().to_string())

print("\nLONGITUDE SUMMARY")
print(sequence["longitude"].describe().to_string())

print("\nDEPTH SUMMARY (km)")
if "depth_km" in sequence.columns:
    sequence["depth_km"] = pd.to_numeric(
        sequence["depth_km"], errors="coerce"
    )
    print(sequence["depth_km"].describe().to_string())

print("\nFARTHEST EVENTS IN LATITUDE/LONGITUDE")
print(
    sequence[["event_id", "time", "magnitude", "place",
            "latitude", "longitude"]]
    .head(10)
    .to_string(index=False)
)

print("\nInspection complete.")


print("\nMAINSHOCK RECORD")
print("----------------")
print(
    df.loc[
        df["event_id"].astype(str) == "us2000h9e2",
        ["event_id", "time", "magnitude", "place",
        "latitude", "longitude", "depth_km"]
    ].to_string(index=False)
)


print("\nDEPTHS OF FIRST 15 EVENTS")
print("-------------------------")
print(
    sequence[
        ["event_id", "time", "magnitude", "depth_km", "place"]
    ].head(15).to_string(index=False)
)

print("\nEVENTS WITH DEPTH BELOW 300 KM")
print("------------------------------")
print("Count:", (sequence["depth_km"] < 300).sum())

print("\nEVENTS WITH DEPTH AT LEAST 600 KM")
print("---------------------------------")
print("Count:", (sequence["depth_km"] >= 600).sum())