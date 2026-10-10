
import pandas as pd
from pathlib import Path

file_path = Path("data/processed/fiji_earthquake_sequence.csv")

df = pd.read_csv(file_path)

df["time"] = pd.to_datetime(df["time"], format="mixed", utc=True)
df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")
df = df.dropna(subset=["time", "magnitude"])

mainshock_time = pd.Timestamp("2018-09-06T15:49:18Z")
end_time = mainshock_time + pd.Timedelta(days=30)

df = df[
    (df["time"] > mainshock_time) &
    (df["time"] <= end_time)
].copy()

first_day_end = mainshock_time + pd.Timedelta(days=1)

first_day = df[
    (df["time"] > mainshock_time) &
    (df["time"] <= first_day_end)
].copy()

print("SEISMORISK: FIRST 24-HOUR PERIOD INSPECTION")
print("===========================================")
print("Mainshock time:", mainshock_time)
print("First 24-hour period ends:", first_day_end)
print("Total events in 30-day window:", len(df))
print("Events in first 24-hour period:", len(first_day))

if first_day.empty:
    print("No events found in the first 24-hour period.")
else:
    print("\nMAGNITUDE SUMMARY")
    print("-----------------")
    print(f"Minimum magnitude: {first_day['magnitude'].min():.2f}")
    print(f"Maximum magnitude: {first_day['magnitude'].max():.2f}")
    print(f"Mean magnitude: {first_day['magnitude'].mean():.2f}")
    print(f"Median magnitude: {first_day['magnitude'].median():.2f}")

    print("\nEVENTS BY MAGNITUDE")
    print("-------------------")
    print(first_day["magnitude"].describe().to_string())

    print("\nFIRST 15 EVENTS BY TIME")
    print("-----------------------")
    print(
        first_day.sort_values("time")[["time", "magnitude"]]
        .head(15)
        .to_string(index=False)
    )

    print("\nLARGEST 10 EVENTS BY MAGNITUDE")
    print("-----------------------------")
    print(
        first_day.sort_values("magnitude", ascending=False)[
            ["time", "magnitude"]
        ]
        .head(10)
        .to_string(index=False)
    )

    duplicate_count = first_day.duplicated().sum()
    print("\nEXACT DUPLICATE ROWS:", duplicate_count)

print("\nInspection complete.")
