import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

MAINSHOCK_ID = "us1000gcii"
EXCLUDE_IDS = {"us1000gcii", "us2000h9e2"}
MAINSHOCK_TIME = pd.Timestamp("2018-08-19T00:19:40.670Z")
DAYS_AFTER = 30

PROJECT_ROOT = Path(__file__).resolve().parents[1]
file_path = PROJECT_ROOT / "data" / "processed" / "fiji_earthquake_sequence.csv"
output_dir = PROJECT_ROOT / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(file_path)
df["time"] = pd.to_datetime(df["time"], format="mixed", utc=True)
df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")
df["depth_km"] = pd.to_numeric(df["depth_km"], errors="coerce")
df = df.dropna(subset=["time", "magnitude", "depth_km"])
df["event_id"] = df["event_id"].astype(str)

end_time = MAINSHOCK_TIME + pd.Timedelta(days=DAYS_AFTER)

df = df[
    (~df["event_id"].isin(EXCLUDE_IDS))
    & (df["time"] > MAINSHOCK_TIME)
    & (df["time"] < end_time)
].copy()

df["hours_after_mainshock"] = (
    df["time"] - MAINSHOCK_TIME
).dt.total_seconds() / 3600

df["days_after_mainshock"] = (
    df["time"] - MAINSHOCK_TIME
).dt.total_seconds() / 86400

df["day_number"] = (
    df["days_after_mainshock"].astype(int) + 1
)

daily_counts = (
    df.groupby("day_number")
    .size()
    .reindex(range(1, DAYS_AFTER + 1), fill_value=0)
)

first_half = daily_counts.loc[1:15]
second_half = daily_counts.loc[16:30]

print("TOTAL EVENTS IN 30 DAYS:", len(df))

print("\nMAGNITUDE SUMMARY")
print(df["magnitude"].describe())

print("\nDEPTH SUMMARY (km)")
print(df["depth_km"].describe())

print("\nEVENTS IN FIRST 24 HOURS:",
      (df["hours_after_mainshock"] < 24).sum())
print("EVENTS IN FIRST 7 DAYS:",
      (df["days_after_mainshock"] < 7).sum())
print("EVENTS IN FIRST 30 DAYS:", len(df))

print("\nEVENTS BY MAGNITUDE")
print(
    pd.cut(
        df["magnitude"],
        bins=[2.5, 3, 4, 5, 6, 10],
        include_lowest=True
    ).value_counts().sort_index()
)

print("\nDAILY ACTIVITY")
print("First 15 days — total:", first_half.sum())
print("First 15 days — mean per day:", round(first_half.mean(), 2))
print("Last 15 days — total:", second_half.sum())
print("Last 15 days — mean per day:", round(second_half.mean(), 2))

print("\nEVENTS WITH MAGNITUDE >= 6")
print(
    df.loc[
        df["magnitude"] >= 6,
        ["event_id", "time", "magnitude", "place", "depth_km"]
    ].to_string(index=False)
)

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

axes[0].hist(df["magnitude"], bins=20, edgecolor="black")
axes[0].set_title("Magnitude Distribution")
axes[0].set_xlabel("Magnitude")
axes[0].set_ylabel("Number of Events")

axes[1].hist(
    df["hours_after_mainshock"],
    bins=30,
    edgecolor="black"
)
axes[1].set_title("Event Timing After Mainshock")
axes[1].set_xlabel("Hours After Mainshock")
axes[1].set_ylabel("Number of Events")

axes[2].bar(daily_counts.index, daily_counts.values)
axes[2].set_title("Daily Earthquake Counts")
axes[2].set_xlabel("Day After Mainshock")
axes[2].set_ylabel("Number of Events")
axes[2].set_xticks(range(0, DAYS_AFTER + 1, 5))

plt.tight_layout()
plt.savefig(output_dir / "fiji_sequence_exploration.png", dpi=300)
plt.close()

print("\nPlot saved to data/processed/fiji_sequence_exploration.png")