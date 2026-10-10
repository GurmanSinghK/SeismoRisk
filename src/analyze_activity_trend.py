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

end_time = MAINSHOCK_TIME + pd.Timedelta(days=DAYS_AFTER)

df = df[
    (~df["event_id"].isin(EXCLUDE_IDS))
    & (df["time"] > MAINSHOCK_TIME)
    & (df["time"] < end_time)
].copy()

df["day_number"] = (
    (df["time"] - MAINSHOCK_TIME).dt.total_seconds() // 86400
).astype(int) + 1

daily_counts = (
    df.groupby("day_number")
    .size()
    .reindex(range(1, DAYS_AFTER + 1), fill_value=0)
)

first_5 = daily_counts.loc[1:5]
days_6_to_15 = daily_counts.loc[6:15]
days_16_to_30 = daily_counts.loc[16:30]

print("SEISMORISK: ACTIVITY TREND ANALYSIS")
print("=" * 45)
print("Total events:", int(daily_counts.sum()))

print("\nPERIOD COMPARISON")
print("Days 1-5: total =", int(first_5.sum()),
      ", daily mean =", round(first_5.mean(), 2))
print("Days 6-15: total =", int(days_6_to_15.sum()),
      ", daily mean =", round(days_6_to_15.mean(), 2))
print("Days 16-30: total =", int(days_16_to_30.sum()),
      ", daily mean =", round(days_16_to_30.mean(), 2))

print("\nDAILY COUNTS")
print(daily_counts.to_string())

plt.figure(figsize=(11, 6))
plt.plot(
    daily_counts.index,
    daily_counts.values,
    marker="o",
    label="Observed daily count"
)
plt.axhline(
    daily_counts.mean(),
    linestyle="--",
    label="Overall daily mean"
)
plt.xlabel("Day after mainshock")
plt.ylabel("Earthquake count")
plt.title("Daily Earthquake Activity After Mainshock")
plt.xticks(range(1, DAYS_AFTER + 1, 2))
plt.legend()
plt.tight_layout()

output_file = OUTPUT_DIR / "activity_trend.png"
plt.savefig(output_file, dpi=300)
plt.close()

print("\nGraph saved to:", output_file)