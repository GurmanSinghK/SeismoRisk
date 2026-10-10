import pandas as pd
from pathlib import Path

DATA_FILE = Path("data/processed/fiji_earthquake_sequence.csv")

MAINSHOCK_ID = "us1000gcii"
EXCLUDE_IDS = {"us1000gcii", "us2000h9e2"}
MAINSHOCK_TIME = pd.Timestamp("2018-08-19T00:19:40.670Z")
DAYS_AFTER = 30

df = pd.read_csv(DATA_FILE)
df["time"] = pd.to_datetime(df["time"], format="mixed", utc=True)
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

df["day_number"] = (
    df["hours_after_mainshock"] // 24
).astype(int) + 1

counts = (
    df.groupby("day_number")
    .size()
    .reindex(range(1, DAYS_AFTER + 1), fill_value=0)
)

print("DAILY COUNT CHECK")
print("=================")
print("Total events:", len(df))
print("Number of days:", len(counts))
print("Sum of daily counts:", counts.sum())
print("Mean:", round(counts.mean(), 4))
print("Sample variance:", round(counts.var(ddof=1), 4))
print("\nCount for each day:")
print(counts.to_string())
print("\nDays with zero events:", int((counts == 0).sum()))
print("Days with more than 20 events:", int((counts > 20).sum()))