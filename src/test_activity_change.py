import pandas as pd
from pathlib import Path
from scipy.stats import ttest_ind

DATA_FILE = Path("data/processed/fiji_earthquake_sequence.csv")

MAINSHOCK_TIME = pd.Timestamp("2018-08-19T00:19:40.670Z")
EXCLUDE_IDS = {"us1000gcii", "us2000h9e2"}
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

df["day_number"] = (
    (df["time"] - MAINSHOCK_TIME).dt.total_seconds() // 86400
).astype(int) + 1

daily_counts = (
    df.groupby("day_number")
    .size()
    .reindex(range(1, DAYS_AFTER + 1), fill_value=0)
)

period_1 = daily_counts.loc[1:5].to_numpy()
period_2 = daily_counts.loc[6:15].to_numpy()

result = ttest_ind(
    period_1,
    period_2,
    equal_var=False
)

print("SEISMORISK: ACTIVITY CHANGE TEST")
print("=" * 40)
print("H0: The mean daily counts are equal.")
print("H1: The mean daily counts differ.")
print("\nDays 1-5:", period_1)
print("Mean:", round(period_1.mean(), 4))
print("Sample standard deviation:", round(period_1.std(ddof=1), 4))
print("\nDays 6-15:", period_2)
print("Mean:", round(period_2.mean(), 4))
print("Sample standard deviation:", round(period_2.std(ddof=1), 4))
print("\nWelch t-statistic:", round(result.statistic, 4))
print("Two-sided p-value:", f"{result.pvalue:.8f}")
print("Significance level: 0.05")

if result.pvalue < 0.05:
    print("Decision: Reject H0.")
    print("Inference: The sample provides evidence of a difference in mean daily counts.")
else:
    print("Decision: Fail to reject H0.")
    print("Inference: The sample does not provide sufficient evidence of a difference in mean daily counts.")

print("\nCaution: Daily counts are skewed, and the first period contains an extreme count.")
print("Treat this as an exploratory result, not a definitive physical conclusion.")