import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ttest_ind

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "fiji_earthquake_sequence.csv"

MAINSHOCK_ID = "us1000gcii"
MAINSHOCK_MAG = 8.2
EXCLUDE_IDS = {MAINSHOCK_ID, "us2000h9e2"}
MAINSHOCK_TIME = pd.Timestamp("2018-08-19T00:19:40.670Z")
DAYS_AFTER = 30
end_time = MAINSHOCK_TIME + pd.Timedelta(days=DAYS_AFTER)

df = pd.read_csv(DATA_FILE)
df["time"] = pd.to_datetime(df["time"], format="mixed", utc=True)
df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")
df["depth_km"] = pd.to_numeric(df["depth_km"], errors="coerce")
df["event_id"] = df["event_id"].astype(str)

sequence = df[
    (~df["event_id"].isin(EXCLUDE_IDS)) &
    (df["time"] > MAINSHOCK_TIME) &
    (df["time"] <= end_time)
].dropna(subset=["magnitude", "depth_km"]).copy()

edges = pd.date_range(start=MAINSHOCK_TIME, periods=DAYS_AFTER + 1, freq="24h")
sequence["day_bin"] = pd.cut(sequence["time"], bins=edges, labels=False, right=False)
daily_counts = sequence["day_bin"].value_counts().reindex(range(DAYS_AFTER), fill_value=0).sort_index()
counts = daily_counts.to_numpy()

mean_cnt = float(counts.mean())
var_cnt = float(counts.var(ddof=1))
vmr = var_cnt / mean_cnt if mean_cnt > 0 else np.nan

if var_cnt > mean_cnt and mean_cnt > 0:
    nb_r = mean_cnt**2 / (var_cnt - mean_cnt)
    nb_p = nb_r / (nb_r + mean_cnt)
else:
    nb_r, nb_p = None, None

p1 = counts[0:5]
p2 = counts[5:15]
p3 = counts[15:30]
tt_res = ttest_ind(p1, p2, equal_var=False)
corr = float(sequence["magnitude"].corr(sequence["depth_km"]))

sequence["depth_group"] = pd.cut(
    sequence["depth_km"],
    bins=[-1, 70, 300, float("inf")],
    labels=["Shallow (<=70 km)", "Intermediate (70-300 km)", "Deep (>300 km)"]
)
depth_counts = sequence["depth_group"].value_counts()

print("SEISMORISK: FINAL RESULTS SUMMARY (DYNAMIC)")
print("=" * 55)
print(f"\n1. DATASET")
print(f"Reference earthquake: {MAINSHOCK_ID}")
print(f"Reference magnitude: {MAINSHOCK_MAG}")
print(f"Analysis period: {DAYS_AFTER} days")
print(f"Events analyzed: {len(sequence)}")

print(f"\n2. DESCRIPTIVE STATISTICS")
print(f"Mean magnitude: {sequence['magnitude'].mean():.4f}")
print(f"Mean depth: {sequence['depth_km'].mean():.4f} km")
print(f"Mean daily count: {mean_cnt:.4f}")
print(f"Daily count variance: {var_cnt:.4f}")
print(f"Variance-to-mean ratio: {vmr:.4f}")

print(f"\n3. ACTIVITY PATTERN")
print(f"Days 1-5 mean: {p1.mean():.1f} events/day")
print(f"Days 6-15 mean: {p2.mean():.1f} events/day")
print(f"Days 16-30 mean: {p3.mean():.1f} events/day")
print(f"Highest daily count: {counts.max()}")
print(f"Lowest daily count: {counts.min()}")

print(f"\n4. COUNT MODEL PARAMETERS")
print(f"Poisson lambda: {mean_cnt:.4f}")
if nb_r is not None:
    print(f"Negative Binomial r: {nb_r:.4f}")
    print(f"Negative Binomial p: {nb_p:.4f}")
else:
    print("Negative Binomial: not fitted (no overdispersion)")

print(f"\n5. ACTIVITY CHANGE TEST (Welch t-test, Days 1-5 vs 6-15)")
print(f"Welch t-statistic: {tt_res.statistic:.4f}")
print(f"Two-sided p-value: {tt_res.pvalue:.4f}")
if tt_res.pvalue < 0.05:
    print("Decision: Reject H0 at alpha = 0.05 level.")
else:
    print("Decision: Fail to reject H0 at alpha = 0.05 level.")

print(f"\n6. MAGNITUDE-DEPTH RELATIONSHIP")
print(f"Pearson correlation: {corr:.4f}")

print(f"\n7. DEPTH GROUPS")
for grp, cnt in depth_counts.items():
    print(f"{grp}: {cnt}")

print("\n8. IMPORTANT METHODOLOGICAL LIMITATIONS")
print("- Only 30 daily observations from a single sequence.")
print("- Daily counts are temporally clustered and overdispersed.")
print("- Heuristic spatial/temporal bounds do not prove every event is a true aftershock.")
print("- Results are descriptive and exploratory, not guaranteed forecasts.")
print("\nSummary completed successfully.")

