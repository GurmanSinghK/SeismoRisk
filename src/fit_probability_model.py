
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from pathlib import Path

file_path = Path("data/processed/fiji_earthquake_sequence.csv")

if not file_path.exists():
    raise FileNotFoundError(f"Dataset not found: {file_path}")

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

df = df.sort_values("time")

if df.empty:
    raise ValueError(
        "No events found in the 30-day period after the mainshock. "
        "Check the dataset timestamps and mainshock time."
    )

duration_days = 30

daily_index = pd.date_range(
    start=mainshock_time.normalize(),
    periods=duration_days + 1,
    freq="D",
    tz="UTC"
)

daily_counts = (
    df.set_index("time")
    .resample("1D")
    .size()
    .reindex(daily_index, fill_value=0)
)

# Exclude the partial first calendar day and use the next 30 full days.
counts = daily_counts.iloc[1:duration_days + 1]

mean_daily = counts.mean()
variance_daily = counts.var(ddof=1)

print("POISSON BASELINE MODEL")
print("----------------------")
print("Events in selected 30-day window:", len(df))
print("Events in 30 complete calendar days:", int(counts.sum()))
print("Observation period (days):", len(counts))
print(f"Mean events per day (lambda): {mean_daily:.3f}")
print(f"Variance of daily counts: {variance_daily:.3f}")

print("\nBASELINE PROBABILITIES")

for k in [0, 1, 2, 3, 5, 10]:
    probability = poisson.pmf(k, mean_daily)
    print(f"P(N = {k}) = {probability:.4f}")

prob_at_least_one = 1 - poisson.pmf(0, mean_daily)

print(
    "\nBaseline probability of at least one event in a day:",
    f"{prob_at_least_one:.4f}"
)

print("\nMODEL CHECK")

if mean_daily > 0:
    ratio = variance_daily / mean_daily
    print(f"Variance-to-mean ratio: {ratio:.3f}")

    if ratio > 1.5:
        print("Daily counts show overdispersion relative to Poisson.")
    elif ratio < 0.67:
        print("Daily counts show underdispersion relative to Poisson.")
    else:
        print("Daily-count variance is reasonably close to the mean.")
else:
    ratio = np.nan
    print("Mean event count is zero; the Poisson rate cannot be meaningfully estimated.")

print(
    "\nImportant: these are baseline estimates for a historical "
    "earthquake sequence, not validated forecasts of future earthquakes."
)

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

days = np.arange(1, duration_days + 1)

# Graph 1: Observed daily counts and Poisson expected mean
plt.figure(figsize=(12, 6))
plt.bar(days, counts.values, label="Observed daily counts")
plt.axhline(
    mean_daily,
    linestyle="--",
    linewidth=2,
    label=f"Poisson expected mean = {mean_daily:.2f}"
)
plt.xlabel("Day after mainshock (full calendar days)")
plt.ylabel("Number of events")
plt.title("Observed Daily Earthquake Counts vs Poisson Mean")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "daily_counts_comparison.png", dpi=300)
plt.show()

# Graph 2: Poisson probability mass function
max_k = max(15, int(counts.max()) + 1)
k_values = np.arange(0, max_k + 1)
probabilities = poisson.pmf(k_values, mean_daily)

plt.figure(figsize=(10, 6))
plt.bar(k_values, probabilities)
plt.xlabel("Number of events in one day")
plt.ylabel("Poisson probability")
plt.title("Poisson Baseline Probability Distribution")
plt.tight_layout()
plt.savefig(output_dir / "poisson_probability.png", dpi=300)
plt.show()

print("\nGraphs saved:")
print(output_dir / "daily_counts_comparison.png")
print(output_dir / "poisson_probability.png")
