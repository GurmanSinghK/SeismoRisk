import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import poisson, nbinom

MAINSHOCK_ID = "us1000gcii"
EXCLUDE_IDS = {"us1000gcii", "us2000h9e2"}
MAINSHOCK_TIME = pd.Timestamp("2018-08-19T00:19:40.670Z")
DAYS_AFTER = 30

DATA_FILE = Path("data/processed/fiji_earthquake_sequence.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE)
df["time"] = pd.to_datetime(df["time"], format="mixed", utc=True)
df["event_id"] = df["event_id"].astype(str)

end_time = MAINSHOCK_TIME + pd.Timedelta(days=DAYS_AFTER)

sequence = df[
    (~df["event_id"].isin(EXCLUDE_IDS))
    & (df["time"] > MAINSHOCK_TIME)
    & (df["time"] < end_time)
].copy()

edges = pd.date_range(
    start=MAINSHOCK_TIME,
    periods=DAYS_AFTER + 1,
    freq="24h"
)

sequence["day"] = pd.cut(
    sequence["time"],
    bins=edges,
    labels=False,
    right=False
)

daily_counts = (
    sequence["day"]
    .value_counts()
    .reindex(range(DAYS_AFTER), fill_value=0)
    .sort_index()
)

counts = daily_counts.to_numpy()
mean_count = counts.mean()
variance_count = counts.var(ddof=1)

print("=" * 55)
print("SEISMORISK: POISSON VS NEGATIVE BINOMIAL")
print("=" * 55)
print("Mainshock ID:", MAINSHOCK_ID)
print("Mainshock time:", MAINSHOCK_TIME)
print("Analysis period:", MAINSHOCK_TIME, "to", end_time)
print("Excluded event IDs:", sorted(EXCLUDE_IDS))
print("Events analyzed:", len(sequence))
print("Daily observations:", len(counts))
print("Mean daily count:", round(mean_count, 4))
print("Sample variance:", round(variance_count, 4))
print("Daily counts:", counts.tolist())

if mean_count <= 0:
    raise ValueError("Mean daily count must be positive.")

lam = mean_count

print("\nPOISSON MODEL")
print("Lambda:", round(lam, 4))

probability_values = [0, 1, 2, 3, 5, 10]

for k in probability_values:
    print(f"P(N={k}) = {poisson.pmf(k, lam):.4f}")

print(f"P(N>=1) = {1 - poisson.pmf(0, lam):.4f}")

print("\nNEGATIVE BINOMIAL MODEL")

if variance_count > mean_count:
    r = mean_count**2 / (variance_count - mean_count)
    p = r / (r + mean_count)

    print("r:", round(r, 4))
    print("p:", round(p, 4))

    for k in probability_values:
        print(f"P(N={k}) = {nbinom.pmf(k, r, p):.4f}")

    print(f"P(N>=1) = {1 - nbinom.pmf(0, r, p):.4f}")
else:
    r = None
    p = None
    print("Negative Binomial method-of-moments fit unavailable.")

print("\nDISPERSION CHECK")
print("Variance-to-mean ratio:", round(variance_count / mean_count, 4))

plt.figure(figsize=(10, 5))
plt.bar(np.arange(1, DAYS_AFTER + 1), counts)
plt.xlabel("Day after mainshock")
plt.ylabel("Number of recorded earthquakes")
plt.title("Daily Earthquake Counts: Fiji Sequence")
plt.xticks(np.arange(1, DAYS_AFTER + 1, 2))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "daily_earthquake_counts.png", dpi=300)
plt.close()

max_k = max(15, int(counts.max()))
k_values = np.arange(max_k + 1)
observed_probs = np.array([(counts == k).mean() for k in k_values])
poisson_probs = poisson.pmf(k_values, lam)

plt.figure(figsize=(10, 6))
plt.bar(
    k_values - 0.2,
    observed_probs,
    width=0.4,
    label="Observed frequencies"
)
plt.bar(
    k_values + 0.2,
    poisson_probs,
    width=0.4,
    label="Poisson model"
)

if r is not None:
    plt.plot(
        k_values,
        nbinom.pmf(k_values, r, p),
        marker="o",
        label="Negative Binomial model"
    )

plt.xlabel("Earthquakes per 24-hour period")
plt.ylabel("Probability / relative frequency")
plt.title("Observed Counts vs Probability Models")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "poisson_vs_negative_binomial.png", dpi=300)
plt.close()

print("\nGraphs saved:")
print(OUTPUT_DIR / "daily_earthquake_counts.png")
print(OUTPUT_DIR / "poisson_vs_negative_binomial.png")
print("\nAnalysis complete.")
print("Probabilities are statistical estimates, not guaranteed forecasts.")