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

counts = (
    sequence["day"]
    .value_counts()
    .reindex(range(DAYS_AFTER), fill_value=0)
    .sort_index()
    .to_numpy()
)

mean_count = counts.mean()
variance_count = counts.var(ddof=1)
lam = mean_count

if mean_count <= 0:
    raise ValueError("Mean daily count must be positive.")

if variance_count <= mean_count:
    raise ValueError(
        "Negative Binomial method-of-moments fit requires variance > mean."
    )

r = mean_count**2 / (variance_count - mean_count)
p = r / (r + mean_count)

max_k = max(20, int(counts.max()))
k_values = np.arange(max_k + 1)

observed_frequency = np.array([
    np.sum(counts == k) for k in k_values
])

poisson_expected = DAYS_AFTER * poisson.pmf(k_values, lam)
nb_expected = DAYS_AFTER * nbinom.pmf(k_values, r, p)

print("=" * 60)
print("SEISMORISK: MODEL VALIDATION")
print("=" * 60)
print("Daily observations:", len(counts))
print("Events analyzed:", len(sequence))
print("Mean:", round(mean_count, 4))
print("Sample variance:", round(variance_count, 4))
print("Poisson lambda:", round(lam, 4))
print("Negative Binomial r:", round(r, 4))
print("Negative Binomial p:", round(p, 4))

print("\nOBSERVED VS EXPECTED FREQUENCIES")
print("Count | Observed days | Poisson expected | NB expected")

for k in range(min(15, max_k + 1)):
    print(
        f"{k:5d} | "
        f"{observed_frequency[k]:13d} | "
        f"{poisson_expected[k]:16.4f} | "
        f"{nb_expected[k]:11.4f}"
    )

print("\nOBSERVED ZERO-COUNT DAYS:", int(np.sum(counts == 0)))
print(
    "Poisson expected zero-count days:",
    round(DAYS_AFTER * poisson.pmf(0, lam), 4)
)
print(
    "Negative Binomial expected zero-count days:",
    round(DAYS_AFTER * nbinom.pmf(0, r, p), 4)
)

print("\nOBSERVED COUNT RANGE:", int(counts.min()), "to", int(counts.max()))

plt.figure(figsize=(11, 6))
plt.bar(
    k_values - 0.25,
    observed_frequency,
    width=0.25,
    label="Observed days"
)
plt.bar(
    k_values,
    poisson_expected,
    width=0.25,
    label="Poisson expected days"
)
plt.bar(
    k_values + 0.25,
    nb_expected,
    width=0.25,
    label="Negative Binomial expected days"
)

plt.xlabel("Earthquakes per 24-hour period")
plt.ylabel("Number of days")
plt.title("Observed and Expected Daily Count Frequencies")
plt.legend()
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "model_validation_frequencies.png",
    dpi=300
)
plt.close()

print("\nGraph saved:")
print(OUTPUT_DIR / "model_validation_frequencies.png")

print(
    "\nNote: This is a descriptive comparison using 30 daily observations. "
    "It does not establish that either model is statistically adequate "
    "or that the selected events are all genuine aftershocks."
)