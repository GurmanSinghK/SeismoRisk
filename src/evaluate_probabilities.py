import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import poisson, nbinom

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "fiji_earthquake_sequence.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["time"] = pd.to_datetime(df["time"], format="mixed", utc=True)
df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")
df["event_id"] = df["event_id"].astype(str)
df = df.dropna(subset=["time", "magnitude"])

MAINSHOCK_ID = "us1000gcii"
EXCLUDE_IDS = {MAINSHOCK_ID, "us2000h9e2"}
mainshock_time = pd.Timestamp("2018-08-19T00:19:40.670Z")
end_time = mainshock_time + pd.Timedelta(days=30)

df = df[
    (~df["event_id"].isin(EXCLUDE_IDS)) &
    (df["time"] > mainshock_time) &
    (df["time"] <= end_time)
].copy().sort_values("time")

edges = pd.date_range(start=mainshock_time, periods=31, freq="24h")
df["day_bin"] = pd.cut(df["time"], bins=edges, labels=False, right=False)
counts = (
    df["day_bin"]
    .value_counts()
    .reindex(range(30), fill_value=0)
    .sort_index()
    .astype(int)
    .to_numpy()
)

train = counts[:20]
test = counts[20:]

mean_train = train.mean()
variance_train = train.var(ddof=1)

if mean_train <= 0:
    raise ValueError("Training mean must be positive.")

poisson_lambda = mean_train
poisson_log_scores = poisson.logpmf(test, poisson_lambda)
poisson_total_log_score = poisson_log_scores.sum()

print("PREDICTIVE PROBABILITY EVALUATION")
print("=================================")
print("Reference mainshock:", MAINSHOCK_ID, "—", mainshock_time)
print("Training days:", len(train))
print("Testing days:", len(test))
print("Training mean:", round(mean_train, 4))
print("Training variance:", round(variance_train, 4))

print("\nPOISSON MODEL")
print("-------------")
print(f"Lambda: {poisson_lambda:.4f}")
print(f"Total test log-likelihood: {poisson_total_log_score:.4f}")
print(f"Mean test log score: {poisson_log_scores.mean():.4f}")

if variance_train > mean_train:
    r = mean_train ** 2 / (variance_train - mean_train)
    p = r / (r + mean_train)

    nb_log_scores = nbinom.logpmf(test, r, p)
    nb_total_log_score = nb_log_scores.sum()

    print("\nNEGATIVE BINOMIAL MODEL")
    print("-----------------------")
    print(f"r: {r:.4f}")
    print(f"p: {p:.4f}")
    print(f"Total test log-likelihood: {nb_total_log_score:.4f}")
    print(f"Mean test log score: {nb_log_scores.mean():.4f}")

    if nb_total_log_score > poisson_total_log_score:
        print("\nThe Negative Binomial model assigned")
        print("higher probability to the observed test counts.")
    elif nb_total_log_score < poisson_total_log_score:
        print("\nThe Poisson model assigned higher")
        print("probability to the observed test counts.")
    else:
        print("\nBoth models achieved equal total log scores.")

    comparison = pd.DataFrame({
        "Day": np.arange(21, 31),
        "Observed count": test,
        "Poisson probability": poisson.pmf(test, poisson_lambda),
        "Negative Binomial probability": nbinom.pmf(test, r, p),
        "Poisson log score": poisson_log_scores,
        "Negative Binomial log score": nb_log_scores
    })

    comparison.to_csv(
        OUTPUT_DIR / "probability_evaluation.csv",
        index=False
    )

    plt.figure(figsize=(12, 6))
    x = np.arange(len(test))
    width = 0.35

    plt.bar(
        x - width / 2,
        poisson_log_scores,
        width,
        label="Poisson"
    )
    plt.bar(
        x + width / 2,
        nb_log_scores,
        width,
        label="Negative Binomial"
    )

    plt.xticks(x, np.arange(21, 31))
    plt.xlabel("Day after mainshock")
    plt.ylabel("Log probability of observed count")
    plt.title("Predictive Log Scores on Test Days (Fiji Sequence)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "predictive_log_scores.png",
        dpi=300
    )
    plt.close()

    print("\nSaved:")
    print(" ", OUTPUT_DIR / "probability_evaluation.csv")
    print(" ", OUTPUT_DIR / "predictive_log_scores.png")

else:
    print("\nNegative Binomial model was not fitted because")
    print("training counts do not show overdispersion.")