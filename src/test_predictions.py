import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import poisson, nbinom

# Anchor to the project root regardless of where the script is run from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH  = PROJECT_ROOT / "data" / "processed" / "fiji_earthquake_sequence.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["time"] = pd.to_datetime(df["time"], format="mixed", utc=True)
df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")
df["event_id"]  = df["event_id"].astype(str)
df = df.dropna(subset=["time", "magnitude"])

# Canonical reference mainshock: us1000gcii, 2018-08-19.
MAINSHOCK_ID  = "us1000gcii"
EXCLUDE_IDS   = {MAINSHOCK_ID, "us2000h9e2"}
mainshock_time = pd.Timestamp("2018-08-19T00:19:40.670Z")
end_time = mainshock_time + pd.Timedelta(days=30)

df = df[
    (~df["event_id"].isin(EXCLUDE_IDS)) &
    (df["time"] > mainshock_time) &
    (df["time"] <= end_time)
].copy().sort_values("time")

# Build 30 daily bins, each one 24 hours, starting from the mainshock.
edges = pd.date_range(start=mainshock_time, periods=31, freq="24h")
df["day_bin"] = pd.cut(df["time"], bins=edges, labels=False, right=False)
counts_series = (
    df["day_bin"]
    .value_counts()
    .reindex(range(30), fill_value=0)
    .sort_index()
)
counts = counts_series.astype(int).to_numpy()

if len(counts) != 30:
    raise ValueError("Expected 30 daily observations.")

# Chronological 20/10 train-test split.
SPLIT = 20
train = counts[:SPLIT]
test  = counts[SPLIT:]

train_mean     = train.mean()
train_variance = train.var(ddof=1)
poisson_lambda = train_mean

if train_variance > train_mean and train_variance > 0:
    r = train_mean ** 2 / (train_variance - train_mean)
    p = r / (r + train_mean)
else:
    r = None
    p = None

poisson_pred = np.full(len(test), poisson_lambda)

# MAE and RMSE computed with numpy (no sklearn dependency).
def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

print("OUT-OF-SAMPLE MODEL EVALUATION")
print("==============================")
print("Reference mainshock:", MAINSHOCK_ID, "—", mainshock_time)
print("Training days:", SPLIT, "  Testing days:", len(test))
print(f"Training mean count:  {train_mean:.3f}")
print(f"Training variance:    {train_variance:.3f}")
print(f"Poisson lambda:       {poisson_lambda:.3f}")
print()
print("POISSON MODEL")
print("-------------")
print(f"MAE:  {mae(test, poisson_pred):.3f}")
print(f"RMSE: {rmse(test, poisson_pred):.3f}")

if r is not None and p is not None:
    nb_mean_pred = r * (1 - p) / p  # NB mean = r*(1-p)/p
    nb_pred = np.full(len(test), nb_mean_pred)
    print()
    print("NEGATIVE BINOMIAL MODEL")
    print("-----------------------")
    print(f"r: {r:.4f}   p: {p:.4f}   NB mean: {nb_mean_pred:.3f}")
    print(f"MAE:  {mae(test, nb_pred):.3f}")
    print(f"RMSE: {rmse(test, nb_pred):.3f}")
else:
    nb_pred = None
    print()
    print("Negative Binomial not fitted: training data has variance ≤ mean.")

results = pd.DataFrame({
    "Day": np.arange(SPLIT + 1, 31),
    "Observed": test,
    "Poisson_prediction": poisson_pred
})
if nb_pred is not None:
    results["NegBinom_prediction"] = nb_pred

results.to_csv(OUTPUT_DIR / "prediction_evaluation.csv", index=False)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(results["Day"], results["Observed"], marker="o", label="Observed counts")
plt.plot(results["Day"], results["Poisson_prediction"], marker="s",
         linestyle="--", label="Poisson mean prediction")
if nb_pred is not None:
    plt.plot(results["Day"], results["NegBinom_prediction"], marker="^",
             linestyle=":", label="NB mean prediction")
plt.xlabel("Day after mainshock")
plt.ylabel("Number of recorded earthquakes")
plt.title("Observed Counts vs Out-of-Sample Model Predictions (Fiji, us1000gcii)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "out_of_sample_predictions.png", dpi=300)
plt.close()

print()
print("Saved:")
print(" ", OUTPUT_DIR / "prediction_evaluation.csv")
print(" ", OUTPUT_DIR / "out_of_sample_predictions.png")
print()
print("Note: 30 daily observations from a single sequence is too small for")
print("reliable out-of-sample evaluation. Treat these metrics as illustrative only.")