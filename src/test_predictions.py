import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import poisson, nbinom
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = Path("data/processed/fiji_earthquake_sequence.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
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

daily_index = pd.date_range(
    start=mainshock_time.normalize(),
    periods=30,
    freq="D",
    tz="UTC"
)

counts = (
    df.set_index("time")
    .resample("1D")
    .size()
    .reindex(daily_index, fill_value=0)
)

counts = counts.astype(int).reset_index(drop=True)

if len(counts) != 30:
    raise ValueError("Expected 30 daily observations.")

split = 20
train = counts.iloc[:split]
test = counts.iloc[split:]

poisson_lambda = train.mean()

train_mean = train.mean()
train_variance = train.var(ddof=1)

if train_variance > train_mean and train_variance > 0:
    r = train_mean ** 2 / (train_variance - train_mean)
    p = r / (r + train_mean)
else:
    r = None
    p = None

poisson_predictions = np.full(len(test), poisson_lambda)

print("OUT-OF-SAMPLE MODEL EVALUATION")
print("==============================")
print("Training days:", len(train))
print("Testing days:", len(test))
print(f"Training mean count: {train_mean:.3f}")
print(f"Training variance: {train_variance:.3f}")
print(f"Poisson lambda: {poisson_lambda:.3f}")

print("\nPOISSON MODEL")
print("-------------")
print(f"MAE: {mean_absolute_error(test, poisson_predictions):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(test, poisson_predictions)):.3f}")

if r is not None and p is not None:
    nb_predictions = np.full(len(test), r * (1 - p) / p)

    print("\nNEGATIVE BINOMIAL MODEL")
    print("-----------------------")
    print(f"Estimated r: {r:.4f}")
    print(f"Estimated p: {p:.4f}")
    print(f"MAE: {mean_absolute_error(test, nb_predictions):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(test, nb_predictions)):.3f}")
else:
    nb_predictions = None
    print("\nNegative Binomial model was not fitted because")
    print("the training data did not show overdispersion.")

results = pd.DataFrame({
    "Day": np.arange(split + 1, 31),
    "Observed": test.to_numpy(),
    "Poisson prediction": poisson_predictions
})

if nb_predictions is not None:
    results["Negative Binomial prediction"] = nb_predictions

results.to_csv(OUTPUT_DIR / "prediction_evaluation.csv", index=False)

plt.figure(figsize=(12, 6))
plt.plot(results["Day"], results["Observed"], marker="o", label="Observed counts")
plt.plot(results["Day"], results["Poisson prediction"], marker="s", label="Poisson prediction")

if nb_predictions is not None:
    plt.plot(
        results["Day"],
        results["Negative Binomial prediction"],
        marker="^",
        label="Negative Binomial prediction"
    )

plt.xlabel("Day after mainshock")
plt.ylabel("Number of recorded earthquakes")
plt.title("Observed Counts vs Out-of-Sample Model Predictions")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "out_of_sample_predictions.png", dpi=300)
plt.show()

print("\nSaved:")
print("data/processed/prediction_evaluation.csv")
print("data/processed/out_of_sample_predictions.png")
print("\nNote: This is a small, single-sequence evaluation.")
print("It does not establish reliable earthquake forecasting ability.")