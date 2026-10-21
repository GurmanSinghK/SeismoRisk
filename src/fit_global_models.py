import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import poisson, nbinom
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_FILE = PROJECT_ROOT / "data" / "processed" / "global_sequence_summary.csv"
EVENTS_FILE  = PROJECT_ROOT / "data" / "processed" / "global_sequence_events.csv"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("SEISMORISK: GLOBAL STATISTICAL COUNT MODELING & EVALUATION")
print("=" * 65)

summary = pd.read_csv(SUMMARY_FILE)
events = pd.read_csv(EVENTS_FILE)

summary["mainshock_time"] = pd.to_datetime(summary["mainshock_time"], utc=True)
summary = summary.sort_values("mainshock_time").reset_index(drop=True)

N_total = len(summary)
counts = summary["assigned_event_count"].to_numpy(dtype=float)

mean_c = np.mean(counts)
var_c  = np.var(counts, ddof=1)
vmr    = var_c / mean_c if mean_c > 0 else np.nan

print(f"Total candidate sequences: {N_total:,}")
print(f"Mean event count per sequence: {mean_c:.3f}")
print(f"Variance of event count:       {var_c:.3f}")
print(f"Variance-to-mean ratio (VMR):  {vmr:.3f}")
print(f"Fraction with zero events:     {(counts == 0).mean():.3%}")
print(f"Fraction with >= 1 event:      {(counts >= 1).mean():.3%}")

# 1. Out-of-sample Chronological Train/Test Split on Sequences (70% Train, 30% Test)
# Leakage prevention: split sequences chronologically by mainshock_time
split_idx = int(0.70 * N_total)
train_df = summary.iloc[:split_idx]
test_df  = summary.iloc[split_idx:]

y_train = train_df["assigned_event_count"].to_numpy(dtype=int)
y_test  = test_df["assigned_event_count"].to_numpy(dtype=int)

train_mean = np.mean(y_train)
train_var  = np.var(y_train, ddof=1)

print("\n--- CHRONOLOGICAL SEQUENCE SPLIT ---")
print(f"Training sequences: {len(y_train):,} ({train_df['mainshock_time'].min().date()} to {train_df['mainshock_time'].max().date()})")
print(f"Testing sequences:  {len(y_test):,} ({test_df['mainshock_time'].min().date()} to {test_df['mainshock_time'].max().date()})")
print(f"Train mean: {train_mean:.3f}, Train var: {train_var:.3f}, VMR: {train_var/train_mean:.3f}")

# Model 1: Baseline Poisson
lam_hat = train_mean

# Model 2: Negative Binomial (Method of Moments)
if train_var > train_mean:
    r_mom = train_mean**2 / (train_var - train_mean)
    p_mom = r_mom / (r_mom + train_mean)
else:
    r_mom, p_mom = 1.0, 0.5

# Model 3: Negative Binomial (Maximum Likelihood Estimation)
def neg_log_lik_nb(params):
    r, p = params
    if r <= 1e-5 or p <= 1e-5 or p >= 1.0 - 1e-5:
        return 1e12
    ll = np.sum(nbinom.logpmf(y_train, r, p))
    return -ll if np.isfinite(ll) else 1e12

opt_res = minimize(neg_log_lik_nb, [r_mom, p_mom], bounds=[(1e-4, 50.0), (1e-4, 0.9999)])
if opt_res.success:
    r_mle, p_mle = opt_res.x
else:
    r_mle, p_mle = r_mom, p_mom

print("\n--- FITTED PARAMETERS ON TRAINING SET ---")
print(f"Poisson: lambda = {lam_hat:.4f}")
print(f"Negative Binomial (MoM): r = {r_mom:.4f}, p = {p_mom:.4f}")
print(f"Negative Binomial (MLE): r = {r_mle:.4f}, p = {p_mle:.4f}")

# 2. Out-of-sample Evaluation on Test Set
# Test Metrics: Log-Likelihood, Brier Score for Binary P(N >= 1), MAE on counts
pois_prob_event = 1.0 - np.exp(-lam_hat)
nb_prob_event   = 1.0 - nbinom.pmf(0, r_mle, p_mle)

y_test_binary = (y_test >= 1).astype(int)

# Brier scores (mean squared error of predicted occurrence probability)
brier_pois = float(np.mean((pois_prob_event - y_test_binary)**2))
brier_nb   = float(np.mean((nb_prob_event - y_test_binary)**2))
brier_base = float(np.mean((np.mean(y_train >= 1) - y_test_binary)**2))

# Test Log-likelihoods
ll_pois = float(np.mean(poisson.logpmf(y_test, lam_hat)))
ll_nb   = float(np.mean(nbinom.logpmf(y_test, r_mle, p_mle)))

# Count Prediction MAE
mae_pois = float(np.mean(np.abs(y_test - lam_hat)))
mae_nb   = float(np.mean(np.abs(y_test - (r_mle * (1 - p_mle) / p_mle))))

print("\n--- OUT-OF-SAMPLE TEST SET METRICS ---")
print(f"Target: Predicting occurrence of >=1 subsequent event (Test prevalence = {y_test_binary.mean():.3%})")
print(f"Empirical Frequency Baseline Brier Score: {brier_base:.4f}")
print(f"Poisson Model (P={pois_prob_event:.3f}):     Brier = {brier_pois:.4f}, Mean Log-lik = {ll_pois:.4f}, MAE = {mae_pois:.2f}")
print(f"Negative Binomial (P={nb_prob_event:.3f}):  Brier = {brier_nb:.4f}, Mean Log-lik = {ll_nb:.4f}, MAE = {mae_nb:.2f}")

eval_summary = pd.DataFrame([
    {"Model": "Poisson Baseline", "Fitted Params": f"lambda={lam_hat:.3f}", "P(N>=1)": pois_prob_event, "Test Brier Score": brier_pois, "Test Mean Log-Lik": ll_pois, "Count MAE": mae_pois},
    {"Model": "Negative Binomial (MLE)", "Fitted Params": f"r={r_mle:.3f}, p={p_mle:.3f}", "P(N>=1)": nb_prob_event, "Test Brier Score": brier_nb, "Test Mean Log-Lik": ll_nb, "Count MAE": mae_nb}
])
eval_summary.to_csv(OUTPUT_DIR / "global_model_evaluation.csv", index=False)

# 3. Observed vs Expected Frequencies on Full Global Dataset
max_plot_k = 15
k_vals = np.arange(0, max_plot_k + 1)
obs_counts = np.array([(counts == k).sum() for k in k_vals])
pois_exp = poisson.pmf(k_vals, mean_c) * N_total
nb_exp   = nbinom.pmf(k_vals, r_mle, p_mle) * N_total

freq_table = pd.DataFrame({
    "Events_k": k_vals,
    "Observed_Sequences": obs_counts,
    "Poisson_Expected": np.round(pois_exp, 1),
    "NegBin_Expected": np.round(nb_exp, 1)
})
freq_table.to_csv(OUTPUT_DIR / "global_count_frequencies.csv", index=False)

# Plot: Observed vs Expected Distribution
plt.figure(figsize=(11, 5.5))
plt.bar(k_vals - 0.2, obs_counts, width=0.4, label="Observed Sequences", color="#22d3ee", alpha=0.85)
plt.plot(k_vals, pois_exp, marker="o", color="#f43f5e", label="Poisson Expected", linewidth=2)
plt.plot(k_vals, nb_exp, marker="s", color="#fbbf24", label="Negative Binomial (MLE)", linewidth=2)
plt.xlabel("Number of assigned events in 30-day window (k)")
plt.ylabel("Number of Sequences")
plt.title("Global Earthquake Sequences: Observed vs Modeled Event Counts")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "global_observed_vs_modeled_counts.png", dpi=300)
plt.close()

# 4. Temporal Rate Analysis (Omori decay empirical fit)
bins = np.array([0, 1, 2, 3, 5, 7, 10, 14, 21, 30])
widths = np.diff(bins)
bin_centers = (bins[:-1] + bins[1:]) / 2.0
decay_series = pd.cut(events["days_after_mainshock"], bins=bins).value_counts().sort_index()
event_rates = decay_series.values / (widths * N_total)

plt.figure(figsize=(9, 5))
plt.plot(bin_centers, event_rates, marker="o", color="#38bdf8", linewidth=2.5, label="Empirical aftershock rate")
plt.xlabel("Days after mainshock")
plt.ylabel("Average events / sequence / day")
plt.title("Empirical Temporal Activity Decay (Omori-Type Pattern)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "global_temporal_decay.png", dpi=300)
plt.close()

print("\nArtifacts and visualizations saved:")
print(f"  {OUTPUT_DIR / 'global_model_evaluation.csv'}")
print(f"  {OUTPUT_DIR / 'global_count_frequencies.csv'}")
print(f"  {OUTPUT_DIR / 'global_observed_vs_modeled_counts.png'}")
print(f"  {OUTPUT_DIR / 'global_temporal_decay.png'}")
print("\nGlobal statistical analysis successfully completed.")
