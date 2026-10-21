# SeismoRisk: Probabilistic Earthquake Sequence Analysis & Forecasting

An academic probability and statistics research project analyzing global earthquake catalogues, modeling aftershock clustering, and evaluating probabilistic event count models (Poisson vs. Negative Binomial).

---

## 1. Project Overview & Scope

SeismoRisk investigates whether post-mainshock seismic activity conforms to memoryless Poisson point processes or exhibits overdispersion best characterized by compound Poisson / Negative Binomial processes. 

Key statistical and scientific components:
- **Global Catalogue Ingestion:** 181,680 earthquakes (M ≥ 4.5, 2000–2025) downloaded via USGS FDSNWS API.
- **Space-Time Sequence Delineation:** KD-tree chord-distance clustering (100 km radius, 30-day temporal window) yielding 9,805 candidate mainshocks (M ≥ 5.5) and 48,338 assigned subsequent events.
- **Temporal Activity Decay:** Empirical evaluation of Omori-type power-law decay rates across time bins.
- **Count Distribution Modeling:** Method of Moments (MoM) and Maximum Likelihood Estimation (MLE) for Poisson and Negative Binomial distributions.
- **Leakage-Free Out-of-Sample Evaluation:** Chronological sequence-level train/test split (70% train [2000–2017], 30% test [2017–2025]) evaluating Log-Likelihood and Brier calibration scores.
- **Interactive Dashboard:** Built with Streamlit, enabling exploration of both the global multi-sequence catalog and a high-resolution regional case study (Fiji deep sequence).

> **Academic Disclaimer:** SeismoRisk is a statistical analysis tool, **not an earthquake early-warning system**. Predictions represent fitted probability distributions over specified space-time windows and should never be used as operational safety forecasts.

---

## 2. Directory Structure

```
SeismoRisk-main/
│
├── app.py                             # Interactive Streamlit analytics dashboard
├── requirements.txt                   # Environment dependencies
├── README.md                          # Comprehensive documentation
│
├── data/
│   ├── raw/
│   │   ├── earthquakes.csv            # Fiji exploratory dataset
│   │   └── global_earthquakes_2000_2025.csv  # USGS global catalog (181k rows)
│   └── processed/
│       ├── earthquakes_clean.csv      # Cleaned regional catalog
│       ├── fiji_earthquake_sequence.csv
│       ├── global_sequence_events.csv # 48,338 assigned events with days and distance
│       ├── global_sequence_summary.csv# 9,805 sequences with event counts
│       ├── global_model_evaluation.csv# Out-of-sample test metrics
│       └── global_count_frequencies.csv
│
├── src/
│   ├── download_global_catalog.py     # USGS FDSNWS automated fetcher
│   ├── usgs_live_client.py            # Official USGS live GeoJSON & FDSNWS client
│   ├── aftershock_model.py            # Modified Omori-Utsu time-decay & probability engine
│   ├── build_sequences.py             # 3D spatial KD-tree sequence builder
│   ├── fit_global_models.py           # MLE fitting, out-of-sample Brier scoring
│   ├── fit_probability_model.py       # Fiji baseline Poisson modeling
│   ├── compare_models.py              # Poisson vs NegBin MoM comparison
│   ├── validate_models.py             # Observed vs expected frequency validation
│   ├── evaluate_probabilities.py      # Log-score evaluation on held-out days
│   ├── test_activity_change.py        # Welch two-sample t-test across periods
│   ├── test_predictions.py            # Out-of-sample count prediction test
│   └── final_results_summary.py       # Fully reproducible dynamic metrics report
│
└── tests/
    └── test_sequences.py              # Comprehensive automated unit test suite
```

---


## 3. Data Provenance & Preprocessing

- **Source API:** USGS Earthquake Catalog API (`https://earthquake.usgs.gov/fdsnws/event/1/query`)
- **Query Criteria:** Start: 2000-01-01, End: 2026-01-01, Min Magnitude: 4.5.
- **Cleaning Filters:**
  - Removal of invalid timestamps (`NaT` records).
  - Magnitude validation ($M \ge 4.5$).
  - Complete coordinate triplets (`latitude`, `longitude`, `depth_km`).
  - Event ID deduplication.
- **Sequence Delineation Heuristic:**
  - Candidate mainshocks: $M \ge 5.5$ not preceded by an equal or larger event within 100 km chord distance and 30 days.
  - Event assignment: Events assigned to the most recent eligible mainshock within 100 km and 30 days.

---

## 4. Execution Workflow

### A. Environment Setup (Windows)

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### B. Run Sequence Ingestion & Statistical Modeling

```powershell
# 1. Rebuild global sequences (spatial KD-tree assignment)
python src/build_sequences.py

# 2. Fit Poisson & NegBin MLE models and run out-of-sample validation
python src/fit_global_models.py

# 3. Run regional dynamic analysis summary
python src/final_results_summary.py
```

### C. Run Comprehensive Automated Tests (9 Unit Test Suites)

```powershell
python -c "from tests.test_sequences import test_global_sequence_bounds, test_time_unit_resolution, test_fiji_sequence_cleaning, test_usgs_geojson_parsing, test_haversine_accuracy, test_omori_rate_integration, test_aftershock_probability_monotonicity, test_qualifying_events_filtering, test_historical_replay_no_data_leakage; test_global_sequence_bounds(); test_time_unit_resolution(); test_fiji_sequence_cleaning(); test_usgs_geojson_parsing(); test_haversine_accuracy(); test_omori_rate_integration(); test_aftershock_probability_monotonicity(); test_qualifying_events_filtering(); test_historical_replay_no_data_leakage(); print('All 9 unit test suites passed!')"
```

### D. Launch the Streamlit Dashboard

```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 5. Live Earthquake & Aftershock Probability Module

The dashboard features **"7. Live Earthquake & Aftershock Analysis"**, which supports:
1. **Live USGS Feed:** Direct connection to `earthquake.usgs.gov/earthquakes/feed/v1.0/summary/` with automatic caching (120s TTL) and on-demand refresh.
2. **Candidate Mainshock Selection:** Choose any recently reported earthquake to inspect magnitude, hypocentral depth, coordinates, and origin time.
3. **Scenario Configuration:** Set forecast duration (1–30 days), search radius (30–250 km chord distance), and minimum magnitude threshold $M_{\text{target}}$.
4. **Modified Omori-Utsu Rate Integration:**
   $$\lambda(t) = \frac{K}{(t + c)^p}$$
   Integrates $\lambda(t)$ analytically over the forecast horizon with Reasenberg-Jones magnitude scaling:
   $$\log_{10} K = a + b \cdot (M_{\text{main}} - M_{\text{target}})$$
   Parameters estimated on historical training sequences ($p = 0.9029$, $c = 0.05\text{ days}$, $b = 0.6920$, $a = -1.250$).
5. **Exact Probabilities & Uncertainty:** Calculates $P(N = 0)$, $P(N \ge 1)$, $P(N \ge 2)$, and 95% predictive intervals for both Poisson and Negative Binomial compound models.
6. **Historical Replay & Prospective Backtesting:** Allows simulating a forecast on past earthquakes at an arbitrary elapsed time, comparing expected counts and probabilities against verified ground-truth events with **zero future-data leakage**.

---

## 6. Statistical Methodology & Key Results

### 1. Count Overdispersion
Across 9,805 global sequences, the mean sequence event count is $\bar{x} = 4.930$, with sample variance $s^2 = 330.738$, yielding a Variance-to-Mean Ratio (VMR) of **67.09**. This violates the Poisson equidispersion property ($\lambda = \mathrm{Var}(X)$) and proves substantial empirical overdispersion.

### 2. Model Fitting (MLE)
- **Poisson:** $\hat{\lambda} = 4.810$ (on 70% chronological training sequences)
- **Negative Binomial (MLE):** $\hat{r} = 0.3108, \hat{p} = 0.0607$

### 3. Out-of-Sample Sequence-Level Validation (2017–2025 Test Set)
- **Occurrence Target:** Probability of at least one subsequent event $P(N \ge 1)$
  - Test Observed Prevalence: **66.15%**
  - Empirical Baseline Brier Score: **0.2245**
  - Negative Binomial (MLE) Brier Score: **0.2303**
  - Poisson Baseline Brier Score: **0.3331** (severely overconfident at $P=0.992$)
- **Mean Out-of-Sample Log-Likelihood:**
  - Negative Binomial: **-2.4570**
  - Poisson: **-9.3068** (penalized heavily for zero-probability tail events)


