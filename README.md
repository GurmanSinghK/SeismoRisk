# SeismoRisk Dashboard

A local Streamlit dashboard for the SeismoRisk earthquake statistics project.

## Project layout

Keep `app.py` and `requirements.txt` in the project root. Place the existing CSV at:

`data/processed/fiji_earthquake_sequence.csv`

The CSV must contain `event_id`, `time`, `magnitude`, `longitude`, `latitude`, and `depth_km`. A `place` column is optional.

## Windows setup

Open Command Prompt or the VS Code terminal in the folder containing `app.py`, then run:

```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If `py` is unavailable, use `python` instead. Streamlit will print a local URL, usually `http://localhost:8501`. Open it in your browser.

## Reuse your existing data

Copy your existing `data` folder into this project root so the expected CSV path exists. Alternatively, leave the default file absent and upload the CSV from the dashboard sidebar.

## Features

- Overview metrics, daily counts and magnitude histogram
- Poisson and Negative Binomial probability comparison
- Observed-versus-expected count table
- Magnitude-depth scatter plot and Pearson correlation
- Descriptive summaries by depth group
- Exploratory Welch two-sample t-test
- Filters for UTC dates, magnitude, depth and excluded event IDs
- CSV downloads and a JSON summary

## Statistical and scientific cautions

- Model parameters are recalculated from the selected daily counts.
- Model probabilities are fitted estimates, not guaranteed forecasts.
- Selected regional events are not independently verified as true aftershocks.
- A high variance-to-mean ratio indicates overdispersion relative to Poisson, but does not prove either candidate model is adequate.
- The t-test is exploratory: counts may be skewed or time-dependent, and short samples with extreme counts make conclusions uncertain.
- This dashboard is not an operational earthquake prediction system and must not be used for safety decisions.

## Troubleshooting

- **CSV not found:** copy the data folder into the project root or use the upload control.
- **Missing columns:** check the CSV header against the required columns above.
- **Command not found:** activate the virtual environment and install requirements.
- **Port already in use:** run `streamlit run app.py --server.port 8502`.
