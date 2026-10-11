
from pathlib import Path
import numpy as np
import pandas as pd

EVENTS_FILE = Path("data/processed/global_sequence_events.csv")
SUMMARY_FILE = Path("data/processed/global_sequence_summary.csv")

events = pd.read_csv(EVENTS_FILE)
summary = pd.read_csv(SUMMARY_FILE)

events["days_after_mainshock"] = pd.to_numeric(
    events["days_after_mainshock"], errors="coerce"
)
events["event_magnitude"] = pd.to_numeric(
    events["event_magnitude"], errors="coerce"
)

events = events.dropna(
    subset=["days_after_mainshock", "event_magnitude", "mainshock_id"]
)

events = events[
    (events["days_after_mainshock"] > 0)
    & (events["days_after_mainshock"] <= 30)
]


print("\nDEBUG: elapsed-time distribution")
print(events["days_after_mainshock"].describe(
    percentiles=[0.5, 0.9, 0.95, 0.99]
).round(3))

print(
    "Events after day 10:",
    int((events["days_after_mainshock"] > 10).sum())
)

print(
    "Events after day 20:",
    int((events["days_after_mainshock"] > 20).sum())
)

print(
    "Maximum elapsed days:",
    events["days_after_mainshock"].max()
)


bins = [0, 1, 2, 3, 5, 7, 10, 14, 21, 30]
labels = [
    "0–1 day",
    "1–2 days",
    "2–3 days",
    "3–5 days",
    "5–7 days",
    "7–10 days",
    "10–14 days",
    "14–21 days",
    "21–30 days",
]

events["time_interval"] = pd.cut(
    events["days_after_mainshock"],
    bins=bins,
    labels=labels,
    right=True,
    include_lowest=True,
)

counts = (
    events.groupby("time_interval", observed=False)
    .size()
    .rename("event_count")
    .reset_index()
)

widths = np.diff(bins)
counts["interval_days"] = widths
counts["events_per_day"] = (
    counts["event_count"] / counts["interval_days"]
)

print("Total candidate sequences:", len(summary))
print(
    "Sequences with zero assigned events:",
    int((summary["assigned_event_count"] == 0).sum()),
)
print("\nEvents by elapsed-time interval:")
print(counts.to_string(index=False))

print("\nMagnitude threshold used in the catalog: >= 4.5")
print("Sequence assignment window: 30 days")
print("Spatial assignment radius: 100 km")
print(
    "\nReminder: these are heuristic sequence assignments, "
    "not confirmed aftershocks."
)
