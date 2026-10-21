
from pathlib import Path
import pandas as pd

catalog = pd.read_csv("data/raw/global_earthquakes_2000_2025.csv")
events = pd.read_csv("data/processed/global_sequence_events.csv")

catalog["time"] = pd.to_datetime(catalog["time"], utc=True, errors="coerce")
events["mainshock_time"] = pd.to_datetime(
    events["mainshock_time"], utc=True, errors="coerce"
)
events["event_time"] = pd.to_datetime(
    events["event_time"], utc=True, errors="coerce"
)

# Check how many catalog events occur after day 10
# relative to each selected mainshock.
summary = pd.read_csv("data/processed/global_sequence_summary.csv")
summary["mainshock_time"] = pd.to_datetime(
    summary["mainshock_time"], utc=True, errors="coerce"
)

catalog = catalog.dropna(
    subset=["time", "latitude", "longitude", "magnitude"]
)

catalog["magnitude"] = pd.to_numeric(catalog["magnitude"], errors="coerce")
catalog = catalog[catalog["magnitude"] >= 4.5]

print("=== DIAGNOSTIC ===")
print("Catalog events:", len(catalog))
print("Assigned events:", len(events))
print("Selected mainshocks:", len(summary))
print("Latest assigned day:", events["days_after_mainshock"].max())

print("\nAssigned events by time interval:")
bins = [0, 1, 3, 5, 7, 10, 15, 20, 30]
print(pd.cut(
    events["days_after_mainshock"],
    bins=bins,
    right=False
).value_counts().sort_index())

print("\nMainshock date range:")
print(summary["mainshock_time"].min(), "to", summary["mainshock_time"].max())

print("\nDiagnostic complete.")
