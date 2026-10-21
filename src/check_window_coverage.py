
from pathlib import Path
import pandas as pd

catalog_path = Path("data/raw/global_earthquakes_2000_2025.csv")
events_path = Path("data/processed/global_sequence_events.csv")

catalog = pd.read_csv(catalog_path)
events = pd.read_csv(events_path)

catalog["time"] = pd.to_datetime(catalog["time"], utc=True, errors="coerce")
events["event_time"] = pd.to_datetime(events["event_time"], utc=True, errors="coerce")
events["mainshock_time"] = pd.to_datetime(events["mainshock_time"], utc=True, errors="coerce")

print("=== ASSIGNED EVENT WINDOW COVERAGE ===")
print("Assigned events after day 5:", (events["days_after_mainshock"] > 5).sum())
print("Assigned events after day 10:", (events["days_after_mainshock"] > 10).sum())
print("Assigned events after day 20:", (events["days_after_mainshock"] > 20).sum())
print("Latest assigned day:", events["days_after_mainshock"].max())

print("\nCatalog date range:")
print(catalog["time"].min(), "to", catalog["time"].max())

print("\nAssigned event date range:")
print(events["event_time"].min(), "to", events["event_time"].max())

print("\nNote: this is a coverage diagnostic, not proof that unassigned events are aftershocks.")
