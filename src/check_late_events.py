
import pandas as pd

catalog = pd.read_csv("data/raw/global_earthquakes_2000_2025.csv")
assigned = pd.read_csv("data/processed/global_sequence_events.csv")

catalog["time"] = pd.to_datetime(catalog["time"], utc=True, errors="coerce")
catalog["magnitude"] = pd.to_numeric(catalog["magnitude"], errors="coerce")

print("=== LATE CATALOGUE EVENTS ===")
print("All catalogue events:", len(catalog))
print("Catalogue events with magnitude >= 4.5:",
      (catalog["magnitude"] >= 4.5).sum())

print("\nAssigned events after day 7:",
      (assigned["days_after_mainshock"] > 7).sum())
print("Assigned events after day 9:",
      (assigned["days_after_mainshock"] > 9).sum())

print("\nLargest assigned day:")
print(assigned["days_after_mainshock"].max())

print("\nCatalogue magnitude distribution:")
print(catalog["magnitude"].describe().round(2))

print("\nCheck complete.")
