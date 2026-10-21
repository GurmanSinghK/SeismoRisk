
from pathlib import Path
import pandas as pd

FILE = Path("data/processed/global_sequence_events.csv")

events = pd.read_csv(FILE)

print("\n=== GLOBAL SEQUENCE VALIDATION ===")
print(f"Assigned events: {len(events):,}")

# Check required columns
required = [
    "mainshock_id",
    "event_id",
    "days_after_mainshock",
    "distance_km",
]

missing = [col for col in required if col not in events.columns]

if missing:
    print("Missing required columns:", missing)
else:
    print("\nMissing values:")
    print(events[required].isna().sum())

    print("\nDays after mainshock:")
    print(events["days_after_mainshock"].describe().round(3))

    print("\nDistance from mainshock (km):")
    print(events["distance_km"].describe().round(3))

    bad_time = (
        (events["days_after_mainshock"] <= 0)
        | (events["days_after_mainshock"] >= 30)
    )

    bad_distance = (
        (events["distance_km"] < 0)
        | (events["distance_km"] > 100.01)
    )

    print("\nEvents outside the 30-day window:", int(bad_time.sum()))
    print("Events beyond 100 km:", int(bad_distance.sum()))

    print("\nDuplicate event assignments:")
    print(int(events["event_id"].duplicated().sum()))

    print("\nValidation check complete.")
