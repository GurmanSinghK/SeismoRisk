
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

catalog = pd.read_csv("data/raw/global_earthquakes_2000_2025.csv")
catalog["time"] = pd.to_datetime(catalog["time"], utc=True, errors="coerce")

for col in ["magnitude", "latitude", "longitude"]:
    catalog[col] = pd.to_numeric(catalog[col], errors="coerce")

catalog = catalog.dropna(
    subset=["event_id", "time", "magnitude", "latitude", "longitude"]
)
catalog = catalog[catalog["magnitude"] >= 4.5]
catalog = catalog.drop_duplicates("event_id").sort_values("time").reset_index(drop=True)

summary = pd.read_csv("data/processed/global_sequence_summary.csv")
summary["mainshock_time"] = pd.to_datetime(summary["mainshock_time"], utc=True)

lat = np.radians(catalog["latitude"].to_numpy())
lon = np.radians(catalog["longitude"].to_numpy())
xyz = np.column_stack((
    np.cos(lat) * np.cos(lon),
    np.cos(lat) * np.sin(lon),
    np.sin(lat),
))

radius_chord = 2 * np.sin(100 / (2 * 6371))
tree = cKDTree(xyz)
times = catalog["time"].astype("int64").to_numpy()
mags = catalog["magnitude"].to_numpy()

# Determine time unit (pandas 3.x = microseconds, older = nanoseconds)
_time_unit = catalog["time"].dtype.unit if hasattr(catalog["time"].dtype, "unit") else np.datetime_data(catalog["time"].dtype)[0]
_US_PER_UNIT = {"ns": 1_000, "us": 1, "ms": 1_000_000}.get(_time_unit, 1)
DAY_INT = int(86_400 * 1_000_000 // _US_PER_UNIT)
WINDOW_INT = 30 * DAY_INT
print(f"Time unit: {_time_unit}, DAY_INT={DAY_INT}")

main_ids = set(summary["mainshock_id"])
main_idx = catalog.index[catalog["event_id"].isin(main_ids)].to_numpy()

eligible_late = 0

for m in main_idx:
    nearby = tree.query_ball_point(xyz[m], radius_chord)
    nearby_arr = np.array(nearby)
    dt_days = (times[nearby_arr] - times[m]) / DAY_INT

    mask = (
        (dt_days >= 10)
        & (dt_days < 30)
        & (mags[nearby_arr] < mags[m])
        & (nearby_arr != m)
    )
    eligible_late += int(mask.sum())

print("Eligible event-mainshock pairs from day 10 to day 30:", eligible_late)
