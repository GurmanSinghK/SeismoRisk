
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

INPUT = Path("data/raw/global_earthquakes_2000_2025.csv")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_MAGNITUDE = 4.5
MAINSHOCK_MAGNITUDE = 5.5
RADIUS_KM = 100.0
WINDOW_DAYS = 30
EARTH_RADIUS_KM = 6371.0

df = pd.read_csv(INPUT)
df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
for col in ["magnitude", "latitude", "longitude"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(
    subset=["event_id", "time", "magnitude", "latitude", "longitude"]
)
df = df[df["magnitude"] >= MIN_MAGNITUDE]
df = df.drop_duplicates("event_id")
df = df.sort_values("time").reset_index(drop=True)

lat = np.radians(df["latitude"].to_numpy())
lon = np.radians(df["longitude"].to_numpy())
xyz = np.column_stack((
    np.cos(lat) * np.cos(lon),
    np.cos(lat) * np.sin(lon),
    np.sin(lat)
))

radius_chord = 2 * np.sin(
    RADIUS_KM / (2 * EARTH_RADIUS_KM)
)
tree = cKDTree(xyz)

times = df["time"].astype("int64").to_numpy()
mags = df["magnitude"].to_numpy()
candidate_idx = np.flatnonzero(mags >= MAINSHOCK_MAGNITUDE)

# Select candidate mainshocks not preceded by a larger
# candidate within the specified space-time window.
selected = []

for idx in candidate_idx:
    nearby = tree.query_ball_point(xyz[idx], radius_chord)
    earlier = [
        j for j in nearby
        if times[idx] - WINDOW_DAYS * 86400 * 10**9
        <= times[j] < times[idx]
        and mags[j] > mags[idx]
        and mags[j] >= MAINSHOCK_MAGNITUDE
    ]
    if not earlier:
        selected.append(idx)

selected = np.array(selected, dtype=int)
selected_tree = cKDTree(xyz[selected])


print("\nDEBUG: selected mainshock time span")
print("Earliest:", df.loc[selected, "time"].min())
print("Latest:", df.loc[selected, "time"].max())

print("\nDEBUG: candidate mainshock magnitudes")
print(df.loc[selected, "magnitude"].describe().round(3))

print("\nDEBUG: assigned-event time span")
print("Earliest catalog event:", df["time"].min())
print("Latest catalog event:", df["time"].max())
print("Selected mainshocks:", len(selected))


# Assign each event to the most recent eligible selected
# mainshock within the radius and time window.
assignments = []

for idx in range(len(df)):
    nearby_local = selected_tree.query_ball_point(
        xyz[idx], radius_chord
    )
    eligible = [
        selected[j] for j in nearby_local
        if times[selected[j]] < times[idx]
        and times[idx] - times[selected[j]]
        < WINDOW_DAYS * 86400 * 10**9
        and mags[selected[j]] > mags[idx]
    ]

    if not eligible:
        continue

    main_idx = max(eligible, key=lambda j: times[j])
    assignments.append({
        "mainshock_id": df.loc[main_idx, "event_id"],
        "mainshock_time": df.loc[main_idx, "time"],
        "mainshock_magnitude": mags[main_idx],
        "event_id": df.loc[idx, "event_id"],
        "event_time": df.loc[idx, "time"],
        "event_magnitude": mags[idx],
        "days_after_mainshock": (
            times[idx] - times[main_idx]
        ) / (86400 * 10**9),
        "distance_km": EARTH_RADIUS_KM * 2 * np.arcsin(
            np.clip(
                np.linalg.norm(xyz[idx] - xyz[main_idx]) / 2,
                0, 1
            )
        )
    })

events = pd.DataFrame(assignments)

if events.empty:
    raise RuntimeError("No candidate sequence events were identified.")


# Build a summary containing ALL selected mainshocks,
# including sequences with zero assigned events.

mainshock_summary = pd.DataFrame({
    "mainshock_id": df.loc[selected, "event_id"].to_numpy(),
    "mainshock_time": df.loc[selected, "time"].to_numpy(),
    "mainshock_magnitude": mags[selected],
})

if events.empty:
    event_summary = pd.DataFrame(columns=[
        "mainshock_id",
        "assigned_event_count",
        "first_event_days",
        "last_event_days",
        "mean_event_magnitude",
    ])
else:
    event_summary = (
        events.groupby("mainshock_id", as_index=False)
        .agg(
            assigned_event_count=("event_id", "size"),
            first_event_days=("days_after_mainshock", "min"),
            last_event_days=("days_after_mainshock", "max"),
            mean_event_magnitude=("event_magnitude", "mean"),
        )
    )

summary = mainshock_summary.merge(
    event_summary,
    on="mainshock_id",
    how="left",
)

summary["assigned_event_count"] = (
    summary["assigned_event_count"].fillna(0).astype(int)
)

summary.to_csv(
    OUT_DIR / "global_sequence_summary.csv",
    index=False,
)

events.to_csv(
    OUT_DIR / "global_sequence_events.csv",
    index=False,
)

print(f"Catalog events: {len(df):,}")
print(f"Selected candidate mainshocks: {len(selected):,}")
print(f"Assigned subsequent events: {len(events):,}")
print(f"All candidate sequences: {len(summary):,}")
print(
    "Sequences with zero assigned events:",
    int((summary["assigned_event_count"] == 0).sum()),
)
print("\nSequence-size summary, including zeros:")
print(summary["assigned_event_count"].describe().round(2))

print("\nSaved:")
print(OUT_DIR / "global_sequence_summary.csv")
print(OUT_DIR / "global_sequence_events.csv")
