
import pandas as pd

df = pd.read_csv("data/processed/earthquakes_clean.csv")

df["time"] = pd.to_datetime(
    df["time"],
    format="mixed",
    utc=True,
    errors="coerce"
)

df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")

df = df.dropna(subset=["time", "magnitude"])
df = df.sort_values("time")

mainshocks = df[df["magnitude"] >= 7.0].copy()
mainshocks = mainshocks.sort_values("magnitude", ascending=False)

print("Total earthquakes with magnitude >= 7:", len(mainshocks))
print("\nTOP 15 CANDIDATE MAINSHOCKS\n")

if mainshocks.empty:
    print("No earthquakes with magnitude >= 7 were found.")
else:
    print(
        mainshocks[
            [
                "event_id",
                "time",
                "magnitude",
                "place",
                "latitude",
                "longitude"
            ]
        ].head(15).to_string(index=False)
    )
