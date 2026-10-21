import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

root = Path(__file__).resolve().parents[1]
df = pd.read_csv(root / "data" / "raw" / "earthquakes.csv")
df["time"] = pd.to_datetime(df["time"], format="mixed", utc=True)

print("Magnitude below 2.5:", (df["magnitude"] < 2.5).sum())
print("Magnitude missing:", df["magnitude"].isna().sum())

plt.figure(figsize=(9, 5))
plt.hist(df["magnitude"].dropna(), bins=40, edgecolor="black")
plt.title("Earthquake Magnitude Distribution (2015–2025)")
plt.xlabel("Magnitude")
plt.ylabel("Number of Events")
plt.tight_layout()
plt.savefig(root / "magnitude_distribution.png", dpi=150)
plt.close()

yearly = df.groupby(df["time"].dt.year).size()

plt.figure(figsize=(9, 5))
yearly.plot(kind="bar", edgecolor="black")
plt.title("Earthquake Events per Year (2015–2025)")
plt.xlabel("Year")
plt.ylabel("Number of Events")
plt.tight_layout()
plt.savefig(root / "yearly_earthquakes.png", dpi=150)
plt.close()


print("\nYearly event counts:")
print(yearly)