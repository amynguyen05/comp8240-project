from pathlib import Path

import pandas as pd

FILES = ["results/fastopic_guardian.csv", "results/fastopic_tmdb.csv"]
METRICS = ["CV", "TD", "Purity", "NMI", "Acc", "F1", "train_s"]

raw = pd.concat([pd.read_csv(f) for f in FILES if Path(f).exists()])
raw = raw[raw["K"] == 50]
embed_s = raw.groupby("dataset")["embed_s"].max()
df = raw.drop_duplicates(subset=["dataset", "K", "seed"], keep="first")

cv_path = Path("results/cv_palmetto.csv")
if cv_path.exists():
    cv = pd.read_csv(cv_path).drop_duplicates(
        subset=["dataset", "K", "seed"], keep="last"
    )
    df = df.merge(cv, on=["dataset", "K", "seed"], how="left")
else:
    df["CV"] = None

rows = []
for name, group in df.groupby("dataset", sort=False):
    row = {"dataset": name, "seeds": len(group)}
    for m in METRICS:
        values = pd.to_numeric(group[m], errors="coerce").dropna()
        row[f"{m}_mean"] = round(values.mean(), 3) if not values.empty else None
        row[f"{m}_sd"] = round(values.std(), 3) if not values.empty else None
    row["embed_s"] = embed_s.get(name)
    rows.append(row)

table = pd.DataFrame(rows)
print(table.to_string(index=False))
table.to_csv("results/summary_constructed.csv", index=False)
