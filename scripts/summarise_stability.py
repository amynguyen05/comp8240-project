from pathlib import Path

import pandas as pd

METRICS = ["CV", "TD", "Purity", "NMI", "Acc", "F1", "train_s"]

df = pd.read_csv("results/stability.csv").drop_duplicates(
    subset=["dataset", "K", "seed"], keep="first"
)
cv_path = Path("results/cv_palmetto.csv")
if cv_path.exists():
    df = df.merge(
        pd.read_csv(cv_path).drop_duplicates(
            subset=["dataset", "K", "seed"], keep="last"
        ),
        on=["dataset", "K", "seed"],
        how="left",
    )
else:
    df["CV"] = None


def setting(row):
    if "-N" in row["dataset"]:
        return "size", row["dataset"].split("-N")[1]
    if "-V" in row["dataset"]:
        return "vocab", row["dataset"].split("-V")[1]
    return "K", str(row["K"])


df[["experiment", "value"]] = df.apply(lambda r: pd.Series(setting(r)), axis=1)
table = (
    df.groupby(["experiment", "value", "dataset", "K"], sort=False)[METRICS]
    .mean()
    .round(3)
    .reset_index()
)
print(table.to_string(index=False))
table.to_csv("results/summary_stability.csv", index=False)
