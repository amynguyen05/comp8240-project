from pathlib import Path

import pandas as pd

MAIN = [
    "results/fastopic_original.csv",
    "results/fastopic_existing.csv",
    "results/fastopic_constructed.csv",
]

transfer = pd.read_csv("results/transfer.csv").drop_duplicates(
    subset=["source", "target", "K", "seed"], keep="last"
)
main = pd.concat([pd.read_csv(f) for f in MAIN if Path(f).exists()])
main = main[main["K"] == 50].drop_duplicates(
    subset=["dataset", "K", "seed"], keep="first"
)

rows = []
for target, g in transfer.groupby("target", sort=False):
    own = main[main["dataset"] == target]
    rows.append(
        {
            "target": target,
            "transfer_Acc": round(g["Acc"].mean(), 3),
            "transfer_Acc_sd": round(g["Acc"].std(), 3),
            "transfer_F1": round(g["F1"].mean(), 3),
            "transfer_F1_sd": round(g["F1"].std(), 3),
            "in_domain_Acc": round(own["Acc"].mean(), 3) if not own.empty else None,
            "in_domain_F1": round(own["F1"].mean(), 3) if not own.empty else None,
            "seeds": len(g),
        }
    )

table = pd.DataFrame(rows)
print(table.to_string(index=False))
table.to_csv("results/summary_transfer.csv", index=False)
