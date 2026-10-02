import pandas as pd

PAPER = {
    "20NG": {"TD": 0.981, "Purity": 0.570, "NMI": 0.522, "Acc": 0.604, "F1": 0.593},
    "NYT": {"TD": 0.999, "Purity": 0.662, "NMI": 0.369, "Acc": 0.754, "F1": 0.596},
    "NeurIPS": {"TD": 0.996},
    "ACL": {"TD": 0.997},
    "Wikitext-103": {"TD": 0.992},
}
PAPER_CPU_S = {
    "20NG": 95.93,
    "NYT": 112.96,
    "NeurIPS": 177.58,
    "ACL": 209.75,
    "Wikitext-103": 552.70,
}
METRICS = ["TD", "Purity", "NMI", "Acc", "F1"]

df = pd.read_csv("results/fastopic_original.csv")
df = df[df["K"] == 50]
df = df.drop_duplicates(subset=["dataset", "K", "seed"], keep="first")

rows = []
for name, group in df.groupby("dataset", sort=False):
    for m in METRICS:
        values = group[m].dropna()
        if values.empty:
            continue
        paper = PAPER.get(name, {}).get(m)
        rows.append(
            {
                "dataset": name,
                "metric": m,
                "paper": paper,
                "ours_mean": round(values.mean(), 3),
                "ours_sd": round(values.std(), 3),
                "gap": round(values.mean() - paper, 3) if paper is not None else None,
                "seeds": len(values),
            }
        )

    # Embedding time is only recorded on the first seed; apply it to every seed.
    total = group["train_s"] + group["embed_s"].max()
    rows.append(
        {
            "dataset": name,
            "metric": "time_s (CPU, embed+train)",
            "paper": PAPER_CPU_S.get(name),
            "ours_mean": round(total.mean(), 1),
            "ours_sd": round(total.std(), 1),
            "gap": (
                round(total.mean() - PAPER_CPU_S[name], 1)
                if name in PAPER_CPU_S
                else None
            ),
            "seeds": len(group),
        }
    )

table = pd.DataFrame(rows)
print(table.to_string(index=False))
table.to_csv("results/summary_original.csv", index=False)
