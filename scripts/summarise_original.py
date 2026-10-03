from pathlib import Path

import pandas as pd

PAPER = {
    "20NG": {
        "CV": 0.412,
        "TD": 0.981,
        "Purity": 0.570,
        "NMI": 0.522,
        "Acc": 0.604,
        "F1": 0.593,
    },
    "NYT": {
        "CV": 0.437,
        "TD": 0.999,
        "Purity": 0.662,
        "NMI": 0.369,
        "Acc": 0.754,
        "F1": 0.596,
    },
    "NeurIPS": {"CV": 0.420, "TD": 0.996},
    "ACL": {"CV": 0.413, "TD": 0.997},
    "Wikitext-103": {"CV": 0.439, "TD": 0.992},
}
PAPER_CPU_S = {
    "20NG": 95.93,
    "NYT": 112.96,
    "NeurIPS": 177.58,
    "ACL": 209.75,
    "Wikitext-103": 552.70,
}
METRICS = ["CV", "TD", "Purity", "NMI", "Acc", "F1"]

raw = pd.read_csv("results/fastopic_original.csv")
raw = raw[raw["K"] == 50]
embed_s = raw.groupby("dataset")[
    "embed_s"
].max()  # from whichever run computed embeddings
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
    for m in METRICS:
        values = pd.to_numeric(group[m], errors="coerce").dropna()
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

    total = group["train_s"] + embed_s[name]
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
