from pathlib import Path

import pandas as pd

MAIN = [
    "results/fastopic_original.csv",
    "results/fastopic_existing.csv",
    "results/fastopic_constructed.csv",
]
METRICS = ["CV", "TD", "Purity", "NMI", "Acc", "F1"]

PAPER = {  # (with ETP, without ETP)
    50: {
        "20NG": {
            "CV": (0.412, 0.368),
            "TD": (0.981, 0.391),
            "Purity": (0.570, 0.401),
            "NMI": (0.522, 0.452),
        },
        "NYT": {
            "CV": (0.437, 0.363),
            "TD": (0.999, 0.338),
            "Purity": (0.662, 0.553),
            "NMI": (0.369, 0.294),
        },
        "NeurIPS": {"CV": (0.420, 0.380), "TD": (0.996, 0.216)},
        "ACL": {"CV": (0.413, 0.373), "TD": (0.997, 0.164)},
        "Wikitext-103": {"CV": (0.439, 0.359), "TD": (0.992, 0.335)},
    },
    10: {
        "20NG": {
            "CV": (0.432, 0.367),
            "TD": (1.000, 0.571),
            "Purity": (0.331, 0.294),
            "NMI": (0.437, 0.408),
        },
        "NYT": {
            "CV": (0.428, 0.371),
            "TD": (1.000, 0.620),
            "Purity": (0.549, 0.512),
            "NMI": (0.316, 0.263),
        },
        "NeurIPS": {"CV": (0.434, 0.379), "TD": (1.000, 0.512)},
        "ACL": {"CV": (0.442, 0.364), "TD": (1.000, 0.412)},
        "Wikitext-103": {"CV": (0.469, 0.416), "TD": (1.000, 0.660)},
    },
}


def load(files):
    frames = [pd.read_csv(f) for f in files if Path(f).exists()]
    if not frames:
        return pd.DataFrame(columns=["dataset", "K", "seed"])
    df = pd.concat(frames).drop_duplicates(
        subset=["dataset", "K", "seed"], keep="first"
    )
    cv_path = Path("results/cv_palmetto.csv")
    if cv_path.exists():
        cv = pd.read_csv(cv_path).drop_duplicates(
            subset=["dataset", "K", "seed"], keep="last"
        )
        df = df.merge(cv, on=["dataset", "K", "seed"], how="left")
    return df


def mean(df, dataset, k, metric):
    if metric not in df.columns:
        return None
    v = pd.to_numeric(
        df[(df["dataset"] == dataset) & (df["K"] == k)][metric], errors="coerce"
    ).dropna()
    return round(v.mean(), 3) if not v.empty else None


def main():
    runs = {
        50: (load(MAIN), load(["results/ablation_no_etp.csv"])),
        10: (load(["results/ablation_k10.csv"]),) * 2,
    }
    rows = []
    for k, (with_df, without_df) in runs.items():
        datasets = sorted(
            {
                d[: -len("-noETP")]
                for d in without_df["dataset"]
                if str(d).endswith("-noETP")
            }
        )
        for d in datasets:
            for m in METRICS:
                w, wo = mean(with_df, d, k, m), mean(without_df, f"{d}-noETP", k, m)
                if w is None and wo is None:
                    continue
                pw, pwo = PAPER.get(k, {}).get(d, {}).get(m, (None, None))
                rows.append(
                    {
                        "K": k,
                        "dataset": d,
                        "metric": m,
                        "ours_with_ETP": w,
                        "ours_without_ETP": wo,
                        "paper_with_ETP": pw,
                        "paper_without_ETP": pwo,
                    }
                )
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    table.to_csv("results/summary_ablation.csv", index=False)


if __name__ == "__main__":
    main()
