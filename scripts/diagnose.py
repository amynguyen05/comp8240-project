import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn import metrics
from sklearn.svm import SVC

THETA, DATA, OUT = Path("results/theta"), Path("datasets"), Path("results/diagnostics")


def labels(name, split):
    return np.loadtxt(DATA / name / f"{split}_labels.txt", dtype=int)


def f1(names):
    rows, per_class = [], []
    for name in names:
        ytr, yte = labels(name, "train"), labels(name, "test")
        for path in sorted(THETA.glob(f"{name}_K50_seed*.npz")):
            z = np.load(path)
            pred = (
                SVC(gamma="scale").fit(z["train_theta"], ytr).predict(z["test_theta"])
            )  # same as topmost _cls
            rows.append(
                {
                    "dataset": name,
                    "seed": path.stem[-1],
                    "Acc": metrics.accuracy_score(yte, pred),
                    **{
                        f"F1 {a}": metrics.f1_score(yte, pred, average=a)
                        for a in ["macro", "weighted", "micro"]
                    },
                }
            )
        p, r, f, s = metrics.precision_recall_fscore_support(yte, pred, zero_division=0)
        per_class.append(
            pd.DataFrame(
                {
                    "dataset": name,
                    "class": range(len(f)),
                    "test docs": s,
                    "F1": f.round(3),
                }
            )
        )
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "f1_by_seed.csv", index=False)
    pd.concat(per_class).to_csv(OUT / "f1_per_class.csv", index=False)
    print(
        pd.DataFrame(rows)
        .groupby("dataset")
        .mean(numeric_only=True)
        .round(3)
        .to_string(),
        "\n",
    )
    print("Per-class F1 (last seed), smallest classes first:")
    for t in per_class:
        print(t.sort_values("test docs").to_string(index=False), "\n")


def nmi(name, ks):
    rows = []
    y = labels(name, "test")
    for k in ks:
        for path in sorted(THETA.glob(f"{name}_K{k}_seed*.npz")):
            c = np.load(path)["test_theta"].argmax(1)
            rows.append(
                {
                    "K": k,
                    "clusters used": len(set(c)),
                    "MI (shared info)": metrics.mutual_info_score(y, c),
                    "H(clusters)": entropy(np.bincount(c)),
                    "H(classes)": entropy(np.bincount(y)),
                    "NMI arithmetic (TopMost)": metrics.normalized_mutual_info_score(
                        y, c
                    ),
                    "NMI min": metrics.normalized_mutual_info_score(
                        y, c, average_method="min"
                    ),
                    "Purity": metrics.cluster.contingency_matrix(y, c).max(0).sum()
                    / len(y),
                }
            )
    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows).groupby("K").mean().round(3)
    table.to_csv(OUT / f"nmi_{name}.csv")
    print(table.to_string())
    print(
        "\nIf MI stays flat while H(clusters) grows, NMI falls only because the denominator grows."
    )


def vocab():
    rows = []
    for v in [20000, 30000, 40000, 50000]:
        name = f"DBpedia-V{v}"
        df = Counter(
            w
            for doc in (DATA / name / "train_texts.txt").read_text().splitlines()
            for w in set(doc.split())
        )
        for path in sorted(Path("results/topwords").glob(f"{name}_K50_seed*.txt")):
            words = [w for line in path.read_text().splitlines() for w in line.split()]
            freq = np.array([df.get(w, 0) for w in words])
            rows.append(
                {
                    "V": v,
                    "median doc freq of top words": np.median(freq),
                    "% top words in <5 docs": 100 * (freq < 5).mean(),
                    "% vocab in <5 docs": 100 * np.mean([c < 5 for c in df.values()]),
                }
            )
    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows).groupby("V").mean().round(2)
    table.to_csv(OUT / "vocab_DBpedia.csv")
    print(table.to_string())
    print(
        "\nIf top words get rarer as V grows, topics stay 'diverse' by using rare, specific words."
    )


if __name__ == "__main__":
    task, *rest = sys.argv[1:]
    {
        "f1": lambda: f1(rest),
        "nmi": lambda: nmi(rest[0], [int(k) for k in rest[1:]]),
        "vocab": vocab,
    }[task]()
