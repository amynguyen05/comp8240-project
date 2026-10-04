import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from fastopic import FASTopic
from topmost import Preprocess
from topmost.data.file_utils import read_text
from topmost.eva import _cls

DATA = Path("datasets")
OUT = Path("results/transfer.csv")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cached_embeddings(name, split):
    path = DATA / name / f"emb_{split}.npy"
    if not path.exists():
        raise SystemExit(f"Missing {path}: run run_fastopic.py {name} first")
    return np.load(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="Wikitext-103")
    parser.add_argument("--targets", nargs="+", default=["20NG", "NYT", "OHSUMED"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--k", type=int, default=50)
    args = parser.parse_args()

    source_texts = read_text(DATA / args.source / "train_texts.txt")
    source_emb = cached_embeddings(args.source, "train")
    targets = {
        t: {
            "train_emb": torch.as_tensor(cached_embeddings(t, "train")),
            "test_emb": torch.as_tensor(cached_embeddings(t, "test")),
            "train_labels": np.loadtxt(DATA / t / "train_labels.txt", dtype=int),
            "test_labels": np.loadtxt(DATA / t / "test_labels.txt", dtype=int),
        }
        for t in args.targets
    }

    new_file = not OUT.exists()
    for seed in args.seeds:
        set_seed(seed)
        model = FASTopic(
            args.k,
            preprocess=Preprocess(stopwords=[], verbose=False),
            num_top_words=15,
            device="cpu",
        )
        start = time.time()
        model.fit_transform(
            source_texts,
            epochs=200,
            learning_rate=0.002,
            preset_doc_embeddings=source_emb,
        )
        train_s = round(time.time() - start, 1)

        for name, t in targets.items():
            train_theta = model.transform(doc_embeddings=t["train_emb"])
            test_theta = model.transform(doc_embeddings=t["test_emb"])
            cls = _cls(train_theta, test_theta, t["train_labels"], t["test_labels"])
            row = {
                "source": args.source,
                "target": name,
                "K": args.k,
                "seed": seed,
                "Acc": round(cls["acc"], 4),
                "F1": round(cls["macro-F1"], 4),
                "train_s": train_s,
            }
            print(row)
            with open(OUT, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                if new_file:
                    writer.writeheader()
                    new_file = False
                writer.writerow(row)


if __name__ == "__main__":
    main()
