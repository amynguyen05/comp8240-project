import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from fastopic import FASTopic
from sentence_transformers import SentenceTransformer
from topmost import Preprocess
from topmost.data.file_utils import read_text
from topmost.eva import _cls, _clustering, _diversity

DATA_DIR = Path("datasets")
RESULTS_DIR = Path("results")
EMBED_MODEL = "all-MiniLM-L6-v2"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load(name):
    d = DATA_DIR / name
    data = {
        "train_texts": read_text(d / "train_texts.txt"),
        "test_texts": read_text(d / "test_texts.txt"),
    }
    if (d / "train_labels.txt").exists():
        data["train_labels"] = np.loadtxt(d / "train_labels.txt", dtype=int)
        data["test_labels"] = np.loadtxt(d / "test_labels.txt", dtype=int)
    return data


def embeddings(name, split, texts):
    """Encode once and cache to datasets/<name>/emb_<split>.npy (gitignored)."""
    cache = DATA_DIR / name / f"emb_{split}.npy"
    if cache.exists():
        return np.load(cache), None
    start = time.time()
    emb = SentenceTransformer(EMBED_MODEL, device="cpu").encode(
        texts, show_progress_bar=True
    )
    seconds = time.time() - start
    np.save(cache, emb)
    return emb, seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--out", default="results/fastopic_original.csv")
    parser.add_argument(
        "--no-etp", action="store_true", help="ablation: softmax instead of ETP"
    )
    args = parser.parse_args()

    data = load(args.dataset)
    labelled = "train_labels" in data
    print(
        f"{args.dataset}: {len(data['train_texts'])} train / {len(data['test_texts'])} test docs"
    )

    train_emb, embed_s = embeddings(args.dataset, "train", data["train_texts"])
    test_emb, _ = embeddings(args.dataset, "test", data["test_texts"])
    if embed_s is not None:
        print(f"Embedded training docs in {embed_s:.1f}s")

    topwords_dir = RESULTS_DIR / "topwords"
    topwords_dir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out)
    label = args.dataset + ("-noETP" if args.no_etp else "")
    new_file = not out_csv.exists()

    for seed in args.seeds:
        set_seed(seed)
        model = FASTopic(
            args.k,
            preprocess=Preprocess(stopwords=[], verbose=False),
            num_top_words=15,
            device="cpu",
        )

        if args.no_etp:
            from fastopic_no_etp import FastopicNoETP

            model.model = FastopicNoETP(args.k)

        start = time.time()
        top_words, train_theta = model.fit_transform(
            data["train_texts"],
            epochs=args.epochs,
            learning_rate=0.002,
            preset_doc_embeddings=train_emb,
        )
        train_s = time.time() - start
        test_theta = model.transform(doc_embeddings=torch.as_tensor(test_emb))

        # Top words saved so CV can be computed later without retraining.
        (topwords_dir / f"{label}_K{args.k}_seed{seed}.txt").write_text(
            "\n".join(top_words)
        )

        row = {
            "dataset": label,
            "K": args.k,
            "seed": seed,
            "TD": _diversity(top_words),
            "Purity": "",
            "NMI": "",
            "Acc": "",
            "F1": "",
            "train_s": round(train_s, 1),
            "embed_s": round(embed_s, 1) if embed_s is not None else "",
        }
        if labelled:
            row.update(_clustering(test_theta, data["test_labels"]))
            cls = _cls(
                train_theta, test_theta, data["train_labels"], data["test_labels"]
            )
            row["Acc"], row["F1"] = cls["acc"], cls["macro-F1"]

        with open(out_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if new_file:
                writer.writeheader()
                new_file = False
            writer.writerow(row)

        embed_s = None  # only the first seed pays the embedding cost
        shown = {
            k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v)
            for k, v in row.items()
        }
        print(shown)


if __name__ == "__main__":
    main()
