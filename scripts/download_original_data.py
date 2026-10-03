import sys
from pathlib import Path

import numpy as np
from topmost import download_dataset
from topmost.data.file_utils import read_text

DATA_DIR = Path("datasets")

# Table 1: (documents, average length, vocabulary, labels)
PAPER = {
    "20NG": (18846, 110.5, 5000, 20),
    "NYT": (9172, 175.4, 10000, 12),
    "NeurIPS": (7237, 2085.9, 10000, None),
    "ACL": (10560, 2023.0, 10000, None),
    "Wikitext-103": (28532, 1355.4, 10000, None),
}


def stats(name):
    d = DATA_DIR / name
    texts = read_text(d / "train_texts.txt") + read_text(d / "test_texts.txt")
    vocab = read_text(d / "vocab.txt")
    avg_len = np.mean([len(t.split()) for t in texts])
    n_labels = None
    if (d / "train_labels.txt").exists():
        labels = np.concatenate(
            [np.loadtxt(d / f"{s}_labels.txt", dtype=int) for s in ("train", "test")]
        )
        n_labels = len(set(labels))
    return len(texts), avg_len, len(vocab), n_labels


def main():
    names = sys.argv[1:] or list(PAPER)
    DATA_DIR.mkdir(exist_ok=True)

    print(f"{'dataset':<14}{'docs':>14}{'avg length':>20}{'vocab':>16}{'labels':>12}")
    for name in names:
        if not (DATA_DIR / name / "train_texts.txt").exists():
            download_dataset(name, cache_path=str(DATA_DIR))
        ours, paper = stats(name), PAPER[name]
        print(
            f"{name:<14}"
            f"{ours[0]:>7} ({paper[0]:>5})"
            f"{ours[1]:>10.1f} ({paper[1]:>7.1f})"
            f"{ours[2]:>8} ({paper[2]:>5})"
            f"{str(ours[3]):>6} ({str(paper[3]):>4})"
        )
    print("\nFormat: ours (paper)")


if __name__ == "__main__":
    main()
