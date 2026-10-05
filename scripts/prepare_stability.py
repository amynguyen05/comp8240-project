import sys
from pathlib import Path

from datasets import load_dataset
from topmost import Preprocess

DATA = Path("datasets")
SEED = 42
SIZES = [15000, 20000, 25000, 30000, 35000, 40000]
VOCABS = [20000, 30000, 40000, 50000]
VOCAB_N = 10000


def build(name, n_train, vocab_size, train, test):
    out = DATA / name
    if (out / "train_texts.txt").exists():
        print(f"{name}: already prepared, skipping")
        return
    tr = train.select(range(n_train))
    te = test.select(range(n_train // 4))
    pre = Preprocess(
        vocab_size=vocab_size, stopwords="English", min_term=1, seed=SEED, verbose=False
    )
    rst = pre.preprocess(
        list(tr["content"]), list(tr["label"]), list(te["content"]), list(te["label"])
    )
    pre.save(
        str(out),
        rst["vocab"],
        rst["train_texts"],
        rst["train_bow"],
        train_labels=rst["train_labels"],
        test_texts=rst["test_texts"],
        test_bow=rst["test_bow"],
        test_labels=rst["test_labels"],
    )
    print(
        f"{name}: train {len(rst['train_texts'])} | test {len(rst['test_texts'])} | "
        f"vocab {len(rst['vocab'])} (requested {vocab_size})"
    )


def main():
    which = sys.argv[1:] or ["size", "vocab"]
    ds = load_dataset("fancyzhx/dbpedia_14")
    train, test = ds["train"].shuffle(seed=SEED), ds["test"].shuffle(seed=SEED)
    if "size" in which:
        for n in SIZES:
            build(f"DBpedia-N{n}", n, 10000, train, test)
    if "vocab" in which:
        for v in VOCABS:
            build(f"DBpedia-V{v}", VOCAB_N, v, train, test)


if __name__ == "__main__":
    main()
