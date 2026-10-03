import sys
import urllib.request
from pathlib import Path

from topmost import Preprocess

DATA_DIR = Path("datasets")
RAW_DIR = DATA_DIR / "raw"


def reuters():
    """NLTK Reuters-21578, ModApte split, single-label documents only."""
    import nltk

    nltk.download("reuters", quiet=True)
    from nltk.corpus import reuters as r

    out = {"train": ([], []), "test": ([], [])}
    for fid in r.fileids():
        cats = r.categories(fid)
        if len(cats) != 1:
            continue
        split = "train" if fid.startswith("training/") else "test"
        out[split][0].append(r.raw(fid))
        out[split][1].append(cats[0])
    return (*out["train"], *out["test"])


def ohsumed():
    """Zenodo 5256848: texts.txt + score.txt, no fixed split, so a stratified 80/20 split (seed 42)."""
    from sklearn.model_selection import train_test_split

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    base = "https://zenodo.org/records/5256848/files"
    for name in ["texts.txt", "score.txt"]:
        if not (RAW_DIR / f"ohsumed_{name}").exists():
            urllib.request.urlretrieve(
                f"{base}/{name}?download=1", RAW_DIR / f"ohsumed_{name}"
            )
    texts = (RAW_DIR / "ohsumed_texts.txt").read_text(encoding="utf-8").splitlines()
    labels = (RAW_DIR / "ohsumed_score.txt").read_text(encoding="utf-8").split()
    assert len(texts) == len(labels), (len(texts), len(labels))
    tr_x, te_x, tr_y, te_y = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )
    return tr_x, tr_y, te_x, te_y


def trec(level):
    """SetFit/TREC-QC, its own train/test split. level = 'label_coarse' (6) or 'label' (50)."""
    from datasets import load_dataset

    ds = load_dataset("SetFit/TREC-QC")
    return (
        ds["train"]["text"],
        ds["train"][level],
        ds["test"]["text"],
        ds["test"][level],
    )


def dbpedia(n_train=20000, n_test=5000):
    """fancyzhx/dbpedia_14, random subset (seed 42) to keep CPU runs small; classes are balanced."""
    from datasets import load_dataset

    ds = load_dataset("fancyzhx/dbpedia_14")
    tr = ds["train"].shuffle(seed=42).select(range(n_train))
    te = ds["test"].shuffle(seed=42).select(range(n_test))
    return tr["content"], tr["label"], te["content"], te["label"]


def bbc():
    """SetFit/bbc-news, unstemmed text, its own train/test split."""
    from datasets import load_dataset

    ds = load_dataset("SetFit/bbc-news")
    return (
        ds["train"]["text"],
        ds["train"]["label"],
        ds["test"]["text"],
        ds["test"]["label"],
    )


LOADERS = {
    "Reuters": reuters,
    "OHSUMED": ohsumed,
    "TREC-coarse": lambda: trec("label_coarse"),
    "TREC-fine": lambda: trec("label"),
    "DBpedia": dbpedia,
    "BBC": bbc,
}


def main():
    names = sys.argv[1:] or list(LOADERS)
    for name in names:
        print(f"\n=== {name} ===")
        tr_x, tr_y, te_x, te_y = LOADERS[name]()
        pre = Preprocess(
            vocab_size=10000, stopwords="English", min_term=1, seed=42, verbose=False
        )
        rst = pre.preprocess(list(tr_x), list(tr_y), list(te_x), list(te_y))
        pre.save(
            str(DATA_DIR / name),
            rst["vocab"],
            rst["train_texts"],
            rst["train_bow"],
            train_labels=rst["train_labels"],
            test_texts=rst["test_texts"],
            test_bow=rst["test_bow"],
            test_labels=rst["test_labels"],
        )
        n_docs = len(rst["train_texts"]) + len(rst["test_texts"])
        avg_len = (rst["train_bow"].sum() + rst["test_bow"].sum()) / n_docs
        n_labels = len(set(rst["train_labels"]) | set(rst["test_labels"]))
        print(
            f"train {len(rst['train_texts'])} | test {len(rst['test_texts'])} | "
            f"avg length {avg_len:.1f} | vocab {len(rst['vocab'])} | labels {n_labels}"
        )


if __name__ == "__main__":
    main()
