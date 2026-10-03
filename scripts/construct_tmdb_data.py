from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, detect
from sklearn.model_selection import GroupShuffleSplit
from topmost import Preprocess

RAW = Path("datasets/raw/tmdb")
OUT = Path("datasets/TMDB")
LOG = Path("results/construction")
SEED = 42

GENRES = ["Horror", "Comedy", "Romance", "Science Fiction", "War", "Western"]
URL = r"https?://\S+|www\.\S+"
MIN_WORDS = 30
DetectorFactory.seed = 0

steps = []


def log(step, df):
    steps.append({"step": step, "docs": len(df)})
    print(f"{step:<50}{len(df):>7}")


def is_english(text):
    try:
        return detect(text) == "en"
    except Exception:
        return False


def main():
    reviews = pd.read_json(RAW / "reviews.jsonl", lines=True)
    films = pd.read_json(RAW / "films.jsonl", lines=True)
    log("raw reviews collected", reviews)

    films["targets"] = films["genres"].apply(lambda g: [x for x in g if x in GENRES])
    single = films[films["targets"].str.len() == 1].copy()
    single["label"] = single["targets"].str[0]
    df = reviews.merge(single[["film_id", "label"]], on="film_id")
    log("films with exactly one of the six genres", df)

    df = df.drop_duplicates("review_id").drop_duplicates("text")
    log("after removing duplicates", df)

    df["text"] = df["text"].str.replace(URL, " ", regex=True)  # links / promotion
    df["words"] = df["text"].str.split().str.len()
    df = df[df["words"] >= MIN_WORDS]
    log(f"after minimum length ({MIN_WORDS} words)", df)

    df = df[df["text"].apply(is_english)]
    log("after keeping English only", df)

    LOG.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(steps).to_csv(LOG / "tmdb_steps.csv", index=False)
    counts = df["label"].value_counts().rename_axis("label").reset_index(name="docs")
    counts.to_csv(LOG / "tmdb_labels.csv", index=False)
    print("\n" + counts.to_string(index=False))

    # split by film, so reviews of the same film never appear in both train and test
    tr, te = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(
            df, groups=df["film_id"]
        )
    )
    train, test = df.iloc[tr], df.iloc[te]

    pre = Preprocess(
        vocab_size=10000, stopwords="English", min_term=1, seed=SEED, verbose=False
    )
    rst = pre.preprocess(
        train["text"].tolist(),
        train["label"].tolist(),
        test["text"].tolist(),
        test["label"].tolist(),
    )
    pre.save(
        str(OUT),
        rst["vocab"],
        rst["train_texts"],
        rst["train_bow"],
        train_labels=rst["train_labels"],
        test_texts=rst["test_texts"],
        test_bow=rst["test_bow"],
        test_labels=rst["test_labels"],
    )
    names = sorted(set(df["label"]))  # TopMost numbers labels in sorted order
    (OUT / "label_names.txt").write_text("\n".join(names))

    n = len(rst["train_texts"]) + len(rst["test_texts"])
    avg = (rst["train_bow"].sum() + rst["test_bow"].sum()) / n
    print(
        f"\nTMDB: train {len(rst['train_texts'])} | test {len(rst['test_texts'])} | "
        f"avg length {avg:.1f} | vocab {len(rst['vocab'])} | labels {len(names)}"
    )


if __name__ == "__main__":
    main()
