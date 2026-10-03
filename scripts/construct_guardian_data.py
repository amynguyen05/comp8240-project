import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from topmost import Preprocess

RAW = Path("datasets/raw/guardian")
OUT = Path("datasets/Guardian")
LOG = Path("results/construction")
SEED = 42

LABELS = {  # Guardian section id -> label (arts sections merged into "culture")
    "australia-news": "australia-news",
    "world": "world",
    "sport": "sport",
    "football": "sport",
    "culture": "culture",
    "music": "culture",
    "film": "culture",
    "books": "culture",
    "tv-and-radio": "culture",
    "stage": "culture",
    "artanddesign": "culture",
    "lifeandstyle": "lifeandstyle",
    "food": "lifeandstyle",
    "environment": "environment",
}
ROUNDUP = r"\b(?:morning mail|afternoon update|full story|quiz|crossword)\b"
SIGNUP = r"Sign up (?:for|to) [^.]{0,120}?(?:email|newsletter)s?\.?"
MIN_WORDS = 150
CAP = 1500

steps = []


def log(step, df):
    steps.append({"step": step, "docs": len(df)})
    print(f"{step:<50}{len(df):>7}")


def main():
    rows = [json.loads(line) for p in sorted(RAW.glob("*.jsonl")) for line in p.open()]
    df = pd.DataFrame(rows)
    log("raw articles collected", df)

    df = df[df["section"].isin(LABELS)].copy()
    df["label"] = df["section"].map(LABELS)
    log("in the six chosen labels", df)

    df = df.drop_duplicates("id").drop_duplicates("text")
    log("after removing duplicates", df)

    df = df[~df["title"].str.contains(ROUNDUP, case=False, regex=True, na=False)]
    log("after removing roundups / quizzes", df)

    df["text"] = df["text"].str.replace(
        SIGNUP, " ", regex=True
    )  # newsletter boilerplate
    df["words"] = df["text"].str.split().str.len()
    df = df[df["words"] >= MIN_WORDS]
    log(f"after minimum length ({MIN_WORDS} words)", df)

    df = df.sample(frac=1, random_state=SEED).groupby("label").head(CAP)
    log(f"after capping at {CAP} per label", df)

    LOG.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(steps).to_csv(LOG / "guardian_steps.csv", index=False)
    counts = df["label"].value_counts().rename_axis("label").reset_index(name="docs")
    counts.to_csv(LOG / "guardian_labels.csv", index=False)
    print("\n" + counts.to_string(index=False))

    train, test = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=SEED
    )
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
        f"\nGuardian: train {len(rst['train_texts'])} | test {len(rst['test_texts'])} | "
        f"avg length {avg:.1f} | vocab {len(rst['vocab'])} | labels {len(names)}"
    )


if __name__ == "__main__":
    main()
