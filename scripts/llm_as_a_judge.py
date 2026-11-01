import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

THETA_DIR = Path("results/theta")
JUDGE_DIR = Path("results/llm_judge")
DATA_DIR = Path("datasets")
SUMMARY_FILE = Path("results/llm_judge_summary.csv")

N_WORDS = 10  # top words shown per topic
N_DOC_WORDS = 60  # documents are cut to their first 60 words
N_SIMILAR_PAIRS = 15  # most similar topic pairs per run
N_RANDOM_PAIRS = 10  # random pairs, as a contrast
SEED = 42

# Prompts: written like an annotation guideline (criteria, anchored scale, reason first, JSON out).
RATING = """You are evaluating topics produced by a topic model, as a careful human annotator would.
Here are the top words of one topic, most important first:
{words}
 
Criterion: can a reader see one clear, nameable theme in these words?
3 = very related: the words clearly describe one theme you could name.
2 = somewhat related: there is a theme, but several words do not fit it.
1 = not related: no clear theme.
First explain your judgement in one sentence, then give the rating.
Reply in JSON only: {{"reason": "<one sentence>", "rating": <1, 2 or 3>, "label": "<short theme name, or none>"}}"""

WORD = """Here are six words. Five belong to the same topic and one does not.
{words}
First explain in one sentence what the five related words have in common, then name the word that does not belong.
Reply in JSON only: {{"reason": "<one sentence>", "intruder": "<the word>"}}"""

DOCUMENT = """A topic model found a topic with these top words: {words}
Below are four documents (preprocessed: lowercased, stopwords removed, truncated).
Three are strongly about this topic and one is not.
{docs}
First explain in one sentence, then give the number of the document that does not belong.
Reply in JSON only: {{"reason": "<one sentence>", "intruder": <1, 2, 3 or 4>}}"""

PAIR = """Here are the top words of two topics produced by a topic model.
Topic A: {a}
Topic B: {b}
Do these two topics describe the same concept, or clearly different concepts?
same = they are about the same thing (redundant topics).
overlap = related, but each has its own distinct focus.
different = they are about clearly different things.
First explain in one sentence, then give your answer.
Reply in JSON only: {{"reason": "<one sentence>", "answer": "<same, overlap or different>"}}"""


def ask(prompt, backend, model):
    """Send one prompt and return (parsed JSON reply, raw text). An unreadable reply gives {}."""
    if backend == "ollama":
        response = requests.post(
            "http://localhost:11434/api/chat",
            timeout=300,
            json={
                "model": model,
                "stream": False,
                "think": False,  # qwen3.5 thinking mode off: answer directly
                "format": "json",
                "options": {"temperature": 0},  # deterministic answers
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        text = response.json()["message"]["content"]
    elif backend == "mock":  # tests the pipeline without an LLM
        text = random.choice(
            [
                '{"reason": "x", "rating": 2}',
                '{"reason": "x", "intruder": 1}',
                '{"reason": "x", "answer": "same"}',
                '{"reason": "x", "answer": "different"}',
            ]
        )
    else:
        raise ValueError(backend)

    match = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(match.group(0)) if match else {}, text
    except json.JSONDecodeError:
        return {}, text


def load_run(run):
    """Topic-word weights, test topic mixes, vocabulary and test texts of one saved run."""
    saved = np.load(THETA_DIR / f"{run}.npz", allow_pickle=True)
    dataset = re.sub(r"-noETP$", "", run.split("_K")[0])  # 20NG-noETP_K50_seed0 -> 20NG
    texts = (DATA_DIR / dataset / "test_texts.txt").read_text().splitlines()
    return saved["beta"], saved["test_theta"], list(saved["vocab"]), texts


def build_items(run, orders):
    """All questions for one run. Each item keeps its prompt, what is shown and (if known) the right answer."""
    beta, theta, vocab, texts = load_run(run)
    rng = random.Random(SEED)
    n_topics = beta.shape[0]
    rank = np.argsort(-beta, axis=1)
    top = [[vocab[j] for j in rank[k, :N_WORDS]] for k in range(n_topics)]
    assign = theta.argmax(1)  # each document's main topic
    items = []

    for k in range(n_topics):
        items.append(
            {
                "task": "rating",
                "topic": k,
                "order": 0,
                "shown": ", ".join(top[k]),
                "prompt": RATING.format(words=", ".join(top[k])),
            }
        )

        # Intruder word: a top-5 word of another topic that ranks in the bottom half for topic k.
        rank_k = {vocab[j]: r for r, j in enumerate(rank[k])}
        pool = [
            w
            for o in range(n_topics)
            if o != k
            for w in top[o][:5]
            if rank_k[w] > len(vocab) // 2 and w not in top[k]
        ]
        # Intruder document: 3 docs assigned to k with the highest theta_k, plus 1 doc assigned elsewhere with a low theta_k.
        own = [i for i in np.argsort(-theta[:, k]) if assign[i] == k][:3]
        others = [i for i in np.argsort(theta[:, k]) if assign[i] != k][:50]
        word_intruder = rng.choice(pool) if pool else None
        doc_intruder = int(rng.choice(others)) if (len(own) == 3 and others) else None

        for o in range(orders):  # same items, shuffled again, to check position bias
            if word_intruder:
                words = top[k][:5] + [word_intruder]
                rng.shuffle(words)
                items.append(
                    {
                        "task": "word",
                        "topic": k,
                        "order": o,
                        "answer": word_intruder,
                        "shown": ", ".join(words),
                        "prompt": WORD.format(words=", ".join(words)),
                    }
                )
            if doc_intruder is not None:
                docs = own + [doc_intruder]
                rng.shuffle(docs)
                shown = "\n".join(
                    f"{n}. {' '.join(texts[i].split()[:N_DOC_WORDS])}"
                    for n, i in enumerate(docs, 1)
                )
                items.append(
                    {
                        "task": "document",
                        "topic": k,
                        "order": o,
                        "answer": docs.index(doc_intruder) + 1,
                        "candidates": [
                            int(i) for i in docs
                        ],  # document ids, to compare across orders
                        "shown": shown,
                        "prompt": DOCUMENT.format(words=", ".join(top[k]), docs=shown),
                    }
                )

    # Topic pairs: the most similar pairs (cosine of topic-word weights) plus random pairs for contrast.
    unit = beta / np.linalg.norm(beta, axis=1, keepdims=True)
    sim = unit @ unit.T
    upper = [(sim[i, j], i, j) for i in range(n_topics) for j in range(i + 1, n_topics)]
    similar = [(i, j) for _, i, j in sorted(upper, reverse=True)[:N_SIMILAR_PAIRS]]
    rest = [(i, j) for _, i, j in upper if (i, j) not in similar]
    pairs = [(i, j, "similar") for i, j in similar] + [
        (i, j, "random") for i, j in rng.sample(rest, min(N_RANDOM_PAIRS, len(rest)))
    ]
    for i, j, kind in pairs:
        for o in range(orders):  # swap A and B to check position bias
            a, b = (i, j) if o == 0 else (j, i)
            items.append(
                {
                    "task": "pair",
                    "topic": f"{i}-{j}",
                    "order": o,
                    "kind": kind,
                    "shown": f"A: {', '.join(top[a])}\nB: {', '.join(top[b])}",
                    "prompt": PAIR.format(a=", ".join(top[a]), b=", ".join(top[b])),
                }
            )
    return items


def score(item, reply):
    """Rating: 1-3. Intrusion: 1 if the intruder was found, else 0. Pair: 1 if judged 'same'. None if unreadable."""
    if item["task"] == "pair":
        answer = str(reply.get("answer", "")).strip().lower()
        return {"same": 1.0, "overlap": 0.0, "different": 0.0}.get(answer)
    if item["task"] == "rating":
        rating = reply.get("rating")
        return rating if rating in (1, 2, 3) else None
    guess = reply.get("intruder")
    if item["task"] == "word":
        return float(str(guess).strip().lower() == item["answer"]) if guess else None
    try:
        return float(int(guess) == item["answer"])
    except (TypeError, ValueError):
        return None


def chosen(row):
    """What the judge picked, as a word or a document id, so answers can be compared across orders."""
    guess = row["reply"].get("intruder") if isinstance(row["reply"], dict) else None
    if row["task"] == "pair":
        return str(row["reply"].get("answer", "")).strip().lower() or None
    if guess is None or row["task"] == "rating":
        return None
    if row["task"] == "word":
        return str(guess).strip().lower()
    try:
        return row["candidates"][int(guess) - 1]
    except (ValueError, IndexError, TypeError):
        return None


def load_cache(run, model):
    """Replies already saved for this run and model, keyed by (task, topic, order)."""
    path = JUDGE_DIR / f"{run}.jsonl"
    cache = {}
    if path.exists():
        for line in path.open():
            c = json.loads(line)
            if c["model"] == model:
                cache[(c["task"], c["topic"], c["order"])] = c
    return cache


def summarise(run, model, cache):
    """One summary row for a run: mean scores per task and how often the two orders agree."""
    df = pd.DataFrame(list(cache.values()))
    df["guess"] = df.apply(chosen, axis=1)
    pairs = df[df.task == "pair"]

    # Position bias: share of items where both orders got the same answer.
    both = (
        df[df.task != "rating"]
        .pivot_table(
            index=["task", "topic"], columns="order", values="guess", aggfunc="first"
        )
        .dropna()
    )
    consistent = (
        (both.nunique(axis=1) == 1).groupby("task").mean() if both.shape[1] > 1 else {}
    )

    return {
        "run": run,
        "model": model,
        "rating_mean": df.loc[df.task == "rating", "score"].mean(),
        "rating_3_share": (df.loc[df.task == "rating", "score"] == 3).mean(),
        "word_intrusion_acc": df.loc[df.task == "word", "score"].mean(),
        "doc_intrusion_acc": df.loc[df.task == "document", "score"].mean(),
        "redundant_similar_pairs": (
            pairs.loc[pairs.kind == "similar", "score"].mean() if len(pairs) else None
        ),
        "redundant_random_pairs": (
            pairs.loc[pairs.kind == "random", "score"].mean() if len(pairs) else None
        ),
        "word_order_consistency": consistent.get("word"),
        "doc_order_consistency": consistent.get("document"),
        "pair_order_consistency": consistent.get("pair"),
        "n_topics": df.loc[df.task == "rating", "topic"].nunique(),
        "unparsed": int(df.score.isna().sum()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Judge FASTopic topics with a local LLM."
    )
    parser.add_argument(
        "runs", nargs="+", help="e.g. 20NG_K50_seed0 (needs results/theta/<run>.npz)"
    )
    parser.add_argument("--backend", default="ollama", choices=["ollama", "mock"])
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument(
        "--orders", type=int, default=2, help="orderings per item (position-bias check)"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="only the first N topics (quick test)"
    )
    args = parser.parse_args()
    JUDGE_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for run in args.runs:
        items = build_items(run, args.orders)
        if (
            args.limit is not None
        ):  # quick test: first N topics, and only the similar pairs among them
            items = [
                i
                for i in items
                if (i["task"] != "pair" and i["topic"] < args.limit)
                or (
                    i["task"] == "pair"
                    and i["kind"] == "similar"
                    and int(i["topic"].split("-")[0]) < args.limit
                )
            ]

        # Ask only what is not cached yet; write each reply straight away so nothing is lost.
        cache = load_cache(run, args.model)
        with (JUDGE_DIR / f"{run}.jsonl").open("a") as f:
            for n, item in enumerate(items, 1):
                key = (item["task"], item["topic"], item["order"])
                if key not in cache:
                    reply, raw = ask(item["prompt"], args.backend, args.model)
                    cache[key] = {
                        **item,
                        "model": args.model,
                        "reply": reply,
                        "raw": raw,
                        "score": score(item, reply),
                    }
                    f.write(json.dumps(cache[key]) + "\n")
                    f.flush()
                if n % 25 == 0:
                    print(f"{run}: {n}/{len(items)}")

        rows.append(summarise(run, args.model, cache))
        print(pd.DataFrame(rows[-1:]).round(3).to_string(index=False), "\n")

    # Update the summary file: replace rows for the same run and model, keep the rest.
    new = pd.DataFrame(rows)
    if SUMMARY_FILE.exists():
        old = pd.read_csv(SUMMARY_FILE)
        keep = ~old.set_index(["run", "model"]).index.isin(
            new.set_index(["run", "model"]).index
        )
        new = pd.concat([old[keep], new])
    new.to_csv(SUMMARY_FILE, index=False)
    print(f"Saved {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
