import argparse
import json
import random
from itertools import combinations
from pathlib import Path

import pandas as pd
from nltk.metrics.agreement import AnnotationTask
from sklearn.metrics import cohen_kappa_score

HUMAN_DIR = Path("results/human")
JUDGE_DIR = Path("results/llm_judge")
KEY_FILE = HUMAN_DIR / "key.csv"
TASKS = ["rating", "word", "document", "pair"]
PAIR_SCALE = {"different": "0", "overlap": "1", "same": "2"}  # pair answers are ordinal
SEED = 0


def read_cache(run, model):
    """The LLM's first-order replies for one run (the order humans see)."""
    rows = []
    for line in (JUDGE_DIR / f"{run}.jsonl").open():
        c = json.loads(line)
        if c["model"] == model and c["order"] == 0:
            rows.append(c)
    return rows


def export(runs, per_task, model):
    items = []
    for run in runs:
        for c in read_cache(run, model):
            shown = c["shown"]
            if c["task"] == "document":  # annotators also need the topic's top words
                words = c["prompt"].split("\n")[0].split(": ", 1)[1]
                shown = f"TOPIC: {words}\n{shown}"
            items.append(
                {"run": run, "task": c["task"], "topic": c["topic"], "shown": shown}
            )

    # Shuffle across runs, then keep the first per task questions of each task.
    random.Random(SEED).shuffle(items)
    key = pd.DataFrame(items).groupby("task", group_keys=False).head(per_task)
    key.insert(0, "id", range(1, len(key) + 1))

    HUMAN_DIR.mkdir(parents=True, exist_ok=True)
    key.to_csv(KEY_FILE, index=False)
    for task in TASKS:
        blank = key.loc[key.task == task, ["id", "shown"]].assign(answer="")
        blank.to_csv(HUMAN_DIR / f"blank_{task}.csv", index=False)
        print(f"{task}: {len(blank)} questions -> {HUMAN_DIR / f'blank_{task}.csv'}")
    print(f"key -> {KEY_FILE}")


def read_annotators():
    """{name: Series of answers indexed by question id}, from <name>_<task>.csv files."""
    files = {}
    for task in TASKS:
        for path in HUMAN_DIR.glob(f"*_{task}.csv"):
            name = path.stem[: -len(task) - 1]
            if name != "blank":
                files.setdefault(name, []).append(path)

    annotators = {}
    for name, paths in sorted(files.items()):
        # Only id and answer are read, so other columns (even ones Excel changed) do not matter.
        answers = pd.concat(
            pd.read_csv(p, dtype=str, usecols=["id", "answer"]) for p in paths
        )
        annotators[name] = (
            answers.set_index(answers.id.astype(int)).answer.str.strip().str.lower()
        )
        print(
            f"{name}: {annotators[name].notna().sum()} answers from {len(paths)} files"
        )
    return annotators


def agree(model):
    key = pd.read_csv(KEY_FILE, dtype={"topic": str})

    # LLM answer and the planted intruder (the truth) for every question, matched on run, task and topic.
    llm = []
    for run in key.run.unique():
        for c in read_cache(run, model):
            reply = c["reply"]
            llm.append(
                {
                    "run": run,
                    "task": c["task"],
                    "topic": str(c["topic"]),
                    "truth": str(c.get("answer", ""))
                    .strip()
                    .lower(),  # empty for rating and pair
                    "LLM": str(
                        reply.get(
                            "rating", reply.get("intruder", reply.get("answer", ""))
                        )
                    )
                    .strip()
                    .lower(),
                }
            )
    table = key.merge(
        pd.DataFrame(llm), on=["run", "task", "topic"], how="left"
    ).set_index("id")

    annotators = read_annotators()
    for name, answers in annotators.items():
        table[name] = answers
    raters = ["LLM"] + list(annotators)
    table[raters] = table[raters].mask(
        table[raters] == ""
    )  # blank answer = not answered

    rows = []
    for task in TASKS:
        t = table[table.task == task]
        if task == "pair":
            t = t.replace(PAIR_SCALE)
        weights = (
            "quadratic" if task in ("rating", "pair") else None
        )  # ordinal: near-misses count partly

        for a, b in combinations(raters, 2):
            both = t[[a, b]].dropna()  # questions both raters answered
            rows.append(
                {
                    "task": task,
                    "raters": f"{a} vs {b}",
                    "items": len(both),
                    "raw agreement": (both[a] == both[b]).mean(),
                    "kappa": cohen_kappa_score(both[a], both[b], weights=weights),
                }
            )
        if len(raters) >= 3:  # Fleiss' kappa over questions that every rater answered
            complete = t[raters].dropna()
            data = [(r, i, v) for r in raters for i, v in complete[r].items()]
            rows.append(
                {
                    "task": task,
                    "raters": "all (Fleiss)",
                    "items": len(complete),
                    "kappa": AnnotationTask(data=data).pi(),
                }
            )
    print("\nAgreement (quadratic-weighted kappa for rating and pair):")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # Intrusion has a right answer, so each rater can also be scored on it.
    print("\nIntrusion accuracy (share of planted intruders found):")
    acc = {
        task: {r: (t[r] == t.truth)[t[r].notna()].mean() for r in raters}
        for task, t in table[table.task.isin(["word", "document"])].groupby("task")
    }
    print(pd.DataFrame(acc).T.round(3).to_string())
    print(
        "\nThe judge does a human-level job if LLM-vs-human agreement is close to human-vs-human."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Validate the LLM judge against human annotators."
    )
    parser.add_argument("action", choices=["export", "agree"])
    parser.add_argument(
        "runs", nargs="*", help="runs to draw questions from (export only)"
    )
    parser.add_argument("--per-task", type=int, default=100, help="questions per task")
    parser.add_argument("--model", default="qwen3.5:9b")
    args = parser.parse_args()

    if args.action == "export":
        export(args.runs, args.per_task, args.model)
    else:
        agree(args.model)


if __name__ == "__main__":
    main()
