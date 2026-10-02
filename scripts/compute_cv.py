import csv
import subprocess
import sys
from pathlib import Path

JAR = Path("palmetto/palmetto-0.1.0-jar-with-dependencies.jar")
INDEX = Path("palmetto/wikipedia/wikipedia_bd")
TOPWORDS = Path("results/topwords")
OUT = Path("results/cv_palmetto.csv")


def cv(topic_file):
    n_topics = len(topic_file.read_text().splitlines())
    result = subprocess.run(
        ["java", "-jar", str(JAR), str(INDEX), "C_V", str(topic_file)],
        capture_output=True,
        text=True,
        check=True,
    )
    # Palmetto prints one line per topic at the end: "<index> <score> <words>"
    lines = result.stdout.strip().splitlines()[-n_topics:]
    scores = [float(line.split()[1]) for line in lines]
    return sum(scores) / len(scores)


def main():
    wanted = set(sys.argv[1:])
    done = set()
    if OUT.exists():
        with open(OUT) as f:
            done = {(r["dataset"], r["K"], r["seed"]) for r in csv.DictReader(f)}

    new_file = not OUT.exists()
    for path in sorted(TOPWORDS.glob("*_K*_seed*.txt")):
        dataset, rest = path.stem.rsplit("_K", 1)
        k, seed = rest.split("_seed")
        if (wanted and dataset not in wanted) or (dataset, k, seed) in done:
            continue

        score = cv(path)
        print(f"{dataset:<14} K={k} seed={seed}  CV={score:.3f}")
        with open(OUT, "a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["dataset", "K", "seed", "CV"])
                new_file = False
            writer.writerow([dataset, k, seed, round(score, 4)])


if __name__ == "__main__":
    main()
