import json
import os
import sys
import time
from pathlib import Path

import requests

API = "https://api.themoviedb.org/3"
KEY = os.environ["TMDB_API_KEY"]
OUT_DIR = Path("datasets/raw/tmdb")
GENRES = [
    "Horror",
    "Comedy",
    "Romance",
    "Science Fiction",
    "Animation",
    "Documentary",
    "War",
    "Western",
]
FILMS_PER_GENRE = 150
MIN_VOTES = 200  # popular enough to have reviews


def get(path, **params):
    params["api_key"] = KEY
    for attempt in range(5):
        resp = requests.get(f"{API}{path}", params=params, timeout=30)
        if resp.status_code == 429:  # rate limited: wait and retry
            time.sleep(2**attempt)
            continue
        resp.raise_for_status()
        time.sleep(0.25)
        return resp.json()
    raise RuntimeError(f"Too many retries for {path}")


def load_ids(path, key):
    return {json.loads(line)[key] for line in path.open()} if path.exists() else set()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    films_path, reviews_path = OUT_DIR / "films.jsonl", OUT_DIR / "reviews.jsonl"
    done_films = load_ids(films_path, "film_id")
    seen_reviews = load_ids(reviews_path, "review_id")

    genre_map = {
        g["name"]: g["id"] for g in get("/genre/movie/list", language="en")["genres"]
    }
    id_to_name = {v: k for k, v in genre_map.items()}

    for genre in sys.argv[1:] or GENRES:
        gid, collected, page = genre_map[genre], 0, 1
        while collected < FILMS_PER_GENRE:
            found = get(
                "/discover/movie",
                with_genres=gid,
                sort_by="popularity.desc",
                **{"vote_count.gte": MIN_VOTES},
                language="en-US",
                page=page,
            )
            if not found["results"]:
                break
            for film in found["results"]:
                collected += 1
                if film["id"] in done_films:
                    continue

                n_reviews, rpage = 0, 1
                while True:
                    rev = get(
                        f"/movie/{film['id']}/reviews", language="en-US", page=rpage
                    )
                    with reviews_path.open("a") as f:
                        for r in rev["results"]:
                            if r["id"] in seen_reviews:
                                continue
                            f.write(
                                json.dumps(
                                    {
                                        "review_id": r["id"],
                                        "film_id": film["id"],
                                        "author": r["author"],
                                        "rating": (r.get("author_details") or {}).get(
                                            "rating"
                                        ),
                                        "created_at": r["created_at"],
                                        "text": r["content"],
                                    }
                                )
                                + "\n"
                            )
                            seen_reviews.add(r["id"])
                            n_reviews += 1
                    if rpage >= rev["total_pages"]:
                        break
                    rpage += 1

                with films_path.open("a") as f:
                    f.write(
                        json.dumps(
                            {
                                "film_id": film["id"],
                                "title": film["title"],
                                "release_date": film.get("release_date"),
                                "genres": [
                                    id_to_name.get(g, str(g)) for g in film["genre_ids"]
                                ],
                                "sampled_under": genre,
                                "n_reviews": n_reviews,
                            }
                        )
                        + "\n"
                    )
                done_films.add(film["id"])
                if collected >= FILMS_PER_GENRE:
                    break
            print(
                f"{genre}: {collected} films checked, {len(seen_reviews)} reviews so far"
            )
            page += 1


if __name__ == "__main__":
    main()
