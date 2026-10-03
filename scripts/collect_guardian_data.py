import json
import os
import sys
import time
from pathlib import Path

import requests

API = "https://content.guardianapis.com/search"
KEY = os.environ["GUARDIAN_API_KEY"]
OUT_DIR = Path("datasets/raw/guardian")
FROM, TO = "2025-01-01", "2026-09-30"
SECTIONS = [
    "australia-news",
    "sport",
    "business",
    "environment",
    "culture",
    "technology",
]


def collect(section, max_pages=50):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{section}.jsonl"
    seen = set()
    if out.exists():
        seen = {json.loads(line)["id"] for line in out.open()}

    page, pages = 1, 1
    with out.open("a") as f:
        while page <= min(pages, max_pages):
            params = {
                "api-key": KEY,
                "section": section,
                "production-office": "aus",  # Guardian Australia only
                "type": "article",  # excludes live blogs, galleries, video
                "from-date": FROM,
                "to-date": TO,
                "show-fields": "bodyText,wordcount,headline",
                "page-size": 200,
                "page": page,
                "order-by": "oldest",
            }
            resp = requests.get(API, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()["response"]
            pages = data["pages"]

            new = 0
            for item in data["results"]:
                if item["id"] in seen:
                    continue
                fields = item.get("fields", {})
                f.write(
                    json.dumps(
                        {
                            "id": item["id"],
                            "section": section,
                            "date": item["webPublicationDate"],
                            "title": fields.get("headline", item["webTitle"]),
                            "wordcount": int(fields.get("wordcount", 0) or 0),
                            "text": fields.get("bodyText", ""),
                        }
                    )
                    + "\n"
                )
                seen.add(item["id"])
                new += 1

            print(
                f"{section}: page {page}/{pages}, +{new} (total {len(seen)}, available {data['total']})"
            )
            page += 1
            time.sleep(0.5)


def main():
    for section in sys.argv[1:] or SECTIONS:
        collect(section)


if __name__ == "__main__":
    main()
