"""
gather.py
---------
Collects all per-index JSON files in prompt/ into a single results.json,
sorted by wiki_bio_test_idx.

Run:
    python3 -m replication.prompt.gather
"""
import glob
import json
import os

PROMPT_DIR  = os.path.dirname(__file__)
OUTPUT_PATH = os.path.join(PROMPT_DIR, "results.json")


def main() -> None:
    pattern = os.path.join(PROMPT_DIR, "*.json")
    paths   = [p for p in glob.glob(pattern) if os.path.abspath(p) != os.path.abspath(OUTPUT_PATH)]

    records: list[dict] = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            records.extend(data)
        else:
            records.append(data)

    records.sort(key=lambda r: r["wiki_bio_test_idx"])

    with open(OUTPUT_PATH, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Collected {len(records)} records from {len(paths)} files → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
